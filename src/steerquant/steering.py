from contextlib import contextmanager
from pathlib import Path

import torch

from .config import artifact_root, config_id
from .data import contrastive_pairs, load_split, prompt
from .io import digest, environment, write_json
from .models import Backend


def hidden_tensor(output):
    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError("Expected residual block output [batch, sequence, hidden]")
    return tensor


@contextmanager
def capture(layers, indices):
    values, handles = {}, []
    try:
        for index in indices:
            def hook(module, inputs, output, index=index):
                values[index] = hidden_tensor(output)[0, -1].detach().float().cpu().clone()
            handles.append(layers[index].register_forward_hook(hook))
        yield values
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def intervene(layer, vector, magnitude):
    """Last prompt token on prefill, then the current token at every decode step.

    Capture and injection both refer to zero-based block OUTPUT (resid_post).
    Hooks are removed on success or failure; alpha zero installs no hook.
    """
    if magnitude == 0 or vector is None:
        yield
        return
    if not torch.isfinite(vector).all():
        raise ValueError("Non-finite steering vector")
    def hook(module, inputs, output):
        tensor = hidden_tensor(output)
        if tensor.shape[-1] != vector.numel():
            raise ValueError("Vector hidden dimension mismatch")
        updated = tensor.clone()
        updated[:, -1, :] = updated[:, -1, :] + magnitude * vector.to(device=tensor.device, dtype=tensor.dtype)
        return (updated, *output[1:]) if isinstance(output, tuple) else updated
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def vector_path(config, precision):
    return artifact_root(config) / "vectors" / f"{precision}.pt"


def extract(config, precision):
    path = vector_path(config, precision)
    if path.exists():
        raise ValueError(f"Vector artifact already exists: {path}")
    rows, manifest = load_split(config["data"]["prepared"], "train", config["data"]["train_limit"], config["seed"])
    pairs = list(contrastive_pairs(rows))
    if not pairs:
        raise ValueError("No contrastive training pairs")
    backend = Backend(config, precision)
    try:
        indices = config["steering"]["layers"]
        # Average within examples first, so ambiguity's two negatives do not double its weight.
        per_example = {r["id"]: {l: torch.zeros(backend.identity["hidden_size"]) for l in indices} for r in rows}
        counts = {r["id"]: 0 for r in rows}
        for n, pair in enumerate(pairs):
            states = []
            for answer in (pair["positive"], pair["negative"]):
                with capture(backend.layers, indices) as values, torch.inference_mode():
                    backend.model(**backend.encode(pair["prompt"], answer), use_cache=False)
                states.append(dict(values))
            for l in indices:
                per_example[pair["id"]][l] += states[0][l] - states[1][l]
            counts[pair["id"]] += 1
            if (n + 1) % 50 == 0:
                print(f"extract {precision}: {n + 1}/{len(pairs)} pairs", flush=True)
        directions, norms, scales = {}, {}, {}
        for l in indices:
            # Equal weight to ambiguous and disambiguated conditions.
            by_condition = []
            for condition in ("ambig", "disambig"):
                examples = [per_example[r["id"]][l] / counts[r["id"]] for r in rows if r["context_condition"] == condition]
                if not examples:
                    raise ValueError("Extraction requires both ambiguous and disambiguated examples")
                by_condition.append(torch.stack(examples).mean(0))
            direction = torch.stack(by_condition).mean(0)
            norm = direction.norm().item()
            if norm < 1e-8 or not torch.isfinite(direction).all():
                raise ValueError("Degenerate extraction direction")
            directions[str(l)] = direction / norm
            norms[str(l)] = norm
        calibration = {l: [] for l in indices}
        for row in rows:
            with capture(backend.layers, indices) as values, torch.inference_mode():
                backend.model(**backend.encode(prompt(row)), use_cache=False)
            for l in indices:
                calibration[l].append(values[l].norm().item())
        for l in indices:
            scales[str(l)] = sum(calibration[l]) / len(calibration[l])
            import math
            if not math.isfinite(scales[str(l)]) or scales[str(l)] <= 0:
                raise ValueError("Invalid calibration activation scale")
        meta = {"schema": 1, "config_id": config_id(config), "model": backend.identity,
                "data_fingerprint": manifest["fingerprint"], "seed": config["seed"],
                "training_ids": [r["id"] for r in rows], "pairs_sha256": digest(pairs), "pairs": len(pairs),
                "method": "mean_difference", "site": "block_output", "extraction_position": "last_answer_token",
                "calibration_position": "last_prompt_token", "raw_vector_norms": norms,
                "activation_scales": scales, "runtime": backend.runtime, "environment": environment()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        torch.save({"vectors": directions, "metadata": meta}, temp)
        temp.replace(path)
        write_json(path.with_suffix(".json"), meta)
        return path
    finally:
        backend.close()


def load_vectors(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def check_vectors(artifact, backend, manifest, config):
    meta = artifact["metadata"]
    for key in ("model_id", "revision", "hidden_size", "chat_template_sha256"):
        if meta["model"][key] != backend.identity[key]:
            raise ValueError(f"Vector/model mismatch: {key}")
    if meta["data_fingerprint"] != manifest["fingerprint"] or meta["seed"] != config["seed"]:
        raise ValueError("Vector training data/seed mismatch")
    if meta["config_id"] != config_id(config):
        raise ValueError("Config differs from extraction; use a separate artifact root")


def random_direction(size, seed, layer):
    generator = torch.Generator().manual_seed(seed + 104729 * layer)
    vector = torch.randn(size, generator=generator)
    return vector / vector.norm()
