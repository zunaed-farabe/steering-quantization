"""Validation sweeps and test evaluation at frozen validation-selected settings."""

import json
from pathlib import Path

import torch

from .config import artifact_root, config_id
from .data import load_capability, load_split, prompt
from .io import digest, environment, file_hash, read_json, read_jsonl, write_json
from .metrics import metrics, parse_answer
from .models import Backend
from .steering import check_vectors, intervene, load_vectors, random_direction, vector_path


def validation_conditions(config, precision):
    conditions = [{"source": "none", "layer": None, "strength": 0.0}]
    if config["evaluation"]["include_prompt_baseline"]:
        conditions.append({"source": "prompt", "layer": None, "strength": 0.0})
    reference = config["precisions"][0]
    for source in config["steering"]["sources"]:
        if source == "native" and precision == reference:
            continue
        for layer in config["steering"]["layers"]:
            for strength in sorted(set(float(a) for a in config["steering"]["strengths"]) - {0.0}):
                conditions.append({"source": source, "layer": layer, "strength": strength})
    return conditions


def read_predictions(run):
    manifest = read_json(Path(run) / "run.json")
    path = Path(run) / "predictions.jsonl"
    if digest(manifest["identity"]) != manifest["run_id"]:
        raise ValueError(f"Modified run identity: {run}")
    if manifest["status"] != "complete" or file_hash(path) != manifest["predictions_sha256"]:
        raise ValueError(f"Incomplete or modified run: {run}")
    rows = read_jsonl(path)
    if len(rows) != manifest["expected_examples"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate or missing predictions: {run}")
    return manifest, rows


def evaluate(config, precision, split="validation", selection_path=None, resume=False, qa_path=None):
    if split not in {"validation", "test", "capability"}:
        raise ValueError("Unknown evaluation split")
    if split == "capability":
        if qa_path is None:
            raise ValueError("Capability evaluation requires --qa-file")
        _, data_manifest = load_split(config["data"]["prepared"], "validation")
        rows = load_capability(qa_path)
    else:
        rows, data_manifest = load_split(config["data"]["prepared"], split, config["data"][f"{split}_limit"], config["seed"])
    if not rows:
        raise ValueError("Empty evaluation split")
    selection_hash = None
    if split != "validation":
        if selection_path is None:
            raise ValueError("Held-out evaluation requires a frozen --selection from validation")
        selection = read_json(selection_path)
        selection_hash = selection.pop("fingerprint")
        if digest(selection) != selection_hash:
            raise ValueError("Selection file was modified")
        if selection["config_id"] != config_id(config) or selection["data_fingerprint"] != data_manifest["fingerprint"]:
            raise ValueError("Selection/config/data mismatch")
        conditions = [{**c["condition"], "operating_points": c["operating_points"]}
                      for c in selection["choices"] if c["precision"] == precision]
        if not conditions:
            raise ValueError("No test conditions for requested precision")
    else:
        conditions = validation_conditions(config, precision)
    reference = config["precisions"][0]
    reference_path = vector_path(config, reference)
    ref_vectors = load_vectors(reference_path)
    native_path = vector_path(config, precision)
    native = load_vectors(native_path) if any(c["source"] == "native" for c in conditions) else None
    if ref_vectors["metadata"]["model"]["precision"] != reference:
        raise ValueError("Reference artifact has the wrong extraction precision")
    if native and native["metadata"]["model"]["precision"] != precision:
        raise ValueError("Native artifact has the wrong extraction precision")
    backend = Backend(config, precision)
    try:
        check_vectors(ref_vectors, backend, data_manifest, config)
        if native:
            check_vectors(native, backend, data_manifest, config)
        env = environment()
        completed = []
        for condition in conditions:
            source, layer, strength = (condition[k] for k in ("source", "layer", "strength"))
            vector, vector_hash, scale = None, None, 0.0
            if layer is not None:
                key = str(layer)
                # Hold original-model activation scale fixed, including native vectors.
                scale = ref_vectors["metadata"]["activation_scales"][key]
                if source == "random":
                    vector = random_direction(backend.identity["hidden_size"], config["seed"], layer)
                    vector_hash = digest(vector.tolist())
                else:
                    chosen = native if source == "native" else ref_vectors
                    vector = chosen["vectors"][key]
                    vector_hash = file_hash(native_path if source == "native" else reference_path)
            identity = {"schema": 1, "config_id": config_id(config), "data_fingerprint": data_manifest["fingerprint"],
                        "examples_sha256": digest(rows), "model": backend.identity, "condition": condition,
                        "split": split, "selection_fingerprint": selection_hash,
                        "capability_file_sha256": file_hash(qa_path) if qa_path else None,
                        "vector_sha256": vector_hash, "reference_vector_sha256": file_hash(reference_path),
                        "magnitude": strength * scale, "source_sha256": env["source_sha256"],
                        "packages": env["packages"], "runtime": backend.runtime}
            run_id = digest(identity)
            directory = artifact_root(config) / "runs" / split / f"{precision}-{source}-{run_id[:16]}"
            directory.mkdir(parents=True, exist_ok=True)
            meta_path, pred_path = directory / "run.json", directory / "predictions.jsonl"
            previous = []
            if meta_path.exists():
                existing = read_json(meta_path)
                if not resume:
                    raise ValueError(f"Run exists; pass --resume: {directory}")
                if existing["identity"] != identity:
                    raise ValueError("Run identity mismatch")
                if existing["status"] == "complete":
                    read_predictions(directory)
                    completed.append(str(directory))
                    continue
                previous = read_jsonl(pred_path) if pred_path.exists() else []
            elif pred_path.exists():
                raise ValueError("Predictions exist without run manifest")
            expected = {r["id"]: r for r in rows}
            if len({r["id"] for r in previous}) != len(previous) or any(r["id"] not in expected for r in previous):
                raise ValueError("Invalid resume predictions")
            for old in previous:
                if any(old[k] != value for k, value in expected[old["id"]].items()):
                    raise ValueError("Resume prediction metadata changed")
            run_manifest = {"identity": identity, "run_id": run_id, "status": "running",
                            "expected_examples": len(rows), "environment": env}
            write_json(meta_path, run_manifest)
            done = {r["id"] for r in previous}
            print(f"evaluate {precision} {source} layer={layer} alpha={strength} {split}", flush=True)
            with pred_path.open("a") as stream:
                for n, row in enumerate(rows):
                    if row["id"] in done:
                        continue
                    target_layer = backend.layers[layer] if layer is not None else None
                    with intervene(target_layer, vector, strength * scale):
                        text, tokens = backend.generate(prompt(row, config["evaluation"]["fairness_prompt"] or source == "prompt"))
                    eos = backend.tokenizer.eos_token_id
                    eos_ids = eos if isinstance(eos, list) else [eos]
                    truncated = len(tokens) >= config["evaluation"]["max_new_tokens"] and bool(tokens) and tokens[-1] not in eos_ids
                    result = {**row, "raw_output": text, "generated_tokens": tokens,
                              "prediction": parse_answer(text, len(row["answers"])), "truncated": truncated}
                    stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    previous.append(result)
                    if (n + 1) % 25 == 0:
                        print(f"  {n + 1}/{len(rows)}", flush=True)
            if len(previous) != len(rows):
                raise ValueError("Missing evaluation predictions")
            scores = metrics(previous)
            write_json(directory / "metrics.json", scores)
            run_manifest.update(status="complete", predictions_sha256=file_hash(pred_path), metrics_sha256=digest(scores))
            write_json(meta_path, run_manifest)
            completed.append(str(directory))
        return completed
    finally:
        backend.close()


def collect_runs(config, split):
    runs = []
    for path in sorted((artifact_root(config) / "runs" / split).glob("*/run.json")):
        meta = read_json(path)
        if meta["status"] != "complete":
            continue
        if meta["identity"]["config_id"] != config_id(config):
            raise ValueError("Artifact root contains another configuration")
        meta, rows = read_predictions(path.parent)
        runs.append((path.parent, meta, rows, metrics(rows)))
    # A report must not combine changed datasets, selections, vectors, or code
    # just because they share an artifact directory and config filename.
    for precision in config["precisions"]:
        subset = [m["identity"] for _, m, _, _ in runs if m["identity"]["model"]["precision"] == precision]
        for key in ("examples_sha256", "selection_fingerprint", "reference_vector_sha256", "source_sha256"):
            if len({digest(item.get(key)) for item in subset}) > 1:
                raise ValueError(f"Mixed campaign artifacts ({precision}, {key}); use a separate artifact root")
    return runs


def select(config, output, accuracy_budget=0.02, invalid_budget=0.01, unknown_budget=0.02):
    if Path(output).exists():
        raise ValueError("Selection is frozen; choose a new path to create another selection")
    if any(not 0 <= v <= 1 for v in (accuracy_budget, invalid_budget, unknown_budget)):
        raise ValueError("Selection budgets must be fractions between zero and one")
    runs = collect_runs(config, "validation")
    if not runs:
        raise ValueError("No complete validation runs")
    _, manifest = load_split(config["data"]["prepared"], "validation")
    choices, diagnostics = [], []
    for precision in config["precisions"]:
        subset = [r for r in runs if r[1]["identity"]["model"]["precision"] == precision]
        baselines = [r for r in subset if r[1]["identity"]["condition"]["source"] == "none"]
        if len(baselines) != 1:
            raise ValueError(f"Expected exactly one complete {precision} baseline")
        baseline = baselines[0]
        base = baseline[3]
        if base["ambig"]["n"] == 0 or base["disambig"]["n"] == 0:
            raise ValueError("Selection requires both evidence conditions")
        desired = validation_conditions(config, precision)
        actual = [r[1]["identity"]["condition"] for r in subset]
        if sorted(map(digest, desired)) != sorted(map(digest, actual)):
            raise ValueError(f"Validation sweep incomplete or duplicated for {precision}")
        for source in sorted({c["source"] for c in desired}):
            candidates = []
            for r in subset:
                if r[1]["identity"]["condition"]["source"] != source:
                    continue
                score = r[3]
                admissible = (score["disambig"]["accuracy"] >= base["disambig"]["accuracy"] - accuracy_budget
                              and score["overall"]["invalid_rate"] <= base["overall"]["invalid_rate"] + invalid_budget
                              and score["disambig"]["unknown_rate"] <= base["disambig"]["unknown_rate"] + unknown_budget
                              and score["overall"]["truncation_rate"] <= base["overall"]["truncation_rate"] + invalid_budget)
                # Baseline and prompt-only control have nothing to tune; always carry them to test.
                if admissible or source in {"none", "prompt"}:
                    candidates.append(r)
            if not candidates:
                diagnostics.append({"precision": precision, "source": source, "status": "no_admissible_setting"})
                continue
            # Avoid rewarding group guessing: prioritize correct uncertainty on ambiguous QA.
            best = max(candidates, key=lambda r: (r[3]["ambig"]["accuracy"], r[3]["disambig"]["accuracy"],
                                                  -abs(r[1]["identity"]["condition"]["strength"])))
            choices.append({"precision": precision, "condition": best[1]["identity"]["condition"],
                            "operating_points": ["baseline" if source in {"none", "prompt"} else "recalibrated"],
                            "validation_run": str(best[0]), "validation_run_id": best[1]["run_id"],
                            "validation_predictions_sha256": best[1]["predictions_sha256"]})
    # Test unchanged deployment as well as per-precision recalibration. Settings
    # are selected ONLY on reference validation, then copied to quantized receivers.
    original = next((c for c in choices if c["precision"] == config["precisions"][0]
                     and c["condition"]["source"] == "reference"), None)
    if original:
        for precision in config["precisions"]:
            existing = next((c for c in choices if c["precision"] == precision and c["condition"] == original["condition"]), None)
            if existing:
                existing["operating_points"].append("unchanged_reference")
            else:
                choices.append({**original, "precision": precision, "operating_points": ["unchanged_reference"]})
    result = {"schema": 1, "config_id": config_id(config), "data_fingerprint": manifest["fingerprint"],
              "rule": "maximize ambiguous accuracy subject to disambiguated accuracy, unknown, invalid and truncation budgets",
              "budgets": {"accuracy_drop": accuracy_budget, "invalid_increase": invalid_budget, "unknown_increase": unknown_budget},
              "choices": choices, "diagnostics": diagnostics}
    result["fingerprint"] = digest(result)
    write_json(output, result)
    return result
