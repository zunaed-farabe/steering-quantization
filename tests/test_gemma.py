from copy import deepcopy
from types import SimpleNamespace

import torch
import pytest
from transformers import Gemma2Config, Gemma2ForCausalLM

from steerquant.config import DEFAULTS
from steerquant.models import Backend, blocks, stopping_ids
from steerquant.steering import capture, intervene


def tiny_gemma():
    torch.manual_seed(8)
    torch.set_num_threads(1)
    config = Gemma2Config(vocab_size=24, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, query_pre_attn_scalar=8, sliding_window=32,
                         max_position_embeddings=128, bos_token_id=2, eos_token_id=1, pad_token_id=0)
    config._attn_implementation = "eager"
    return Gemma2ForCausalLM(config).eval()


def test_gemma2_adapter_capture_and_reversible_intervention():
    model = tiny_gemma()
    layers = blocks(model)
    ids = torch.tensor([[2, 3, 5]])
    with capture(layers, [0]) as values, torch.inference_mode():
        result = model(ids, output_hidden_states=True, use_cache=False)
    assert torch.equal(values[0], result.hidden_states[1][0, -1].float())
    baseline = result.logits.clone()
    with intervene(layers[0], torch.arange(32).float(), 0.1), torch.inference_mode():
        changed = model(ids, use_cache=False).logits
    assert not torch.allclose(changed[:, -1], baseline[:, -1])
    with torch.inference_mode():
        assert torch.equal(model(ids, use_cache=False).logits, baseline)
    assert not layers[0]._forward_hooks


def test_checkpoint_end_of_turn_stops_generation_before_token_cap():
    model = tiny_gemma()
    model.generation_config.eos_token_id = [1, 7]
    model.generation_config.forced_bos_token_id = 7
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0,
                                decode=lambda ids, **kwargs: "")
    backend = Backend.__new__(Backend)
    backend.model, backend.tokenizer = model, tokenizer
    backend.config = {"evaluation": {"max_new_tokens": 4}}
    backend.eos_token_ids = stopping_ids(model, tokenizer)
    backend.encode = lambda text: {"input_ids": torch.tensor([[2]]), "attention_mask": torch.ones(1, 1, dtype=torch.long)}
    assert backend.eos_token_ids == [1, 7]
    _, generated = backend.generate("fixture")
    assert generated == [7]


def test_stop_ids_fallback_and_deduplication():
    tokenizer = SimpleNamespace(eos_token_id=1)
    assert stopping_ids(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None)), tokenizer) == [1]
    assert stopping_ids(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 7, 7])), tokenizer) == [1, 7]


@pytest.mark.parametrize("nonzero_accuracy", [0.9, 1.0])
def test_selection_can_reject_every_nonzero_vector(tmp_path, monkeypatch, nonzero_accuracy):
    import steerquant.experiments as experiments
    config = deepcopy(DEFAULTS)
    config["precisions"] = ["bf16"]
    config["steering"].update(layers=[0], strengths=[0, 0.1], sources=["reference", "random"])
    config["evaluation"]["include_prompt_baseline"] = False
    runs = []
    for c in experiments.validation_conditions(config, "bf16"):
        score = {"ambig": {"n": 10, "accuracy": 1.0 if c["source"] == "none" else nonzero_accuracy},
                 "disambig": {"n": 10, "accuracy": 1.0, "unknown_rate": 0.0},
                 "overall": {"invalid_rate": 0.0, "truncation_rate": 0.0}}
        meta = {"identity": {"model": {"precision": "bf16"}, "condition": c},
                "run_id": c["source"], "predictions_sha256": "fixture"}
        runs.append((tmp_path / c["source"], meta, [], score))
    monkeypatch.setattr(experiments, "collect_runs", lambda *args: runs)
    monkeypatch.setattr(experiments, "load_split", lambda *args: ([], {"fingerprint": "fixture"}))
    result = experiments.select(config, tmp_path / "selection.json")
    for choice in result["choices"]:
        if choice["condition"]["source"] != "none":
            assert choice["condition"]["strength"] == 0
            assert choice["operating_points"] == ["no_intervention"]
    assert any(d["status"] == "no_nonzero_reference_selected" for d in result["diagnostics"])


def test_prompt_control_is_carried_to_test_even_outside_the_budgets(tmp_path, monkeypatch):
    import steerquant.experiments as experiments
    config = deepcopy(DEFAULTS)
    config["precisions"] = ["bf16"]
    config["steering"].update(layers=[0], strengths=[0, 0.1], sources=["reference"])
    runs = []
    for c in experiments.validation_conditions(config, "bf16"):
        score = {"ambig": {"n": 10, "accuracy": 1.0},
                 "disambig": {"n": 10, "accuracy": 0.5 if c["source"] == "prompt" else 1.0, "unknown_rate": 0.0},
                 "overall": {"invalid_rate": 0.0, "truncation_rate": 0.0}}
        meta = {"identity": {"model": {"precision": "bf16"}, "condition": c},
                "run_id": c["source"], "predictions_sha256": "fixture"}
        runs.append((tmp_path / c["source"], meta, [], score))
    monkeypatch.setattr(experiments, "collect_runs", lambda *args: runs)
    monkeypatch.setattr(experiments, "load_split", lambda *args: ([], {"fingerprint": "fixture"}))
    result = experiments.select(config, tmp_path / "selection.json")
    assert any(c["condition"]["source"] == "prompt" for c in result["choices"])
