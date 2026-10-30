"""Offline integration exercise. Synthetic data + random tiny model, no bias claims."""

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from .config import load_config
from .data import prepare
from .io import ensure_new_directory, write_json, write_jsonl
from .models import lock_config


def fixture_rows(groups=15):
    rows = []
    for group in range(groups):
        for condition in ("ambig", "disambig"):
            for polarity in ("neg", "nonneg"):
                # Synthetic attributed action, not a social-bias benchmark.
                gold = 2 if condition == "ambig" else (0 if group % 2 else 1)
                rows.append({"example_id": len(rows), "question_index": group, "category": "Synthetic",
                             "context_condition": condition, "question_polarity": polarity,
                             "context": f"Two fictional people discuss object {group}." + (f" Person {gold + 1} moved it." if condition == "disambig" else ""),
                             "question": "Who moved the object?", "ans0": "Person 1", "ans1": "Person 2", "ans2": "Cannot determine",
                             "label": gold, "target_loc": 0 if polarity == "neg" else 1,
                             "answer_info": {"ans0": ["Person 1", "group1"], "ans1": ["Person 2", "group2"], "ans2": ["Cannot determine", "unknown"]},
                             "additional_metadata": {"stereotyped_groups": ["group1"], "synthetic": True}})
    return rows


def setup_smoke(destination):
    dest = ensure_new_directory(destination).resolve()
    checkpoint = dest / "tiny-model"
    checkpoint.mkdir()
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "A": 3, "B": 4, "C": 5, "D": 6,
             "user": 7, "assistant": 8, ":": 9, ".": 10, "Person": 11, "1": 12, "2": 13}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    fast.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    fast.save_pretrained(checkpoint)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=48,
                                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=512, eos_token_id=2, pad_token_id=1))
    model.save_pretrained(checkpoint)
    raw = dest / "synthetic.jsonl"
    write_jsonl(raw, fixture_rows())
    prepare(raw, dest / "prepared", seed=42)
    settings = {"model": {"id": str(checkpoint), "device": "cpu", "compute_dtype": "float32"},
                "precisions": ["fp32"], "data": {"prepared": str(dest / "prepared"), "train_limit": 8,
                                               "validation_limit": 4, "test_limit": 4},
                "steering": {"layers": [0], "strengths": [-0.05, 0.0, 0.05], "sources": ["reference", "random"]},
                "evaluation": {"max_new_tokens": 1}, "output": {"root": str(dest / "artifacts")}}
    write_json(dest / "draft.json", settings)
    config = lock_config(load_config(dest / "draft.json"))
    write_json(dest / "config.json", config)
    return config


def run_smoke(destination):
    from .analysis import summarize
    from .experiments import evaluate, select
    from .steering import extract
    torch.set_num_threads(1)
    config = setup_smoke(destination)
    dest = Path(destination)
    extract(config, "fp32")
    evaluate(config, "fp32")
    # Permissive smoke budgets exercise the workflow on a random model only.
    select(config, dest / "selection.json", accuracy_budget=1.0, invalid_budget=1.0, unknown_budget=1.0)
    evaluate(config, "fp32", "test", dest / "selection.json")
    summarize(config, dest / "validation-report", "validation", bootstrap=100)
    summarize(config, dest / "test-report", "test", bootstrap=100)
    controls = dest / "controls.jsonl"
    write_jsonl(controls, [{"id": "control-1", "group": "control-1", "question": "What is 1 plus 1?",
                            "answers": ["zero", "one", "two", "three"], "gold": 2}])
    evaluate(config, "fp32", "capability", dest / "selection.json", qa_path=controls)
    summarize(config, dest / "capability-report", "capability", bootstrap=100)
    return dest
