from copy import deepcopy
from pathlib import Path

import yaml

from .io import digest

DEFAULTS = {
    "seed": 42,
    "model": {"id": "Qwen/Qwen2.5-7B-Instruct", "revision": "main", "device": "cuda:0",
              "compute_dtype": "bfloat16", "attention": "eager"},
    "precisions": ["bf16", "int8", "nf4"],
    "data": {"prepared": "data/prepared/gender", "train_limit": 128, "validation_limit": None, "test_limit": None},
    "steering": {"layers": [13], "strengths": [-0.05, -0.025, 0.0, 0.01, 0.025, 0.05, 0.1],
                 "method": "mean_difference", "positions": "last", "sources": ["reference", "native", "random"]},
    "evaluation": {"max_new_tokens": 12, "fairness_prompt": False, "include_prompt_baseline": True},
    "output": {"root": "artifacts/gender-seed42"},
}


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    config = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key not in config:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(config[key], dict):
            if not isinstance(value, dict) or set(value) - set(config[key]):
                raise ValueError(f"Unknown or invalid options in {key}")
            config[key].update(value)
        else:
            config[key] = value
    validate(config)
    return config


def validate(c):
    if type(c["seed"]) is not int or c["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    allowed = {"bf16", "fp32", "int8", "nf4"}
    if not c["precisions"] or set(c["precisions"]) - allowed or len(set(c["precisions"])) != len(c["precisions"]):
        raise ValueError(f"Precisions must be distinct entries from {sorted(allowed)}")
    if c["precisions"][0] not in {"bf16", "fp32"}:
        raise ValueError("The first precision must be the floating-point reference")
    if c["model"]["compute_dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported compute dtype")
    if not c["steering"]["layers"] or any(type(n) is not int or n < 0 for n in c["steering"]["layers"]):
        raise ValueError("Layers are zero-based nonnegative block indices")
    if len(set(c["steering"]["layers"])) != len(c["steering"]["layers"]):
        raise ValueError("Layer indices must be distinct")
    if c["steering"]["method"] != "mean_difference" or c["steering"]["positions"] != "last":
        raise ValueError("v1 implements mean_difference with last-position interventions")
    if set(c["steering"]["sources"]) - {"reference", "native", "random"}:
        raise ValueError("Unknown vector source")
    if len(set(c["steering"]["sources"])) != len(c["steering"]["sources"]):
        raise ValueError("Vector sources must be distinct")
    import math
    if not c["steering"]["strengths"] or any(not math.isfinite(float(a)) for a in c["steering"]["strengths"]):
        raise ValueError("Steering strengths must be finite")
    if c["evaluation"]["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    for key in ("train_limit", "validation_limit", "test_limit"):
        v = c["data"][key]
        if v is not None and (type(v) is not int or v < 1):
            raise ValueError(f"{key} must be positive or null")
    if c["data"]["train_limit"] is not None and c["data"]["train_limit"] < 4:
        raise ValueError("train_limit must be at least four to include both evidence conditions")


def config_id(config):
    return digest(config)


def artifact_root(config):
    return Path(config["output"]["root"])
