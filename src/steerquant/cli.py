import argparse
import json
import sys
from pathlib import Path

from .config import artifact_root, load_config
from .io import write_json


def parser():
    p = argparse.ArgumentParser(description="Bias-reduction steering under weight quantization")
    commands = p.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-bbq", help="Download pinned official BBQ data and scoring metadata")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--axes", nargs="+", choices=["gender", "age", "race", "religion", "nationality", "appearance", "disability", "socioeconomic"], default=["gender"])
    fetch.add_argument("--revision", default="main")
    prep = commands.add_parser("prepare", help="Create scenario-grouped train/validation/test splits")
    prep.add_argument("--raw", required=True)
    prep.add_argument("--metadata")
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=42)
    for name, help_text in (("lock", "Resolve an immutable model revision"), ("plan", "Inspect configuration and evaluation budget"),
                            ("extract", "Extract and save per-precision steering vectors"), ("evaluate", "Run a validation sweep or frozen held-out settings"),
                            ("select", "Freeze settings selected on validation"), ("summarize", "Produce CSV, JSON, Markdown, and validation curves")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True)
        if name in {"lock", "select", "summarize"}:
            sub.add_argument("--output", required=True)
        if name in {"extract", "evaluate"}:
            sub.add_argument("--precision", choices=["all", "bf16", "fp32", "int8", "nf4"], default="all")
        if name in {"evaluate", "summarize"}:
            sub.add_argument("--split", choices=["validation", "test", "capability"], default="validation")
        if name == "evaluate":
            sub.add_argument("--selection")
            sub.add_argument("--qa-file", help="Held-out generic/neutral QA JSONL for capability split")
            sub.add_argument("--resume", action="store_true")
        if name == "select":
            sub.add_argument("--accuracy-budget", type=float, default=0.02)
            sub.add_argument("--invalid-budget", type=float, default=0.01)
            sub.add_argument("--unknown-budget", type=float, default=0.02)
        if name == "summarize":
            sub.add_argument("--bootstrap", type=int, default=1000)
    smoke = commands.add_parser("smoke", help="Offline tiny random-model end-to-end integration test")
    smoke.add_argument("--output", required=True)
    return p


def dispatch(args):
    if args.command == "fetch-bbq":
        from .data import fetch_bbq
        return fetch_bbq(args.output, args.axes, args.revision)
    if args.command == "prepare":
        from .data import prepare
        return prepare(args.raw, args.output, args.metadata, args.seed)
    if args.command == "smoke":
        from .smoke import run_smoke
        return str(run_smoke(args.output))
    config = load_config(args.config)
    if args.command == "lock":
        from .models import lock_config
        if Path(args.output).exists():
            raise ValueError("Locked config already exists; choose a new path")
        write_json(args.output, lock_config(config))
        return args.output
    if args.command == "plan":
        from .experiments import validation_conditions
        from .data import load_split
        manifest = None
        count = None
        if Path(config["data"]["prepared"]).exists():
            rows, manifest = load_split(config["data"]["prepared"], "validation", config["data"]["validation_limit"], config["seed"])
            count = len(rows)
        cells = {precision: len(validation_conditions(config, precision)) for precision in config["precisions"]}
        return {"config": config, "validation_conditions": cells,
                "validation_generations": sum(cells.values()) * count if count else None,
                "data_fingerprint": manifest["fingerprint"] if manifest else None,
                "note": "Single device per process; quantized runs require CUDA. Activation quantization is not implemented in v1."}
    if args.command in {"extract", "evaluate"}:
        from .experiments import evaluate
        from .steering import extract
        precisions = config["precisions"] if args.precision == "all" else [args.precision]
        if any(p not in config["precisions"] for p in precisions):
            raise ValueError("Precision is not enabled in config")
        result = []
        for precision in precisions:
            if args.command == "extract":
                result.append(str(extract(config, precision)))
            else:
                result.extend(evaluate(config, precision, args.split, args.selection, args.resume, args.qa_file))
        return result
    if args.command == "select":
        from .experiments import select
        return select(config, args.output, args.accuracy_budget, args.invalid_budget, args.unknown_budget)
    if args.command == "summarize":
        from .analysis import summarize
        return str(summarize(config, args.output, args.split, args.bootstrap))
    raise ValueError("Unknown command")


def main():
    args = parser().parse_args()
    try:
        result = dispatch(args)
    except (ValueError, OSError) as exc:
        print(f"steerquant: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
