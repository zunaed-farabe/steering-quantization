"""Go/no-go gate after the GPU check, before the long campaign starts.

Fails if any precision did not finish, produced unusable vectors, or if the
UNSTEERED model's answers are too often unparseable or truncated. A high baseline
invalid rate means accuracy would mostly measure answer formatting, so the prompt
or parser must be fixed (and the fix documented) before spending the main run.
Steered conditions are reported but never gated: their invalid rates are results.
Reads only the GPU-check validation subset; it never touches test data.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_conditions(config, precision):
    nonzero = len({float(a) for a in config["steering"]["strengths"]} - {0.0})
    sources = [s for s in config["steering"]["sources"] if not (s == "native" and precision == config["precisions"][0])]
    return 1 + bool(config["evaluation"]["include_prompt_baseline"]) + nonzero * len(config["steering"]["layers"]) * len(sources)


def gate(config_path, max_invalid=0.05, max_truncation=0.05):
    config = read(config_path)
    root = Path(config["output"]["root"])
    report = {"config": str(config_path), "max_invalid": max_invalid, "max_truncation": max_truncation,
              "precisions": {}, "problems": []}
    runs = {}
    for meta_path in sorted((root / "runs" / "validation").glob("*/run.json")):
        meta = read(meta_path)
        precision = meta["identity"]["model"]["precision"]
        if meta["status"] != "complete":
            report["problems"].append(f"{precision}: incomplete run {meta_path.parent.name}")
            continue
        runs.setdefault(precision, []).append((meta["identity"]["condition"], meta_path.parent))
    for precision in config["precisions"]:
        entry = report["precisions"][precision] = {}
        vector = root / "vectors" / f"{precision}.json"
        if not vector.exists():
            report["problems"].append(f"{precision}: no extracted vector ({vector})")
        else:
            meta = read(vector)
            values = list(meta["raw_vector_norms"].values()) + list(meta["activation_scales"].values())
            entry["activation_scales"] = meta["activation_scales"]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values):
                report["problems"].append(f"{precision}: non-finite or zero vector norm/activation scale")
        done = runs.get(precision, [])
        entry["complete_runs"] = len(done)
        if len(done) != expected_conditions(config, precision):
            report["problems"].append(f"{precision}: {len(done)} complete runs, expected {expected_conditions(config, precision)}")
        entry["conditions"] = {}
        for condition, directory in done:
            rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            invalid = [r for r in rows if r["prediction"] is None]
            truncated = [r for r in rows if r.get("truncated")]
            name = condition["source"] if condition["layer"] is None else f"{condition['source']} alpha={condition['strength']}"
            entry["conditions"][name] = {
                "n": len(rows), "invalid": len(invalid), "truncated": len(truncated),
                "invalid_rate": len(invalid) / len(rows), "truncation_rate": len(truncated) / len(rows),
                "most_common_outputs": Counter(r["raw_output"] for r in rows).most_common(5),
                "invalid_examples": [r["raw_output"] for r in invalid[:5]]}
        base = entry["conditions"].get("none")
        if base is None:
            report["problems"].append(f"{precision}: no unsteered baseline run")
        else:
            if base["invalid_rate"] > max_invalid:
                report["problems"].append(f"{precision}: unsteered invalid rate {base['invalid_rate']:.3f} > {max_invalid} "
                                          f"(examples: {base['invalid_examples'][:3]})")
            if base["truncation_rate"] > max_truncation:
                report["problems"].append(f"{precision}: unsteered truncation rate {base['truncation_rate']:.3f} > {max_truncation}")
    report["passed"] = not report["problems"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Locked GPU-check config")
    parser.add_argument("--max-invalid", type=float, default=0.05)
    parser.add_argument("--max-truncation", type=float, default=0.05)
    parser.add_argument("--output", default="logs/gate.json")
    args = parser.parse_args()
    report = gate(args.config, args.max_invalid, args.max_truncation)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for precision, entry in report["precisions"].items():
        for name, c in entry.get("conditions", {}).items():
            print(f"{precision:5s} {name:22s} n={c['n']:4d} invalid={c['invalid_rate']:.3f} truncated={c['truncation_rate']:.3f} "
                  f"top outputs={[o for o, _ in c['most_common_outputs'][:3]]}")
    print("GATE PASSED" if report["passed"] else "GATE FAILED:\n  " + "\n  ".join(report["problems"]))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
