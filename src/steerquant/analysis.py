import csv
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np

from .config import config_id
from .experiments import collect_runs
from .io import ensure_new_directory, write_json
from .metrics import transitions
from .steering import load_vectors, vector_path


def paired_interval(before, after, condition=None, bootstrap=1000, seed=42, outcome="accuracy"):
    """Paired scenario-cluster bootstrap; answer permutations are not independent."""
    b, a = {r["id"]: r for r in before}, {r["id"]: r for r in after}
    if b.keys() != a.keys():
        raise ValueError("Paired comparison requires identical examples")
    by_group = defaultdict(list)
    for key, old in b.items():
        new = a[key]
        if condition and old["context_condition"] != condition:
            continue
        if outcome == "accuracy":
            delta = int(new["prediction"] == new["gold"]) - int(old["prediction"] == old["gold"])
        elif outcome == "stereotypical_error":
            if old["target"] is None:
                continue
            delta = int(new["prediction"] == new["target"] and new["prediction"] != new["gold"]) - int(old["prediction"] == old["target"] and old["prediction"] != old["gold"])
        else:
            raise ValueError("Unknown outcome")
        by_group[old["group"]].append(delta)
    if not by_group:
        return {"delta": None, "ci95": None, "groups": 0}
    sums = np.array([sum(v) for v in by_group.values()], dtype=float)
    counts = np.array([len(v) for v in by_group.values()])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstrap):
        sample = rng.integers(0, len(sums), len(sums))
        values.append(float(sums[sample].sum() / counts[sample].sum()))
    return {"delta": float(sums.sum() / counts.sum()), "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "groups": len(sums), "resamples": bootstrap, "unit": "scenario_group"}


def response_svg(records, path):
    curves = defaultdict(list)
    for r in records:
        if r["layer"] is not None:
            curves[(r["precision"], r["source"], r["layer"])].append((r["strength"], r["ambig_accuracy"]))
    baseline = {r["precision"]: r["ambig_accuracy"] for r in records if r["source"] == "none"}
    for label, points in curves.items():
        if baseline.get(label[0]) is not None:
            points.append((0.0, baseline[label[0]]))
    width, height, left, top, plot_width, plot_height = 1000, 520, 75, 45, 610, 385
    strengths = [x for points in curves.values() for x, y in points]
    lo, hi = min(strengths, default=-0.1), max(strengths, default=0.1)
    if hi == lo:
        hi = lo + 1
    def x(value):
        return left + (value - lo) / (hi - lo) * plot_width
    def y(value):
        return top + (1 - value) * plot_height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#202532">',
             '<text x="75" y="24" font-size="17">Validation response curves — ambiguous-context accuracy</text>']
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.extend([f'<line x1="{left}" x2="{left+plot_width}" y1="{y(tick)}" y2="{y(tick)}" stroke="#ddd"/>',
                      f'<text x="{left-40}" y="{y(tick)+4}">{tick:.2f}</text>'])
    for tick in np.linspace(lo, hi, 5):
        parts.append(f'<text x="{x(tick)-18}" y="{top+plot_height+25}">{tick:.3g}</text>')
    palette = ["#2563eb", "#c2410c", "#15803d", "#9333ea", "#be123c", "#0e7490", "#854d0e"]
    for i, (label, points) in enumerate(sorted(curves.items())):
        points = sorted((a, b) for a, b in points if b is not None)
        color = palette[i % len(palette)]
        coords = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for a, b in points:
            parts.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="3" fill="{color}"/>')
        parts.append(f'<text x="715" y="{55+20*i}" fill="{color}">{escape(str(label))}</text>')
    parts.extend(['<text x="220" y="485">Strength × original-model activation norm</text>',
                  '<text x="75" y="510">Inspect invalid, truncation, and disambiguated accuracy alongside these curves.</text>', '</g></svg>'])
    Path(path).write_text("\n".join(parts))


def summarize(config, destination, split="validation", bootstrap=1000):
    if bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    runs = collect_runs(config, split)
    if not runs:
        raise ValueError("No completed runs")
    dest = ensure_new_directory(destination)
    baselines = {m["identity"]["model"]["precision"]: rows for _, m, rows, _ in runs
                 if m["identity"]["condition"]["source"] == "none"}
    for precision in baselines:
        count = sum(m["identity"]["condition"]["source"] == "none" and m["identity"]["model"]["precision"] == precision
                    for _, m, _, _ in runs)
        if count != 1:
            raise ValueError(f"Expected exactly one {precision} baseline")
    output, detailed = [], []
    for directory, meta, rows, score in runs:
        identity, c = meta["identity"], meta["identity"]["condition"]
        precision = identity["model"]["precision"]
        if precision not in baselines:
            raise ValueError(f"Missing {precision} baseline")
        base = baselines[precision]
        comparison = {name: paired_interval(base, rows, condition, bootstrap, config["seed"])
                      for name, condition in (("overall", None), ("ambig", "ambig"), ("disambig", "disambig"))}
        comparison["stereotypical_error"] = paired_interval(base, rows, None, bootstrap, config["seed"], "stereotypical_error")
        record = {"run_id": meta["run_id"], "precision": precision, "source": c["source"],
                  "layer": c["layer"], "strength": c["strength"],
                  "operating_points": ",".join(c.get("operating_points", [])),
                  "n": score["overall"]["n"], "accuracy": score["overall"]["accuracy"],
                  "ambig_accuracy": score["ambig"]["accuracy"], "disambig_accuracy": score["disambig"]["accuracy"],
                  "invalid_rate": score["overall"]["invalid_rate"], "truncation_rate": score["overall"]["truncation_rate"],
                  "unnecessary_unknown_rate": score["disambig"]["unknown_rate"],
                  "stereotypical_error_rate": score["overall"]["stereotypical_error_rate"],
                  "ambig_bias_score": score["ambig"]["bbq_bias_score"],
                  "disambig_bias_score": score["disambig"]["bbq_bias_score"],
                  "accuracy_delta": comparison["overall"]["delta"],
                  "accuracy_delta_ci95_low": comparison["overall"]["ci95"][0],
                  "accuracy_delta_ci95_high": comparison["overall"]["ci95"][1]}
        output.append(record)
        detailed.append({"run": str(directory), "metrics": score, "paired_vs_own_baseline": comparison,
                         "transitions": transitions(base, rows)})
    geometry = []
    reference = load_vectors(vector_path(config, config["precisions"][0]))
    for precision in config["precisions"]:
        path = vector_path(config, precision)
        if path.exists():
            vectors = load_vectors(path)
            for layer, v in vectors["vectors"].items():
                geometry.append({"precision": precision, "layer": int(layer),
                                 "cosine_to_reference": float(v @ reference["vectors"][layer]),
                                 "raw_norm": vectors["metadata"]["raw_vector_norms"][layer]})
    write_json(dest / "analysis.json", {"schema": 1, "config_id": config_id(config), "split": split,
                                        "runs": detailed, "vector_geometry": geometry,
                                        "caution": "Within-extraction-seed descriptive intervals; not equivalence tests or pooled model-level inference."})
    with (dest / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    if split == "validation":
        response_svg(output, dest / "response.svg")
    lines = ["# Steering under weight quantization", "", f"Split: **{split}**. Config: `{config_id(config)[:16]}`.", "",
             "Synthetic smoke results validate plumbing only; real-model results require a GPU run.", "",
             "| Precision | Source | Layer | Strength | Ambiguous acc. | Disambiguated acc. | Invalid |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in output:
        def fmt(value):
            return f"{value:.3f}" if value is not None else "—"
        lines.append(f"| {r['precision']} | {r['source']} | {r['layer']} | {r['strength']} | {fmt(r['ambig_accuracy'])} | {fmt(r['disambig_accuracy'])} | {fmt(r['invalid_rate'])} |")
    lines += ["", "Paired 95% intervals resample whole scenario groups. Multiple strengths are exploratory; no multiplicity correction or equivalence claim is made.",
              "See analysis.json for bias denominators, group-pair results, answer transitions, and paired intervals. Missing bias targets reduce annotation coverage; unknown and invalid outputs remain explicit."]
    (dest / "report.md").write_text("\n".join(lines) + "\n")
    return dest
