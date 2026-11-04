"""Descriptive bias scan: unsteered and prompt-instruction baselines per category and model.

Reads complete validation runs from any configs (baseline-only scans or dose sweeps) and
reports how much stereotyped answering there is to remove. It never chooses settings.
"""

from pathlib import Path

from .experiments import collect_runs
from .io import write_json

# Below this many validation scenario groups the scenario-group bootstrap has too few
# units to be meaningful, so a category is described but not eligible for a dose test.
MIN_GROUPS = 8


def describe(rows, score):
    disambig = [r for r in rows if r["context_condition"] == "disambig" and r["target"] is not None]
    pro = [r for r in disambig if r["gold"] == r["target"]]
    anti = [r for r in disambig if r["gold"] != r["target"]]
    def accuracy(subset):
        return sum(r["prediction"] == r["gold"] for r in subset) / len(subset) if subset else None
    gap = accuracy(pro) - accuracy(anti) if pro and anti else None
    stereotyped = [r for r in rows if r["target"] is not None and r["prediction"] == r["target"] and r["prediction"] != r["gold"]]
    return {"n": len(rows), "groups": len({r["group"] for r in rows}),
            "stereotyped_answers": len(stereotyped), "stereotyped_rate": score["overall"]["stereotypical_error_rate"],
            "stereotyped_in_ambig": sum(r["context_condition"] == "ambig" for r in stereotyped),
            "ambig_accuracy": score["ambig"]["accuracy"], "disambig_accuracy": score["disambig"]["accuracy"],
            "disambig_unknown_rate": score["disambig"]["unknown_rate"],
            "pro_minus_anti_accuracy": gap, "ambig_bias_score": score["ambig"]["bbq_bias_score"],
            "invalid_rate": score["overall"]["invalid_rate"], "truncation_rate": score["overall"]["truncation_rate"],
            "letter_with_option_answers": sum(r.get("answer_format") == "letter_with_option" for r in rows)}


def scan(configs, output):
    records = []
    for config in configs:
        runs = collect_runs(config, "validation")
        found = {}
        for _, meta, rows, score in runs:
            condition = meta["identity"]["condition"]
            if condition["layer"] is None and meta["identity"]["model"]["precision"] == config["precisions"][0]:
                found[condition["source"]] = (rows, score)
        if "none" not in found:
            raise ValueError(f"No complete unsteered baseline for {config['output']['root']}")
        rows = found["none"][0]
        record = {"model": config["model"]["id"], "category": rows[0]["category"], "root": config["output"]["root"],
                  "unsteered": describe(*found["none"])}
        if "prompt" in found:
            record["prompt"] = describe(*found["prompt"])
        record["eligible_for_dose_test"] = record["unsteered"]["groups"] >= MIN_GROUPS
        records.append(record)
    records.sort(key=lambda r: (r["category"], r["model"]))
    result = {"schema": 1, "split": "validation", "min_groups_for_dose_test": MIN_GROUPS, "records": records}
    output = Path(output)
    write_json(output, result)
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return {"records": len(records), "output": str(output)}


def markdown(result):
    def pct(value):
        return "—" if value is None else f"{100 * value:.1f}%"
    def signed(value):
        return "—" if value is None else f"{100 * value:+.1f}"
    lines = ["# Bias scan: unsteered validation baselines", "",
             "Stereotyped answer: matches BBQ's stereotype-consistent target and is wrong. Gap: accuracy on "
             "disambiguated questions whose answer fits the stereotype minus those whose answer goes against it, in points.",
             "",
             "| Category | Model | Questions | Scenarios | Stereotyped (ambiguous) | Ambiguous acc. | Disambiguated acc. | Gap | "
             "Unnecessary unknown | Invalid | Prompt: stereotyped | Prompt: disambiguated acc. | Dose-testable |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["records"]:
        u, p = r["unsteered"], r.get("prompt")
        lines.append(f"| {r['category']} | {r['model'].split('/')[-1]} | {u['n']} | {u['groups']} | "
                     f"{u['stereotyped_answers']} ({u['stereotyped_in_ambig']}), {pct(u['stereotyped_rate'])} | "
                     f"{pct(u['ambig_accuracy'])} | {pct(u['disambig_accuracy'])} | {signed(u['pro_minus_anti_accuracy'])} | "
                     f"{pct(u['disambig_unknown_rate'])} | {pct(u['invalid_rate'])} | "
                     f"{pct(p['stereotyped_rate']) if p else '—'} | {pct(p['disambig_accuracy']) if p else '—'} | "
                     f"{'yes' if r['eligible_for_dose_test'] else 'no'} |")
    lines += ["", f"Dose-testable means at least {result['min_groups_for_dose_test']} validation scenarios, the minimum for the "
              "scenario-group bootstrap. Categories below it are descriptive only: their intervals would rest on 3–5 scenarios.", ""]
    return "\n".join(lines)
