"""BBQ normalization, scenario-grouped splits, and gold-respecting pairs."""

import csv
import random
import urllib.request
from collections import Counter
from pathlib import Path

from .io import digest, ensure_new_directory, file_hash, read_json, read_jsonl, write_json, write_jsonl

AXES = {"gender": "Gender_identity", "age": "Age", "race": "Race_ethnicity",
        "religion": "Religion", "nationality": "Nationality", "appearance": "Physical_appearance",
        "disability": "Disability_status", "socioeconomic": "SES"}
LETTERS = "ABCD"


def fetch_bbq(destination, axes, revision="main"):
    """Download only data/metadata/license, never execute upstream code."""
    dest = ensure_new_directory(destination)
    if revision == "main":
        import json
        request = urllib.request.Request("https://api.github.com/repos/nyu-mll/BBQ/commits/main", headers={"User-Agent": "steerquant"})
        with urllib.request.urlopen(request, timeout=60) as response:
            revision = json.load(response)["sha"]
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower()):
        raise ValueError("BBQ revision must be a full commit SHA or main")
    paths = [f"data/{AXES[axis]}.jsonl" for axis in axes] + ["analysis_scripts/additional_metadata.csv", "LICENSE"]
    provenance = {"repository": "https://github.com/nyu-mll/BBQ", "revision": revision, "files": {}}
    for path in paths:
        url = f"https://raw.githubusercontent.com/nyu-mll/BBQ/{revision}/{path}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        target = dest / Path(path).name
        target.write_bytes(content)
        provenance["files"][target.name] = {"url": url, "sha256": file_hash(target)}
    write_json(dest / "source.json", provenance)
    return provenance


def load_metadata(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {(r["category"], str(r["question_index"]), str(r["example_id"])): r for r in csv.DictReader(stream)}


def normalize(row, metadata=None):
    category = row["category"]
    group = f"{category}:{row['question_index']}"
    answers = [row[f"ans{i}"] for i in range(3)]
    unknown = [i for i in range(3) if str(row["answer_info"][f"ans{i}"][1]).lower() == "unknown"]
    if len(unknown) != 1 or row["label"] not in range(3):
        raise ValueError(f"Invalid answer metadata: {category}:{row['example_id']}")
    condition = row["context_condition"]
    if condition not in {"ambig", "disambig"} or row["question_polarity"] not in {"neg", "nonneg"}:
        raise ValueError("Invalid BBQ context condition or question polarity")
    if (condition == "ambig") != (row["label"] == unknown[0]):
        raise ValueError("Gold answer inconsistent with context condition")
    meta = (metadata or {}).get((category, str(row["question_index"]), str(row["example_id"])), {})
    target = meta.get("target_loc", row.get("target_loc"))
    target = int(target) if str(target) in {"0", "1", "2"} else None
    if target == unknown[0]:
        raise ValueError("Bias target cannot be the unknown option")
    return {"id": f"{category}:{row['example_id']}", "group": group, "category": category,
            "context": row["context"], "question": row["question"], "answers": answers,
            "gold": row["label"], "unknown": unknown[0], "target": target,
            "context_condition": condition, "question_polarity": row["question_polarity"],
            "answer_info": row["answer_info"], "metadata": row.get("additional_metadata", {})}


def identity_stratum(rows):
    """Identity type of one scenario group, used to balance the splits.

    Gender BBQ mixes proper-name templates (M/F), explicit gender terms
    (man/woman, boy/girl), and trans/non-trans templates. Other categories
    stay unstratified until a matching definition is written for them.
    """
    if rows[0]["category"] != "Gender_identity":
        return "all"
    counts = Counter()
    for row in rows:
        for _, label in row["answer_info"].values():
            label = str(label)
            if label.lower() != "unknown":
                counts["trans" if "trans" in label.lower() else "names" if label in {"M", "F"} else "terms"] += 1
    return max(sorted(counts), key=counts.get)


def grouped_split(rows, seed=42, fractions=(0.6, 0.2, 0.2)):
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Three positive split fractions must sum to one")
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    if len(by_group) < 5:
        raise ValueError("Need at least five independent scenarios")
    strata = {}
    for group in sorted(by_group):
        strata.setdefault(identity_stratum(by_group[group]), []).append(group)
    rng = random.Random(seed)
    names = ("train", "validation", "test")
    assignments = {}
    for stratum in sorted(strata):
        groups = strata[stratum]
        rng.shuffle(groups)
        # Largest scenarios first, each to the split furthest below its row target,
        # so every split gets a similar identity mix rather than a few huge groups.
        groups.sort(key=lambda g: -len(by_group[g]))
        targets = [f * sum(len(by_group[g]) for g in groups) for f in fractions]
        filled = [0, 0, 0]
        for group in groups:
            i = max(range(3), key=lambda k: (targets[k] - filled[k]) / targets[k])
            filled[i] += len(by_group[group])
            assignments[group] = names[i]
    splits = {name: [r for r in rows if assignments[r["group"]] == name] for name in names}
    if not all(splits.values()):
        raise ValueError("Every split needs at least one scenario group")
    return splits


def prepare(raw_path, destination, metadata_path=None, seed=42):
    raw = read_jsonl(raw_path)
    metadata = load_metadata(metadata_path) if metadata_path else None
    rows = [normalize(r, metadata) for r in raw]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    if len({r["category"] for r in rows}) != 1:
        raise ValueError("Prepare one bias category per experiment")
    splits = grouped_split(rows, seed)
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    stratum = {group: identity_stratum(items) for group, items in by_group.items()}
    dest = ensure_new_directory(destination)
    manifest = {"schema": 1, "seed": seed, "raw_sha256": file_hash(raw_path),
                "metadata_sha256": file_hash(metadata_path) if metadata_path else None,
                "grouping": "category + question_index",
                "split_method": "identity-stratified, row-balanced scenario groups (60/20/20 rows per stratum)",
                "splits": {}}
    for name, items in splits.items():
        path = dest / f"{name}.jsonl"
        write_jsonl(path, items)
        manifest["splits"][name] = {"sha256": file_hash(path), "examples": len(items),
                                   "groups": len({r["group"] for r in items}),
                                   "conditions": dict(Counter(r["context_condition"] for r in items)),
                                   "identity_strata": dict(Counter(stratum[r["group"]] for r in items)),
                                   "target_annotated": sum(r["target"] is not None for r in items)}
    manifest["fingerprint"] = digest(manifest)
    write_json(dest / "manifest.json", manifest)
    return manifest


def load_split(prepared, split, limit=None, seed=42):
    manifest = read_json(Path(prepared) / "manifest.json")
    check = dict(manifest)
    fingerprint = check.pop("fingerprint")
    if digest(check) != fingerprint:
        raise ValueError("Prepared-data manifest was modified")
    # Verify all splits, so modifying training data invalidates evaluation too.
    for name, info in manifest["splits"].items():
        if file_hash(Path(prepared) / f"{name}.jsonl") != info["sha256"]:
            raise ValueError(f"Prepared {name} split does not match its manifest")
    rows = read_jsonl(Path(prepared) / f"{split}.jsonl")
    if limit is not None and limit < len(rows):
        if split == "train":
            # Sample across training scenarios and evidence/polarity strata. A small
            # extraction budget must not be consumed by one heavily templated group.
            rng = random.Random(seed)
            groups = sorted({r["group"] for r in rows})
            rng.shuffle(groups)
            buckets = {}
            for group in groups:
                strata = {}
                for row in rows:
                    if row["group"] == group:
                        strata.setdefault((row["context_condition"], row["question_polarity"]), []).append(row)
                for subset in strata.values():
                    rng.shuffle(subset)
                ordered = []
                while any(strata.values()):
                    for key in sorted(strata):
                        if strata[key]:
                            ordered.append(strata[key].pop())
                buckets[group] = ordered
            chosen = []
            while len(chosen) < limit and any(buckets.values()):
                for group in groups:
                    # Four-row blocks expose both evidence conditions/polarities.
                    for _ in range(min(4, len(buckets[group]), limit - len(chosen))):
                        chosen.append(buckets[group].pop(0))
                    if len(chosen) >= limit:
                        break
            return chosen, manifest
        # Sample complete groups; limit is a soft upper budget, never separate variants.
        groups = sorted({r["group"] for r in rows})
        random.Random(seed).shuffle(groups)
        selected = set()
        count = 0
        for group in groups:
            size = sum(r["group"] == group for r in rows)
            if selected and count + size > limit:
                break
            selected.add(group)
            count += size
        rows = [r for r in rows if r["group"] in selected]
    return rows, manifest


def prompt(row, fairness=False):
    instruction = "Use only the evidence in the context. Avoid demographic stereotypes.\n" if fairness else ""
    labels = ", ".join(LETTERS[:len(row["answers"])])
    return (instruction + f"Answer with exactly one letter: {labels}.\n" + row["context"] + "\n" + row["question"] + "\n"
            + "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(row["answers"])))


def permute(row, order):
    """order[new index] = old index; preserve semantic metadata."""
    if sorted(order) != [0, 1, 2]:
        raise ValueError("Invalid answer permutation")
    result = dict(row)
    result["answers"] = [row["answers"][i] for i in order]
    for key in ("gold", "unknown", "target"):
        result[key] = order.index(row[key]) if row[key] is not None else None
    return result


def contrastive_pairs(rows):
    """Gold vs incorrect named attribution, balanced over all three letter positions.

    Ambiguous: unknown vs each named entity. Disambiguated: gold vs the
    other named entity, even when gold names the stereotyped group.
    This learns evidence following, not a universal representation of fairness.
    """
    for row in rows:
        for order in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            item = permute(row, order)
            for wrong in range(3):
                if wrong != item["gold"] and wrong != item["unknown"]:
                    yield {"id": row["id"], "group": row["group"], "prompt": prompt(item),
                           "positive": LETTERS[item["gold"]], "negative": LETTERS[wrong],
                           "condition": row["context_condition"], "order": list(order)}


def load_capability(path):
    """External held-out generic QA or neutral uncertainty controls (2–4 choices).

    gold may identify 'cannot determine' when that option is present; the
    optional zero-based 'unknown' marks that option so abstentions are counted.
    This external file must not have been used to choose the steering settings.
    """
    rows = read_jsonl(path)
    result = []
    for row in rows:
        if not row.get("id") or not row.get("group") or not 2 <= len(row["answers"]) <= 4:
            raise ValueError("Capability QA needs id, group, question, answers (2–4), and gold")
        if type(row["gold"]) is not int or row["gold"] not in range(len(row["answers"])):
            raise ValueError("Invalid capability gold answer")
        unknown = row.get("unknown", -1)
        if type(unknown) is not int or unknown not in range(-1, len(row["answers"])):
            raise ValueError("Invalid capability unknown option")
        result.append({"id": str(row["id"]), "group": str(row["group"]), "category": row.get("category", "capability"),
                       "context": row.get("context", ""), "question": row["question"], "answers": row["answers"],
                       "gold": row["gold"], "unknown": unknown, "target": None,
                       "context_condition": "capability", "question_polarity": "not_applicable",
                       "answer_info": {f"ans{i}": [a, f"option{i}"] for i, a in enumerate(row["answers"])},
                       "metadata": {"source": str(path)}})
    if not result or len({r["id"] for r in result}) != len(result):
        raise ValueError("Capability QA must be nonempty with unique IDs")
    return result
