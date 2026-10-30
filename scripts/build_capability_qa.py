"""Build the held-out capability / neutral-uncertainty QA file used at frozen settings.

Two demographically neutral parts, 2,000 questions in total:
  mmlu                 1,000 MMLU test questions (4 options), evenly spread over 52 subjects.
  squad_answerable       500 SQuAD 2.0 dev questions: correct span, a distractor span from the
                           same passage, and "Cannot be determined" (an unnecessary abstention).
  squad_unanswerable     500 SQuAD 2.0 dev unanswerable questions: the plausible-but-wrong span,
                           a distractor span, and "Cannot be determined" (the correct answer).
Social-attitude MMLU subjects and SQuAD articles about ethnic/religious/social groups are excluded,
and any question whose question or options mention demographic terms is dropped.
Downloads are verified against pinned SHA-256 hashes; the build is deterministic for a seed.
Standard library only. Usage:
  python scripts/build_capability_qa.py --downloads data/downloads --output data/capability
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import string
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

SOURCES = {
    "mmlu-data.tar": ("https://people.eecs.berkeley.edu/~hendrycks/data.tar",
                      "bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b"),
    "mmlu-LICENSE": ("https://raw.githubusercontent.com/hendrycks/test/master/LICENSE",
                     "5d841b9eee8bf0d721359b79fbae956a8b426d3139bfefee32149c7d4f2a1a9c"),
    "squad-dev-v2.0.json": ("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
                            "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"),
}
MMLU_EXCLUDED = ["human_sexuality", "moral_disputes", "moral_scenarios", "sociology", "world_religions"]
SQUAD_EXCLUDED = ["Black_Death", "Economic_inequality", "Huguenot", "Imperialism", "Islamism", "Normans", "Yuan_dynasty"]
DEMOGRAPHIC = re.compile(
    r"\b(wom[ae]n|m[ae]n|females?|males?|genders?|sex|sexual|sexuality|girls?|boys?|wi(fe|ves)|husbands?|"
    r"mothers?|fathers?|daughters?|sons?|races?|racial|racism|ethnic|ethnicity|blacks?|whites?|africans?|"
    r"asians?|hispanics?|latin[oa]s?|arabs?|jews?|jewish|muslims?|islam|islamic|christians?|catholics?|"
    r"protestants?|hindus?|buddhists?|religions?|religious|gay|lesbian|homosexual|transgender|immigrants?|"
    r"immigration|slaves?|slavery|natives?|indigenous|tribes?|tribal|elderly|disabled|disability|poor|"
    r"wealthy|minorit(y|ies))\b", re.IGNORECASE)
UNKNOWN = "Cannot be determined"
PERMUTATIONS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(downloads):
    downloads.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = downloads / name
        if not path.exists():
            print(f"downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=300) as response, open(path, "wb") as out:
                shutil.copyfileobj(response, out)
        if sha256(path) != expected:
            raise ValueError(f"{name} does not match its pinned SHA-256; upstream changed or the download is corrupt")


def neutral(*texts):
    return not any(DEMOGRAPHIC.search(t) for t in texts)


def allocate(capacity, total, rng):
    """Spread `total` as evenly as possible over groups, never above a group's capacity."""
    if sum(capacity.values()) < total:
        raise ValueError("Not enough eligible questions")
    quota = dict.fromkeys(capacity, 0)
    order = sorted(capacity)
    rng.shuffle(order)
    while total:
        for group in order:
            if total and quota[group] < capacity[group]:
                quota[group] += 1
                total -= 1
    return quota


def mmlu_rows(tar_path, count, rng):
    eligible = {}
    with tarfile.open(tar_path) as tar:  # read in memory; nothing is extracted to disk
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not (member.isfile() and member.name.startswith("data/test/") and member.name.endswith("_test.csv")):
                continue
            subject = member.name.split("/")[-1][:-len("_test.csv")]
            if subject in MMLU_EXCLUDED:
                continue
            rows = csv.reader(io.TextIOWrapper(tar.extractfile(member), encoding="utf-8"))
            eligible[subject] = [(i, r) for i, r in enumerate(rows)
                                 if len(r) == 6 and r[5] in "ABCD" and neutral(*r[:5])]
    seen = set()  # MMLU repeats some questions, sometimes across subjects; keep the first copy
    for subject in sorted(eligible):
        unique = []
        for i, r in eligible[subject]:
            key = (norm(r[0]), tuple(sorted(norm(x) for x in r[1:5])))
            if key not in seen:
                seen.add(key)
                unique.append((i, r))
        eligible[subject] = unique
    quota = allocate({s: len(v) for s, v in eligible.items()}, count, rng)
    result = []
    for subject in sorted(eligible):
        for i, r in sorted(rng.sample(eligible[subject], quota[subject])):
            result.append({"id": f"mmlu:{subject}:{i}", "group": f"mmlu:{subject}", "category": "mmlu",
                           "question": r[0], "answers": r[1:5], "gold": "ABCD".index(r[5])})
    return result


def norm(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def overlaps(a, b):
    return a == b or a in b or b in a


def mode(texts):
    counts = Counter(texts)
    return max(texts, key=lambda t: (counts[t], -texts.index(t)))


def squad_candidates(squad_path, rng):
    data = json.loads(Path(squad_path).read_text(encoding="utf-8"))
    candidates = {"squad_answerable": {}, "squad_unanswerable": {}}
    seen = set()
    for article in data["data"]:
        title = article["title"]
        if title in SQUAD_EXCLUDED:
            continue
        for paragraph in article["paragraphs"]:
            spans = sorted({a["text"].strip() for q in paragraph["qas"]
                            for a in q.get("answers", []) + q.get("plausible_answers", []) if norm(a["text"])})
            for q in paragraph["qas"]:
                impossible = q["is_impossible"]
                key = "plausible_answers" if impossible else "answers"
                variants = [a["text"].strip() for a in q.get(key, []) if norm(a["text"])]
                if not variants or (paragraph["context"], norm(q["question"])) in seen:
                    continue
                seen.add((paragraph["context"], norm(q["question"])))
                main = mode(variants)
                blocked = [norm(v) for v in variants]
                distractors = [s for s in spans if not any(overlaps(norm(s), b) for b in blocked)]
                if not distractors:
                    continue
                distractor = rng.choice(distractors)
                if not neutral(q["question"], main, distractor):
                    continue
                category = "squad_unanswerable" if impossible else "squad_answerable"
                candidates[category].setdefault(title, []).append(
                    {"id": f"squad:{q['id']}", "group": f"squad:{title}", "category": category,
                     "context": paragraph["context"], "question": q["question"].strip(),
                     "options": [main, distractor, UNKNOWN]})
    return candidates


def squad_rows(candidates, count, rng):
    result = []
    for category in sorted(candidates):
        by_title = candidates[category]
        quota = allocate({t: len(v) for t, v in by_title.items()}, count, rng)
        chosen = []
        for title in sorted(by_title):
            chosen += rng.sample(sorted(by_title[title], key=lambda r: r["id"]), quota[title])
        rng.shuffle(chosen)
        for n, row in enumerate(chosen):
            # Cycle all six orders so gold and "Cannot be determined" are balanced over A/B/C.
            order = PERMUTATIONS[n % 6]
            answers = [row["options"][i] for i in order]
            unknown = answers.index(UNKNOWN)
            gold = unknown if category == "squad_unanswerable" else answers.index(row["options"][0])
            result.append({"id": row["id"], "group": row["group"], "category": category, "context": row["context"],
                           "question": row["question"], "answers": answers, "gold": gold, "unknown": unknown})
    return sorted(result, key=lambda r: r["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--downloads", default="data/downloads")
    parser.add_argument("--output", default="data/capability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmlu", type=int, default=1000)
    parser.add_argument("--squad-per-kind", type=int, default=500)
    args = parser.parse_args()
    downloads, output = Path(args.downloads), Path(args.output)
    if output.exists() and any(output.iterdir()):
        sys.exit(f"Refusing to overwrite nonempty {output}")
    fetch(downloads)
    rng = random.Random(args.seed)
    rows = mmlu_rows(downloads / "mmlu-data.tar", args.mmlu, rng)
    rows += squad_rows(squad_candidates(downloads / "squad-dev-v2.0.json", rng), args.squad_per_kind, rng)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    output.mkdir(parents=True, exist_ok=True)
    qa = output / "qa.jsonl"
    with open(qa, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    shutil.copyfile(downloads / "mmlu-LICENSE", output / "LICENSE-MMLU.txt")
    summary = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        summary[category] = {"questions": len(subset), "groups": len({r["group"] for r in subset}),
                             "gold_letters": dict(sorted(Counter("ABCD"[r["gold"]] for r in subset).items())),
                             "unknown_letters": dict(sorted(Counter("ABCD"[r["unknown"]] for r in subset if "unknown" in r).items()))}
    manifest = {"schema": 1, "seed": args.seed, "file": "qa.jsonl", "sha256": sha256(qa), "questions": len(rows),
                "parts": summary,
                "sources": {name: {"url": url, "sha256": digest} for name, (url, digest) in SOURCES.items()},
                "licenses": {"mmlu": "MIT (LICENSE-MMLU.txt)", "squad_v2_dev": "CC BY-SA 4.0; squad_* rows are adapted from it"},
                "rules": {"mmlu_split": "test", "mmlu_excluded_subjects": MMLU_EXCLUDED,
                          "squad_split": "dev v2.0", "squad_excluded_articles": SQUAD_EXCLUDED,
                          "demographic_filter": DEMOGRAPHIC.pattern, "filter_applies_to": "question and answer options",
                          "unknown_option": UNKNOWN,
                          "squad_distractor": "a span answering another question in the same passage, not overlapping any answer variant",
                          "allocation": "even over groups (MMLU subject, SQuAD article), seeded",
                          "option_order": "SQuAD rows cycle all six orders; MMLU keeps its original order"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: manifest[k] for k in ("questions", "sha256", "parts")}, indent=2))


if __name__ == "__main__":
    main()
