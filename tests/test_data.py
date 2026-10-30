from collections import Counter

import pytest

from steerquant.data import contrastive_pairs, grouped_split, load_split, normalize, prepare
from steerquant.io import write_jsonl


def test_related_scenarios_never_cross_splits(raw_rows):
    rows = [normalize(r) for r in raw_rows]
    splits = grouped_split(rows, seed=11)
    groups = [{r["group"] for r in split} for split in splits.values()]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    assert sum(map(len, splits.values())) == len(rows)
    assert splits == grouped_split(rows, seed=11)
    for split in splits.values():
        assert set(r["context_condition"] for r in split) == {"ambig", "disambig"}


def test_gold_stereotyped_group_remains_positive(raw_rows):
    rows = [normalize(r) for r in raw_rows]
    stereotyped_correct = next(r for r in rows if r["gold"] == r["target"])
    pairs = list(contrastive_pairs([stereotyped_correct]))
    assert len(pairs) == 3
    assert Counter(p["positive"] for p in pairs) == {"A": 1, "B": 1, "C": 1}
    for pair in pairs:
        assert pair["positive"] != pair["negative"]
        assert pair["positive"] == "ABC"[pair["order"].index(stereotyped_correct["gold"])]


def test_ambiguous_pairs_abstain_without_letter_prior(raw_rows):
    row = normalize(raw_rows[0])
    pairs = list(contrastive_pairs([row]))
    assert len(pairs) == 6
    assert Counter(p["positive"] for p in pairs) == {"A": 2, "B": 2, "C": 2}
    assert Counter(p["negative"] for p in pairs) == {"A": 2, "B": 2, "C": 2}


def test_manifest_tampering_and_group_budget(raw_rows, tmp_path):
    raw, dest = tmp_path / "raw.jsonl", tmp_path / "prepared"
    write_jsonl(raw, raw_rows)
    manifest = prepare(raw, dest)
    rows, _ = load_split(dest, "validation", limit=5)
    assert len(rows) == 4  # Never take a partial four-variant scenario.
    assert manifest["splits"]["train"]["target_annotated"] > 0
    with (dest / "test.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="does not match"):
        load_split(dest, "train")


def test_invalid_gold_evidence_rejected(raw_rows):
    bad = dict(raw_rows[0], label=0)
    with pytest.raises(ValueError, match="inconsistent"):
        normalize(bad)


def test_duplicate_ids_rejected(raw_rows, tmp_path):
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, raw_rows + [raw_rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(path, tmp_path / "prepared")
