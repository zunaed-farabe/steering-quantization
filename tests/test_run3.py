"""Run 3: stereotype contrast, lenient-but-exact answer parsing, bias scan, configs and data."""

import importlib.util
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import steerquant.dose as dose
import steerquant.scan as scan
from steerquant.config import DEFAULTS, config_id, contrast, load_config
from steerquant.data import contrastive_pairs, load_split, normalize, pairs_for, stereotype_pairs
from steerquant.experiments import validation_conditions
from steerquant.metrics import answer_format, metrics, parse_answer

ROOT = Path(__file__).resolve().parents[1]


def row(gold=0, target=0, unknown=2, condition="disambig", id_="x:1"):
    return {"id": id_, "group": "x:1", "category": "Race_ethnicity", "context": "Two people talked.",
            "question": "Who was rude?", "answers": ["The first", "The second", "Can't answer"],
            "gold": gold, "target": target, "unknown": unknown, "context_condition": condition,
            "question_polarity": "neg", "answer_info": {}, "metadata": {}}


# --- Stereotype contrast ---------------------------------------------------------------

@pytest.mark.parametrize("target,unknown", [(0, 2), (1, 2), (0, 1), (2, 0)])
def test_stereotype_pairs_contrast_the_other_named_answer_with_the_target(target, unknown):
    original = row(gold=unknown, target=target, unknown=unknown, condition="ambig")
    other = ({0, 1, 2} - {target, unknown}).pop()
    pairs = list(stereotype_pairs([original]))
    assert len(pairs) == 3
    for pair in pairs:
        order = pair["order"]
        assert pair["positive"] == "ABC"[order.index(other)]
        assert pair["negative"] == "ABC"[order.index(target)]
        assert "ABC"[order.index(unknown)] not in (pair["positive"], pair["negative"])
    # Each letter is positive once and negative once, so no letter prior is learned.
    assert Counter(p["positive"] for p in pairs) == {"A": 1, "B": 1, "C": 1}
    assert Counter(p["negative"] for p in pairs) == {"A": 1, "B": 1, "C": 1}


def test_stereotype_pairs_ignore_gold_and_skip_unannotated_rows():
    agrees = list(stereotype_pairs([row(gold=0, target=0)]))
    disagrees = list(stereotype_pairs([row(gold=1, target=0)]))
    assert [(p["positive"], p["negative"]) for p in agrees] == [(p["positive"], p["negative"]) for p in disagrees]
    assert list(stereotype_pairs([row(target=None)])) == []


def test_evidence_contrast_is_unchanged_from_runs_1_and_2(raw_rows):
    rows = [normalize(r) for r in raw_rows[:8]]
    assert pairs_for("evidence", rows) == list(contrastive_pairs(rows))
    with pytest.raises(ValueError, match="Unknown contrast"):
        pairs_for("gold", rows)


def test_training_sample_can_require_a_target(tmp_path, raw_rows):
    from steerquant.data import prepare
    raw = [dict(r) for r in raw_rows]
    for r in raw[::2]:
        r["target_loc"] = None
    from steerquant.io import write_jsonl
    write_jsonl(tmp_path / "raw.jsonl", raw)
    prepare(tmp_path / "raw.jsonl", tmp_path / "prepared")
    rows, _ = load_split(tmp_path / "prepared", "train", 8, 42, require_target=True)
    assert rows and all(r["target"] is not None for r in rows)
    assert {r["context_condition"] for r in rows} == {"ambig", "disambig"}


# --- Config ------------------------------------------------------------------------------

def test_contrast_is_optional_so_existing_config_identities_do_not_change():
    old = load_config(ROOT / "configs/dose_qwen.yaml")
    assert "contrast" not in old["steering"] and contrast(old) == "evidence"
    assert config_id(old) == config_id(load_config(ROOT / "configs/dose_qwen.yaml"))
    new = load_config(ROOT / "configs/run3/dose_qwen_race.yaml")
    assert contrast(new) == "stereotype"


def test_invalid_contrast_and_layerless_steering_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("steering:\n  contrast: gold\n")
    with pytest.raises(ValueError, match="contrast"):
        load_config(bad)
    bad.write_text("steering:\n  layers: []\n  sources: [reference]\n")
    with pytest.raises(ValueError, match="Layers"):
        load_config(bad)


def test_run3_configs_are_generated_and_current():
    spec = importlib.util.spec_from_file_location("make_run3_configs", ROOT / "scripts/make_run3_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, text in module.files().items():
        assert (ROOT / "configs/run3" / name).read_text(encoding="utf-8") == text, name


@pytest.mark.parametrize("model,layers", [("qwen", [8, 13, 20]), ("gemma", [12, 20, 28])])
def test_run3_configs_match_the_plan(model, layers):
    roots = set()
    for axis in ("age", "religion", "nationality", "appearance", "disability", "socioeconomic"):
        config = load_config(ROOT / f"configs/run3/scan_{model}_{axis}.yaml")
        assert config["precisions"] == ["bf16"] and config["steering"]["sources"] == []
        assert [c["source"] for c in validation_conditions(config, "bf16")] == ["none", "prompt"]
        roots.add(config["output"]["root"])
    for axis in ("race", "gender"):
        config = load_config(ROOT / f"configs/run3/dose_{model}_{axis}.yaml")
        assert config["steering"]["layers"] == layers and contrast(config) == "stereotype"
        assert config["steering"]["sources"] == ["reference", "random"]
        assert config["data"]["prepared"] == f"data/prepared/{axis}"
        # Same grid as run 2, so the two contrasts are compared at identical settings.
        assert config["steering"]["strengths"] == load_config(ROOT / f"configs/dose_{model}.yaml")["steering"]["strengths"]
        assert len(validation_conditions(config, "bf16")) == 50
        roots.add(config["output"]["root"])
    assert len(roots) == 8 and all(r.startswith("artifacts/run3/") for r in roots)


# --- Data --------------------------------------------------------------------------------

@pytest.mark.parametrize("axis,groups", [("race", 10), ("gender", 11), ("age", 5), ("religion", 4), ("nationality", 4),
                                         ("appearance", 5), ("disability", 4), ("socioeconomic", 3)])
def test_prepared_splits_verify_and_have_the_documented_scenario_counts(axis, groups):
    rows, manifest = load_split(ROOT / f"data/prepared/{axis}", "validation")
    assert len({r["group"] for r in rows}) == manifest["splits"]["validation"]["groups"] == groups
    assert (groups >= scan.MIN_GROUPS) == (axis in {"race", "gender"})


@pytest.mark.parametrize("axis", ["race", "gender"])
def test_dose_categories_give_stereotype_pairs_for_every_training_row(axis):
    rows, _ = load_split(ROOT / f"data/prepared/{axis}", "train", 128, 42, require_target=True)
    assert len(rows) == 128 and {r["context_condition"] for r in rows} == {"ambig", "disambig"}
    pairs = list(stereotype_pairs(rows))
    assert len(pairs) == 3 * len(rows)


def test_bbq_target_is_the_stereotype_consistent_answer_for_both_polarities():
    rows, _ = load_split(ROOT / "data/prepared/age", "validation")
    for r in rows:
        label = str(r["answer_info"][f"ans{r['target']}"][1]).lower()
        named = label in [str(g).lower() for g in r["metadata"]["stereotyped_groups"]]
        assert named == (r["question_polarity"] == "neg"), r["id"]


# --- Answer parsing ----------------------------------------------------------------------

OPTIONS = ["The grandfather", "The grandson", "Can't answer"]


@pytest.mark.parametrize("text,expected,form", [
    ("B", 1, "letter"), (" C. ", 2, "letter"),
    ("C. Can't answer", 2, "letter_with_option"), ("C. Can’t answer.", 2, "letter_with_option"),
    ("(A) The grandfather", 0, "letter_with_option"), ("a) the grandfather", 0, "letter_with_option"),
    ("A. Can't answer", None, None),            # letter and text disagree
    ("A. The grandfather because he", None, None),  # explanation
    ("A.  Both", None, None), ("Can't answer", None, None), ("A or B", None, None), ("", None, None)])
def test_letter_with_its_own_option_text_is_accepted_and_nothing_else(text, expected, form):
    assert parse_answer(text, 3, OPTIONS) == expected
    assert answer_format(text, 3, OPTIONS) == form


def test_without_options_the_parser_stays_strict():
    assert parse_answer("C. Can't answer") is None


# --- Bias scan and verdict diagnostics -----------------------------------------------------

def scan_rows(groups, stereotyped):
    rows = []
    for g in range(groups):
        for n in range(4):
            ambig = n < 2
            gold = 1 if ambig else (0 if n == 2 else 2)
            prediction = 0 if (ambig and n == 0 and g < stereotyped) else gold
            rows.append({"id": f"{g}:{n}", "group": f"g{g}", "category": "Age", "context_condition": "ambig" if ambig else "disambig",
                         "gold": gold, "unknown": 1, "target": 0, "prediction": prediction, "truncated": False,
                         "answer_format": "letter", "answer_info": {"ans0": ["old", "old"], "ans1": ["?", "unknown"], "ans2": ["young", "nonOld"]}})
    return rows


def test_scan_reports_room_and_marks_small_categories_descriptive(monkeypatch, tmp_path):
    def fake_runs(config, split):
        rows = scan_rows(config["groups"], 2)
        return [(None, {"identity": {"model": {"precision": "bf16"}, "condition": {"source": s, "layer": None, "strength": 0.0}}},
                 rows, metrics(rows)) for s in ("none", "prompt")]
    monkeypatch.setattr(scan, "collect_runs", fake_runs)
    configs = []
    for groups in (4, 10):
        c = deepcopy(DEFAULTS)
        c["precisions"], c["groups"] = ["bf16"], groups
        configs.append(c)
    scan.scan(configs, tmp_path / "scan.json")
    text = (tmp_path / "scan.md").read_text(encoding="utf-8")
    from steerquant.io import read_json
    records = read_json(tmp_path / "scan.json")["records"]
    assert [r["eligible_for_dose_test"] for r in records] == [False, True]
    assert records[0]["unsteered"]["stereotyped_answers"] == 2
    assert records[0]["unsteered"]["pro_minus_anti_accuracy"] == 0
    assert "| Age |" in text


def test_direction_check_counts_negative_strengths_that_raise_stereotyping(monkeypatch):
    from test_dose import fixture_rows, run
    runs = [run({"source": "none", "layer": None, "strength": 0.0}, fixture_rows()),
            run({"source": "reference", "layer": 13, "strength": -0.3}, fixture_rows(stereotyped=6)),
            run({"source": "random", "layer": 13, "strength": -0.3}, fixture_rows())]
    monkeypatch.setattr(dose, "collect_runs", lambda *args: runs)
    config = deepcopy(DEFAULTS)
    config["precisions"] = ["bf16"]
    config["steering"]["contrast"] = "stereotype"
    outcome = dose.verdict(config, bootstrap=200)
    assert outcome["verdict"] == "FAIL"  # the pass rule itself is unchanged
    assert outcome["direction_check"] == {"negative_settings": 1, "raise_more_than_random": 1, "lower_more_than_random": 0}
    assert "Direction check" in dose.markdown(outcome)
