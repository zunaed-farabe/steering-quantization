from copy import deepcopy

import pytest

import steerquant.dose as dose
from steerquant.config import DEFAULTS, load_config
from steerquant.experiments import validation_conditions
from steerquant.metrics import metrics


def fixture_rows(stereotyped=2, invalid=0, unknown_errors=0, groups=12):
    """Twelve scenarios, ten ambiguous and ten disambiguated questions each."""
    rows = []
    for g in range(groups):
        for n in range(20):
            ambig = n < 10
            gold = 1 if ambig else 2
            prediction = gold
            if ambig and n < stereotyped:
                prediction = 0  # the stereotyped target
            elif ambig and n < stereotyped + invalid:
                prediction = None
            elif not ambig and n - 10 < unknown_errors:
                prediction = 1  # an unnecessary unknown
            rows.append({"id": f"{g}:{n}", "group": f"g{g}", "context_condition": "ambig" if ambig else "disambig",
                         "gold": gold, "unknown": 1, "target": 0, "prediction": prediction, "truncated": False,
                         "answer_info": {"ans0": ["man", "M"], "ans1": ["Unknown", "unknown"], "ans2": ["woman", "F"]}})
    return rows


def run(condition, rows):
    meta = {"identity": {"model": {"precision": "bf16"}, "condition": condition}}
    return (None, meta, rows, metrics(rows))


def result(monkeypatch, reference, random, baseline=None):
    runs = [run({"source": "none", "layer": None, "strength": 0.0}, baseline or fixture_rows()),
            run({"source": "reference", "layer": 13, "strength": 0.3}, reference),
            run({"source": "random", "layer": 13, "strength": 0.3}, random)]
    monkeypatch.setattr(dose, "collect_runs", lambda *args: runs)
    config = deepcopy(DEFAULTS)
    config["precisions"] = ["bf16"]
    return dose.verdict(config, bootstrap=200)


def test_vector_that_beats_random_within_limits_passes(monkeypatch):
    outcome = result(monkeypatch, fixture_rows(stereotyped=0), fixture_rows())
    assert outcome["verdict"] == "PASS"
    assert outcome["settings"][0]["reduction"] == pytest.approx(0.1)
    assert outcome["settings"][0]["changed_answers"] == 24
    assert outcome["settings"][0]["random_changed_answers"] == 0


def test_vector_no_better_than_random_fails(monkeypatch):
    outcome = result(monkeypatch, fixture_rows(stereotyped=0), fixture_rows(stereotyped=0))
    assert outcome["verdict"] == "FAIL"
    assert outcome["settings"][0]["lowers_stereotyping"]
    assert not outcome["settings"][0]["beats_random"]


def test_small_reduction_fails_even_when_it_beats_random(monkeypatch):
    baseline = fixture_rows()
    for row in baseline[:1]:
        row["prediction"] = 0
    almost_same = deepcopy(baseline)
    almost_same[0]["prediction"] = almost_same[0]["gold"]
    outcome = result(monkeypatch, almost_same, deepcopy(baseline), baseline)
    assert outcome["settings"][0]["reduction"] < dose.RULE["min_reduction"]
    assert outcome["verdict"] == "FAIL"


def test_reducing_bias_by_abstaining_too_much_fails(monkeypatch):
    outcome = result(monkeypatch, fixture_rows(stereotyped=0, unknown_errors=5), fixture_rows())
    assert outcome["verdict"] == "FAIL"
    assert {"disambig_accuracy", "unnecessary_unknown"} <= set(outcome["settings"][0]["broken_limits"])


def test_unscoreable_baseline_is_not_a_result(monkeypatch):
    broken = fixture_rows(invalid=3)
    outcome = result(monkeypatch, broken, broken, broken)
    assert outcome["verdict"] == "NOT_SCOREABLE"


def test_report_shows_the_verdict_and_every_setting(monkeypatch):
    text = dose.markdown(result(monkeypatch, fixture_rows(stereotyped=0), fixture_rows()))
    assert "**Verdict: PASS**" in text
    assert "| 13 | +0.3 |" in text


@pytest.mark.parametrize("name,layers", [("qwen", [8, 13, 20]), ("gemma", [12, 20, 28])])
def test_dose_configs_are_bf16_validation_sweeps_with_their_own_roots(name, layers):
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / f"configs/dose_{name}.yaml")
    assert config["precisions"] == ["bf16"]
    assert config["steering"]["layers"] == layers
    assert config["steering"]["sources"] == ["reference", "random"]
    assert config["output"]["root"] == f"artifacts/dose-{name}"
    assert len(validation_conditions(config, "bf16")) == 50
