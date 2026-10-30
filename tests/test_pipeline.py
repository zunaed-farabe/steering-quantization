import pytest

from steerquant.analysis import paired_interval
from steerquant.config import load_config
from steerquant.experiments import evaluate, read_predictions, select, validation_conditions
from steerquant.io import read_json
from steerquant.smoke import run_smoke


def test_offline_end_to_end_and_artifact_integrity(tmp_path):
    dest = tmp_path / "smoke"
    run_smoke(dest)
    config = load_config(dest / "config.json")
    selection = read_json(dest / "selection.json")
    assert selection["choices"]
    assert any("unchanged_reference" in c["operating_points"] for c in selection["choices"])
    assert (dest / "validation-report" / "response.svg").exists()
    assert (dest / "test-report" / "metrics.csv").exists()
    assert (dest / "capability-report" / "metrics.csv").exists()
    evaluate(config, "fp32", resume=True)
    with pytest.raises(ValueError, match="Run exists"):
        evaluate(config, "fp32")
    with pytest.raises(ValueError, match="frozen"):
        evaluate(config, "fp32", split="test")
    run = next((dest / "artifacts" / "runs" / "validation").iterdir())
    with (run / "predictions.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="modified"):
        read_predictions(run)


def test_missing_validation_sweep_cannot_select(tmp_path):
    from copy import deepcopy
    from steerquant.config import DEFAULTS
    config = deepcopy(DEFAULTS)
    config["output"]["root"] = str(tmp_path)
    with pytest.raises(ValueError, match="No complete"):
        select(config, tmp_path / "selection.json")


def test_reference_native_not_duplicated():
    from steerquant.config import DEFAULTS
    conditions = validation_conditions(DEFAULTS, "bf16")
    assert not any(c["source"] == "native" for c in conditions)
    assert len([c for c in conditions if c["source"] == "none"]) == 1


def test_paired_bootstrap_uses_group_units():
    before = [{"id": str(i), "group": "one" if i < 3 else "two", "gold": 1,
               "prediction": 0, "context_condition": "ambig"} for i in range(4)]
    after = [{**r, "prediction": 1} for r in before]
    result = paired_interval(before, after, bootstrap=100)
    assert result["groups"] == 2
    assert result["delta"] == 1
    assert result["ci95"] == [1, 1]
