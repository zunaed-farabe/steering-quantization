"""Run 3 on the tiny offline model, and the runner's commit-and-push step on a local remote."""

import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import steerquant.dose as dose
from steerquant.experiments import evaluate
from steerquant.io import read_json, read_jsonl
from steerquant.scan import scan
from steerquant.data import prepare
from steerquant.io import write_jsonl
from steerquant.smoke import fixture_rows, setup_smoke
from steerquant.steering import extract, load_vectors

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke_config(tmp_path):
    torch.set_num_threads(1)
    return setup_smoke(tmp_path / "smoke")


def polarity_worded(tmp_path):
    """The smoke questions, with BBQ-like different wording for each question polarity."""
    rows = fixture_rows()
    for row in rows:
        if row["question_polarity"] == "nonneg":
            row["question"] = "Who did not move the object at all?"
    write_jsonl(tmp_path / "polar.jsonl", rows)
    prepare(tmp_path / "polar.jsonl", tmp_path / "polar-prepared", seed=42)
    return str(tmp_path / "polar-prepared")


def test_stereotype_contrast_cancels_when_both_polarities_are_worded_identically(smoke_config, tmp_path):
    # The target flips with question polarity, so with identical wording the paired rows
    # contrast the same two activations in opposite directions and sum to exactly zero.
    # Only the polarity-dependent part survives: whether the answer fits the stereotype.
    stereotype = deepcopy(smoke_config)
    stereotype["steering"]["contrast"] = "stereotype"
    stereotype["output"]["root"] = str(tmp_path / "identical")
    with pytest.raises(ValueError, match="Degenerate"):
        extract(stereotype, "fp32")


def test_stereotype_vector_differs_from_evidence_and_records_its_geometry(smoke_config, tmp_path):
    prepared = polarity_worded(tmp_path)
    evidence = deepcopy(smoke_config)
    evidence["data"]["prepared"] = prepared
    evidence["output"]["root"] = str(tmp_path / "evidence")
    stereotype = deepcopy(evidence)
    stereotype["steering"]["contrast"] = "stereotype"
    stereotype["output"]["root"] = str(tmp_path / "stereotype")
    old, new = load_vectors(extract(evidence, "fp32")), load_vectors(extract(stereotype, "fp32"))
    assert old["metadata"]["contrast"] == "evidence" and old["metadata"]["diagnostics"] == {}
    assert new["metadata"]["contrast"] == "stereotype"
    cosine = new["metadata"]["diagnostics"]["cosine_with_evidence_direction"]["0"]
    assert -1 <= cosine <= 1
    assert cosine == pytest.approx(float(new["vectors"]["0"] @ old["vectors"]["0"]), abs=1e-5)
    assert float(new["vectors"]["0"] @ old["vectors"]["0"]) < 0.999

    evaluate(stereotype, "fp32")
    rows = read_jsonl(next((tmp_path / "stereotype" / "runs" / "validation").glob("*/predictions.jsonl")))
    assert all("answer_format" in r for r in rows)
    result = dose.verdict(stereotype, bootstrap=50)
    assert result["contrast"] == "stereotype"
    assert result["vector_diagnostics"]["cosine_with_evidence_direction"]["0"] == pytest.approx(cosine)
    assert result["direction_check"]["negative_settings"] == 1


def test_baseline_only_scan_needs_no_vectors(smoke_config, tmp_path):
    config = deepcopy(smoke_config)
    config["steering"].update(sources=[], layers=[], strengths=[0.0])
    config["output"]["root"] = str(tmp_path / "scan-root")
    runs = evaluate(config, "fp32")
    assert len(runs) == 2 and not (tmp_path / "scan-root" / "vectors").exists()
    for run in runs:
        assert read_json(Path(run) / "run.json")["identity"]["reference_vector_sha256"] is None
    scan([config], tmp_path / "scan.json")
    assert (tmp_path / "scan.md").read_text(encoding="utf-8").count("| Synthetic |") == 1


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.mark.skipif(not all(shutil.which(t) for t in ("bash", "flock", "git")), reason="needs Linux bash, flock and git")
def test_finish_pushes_only_the_archive_and_a_failed_push_never_fails_the_campaign(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    (work / "scripts").mkdir(parents=True)
    git(work, "init", "-q")
    git(work, "config", "user.email", "runner@example.invalid")
    git(work, "config", "user.name", "Runner")
    shutil.copy(ROOT / "scripts/campaign_lib.sh", work / "scripts/campaign_lib.sh")
    (work / "scripts/package_results.py").write_text(
        "import sys\nopen(sys.argv[sys.argv.index('--output') + 1], 'wb').write(b'results')\n")
    (work / "notes.txt").write_text("tracked\n")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "start")
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-q", "-u", "origin", "HEAD")
    (work / "notes.txt").write_text("edited, must not be committed\n")
    (work / "stray.txt").write_text("untracked, must not be committed\n")

    def finish(name):
        script = f'set -Eeuo pipefail; RESUME=resume; source scripts/campaign_lib.sh; finish {name} "Done here" --only {name}'
        done = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True,
                              env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PUSH_RETRY_SECONDS": "0"})
        return done.returncode, (work / "logs/campaign-status.txt").read_text()

    code, status = finish("demo")
    assert code == 0 and "pushed to GitHub" in status
    zips = sorted(p.name for p in work.glob("demo-results-*.zip"))
    assert len(zips) == 1
    pushed = git(work, "--git-dir", str(remote), "show", "--name-only", "--format=%s", "HEAD").split()
    assert pushed[:2] == ["Results:", zips[0]] and pushed[-1] == zips[0] and "notes.txt" not in pushed
    assert "edited" in (work / "notes.txt").read_text()

    code, status = finish("demo")  # resuming a finished campaign reuses its archive
    assert code == 0 and sorted(p.name for p in work.glob("demo-results-*.zip")) == zips

    git(work, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    code, status = finish("other")
    assert code == 0 and "NOT pushed" in status


SSH_TOOLS = ("bash", "flock", "git", "ssh-agent", "ssh-add", "ssh-keygen", "timeout")


@pytest.mark.skipif(not all(shutil.which(t) for t in SSH_TOOLS), reason="needs Linux bash, flock, git and OpenSSH")
@pytest.mark.parametrize("line,unlocks", [("github_passkey='pa ss=wo#rd'", True), ("github_passkey=wrong", False)])
def test_ssh_key_is_unlocked_from_the_env_file_without_ever_prompting(tmp_path, line, unlocks):
    # A throwaway key in a temporary folder; the user's own keys and agent are never touched.
    key = tmp_path / "key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "pa ss=wo#rd", "-C", "throwaway", "-f", str(key)], check=True)
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    git(work, "init", "-q")
    git(work, "remote", "add", "origin", "git@example.invalid:someone/repo.git")
    shutil.copy(ROOT / "scripts/campaign_lib.sh", work / "scripts/campaign_lib.sh")
    (work / ".env").write_text(line + "\n")
    script = ('set -Eeuo pipefail; RESUME=x; source scripts/campaign_lib.sh; '
              'if unlock_ssh_key; then ssh-add -l; echo UNLOCKED; else echo LOCKED; fi; '
              'pid=$OWN_AGENT; stop_own_agent; sleep 0.5; if kill -0 "$pid" 2>/dev/null; then echo AGENT_LEFT; fi')
    done = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True, timeout=120,
                          stdin=subprocess.DEVNULL,
                          env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GITHUB_SSH_KEY": str(key)})
    assert done.returncode == 0, done.stderr
    assert ("UNLOCKED" in done.stdout) == unlocks
    assert ("throwaway" in done.stdout) == unlocks
    assert "AGENT_LEFT" not in done.stdout
    assert "pa ss" not in done.stdout + done.stderr  # the passphrase is never printed
