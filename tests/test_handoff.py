import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def helper(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handoff_contains_required_files_but_no_environment_or_weights(tmp_path):
    module = helper("make_handoff")
    root = tmp_path / "project"
    for name in module.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    for name in (".env", ".venv/private.txt", ".git/config", "artifacts/model.pt", "src/steerquant/__pycache__/cached.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must not ship")
    destination, count, checksum = module.build_bundle(root, tmp_path / "handoff.zip")
    assert checksum == hashlib.sha256(destination.read_bytes()).hexdigest()
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("steering-quantization/BUNDLE_INFO.json"))
        assert len(manifest["files"]) == count
        for name, entry in manifest["files"].items():
            assert hashlib.sha256(archive.read("steering-quantization/" + name)).hexdigest() == entry["sha256"]
        assert not any(".venv" in name or ".env" in name or "artifacts/" in name for name in archive.namelist())


def test_result_packager_preserves_partial_runs_and_excludes_smoke_weights(tmp_path, monkeypatch):
    module = helper("package_results")
    root = tmp_path / "project"
    monkeypatch.setattr(module, "__file__", str(root / "scripts/package_results.py"))
    for name in ("artifacts/qwen-gender-seed42/runs/validation/partial/run.json", "logs/failure.log", "configs/main.locked.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    for name in (".env", "artifacts/offline-smoke/tiny-model/model.safetensors"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded")
    path, _ = module.package(tmp_path / "results.zip")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert any(name.endswith("failure.log") for name in names)
        assert any("partial/run.json" in name for name in names)
        assert not any(".env" in name or "offline-smoke" in name for name in names)
    with pytest.raises(ValueError, match="overwrite"):
        module.package(path)


def test_gpu_diagnostic_records_a_report_without_exposing_environment(monkeypatch):
    module = helper("check_gpu")
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token-test-sentinel")
    result = module.diagnostic()
    assert "ready_for_gpu_check" in result
    assert "not-a-real-token-test-sentinel" not in json.dumps(result)
