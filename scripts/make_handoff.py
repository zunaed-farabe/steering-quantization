"""Build a source + audited-data handoff without environments, credentials, or weights."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TOP_FILES = {"README.md", "START_HERE.md", "PROJECT_IDEA.md", "FILE_GUIDE.md", "GPU_RUNBOOK.md",
             "pyproject.toml", "LICENSE", ".gitignore", ".gitattributes", "run_experiment.sh"}
CAPABILITY_FILES = {"qa.jsonl", "manifest.json", "README.md", "LICENSE-MMLU.txt"}
SOURCE_FOLDERS = {"src", "tests", "configs", "docs", "examples", "scripts", ".github"}
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".md", ".txt"}
RAW_FILES = {"Gender_identity.jsonl", "Age.jsonl", "Race_ethnicity.jsonl", "additional_metadata.csv", "source.json", "LICENSE"}
PREPARED_FILES = {"train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"}
REQUIRED = {"START_HERE.md", "PROJECT_IDEA.md", "FILE_GUIDE.md", "GPU_RUNBOOK.md", "pyproject.toml",
            "src/steerquant/cli.py", "data/raw/bbq/Gender_identity.jsonl", "data/raw/bbq/additional_metadata.csv",
            "data/raw/bbq/source.json", "data/raw/bbq/LICENSE"} | {f"data/prepared/gender/{name}" for name in PREPARED_FILES}


def include(relative):
    parts = relative.parts
    if any(part in {"__pycache__", ".pytest_cache", ".venv", ".git"} or part.endswith(".egg-info") for part in parts):
        return False
    if len(parts) == 1:
        return relative.name in TOP_FILES
    if parts[0] in SOURCE_FOLDERS:
        return relative.suffix in SUFFIXES and relative.name != ".env"
    if parts[:3] == ("data", "raw", "bbq") and len(parts) == 4:
        return relative.name in RAW_FILES
    if parts[:3] == ("data", "prepared", "gender") and len(parts) == 4:
        return relative.name in PREPARED_FILES
    if parts[:2] == ("data", "capability") and len(parts) == 3:
        return relative.name in CAPABILITY_FILES
    return False


def build_bundle(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and not path.is_symlink() and include(relative):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Bundle input escapes project root")
            files[relative.as_posix()] = path
    missing = REQUIRED - files.keys()
    if missing:
        raise ValueError(f"Missing required handoff inputs: {sorted(missing)}")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    inventory = {}
    for name, path in files.items():
        inventory[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"format": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit, "git_dirty": dirty,
                "purpose": "GPU handoff: code, study idea, instructions, and audited official BBQ data",
                "excluded": ["Python environments", "credentials", "model weights/cache", "prior experiment outputs", "Git history"],
                "files": inventory}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files.items():
            archive.write(path, "steering-quantization/" + name)
        archive.writestr("steering-quantization/BUNDLE_INFO.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temp.replace(destination)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{checksum}  {destination.name}\n")
    return destination, len(files), checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="../steering-quantization-handoff.zip")
    args = parser.parse_args()
    try:
        path, count, checksum = build_bundle(Path(__file__).resolve().parents[1], args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"archive": str(path), "files": count + 1, "bytes": path.stat().st_size, "sha256": checksum}, indent=2))


if __name__ == "__main__":
    main()
