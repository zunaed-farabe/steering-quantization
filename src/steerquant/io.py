"""Artifact IO and fingerprints. No model or dataset code runs on import."""

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Explicit UTF-8 and "\n" so hashed files are byte-identical on every OS.
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def environment():
    packages = {}
    for package in ("torch", "transformers", "numpy", "accelerate", "bitsandbytes", "steerquant"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    # Include source content even before the first commit or with a dirty checkout.
    source = {str(p.relative_to(Path(__file__).parent)): file_hash(p)
              for p in sorted(Path(__file__).parent.rglob("*.py"))}
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages,
            "git_commit": commit, "git_dirty": dirty, "source_sha256": digest(source)}


def ensure_new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
