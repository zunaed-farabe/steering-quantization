"""Create a portable result archive, including partial runs for diagnosis."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package(destination, artifact_roots=None):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}")
    artifact_roots = artifact_roots or [root / "artifacts/gpu-check", root / "artifacts/qwen-gender-seed42"]
    files = {}
    for name in artifact_roots:
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact roots must be inside this project")
        if path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and not f.is_symlink() and f.suffix in {".json", ".jsonl", ".pt", ".csv", ".svg", ".md", ".log", ".txt"}:
                    files[str(f.relative_to(root))] = f
    for folder in (root / "logs", root / "reports", root / "configs"):
        if folder.exists():
            for f in folder.rglob("*"):
                if f.is_file() and not f.is_symlink() and f.suffix in {".json", ".jsonl", ".yaml", ".yml", ".csv", ".svg", ".md", ".log", ".txt"}:
                    files[str(f.relative_to(root))] = f
    for name in ("BUNDLE_INFO.json", "data/raw/bbq/source.json", "data/raw/bbq/LICENSE", "data/prepared/gender/manifest.json", "docs/data-audit.json"):
        path = root / name
        if path.is_file():
            files[name] = path
    if not any(name.startswith(("artifacts/", "logs/", "reports/")) for name in files):
        raise ValueError("No result artifacts or diagnostic logs found")
    inventory = {name: {"bytes": path.stat().st_size, "sha256": hash_file(path)} for name, path in sorted(files.items())}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, "steering-results/" + name)
        archive.writestr("steering-results/RESULTS_MANIFEST.json", json.dumps({"files": inventory}, indent=2) + "\n")
    temp.replace(destination)
    return destination, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="steering-results.zip")
    parser.add_argument("--artifact-root", action="append", help="Repeat for nondefault artifact roots")
    args = parser.parse_args()
    try:
        path, count = package(args.output, args.artifact_root)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Created {path} with {count} result/provenance files")


if __name__ == "__main__":
    main()
