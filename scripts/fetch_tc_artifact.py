#!/usr/bin/env python3
"""Download one archived TC artifact and verify its declared SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import yaml


def _entries(manifest: dict):
    real = dict(manifest["real_archive"])
    real["id"] = "real_archive"
    yield real
    if "real_train_archive" in manifest:
        real_train = dict(manifest["real_train_archive"])
        real_train["id"] = "real_train_archive"
        yield real_train
    if "real_validation_archive" in manifest:
        real_validation = dict(manifest["real_validation_archive"])
        real_validation["id"] = "real_validation_archive"
        yield real_validation
    yield from manifest["generated_sample_banks"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_id")
    parser.add_argument("--manifest", type=Path, default=Path("case_studies/tropical_cyclone_bt/artifacts.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    matches = [entry for entry in _entries(manifest) if entry["id"] == args.artifact_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate artifact id {args.artifact_id!r}.")
    entry = matches[0]
    url = entry.get("archive_url")
    expected = entry.get("archive_sha256") or entry.get("sha256")
    if not url or url == "PENDING" or not expected or expected == "PENDING":
        raise ValueError("Artifact URL and SHA-256 must be frozen before download.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, tempfile.NamedTemporaryFile(
        dir=args.output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            temporary.write(chunk)
    observed = digest.hexdigest()
    if observed != expected:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch: expected {expected}, observed {observed}.")
    shutil.move(str(temporary_path), args.output)
    print(f"verified {args.artifact_id}: {observed} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
