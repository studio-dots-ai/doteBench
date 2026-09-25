#!/usr/bin/env python3
"""Build the redistributable audio layer used by doteBench materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ALLOWED = {"allow_with_attribution", "allow_model_layer_with_attribution"}
OUTPUT_FILES = {
    "audio",
    "audio-provenance.jsonl",
    "bundle-manifest.json",
    "licenses",
    "NOTICE.md",
}
LICENSE_FILES = {"Apache-2.0.txt", "CC-BY-4.0.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_extras(path: Path, expected: set[str]) -> None:
    """Reject stale or unrelated entries before reusing a bundle directory."""
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Bundle path must be a directory: {path}")
    extra = sorted(item.name for item in path.iterdir() if item.name not in expected)
    if extra:
        raise ValueError(f"Unexpected bundle entries in {path}: {extra[:5]}")


def build(root: Path, audio_root: Path, output: Path) -> dict:
    ledger = [
        json.loads(line)
        for line in (root / "release/audio-provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = sorted(
        (record for record in ledger if record["release_decision"] in ALLOWED),
        key=lambda item: item["audio_sha256"],
    )
    if len(selected) != 1027:
        raise ValueError(f"Expected 1,027 bundle assets, found {len(selected)}")
    reject_extras(output, OUTPUT_FILES)
    destination = output / "audio"
    expected_audio = {f"{record['audio_sha256']}.wav" for record in selected}
    reject_extras(destination, expected_audio)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for record in selected:
        digest = record["audio_sha256"]
        source = audio_root / f"{digest}.wav"
        target = destination / source.name
        if sha256(source) != digest:
            raise ValueError(f"Source audio checksum mismatch: {digest}")
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256(target) != digest:
            raise ValueError(f"Bundle audio checksum mismatch: {digest}")
        files[digest] = {"sha256": digest, "size_bytes": target.stat().st_size}
    reject_extras(destination, expected_audio)
    if {item.name for item in destination.iterdir()} != expected_audio:
        raise ValueError("Bundle audio directory is incomplete")
    canonical = [{"asset_id": key, **files[key]} for key in sorted(files)]
    aggregate = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "benchmark_version": json.loads((root / "release/shards.json").read_text())[
            "version"
        ],
        "asset_count": len(files),
        "audio_digest": aggregate,
        "files": files,
    }
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "audio-provenance.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in selected),
        encoding="utf-8",
    )
    licenses = output / "licenses"
    reject_extras(licenses, LICENSE_FILES)
    licenses.mkdir(exist_ok=True)
    for name in ("Apache-2.0.txt", "CC-BY-4.0.txt"):
        shutil.copyfile(root / "release/licenses" / name, licenses / name)
    shutil.copyfile(root / "release/AUDIO-NOTICE.md", output / "NOTICE.md")
    reject_extras(output, OUTPUT_FILES)
    if {item.name for item in output.iterdir()} != OUTPUT_FILES:
        raise ValueError("Bundle directory is incomplete")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(
        args.root.resolve(), args.audio_root.resolve(), args.output.resolve()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
