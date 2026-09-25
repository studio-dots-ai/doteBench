"""Prepare and identify model snapshots used by data materialization."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

from dotebench.dataset import read_json, relative_path, resources

RECEIPT = ".dotebench-materialization-model.json"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def model_locks() -> dict[str, dict]:
    return read_json(resources() / "release/materialization-models.json")["models"]


def model_directory(root: Path, model_id: str) -> Path:
    return relative_path(Path(root), model_locks()[model_id]["directory"])


def locked_identity(model_id: str) -> dict[str, str]:
    """Return the public identity fixed by the materialization model lock."""
    lock = model_locks()[model_id]
    return {
        "model": model_id,
        "repository": lock["repository"],
        "revision": lock["revision"],
        "lock_sha256": _digest(lock),
    }


def _files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file()
        and not {".git", ".cache", "__pycache__"}.intersection(
            path.relative_to(root).parts
        )
        and path.name not in {RECEIPT, RECEIPT + ".pending"}
    }


def _stamp(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _hashes(path: Path) -> tuple[dict[str, int | str], str]:
    stamp = _stamp(path)
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {stamp['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    if stamp != _stamp(path):
        raise ValueError(f"Model asset changed while hashing: {path}")
    return {**stamp, "sha256": sha256.hexdigest()}, git_blob.hexdigest()


def _hub_files(lock: dict) -> dict[str, dict]:
    revision = lock["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Materialization model revision must be a full commit SHA")
    response = requests.get(
        "https://huggingface.co/api/models/"
        + quote(lock["repository"], safe="/")
        + "/revision/"
        + revision,
        params={"blobs": "true"},
        timeout=60,
    )
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != revision:
        raise ValueError("Hub model revision mismatch")
    result = {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        relative_path(Path("."), name)
        if name in result:
            raise ValueError("Duplicate Hub model file")
        lfs = item.get("lfs")
        result[name] = {
            "size": item["size"],
            "sha256" if lfs else "git_blob": lfs["sha256"]
            if lfs
            else item["blobId"],
        }
    if not result:
        raise ValueError("Hub model has no files")
    return result


def _identity(model_id: str, lock: dict, files: dict) -> dict[str, str]:
    return {
        **locked_identity(model_id),
        "files_sha256": _digest(
            {name: item["sha256"] for name, item in files.items()}
        ),
    }


def prepare_model(model_id: str, root: Path, *, accepted: set[str]) -> dict[str, str]:
    """Verify a complete pinned snapshot and register an immutable local receipt."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Materialization model directory is missing: {root}")
    lock = model_locks()[model_id]
    acceptance = lock.get("license_acceptance")
    if acceptance and acceptance not in accepted:
        raise ValueError(
            f"Model {model_id} requires --accept-license {acceptance}"
        )
    paths = _files(root)
    expected = _hub_files(lock)
    if set(paths) != set(expected):
        raise ValueError(f"Pinned model file set mismatch: {model_id}")
    files = {}
    for name, path in sorted(paths.items()):
        actual, git_blob = _hashes(path)
        for key, value in expected[name].items():
            observed = git_blob if key == "git_blob" else actual[key]
            if observed != value:
                raise ValueError(f"Model asset mismatch: {model_id}/{name}")
        files[name] = actual
    identity = _identity(model_id, lock, files)
    payload = {"identity": identity, "files": files}
    temporary = root / (RECEIPT + ".pending")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / RECEIPT)
    return identity


def prepared_identity(
    model_id: str, root: Path, *, full_check: bool = True
) -> dict[str, str]:
    """Return the pinned identity after rehashing every registered model file."""
    root = Path(root).resolve()
    receipt = read_json(root / RECEIPT)
    lock = model_locks()[model_id]
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty materialization model receipt")
    expected_identity = _identity(model_id, lock, files)
    if receipt.get("identity") != expected_identity:
        raise ValueError(f"Materialization model identity mismatch: {model_id}")
    paths = _files(root)
    if set(paths) != set(files):
        raise ValueError(f"Prepared model file set changed: {model_id}")
    for name, item in files.items():
        if _stamp(paths[name]) != {key: item[key] for key in ("size", "mtime_ns")}:
            raise ValueError(f"Prepared model asset changed: {model_id}/{name}")
        if full_check and _hashes(paths[name])[0]["sha256"] != item["sha256"]:
            raise ValueError(f"Prepared model checksum changed: {model_id}/{name}")
    return expected_identity
