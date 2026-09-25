"""Prepare local model assets from explicit locks and verify file receipts."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

from dotebench.dataset import read_json, relative_path

RECEIPT = ".dotebench-evaluation-model.json"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _files(root):
    return {
        str(p.relative_to(root)): p
        for p in root.rglob("*")
        if p.is_file()
        and not {".git", ".cache", "__pycache__"}.intersection(
            p.relative_to(root).parts
        )
        and p.name not in {RECEIPT, RECEIPT + ".pending"}
    }


def _stamp(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _hashes(path):
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


def _hub_files(lock):
    """Get blob identities at the pinned commit; this downloads metadata only."""
    revision = lock["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Metric model revision must be a full commit SHA")
    url = (
        "https://huggingface.co/api/models/"
        + quote(lock["repository"], safe="/")
        + "/revision/"
        + revision
    )
    response = requests.get(url, params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != revision:
        raise ValueError("Hub model revision mismatch")
    files = {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        relative_path(Path("."), name)
        if name in files:
            raise ValueError("Duplicate Hub file")
        lfs = item.get("lfs")
        files[name] = {
            "size": item["size"],
            "sha256" if lfs else "git_blob": lfs["sha256"] if lfs else item["blobId"],
        }
    if not files:
        raise ValueError("Hub model has no files")
    return files


def _identity(model_id, lock_sha256, files):
    return {
        "model": model_id,
        "lock_sha256": lock_sha256,
        "files_sha256": digest({name: item["sha256"] for name, item in files.items()}),
    }


def validate_identity(model_id, identity, *, lock=None):
    """Check identity structure and, when supplied, the caller-owned model lock."""
    if (
        not isinstance(identity, dict)
        or identity.get("model") != model_id
        or not isinstance(identity.get("lock_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity["lock_sha256"])
        or (lock is not None and identity["lock_sha256"] != digest(lock))
        or not isinstance(identity.get("files_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity["files_sha256"])
    ):
        raise ValueError(f"Metric model identity mismatch: {model_id}")


def prepare_model(model_id, root, *, lock):
    """Validate existing assets and atomically register their immutable identity.

    Hugging Face snapshots must contain the complete pinned repository. Other
    assets verify the locked checkpoint and include local loader files in the
    receipt. No weight files are downloaded or modified.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Model preparation requires a directory")
    paths = _files(root)
    if "repository" in lock:
        expected = _hub_files(lock)
        if set(paths) != set(expected):
            raise ValueError(f"Pinned model file set mismatch: {model_id}")
    else:
        expected = {lock["filename"]: {"sha256": lock["sha256"]}}
        if not set(expected) <= set(paths):
            raise ValueError(f"Missing locked checkpoint: {model_id}")
    files = {}
    for name, path in sorted(paths.items()):
        actual, git_blob = _hashes(path)
        for key, value in expected.get(name, {}).items():
            if (git_blob if key == "git_blob" else actual[key]) != value:
                raise ValueError(
                    f"Model asset checksum or size mismatch: {model_id}/{name}"
                )
        files[name] = actual
    identity = _identity(model_id, digest(lock), files)
    receipt = {"identity": identity, "files": files}
    temp = root / (RECEIPT + ".pending")
    temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temp.replace(root / RECEIPT)
    return identity


def prepared_identity(model_id, model_path, *, lock=None, full_check=False):
    """Read a verified receipt; startup checks stamps, --check rehashes files."""
    path = Path(model_path).resolve()
    root = path if path.is_dir() else path.parent
    if lock is not None and path.is_file() and path.name != lock.get("filename"):
        raise ValueError("Checkpoint path differs from the model lock")
    receipt = read_json(root / RECEIPT)
    files = receipt["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty model receipt")
    validate_identity(model_id, receipt["identity"], lock=lock)
    if receipt["identity"] != _identity(
        model_id, receipt["identity"]["lock_sha256"], files
    ):
        raise ValueError("Model receipt content mismatch")
    paths = _files(root)
    if set(paths) != set(files):
        raise ValueError("Prepared model file set changed")
    for name, item in files.items():
        if full_check:
            actual, _ = _hashes(paths[name])
            if actual["sha256"] != item["sha256"]:
                raise ValueError(f"Prepared model checksum changed: {name}")
        if _stamp(paths[name]) != {key: item[key] for key in ("size", "mtime_ns")}:
            raise ValueError(f"Prepared model asset changed; prepare it again: {name}")
    return receipt["identity"]
