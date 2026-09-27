"""Content identities for materialized audio and receipts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_digest(records: Mapping[str, Mapping[str, object]]) -> str:
    """Hash a canonical asset-id to content fingerprint inventory."""
    canonical = [
        {
            "asset_id": asset_id,
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for asset_id, record in sorted(records.items())
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
