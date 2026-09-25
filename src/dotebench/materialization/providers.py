"""HTTP transport for repository-owned materialization model services."""

from __future__ import annotations

import base64
from pathlib import Path

import requests

from .manifest import AssetSpec


def synthesize(
    url: str,
    spec: AssetSpec,
    *,
    reference_audio: Path | None = None,
    timeout: float = 1800,
) -> bytes:
    payload = {
        "text": spec.source_text,
        "language": spec.language,
        "seed": spec.recipe["seed"],
        "model_id": spec.recipe["model_id"],
        "parameters": spec.recipe["parameters"],
    }
    if reference_audio is not None:
        payload["reference_audio"] = base64.b64encode(
            reference_audio.read_bytes()
        ).decode("ascii")
    response = requests.post(
        url.rstrip("/") + "/synthesize", json=payload, timeout=timeout
    )
    response.raise_for_status()
    if response.headers.get("content-type", "").split(";", 1)[0] != "audio/wav":
        raise RuntimeError("Materialization provider did not return audio/wav")
    if not response.content:
        raise RuntimeError("Materialization provider returned empty audio")
    return response.content
