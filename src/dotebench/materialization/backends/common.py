"""Shared HTTP contract for materialization synthesis backends."""

from __future__ import annotations

import base64
import hashlib
import io
import random
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from dotebench.services.utils.http import service_app


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str
    seed: int = Field(ge=0)
    model_id: str
    parameters: dict
    reference_audio: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def decoded_audio(value: str | None) -> bytes | None:
    if value is None:
        return None
    return base64.b64decode(value, validate=True)


def wav_bytes(samples, sample_rate: int) -> bytes:
    output = io.BytesIO()
    sf.write(output, np.asarray(samples).reshape(-1), sample_rate, format="WAV")
    return output.getvalue()


def source_identity(upstream: Path, *, source: str, commit: str) -> dict[str, str]:
    """Verify that a service imports the registered official upstream checkout."""
    upstream = Path(upstream).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ValueError(f"Official source commit mismatch: {upstream}")
    configured = subprocess.check_output(
        ["git", "-C", str(upstream), "remote", "get-url", "origin"], text=True
    ).strip()
    normalized = configured.removesuffix(".git").rstrip("/").lower()
    expected = source.removesuffix(".git").rstrip("/").lower()
    if normalized != expected:
        raise ValueError(f"Official source remote mismatch: {upstream}")
    status = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if status.strip():
        raise ValueError(f"Official source checkout has tracked modifications: {upstream}")
    return {"repository": source, "commit": actual}


def implementation_identity(path: Path, module: str) -> dict[str, str]:
    return {
        "module": module,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def create_app(
    *,
    provider: str,
    identity: dict,
    synthesize: Callable[[SynthesisRequest], bytes],
) -> FastAPI:
    app = service_app(
        title=f"doteBench {provider} materialization",
        health=lambda: {"status": "ready", "provider": provider, "identity": identity},
    )
    lock = threading.Lock()

    @app.post("/synthesize")
    def invoke(request: SynthesisRequest):
        try:
            with lock:
                result = synthesize(request)
            if not result:
                raise ValueError("Synthesis returned empty audio")
            return Response(result, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return app
