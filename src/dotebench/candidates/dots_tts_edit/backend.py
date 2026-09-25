"""dots.tts.edit SDK hosted in its own model environment."""

import argparse
import base64
import hashlib
import io
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib.metadata import packages_distributions, version
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from dotebench import RELEASE
from dotebench.domain import Audio, InfrastructureError
from dotebench.models.requests import NativeRequest


class Runtime:
    def __init__(
        self, *, model_path, scratch_dir, runtime_options=None, runtime_factory=None
    ):
        self.model_path = Path(model_path)
        self.scratch_dir = Path(scratch_dir)
        self.runtime_options = dict(runtime_options or {})
        self.runtime_factory = runtime_factory
        self.sdk_version = (
            {
                name: version(name)
                for name in packages_distributions().get("dots_tts", [])
            }
            if runtime_factory is None
            else "injected"
        )
        self.runtime = None
        self.last_call_metadata = {}

    def prepare(self) -> None:
        if not self.model_path.is_dir():
            raise ValueError("model_path must identify a prepared local checkpoint")
        if self.runtime_factory is None:
            from dots_tts.edit_runtime import DotsTtsEditRuntime

            self.runtime_factory = DotsTtsEditRuntime.from_pretrained
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = self.runtime_factory(
            str(self.model_path), **self.runtime_options
        )

    def invoke(self, compiled, audio: Audio) -> Audio:
        if self.runtime is None:
            raise InfrastructureError("Candidate has not been prepared")
        if compiled.transport != "dots_sdk":
            raise InfrastructureError("dots.tts.edit received an unsupported request")
        with TemporaryDirectory(dir=self.scratch_dir, prefix="source-") as directory:
            source = Path(directory) / "source.wav"
            source.write_bytes(audio.data)
            payload = compiled.payload(audio, source)
            self.last_call_metadata = {
                "native_request": asdict(compiled),
                "scope": compiled.scope,
            }
            use_xvector = dict(compiled.fields).get("use_xvector", "auto")
            self.last_call_metadata["resolved_use_xvector"] = (
                compiled.scope != "emotion" if use_xvector == "auto" else use_xvector
            )
            result = self.runtime.generate_edit(**payload)
            self.last_call_metadata["response_metadata"] = {
                key: result[key]
                for key in (
                    "fid",
                    "request_id",
                    "sample_rate",
                    "duration_seconds",
                    "time_used",
                    "rtf",
                    "source_text",
                    "target_text",
                )
                if key in result
            }
        samples = result["audio"]
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().float().numpy().squeeze()
        buffer = io.BytesIO()
        sf.write(buffer, samples, result["sample_rate"], format="WAV", subtype="PCM_16")
        return Audio(buffer.getvalue())

    def close(self) -> None:
        self.runtime = None


def create_app(runtime):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, ConfigDict
    from typing import Any

    class NativeCall(BaseModel):
        model_config = ConfigDict(extra="forbid")
        endpoint: str
        fields: list[tuple[str, Any]]
        audio_field: str
        audio_encoding: str
        response_encoding: str
        selected_operation_index: int | None
        scope: str
        transport: str
        case_id: str | None

    class Request(BaseModel):
        model_config = ConfigDict(extra="forbid")
        request: NativeCall
        audio: str

    @asynccontextmanager
    async def lifespan(app):
        try:
            runtime.prepare()
            yield
        finally:
            runtime.close()

    app = FastAPI(version=RELEASE, lifespan=lifespan)
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "candidate": "dots_tts_edit",
            "runtime": {
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "package": runtime.sdk_version,
            },
        }

    @app.post("/invoke")
    def invoke(request: Request):
        try:
            values = request.request.model_dump()
            values["fields"] = tuple(tuple(pair) for pair in values["fields"])
            compiled = NativeRequest(**values)
            audio = Audio(base64.b64decode(request.audio, validate=True))
            audio.validate()
        except Exception as exc:
            raise HTTPException(
                422, detail={"kind": "infrastructure", "message": str(exc)}
            ) from exc
        with lock:
            try:
                result = runtime.invoke(compiled, audio)
                result.validate()
            except InfrastructureError as exc:
                raise HTTPException(
                    503, detail={"kind": "infrastructure", "message": str(exc)}
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    500, detail={"kind": "candidate", "message": str(exc)}
                ) from exc
            return {
                "audio": base64.b64encode(result.data).decode("ascii"),
                "metadata": runtime.last_call_metadata,
            }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(
            Runtime(
                model_path=args.model_path,
                scratch_dir=args.scratch_dir,
                runtime_options={"precision": args.precision},
            )
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
