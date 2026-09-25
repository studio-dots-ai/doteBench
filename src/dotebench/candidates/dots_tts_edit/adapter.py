"""HTTP adapter for the dots.tts.edit service."""

import base64
from dataclasses import asdict
from pathlib import Path

import requests

from dotebench.domain import Audio, InfrastructureError
from dotebench.models.base import CompiledCandidate
from .compilers.one_take import OneTakeCompiler
from .compilers.sequential import SequentialCompiler


class DotsTtsEdit(CompiledCandidate):
    candidate_name = "dots_tts_edit"

    def __init__(
        self,
        *,
        url,
        scratch_dir,
        generation_options=None,
        compiler="one_take",
        trace_dir=None,
        timeout=930,
        session=None,
    ):
        options = dict(generation_options or {})
        if set(options) - {
            "use_xvector",
            "speaker_scale",
            "ode_method",
            "num_steps",
            "guidance_scale",
        }:
            raise ValueError("Unsupported generation options")
        implementations = {
            "one_take": OneTakeCompiler,
            "sequential": SequentialCompiler,
        }
        if compiler not in implementations:
            raise ValueError("compiler must be one_take or sequential")
        super().__init__(
            compiler=implementations[compiler](options),
            trace_dir=Path(trace_dir) if trace_dir else Path(scratch_dir) / "sequences",
        )
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.runtime = None

    def prepare(self):
        try:
            response = self.session.get(self.url + "/health", timeout=self.timeout)
            response.raise_for_status()
            health = response.json()
            if (
                health["candidate"] != self.candidate_name
                or health["status"] != "ready"
            ):
                raise ValueError("Candidate service identity mismatch")
            self.runtime = health["runtime"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise InfrastructureError(
                "Candidate service is unavailable or mismatched"
            ) from exc

    def invoke(self, compiled, audio: Audio) -> Audio:
        self.last_call_metadata = {
            "native_request": asdict(compiled),
            "scope": compiled.scope,
        }
        try:
            response = self.session.post(
                self.url + "/invoke",
                json={
                    "request": asdict(compiled),
                    "audio": base64.b64encode(audio.data).decode("ascii"),
                },
                timeout=self.timeout,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise InfrastructureError("Candidate service transport failed") from exc
        if not response.ok:
            try:
                detail = response.json()["detail"]
            except (ValueError, KeyError, TypeError) as exc:
                raise InfrastructureError("Malformed candidate error response") from exc
            if not isinstance(detail, dict) or not isinstance(
                detail.get("message"), str
            ):
                raise InfrastructureError("Malformed candidate error response")
            if detail.get("kind") == "candidate":
                raise RuntimeError(detail["message"])
            if detail.get("kind") != "infrastructure":
                raise InfrastructureError("Malformed candidate error response")
            raise InfrastructureError(detail["message"])
        try:
            result = response.json()
            if not isinstance(result["metadata"], dict):
                raise TypeError("Candidate metadata must be an object")
            self.last_call_metadata.update(result["metadata"])
            return Audio(base64.b64decode(result["audio"], validate=True))
        except (ValueError, KeyError, TypeError) as exc:
            raise InfrastructureError("Invalid candidate service response") from exc

    def runtime_identity(self):
        return self.runtime

    def close(self):
        self.session.close()
