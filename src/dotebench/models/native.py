"""Shared transport and lifecycle for native model services."""

import base64
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from dotebench.domain import InfrastructureError

from .base import Audio, CompiledCandidate
from .requests import NativeRequest


class NativeCandidate(CompiledCandidate):
    """Transport one bound model call to a native candidate service."""

    def __init__(
        self,
        *,
        url: str,
        scratch_dir: str,
        timeout: float = 930,
        session=None,
        request_observer=None,
        require_runtime_identity: bool = False,
        compiler,
        trace_dir: str | None = None,
    ):
        self.url = url.rstrip("/")
        self.scratch_dir = Path(scratch_dir)
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.require_runtime_identity = require_runtime_identity
        self.request_observer = request_observer
        self.ready = False
        self.runtime = None
        super().__init__(
            compiler=compiler,
            trace_dir=Path(trace_dir) if trace_dir else self.scratch_dir / "sequences",
        )

    def is_candidate_failure(self, detail):
        return (
            isinstance(detail, dict) and detail.get("error_code") == "no_audio_tokens"
        )

    def prepare(self):
        try:
            response = self.session.get(self.url + "/health", timeout=self.timeout)
            response.raise_for_status()
            health = response.json()
            if (
                health.get("status") == "not_loaded"
                or health.get("model_loaded") is False
            ):
                raise InfrastructureError("Native model is not loaded")
        except requests.RequestException as exc:
            raise InfrastructureError("Native model service is unavailable") from exc
        self.runtime = health.get("runtime")
        if self.require_runtime_identity and self.runtime is None:
            raise InfrastructureError(
                "Native service did not provide its runtime identity"
            )
        if self.runtime is not None:
            from dotebench.candidates.common.runtime import runtime_identity

            if self.runtime != runtime_identity(self.candidate_name):
                raise InfrastructureError("Native service runtime identity mismatch")
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.ready = True

    def invoke(self, compiled: NativeRequest, audio: Audio) -> Audio:
        if not self.ready:
            raise InfrastructureError("Native candidate has not been prepared")
        if compiled.transport != "http":
            raise InfrastructureError("Native service received a non-HTTP request")
        self.last_call_metadata = {
            "native_request": asdict(compiled),
            "scope": compiled.scope,
        }
        with TemporaryDirectory(dir=self.scratch_dir, prefix="source-") as directory:
            source = Path(directory) / "source.wav"
            source.write_bytes(audio.data)
            payload = compiled.payload(audio, source)
            if self.request_observer is not None:
                self.request_observer(compiled.case_id, compiled, payload)
            try:
                response = self.session.post(
                    self.url + compiled.endpoint, json=payload, timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                raise InfrastructureError("Native candidate transport failed") from exc
            if not response.ok:
                try:
                    detail = response.json().get("detail")
                except (ValueError, AttributeError):
                    detail = None
                message = f"Native request failed: HTTP {response.status_code}: {response.text[:2000]}"
                if self.is_candidate_failure(detail):
                    raise RuntimeError(message)
                raise InfrastructureError(message)
            try:
                self.last_call_metadata["response_headers"] = dict(
                    getattr(response, "headers", {})
                )
                if compiled.response_encoding == "wav":
                    data = response.content
                elif compiled.response_encoding == "audio_base64":
                    self.last_call_metadata["response_metadata"] = {
                        k: v for k, v in response.json().items() if k != "audio_base64"
                    }
                    data = base64.b64decode(
                        response.json()["audio_base64"], validate=True
                    )
                else:
                    raise InfrastructureError("Unsupported native response encoding")
            except (KeyError, TypeError, ValueError) as exc:
                raise InfrastructureError(
                    "Malformed native generation response"
                ) from exc
            return Audio(data)

    def runtime_identity(self):
        if self.runtime is not None:
            from dotebench.candidates.common.runtime import runtime_identity

            if self.runtime != runtime_identity(self.candidate_name):
                raise InfrastructureError(
                    "Native runtime sources changed during generation"
                )
        return self.runtime

    def close(self):
        self.ready = False
        self.session.close()
