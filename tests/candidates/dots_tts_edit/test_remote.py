import io
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

from dotebench.candidates.dots_tts_edit.adapter import DotsTtsEdit
from dotebench.compilers.one_take import OneTakeCompiler
from dotebench.domain import Audio, GenerationRequest, InfrastructureError
from dotebench.models.requests import NativeRequest
from dotebench.services import resolve_service_graph, start_services


def wav():
    stream = io.BytesIO()
    sf.write(stream, np.zeros(160), 16000, format="WAV")
    return Audio(stream.getvalue())


class CallCompiler(OneTakeCompiler):
    candidate_name = "fake"

    def __init__(self, action="echo"):
        self.action = action

    def identity(self):
        return {"sha256": "test-compiler"}

    def bind_request(self, request, resolved):
        return NativeRequest(
            "/model-call",
            (("action", self.action),),
            "audio",
            "base64",
            "wav",
            None,
            "test",
            transport="test",
        )


def service_graph(port):
    code = f"""
import uvicorn
from dotebench.domain import InfrastructureError
from dotebench.candidates.dots_tts_edit.backend import create_app

class Backend:
    sdk_version = "fixture"
    last_call_metadata = {{}}

    def prepare(self):
        pass

    def runtime_identity(self):
        return {{"sha256": "runtime"}}

    def invoke(self, request, audio):
        action = dict(request.fields)["action"]
        if action == "infrastructure":
            raise InfrastructureError("failure classification sentinel")
        if action == "candidate":
            raise ValueError("failure classification sentinel")
        self.last_call_metadata = {{"server": True, "scope": request.scope}}
        return audio

    def close(self):
        pass

uvicorn.run(create_app(Backend()), host="127.0.0.1", port={port})
"""
    return resolve_service_graph(
        {
            "candidate": {
                "config_key": "test",
                "instance_key": "test",
                "url": f"http://127.0.0.1:{port}",
                "health_path": "/health",
                "startup_timeout_sec": 30,
                "command": [sys.executable, "-c", code],
            }
        }
    )


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_compiled_call_http_roundtrip_and_request_boundary(tmp_path):
    graph = service_graph(free_port())

    with start_services(
        graph=graph, project_root=Path.cwd(), log_dir=tmp_path / "logs"
    ):
        remote = DotsTtsEdit(
            url=graph.direct_services.url("candidate"), scratch_dir=tmp_path
        )
        remote.compiler = CallCompiler()
        remote.prepare()
        audio = wav()
        result = remote.generate(GenerationRequest("one", "en", audio, "hello"))
        assert result == audio
        assert remote.last_execution_trace.is_file()
        assert remote.runtime_identity()["package"] == "fixture"
        assert remote.last_call_metadata["server"] is True

        # The service accepts bound calls only; raw benchmark instructions have no API.
        assert (
            requests.post(
                remote.url + "/generate",
                json={"instruction_xml": "hello"},
                timeout=5,
            ).status_code
            == 404
        )
        assert (
            requests.post(
                remote.url + "/invoke",
                json={"request": {"instruction_xml": "hello"}, "audio": "bad"},
                timeout=5,
            ).status_code
            == 422
        )
        remote.close()


@pytest.mark.parametrize(
    "action,expected",
    [
        ("infrastructure", InfrastructureError),
        ("candidate", RuntimeError),
    ],
)
def test_remote_failure_classification(tmp_path, action, expected):
    graph = service_graph(free_port())

    with start_services(
        graph=graph, project_root=Path.cwd(), log_dir=tmp_path / action
    ):
        remote = DotsTtsEdit(
            url=graph.direct_services.url("candidate"), scratch_dir=tmp_path
        )
        remote.compiler = CallCompiler(action)
        remote.prepare()
        with pytest.raises(expected) as caught:
            remote.generate(GenerationRequest("one", "en", wav(), "hello"))
        assert type(caught.value) is expected
        remote.close()
