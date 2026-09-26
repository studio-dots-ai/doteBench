"""Exercise the native HTTP boundary without loading model weights."""

import base64
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient


def test_health_and_final_edit_messages(monkeypatch, tmp_path):
    from dotebench.candidates.common import runtime

    errors = ModuleType("generation_errors")
    errors.NoAudioTokensError = RuntimeError
    monkeypatch.setitem(sys.modules, errors.__name__, errors)
    monkeypatch.setenv("DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setattr(runtime, "activate_runtime", lambda name: tmp_path)
    from dotebench.candidates.ming_uniaudio import backend as service

    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(1600), 16000)
    captured = []

    def speech_edit(**kwargs):
        captured.append(kwargs)
        Path(kwargs["output_wav_path"]).write_bytes(source.read_bytes())
        return None, "Hello"

    monkeypatch.setattr(
        service, "MODEL", SimpleNamespace(speech_edit=speech_edit, sample_rate=16000)
    )
    monkeypatch.setattr(service, "OUTPUT_ROOT", tmp_path / "outputs")
    seeded = Mock()
    monkeypatch.setattr(service, "seed_everything", seeded)
    client = TestClient(service.app)
    for method in [client.get, client.post]:
        assert method("/health").json()["runtime"] == service.RUNTIME_IDENTITY
    instruction = "Please recognize the language of this speech and transcribe it. And change the emotion to happy mood."
    response = client.post(
        "/edit",
        json={
            "audio_path": str(source),
            "instruction": instruction,
            "seed": 1895,
            "use_cot": False,
            "max_audio_seconds": 45,
        },
    )
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["audio_base64"]) == source.read_bytes()
    assert captured[0]["messages"] == [
        {
            "role": "HUMAN",
            "content": [
                {"type": "audio", "audio": str(source), "target_sample_rate": 16000},
                {"type": "text", "text": f"<prompt>{instruction}\n</prompt>"},
            ],
        }
    ]
    assert captured[0]["use_cot"] is False
    assert captured[0]["max_audio_seconds"] == 45
    seeded.assert_called_once_with(1895)
    assert (
        client.post(
            "/edit",
            json={"audio_path": str(source), "instruction": "<prompt>bad</prompt>"},
        ).status_code
        == 422
    )
