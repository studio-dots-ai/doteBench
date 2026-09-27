"""Exercise all Step-Audio-EditX model call boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from fastapi.testclient import TestClient


def test_native_endpoints_forward_exact_controls(monkeypatch, tmp_path):
    from dotebench.candidates.common import runtime

    monkeypatch.setenv("DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path))
    monkeypatch.setattr(runtime, "activate_runtime", lambda name: tmp_path)
    from dotebench.candidates.step_audio_editx import backend as service

    model = SimpleNamespace(
        **{
            name: Mock(return_value=(torch.zeros(1600), 16000, {}))
            for name in ["clone", "edit", "edit_freeform"]
        }
    )
    monkeypatch.setattr(service, "MODEL", model)
    monkeypatch.setattr(service, "OUTPUT_ROOT", tmp_path)
    client = TestClient(service.app)
    for method in [client.get, client.post]:
        assert method("/health").json()["runtime"] == service.RUNTIME_IDENTITY
    assert client.get("/info").status_code == 200
    assert (
        client.post(
            "/tts",
            json={
                "prompt_audio_path": "source.wav",
                "prompt_text": "Hello",
                "text": "Goodbye",
                "max_new_tokens": 1875,
            },
        ).status_code
        == 200
    )
    model.clone.assert_called_once_with(
        "source.wav",
        "Hello",
        "Goodbye",
        max_new_tokens=1875,
        return_generation_metadata=True,
    )
    assert (
        client.post(
            "/edit",
            json={
                "audio_path": "source.wav",
                "transcript": "Hello",
                "edit_type": "speed",
                "edit_info": "faster",
                "max_new_tokens": 1875,
            },
        ).status_code
        == 200
    )
    model.edit.assert_called_once_with(
        prompt_wav_path="source.wav",
        prompt_text="Hello",
        edit_type="speed",
        edit_info="faster",
        target_text=None,
        max_new_tokens=1875,
        return_generation_metadata=True,
    )
    instruction = "Raise the pitch of the entire utterance by 3 semitones. The text corresponding to the audio is: Hello"
    assert (
        client.post(
            "/edit_freeform",
            json={
                "audio_path": "source.wav",
                "instruction": instruction,
                "max_new_tokens": 1875,
            },
        ).status_code
        == 200
    )
    model.edit_freeform.assert_called_once_with(
        prompt_wav_path="source.wav",
        instruction=instruction,
        max_new_tokens=1875,
        return_generation_metadata=True,
    )
