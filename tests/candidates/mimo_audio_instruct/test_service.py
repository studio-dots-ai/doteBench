"""Verify text, instruction, reference audio and sampling reach native TTS."""

import base64
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient


def test_tts_boundary_and_reference_cleanup(monkeypatch, tmp_path):
    from dotebench.candidates.common import runtime

    monkeypatch.setenv("DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path))
    monkeypatch.setattr(runtime, "activate_runtime", lambda name: tmp_path)
    package = ModuleType("src")
    package.__path__ = []
    mimo_package = ModuleType("src.mimo_audio")
    mimo_package.__path__ = []
    modeling = ModuleType("src.mimo_audio.modeling_mimo_audio")

    class MiMoStopper:
        def __init__(self, **kwargs):
            self.options = kwargs

    modeling.MiMoStopper = MiMoStopper
    monkeypatch.setitem(sys.modules, "src", package)
    monkeypatch.setitem(sys.modules, "src.mimo_audio", mimo_package)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    from dotebench.candidates.mimo_audio_instruct import backend as service

    audio = io.BytesIO()
    sf.write(audio, np.zeros(1600), 16000, format="WAV")
    captured = []

    def prompt(text, **kwargs):
        assert Path(kwargs["prompt_speech"]).read_bytes() == audio.getvalue()
        captured.append((text, kwargs))
        return "INPUT_IDS"

    metadata = {
        "max_new_tokens": 282,
        "generated_step_count": 10,
        "decoded_audio_frame_count": 10,
        "truncated_by_token_limit": False,
    }
    model = SimpleNamespace(
        device="cpu",
        tokenizer=SimpleNamespace(eos_token_id=1),
        eostm_idx=2,
        im_end_idx=3,
        group_size=4,
        audio_channels=8,
        get_tts_sft_prompt=prompt,
        forward=Mock(return_value=(torch.zeros(2400), metadata)),
    )
    monkeypatch.setattr(service, "MODEL", model)
    monkeypatch.setattr(service, "TEMP_DIR", str(tmp_path))
    seeded = Mock()
    monkeypatch.setattr(service, "_set_seed", seeded)
    client = TestClient(service.app)
    assert client.get("/health").json()["runtime"] == service.RUNTIME_IDENTITY
    payload = {
        "text": "你好",
        "instruct": "用开心的语气说。",
        "read_text_only": True,
        "seed": 42,
        "max_new_tokens": 282,
        "prompt_speech": {
            "data": base64.b64encode(audio.getvalue()).decode(),
            "format": "wav",
        },
    }
    response = client.post("/tts", json=payload)
    assert response.status_code == 200, response.text
    waveform, rate = sf.read(io.BytesIO(response.content))
    assert rate == 24000 and len(waveform) == 2400
    assert captured[0][0] == "你好"
    assert captured[0][1]["instruct"] == "用开心的语气说。"
    assert captured[0][1]["read_text_only"] is True
    seeded.assert_called_once_with(42)
    call = model.forward.call_args
    assert call.args == ("INPUT_IDS",)
    assert call.kwargs["max_new_tokens"] == 282 and call.kwargs["task_name"] == "tts"
    assert call.kwargs["return_generation_metadata"] is True
    assert not list(tmp_path.iterdir())
