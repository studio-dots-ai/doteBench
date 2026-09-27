"""Verify final native model arguments without loading weights."""

import base64
import io
import sys
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from dotebench.candidates.fireredtts3_instruct.backend import create_app


@pytest.mark.parametrize("endpoint", ["semantic_edit", "acoustic_edit"])
def test_full_audio_and_single_call(endpoint, tmp_path, monkeypatch):
    fake = ModuleType("fireredtts3.utils.utils")
    fake.fix_seed = lambda seed: None
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    calls = []

    class Model:
        def generate_semantic_edit(self, **kwargs):
            calls.append(("semantic_edit", kwargs))
            return torch.zeros(1, 240), 24000, "model edited text"

        def generate_acoustic_edit(self, **kwargs):
            calls.append(("acoustic_edit", kwargs))
            return torch.zeros(1, 240), 24000

    source = tmp_path / "input.wav"
    sf.write(source, np.zeros(21 * 16000, dtype=np.float32), 16000, subtype="FLOAT")
    client = TestClient(create_app(Model(), {"test": "identity"}, tmp_path / "outputs"))
    response = client.post(
        "/" + endpoint,
        json={
            "case_id": "test",
            "audio_path": str(source),
            "instruction": "instruction",
        },
    )
    assert response.status_code == 200, response.text
    assert len(calls) == 1 and calls[0][0] == endpoint
    args = calls[0][1]
    assert args["audio_in"].shape == (1, 21 * 16000)
    assert set(args) == {
        "instruction",
        "audio_in",
        "audio_in_sr",
        "n_timesteps",
        "inference_cfg",
        "seed",
    }
    assert (
        args["n_timesteps"] == 10
        and args["seed"] == 1234
        and args["inference_cfg"] == 1.2
    )
    output, sr = sf.read(io.BytesIO(base64.b64decode(response.json()["audio_base64"])))
    assert sr == 24000 and len(output) == 240
    assert client.get("/health").json()["runtime"] == {"test": "identity"}
    assert response.json()["metadata"]["case_id"] == "test"
