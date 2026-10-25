"""Official PE audio preparation and generation wrapper with a CPU stub model."""

import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="module")
def official(tmp_path_factory):
    pytest.importorskip("torch")
    source = os.environ.get("DOTEBENCH_AUK_SOURCE")
    if not source:
        pytest.skip("Set DOTEBENCH_AUK_SOURCE to the pinned official checkout")
    from dotebench.candidates.common.runtime import prepare_runtime

    runtime = prepare_runtime(
        "auk_base_pe",
        upstream=source,
        destination=tmp_path_factory.mktemp("auk-pe-runtime"),
    )
    sys.path.insert(0, str(runtime / "src"))
    from auk.infer import pe

    from dotebench.candidates.auk_base_pe import backend

    yield pe, backend
    sys.path.remove(str(runtime / "src"))


def request(backend, tmp_path, seconds=1):
    path = tmp_path / "input.wav"
    sf.write(path, np.full(round(seconds * 16000), 0.1), 16000)
    return backend.PERequest(
        audio=str(path),
        source_text="hello world",
        language="en",
        prepared={
            "instruction": "Use this exact cached instruction.",
            "task_type": "speed_edit",
            "operation_subtype": None,
            "params_extracted": {"speed_multiplier": 1.5},
            "prepared_params": {"speed_multiplier": 1.5},
            "language": "en",
            "text_language": None,
        },
        metadata={"source_audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    )


def test_current_audio_duration_and_provided_transcript(
    official, tmp_path, monkeypatch
):
    pe, backend = official
    monkeypatch.setattr(pe, "_vad_speech_bounds", lambda _: None)
    monkeypatch.setattr(
        pe, "_render_instruction", Mock(side_effect=AssertionError("cached"))
    )
    enhancer = pe.PromptEnhancer.__new__(pe.PromptEnhancer)
    enhancer._call_llm = Mock(side_effect=AssertionError("no LLM in audio preparation"))
    enhancer._transcribe = Mock(side_effect=AssertionError("provided transcript"))
    durations = []
    for seconds in [1, 2]:
        with backend.prepare_cached(
            enhancer, request(backend, tmp_path, seconds)
        ) as prepared:
            durations.append(prepared.gen_seconds)
            assert prepared.asr.text == "hello world"
            assert prepared.instruction == "Use this exact cached instruction."
            assert prepared.duration_details["original_duration_sec"] == seconds
            assert prepared.duration_source == "input_duration_div_speed"
    assert durations == pytest.approx([0.66, 1.34])
    enhancer._call_llm.assert_not_called()
    enhancer._transcribe.assert_not_called()


def test_official_generation_wrapper_and_cleanup(official, tmp_path, monkeypatch):
    import torch

    pe, backend = official
    monkeypatch.setattr(pe, "_vad_speech_bounds", lambda _: (0.2, 0.8))
    req = request(backend, tmp_path)
    model = Mock()
    model.generate.return_value = (torch.zeros(1, 240), 24000)
    directory = tmp_path / "trace"
    directory.mkdir()
    output, metadata = backend.generate(
        model, req, pe.PromptEnhancer.__new__(pe.PromptEnhancer), directory
    )
    args, kwargs = model.generate.call_args
    assert args[0][0]["content"][0]["text"] == req.prepared["instruction"]
    assert kwargs["seed"] == 44
    assert kwargs["nfe"] == 32
    assert kwargs["cfg_strength"] == 2.0
    assert kwargs["audio"] != req.audio
    assert not Path(kwargs["audio"]).exists()
    assert Path(metadata["model_input"]["path"]).is_file()
    assert metadata["pe"]["duration_details"]["vad_bounds_sec"] == [0.2, 0.8]
    assert output.startswith(b"RIFF")


def test_service_preserves_candidate_error_boundary(official, tmp_path, monkeypatch):
    pe, backend = official
    from fastapi import HTTPException

    monkeypatch.setattr(backend.base, "engine", object())
    monkeypatch.setenv("DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path))
    monkeypatch.setattr(
        backend, "generate", Mock(side_effect=pe.PromptEnhancerError("PE failure"))
    )
    with pytest.raises(HTTPException) as error:
        backend.edit_pe(request(backend, tmp_path))
    assert error.value.status_code == 422
    assert error.value.detail["error_code"] == "candidate_error"
    monkeypatch.setattr(
        backend, "generate", Mock(side_effect=OSError("storage failure"))
    )
    with pytest.raises(OSError):
        backend.edit_pe(request(backend, tmp_path))


def test_instruct_tts_retries_with_failure_feedback(official):
    pe, _ = official
    from dotebench.candidates.auk_base_pe.instruction_cache import (
        INSTRUCT_TTS_ERROR,
        OfficialPE,
        prepare_record,
    )

    backend = OfficialPE.__new__(OfficialPE)
    backend.pe = pe
    backend.api_key = "test-secret"
    backend.enhancer = Mock()
    backend.enhancer._classify.return_value = (
        pe._Classified(True, "instruct_tts", None, {}, False, "test", "en", "en"),
        None,
    )
    item = {
        "instruction_xml": '<emo type="happy">Hello</emo>',
        "source_text": "Hello",
        "language": "en",
    }
    record = prepare_record(item, {}, backend)
    assert record["status"] == "unsupported"
    assert len(record["attempts"]) == 5
    assert INSTRUCT_TTS_ERROR in record["attempts"][1]["instruction"]
    backend.enhancer._normalize_params.assert_not_called()
