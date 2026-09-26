"""The candidate computes its own alignment from raw inputs and owns its cache."""

import base64
import hashlib
import io
import json
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from dotebench.candidates.auk_base.aligner import create_app
from dotebench.candidates.auk_base.adapter import AuKBase
from dotebench.candidates.auk_base.alignment import CandidateAligner
from dotebench.domain import (
    AlignmentUnit,
    Audio,
    InfrastructureError,
    SourceAlignment,
)
from dotebench.instructions import parse_instruction
from dotebench.text import token_spans
from tests.candidates.auk_base.test_adapter import request, synthetic_alignment


def client(tmp_path):
    q = request('left <rate factor="2">middle</rate> right')
    identity = {"weights_sha256": "weights", "implementation_sha256": "implementation"}
    model = CandidateAligner("http://aligner", tmp_path)
    model.session = Mock()
    model.session.get.return_value.json.return_value = {
        "status": "ready",
        "role": "candidate-aligner",
        "identity": identity,
    }
    model.session.post.return_value.json.return_value = {
        "identity": identity,
        "segments": [vars(s) for s in synthetic_alignment(q).segments],
    }
    return model, q, identity


def test_online_alignment_and_cache(tmp_path):
    model, q, _ = client(tmp_path)
    text = parse_instruction(q.instruction_xml).source_text
    first, record = model.align(q.source_audio, text, q.language)
    second, reused = model.align(q.source_audio, text, q.language)
    assert first == second == synthetic_alignment(q)
    assert record == reused
    assert model.session.post.call_count == 1
    payload = model.session.post.call_args.kwargs["json"]
    assert set(payload) == {"audio", "text", "language", "identity"}
    assert payload["text"] == text
    assert first.audio_sha256 == hashlib.sha256(q.source_audio.data).hexdigest()
    path = next(tmp_path.glob("*.json"))
    corrupted = json.loads(path.read_text())
    corrupted["alignment"]["segments"][0]["end"] = 100
    path.write_text(json.dumps(corrupted))
    with pytest.raises(InfrastructureError, match="checksum"):
        model.align(q.source_audio, text, q.language)


def test_model_identity_cannot_change_during_generation(tmp_path):
    model, q, _ = client(tmp_path)
    model.prepare()
    model.session.get.return_value.json.return_value["identity"] = {
        "weights_sha256": "changed"
    }
    with pytest.raises(InfrastructureError, match="identity changed"):
        model.align(q.source_audio, "left middle right", "en")


def test_unlocatable_response_is_candidate_failure(tmp_path):
    model, q, _ = client(tmp_path)
    model.session.post.return_value.json.return_value["segments"] = []
    with pytest.raises(ValueError):
        model.align(q.source_audio, "left middle right", "en")
    assert not list(tmp_path.glob("*.json"))


def test_malformed_cache_is_infrastructure_failure(tmp_path):
    model, q, _ = client(tmp_path)
    model.align(q.source_audio, "left middle right", "en")
    next(tmp_path.glob("*.json")).write_text("{broken")
    with pytest.raises(InfrastructureError, match="storage or response"):
        model.align(q.source_audio, "left middle right", "en")


def test_malformed_service_response_is_infrastructure_failure(tmp_path):
    model, q, _ = client(tmp_path)
    del model.session.post.return_value.json.return_value["segments"]
    with pytest.raises(InfrastructureError, match="storage or response"):
        model.align(q.source_audio, "left middle right", "en")


def test_service_rejects_alignment_annotations():
    measure = Mock(return_value={"segments": []})
    identity = {"weights_sha256": "fixed"}
    app = create_app(measure, identity)
    invoke = next(route.endpoint for route in app.routes if route.path == "/align")
    payload = {
        "audio": "wav",
        "text": "words",
        "language": "en",
        "identity": identity,
    }
    with pytest.raises(HTTPException) as error:
        invoke({**payload, "alignment": []})
    assert error.value.status_code == 400
    measure.assert_not_called()
    assert invoke(payload) == {"segments": [], "identity": identity}
    assert measure.call_args.args[0] == {
        k: payload[k] for k in ("audio", "text", "language")
    }


def test_sequential_auk_realigns_each_current_local_audio_but_not_whole(tmp_path):
    source = request(
        '<sub targ="new">old</sub> <pitch semitones="2">middle</pitch> '
        '<rate factor="2">right</rate>'
    )
    info = sf.info(io.BytesIO(source.source_audio.data))

    def output(value):
        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.full(info.frames, value, dtype="float32"),
            info.samplerate,
            format="WAV",
            subtype="FLOAT",
        )
        return Audio(buffer.getvalue())

    outputs = [output(value) for value in (0.1, 0.2, 0.3)]
    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    responses = []
    for audio in outputs:
        response = Mock(ok=True, headers={})
        response.json.return_value = {
            "audio_base64": base64.b64encode(audio.data).decode(),
            "metadata": {},
        }
        responses.append(response)
    session.post.side_effect = responses

    alignment_inputs = []

    class Aligner:
        def align(self, audio, text, language):
            alignment_inputs.append((audio, text, language))
            duration = sf.info(io.BytesIO(audio.data)).duration
            spans = token_spans(text)
            segments = tuple(
                AlignmentUnit(
                    text[a:b],
                    i * duration / len(spans),
                    (i + 1) * duration / len(spans),
                )
                for i, (a, b) in enumerate(spans)
            )
            return (
                SourceAlignment(
                    hashlib.sha256(audio.data).hexdigest(),
                    hashlib.sha256(text.encode()).hexdigest(),
                    segments,
                ),
                {"input_audio_sha256": hashlib.sha256(audio.data).hexdigest()},
            )

        def close(self):
            pass

    model = AuKBase(
        url="http://service",
        scratch_dir=str(tmp_path / "scratch"),
        trace_dir=str(tmp_path / "traces"),
        compiler="sequential",
        aligner_url="http://aligner",
        session=session,
    )
    model.aligner = Aligner()
    model.prepare()

    assert model.generate(source) == outputs[-1]
    assert session.post.call_count == 3
    assert [(text, language) for _, text, language in alignment_inputs] == [
        ("new middle right", "en"),
        ("new middle right", "en"),
    ]
    assert [audio.data for audio, _, _ in alignment_inputs] == [
        outputs[0].data,
        outputs[1].data,
    ]
    payloads = [call.kwargs["json"] for call in session.post.call_args_list]
    assert payloads[0]["metadata"]["generation_scope"] == "whole"
    assert [payload["metadata"]["source_audio_sha256"] for payload in payloads[1:]] == [
        hashlib.sha256(outputs[0].data).hexdigest(),
        hashlib.sha256(outputs[1].data).hexdigest(),
    ]


def test_aligner_entrypoint_loads_registered_model(monkeypatch, tmp_path):
    import importlib.metadata
    import sys
    from unittest.mock import Mock

    import uvicorn

    from dotebench.candidates.auk_base import aligner as auk_aligner
    from dotebench.services.backends import qwen3_aligner

    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    build = Mock(return_value=lambda payload: {})
    monkeypatch.setattr(qwen3_aligner, "build", build)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aligner",
            "--model-path",
            str(tmp_path),
            "--port",
            "18001",
            "--host",
            "127.0.0.2",
        ],
    )
    auk_aligner.main()
    build.assert_called_once_with(tmp_path, "cuda:0")
    assert run.call_args.kwargs == {"host": "127.0.0.2", "port": 18001}
