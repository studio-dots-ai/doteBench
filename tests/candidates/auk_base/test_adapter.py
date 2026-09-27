"""AuK compilation: positioning, duration, isolation, and input provenance."""

import hashlib
import io
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from dotebench.candidates.auk_base.compile import compile_request as compile_native
from dotebench.domain import (
    AlignmentUnit,
    Audio,
    GenerationRequest,
    InfrastructureError,
    SourceAlignment,
)
from dotebench.instructions import parse_instruction
from dotebench.text import token_spans


def request(xml, sr=22050):
    inst = parse_instruction(xml)
    units = tuple(
        AlignmentUnit(inst.source_text[a:b], i * 0.4, (i + 1) * 0.4)
        for i, (a, b) in enumerate(token_spans(inst.source_text))
    )
    buf = io.BytesIO()
    sf.write(
        buf,
        np.zeros(round((len(units) * 0.4 + 0.2) * sr)),
        sr,
        format="WAV",
        subtype="FLOAT",
    )
    audio = Audio(buf.getvalue())
    return GenerationRequest("case", "en", audio, xml)


def fields(xml, **kwargs):
    return dict(compile_request(request(xml, **kwargs)).fields)


def test_replacement_uses_only_selected_target():
    r = request('<sub targ="new words">old</sub> stay <sub targ="SECRET">other</sub>')
    data = dict(compile_request(r).fields)
    assert data["instruction"] == "Change the spoken content to: new words stay other"
    assert data["metadata"]["target_text"] == "new words stay other"
    assert "SECRET" not in str(data)
    assert data["start_frame"] == 0
    assert (
        data["metadata"]["end_sample"]
        == sf.info(io.BytesIO(r.source_audio.data)).frames
    )


def test_following_operations_do_not_change_selected_request():
    assert fields("<ins>new</ins>old words") == fields(
        '<ins>new</ins><sub targ="other">old words</sub>'
    )


@pytest.mark.parametrize(
    "xml",
    [
        '<sub targ="new">old</sub> words',
        "<ins>new </ins>old words",
        "<del>old </del>words",
        "<del>all words</del>",
        '<emo type="happy" level="2">all words</emo>.',
        '<rate factor="2">all words</rate>',
        '<pitch semitones="2">all words</pitch>',
    ],
)
def test_whole_requests_ignore_alignment(xml):
    r = request(xml)
    good = compile_request(r)
    assert compile_request(r, alignment=None) == good
    bad = replace(
        synthetic_alignment(r),
        audio_sha256="bad",
        text_sha256="bad",
        segments=(),
    )
    assert compile_request(r, alignment=bad) == good
    assert dict(good.fields)["metadata"]["generation_scope"] == "whole"


def test_whole_target_duration_uses_utf8_ratio():
    r = request('你好 <sub targ="longer words">世界</sub>')
    data = dict(compile_request(r).fields)
    duration = sf.info(io.BytesIO(r.source_audio.data)).duration
    target = "你好 longer words"
    expected = max(
        1,
        round(
            50
            * duration
            * len(target.encode())
            / len(parse_instruction(r.instruction_xml).source_text.encode())
        ),
    )
    assert data["target_frames"] == expected


def test_full_speech_deletion_has_one_frame():
    data = fields("<del>all words</del>")
    assert data["target_frames"] == 1
    assert data["metadata"]["target_text"] == ""
    assert data["instruction"] == "Remove all spoken content."


def test_repeated_word_uses_selected_text_position():
    data = fields('same <sub targ="new">same</sub> same')
    assert data["metadata"]["target_text"] == "same new same"


def test_full_rate_includes_audio_tail():
    r = request('<rate factor="2">all words</rate>')
    data = dict(compile_request(r).fields)
    duration = sf.info(io.BytesIO(r.source_audio.data)).duration
    assert data["target_frames"] == round(duration / 2 * 50)
    assert (
        data["metadata"]["end_sample"]
        == sf.info(io.BytesIO(r.source_audio.data)).frames
    )


def test_local_rate_and_modes():
    r = request('left <rate factor="2">middle</rate> right')
    clean = dict(compile_request(r, "clean").fields)
    bridge = dict(compile_request(r, "bridge").fields)
    assert bridge == {**clean, "context_mode": "bridge"}
    assert clean["target_frames"] == 10
    assert clean["metadata"]["start_sample"] == 8820


def test_local_alignment_provenance():
    r = request('left <rate factor="2">middle</rate> right')
    with pytest.raises(ValueError, match="requires"):
        compile_request(r, alignment=None)
    with pytest.raises(ValueError, match="audio checksum"):
        compile_request(
            r, alignment=replace(synthetic_alignment(r), audio_sha256="bad")
        )
    with pytest.raises(ValueError, match="transcript checksum"):
        compile_request(r, alignment=replace(synthetic_alignment(r), text_sha256="bad"))
    with pytest.raises(ValueError, match="in order"):
        compile_request(
            r,
            alignment=replace(
                synthetic_alignment(r),
                segments=tuple(reversed(synthetic_alignment(r).segments)),
            ),
        )


@pytest.mark.parametrize("sr", [16000, 22050, 24000, 32000, 44100, 48000])
def test_whole_source_rate_boundaries(sr):
    r = request('left <sub targ="new">old</sub> right', sr=sr)
    data = dict(compile_request(r).fields)
    assert data["metadata"]["start_sample"] == 0
    assert (
        data["metadata"]["end_sample"]
        == sf.info(io.BytesIO(r.source_audio.data)).frames
    )
    assert data["metadata"]["generated_samples"] == round(
        data["target_frames"] * sr / 50
    )


def test_insertion_uses_canonical_word_boundaries():
    data = fields("<ins>new</ins>old words")
    assert data["metadata"]["target_text"] == "new old words"


@pytest.mark.parametrize(
    "sr,samples", [(24000, 106560), (44100, 31752), (22050, 15876)]
)
def test_whole_source_frames_use_exact_sample_count(sr, samples):
    r = request('<sub targ="new">old</sub>')
    buf = io.BytesIO()
    sf.write(buf, np.zeros(samples), sr, format="WAV", subtype="FLOAT")
    r = replace(r, source_audio=Audio(buf.getvalue()))
    data = dict(compile_request(r).fields)
    assert data["end_frame"] == (samples * 50 + sr - 1) // sr


@pytest.mark.parametrize(
    "code,error",
    [("candidate_error", RuntimeError), ("out_of_memory", InfrastructureError)],
)
def test_candidate_failure_and_infrastructure_are_distinct(code, error, tmp_path):
    from unittest.mock import Mock

    from dotebench.candidates.auk_base.adapter import AuKBase

    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    response = session.post.return_value
    response.ok = False
    response.json.return_value = {"detail": {"error_code": code}}
    response.text = code
    response.status_code = 422
    model = AuKBase(url="http://service", scratch_dir=str(tmp_path), session=session)
    model.prepare()
    with pytest.raises(error):
        model.generate(request('<sub targ="new">old</sub>'))
    assert session.post.call_count == 1


def synthetic_alignment(request):
    text = parse_instruction(request.instruction_xml).source_text
    return SourceAlignment(
        hashlib.sha256(request.source_audio.data).hexdigest(),
        hashlib.sha256(text.encode()).hexdigest(),
        tuple(
            AlignmentUnit(text[a:b], i * 0.4, (i + 1) * 0.4)
            for i, (a, b) in enumerate(token_spans(text))
        ),
    )


_AUTO = object()


def compile_request(request, context_mode="clean", *, alignment=_AUTO):
    if alignment is _AUTO:
        alignment = synthetic_alignment(request)
    return compile_native(request, context_mode, alignment=alignment)


@pytest.mark.parametrize("context", ["clean", "bridge"])
@pytest.mark.parametrize("interval", [(0.4, 0.4), (0.401, 0.402)])
def test_empty_local_window_is_identity(context, interval, tmp_path):
    from dotebench.candidates.auk_base.adapter import AuKBase

    r = request('before <pitch semitones="-3">word</pitch> after')
    alignment = synthetic_alignment(r)
    units = list(alignment.segments)
    units[1] = replace(units[1], start=interval[0], end=interval[1])
    alignment = replace(alignment, segments=tuple(units))
    compiled = compile_native(r, context, alignment=alignment)
    assert compiled.transport == "identity"
    assert dict(compiled.fields)["metadata"]["source_window"] == list(interval)
    model = AuKBase(url="http://unused", scratch_dir=str(tmp_path))
    model.ready = True
    assert model.invoke(compiled, r.source_audio) is r.source_audio
    assert model.last_call_metadata["scope"] == "identity"


@pytest.mark.parametrize("context", ["clean", "bridge"])
def test_identity_step_preserves_current_audio_and_continues(
    context, tmp_path, monkeypatch
):
    import json
    from dataclasses import asdict

    from dotebench.candidates.auk_base.adapter import AuKBase
    from dotebench.models.native import NativeCandidate

    r = request(
        '<pitch semitones="-3">old</pitch> next <emo type="happy" level="2">last</emo>'
    )
    alignment = synthetic_alignment(r)
    units = list(alignment.segments)
    units[0] = replace(units[0], start=0.0, end=0.0)
    alignment = replace(alignment, segments=tuple(units))
    model = AuKBase(
        url="http://unused",
        scratch_dir=str(tmp_path),
        compiler="sequential",
        context_mode=context,
    )
    model.ready = True
    monkeypatch.setattr(
        model, "resolve_alignment", lambda audio, text, language: (alignment, {})
    )
    calls = []
    output = request("another utterance").source_audio

    def invoke(self, compiled, audio):
        calls.append((compiled, audio))
        self.last_call_metadata = {"native_request": asdict(compiled)}
        return output

    monkeypatch.setattr(NativeCandidate, "invoke", invoke)
    assert model.generate(r) is output
    assert len(calls) == 1 and calls[0][1] is r.source_audio
    trace = json.loads(model.last_execution_trace.read_text())
    assert trace["status"] == "ok" and len(trace["steps"]) == 2
    first = trace["steps"][0]
    assert first["call"]["scope"] == "identity"
    assert first["input_audio"]["sha256"] == first["output_audio"]["sha256"]
