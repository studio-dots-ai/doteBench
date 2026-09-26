from pathlib import Path

import numpy as np
import pytest

from dotebench.candidates.dots_tts_edit.backend import Runtime as DotsRuntime
from dotebench.candidates.dots_tts_edit.compilers.one_take import OneTakeCompiler
from dotebench.instructions import parse_instruction
from dotebench.models.base import Audio, GenerationRequest


@pytest.mark.parametrize(
    ("xml", "native"),
    [
        (
            '<pitch semitones="-4">降低</pitch> &amp; <rate factor="1.20">faster</rate>',
            "<pitch, semitones=-4>降低</pitch> &amp; <rate, factor=1.20>faster</rate>",
        ),
        (
            '<emo type="happy" level="2">Hello</emo> <pitch semitones="+3">there</pitch><pause act="red"/>',
            '<emo type="happy" level="2">Hello</emo> <pitch, semitones=+3>there</pitch><pause act="red" level="2"/>',
        ),
        (
            '"Yes, yes," she <rate factor="1.12">hurried</rate>, pulling her hand <rate factor="0.88">gently</rate> away from him. Presently it stole back to his coat sleeve.',
            '"Yes, yes," she <rate, factor=1.12>hurried</rate>, pulling her hand <rate, factor=0.88>gently</rate> away from him. Presently it stole back to his coat sleeve.',
        ),
    ],
)
def test_native_training_dialect(xml, native):
    from dotebench.candidates.dots_tts_edit.compile import compile_request

    request = GenerationRequest("fixture", "en", Audio(b""), xml)
    result = compile_request(request, {})
    assert result["instruction"] == native
    parsed = parse_instruction(xml)
    assert result["source_text"] == parsed.source_text
    assert result["target_text"] == parsed.target_text
    assert request.instruction_xml == xml


def test_public_runtime_contract(tmp_path):
    calls = []
    import io

    import soundfile as sf

    stream = io.BytesIO()
    sf.write(stream, np.zeros(160), 16000, format="WAV", subtype="PCM_16")
    source_bytes = stream.getvalue()

    class Runtime:
        def generate_edit(self, **kwargs):
            calls.append(kwargs)
            assert Path(kwargs["source_audio_path"]).read_bytes() == source_bytes
            return {"audio": np.ones(64), "sample_rate": 16000}

    def factory(path, **kwargs):
        assert path == str(tmp_path)
        return Runtime()

    model = DotsRuntime(
        model_path=str(tmp_path),
        scratch_dir=str(tmp_path / "scratch"),
        runtime_factory=factory,
    )
    instruction = 'a <pitch semitones="3">word</pitch>'
    model.prepare()
    request = GenerationRequest("one", "en", Audio(source_bytes), instruction)
    compiled = OneTakeCompiler({"num_steps": 10}).bind_request(request, {})
    audio = model.invoke(compiled, request.source_audio)
    audio.validate()
    assert calls[0]["instruction"] == "a <pitch, semitones=3>word</pitch>"
    assert calls[0]["source_text"] == "[EN]a word"
    assert calls[0]["target_text"] == "[EN]a word"
    assert calls[0]["num_steps"] == 10
    assert not list((tmp_path / "scratch").glob("source-*"))
    model.close()
    assert model.runtime is None


def test_public_runtime_preserves_paper_language_prefixes():
    from dotebench.candidates.dots_tts_edit.compile import (
        compile_public_runtime_request,
    )

    result = compile_public_runtime_request(
        GenerationRequest(
            "zh", "zh", Audio(b""), '请<rate factor="0.80">慢说</rate>。'
        ),
        {},
    )
    assert result["source_text"] == result["target_text"] == "[ZH]请慢说。"
    assert result["instruction"] == "请<rate, factor=0.80>慢说</rate>。"


def test_fixed_model_pause_strength_and_derived_transcripts():
    import pytest

    from dotebench.candidates.dots_tts_edit.compile import compile_request
    from dotebench.instructions import parse_instruction

    xml = (
        'Say <sub targ="goodbye">hello</sub><pause act="ins"/> world<pause act="red"/>'
    )
    request = GenerationRequest("one", "en", Audio(b""), xml)
    result = compile_request(request, {})
    assert (
        result["instruction"]
        == 'Say <sub targ="goodbye">hello</sub><pause act="ins" level="2"/> world<pause act="red" level="2"/>'
    )
    assert result["source_text"] == parse_instruction(xml).source_text
    assert result["target_text"] == parse_instruction(xml).target_text
    with pytest.raises(ValueError):
        parse_instruction(result["instruction"])
    assert request.instruction_xml == xml
