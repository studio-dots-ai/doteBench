"""dots.tts.edit one-take requests match the public SDK payload."""

from dotebench.candidates.dots_tts_edit.compile import compile_public_runtime_request
from dotebench.candidates.dots_tts_edit.compilers.one_take import OneTakeCompiler
from tests.candidates.helpers import assert_cli_wiring, audio_request


def test_one_take_is_the_same_sdk_payload():
    options = {"use_xvector": "auto", "num_steps": 10}
    original = audio_request('<rate factor="1.2">hello</rate>')
    compiler = OneTakeCompiler(options)
    (step,) = compiler.compile(original).steps
    compiled = compiler.bind(step, original.source_audio, {})
    assert compiled.transport == "dots_sdk"
    assert dict(compiled.fields) == compile_public_runtime_request(original, options)


def test_cli_wiring(monkeypatch, tmp_path):
    assert_cli_wiring(monkeypatch, tmp_path, "dots_tts_edit")
