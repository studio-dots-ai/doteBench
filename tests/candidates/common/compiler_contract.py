"""Reusable assertions for a candidate's one-take compiler."""

from tests.candidates.helpers import audio_request


def assert_one_take_matches_request_compiler(compiler_class, compile_request):
    compiler = compiler_class()
    original = audio_request('<pitch semitones="2">hello</pitch>')
    (step,) = compiler.compile(original).steps
    assert compiler.bind(step, original.source_audio, {}) == compile_request(original)
