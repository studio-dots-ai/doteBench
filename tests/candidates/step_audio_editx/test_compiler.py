"""Step-Audio-EditX one-take compiler contract."""

from dotebench.candidates.step_audio_editx.compile import compile_request
from dotebench.candidates.step_audio_editx.compilers.one_take import OneTakeCompiler
from tests.candidates.common.compiler_contract import (
    assert_one_take_matches_request_compiler,
)


def test_one_take_preserves_single_call():
    assert_one_take_matches_request_compiler(OneTakeCompiler, compile_request)
