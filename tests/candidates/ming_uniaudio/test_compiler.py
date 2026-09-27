"""Ming-UniAudio one-take compiler contract."""

from dotebench.candidates.ming_uniaudio.compile import compile_request
from dotebench.candidates.ming_uniaudio.compilers.one_take import OneTakeCompiler
from tests.candidates.common.compiler_contract import (
    assert_one_take_matches_request_compiler,
)


def test_one_take_preserves_single_call():
    assert_one_take_matches_request_compiler(OneTakeCompiler, compile_request)
