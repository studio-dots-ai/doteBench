"""FireRedTTS3-Instruct one-take compiler contract."""

from dotebench.candidates.fireredtts3_instruct.compile import compile_request
from dotebench.candidates.fireredtts3_instruct.compilers.one_take import OneTakeCompiler
from tests.candidates.common.compiler_contract import (
    assert_one_take_matches_request_compiler,
)
from tests.candidates.helpers import assert_cli_wiring


def test_one_take_preserves_single_call():
    assert_one_take_matches_request_compiler(OneTakeCompiler, compile_request)


def test_cli_wiring(monkeypatch, tmp_path):
    assert_cli_wiring(monkeypatch, tmp_path, "fireredtts3_instruct")
