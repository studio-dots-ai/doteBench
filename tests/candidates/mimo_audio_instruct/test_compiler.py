"""MiMo-Audio-Instruct one-take and sequential compiler contracts."""

from dotebench.candidates.mimo_audio_instruct.compile import compile_request
from dotebench.candidates.mimo_audio_instruct.compilers.one_take import OneTakeCompiler
from dotebench.compilers.sequential import SequentialCompiler
from dotebench.domain import GenerationRequest
from tests.candidates.common.compiler_contract import (
    assert_one_take_matches_request_compiler,
)
from tests.candidates.helpers import audio_request


def test_one_take_preserves_single_call():
    assert_one_take_matches_request_compiler(OneTakeCompiler, compile_request)


def test_sequential_delegates_each_single_tag_to_one_take():
    one_take = OneTakeCompiler()
    compiler = SequentialCompiler(one_take)
    original = audio_request(
        '<ins>new</ins><pitch semitones="2">word</pitch><pause act="ins"/>'
    )
    plan = compiler.compile(original)
    assert [step.operation_index for step in plan.steps] == [0, 1, 2]
    assert all(step.binder == "one_take" for step in plan.steps)
    for step in plan.steps:
        rebound = GenerationRequest(
            step.case_id,
            step.language,
            original.source_audio,
            step.instruction_xml,
        )
        (expected_step,) = one_take.compile(rebound).steps
        assert compiler.resolve(step.binder).bind(
            step, original.source_audio, {}
        ) == one_take.bind(expected_step, original.source_audio, {})
