"""Selection, contextual edits and numeric controls at the candidate boundary."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from dotebench.candidates.fireredtts3_instruct.adapter import FireRedTTS3Instruct
from dotebench.candidates.fireredtts3_instruct.compile import (
    compile_request,
    describe_request,
    mapped_rate,
    occurrences,
)
from dotebench.domain import Audio, GenerationRequest, InfrastructureError


def request(xml, language="en"):
    return GenerationRequest("case", language, Audio(b""), xml)


@pytest.mark.parametrize(
    "xml",
    [
        '<ins>NASA cut </ins><sub targ="wrong">This is the original sentence</sub>.',
        '<del>This </del><sub targ="wrong">is a sentence</sub>.',
        "<ins>Hello </ins><del>world</del>",
        '<sub targ="new">word</sub> <sub targ="wrong">word</sub>',
        '汤七七呀<sub targ="你别这样">汤七七呀</sub><emo type="happy" level="2">我是知识分子</emo>',
        '<pitch semitones="4">Hello</pitch> <sub targ="wrong">world</sub>',
        '<pause act="red"/><pitch semitones="3">Hello</pitch>',
    ],
)
def test_only_first_and_other_tags_do_not_restrict_context(xml):
    q = request(xml)
    plan = describe_request(q)
    assert plan["selected_operation_index"] == 0
    assert "wrong" not in plan["instruction"]
    assert "instruction_xml" not in dict(compile_request(q).fields)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0.97", "0.9"),
        ("1.04", "1.1"),
        ("1.25", "1.2"),
        ("0.85", "0.9"),
        ("1.99", "2.0"),
        ("0.1", "0.5"),
        ("3", "2.0"),
        ("1", "1.0"),
    ],
)
def test_rate_mapping(value, expected):
    assert mapped_rate(value) == Decimal(expected)


def test_overlapping_repeats_are_not_unique():
    assert occurrences("哈哈哈", "哈哈") == 2


@pytest.mark.parametrize(
    "code, error",
    [("generation_failed", RuntimeError), ("out_of_memory", InfrastructureError)],
)
def test_prediction_failure_and_infrastructure_are_distinct(code, error, tmp_path):
    from tests.candidates.helpers import audio_request

    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    response = session.post.return_value
    response.ok = False
    response.json.return_value = {"detail": {"error_code": code}}
    response.text = code
    response.status_code = 500
    model = FireRedTTS3Instruct(
        url="http://service", scratch_dir=str(tmp_path), session=session
    )
    model.prepare()
    with pytest.raises(error):
        model.generate(audio_request('<pitch semitones="9">hello</pitch>'))
    assert session.post.call_count == 1
