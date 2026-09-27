import pytest

from dotebench.compilers.sequential.planner import plan_operations
from dotebench.instructions import parse_instruction


@pytest.mark.parametrize(
    "xml",
    [
        '<ins>Hello</ins><sub targ="world">earth</sub>!',
        '<del>Hello</del><sub targ="world">earth</sub>!',
        '<sub targ="one">same</sub> same <sub targ="two">same</sub>',
        "<del>all</del><ins>new</ins>",
        "<ins>你好</ins><del>世界</del><ins>朋友</ins>",
        '<emo type="happy" level="2">same</emo><ins>new</ins> same',
        '<rate factor="2">a</rate> <rate factor="0.5">b</rate>',
        '<pause act="ins"/>a<pause act="red"/>',
    ],
)
def test_plan_xml_ownership(xml):
    parsed = parse_instruction(xml)
    steps = plan_operations(xml)
    assert len(steps) == len(parsed.operations)
    text = parsed.source_text
    for step in steps:
        single = parse_instruction(step.instruction_xml)
        assert len(single.operations) == 1
        assert single.source_text == step.source_text == text
        text = single.target_text
    assert text == parsed.target_text


def test_acoustic_span_excludes_adjacent_insert():
    steps = plan_operations('<emo type="happy" level="2">same</emo><ins>new</ins> same')
    assert [s.operation_index for s in steps] == [1, 0]
    op = parse_instruction(steps[1].instruction_xml).operations[0]
    assert op.content == "same"


def test_pause_serialization_uses_empty_tags():
    steps = plan_operations('a<pause act="ins"/>b<pause act="red"/>')
    assert [step.instruction_xml for step in steps] == [
        'a<pause act="ins"/>b',
        'ab<pause act="red"/>',
    ]


def test_text_chain_accepts_semantically_irrelevant_whitespace_rendering():
    xml = " b <del>b</del>-<ins>a</ins> b "
    instruction = parse_instruction(xml)
    steps = plan_operations(xml)

    assert [step.operation_index for step in steps] == [0, 1]
    assert steps[-1].target_text == "b - a b"
    assert "".join(steps[-1].target_text.split()) == "".join(
        instruction.target_text.split()
    )


@pytest.mark.parametrize(
    ("suffix", "kind", "content"),
    [
        ('<emo type="happy" level="2"> b</emo> ', "emo", "b"),
        ('<pause act="ins"/> b ', "pause", ""),
    ],
)
def test_whitespace_drift_keeps_later_operation_on_its_owned_word_boundary(
    suffix, kind, content
):
    xml = " b <del>b</del>-<ins>a</ins>" + suffix
    steps = plan_operations(xml)

    assert steps[-1].source_text == "b - a b"
    parsed = parse_instruction(steps[-1].instruction_xml)
    assert len(parsed.operations) == 1
    assert parsed.operations[0].kind == kind
    assert parsed.operations[0].content == content
    assert parsed.source_text == parsed.target_text == steps[-1].source_text
