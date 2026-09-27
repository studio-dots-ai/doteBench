import pytest

from dotebench.instructions import parse_instruction


def test_chinese_text_spans():
    value = parse_instruction('甲<ins>乙</ins><sub targ="丁">丙</sub><del>戊</del>己')
    assert value.source_text == "甲丙戊己"
    assert value.target_text == "甲乙丁己"
    ins, sub, delete = value.operations
    assert (ins.source.start, ins.source.end, ins.target.start, ins.target.end) == (
        1,
        1,
        1,
        2,
    )
    assert (sub.source.start, sub.source.end, sub.target.start, sub.target.end) == (
        1,
        2,
        2,
        3,
    )
    assert (
        delete.source.start,
        delete.source.end,
        delete.target.start,
        delete.target.end,
    ) == (2, 3, 3, 3)


def test_english_normalization():
    value = parse_instruction(
        'The <sub targ="large">small </sub><del>old </del>house<ins>is blue</ins>.'
    )
    assert value.source_text == "The small old house."
    assert value.target_text == "The large house is blue."
    assert (
        value.target_text[
            value.operations[0].target.start : value.operations[0].target.end
        ]
        == "large"
    )


def test_xml_escapes():
    value = parse_instruction('A &amp; <sub targ="&quot;B&quot; &lt; C">D &gt; E</sub>')
    assert value.source_text == "A & D > E"
    assert value.target_text == 'A &"B" < C'


def test_mixed_acoustic_tags_do_not_change_spacing():
    value = parse_instruction(
        '&quot;<emo type="happy" level="2">Hello</emo>, <sub targ="world">earth</sub>!'
    )
    assert value.target_text == '"Hello, world!'
    assert value.operations[0].target.start == 1


def test_acoustic_offsets_preserve_whitespace():
    value = parse_instruction(
        '甲 <pitch semitones="-2">乙</pitch>  <rate factor="1.2">丙</rate><pause act="ins"/>丁'
    )
    assert value.source_text == value.target_text == "甲 乙  丙丁"
    assert [(op.kind, op.source.start, op.source.end) for op in value.operations] == [
        ("pitch", 2, 3),
        ("rate", 5, 6),
        ("pause", 6, 6),
    ]


@pytest.mark.parametrize(
    "xml",
    [
        "<foo>x</foo>",
        '<ins bad="1">x</ins>',
        "<sub>x</sub>",
        '<sub targ="">x</sub>',
        '<emo type="bogus" level="2">x</emo>',
        '<emo type="happy" level="4">x</emo>',
        '<pitch semitones="nan">x</pitch>',
        '<rate factor="0">x</rate>',
        '<pause act="delete"/>',
        '<pause act="ins">x</pause>',
        "<ins><del>x</del></ins>",
        '<emo type="happy" level="2"><rate factor="1.2">x</rate></emo>',
        "<ins>x<del>y</ins></del>",
        "<ins>x",
        "<ins/>",
        "<!DOCTYPE x><ins>x</ins>",
        '<?xml version="1.0"?><ins>x</ins>',
        "A & B",
        '<sub targ="x" targ="y">z</sub>',
        "<rate, factor=1.2>x</rate>",
    ],
)
def test_reject_invalid(xml):
    with pytest.raises(ValueError):
        parse_instruction(xml)


def test_frozen_text_validation():
    with pytest.raises(ValueError, match="source transcript"):
        parse_instruction("abc", source_text="def")
    with pytest.raises(ValueError, match="target transcript"):
        parse_instruction("abc", target_text="def")
