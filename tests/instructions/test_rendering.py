"""Public rendering contracts for XML and candidate-selected text edits."""

import pytest

from dotebench.instructions import (
    Span,
    parse_instruction,
    render_replacement,
    render_source,
    render_target,
)


@pytest.mark.parametrize(
    "xml, source, target",
    [
        ('  A <sub targ="new">old</sub> word !  ', "  A old word !  ", "A new word!"),
        ("<ins>Hello</ins><del>old</del> world!", "old world!", "Hello world!"),
        ("<del>all</del>", "all", ""),
        ('你<sub targ="朋友">同学</sub>好。', "你同学好。", "你朋友好。"),
        ('  <emo type="happy" level="2">a  b</emo>\n', "  a  b\n", "  a  b\n"),
        ('a<pause act="ins"/>  b', "a  b", "a  b"),
        ("Tom<ins>'s</ins> book", "Tom book", "Tom's book"),
        ("a<ins>-</ins>word", "aword", "a-word"),
    ],
)
def test_public_xml_renderers(xml, source, target):
    assert render_source(xml) == source
    assert render_target(xml) == target
    parsed = parse_instruction(xml, source_text=source, target_text=target)
    assert (parsed.source_text, parsed.target_text) == (source, target)


@pytest.mark.parametrize("render", [render_source, render_target])
@pytest.mark.parametrize(
    "xml",
    [
        "",
        "<!DOCTYPE x>text",
        "<ins>x",
        "<unknown>x</unknown>",
        '<emo type="happy" level="2"><sub targ="new">old</sub></emo>',
    ],
)
def test_renderers_validate_complete_instruction(render, xml):
    with pytest.raises(ValueError):
        render(xml)


@pytest.mark.parametrize(
    "text, span, replacement, expected",
    [
        ("old word!", Span(0, 3), "new", "new word!"),
        ("hello!", Span(5, 5), "world", "hello world!"),
        ("all", Span(0, 3), "", ""),
        ("你好世界", Span(2, 4), "朋友", "你好朋友"),
        ("Tom book", Span(3, 3), "'s", "Tom's book"),
        ("aword", Span(1, 1), "-", "a-word"),
        (" b b- b ", Span(3, 4), "", "b - b"),
    ],
)
def test_selected_replacement(text, span, replacement, expected):
    assert render_replacement(text, span, replacement) == expected


@pytest.mark.parametrize("span", [Span(-1, 0), Span(2, 1), Span(0, 4)])
def test_replacement_rejects_out_of_bounds_span(span):
    with pytest.raises(ValueError, match="span"):
        render_replacement("abc", span, "x")
