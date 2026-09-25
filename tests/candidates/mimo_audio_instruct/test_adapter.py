"""MiMo-Audio-Instruct compilation and transport contracts."""

import pytest

from dotebench.candidates.mimo_audio_instruct.adapter import MiMoAudioInstruct
from dotebench.instructions import parse_instruction
from tests.candidates.helpers import (
    assert_cli_wiring,
    assert_first_acoustic_uses_complete_target,
    assert_native_transport_one_request,
    assert_same_position_keeps_xml_order,
)
from tests.candidates.helpers import (
    audio_request as request,
)

CANDIDATE = MiMoAudioInstruct
CANDIDATE_NAME = "mimo_audio_instruct"


def test_first_acoustic_only_with_complete_target(tmp_path):
    assert_first_acoustic_uses_complete_target(CANDIDATE, tmp_path)


def test_same_position_keeps_xml_order(tmp_path):
    assert_same_position_keeps_xml_order(CANDIDATE, tmp_path)


def test_native_transport_one_request(tmp_path):
    assert_native_transport_one_request(CANDIDATE, tmp_path)


def test_cli_wiring(monkeypatch, tmp_path):
    assert_cli_wiring(monkeypatch, tmp_path, CANDIDATE_NAME)


def test_mimo_chinese_and_escaping(tmp_path):
    q = request(
        '<sub targ="你好 &amp; 再见">你好</sub><emo type="happy" level="2">朋友</emo>',
        "zh",
    )
    actual = dict(
        MiMoAudioInstruct(url="http://native", scratch_dir=str(tmp_path))
        .compiler.bind_request(resolved={}, request=q)
        .fields
    )
    assert actual["text"] == parse_instruction(q.instruction_xml).target_text
    assert actual["instruct"] == "用开心的语气说。"
    assert actual["read_text_only"] is True


@pytest.mark.parametrize(
    "xml,language,expected",
    [
        (
            '<rate factor="1.2">甲</rate><pitch semitones="-3">乙</pitch>',
            "zh",
            "用原来1.2倍的语速说。",
        ),
        (
            '<rate factor="0.8">one</rate><rate factor="1.2">two</rate>',
            "en",
            "Speak at 0.8 times the original speaking rate.",
        ),
        (
            '<pitch semitones="-3">甲</pitch><pitch semitones="3">乙</pitch>',
            "zh",
            "将整段话的音高降低3个半音。",
        ),
        (
            '<pitch semitones="3">one</pitch><pitch semitones="-3">two</pitch>',
            "en",
            "Speak with the pitch raised by 3 semitones.",
        ),
        (
            '甲<pause act="red"/>乙<pause act="ins"/>',
            "zh",
            "减少整段话中的停顿。",
        ),
        (
            'one<pause act="ins"/>two<pause act="red"/>',
            "en",
            "Speak with longer pauses throughout.",
        ),
        (
            '<emo type="melancholic" level="2">甲</emo><emo type="happy" level="2">乙</emo>',
            "zh",
            "用忧郁的语气说。",
        ),
    ],
)
def test_mimo_exact_global_control(xml, language, expected, tmp_path):
    q = request(xml, language)
    compiled = MiMoAudioInstruct(
        url="http://native", scratch_dir=str(tmp_path)
    ).compiler.bind_request(resolved={}, request=q)
    fields = dict(compiled.fields)
    assert fields["instruct"] == expected
    assert fields["text"] == parse_instruction(q.instruction_xml).target_text
    assert compiled.scope == "global"
