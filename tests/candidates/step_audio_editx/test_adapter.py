"""Step-Audio-EditX compilation and transport contracts."""

from dotebench.candidates.step_audio_editx.adapter import StepAudioEditX
from tests.candidates.helpers import (
    assert_cli_wiring,
    assert_first_acoustic_uses_complete_target,
    assert_native_transport_one_request,
    assert_same_position_keeps_xml_order,
)
from tests.candidates.helpers import (
    audio_request as request,
)

CANDIDATE = StepAudioEditX
CANDIDATE_NAME = "step_audio_editx"


def test_first_acoustic_only_with_complete_target(tmp_path):
    assert_first_acoustic_uses_complete_target(CANDIDATE, tmp_path)


def test_same_position_keeps_xml_order(tmp_path):
    assert_same_position_keeps_xml_order(CANDIDATE, tmp_path)


def test_native_transport_one_request(tmp_path):
    assert_native_transport_one_request(CANDIDATE, tmp_path)


def test_cli_wiring(monkeypatch, tmp_path):
    assert_cli_wiring(monkeypatch, tmp_path, CANDIDATE_NAME)


def test_step_speed_sends_enum_not_factor(tmp_path):
    q = request('<rate factor="1.2">Hello</rate>')
    actual = StepAudioEditX(
        url="http://native", scratch_dir=str(tmp_path)
    ).compiler.bind_request(resolved={}, request=q)
    assert actual.endpoint == "/edit"
    assert dict(actual.fields)["edit_info"] == "faster"
    assert "1.2" not in str(actual.fields)


def test_step_expanded_instruction_contains_source_once(tmp_path):
    q = request('<pitch semitones="3">Same word</pitch> Same word')
    compiled = StepAudioEditX(
        url="http://native", scratch_dir=str(tmp_path)
    ).compiler.bind_request(resolved={}, request=q)
    assert compiled.endpoint == "/edit_freeform"
    assert (
        dict(compiled.fields)["instruction"]
        == "Raise the pitch of the entire utterance by 3 semitones. The text corresponding to the audio is: Same word Same word"
    )
