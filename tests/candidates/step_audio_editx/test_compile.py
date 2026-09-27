"""Compositional requests use one global control and the complete target text."""

import pytest

from dotebench.candidates.step_audio_editx.compile import compile_request
from dotebench.domain import Audio, GenerationRequest
from dotebench.instructions import parse_instruction


@pytest.mark.parametrize(
    "tag,endpoint,control",
    [
        ('<emo type="happy" level="2">word</emo>', "/edit", "happy"),
        ('<rate factor="1.2">word</rate>', "/edit", "faster"),
        ('<pitch semitones="3">word</pitch>', "/edit_freeform", "Raise the pitch"),
        ('<pause act="ins"/>word', "/edit_freeform", "Increase the pauses"),
        ('<pause act="red"/>word', "/edit_freeform", "Remove any silent portions"),
    ],
)
@pytest.mark.parametrize(
    "language,source,target", [("en", "old", "new"), ("zh", "旧词", "新词")]
)
def test_target_transcript_and_first_control(
    tag, endpoint, control, language, source, target
):
    xml = (
        f'<sub targ="{target}">{source}</sub> {tag} <pitch semitones="-7">last</pitch>'
    )
    instruction = parse_instruction(xml)
    result = compile_request(GenerationRequest("case", language, Audio(b""), xml))
    fields = dict(result.fields)
    assert result.endpoint == endpoint
    assert result.scope == "global"
    assert result.selected_operation_index == 1
    assert instruction.target_text in str(fields)
    assert source not in str(fields)
    assert "-7" not in str(fields)
    assert "saying exactly" not in str(fields)
    if endpoint == "/edit":
        assert fields["transcript"] == instruction.target_text
        assert fields["edit_info"] == control
    else:
        assert fields["instruction"].startswith(control)
        assert fields["instruction"].endswith(
            "The text corresponding to the audio is: " + instruction.target_text
        )
