"""Ming-UniAudio compilation, transport, and failure contracts."""

from unittest.mock import Mock

import pytest

from dotebench.candidates.ming_uniaudio.adapter import MingUniAudio
from dotebench.candidates.ming_uniaudio.compile import MING_PREFIX
from dotebench.domain import InfrastructureError
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

CANDIDATE = MingUniAudio
CANDIDATE_NAME = "ming_uniaudio"


def test_first_acoustic_only_with_complete_target(tmp_path):
    assert_first_acoustic_uses_complete_target(CANDIDATE, tmp_path)


def test_same_position_keeps_xml_order(tmp_path):
    assert_same_position_keeps_xml_order(CANDIDATE, tmp_path)


def test_native_transport_one_request(tmp_path):
    assert_native_transport_one_request(CANDIDATE, tmp_path)


def test_cli_wiring(monkeypatch, tmp_path):
    assert_cli_wiring(monkeypatch, tmp_path, CANDIDATE_NAME)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_ming_prefix_and_no_second_emotion(language, tmp_path):
    q = request(
        '<emo type="happy" level="2">Hello</emo> <emo type="sad" level="1">world</emo>',
        language,
    )
    actual = dict(
        MingUniAudio(url="http://native", scratch_dir=str(tmp_path))
        .compiler.bind_request(resolved={}, request=q)
        .fields
    )
    assert actual["instruction"] == MING_PREFIX + "change the emotion to happy mood."
    assert actual["use_cot"] is False
    assert actual["instruction"].count(MING_PREFIX) == 1


def test_ming_text_keeps_all_edits(tmp_path):
    q = request('<sub targ="Hi">Hello</sub> <del>old</del> <ins>new</ins> world')
    actual = dict(
        MingUniAudio(url="http://native", scratch_dir=str(tmp_path))
        .compiler.bind_request(resolved={}, request=q)
        .fields
    )["instruction"]
    assert "substitute 'Hello' with 'Hi'" in actual
    assert "delete 'old'" in actual
    assert "insert 'new'" in actual


def test_unprepared_is_infrastructure(tmp_path):
    adapter = MingUniAudio(url="http://native", scratch_dir=str(tmp_path))
    with pytest.raises(InfrastructureError):
        adapter.generate(request('<pitch semitones="3">Hi</pitch>'))


def test_ming_text_preserves_xml_node_whitespace(tmp_path):
    q = request("Hello<ins> dear &amp; kind </ins>world<del> today</del>.")
    compiled = MingUniAudio(
        url="http://native", scratch_dir=str(tmp_path)
    ).compiler.bind_request(resolved={}, request=q)
    instruction = dict(compiled.fields)["instruction"]
    assert "insert ' dear & kind '" in instruction
    assert "delete ' today'" in instruction
    assert "final transcript is exactly: 'Hello dear & kind world.'" in instruction
    assert parse_instruction(q.instruction_xml).operations[0].content == " dear & kind "


@pytest.mark.parametrize(
    "xml,expected",
    [
        ('<rate factor="0.7">Hello</rate>', "adjusts the speed to 0.7."),
        ('<pitch semitones="-3">Hello</pitch>', "shifts the pitch by -3 steps."),
    ],
)
def test_ming_cookbook_acoustic_commands(xml, expected, tmp_path):
    compiled = MingUniAudio(
        url="http://native", scratch_dir=str(tmp_path)
    ).compiler.bind_request(resolved={}, request=request(xml))
    assert dict(compiled.fields)["instruction"] == MING_PREFIX + expected


@pytest.mark.parametrize(
    "status,detail,exception",
    [
        (500, {"error_code": "no_audio_tokens"}, RuntimeError),
        (500, "CUDA out of memory", InfrastructureError),
        (422, "Invalid request", InfrastructureError),
        (503, "Model is not loaded", InfrastructureError),
    ],
)
def test_native_failure_contract(status, detail, exception, tmp_path):
    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    response = session.post.return_value
    response.ok = False
    response.status_code = status
    response.text = str(detail)
    response.json.return_value = {"detail": detail}
    adapter = MingUniAudio(
        url="http://native", scratch_dir=str(tmp_path), session=session
    )
    adapter.prepare()
    with pytest.raises(exception) as caught:
        adapter.generate(request('<pitch semitones="3">Hello</pitch>'))
    assert type(caught.value) is exception
    adapter.close()


def test_verified_service_requires_runtime_identity(tmp_path):
    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    adapter = MingUniAudio(
        url="http://native",
        scratch_dir=str(tmp_path),
        session=session,
        require_runtime_identity=True,
    )
    with pytest.raises(InfrastructureError, match="did not provide"):
        adapter.prepare()
    session.post.assert_not_called()


def test_mismatched_runtime_is_infrastructure_error(tmp_path):
    session = Mock()
    session.get.return_value.json.return_value = {
        "status": "healthy",
        "runtime": {"sha256": "wrong"},
    }
    adapter = MingUniAudio(
        url="http://native",
        scratch_dir=str(tmp_path),
        session=session,
        require_runtime_identity=True,
    )
    with pytest.raises(InfrastructureError, match="identity mismatch"):
        adapter.prepare()
    session.post.assert_not_called()


@pytest.mark.parametrize(
    ("xml", "expected"),
    [
        ("<ins>Hello</ins> world", True),
        ("<del>Hello</del> world", True),
        ('<sub targ="Hello">Hello</sub> world', True),
        ('<ins>你好</ins><del>世界</del><sub targ="朋友">大家</sub>', True),
        ('<emo type="happy" level="2">Hello</emo>', False),
        ('<pitch semitones="3">Hello</pitch>', False),
        ('<rate factor="0.8">Hello</rate>', False),
        ('Hello<pause act="ins"/>', False),
        ('<ins>Hello</ins><pause act="ins"/> world', False),
        ('<sub targ="Hi">Hello</sub> <pitch semitones="3">world</pitch>', False),
        ("Hello world", False),
    ],
)
def test_ming_cot_depends_on_all_operation_tags(xml, expected, tmp_path):
    candidate = MingUniAudio(url="http://native", scratch_dir=str(tmp_path))
    fields = dict(
        candidate.compiler.bind_request(resolved={}, request=request(xml)).fields
    )
    assert fields["use_cot"] is expected
    assert fields["seed"] == 1895
    assert fields["max_audio_seconds"] == 45.0


def test_ming_sequential_cot_at_transport_boundary(tmp_path):
    import base64

    q = request('<pitch semitones="3">world</pitch><ins>Hello</ins><pause act="ins"/>')
    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    response = session.post.return_value
    response.ok = True
    response.headers = {}
    response.json.return_value = {
        "audio_base64": base64.b64encode(q.source_audio.data).decode()
    }
    observed = []
    candidate = MingUniAudio(
        url="http://native",
        scratch_dir=str(tmp_path / "scratch"),
        trace_dir=str(tmp_path / "traces"),
        compiler="sequential",
        session=session,
        request_observer=lambda cid, compiled, payload: observed.append(payload.copy()),
    )
    candidate.prepare()
    try:
        candidate.generate(q)
    finally:
        candidate.close()
    assert [p["use_cot"] for p in observed] == [True, False, False]
