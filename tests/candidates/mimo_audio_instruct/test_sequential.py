"""Sequential calls retain audio chains and generation metadata."""

import base64
import json
from unittest.mock import Mock

from dotebench.candidates.mimo_audio_instruct.adapter import MiMoAudioInstruct
from dotebench.domain import GenerationRequest
from tests.candidates.helpers import session_with, wav


def test_mimo_sequence_keeps_empty_instruct_global_calls_audio_chain_and_headers(
    tmp_path,
):
    original = wav(0.0)
    outputs = [wav(0.1), wav(0.2), wav(0.3)]
    responses = []
    for index, output in enumerate(outputs):
        responses.append(
            Mock(
                ok=True,
                content=output.data,
                headers={
                    "X-Generated-Step-Count": str(282 if index == 1 else 10),
                    "X-Truncated-By-Token-Limit": ("true" if index == 1 else "false"),
                },
            )
        )
    session = session_with(responses)
    observed = []

    def observe(case_id, compiled, payload):
        observed.append((case_id, compiled, payload))

    candidate = MiMoAudioInstruct(
        url="http://mimo",
        scratch_dir=str(tmp_path / "scratch"),
        trace_dir=str(tmp_path / "traces"),
        compiler="sequential",
        session=session,
        request_observer=observe,
    )
    candidate.compilation_identity = lambda: {"version": "test"}
    candidate.prepare()

    request = GenerationRequest(
        "mimo-case",
        "en",
        original,
        '<ins>A</ins><pitch semitones="2">B</pitch><pause act="ins"/>',
    )
    assert candidate.generate(request) == outputs[-1]

    assert session.post.call_count == 3
    payloads = [call.kwargs["json"] for call in session.post.call_args_list]
    assert [payload["instruct"] for payload in payloads] == [
        "",
        "Speak with the pitch raised by 2 semitones.",
        "Speak with longer pauses throughout.",
    ]
    assert [payload["text"] for payload in payloads] == ["A B", "A B", "A B"]
    assert [compiled.scope for _, compiled, _ in observed] == [
        "text",
        "global",
        "global",
    ]
    assert [
        base64.b64decode(payload["prompt_speech"]["data"], validate=True)
        for payload in payloads
    ] == [original.data, outputs[0].data, outputs[1].data]

    trace = json.loads(candidate.last_execution_trace.read_text())
    assert [step["operation_index"] for step in trace["steps"]] == [0, 1, 2]
    assert [step["call"]["scope"] for step in trace["steps"]] == [
        "text",
        "global",
        "global",
    ]
    assert [
        step["call"]["response_headers"]["X-Truncated-By-Token-Limit"]
        for step in trace["steps"]
    ] == ["false", "true", "false"]
    assert [
        step["call"]["response_headers"]["X-Generated-Step-Count"]
        for step in trace["steps"]
    ] == ["10", "282", "10"]
    assert (
        trace["steps"][0]["output_audio"]["sha256"]
        == trace["steps"][1]["input_audio"]["sha256"]
    )
    assert (
        trace["steps"][1]["output_audio"]["sha256"]
        == trace["steps"][2]["input_audio"]["sha256"]
    )
