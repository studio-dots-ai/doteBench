"""Sequential calls retain audio chains and generation metadata."""

import base64
import json
from unittest.mock import Mock

from dotebench.candidates.step_audio_editx.adapter import StepAudioEditX
from dotebench.domain import GenerationRequest
from tests.candidates.helpers import session_with, wav


def test_step_sequence_retains_json_generation_metadata(tmp_path):
    original = wav(0.0)
    outputs = [wav(0.1), wav(0.2)]
    metadata = [
        {
            "max_new_tokens": 1875,
            "generated_token_count": 100,
            "finish_reason": "stop",
            "truncated_by_token_limit": False,
        },
        {
            "max_new_tokens": 1875,
            "generated_token_count": 1875,
            "finish_reason": "length",
            "truncated_by_token_limit": True,
        },
    ]
    responses = []
    for output, generation in zip(outputs, metadata):
        response = Mock(ok=True, headers={})
        response.json.return_value = {
            "status": "success",
            "audio_base64": base64.b64encode(output.data).decode("ascii"),
            **generation,
        }
        responses.append(response)
    session = session_with(responses)
    observed_inputs = []

    def observe(_case_id, compiled, payload):
        with open(payload[compiled.audio_field], "rb") as stream:
            observed_inputs.append(stream.read())

    candidate = StepAudioEditX(
        url="http://step",
        scratch_dir=str(tmp_path / "scratch"),
        trace_dir=str(tmp_path / "traces"),
        compiler="sequential",
        session=session,
        request_observer=observe,
    )
    candidate.compilation_identity = lambda: {"version": "test"}
    candidate.prepare()

    request = GenerationRequest(
        "step-case",
        "en",
        original,
        '<rate factor="0.8">A</rate><pitch semitones="2">B</pitch>',
    )
    assert candidate.generate(request) == outputs[-1]
    assert observed_inputs == [original.data, outputs[0].data]

    trace = json.loads(candidate.last_execution_trace.read_text())
    captured = [step["call"]["response_metadata"] for step in trace["steps"]]
    assert captured == [
        {"status": "success", **metadata[0]},
        {"status": "success", **metadata[1]},
    ]
    assert all("audio_base64" not in record for record in captured)
