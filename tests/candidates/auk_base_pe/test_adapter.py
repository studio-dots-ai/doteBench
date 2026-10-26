"""Cache requirements, shared execution, and generation failure accounting."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dotebench.candidates.auk_base_pe.adapter import AuKBasePE
from dotebench.candidates.auk_base_pe.instruction_cache import digest
from dotebench.domain import InfrastructureError
from dotebench.execution import GenerationRunner
from tests.candidates.helpers import assert_cli_wiring, audio_request, wav


def make_cache(tmp_path, plan, statuses=None):
    directory = tmp_path / "cache"
    (directory / "records").mkdir(parents=True)
    identity = {"schema": 1, "scope": "instruction_only", "model": "test"}
    (directory / "manifest.json").write_text(json.dumps({"identity": identity}))
    paths = []
    for index, step in enumerate(plan.steps):
        item = {
            "instruction_xml": step.instruction_xml,
            "source_text": step.source_text,
            "language": step.language,
        }
        key = digest({"identity": identity, "input": item})
        record = {
            "key": key,
            "identity": identity,
            "input": item,
            "status": statuses[index] if statuses else "ok",
            "attempts": [{"error": "official refusal reason", "calls": []}],
            "output": {"instruction": f"official instruction {index}"},
        }
        path = directory / "records" / f"{key}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    return paths


def candidate(tmp_path, compiler="one_take"):
    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    return AuKBasePE(
        instruction_cache_dir=tmp_path / "cache",
        compiler=compiler,
        url="http://native",
        scratch_dir=tmp_path / "scratch",
        session=session,
    )


def test_one_take_resolves_full_xml_without_selecting_an_operation(tmp_path):
    model = candidate(tmp_path)
    request = audio_request(
        '<sub targ="bye">hello</sub> <pitch semitones="3">world</pitch>'
    )
    plan = model.compiler.compile(request)
    assert len(plan.steps) == 1
    requirement = plan.steps[0].requirements[0]
    assert requirement.instruction_xml == request.instruction_xml
    assert requirement.name == "pe_instruction"
    paths = make_cache(tmp_path, plan)
    model.prepare()
    value = requirement.resolve(model, request.source_audio)
    native = model.compiler.bind(
        plan.steps[0], request.source_audio, {requirement.name: value}
    )
    assert native.selected_operation_index is None
    assert native.scope == "whole"
    assert dict(native.fields)["prepared"]["instruction"] == "official instruction 0"
    assert value[1]["path"] == str(paths[0])


def test_sequential_uses_shared_planner_and_previous_audio(tmp_path):
    model = candidate(tmp_path, "sequential")
    request = audio_request(
        '<sub targ="bye">hello</sub> <pitch semitones="3">world</pitch>'
    )
    plan = model.compiler.compile(request)
    assert len(plan.steps) == 2
    assert plan.steps[1].source_text == "bye world"
    make_cache(tmp_path, plan)
    outputs = [wav(0.1), wav(0.2)]
    payloads = []

    def post(url, *, json, timeout):
        payloads.append((json, Path(json["audio"]).read_bytes()))
        response = Mock(ok=True, headers={})
        response.json.return_value = {
            "audio_base64": base64.b64encode(outputs[len(payloads) - 1].data).decode()
        }
        return response

    model.session.post.side_effect = post
    model.prepare()
    assert model.generate(request) == outputs[1]
    assert payloads[0][1] == request.source_audio.data
    assert payloads[1][1] == outputs[0].data
    assert payloads[1][0]["source_text"] == "bye world"
    trace = json.loads(model.last_execution_trace.read_text())
    assert [s["operation_index"] for s in trace["steps"]] == [0, 1]
    assert all(s["requirements"][0]["name"] == "pe_instruction" for s in trace["steps"])


@pytest.mark.parametrize("status", ["unsupported", "error", "pending", "missing"])
def test_refusal_is_candidate_failure_but_cache_errors_abort(tmp_path, status):
    model = candidate(tmp_path)
    request = audio_request('<pitch semitones="3">hello</pitch>')
    paths = make_cache(tmp_path, model.compiler.compile(request), [status])
    if status == "missing":
        paths[0].unlink()
    model.prepare()
    store = Mock()
    store.source.return_value = request.source_audio
    runner = GenerationRunner(load=Mock(), store=store, candidate=model, config={})
    case = SimpleNamespace(
        id=request.id,
        language=request.language,
        instruction_xml=request.instruction_xml,
    )
    if status == "unsupported":
        result = runner.execute(case)
        assert result.status == "candidate_error"
        assert "official refusal reason" in result.error
        assert runner.aggregate([result]) == {
            "attempted": 1,
            "generated": 0,
            "failed": 1,
        }
        trace = json.loads(model.last_execution_trace.read_text())
        assert trace["steps"][0]["call"]["pe_cache"]["path"] == str(paths[0])
    else:
        with pytest.raises(InfrastructureError):
            runner.execute(case)
    model.session.post.assert_not_called()
    store.save_audio.assert_not_called()


def test_native_candidate_error_response(tmp_path):
    model = candidate(tmp_path)
    request = audio_request('<pitch semitones="3">hello</pitch>')
    make_cache(tmp_path, model.compiler.compile(request))
    model.prepare()
    response = model.session.post.return_value
    response.ok = False
    response.status_code = 422
    response.text = "PE generation failed"
    response.json.return_value = {"detail": {"error_code": "candidate_error"}}
    with pytest.raises(RuntimeError, match="PE generation failed") as error:
        model.generate(request)
    assert not isinstance(error.value, InfrastructureError)


def test_cli_wiring(monkeypatch, tmp_path):
    cache = str(tmp_path / "cache")
    kwargs = assert_cli_wiring(
        monkeypatch,
        tmp_path,
        "auk_base_pe",
        overrides=[f"candidate.options.instruction_cache_dir={cache}"],
    )
    assert kwargs["candidate"].instruction_cache_dir == cache
    assert kwargs["config"]["generation"]["options"]["instruction_cache_dir"] == cache
