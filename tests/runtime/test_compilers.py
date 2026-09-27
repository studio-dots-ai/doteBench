"""Compiler composition and the common multi-call executor."""

import io
import json

import numpy as np
import pytest
import soundfile as sf

from dotebench.compilers.base import CompilationError
from dotebench.compilers.one_take import OneTakeCompiler
from dotebench.compilers.sequential import SequentialCompiler
from dotebench.domain import Audio, GenerationRequest, InfrastructureError
from dotebench.models.base import CompiledCandidate
from dotebench.models.requests import NativeRequest


def wav(value=0.0):
    stream = io.BytesIO()
    sf.write(
        stream,
        np.full(160, value, dtype="float32"),
        16000,
        format="WAV",
        subtype="FLOAT",
    )
    return Audio(stream.getvalue())


def request(xml):
    return GenerationRequest("case", "en", wav(), xml)


class FakeOneTake(OneTakeCompiler):
    candidate_name = "fake"

    def bind_request(self, request, resolved):
        return NativeRequest(
            "/edit",
            (("instruction_xml", request.instruction_xml),),
            "audio",
            "file",
            "wav",
            0,
            "test",
        )


class FakeCandidate(CompiledCandidate):
    candidate_name = "fake"

    def __init__(self, compiler, trace_dir, outputs):
        super().__init__(compiler=compiler, trace_dir=trace_dir)
        self.outputs = iter(outputs)
        self.inputs = []

    def compilation_identity(self):
        return {"sha256": "test-compiler"}

    def prepare(self):
        pass

    def invoke(self, request, audio):
        self.inputs.append((request, audio))
        self.last_call_metadata = {"scope": request.scope}
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return output

    def close(self):
        pass


def test_executor_chains_audio_and_binds_root_identity(tmp_path):
    compiler = SequentialCompiler(FakeOneTake())
    outputs = [wav(0.1), wav(0.2)]
    candidate = FakeCandidate(compiler, tmp_path, outputs)
    original = request('<ins>new</ins><pitch semitones="2">word</pitch>')
    assert candidate.generate(original) == outputs[-1]
    assert [audio for _, audio in candidate.inputs] == [
        original.source_audio,
        outputs[0],
    ]
    assert all(bound.case_id == original.id for bound, _ in candidate.inputs)
    assert (
        json.loads(candidate.last_execution_trace.read_text())["compiler"]
        == candidate.compilation_identity()
    )


@pytest.mark.parametrize(
    "compiler",
    [FakeOneTake(), SequentialCompiler(FakeOneTake())],
    ids=["one_take", "sequential"],
)
def test_one_take_and_sequential_use_the_same_executor(tmp_path, compiler):
    output = wav(0.1)
    candidate = FakeCandidate(compiler, tmp_path, [output])
    original = request('<pitch semitones="2">word</pitch>')
    assert candidate.generate(original) == output
    trace = json.loads(candidate.last_execution_trace.read_text())
    assert trace["status"] == "ok"
    assert len(trace["steps"]) == 1
    assert (
        trace["steps"][0]["input_audio"]["sha256"]
        != trace["steps"][0]["output_audio"]["sha256"]
    )


@pytest.mark.parametrize(
    "failure,exception",
    [
        (Audio(b"invalid"), Exception),
        (RuntimeError("candidate failed"), RuntimeError),
        (InfrastructureError("service failed"), InfrastructureError),
    ],
)
def test_executor_failure_trace_and_error_boundary(tmp_path, failure, exception):
    compiler = SequentialCompiler(FakeOneTake())
    first = wav(0.1)
    candidate = FakeCandidate(compiler, tmp_path, [first, failure])
    original = request('<ins>new</ins><pitch semitones="2">word</pitch>')
    with pytest.raises(exception):
        candidate.generate(original)
    trace = json.loads(candidate.last_execution_trace.read_text())
    assert trace["status"] == "failed"
    assert [step["status"] for step in trace["steps"]] == ["ok", "failed"]
    assert (
        trace["steps"][0]["output_audio"]["sha256"]
        == trace["steps"][1]["input_audio"]["sha256"]
    )
    index = json.loads((tmp_path / "index.jsonl").read_text())
    assert index["status"] == "failed"


def test_planning_and_binding_faults_are_compilation_errors(tmp_path):
    class BadPlan(FakeOneTake):
        def compile(self, request):
            raise ValueError("bad compiler config")

    candidate = FakeCandidate(BadPlan(), tmp_path / "plan", [])
    with pytest.raises(CompilationError, match="planning"):
        candidate.generate(request('<pitch semitones="2">word</pitch>'))

    class BadBind(FakeOneTake):
        def bind_request(self, request, resolved):
            raise ValueError("bad bound call")

    candidate = FakeCandidate(BadBind(), tmp_path / "bind", [])
    with pytest.raises(CompilationError, match="binding"):
        candidate.generate(request('<pitch semitones="2">word</pitch>'))
    trace = json.loads(candidate.last_execution_trace.read_text())
    assert trace["steps"][0]["status"] == "failed"
