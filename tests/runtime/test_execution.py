import io
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import soundfile as sf

from dotebench.domain import Annotations, AudioAsset, Case
from dotebench.evaluation.base import EvaluationProtocol, EvaluationResult
from dotebench.execution import EvaluationRunner, GenerationRunner
from dotebench.instructions import parse_instruction
from dotebench.models.base import Audio, CandidateModel
from dotebench.storage import RunStore, sha256


@pytest.fixture
def fixture(tmp_path):
    stream = io.BytesIO()
    sf.write(stream, np.sin(np.arange(800) * 0.1), 16000, format="WAV")
    data = stream.getvalue()
    (tmp_path / "source.wav").write_bytes(data)
    xml = 'say <sub targ="world">hello</sub>'
    instruction = parse_instruction(xml)
    case = Case(
        "test-1",
        "en",
        AudioAsset("source.wav", sha256(data)),
        xml,
        Annotations(),
        instruction,
    )
    return case, RunStore(tmp_path / "run", tmp_path)


class FixtureCandidate(CandidateModel):
    def compilation_identity(self):
        return {"sha256": "fixture-compiler"}

    def prepare(self):
        pass

    def generate(self, request):
        return Audio(request.source_audio.data)

    def close(self):
        pass


class Double(FixtureCandidate):
    def generate(self, request):
        assert not hasattr(request, "annotations")
        assert not hasattr(request, "case")
        with pytest.raises(FrozenInstanceError):
            request.instruction_xml = "overwrite"
        return Audio(request.source_audio.data)


class Evaluator(EvaluationProtocol):
    def identity(self):
        return {"fixture": "evaluator"}

    closed = False

    def prepare(self):
        pass

    def close(self):
        self.closed = True

    def evaluate(self, request):
        return EvaluationResult(
            request.case.id, {"score": float(request.generation.status == "ok")}
        )

    def aggregate(self, results):
        return {"score": sum(r.metrics["score"] for r in results) / len(results)}


def generate(case, store, component=None):
    return GenerationRunner(
        load=lambda: [case],
        store=store,
        candidate=component or FixtureCandidate(),
        config={"dataset": ["fixture-v1"]},
    ).run()


def evaluate(case, store, component=None, version="fixture-v1"):
    return EvaluationRunner(
        load=lambda: [case],
        store=store,
        evaluator=component or Evaluator(),
        config={"dataset": [version], "evaluation": {}},
    ).run()


@pytest.mark.parametrize("candidate", [FixtureCandidate, Double])
def test_full_workflow(fixture, candidate):
    case, store = fixture
    assert generate(case, store, candidate())["generated"] == 1
    assert (store.root / "audio/test-1.wav").read_bytes() == store.source(case).data
    protocol = Evaluator()
    assert evaluate(case, store, protocol) == {"score": 1.0}
    assert protocol.closed
    assert store.read_json("summary.json")["status"] == "complete"
    assert store.read_json("run.json")["case_ids"] == ["test-1"]


@pytest.mark.parametrize("failure", ["raise", "missing", "invalid"])
def test_candidate_failure_scored_by_protocol(fixture, failure):
    case, store = fixture

    class Bad(FixtureCandidate):
        closed = False

        def generate(self, request):
            if failure == "raise":
                raise RuntimeError("candidate failed")
            return None if failure == "missing" else Audio(b"invalid")

        def close(self):
            self.closed = True

    model = Bad()
    assert generate(case, store, model)["failed"] == 1
    assert model.closed
    assert evaluate(case, store)["score"] == 0


def test_duplicate_selection(fixture):
    case, store = fixture
    with pytest.raises(ValueError, match="unique"):
        GenerationRunner(
            load=lambda: [case, case],
            store=store,
            candidate=FixtureCandidate(),
            config={},
        ).run()
    assert not store.root.exists()


def test_bad_source_is_infrastructure(fixture):
    case, store = fixture
    case = replace(case, source_audio=AudioAsset("source.wav", "0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        generate(case, store)
    assert store.read_json("run.json")["generation_status"] == "incomplete"
    assert store.rows("generation.jsonl") == []


@pytest.mark.parametrize("failure", ["missing", "corrupt", "duplicate", "version"])
def test_evaluation_integrity(fixture, failure):
    case, store = fixture
    generate(case, store)
    if failure == "missing":
        (store.root / "audio/test-1.wav").unlink()
    if failure == "corrupt":
        (store.root / "audio/test-1.wav").write_bytes(b"bad")
    if failure == "duplicate":
        store.append("generation.jsonl", store.rows("generation.jsonl")[0])
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate(case, store, version="v2" if failure == "version" else "fixture-v1")
    assert store.read_json("run.json").get("evaluation_status") != "complete"


def test_evaluator_error_no_success(fixture):
    case, store = fixture
    generate(case, store)

    class Bad(Evaluator):
        def evaluate(self, request):
            raise ConnectionError("metric unavailable")

    evaluator = Bad()
    with pytest.raises(ConnectionError):
        evaluate(case, store, evaluator)
    assert evaluator.closed
    assert store.read_json("summary.json")["status"] == "incomplete"


def test_close_failure_no_success(fixture):
    case, store = fixture
    generate(case, store)

    class Bad(Evaluator):
        def close(self):
            raise RuntimeError("release failed")

    with pytest.raises(RuntimeError):
        evaluate(case, store, Bad())
    assert store.read_json("summary.json")["status"] == "incomplete"


def test_template_cannot_be_overridden():
    with pytest.raises(TypeError):

        class BadRunner(GenerationRunner):
            def run(self):
                pass


def test_transport_fault_is_infrastructure(fixture):
    from dotebench.domain import InfrastructureError

    case, store = fixture

    class Bad(FixtureCandidate):
        def generate(self, request):
            raise InfrastructureError("transport down")

    with pytest.raises(InfrastructureError):
        generate(case, store, Bad())
    assert store.read_json("run.json")["generation_status"] == "incomplete"
    assert store.rows("generation.jsonl") == []


def test_finish_failure_is_incomplete(fixture):
    case, store = fixture
    generate(case, store)

    class Bad(EvaluationRunner):
        def finish(self, summary):
            super().finish(summary)
            raise OSError("finalization failure")

    with pytest.raises(OSError):
        Bad(
            load=lambda: [case],
            store=store,
            evaluator=Evaluator(),
            config={"dataset": ["fixture-v1"], "evaluation": {}},
        ).run()
    assert store.read_json("run.json")["evaluation_status"] == "incomplete"
    assert store.read_json("summary.json")["status"] == "incomplete"


def test_compiler_change_cannot_complete_run(fixture):
    case, store = fixture

    class ChangingCompiler(FixtureCandidate):
        changed = False
        closed = False

        def compilation_identity(self):
            return {"version": "changed" if self.changed else "original"}

        def generate(self, request):
            self.changed = True
            return super().generate(request)

        def close(self):
            self.closed = True

    component = ChangingCompiler()
    with pytest.raises(ValueError, match="Compiler changed"):
        generate(case, store, component)
    assert component.closed
    assert store.read_json("run.json")["generation_status"] == "incomplete"


def test_generation_records_compiler_identity(fixture):
    case, store = fixture
    component = FixtureCandidate()
    generate(case, store, component)
    assert (
        store.read_json("run.json")["compilation"] == component.compilation_identity()
    )


def test_generation_excludes_evaluation_alignment(fixture):
    from dotebench.domain import (
        AlignmentUnit,
        SourceAlignment,
    )

    case, store = fixture
    alignment = SourceAlignment(
        case.source_audio.sha256,
        sha256(case.source_text.encode()),
        (AlignmentUnit("say", 0, 0.02), AlignmentUnit("hello", 0.02, 0.05)),
    )
    case = replace(
        case,
        annotations=Annotations(source_emotion="happy", source_alignment=alignment),
    )

    class Inspect(FixtureCandidate):
        def generate(self, request):
            assert not hasattr(request, "source_alignment")
            assert not hasattr(request, "instruction")
            assert not hasattr(request, "annotations")
            assert not hasattr(request, "source_emotion")
            return request.source_audio

    assert generate(case, store, Inspect())["generated"] == 1


def test_evaluation_uses_manifest_case_independently_of_compiler(fixture, monkeypatch):
    from dotebench import compilation

    case, store = fixture
    generate(case, store)
    (store.root / "trace.json").write_text(
        '{"instruction_xml":"unrelated instruction"}'
    )

    def unavailable(*args, **kwargs):
        raise AssertionError("Evaluation must not resolve a compiler")

    monkeypatch.setattr(compilation, "compiler_identity", unavailable)

    class Inspect(Evaluator):
        def evaluate(self, request):
            assert request.case is case
            assert request.case.instruction_xml == case.instruction_xml
            assert request.source_audio == store.source(case)
            assert not hasattr(request, "compiler")
            return super().evaluate(request)

    assert evaluate(case, store, Inspect()) == {"score": 1.0}


@pytest.mark.parametrize("drift", [False, True])
def test_prepared_protocol_identity_is_recorded_and_frozen(fixture, drift):
    case, store = fixture
    generate(case, store)

    class Identified(Evaluator):
        def __init__(self):
            self.snapshot = {
                "protocol_version": "test",
                "services": {"model": "original"},
            }

        def identity(self):
            return self.snapshot

        def evaluate(self, request):
            if drift:
                self.snapshot["services"]["model"] = "changed"
            return super().evaluate(request)

    if drift:
        with pytest.raises(ValueError, match="identity changed"):
            evaluate(case, store, Identified())
    else:
        evaluate(case, store, Identified())
    assert (
        store.read_json("run.json")["evaluation"]["identity"]["services"]["model"]
        == "original"
    )


@pytest.mark.parametrize("field", ["identity", "protocol_version"])
def test_caller_cannot_override_protocol_identity(fixture, field):
    case, store = fixture
    generate(case, store)
    runner = EvaluationRunner(
        load=lambda: [case],
        store=store,
        evaluator=Evaluator(),
        config={"dataset": ["fixture-v1"], "evaluation": {field: "claimed"}},
    )
    with pytest.raises(ValueError, match="supplied by the prepared protocol"):
        runner.run()


def test_evaluation_requires_explicit_identity(fixture):
    case, store = fixture
    generate(case, store)

    class MissingIdentity(Evaluator):
        def identity(self):
            return

    with pytest.raises(TypeError, match="identity must be a mapping"):
        evaluate(case, store, MissingIdentity())
