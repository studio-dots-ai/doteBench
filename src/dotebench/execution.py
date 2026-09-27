"""Sequential template workflow with injected generation/evaluation capabilities."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import final

from dotebench.domain import InfrastructureError
from dotebench.evaluation.base import EvaluationRequest, EvaluationResult
from dotebench.models.base import Audio, GenerationRequest, GenerationResult


class BenchmarkRunner(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            raise TypeError("Runner extensions implement hooks, not run()")

    def __init__(self, *, load, store, component, config):
        self.load = load
        self.store = store
        self.component = component
        self.config = config

    @final
    def run(self):
        initialized = False
        completed = False
        try:
            cases = tuple(self.load())
            ids = [case.id for case in cases]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("Selection must contain unique, nonempty case IDs")
            self.begin(cases)
            initialized = True
            self.component.prepare()
            self.prepared()
            results = []
            for case in cases:
                result = self.execute(case)
                self.record(result)
                results.append(result)
            summary = self.aggregate(results)
            # Closing is part of completion; its failure cannot certify success.
            completed = True
        except BaseException:
            if initialized:
                self.mark("incomplete")
            raise
        finally:
            try:
                self.component.close()
            except BaseException:
                if initialized:
                    self.mark("incomplete")
                raise
        if completed:
            try:
                self.finish(summary)
            except BaseException:
                self.mark("incomplete")
                raise
            return summary

    def prepared(self):
        pass

    def mark(self, status):
        config = self.store.read_json("run.json")
        config[self.stage] = status
        self.store.write_json("run.json", config)
        if status == "incomplete" and self.stage == "evaluation_status":
            self.store.write_json("summary.json", {"status": "incomplete"})

    @abstractmethod
    def begin(self, cases): ...

    @abstractmethod
    def execute(self, case): ...

    @abstractmethod
    def record(self, result): ...

    @abstractmethod
    def aggregate(self, results): ...

    @abstractmethod
    def finish(self, summary): ...


class GenerationRunner(BenchmarkRunner):
    def __init__(self, *, load, store, candidate, config):
        super().__init__(load=load, store=store, component=candidate, config=config)

    stage = "generation_status"

    def begin(self, cases):
        self.compiler_identity = self.component.compilation_identity()
        config = {
            **self.config,
            "case_ids": [c.id for c in cases],
            self.stage: "running",
        }
        if self.compiler_identity is not None:
            config["compilation"] = self.compiler_identity
        self.store.initialize(config)

    def prepared(self):
        runtime = self.component.runtime_identity()
        if runtime is not None:
            config = self.store.read_json("run.json")
            config["runtime"] = runtime
            self.store.write_json("run.json", config)

    def execute(self, case):
        source = self.store.source(case)
        request = GenerationRequest(
            case.id,
            case.language,
            source,
            case.instruction_xml,
        )
        try:
            audio = self.component.generate(request)
            if not isinstance(audio, Audio):
                raise ValueError("Candidate returned no Audio")
            audio.validate()
        except InfrastructureError:
            raise
        except Exception as exc:
            return GenerationResult(
                case.id, "candidate_error", error=f"{type(exc).__name__}: {exc}"
            )
        # Storage errors are infrastructure failures, outside the candidate handler.
        return self.store.save_audio(case.id, audio)

    def record(self, result):
        self.store.record_generation(result)

    def aggregate(self, results):
        return {
            "attempted": len(results),
            "generated": sum(r.status == "ok" for r in results),
            "failed": sum(r.status == "candidate_error" for r in results),
        }

    def finish(self, summary):
        config = self.store.read_json("run.json")
        if self.component.compilation_identity() != self.compiler_identity:
            raise ValueError("Compiler changed during generation")
        if self.component.runtime_identity() != config.get("runtime"):
            raise ValueError("Runtime changed during generation")
        config.update(generation_status="complete", generation_summary=summary)
        self.store.write_json("run.json", config)


class EvaluationRunner(BenchmarkRunner):
    def __init__(self, *, load, store, evaluator, config):
        super().__init__(load=load, store=store, component=evaluator, config=config)

    stage = "evaluation_status"

    def begin(self, cases):
        config = self.store.read_json("run.json")
        if config.get("generation_status") != "complete":
            raise ValueError("Generation is incomplete")
        if config["case_ids"] != [c.id for c in cases]:
            raise ValueError("Evaluation case selection mismatch")
        if config.get("dataset") != self.config.get("dataset") or config.get(
            "dataset_version"
        ) != self.config.get("dataset_version"):
            raise ValueError("Dataset version or manifest checksum mismatch")
        if (self.store.root / "evaluation.jsonl").exists():
            raise FileExistsError("Evaluation already exists; use a separate run")
        rows = self.store.rows("generation.jsonl")
        if [r["id"] for r in rows] != config["case_ids"]:
            raise ValueError("Missing or duplicate generation records")
        self.generations = {r["id"]: GenerationResult(**r) for r in rows}
        if {"identity", "protocol_version"} & self.config["evaluation"].keys():
            raise ValueError("Evaluation identity is supplied by the prepared protocol")
        config.update(evaluation_status="running", evaluation=self.config["evaluation"])
        if "software" in self.config:
            config["evaluation_software"] = self.config["software"]
        if "services" in self.config:
            config["evaluation_services"] = self.config["services"]
        self.store.write_json("run.json", config)
        self.store.write_json("summary.json", {"status": "incomplete"})

    def prepared(self):
        self.evaluation_identity = deepcopy(self.component.identity())
        if not isinstance(self.evaluation_identity, dict):
            raise TypeError("Evaluation identity must be a mapping")
        config = self.store.read_json("run.json")
        config["evaluation"]["identity"] = self.evaluation_identity
        self.store.write_json("run.json", config)

    def execute(self, case):
        generation = self.generations[case.id]
        request = EvaluationRequest(
            case, generation, self.store.source(case), self.store.generated(generation)
        )
        result = self.component.evaluate(request)
        if not isinstance(result, EvaluationResult) or result.id != case.id:
            raise ValueError("Evaluator returned mismatched result")
        return result

    def record(self, result):
        self.store.append(
            "evaluation.jsonl", {"id": result.id, "metrics": dict(result.metrics)}
        )

    def aggregate(self, results):
        return self.component.aggregate(tuple(results))

    def finish(self, summary):
        if self.component.identity() != self.evaluation_identity:
            raise ValueError("Evaluation identity changed during execution")
        self.store.write_json(
            "summary.json", {"status": "complete", "metrics": summary}
        )
        self.mark("complete")
