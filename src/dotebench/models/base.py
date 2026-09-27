"""Candidate runtime contracts."""

from abc import ABC, abstractmethod
from pathlib import Path

from dotebench.domain import Audio, GenerationRequest, GenerationResult

__all__ = [
    "Audio",
    "CandidateModel",
    "CompiledCandidate",
    "GenerationRequest",
    "GenerationResult",
]


class CandidateModel(ABC):
    def compilation_identity(self):
        return None

    def runtime_identity(self):
        return None

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def generate(self, request: GenerationRequest) -> Audio: ...

    @abstractmethod
    def close(self) -> None: ...


class CompiledCandidate(CandidateModel):
    """A one-call backend driven by a separately selected compiler."""

    candidate_name: str

    def __init__(self, *, compiler, trace_dir):
        if compiler.candidate_name != self.candidate_name:
            raise ValueError("Compiler and candidate do not match")
        self.compiler = compiler
        self.trace_dir = Path(trace_dir)
        self.last_call_metadata = {}

    def compilation_identity(self):
        return self.compiler.identity()

    def generate(self, request: GenerationRequest) -> Audio:
        from .executor import execute_plan

        return execute_plan(self, self.compiler, request)

    @abstractmethod
    def invoke(self, request, audio: Audio) -> Audio: ...
