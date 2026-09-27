"""Small Python interfaces for candidate-owned instruction compilation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from dotebench.domain import Audio, GenerationRequest, InfrastructureError
from dotebench.models.requests import NativeRequest


@dataclass(frozen=True)
class CompileStep:
    """One model call whose audio input is supplied by the executor."""

    case_id: str
    language: str
    operation_index: int | None
    instruction_xml: str
    source_text: str
    target_text: str
    binder: str
    requirements: tuple[object, ...] = ()


@dataclass(frozen=True)
class CompilePlan:
    steps: tuple[CompileStep, ...]


class CompilationError(InfrastructureError):
    """The selected compiler could not produce a valid executable call."""


class Requirement(ABC):
    name: str

    @abstractmethod
    def resolve(self, candidate, audio: Audio): ...

    def record(self, value):
        return {"name": self.name}


class Compiler(ABC):
    """Compile an instruction and bind each planned call to its current audio."""

    candidate_name: str
    name: str
    dependencies: tuple["Compiler", ...] = ()

    def identity(self):
        from dotebench.compilation import compiler_identity

        return compiler_identity(self.candidate_name, compiler=self.name)

    def resolve(self, name: str) -> "Compiler":
        matches = []
        visiting = set()
        visited = set()

        def visit(compiler):
            key = id(compiler)
            if key in visiting:
                raise ValueError("Compiler dependency cycle")
            if key in visited:
                return
            visiting.add(key)
            if compiler.name == name:
                matches.append(compiler)
            for dependency in compiler.dependencies:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        visit(self)
        unique = {id(compiler): compiler for compiler in matches}
        if len(unique) != 1:
            raise ValueError(f"Compiler dependency is not uniquely resolvable: {name}")
        return next(iter(unique.values()))

    @abstractmethod
    def compile(self, request: GenerationRequest) -> CompilePlan: ...

    @abstractmethod
    def bind(
        self,
        step: CompileStep,
        current_audio: Audio,
        resolved: Mapping[str, object],
    ) -> NativeRequest: ...
