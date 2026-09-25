"""Shared compilation of a complete instruction into one model call."""

from abc import abstractmethod
from collections.abc import Mapping

from dotebench.domain import GenerationRequest
from dotebench.instructions import parse_instruction
from dotebench.models.requests import NativeRequest

from .base import CompilePlan, Compiler, CompileStep


class OneTakeCompiler(Compiler):
    """Base for a candidate's sole implementation of one model call."""

    name = "one_take"

    def compile(self, request: GenerationRequest) -> CompilePlan:
        parsed = parse_instruction(request.instruction_xml)
        return CompilePlan(
            (
                CompileStep(
                    case_id=request.id,
                    language=request.language,
                    operation_index=None,
                    instruction_xml=request.instruction_xml,
                    source_text=parsed.source_text,
                    target_text=parsed.target_text,
                    binder=self.name,
                    requirements=self.requirements(request, parsed),
                ),
            )
        )

    def requirements(self, request, parsed) -> tuple[object, ...]:
        return ()

    def bind(self, step, current_audio, resolved):
        request = GenerationRequest(
            step.case_id,
            step.language,
            current_audio,
            step.instruction_xml,
        )
        return self.bind_request(request, resolved)

    @abstractmethod
    def bind_request(
        self, request: GenerationRequest, resolved: Mapping[str, object]
    ) -> NativeRequest: ...
