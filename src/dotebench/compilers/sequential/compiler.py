"""Shared sequential compiler composed from a candidate's one_take compiler."""

from dataclasses import replace

from dotebench.domain import GenerationRequest

from ..base import CompilePlan, Compiler
from .planner import plan_operations


class SequentialCompiler(Compiler):
    name = "sequential"

    def __init__(self, one_take: Compiler):
        if one_take.name != "one_take":
            raise ValueError("SequentialCompiler requires a one_take compiler")
        self.one_take = one_take
        self.candidate_name = one_take.candidate_name
        self.dependencies = (one_take,)

    def compile(self, request: GenerationRequest) -> CompilePlan:
        steps = []
        for edit in plan_operations(request.instruction_xml):
            single = replace(request, instruction_xml=edit.instruction_xml)
            compiled = self.one_take.compile(single)
            if len(compiled.steps) != 1:
                raise ValueError("one_take compiler must produce exactly one step")
            steps.append(
                replace(
                    compiled.steps[0],
                    operation_index=edit.operation_index,
                    source_text=edit.source_text,
                    target_text=edit.target_text,
                )
            )
        return CompilePlan(tuple(steps))

    def bind(self, step, current_audio, resolved):
        return self.one_take.bind(step, current_audio, resolved)
