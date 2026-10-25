from dataclasses import dataclass

from dotebench.compilers.base import Requirement
from dotebench.compilers.one_take import OneTakeCompiler as BaseOneTakeCompiler

from ..compile import compile_request


@dataclass(frozen=True)
class PEInstructionRequirement(Requirement):
    instruction_xml: str
    source_text: str
    language: str
    name: str = "pe_instruction"

    def resolve(self, candidate, audio):
        return candidate.resolve_instruction(
            {
                "instruction_xml": self.instruction_xml,
                "source_text": self.source_text,
                "language": self.language,
            }
        )

    def record(self, value):
        _prepared, provenance = value
        return {"name": self.name, "provenance": provenance}


class OneTakeCompiler(BaseOneTakeCompiler):
    candidate_name = "auk_base_pe"

    def requirements(self, request, parsed):
        return (
            PEInstructionRequirement(
                request.instruction_xml, parsed.source_text, request.language
            ),
        )

    def bind_request(self, request, resolved):
        return compile_request(request, *resolved["pe_instruction"])
