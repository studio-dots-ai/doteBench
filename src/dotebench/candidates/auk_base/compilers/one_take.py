from dataclasses import dataclass, replace

from dotebench.compilers.base import Requirement
from dotebench.compilers.one_take import OneTakeCompiler as BaseOneTakeCompiler

from ..compile import compile_request

DEPENDENCIES = ()


@dataclass(frozen=True)
class AlignmentRequirement(Requirement):
    text: str
    language: str
    name: str = "candidate_alignment"

    def resolve(self, candidate, audio):
        return candidate.resolve_alignment(audio, self.text, self.language)

    def record(self, value):
        _alignment, provenance = value
        return {"name": self.name, "provenance": provenance}


class OneTakeCompiler(BaseOneTakeCompiler):
    candidate_name = "auk_base"

    def __init__(self, context_mode="clean"):
        if context_mode not in {"clean", "bridge"}:
            raise ValueError("Expected clean or bridge context mode")
        self.context_mode = context_mode

    @staticmethod
    def _whole(parsed):
        _, op = min(
            enumerate(parsed.operations),
            key=lambda pair: (pair[1].source.start, pair[0]),
        )
        outside = (
            parsed.source_text[: op.source.start] + parsed.source_text[op.source.end :]
        )
        return op.kind in {"sub", "ins", "del"} or (
            op.kind in {"emo", "rate", "pitch"}
            and not any(ch.isalnum() for ch in outside)
        )

    def requirements(self, request, parsed):
        if self._whole(parsed):
            return ()
        return (AlignmentRequirement(parsed.source_text, request.language),)

    def bind_request(self, request, resolved):
        alignment = provenance = None
        if "candidate_alignment" in resolved:
            alignment, provenance = resolved["candidate_alignment"]
        compiled = compile_request(request, self.context_mode, alignment=alignment)
        if provenance is None:
            return compiled
        fields = dict(compiled.fields)
        fields["metadata"] = {
            **fields["metadata"],
            "candidate_alignment": provenance,
        }
        return replace(compiled, fields=tuple(fields.items()))
