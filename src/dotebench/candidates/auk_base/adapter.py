"""AuK Base candidate transport."""

from dataclasses import asdict
from pathlib import Path

from dotebench.domain import InfrastructureError
from dotebench.models.native import NativeCandidate

from .alignment import CandidateAligner
from .compilers.one_take import OneTakeCompiler
from .compilers.sequential import SequentialCompiler


class AuKBase(NativeCandidate):
    candidate_name = "auk_base"

    def __init__(
        self,
        *,
        context_mode="clean",
        aligner_url=None,
        alignment_cache_dir=None,
        compiler="one_take",
        **kwargs,
    ):
        implementations = {
            "one_take": OneTakeCompiler,
            "sequential": SequentialCompiler,
        }
        if compiler not in implementations:
            raise ValueError("compiler must be one_take or sequential")
        super().__init__(
            compiler=implementations[compiler](context_mode),
            **kwargs,
        )
        self.aligner = (
            CandidateAligner(
                aligner_url,
                alignment_cache_dir or Path(self.scratch_dir) / "candidate-alignments",
            )
            if aligner_url
            else None
        )

    def invoke(self, compiled, audio):
        if compiled.transport == "identity":
            if not self.ready:
                raise InfrastructureError("Native candidate has not been prepared")
            self.last_call_metadata = {
                "native_request": asdict(compiled),
                "scope": "identity",
                "response_metadata": dict(compiled.fields)["metadata"],
            }
            return audio
        return super().invoke(compiled, audio)

    def is_candidate_failure(self, detail):
        return (
            isinstance(detail, dict) and detail.get("error_code") == "candidate_error"
        ) or super().is_candidate_failure(detail)

    def resolve_alignment(self, audio, text, language):
        if self.aligner is None:
            raise InfrastructureError(
                "AuK Base local editing requires its own aligner_url"
            )
        return self.aligner.align(audio, text, language)

    def close(self):
        if self.aligner is not None:
            self.aligner.close()
        super().close()
