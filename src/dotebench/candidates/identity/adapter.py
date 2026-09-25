"""Identity candidate for validating the complete generation workflow."""

from dotebench.models.base import Audio, CandidateModel, GenerationRequest

from .compilers.one_take import compile_request


class Identity(CandidateModel):
    def compilation_identity(self):
        from dotebench.compilation import compiler_identity

        return compiler_identity("identity")

    def prepare(self) -> None:
        pass

    def generate(self, request: GenerationRequest) -> Audio:
        return compile_request(request)

    def close(self) -> None:
        pass
