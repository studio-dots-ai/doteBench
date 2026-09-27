"""FireRedTTS3-Instruct runtime backend."""

from dotebench.models.native import NativeCandidate

from .compilers.one_take import OneTakeCompiler
from .compilers.sequential import SequentialCompiler


class FireRedTTS3Instruct(NativeCandidate):
    candidate_name = "fireredtts3_instruct"

    def __init__(self, *, compiler="one_take", **kwargs):
        implementations = {
            "one_take": OneTakeCompiler,
            "sequential": SequentialCompiler,
        }
        if compiler not in implementations:
            raise ValueError("compiler must be one_take or sequential")
        super().__init__(compiler=implementations[compiler](), **kwargs)

    def is_candidate_failure(self, detail):
        return (
            isinstance(detail, dict) and detail.get("error_code") == "generation_failed"
        )
