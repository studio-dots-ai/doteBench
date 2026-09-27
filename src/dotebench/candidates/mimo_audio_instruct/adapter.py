"""MiMo-Audio-Instruct runtime backend."""

from dotebench.models.native import NativeCandidate

from .compilers.one_take import OneTakeCompiler
from .compilers.sequential import SequentialCompiler


class MiMoAudioInstruct(NativeCandidate):
    candidate_name = "mimo_audio_instruct"

    def __init__(self, *, compiler="one_take", **kwargs):
        implementations = {
            "one_take": OneTakeCompiler,
            "sequential": SequentialCompiler,
        }
        if compiler not in implementations:
            raise ValueError("compiler must be one_take or sequential")
        super().__init__(compiler=implementations[compiler](), **kwargs)
