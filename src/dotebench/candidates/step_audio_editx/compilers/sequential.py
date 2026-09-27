from dotebench.compilers.sequential import SequentialCompiler as BaseSequentialCompiler

from .one_take import OneTakeCompiler

DEPENDENCIES = ("one_take",)


class SequentialCompiler(BaseSequentialCompiler):
    candidate_name = "step_audio_editx"

    def __init__(self):
        super().__init__(OneTakeCompiler())
