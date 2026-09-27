from dotebench.compilers.sequential import SequentialCompiler as BaseSequentialCompiler

from .one_take import OneTakeCompiler

DEPENDENCIES = ("one_take",)


class SequentialCompiler(BaseSequentialCompiler):
    candidate_name = "dots_tts_edit"

    def __init__(self, generation_options=None):
        super().__init__(OneTakeCompiler(generation_options))
