from dotebench.compilers.sequential import SequentialCompiler as BaseSequentialCompiler

from .one_take import OneTakeCompiler

DEPENDENCIES = ("one_take",)


class SequentialCompiler(BaseSequentialCompiler):
    candidate_name = "auk_base"

    def __init__(self, context_mode="clean"):
        super().__init__(OneTakeCompiler(context_mode))
