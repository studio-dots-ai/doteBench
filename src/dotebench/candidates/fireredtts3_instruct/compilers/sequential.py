from dotebench.compilers.sequential import SequentialCompiler as BaseSequentialCompiler

from .one_take import OneTakeCompiler

DEPENDENCIES = ("one_take",)


class SequentialCompiler(BaseSequentialCompiler):
    candidate_name = "fireredtts3_instruct"

    def __init__(self):
        super().__init__(OneTakeCompiler())
