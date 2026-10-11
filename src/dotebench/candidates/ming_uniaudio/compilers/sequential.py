from dotebench.compilers.sequential import SequentialCompiler as BaseSequentialCompiler

from .one_take import OneTakeCompiler


class SequentialCompiler(BaseSequentialCompiler):
    candidate_name = "ming_uniaudio"

    def __init__(self):
        super().__init__(OneTakeCompiler())
