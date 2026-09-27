from dotebench.compilers.one_take import OneTakeCompiler as BaseOneTakeCompiler

from ..compile import compile_request

DEPENDENCIES = ()


class OneTakeCompiler(BaseOneTakeCompiler):
    candidate_name = "step_audio_editx"

    def bind_request(self, request, resolved):
        return compile_request(request)
