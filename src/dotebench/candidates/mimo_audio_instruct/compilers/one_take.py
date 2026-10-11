from dotebench.compilers.one_take import OneTakeCompiler as BaseOneTakeCompiler

from ..compile import compile_request


class OneTakeCompiler(BaseOneTakeCompiler):
    candidate_name = "mimo_audio_instruct"

    def bind_request(self, request, resolved):
        return compile_request(request)
