from dotebench.compilers.one_take import OneTakeCompiler as BaseOneTakeCompiler

from ..compile import compile_request

DEPENDENCIES = ()


class OneTakeCompiler(BaseOneTakeCompiler):
    candidate_name = "ming_uniaudio"

    def bind_request(self, request, resolved):
        return compile_request(request)
