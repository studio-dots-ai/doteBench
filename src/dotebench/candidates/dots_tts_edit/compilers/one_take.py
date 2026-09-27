from dotebench.compilers.one_take import OneTakeCompiler as BaseOneTakeCompiler
from dotebench.instructions import parse_instruction
from dotebench.models.requests import NativeRequest

from ..compile import compile_public_runtime_request

DEPENDENCIES = ()


class OneTakeCompiler(BaseOneTakeCompiler):
    candidate_name = "dots_tts_edit"

    def __init__(self, generation_options=None):
        self.generation_options = dict(generation_options or {})

    def bind_request(self, request, resolved):
        fields = compile_public_runtime_request(request, self.generation_options)
        kinds = {
            op.kind for op in parse_instruction(request.instruction_xml).operations
        }
        return NativeRequest(
            "generate_edit",
            tuple(fields.items()),
            "source_audio_path",
            "file",
            "dots_result",
            None,
            "emotion" if kinds == {"emo"} else "xml",
            transport="dots_sdk",
        )
