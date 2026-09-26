"""Deterministic mimo-audio-instruct model request compilation."""

from dotebench.candidates.common.selection import (
    number,
    select_instruction,
)
from dotebench.instructions import Operation, parse_instruction
from dotebench.models.requests import NativeRequest

ZH_EMOTIONS = {
    "afraid": "害怕",
    "angry": "生气",
    "happy": "开心",
    "melancholic": "忧郁",
    "sad": "悲伤",
    "surprised": "惊讶",
    "calm": "平静",
    "disgusted": "厌恶",
    "excited": "兴奋",
}


def global_command(op: Operation, language: str) -> str:
    p = dict(op.params)
    if language == "zh":
        if op.kind == "emo":
            return f"用{ZH_EMOTIONS[p['type']]}的语气说。"
        if op.kind == "rate":
            return f"用原来{number(p['factor'])}倍的语速说。"
        if op.kind == "pitch":
            value = float(p["semitones"])
            return f"将整段话的音高{'提高' if value > 0 else '降低'}{number(str(abs(value)))}个半音。"
        if op.kind == "pause":
            return f"{'增加' if p['act'] == 'ins' else '减少'}整段话中的停顿。"
    else:
        if op.kind == "emo":
            return f"Speak in a {p['type']} voice."
        if op.kind == "rate":
            return f"Speak at {number(p['factor'])} times the original speaking rate."
        if op.kind == "pitch":
            value = float(p["semitones"])
            return f"Speak with the pitch {'raised' if value > 0 else 'lowered'} by {number(str(abs(value)))} semitones."
        if op.kind == "pause":
            return f"Speak with {'longer' if p['act'] == 'ins' else 'shorter'} pauses throughout."
    raise ValueError(f"Unsupported acoustic operation: {op.kind}")


def compile_request(request):
    parsed = parse_instruction(request.instruction_xml)
    selected = select_instruction(parsed)
    fields = {
        "text": parsed.target_text,
        "instruct": global_command(selected.operation, request.language)
        if selected.operation
        else "",
        "read_text_only": True,
        "max_new_tokens": 282,
        "seed": 42,
        "output_format": "wav",
    }
    return NativeRequest(
        "/tts",
        tuple(fields.items()),
        "prompt_speech",
        "base64",
        "wav",
        selected.operation_index,
        "global" if selected.operation else "text",
    )
