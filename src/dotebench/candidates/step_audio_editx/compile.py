"""Deterministic step-audio-editx model request compilation."""

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

STEP_EMOTIONS = {
    "afraid": "fear",
    "angry": "angry",
    "happy": "happy",
    "melancholic": "depressed",
    "sad": "sad",
    "surprised": "surprised",
    "calm": "remove",
    "disgusted": "disgusted",
    "excited": "excited",
}


def step_global_command(op: Operation, language: str = "en") -> str:
    p = dict(op.params)
    if language == "zh":
        if op.kind == "emo":
            return f"将整段话以{ZH_EMOTIONS[p['type']]}的情绪说出。"
        if op.kind == "rate":
            return f"将整段话的语速调整为{p['factor']}倍。"
        if op.kind == "pitch":
            v = float(p["semitones"])
            return f"将整段话的音高{'提高' if v > 0 else '降低'}{number(str(abs(v)))}个半音。"
        if op.kind == "pause":
            return f"将整段话中的停顿{'增加' if p['act'] == 'ins' else '减少'}。"
    if op.kind == "emo":
        return f"Speak the entire utterance with a {p['type']} emotion."
    if op.kind == "rate":
        return f"Adjust the speaking rate of the entire utterance to {p['factor']}x."
    if op.kind == "pitch":
        v = float(p["semitones"])
        return f"{'Increase' if v > 0 else 'Decrease'} the pitch of the entire utterance by {number(str(abs(v)))} semitones."
    if op.kind == "pause":
        return f"{'Increase' if p['act'] == 'ins' else 'Reduce'} the pauses throughout the entire utterance."
    raise ValueError(op.kind)


def compile_request(request):
    parsed = parse_instruction(request.instruction_xml)
    selected = select_instruction(parsed)
    source = parsed.target_text if selected.composition else parsed.source_text
    fields = {"max_new_tokens": 1875}
    endpoint = "/edit"
    audio_field = "audio_path"
    op = selected.operation
    if op is None:
        endpoint = "/tts"
        audio_field = "prompt_audio_path"
        fields.update(prompt_text=source, text=parsed.target_text)
    else:
        p = dict(op.params)
        if op.kind == "emo":
            fields.update(
                transcript=source,
                edit_type="emotion",
                edit_info=STEP_EMOTIONS[p["type"]],
            )
        elif op.kind == "rate":
            fields.update(
                transcript=source,
                edit_type="speed",
                edit_info="faster" if float(p["factor"]) > 1 else "slower",
            )
        elif op.kind == "pause" and p["act"] != "ins":
            if selected.composition:
                fields["instruction"] = (
                    "Remove any silent portions from the given audio while preserving "
                    "the voice content clearly. Ensure that the speech quality remains "
                    "intact with minimal distortion, and eliminate all silence from the audio.\n"
                    f" The text corresponding to the audio is: {source}"
                )
                endpoint = "/edit_freeform"
            else:
                fields.update(transcript=source, edit_type="vad", edit_info=None)
        else:
            if op.kind == "pitch":
                value = float(p["semitones"])
                instruction = f"{'Raise' if value > 0 else 'Lower'} the pitch of the entire utterance by {number(str(abs(value)))} semitones."
            else:
                instruction = step_global_command(op)
            fields["instruction"] = (
                instruction + f" The text corresponding to the audio is: {source}"
            )
            endpoint = "/edit_freeform"
    return NativeRequest(
        endpoint,
        tuple(fields.items()),
        audio_field,
        "file",
        "audio_base64",
        selected.operation_index,
        "global" if op else "text",
    )
