"""Deterministic ming-uniaudio model request compilation."""

from dotebench.candidates.common.selection import (
    TEXT_KINDS,
    SelectedInstruction,
    exact_target,
    number,
    select_instruction,
)
from dotebench.instructions import Operation, parse_instruction
from dotebench.models.requests import NativeRequest

MING_PREFIX = "Please recognize the language of this speech and transcribe it. And "


def ming_global_command(op: Operation) -> str:
    p = dict(op.params)
    if op.kind == "emo":
        return f"change the emotion to {p['type']} mood."
    if op.kind == "pitch":
        return f"shifts the pitch by {number(p['semitones'])} steps."
    if op.kind == "rate":
        return f"adjusts the speed to {number(p['factor'])}."
    if op.kind == "pause":
        return f"{'increase' if p['act'] == 'ins' else 'reduce'} the pauses throughout the entire utterance."
    raise ValueError(f"Unsupported acoustic operation: {op.kind}")


def ming_instruction(selected: SelectedInstruction, language: str) -> str:
    instruction = selected.instruction
    if selected.operation is None:
        clauses = []
        for op in instruction.operations:
            p = dict(op.params)
            source = op.content
            if op.kind == "sub":
                clauses.append(f"substitute '{source}' with '{p['targ']}'")
            elif op.kind == "ins":
                clauses.append(f"insert '{op.content}'")
            elif op.kind == "del":
                clauses.append(f"delete '{source}'")
        return (
            MING_PREFIX
            + f"edit the speech so the final transcript is exactly: '{instruction.target_text}'. Specifically, {'; '.join(clauses)}."
        )
    command = ming_global_command(selected.operation)
    if selected.composition:
        command = exact_target(instruction.target_text, language) + " " + command
    elif selected.operation.kind == "pause":
        command += f" Keep the spoken content exactly: '{instruction.target_text}'."
    return MING_PREFIX + command


def compile_request(request):
    parsed = parse_instruction(request.instruction_xml)
    selected = select_instruction(parsed)
    fields = {
        "instruction": ming_instruction(selected, request.language),
        "seed": 1895,
        "use_cot": bool(parsed.operations)
        and all(op.kind in TEXT_KINDS for op in parsed.operations),
        "max_audio_seconds": 45.0,
    }
    return NativeRequest(
        "/edit",
        tuple(fields.items()),
        "audio_path",
        "file",
        "audio_base64",
        selected.operation_index,
        "global" if selected.operation else "text",
    )
