"""One source-ordered edit using FireRedTTS3-Instruct's native interfaces."""

import json
from decimal import Decimal

from dotebench.instructions import Span, parse_instruction, render_replacement
from dotebench.models.requests import NativeRequest
from dotebench.text import token_spans


def quote(text):
    return "'" + text + "'" if "'" not in text else json.dumps(text, ensure_ascii=False)


def occurrences(text, phrase):
    # Count overlaps too: str.count alone would misclassify repeated Chinese text.
    return sum(text.startswith(phrase, i) for i in range(len(text))) if phrase else 0


def unique_context(text, start, end):
    """Smallest lexical context enclosing a single edit, irrespective of other tags."""
    units = token_spans(text)
    left = sorted({0, start, *(a for a, _ in units if a <= start)})
    right = sorted({len(text), end, *(b for _, b in units if b >= end)})
    candidates = sorted(
        ((a, b) for a in left for b in right if a < b),
        key=lambda pair: (pair[1] - pair[0], pair[0]),
    )
    for a, b in candidates:
        if occurrences(text, text[a:b]) == 1:
            return a, b
    raise ValueError("Cannot identify an edit in an empty source transcript")


def semantic_instruction(text, op):
    start, end = op.source.start, op.source.end
    original = text[start:end]
    replacement = (
        dict(op.params)["targ"]
        if op.kind == "sub"
        else op.content
        if op.kind == "ins"
        else ""
    )
    if op.kind in {"sub", "del"} and occurrences(text, original) == 1:
        command = (
            f"Replace {quote(original)} with {quote(replacement)}."
            if op.kind == "sub"
            else f"delete {quote(original)}"
        )
        return command, (start, end), replacement
    if op.kind == "ins":
        units = token_spans(text)
        # Start exactly at the insertion point (ignoring adjacent whitespace).
        rstart = start + len(text[start:]) - len(text[start:].lstrip())
        lend = len(text[:start].rstrip())
        for a, b in units:
            if a == rstart and occurrences(text, text[a:b]) == 1:
                return (
                    f"insert {quote(replacement)} before the character or word {quote(text[a:b])}",
                    (start, end),
                    replacement,
                )
        for a, b in reversed(units):
            if b == lend and occurrences(text, text[a:b]) == 1:
                return (
                    f"insert {quote(replacement)} after the character or word {quote(text[a:b])}",
                    (start, end),
                    replacement,
                )
    # A single contextual replacement preserves every unselected source character.
    a, b = unique_context(text, start, end)
    changed = render_replacement(text[a:b], Span(start - a, end - a), replacement)
    return f"Replace {quote(text[a:b])} with {quote(changed)}.", (a, b), changed


def mapped_rate(value):
    value = Decimal(value)
    if value == 1:
        return Decimal("1.0")
    grid = [
        Decimal(n) / 10 for n in range(5, 21) if (Decimal(n) / 10 - 1) * (value - 1) > 0
    ]
    return min(grid, key=lambda x: (abs(x - value), abs(x - 1)))


def describe_request(request):
    parsed = parse_instruction(request.instruction_xml)
    index, op = min(
        enumerate(parsed.operations),
        key=lambda x: (x[1].source.start, x[0]),
    )
    params = dict(op.params)
    text = parsed.source_text
    metadata = {
        "selected_operation_index": index,
        "kind": op.kind,
        "source_span": [op.source.start, op.source.end],
        "original_params": params,
    }
    if op.kind in {"ins", "del", "sub"}:
        instruction, span, replacement = semantic_instruction(text, op)
        endpoint, scope = "/semantic_edit", "text"
        metadata.update(instruction_span=list(span), replacement=replacement)
    else:
        endpoint, scope = "/acoustic_edit", "global"
        if op.kind == "rate":
            actual = format(mapped_rate(params["factor"]), ".1f")
            instruction = f"adjust the speed to {actual}x"
            metadata["actual_params"] = {"factor": actual}
        elif op.kind == "pitch":
            value = Decimal(params["semitones"])
            grid = [n for n in range(-6, 7) if n and n * value > 0]
            actual = min(grid, key=lambda n: (abs(Decimal(n) - value), abs(n)))
            instruction = (
                f"shift the pitch by {actual} step{'s' if abs(actual) != 1 else ''}"
            )
            metadata["actual_params"] = {"semitones": actual}
        elif op.kind == "emo":
            instruction = f"adjust the emotion to {params['type']}"
        elif op.kind == "pause":
            instruction = (
                "increase the pauses" if params["act"] == "ins" else "reduce the pauses"
            )
        else:
            raise ValueError(f"Unsupported operation: {op.kind}")
    metadata.update(
        endpoint=endpoint,
        scope=scope,
        instruction=instruction,
        template_support="extension" if op.kind in {"emo", "pause"} else "official",
    )
    return metadata


def compile_request(request):
    plan = describe_request(request)
    return NativeRequest(
        plan["endpoint"],
        (
            ("case_id", request.id),
            ("instruction", plan["instruction"]),
            ("seed", 1234),
            ("n_timesteps", 10),
            ("inference_cfg", 1.2),
        ),
        "audio_path",
        "file",
        "audio_base64",
        plan["selected_operation_index"],
        plan["scope"],
    )
