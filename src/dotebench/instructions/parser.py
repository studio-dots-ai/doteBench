"""Strict XML editing instructions with Unicode character coordinates."""

import math
import xml.etree.ElementTree as ET

from .rendering import _render_parts
from .types import InstructionRegion, Operation, ParsedInstruction, Span

ATTRS = {
    "ins": set(),
    "del": set(),
    "sub": {"targ"},
    "emo": {"type", "level"},
    "pitch": {"semitones"},
    "rate": {"factor"},
    "pause": {"act"},
}
EMOTIONS = {
    "angry",
    "happy",
    "sad",
    "afraid",
    "disgusted",
    "surprised",
    "calm",
    "melancholic",
    "neutral",
}


def parse_instruction(
    instruction_xml: str,
    *,
    source_text: str | None = None,
    target_text: str | None = None,
) -> ParsedInstruction:
    """Parse a fragment; edits cannot nest. Insert/delete/pause spans may be empty.

    Text edits use canonical word-boundary whitespace rendering. Instructions
    containing only acoustic edits preserve transcript whitespace exactly.
    """
    if (
        not isinstance(instruction_xml, str)
        or not instruction_xml
        or "<!" in instruction_xml
        or "<?" in instruction_xml
    ):
        raise ValueError("Expected an XML instruction fragment without declarations")
    try:
        root = ET.fromstring("<instruction>" + instruction_xml + "</instruction>")
    except ET.ParseError as exc:
        raise ValueError(f"Invalid instruction XML: {exc}") from exc
    parts = []
    if root.text:
        parts.append(("equal", root.text, root.text, ()))
    for node in root:
        kind, attrs, body = node.tag, node.attrib, node.text or ""
        if kind not in ATTRS or set(attrs) != ATTRS[kind] or len(node):
            raise ValueError(f"Unsupported tag, attributes, or nesting: {kind}")
        if kind == "pause":
            if body or attrs["act"] not in {"ins", "red"}:
                raise ValueError("Invalid pause instruction")
        elif not body:
            raise ValueError(f"Empty {kind} region")
        if kind == "sub" and not attrs["targ"]:
            raise ValueError("Empty substitution target")
        if kind == "emo" and (
            attrs["type"] not in EMOTIONS or attrs["level"] not in {"1", "2", "3"}
        ):
            raise ValueError("Invalid emotion parameters")
        if kind in {"pitch", "rate"}:
            value = float(attrs["semitones" if kind == "pitch" else "factor"])
            if not math.isfinite(value) or (kind == "rate" and value <= 0):
                raise ValueError("Invalid acoustic parameter")
        source = "" if kind in {"ins", "pause"} else body
        target = (
            "" if kind in {"del", "pause"} else attrs["targ"] if kind == "sub" else body
        )
        parts.append((kind, source, target, tuple(sorted(attrs.items()))))
        if node.tail:
            parts.append(("equal", node.tail, node.tail, ()))
    source, target = _render_parts(parts)
    text_edit = any(p[0] in {"ins", "del", "sub"} for p in parts)
    if source_text is not None and source != source_text:
        raise ValueError("Instruction source transcript mismatch")
    if target_text is not None and target != target_text:
        raise ValueError("Instruction target transcript mismatch")
    # Rendering only changes whitespace. Carry XML-node ownership through that
    # transformation instead of searching rendered text for repeated strings.
    owners = [
        (ch, index)
        for index, part in enumerate(parts)
        for ch in part[2]
        if not ch.isspace()
    ]
    positions = [[] for _ in parts]
    cursor = 0
    for offset, ch in enumerate(target):
        if ch.isspace():
            continue
        if cursor >= len(owners) or owners[cursor][0] != ch:
            raise ValueError("Target rendering changed XML content")
        positions[owners[cursor][1]].append(offset)
        cursor += 1
    if cursor != len(owners):
        raise ValueError("Target rendering lost XML content")
    operations = []
    regions = []
    spos = tpos = 0
    for index, (kind, src, dst, attrs) in enumerate(parts):
        if text_edit:
            start = positions[index][0] if positions[index] else tpos
            end = positions[index][-1] + 1 if positions[index] else tpos
        else:
            start, end = tpos, tpos + len(dst)
        regions.append(
            InstructionRegion(kind, Span(spos, spos + len(src)), Span(start, end))
        )
        if kind != "equal":
            operations.append(
                Operation(
                    kind,
                    Span(spos, spos + len(src)),
                    Span(start, end),
                    attrs,
                    dst if kind == "ins" else src,
                )
            )
        spos += len(src)
        tpos = end
    return ParsedInstruction(source, target, tuple(operations), tuple(regions))
