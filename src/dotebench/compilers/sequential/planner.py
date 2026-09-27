"""Single-tag editing plans with XML-owned transcript coordinates."""

from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

from dotebench.instructions import parse_instruction

TEXT_KINDS = {"sub", "ins", "del"}


@dataclass(frozen=True)
class EditStep:
    operation_index: int
    instruction_xml: str
    source_text: str
    target_text: str


def _fragment(text, start, end, op):
    attrs = "".join(f" {key}={quoteattr(value)}" for key, value in op.params)
    if op.kind == "pause":
        return escape(text[:start]) + f"<pause{attrs}/>" + escape(text[end:])
    body = op.content if op.kind == "ins" else text[start:end]
    return (
        escape(text[:start])
        + f"<{op.kind}{attrs}>"
        + escape(body)
        + f"</{op.kind}>"
        + escape(text[end:])
    )


def _nonspace(text):
    return tuple(ch for ch in text if not ch.isspace())


def _owned_span(text, owners, region_index):
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    selected = [p for p, owner in zip(positions, owners) if owner == region_index]
    if selected:
        return selected[0], selected[-1] + 1
    anchor = next(
        (p for p, owner in zip(positions, owners) if owner > region_index),
        len(text),
    )
    return anchor, anchor


def plan_operations(instruction_xml: str) -> tuple[EditStep, ...]:
    """Execute text tags first, then acoustic tags, preserving each XML order.

    Character ownership survives whitespace rendering, including repeated text.
    Acoustic spans use the original target coordinates when rendering is exact,
    and XML ownership when sequential rendering only changes whitespace.
    """
    instruction = parse_instruction(instruction_xml)
    if not instruction.operations:
        raise ValueError("Sequential editing requires at least one XML operation")
    if len(instruction.operations) == 1:
        return (
            EditStep(
                0, instruction_xml, instruction.source_text, instruction.target_text
            ),
        )
    text = instruction.source_text
    owners = [
        index
        for index, region in enumerate(instruction.regions)
        for ch in text[region.source.start : region.source.end]
        if not ch.isspace()
    ]
    op_regions = [i for i, r in enumerate(instruction.regions) if r.kind != "equal"]
    steps = []
    for index, op in enumerate(instruction.operations):
        if op.kind not in TEXT_KINDS:
            continue
        region_index = op_regions[index]
        positions = [i for i, ch in enumerate(text) if not ch.isspace()]
        if not steps:
            start, end = op.source.start, op.source.end
        else:
            start, end = _owned_span(text, owners, region_index)
        fragment = _fragment(text, start, end, op)
        parsed = parse_instruction(fragment)
        steps.append(EditStep(index, fragment, text, parsed.target_text))
        replacement = (
            dict(op.params)["targ"]
            if op.kind == "sub"
            else op.content
            if op.kind == "ins"
            else ""
        )
        owners = (
            [owner for p, owner in zip(positions, owners) if p < start]
            + [region_index for ch in replacement if not ch.isspace()]
            + [owner for p, owner in zip(positions, owners) if p >= end]
        )
        text = parsed.target_text
        if len(owners) != sum(not ch.isspace() for ch in text):
            raise ValueError("Sequential rendering lost XML ownership")
    exact_target = text == instruction.target_text
    if not exact_target and _nonspace(text) != _nonspace(instruction.target_text):
        raise ValueError("Sequential target differs from the original XML target")
    for index, op in enumerate(instruction.operations):
        if op.kind in TEXT_KINDS:
            continue
        if exact_target:
            start, end = op.target.start, op.target.end
        else:
            start, end = _owned_span(text, owners, op_regions[index])
        fragment = _fragment(text, start, end, op)
        parsed = parse_instruction(fragment)
        steps.append(EditStep(index, fragment, text, parsed.target_text))
    return tuple(steps)
