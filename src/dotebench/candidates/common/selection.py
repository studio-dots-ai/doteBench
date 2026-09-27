"""Shared XML operation ordering and selection."""

from dataclasses import dataclass

from dotebench.instructions import Operation, ParsedInstruction

TEXT_KINDS = frozenset({"ins", "del", "sub"})


@dataclass(frozen=True)
class SelectedInstruction:
    instruction: ParsedInstruction
    operation_index: int | None
    composition: bool

    @property
    def operation(self) -> Operation | None:
        return (
            None
            if self.operation_index is None
            else self.instruction.operations[self.operation_index]
        )

    @property
    def has_text_edits(self):
        return any(op.kind in TEXT_KINDS for op in self.instruction.operations)


def select_instruction(instruction: ParsedInstruction) -> SelectedInstruction:
    ordered = sorted(
        enumerate(instruction.operations),
        key=lambda item: (item[1].source.start, item[0]),
    )
    first = next((index for index, op in ordered if op.kind not in TEXT_KINDS), None)
    families = {
        "text"
        if op.kind in TEXT_KINDS
        else "prosody"
        if op.kind in {"pitch", "rate"}
        else op.kind
        for op in instruction.operations
    }
    return SelectedInstruction(instruction, first, len(families) > 1)


def number(value: str) -> str:
    from decimal import Decimal

    return format(Decimal(value).normalize(), "f")


def exact_target(text: str, language: str) -> str:
    return (
        f"请完整说出以下文本：“{text}”。"
        if language == "zh"
        else f'Output one spoken utterance saying exactly: "{text}".'
    )
