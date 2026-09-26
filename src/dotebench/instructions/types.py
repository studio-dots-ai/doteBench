"""Immutable instruction records with Unicode character coordinates."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Operation:
    kind: Literal["ins", "del", "sub", "emo", "pitch", "rate", "pause"]
    source: Span
    target: Span
    params: tuple[tuple[str, str], ...] = ()
    content: str = ""


@dataclass(frozen=True)
class InstructionRegion:
    """A paired XML node, including unchanged text between operations."""

    kind: str
    source: Span
    target: Span


@dataclass(frozen=True)
class ParsedInstruction:
    source_text: str
    target_text: str
    operations: tuple[Operation, ...]
    regions: tuple[InstructionRegion, ...] = ()
