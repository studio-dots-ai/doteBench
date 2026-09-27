"""Public XML instruction semantics, rendering and coordinate types."""

from .parser import parse_instruction
from .rendering import render_replacement
from .types import InstructionRegion, Operation, ParsedInstruction, Span

__all__ = [
    "InstructionRegion",
    "Operation",
    "ParsedInstruction",
    "Span",
    "parse_instruction",
    "render_replacement",
    "render_source",
    "render_target",
]


def render_source(instruction_xml: str) -> str:
    """Validate an XML fragment and return its exact source transcript."""
    return parse_instruction(instruction_xml).source_text


def render_target(instruction_xml: str) -> str:
    """Validate an XML fragment and return its canonical target transcript.

    Text edits normalize word boundaries and punctuation. Acoustic-only
    instructions preserve the original transcript whitespace.
    """
    return parse_instruction(instruction_xml).target_text
