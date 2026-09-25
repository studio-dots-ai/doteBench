"""Compile benchmark XML to the instruction dialect used to train dots.tts.edit."""

import re
import xml.etree.ElementTree as ET

from dotebench.instructions import parse_instruction


def model_instruction(instruction_xml):
    """Use the same fixed pause strength for XML, NL, and expert checkpoints."""
    parse_instruction(instruction_xml)

    def pause(match):
        tag = match.group(0)
        end = "/>" if tag.endswith("/>") else ">"
        return tag[: -len(end)].rstrip() + ' level="2"' + end

    def prosody(match):
        # Read attribute strings directly: float conversion would lose lexical
        # precision (for example 1.20), changing the model's token sequence.
        node = ET.fromstring(match.group(0) + "</" + match.group(1) + ">")
        attribute = "semitones" if node.tag == "pitch" else "factor"
        return f"<{node.tag}, {attribute}={node.attrib[attribute]}>"

    native = re.sub(r"<(pitch|rate)\b[^>]*>", prosody, instruction_xml)
    return re.sub(r"<pause\b[^>]*>", pause, native)


def compile_request(request, options):
    parsed = parse_instruction(request.instruction_xml)
    return {
        "instruction": model_instruction(request.instruction_xml),
        "source_text": parsed.source_text,
        "target_text": parsed.target_text,
        **options,
    }


def compile_public_runtime_request(request, options):
    """Match the transcript tokens added by the paper's language-aware runtime.

    The public edit runtime encodes explicit transcripts verbatim and has no
    language argument. Keep this transport mapping separate from plain rendered
    transcripts returned by compile_request.
    """
    compiled = compile_request(request, options)
    prefix = {"en": "[EN]", "zh": "[ZH]"}[request.language]
    for key in ("source_text", "target_text"):
        text = compiled[key].strip()
        compiled[key] = text if not text or text.startswith(prefix) else prefix + text
    compiled["instruction"] = compiled["instruction"].strip()
    return compiled
