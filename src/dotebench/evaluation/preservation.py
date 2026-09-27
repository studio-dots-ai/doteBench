"""Evaluator-owned XML preservation selection and WDTW orchestration."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Literal

from ..alignment import (
    alignment_map,
    alignment_to_segments,
    token_indices,
)
from ..instructions import parse_instruction
from ..text import tokenize_text

_PreservedOperationKind = Literal["pitch", "rate"]


def select_preserved_segments(
    instruction,
    source,
    target,
    *,
    operation_kind: _PreservedOperationKind,
):
    """Select unchanged words plus an operation's orthogonal-property scope."""
    if operation_kind not in ("pitch", "rate"):
        raise ValueError(f"Unsupported preservation operation: {operation_kind}")
    source_tokens = tokenize_text(instruction.source_text)
    target_tokens = tokenize_text(instruction.target_text)
    source_mask, target_mask = set(), set()
    for operation in instruction.operations:
        if operation.kind == operation_kind:
            continue
        source_mask.update(token_indices(instruction.source_text, operation.source))
        target_mask.update(token_indices(instruction.target_text, operation.target))
    source_map = alignment_map(instruction.source_text, source)
    target_map = alignment_map(instruction.target_text, target)
    pairs = []
    for region in instruction.regions:
        if region.kind != "equal" and region.kind != operation_kind:
            continue
        si = sorted(token_indices(instruction.source_text, region.source) - source_mask)
        ti = sorted(token_indices(instruction.target_text, region.target) - target_mask)
        if [source_tokens[i] for i in si] != [target_tokens[j] for j in ti]:
            raise ValueError("XML preservation region has inconsistent word units")
        pairs.extend(zip(si, ti, strict=True))
    pairs = list(dict.fromkeys(pairs))
    kept = [(source[source_map[i]], target[target_map[j]]) for i, j in pairs]
    kept = [(s, t) for s, t in kept if s.duration > 0]
    return [s for s, _ in kept], [t for _, t in kept], len(kept)


def evaluate_preservation(payload, measure_wdtw):
    """Select XML-owned scopes, request measurements, and attach audit intervals."""
    instruction = parse_instruction(
        payload["instruction_xml"],
        source_text=payload["source_text"],
        target_text=payload["target_text"],
    )
    source = alignment_to_segments(payload["source_alignments"])
    target = alignment_to_segments(payload["target_alignments"])
    for segment in source + target:
        if (
            not math.isfinite(segment.start)
            or not math.isfinite(segment.end)
            or segment.start < 0
            or segment.end < segment.start
        ):
            raise ValueError("Alignment requires finite 0 <= start <= end")
    ds, dt, _ = select_preserved_segments(
        instruction, source, target, operation_kind="pitch"
    )
    fs, ft, _ = select_preserved_segments(
        instruction, source, target, operation_kind="rate"
    )

    selection = {
        "duration_source": [asdict(s) for s in ds],
        "duration_target": [asdict(s) for s in dt],
        "f0_source": [asdict(s) for s in fs],
        "f0_target": [asdict(s) for s in ft],
    }
    result = measure_wdtw(
        {
            **selection,
            "source_audio": payload["source_audio"],
            "target_audio": payload["target_audio"],
        }
    )
    return {**result, "selection": selection}
