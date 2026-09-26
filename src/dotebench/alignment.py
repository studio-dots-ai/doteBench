"""Alignment validation and pure query extraction over fixed text units."""

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .text import normalize_word, token_spans, tokenize_text


@dataclass(frozen=True)
class WordSegment:
    word: str
    start: float
    end: float
    raw_word: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


def alignment_to_segments(
    alignments: Iterable[Mapping[str, Any]],
) -> list[WordSegment]:
    """Convert aligner response entries to normalized word-duration segments."""
    segments: list[WordSegment] = []
    for item in alignments or []:
        if item.get("is_punctuation") is True:
            raise ValueError("Alignment cannot contain punctuation-only units")
        raw_word = item["word"]
        tokens = tokenize_text(raw_word)
        if not tokens:
            raise ValueError("Alignment cannot contain empty text units")
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Alignment item missing numeric start/end: {item}"
            ) from exc
        if end < start:
            raise ValueError(f"Alignment item has end before start: {item}")
        if len(tokens) != 1:
            raise ValueError(
                "Alignment item must contain exactly one canonical text unit"
            )
        segments.append(
            WordSegment(word=tokens[0], start=start, end=end, raw_word=raw_word)
        )
    return segments


def alignment_map(text, segments):
    """Associate the complete forced transcript in order, never by subsequence."""
    tokens = tokenize_text(text)
    if tokens != [segment.word for segment in segments]:
        raise ValueError("Forced alignment units do not match the complete transcript")
    return dict(enumerate(range(len(tokens))))


def _mask_tokens_from_char_flags(text: str, char_flags: Sequence[bool]) -> list[bool]:
    if len(text) != len(char_flags):
        raise ValueError("Character edit mask does not match rendered text")
    normalized_parts: list[str] = []
    normalized_edited: list[bool] = []
    for char, is_edited in zip(text, char_flags, strict=True):
        normalized_char = unicodedata.normalize("NFKC", char)
        normalized_parts.append(normalized_char)
        normalized_edited.extend([is_edited] * len(normalized_char))
    normalized_text = "".join(normalized_parts)
    spans = token_spans(normalized_text)
    span_tokens = [normalize_word(normalized_text[start:end]) for (start, end) in spans]
    assert span_tokens == tokenize_text(text)
    token_mask = [any(normalized_edited[start:end]) for (start, end) in spans]
    for edited_index, is_edited in enumerate(normalized_edited):
        if not is_edited or normalized_text[edited_index] not in {"'", "’", "-"}:
            continue
        for token_index, (start, end) in enumerate(spans):
            if end == edited_index or start == edited_index + 1:
                token_mask[token_index] = True
    return token_mask


def token_indices(text, span):
    flags = [span.start <= i < span.end for i in range(len(text))]
    return {
        i
        for i, masked in enumerate(_mask_tokens_from_char_flags(text, flags))
        if masked
    }


def operation_pairs(instruction, operation, source, target):
    """Pair acoustic-operation units at their XML positions."""
    alignment_map(instruction.source_text, source)
    alignment_map(instruction.target_text, target)
    si = sorted(token_indices(instruction.source_text, operation.source))
    ti = sorted(token_indices(instruction.target_text, operation.target))
    if [source[i].word for i in si] != [target[i].word for i in ti]:
        raise ValueError("XML acoustic operation has inconsistent word units")
    return [(source[i], target[j]) for i, j in zip(si, ti, strict=True)]


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    return float(value)


def _field(segment, key):
    if isinstance(segment, Mapping):
        return segment[key]
    return getattr(segment, key)


def validate_alignment(text, alignment, audio_duration):
    duration = _finite(audio_duration, "audio duration")
    if duration < 0:
        raise ValueError("Negative audio duration")
    spans = token_spans(text)
    if len(spans) != len(alignment):
        raise ValueError("Alignment does not match the complete transcript")
    previous_start = previous_end = 0.0
    for (a, b), item in zip(spans, alignment, strict=True):
        word = _field(item, "word")
        if not isinstance(word, str) or normalize_word(word) != normalize_word(
            text[a:b]
        ):
            raise ValueError(
                "Alignment units do not match the complete transcript in order"
            )
        if len(token_spans(word)) != 1:
            raise ValueError("Alignment item must contain one text unit")
        start = _finite(_field(item, "start"), "alignment start")
        end = _finite(_field(item, "end"), "alignment end")
        if (
            not 0 <= start <= end <= duration
            or start < previous_start
            or end < previous_end
        ):
            raise ValueError("Invalid alignment time or order")
        previous_start, previous_end = start, end
    return spans


def query_alignment(text, alignment, queries, audio_duration):
    """Resolve tagged positions without tokenizing or aligning the audio again."""
    spans = validate_alignment(text, alignment, audio_duration)
    answers = []
    for query in queries:
        if query.count("<q>") != 1 or query.count("</q>") != 1:
            raise ValueError("Query requires exactly one tag pair")
        a, close = query.index("<q>"), query.index("</q>")
        b = close - 3
        if b < a or query[:a] + query[a + 3 : close] + query[close + 4 :] != text:
            raise ValueError("Query must recover the exact complete transcript")
        if any(x < boundary < y for x, y in spans for boundary in (a, b)):
            raise ValueError("Query boundary cuts an alignment unit")
        selected = [i for i, (x, y) in enumerate(spans) if a <= x and y <= b]
        if a == b:
            left = [i for i, (_, y) in enumerate(spans) if y <= a]
            point = _field(alignment[left[-1]], "end") if left else 0.0
            if a == len(text):
                point = audio_duration
            start = end = point
        elif selected:
            start = _field(alignment[selected[0]], "start")
            end = _field(alignment[selected[-1]], "end")
        else:
            raise ValueError("Query region contains no alignment units")
        answers.append({"start": start, "end": end})
    return answers


def query_intervals(alignment, text, operations, *, side, audio_duration):
    tokens = token_spans(text)
    queries, plans = [], []

    def query(start, end):
        queries.append(text[:start] + "<q>" + text[start:end] + "</q>" + text[end:])
        return len(queries) - 1

    for index, operation in operations:
        span = getattr(operation, side)
        if span.end > span.start:
            plans.append((index, query(span.start, span.end), None, False))
        else:
            if any(a < span.start < b for a, b in tokens):
                raise ValueError("Edit point cuts an alignment unit")
            left = [t for t in tokens if t[1] <= span.start]
            right = [t for t in tokens if t[0] >= span.start]
            plans.append(
                (
                    index,
                    query(*left[-1]) if left else None,
                    query(*right[0]) if right else None,
                    True,
                )
            )
    answers = query_alignment(text, alignment, queries, audio_duration)
    return {
        index: (
            (
                answers[left]["end"] if left is not None else 0.0,
                answers[right]["start"] if right is not None else audio_duration,
            )
            if point
            else (answers[left]["start"], answers[left]["end"])
        )
        for index, left, right, point in plans
    }


def bound_segments(segments, duration):
    """Intersect model predictions with the physical audio support once."""
    result = []
    for segment in segments:
        start = _finite(segment["start"], "alignment start")
        end = _finite(segment["end"], "alignment end")
        if end < start:
            raise ValueError("Alignment end precedes start")
        result.append(
            {
                **segment,
                "start": min(duration, max(0.0, start)),
                "end": min(duration, max(0.0, end)),
            }
        )
    return result
