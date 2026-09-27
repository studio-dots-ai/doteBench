"""Canonical instruction rendering and text replacement boundaries."""

import re

from .types import Span


def render_replacement(text: str, span: Span, replacement: str) -> str:
    """Replace one Unicode character span using canonical text-edit spacing.

    An empty span inserts text; an empty replacement deletes the span. For a
    contextual edit, pass the context text and a span relative to that context.
    The result follows text-edit whitespace rules even for an empty replacement.
    """
    if not 0 <= span.start <= span.end <= len(text):
        raise ValueError("Replacement span is outside the transcript")
    return _normalize_parts([text[: span.start], replacement, text[span.end :]])


def _render_parts(parts):
    """Render validated XML parts without losing their text-edit boundaries."""
    source = "".join(p[1] for p in parts)
    groups = [""]
    text_edit = False
    for kind, _, dst, _ in parts:
        if kind in {"ins", "del", "sub"}:
            text_edit = True
            groups.extend([dst, ""])
        else:
            groups[-1] += dst
    target = _normalize_parts(groups) if text_edit else "".join(p[2] for p in parts)
    return source, target


def _normalize_parts(parts):
    parts = [p.strip() for p in parts]
    rendered = ""
    ascii_word = lambda s: bool(s and re.fullmatch("[A-Za-z0-9]", s))
    suffix = lambda s: bool(re.match(r"(?:s|t|re|ve|ll|d|m)\b", s, re.IGNORECASE))
    for i, part in enumerate(parts):
        if not part:
            continue
        following = next((p for p in parts[i + 1 :] if p), "")
        connector = bool(
            rendered
            and following
            and ascii_word(rendered[-1])
            and (
                (part == "-" and ascii_word(following[0]))
                or (part in {"'", "’"} and (suffix(following) or rendered[-1] in "sS"))
            )
        )
        internal = bool(
            rendered
            and (
                (
                    part[0] in {"'", "’", "-"}
                    and len(part) > 1
                    and ascii_word(rendered[-1])
                    and ascii_word(part[1])
                )
                or (
                    rendered[-1] in {"'", "’", "-"}
                    and len(rendered) > 1
                    and ascii_word(rendered[-2])
                    and ascii_word(part[0])
                    and (rendered[-1] == "-" or suffix(part))
                )
            )
        )
        if (
            rendered
            and not connector
            and not internal
            and (ascii_word(rendered[-1]) or ascii_word(part[0]))
        ):
            rendered += " "
        rendered += part
    return re.sub(
        r"\s+([,.;:!?，。？！；：])", r"\1", re.sub(r"\s+", " ", rendered).strip()
    )
