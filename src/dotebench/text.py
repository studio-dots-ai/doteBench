"""Canonical alignment text units and their positions in the transcript."""

import unicodedata
from typing import Any, List, Optional, Tuple


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        13312 <= code <= 19903
        or 19968 <= code <= 40959
        or 63744 <= code <= 64255
        or (131072 <= code <= 173791)
        or (173824 <= code <= 177983)
        or (177984 <= code <= 178207)
        or (178208 <= code <= 183983)
    )


def normalize_word(text: Any) -> str:
    """Normalize a word/token for transcript and alignment matching."""
    value = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    kept: List[str] = []
    for ch in value:
        if ch.isalnum() or _is_cjk(ch):
            kept.append(ch)
    return "".join(kept)


def tokenize_text(text: str) -> List[str]:
    """Normalize the same units used by alignment inference and queries."""
    return [normalize_word(text[a:b]) for a, b in token_spans(text)]


def token_spans(text: str) -> List[Tuple[int, int]]:
    """Return Unicode character spans for the canonical transcript units."""
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, char in enumerate(text):
        if _is_cjk(char):
            if start is not None:
                spans.append((start, index))
                start = None
            spans.append((index, index + 1))
        elif char.isalnum():
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans
