"""WER/CER computations and their HTTP service backend."""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import asdict, dataclass

import jiwer

from dotebench.services.utils.http import json_endpoint, service_app


@dataclass
class WERResult:
    """Result of a WER/CER computation."""

    error_rate: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int
    hyp_length: int
    reference_normalized: str
    hypothesis_normalized: str
    metric: str

    def to_dict(self) -> dict:
        return asdict(self)


_CJK_RANGES = (
    (19968, 40959),
    (13312, 19903),
    (131072, 173791),
    (173824, 177983),
    (177984, 178207),
    (178208, 183983),
    (63744, 64255),
    (194560, 195103),
)


def _is_cjk_char(cp: int) -> bool:
    return any(lo <= cp <= hi for (lo, hi) in _CJK_RANGES)


def _insert_script_boundary_spaces(text: str) -> str:
    """Insert spaces at CJK↔Latin and CJK↔digit script-switching boundaries.

    Ensures consistent tokenization regardless of input spacing:
        "一个apple"  → "一个 apple"
        "第3个"      → "第 3 个"
        "一个 apple" → "一个 apple"  (already spaced — no change)

    This prevents mismatches like "一个apple" vs "一个 apple" being treated
    as different tokens in WER mode.
    """
    if not text:
        return text
    result = [text[0]]
    for i in range(1, len(text)):
        prev_cp = ord(text[i - 1])
        curr_cp = ord(text[i])
        prev_cjk = _is_cjk_char(prev_cp)
        curr_cjk = _is_cjk_char(curr_cp)
        prev_alnum = text[i - 1].isascii() and text[i - 1].isalnum()
        curr_alnum = text[i].isascii() and text[i].isalnum()
        if prev_cjk and curr_alnum or (prev_alnum and curr_cjk):
            result.append(" ")
        result.append(text[i])
    return "".join(result)


def normalize_text(text: str) -> str:
    """Normalize text for error rate computation.

    - Unicode NFKC normalization
    - Lowercase
    - Insert spaces at CJK↔Latin and CJK↔digit script boundaries
    - Remove punctuation (keep alphanumeric, CJK, spaces)
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _insert_script_boundary_spaces(text)
    text = re.sub("[^\\w\\s]", " ", text)
    text = re.sub("\\s+", " ", text).strip()
    return text


def mixed_language_tokens(text: str) -> list[str]:
    """Tokenize CJK as characters and non-CJK runs by whitespace-delimited words."""
    normalized = normalize_text(text)
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current.clear()

    for char in normalized:
        if char.isspace():
            flush()
            continue
        if _is_cjk_char(ord(char)):
            flush()
            tokens.append(char)
            continue
        current.append(char)
    flush()
    return tokens


def _compute_token_error_rate(
    reference_tokens: list[str],
    hypothesis_tokens: list[str],
    *,
    reference_normalized: str,
    hypothesis_normalized: str,
    metric: str,
) -> WERResult:
    if not reference_tokens:
        error_rate = 0.0 if not hypothesis_tokens else float("inf")
        return WERResult(
            error_rate=error_rate,
            substitutions=0,
            deletions=0,
            insertions=len(hypothesis_tokens),
            hits=0,
            ref_length=0,
            hyp_length=len(hypothesis_tokens),
            reference_normalized=reference_normalized,
            hypothesis_normalized=hypothesis_normalized,
            metric=metric,
        )
    output = jiwer.process_words(
        " ".join(reference_tokens), " ".join(hypothesis_tokens)
    )
    return WERResult(
        error_rate=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        ref_length=len(reference_tokens),
        hyp_length=len(hypothesis_tokens),
        reference_normalized=reference_normalized,
        hypothesis_normalized=hypothesis_normalized,
        metric=metric,
    )


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    """Compute Word Error Rate.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted / transcribed text.

    Returns:
        WERResult with error rate and detailed statistics.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    if not ref_norm:
        error_rate = 0.0 if not hyp_norm else float("inf")
        hyp_words = hyp_norm.split() if hyp_norm else []
        return WERResult(
            error_rate=error_rate,
            substitutions=0,
            deletions=0,
            insertions=len(hyp_words),
            hits=0,
            ref_length=0,
            hyp_length=len(hyp_words),
            reference_normalized=ref_norm,
            hypothesis_normalized=hyp_norm,
            metric="wer",
        )
    output = jiwer.process_words(ref_norm, hyp_norm)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    return WERResult(
        error_rate=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        ref_length=len(ref_words),
        hyp_length=len(hyp_words),
        reference_normalized=ref_norm,
        hypothesis_normalized=hyp_norm,
        metric="wer",
    )


def compute_cer(reference: str, hypothesis: str) -> WERResult:
    """Compute Character Error Rate.

    Suitable for Chinese and other CJK languages where word
    segmentation is ambiguous. Each character becomes a token.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted / transcribed text.

    Returns:
        WERResult with error rate and detailed statistics (metric="cer").
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    ref_chars = ref_norm.replace(" ", "")
    hyp_chars = hyp_norm.replace(" ", "")
    if not ref_chars:
        error_rate = 0.0 if not hyp_chars else float("inf")
        return WERResult(
            error_rate=error_rate,
            substitutions=0,
            deletions=0,
            insertions=len(hyp_chars),
            hits=0,
            ref_length=0,
            hyp_length=len(hyp_chars),
            reference_normalized=ref_norm,
            hypothesis_normalized=hyp_norm,
            metric="cer",
        )
    ref_spaced = " ".join(ref_chars)
    hyp_spaced = " ".join(hyp_chars)
    output = jiwer.process_words(ref_spaced, hyp_spaced)
    return WERResult(
        error_rate=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        ref_length=len(ref_chars),
        hyp_length=len(hyp_chars),
        reference_normalized=ref_norm,
        hypothesis_normalized=hyp_norm,
        metric="cer",
    )


def compute_mixed_error_rate(reference: str, hypothesis: str) -> WERResult:
    """Compute error rate with CJK chars and non-CJK whitespace words as units."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    return _compute_token_error_rate(
        mixed_language_tokens(reference),
        mixed_language_tokens(hypothesis),
        reference_normalized=ref_norm,
        hypothesis_normalized=hyp_norm,
        metric="mixed",
    )


def compute_error_rate(
    reference: str, hypothesis: str, language: str = "auto"
) -> WERResult:
    """Compute error rate with automatic metric selection.

    - auto/mixed → CJK chars + non-CJK whitespace words
    - Chinese language hints → CER
    - Other explicit language hints → WER

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted / transcribed text.
        language: "auto"/"mixed", "zh"/"chinese" (→ CER), or anything else (→ WER).

    Returns:
        WERResult.
    """
    normalized_language = language.lower()
    if normalized_language in ("auto", "mixed"):
        return compute_mixed_error_rate(reference, hypothesis)
    if normalized_language in ("zh", "chinese", "zho", "cmn"):
        return compute_cer(reference, hypothesis)
    return compute_wer(reference, hypothesis)


def build():
    def measure(payload):
        result = compute_error_rate(
            payload["reference"],
            payload["hypothesis"],
            payload.get("language", "auto"),
        ).to_dict()
        # Empty references have undefined rate with insertions. Preserve
        # error counts and expose JSON null rather than invalid Infinity.
        if not math.isfinite(result["error_rate"]):
            result["error_rate"] = None
            result["status"] = "empty_reference"
        return result

    return measure


def create_app(measure):
    app = service_app(
        title="wer",
        health=lambda: {
            "status": "ready",
            "ability": "wer",
            "model_identity": None,
            "configuration": getattr(measure, "configuration", {}),
        },
        close=getattr(measure, "close", None),
    )
    app.post("/measure")(json_endpoint(measure))
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    measure = build()
    import uvicorn

    uvicorn.run(create_app(measure), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
