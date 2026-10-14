"""Duration/F0 computations and their WDTW HTTP service backend."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from dotebench.alignment import WordSegment

from dotebench.services.utils.http import json_endpoint, request_json, service_app


MISSING_TARGET_F0_PENALTY_SEMITONES = 8.0


def f0_statistics(span, keys):
    """Read nullable F0 statistics; malformed service data is not unvoiced audio."""
    if not isinstance(span, Mapping) or any(key not in span for key in keys):
        raise ValueError("F0 service returned missing statistic fields")
    values = [span[key] for key in keys]
    for value in values:
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("F0 service returned an invalid statistic")
    return None if any(value is None for value in values) else values


def _duration_local_cost(
    source: WordSegment, target: WordSegment, *, word_mismatch_penalty: float
) -> float:
    duration_cost = abs(source.duration - target.duration)
    if source.word == target.word:
        return duration_cost
    return source.duration + target.duration + word_mismatch_penalty


def _duration_dtw_distance(
    source: Sequence[WordSegment],
    target: Sequence[WordSegment],
    *,
    word_mismatch_penalty: float = 1.0,
) -> float:
    if not source and not target:
        return 0.0
    if not source:
        return sum(seg.duration + word_mismatch_penalty for seg in target)
    if not target:
        return sum(seg.duration + word_mismatch_penalty for seg in source)
    rows = len(source) + 1
    cols = len(target) + 1
    dp = [[math.inf] * cols for _ in range(rows)]
    dp[0][0] = 0.0
    for i in range(1, rows):
        for j in range(1, cols):
            cost = _duration_local_cost(
                source[i - 1],
                target[j - 1],
                word_mismatch_penalty=word_mismatch_penalty,
            )
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return float(dp[-1][-1])


def compute_duration_preservation(source, target):
    normalizer = sum(s.duration for s in source) + sum(t.duration for t in target)
    distance = _duration_dtw_distance(source, target) if normalizer else 0.0
    return {
        "wdtw_dur": distance / normalizer if normalizer else None,
        "distance": distance,
        "normalizer": normalizer,
        "status": "ok" if normalizer else "no_segments",
        "mode": "preserved",
    }


@dataclass(frozen=True)
class F0PreservationResult:
    wdtw_f0: float | None
    distance: float
    normalizer: float
    status: str
    word_results: list[dict[str, Any]]
    error: str | None = None
    unit: str = "semitones"

    def to_dict(self, include_words: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "wdtw_f0": self.wdtw_f0,
            "distance": self.distance,
            "normalizer": self.normalizer,
            "status": self.status,
            "unit": self.unit,
        }
        if self.error:
            data["error"] = self.error
        if include_words:
            data["word_results"] = self.word_results
        return data


def compute_f0_preservation(
    *,
    source_segments: Sequence[WordSegment],
    target_segments: Sequence[WordSegment],
    source_f0_spans: Sequence[Mapping[str, Any]],
    target_f0_spans: Sequence[Mapping[str, Any]],
) -> F0PreservationResult:
    """Compute equal-word-weighted F0 preservation error in semitones."""
    if len(source_segments) != len(target_segments):
        raise ValueError(
            f"Source and target preserved segment counts must match: source={len(source_segments)}, target={len(target_segments)}"
        )
    eligible_count = len(source_segments)
    if eligible_count == 0:
        return F0PreservationResult(
            wdtw_f0=None,
            distance=0.0,
            normalizer=0.0,
            status="no_preserved_words",
            word_results=[],
        )
    if len(source_f0_spans) != eligible_count or len(target_f0_spans) != eligible_count:
        raise ValueError(
            f"F0 span count must match preserved word count: eligible={eligible_count}, source={len(source_f0_spans)}, target={len(target_f0_spans)}"
        )
    word_results: list[dict[str, Any]] = []
    total_error = 0.0
    source_valid_count = 0
    statistic_keys = ("min_f0_hz", "max_f0_hz", "mean_f0_hz")
    for index in range(eligible_count):
        source_f0 = source_f0_spans[index]
        target_f0 = target_f0_spans[index]
        errors: dict[str, float] = {}
        source_values = (
            f0_statistics(source_f0, statistic_keys)
            if source_segments[index].duration > 0
            else None
        )
        target_values = (
            f0_statistics(target_f0, statistic_keys)
            if target_segments[index].duration > 0
            else None
        )
        word_result: dict[str, Any] = {
            "index": index,
            "word": source_segments[index].word,
            "source_span": source_segments[index].to_dict(),
            "target_span": target_segments[index].to_dict(),
            "source_f0": dict(source_f0),
            "target_f0": dict(target_f0),
            "status": "invalid_source_f0",
        }
        if source_values is not None:
            if target_values is None:
                # The empty-output scoring convention, applied per source-valid word.
                word_error = MISSING_TARGET_F0_PENALTY_SEMITONES
                word_result["status"] = "penalized_missing_target_f0"
            else:
                for key, source_value, target_value in zip(
                    statistic_keys, source_values, target_values, strict=True
                ):
                    errors[key.removesuffix("_f0_hz") + "_error_semitones"] = abs(
                        12.0 * math.log2(target_value / source_value)
                    )
                word_error = sum(errors.values()) / len(errors)
                word_result.update(errors)
                word_result["status"] = "ok"
            word_result["word_error_semitones"] = word_error
            total_error += word_error
            source_valid_count += 1
        word_results.append(word_result)
    return F0PreservationResult(
        wdtw_f0=total_error / source_valid_count if source_valid_count else None,
        distance=total_error,
        normalizer=float(source_valid_count),
        status=(
            "penalized_missing_target_f0"
            if any(w["status"] == "penalized_missing_target_f0" for w in word_results)
            else "ok"
            if source_valid_count
            else "no_valid_f0"
        ),
        word_results=word_results,
    )


def measure_spans(payload, measure_f0):
    """Compare caller-selected segment pairs over the two complete recordings."""

    def segments(name):
        if name not in payload or not isinstance(payload[name], list):
            raise ValueError(f"WDTW requires a segment list: {name}")
        result = [
            WordSegment(
                word=item["word"],
                start=float(item["start"]),
                end=float(item["end"]),
                raw_word=item.get("raw_word", item["word"]),
            )
            for item in payload[name]
        ]
        for segment in result:
            if (
                not math.isfinite(segment.start)
                or not math.isfinite(segment.end)
                or segment.start < 0
                or segment.end < segment.start
            ):
                raise ValueError("WDTW segments require finite 0 <= start <= end")
        return result

    ds, dt = segments("duration_source"), segments("duration_target")
    fs, ft = segments("f0_source"), segments("f0_target")
    if len(fs) != len(ft):
        raise ValueError("Source and target F0 segment counts must match")

    def f0(audio, segments):
        if not segments:
            return []
        indices = [i for i, s in enumerate(segments) if s.duration > 0]
        if not indices:
            return [{} for _ in segments]
        response = measure_f0(
            {
                "audio": audio,
                "spans": [
                    {"start": s.start, "end": s.end}
                    for s in (segments[i] for i in indices)
                ],
            }
        )
        if len(response["spans"]) != len(indices):
            raise ValueError("F0 service did not return every requested span")
        result = [{} for _ in segments]
        for i, value in zip(indices, response["spans"], strict=True):
            result[i] = value
        return result

    return {
        "wdtw_dur": compute_duration_preservation(ds, dt),
        "wdtw_f0": compute_f0_preservation(
            source_segments=fs,
            target_segments=ft,
            source_f0_spans=f0(payload["source_audio"], fs),
            target_f0_spans=f0(payload["target_audio"], ft),
        ).to_dict(),
    }


def build(*, f0_url, f0_timeout_sec=300.0):
    if not f0_url:
        raise ValueError("WDTW requires an F0 service URL")

    def measure_f0(payload):
        with requests.Session() as session:
            return request_json(
                session,
                "POST",
                f0_url.rstrip("/") + "/measure",
                payload=payload,
                timeout=f0_timeout_sec,
            )

    def measure(payload):
        return measure_spans(payload, measure_f0)

    measure.configuration = {"f0_url": f0_url.rstrip("/")}
    return measure


def create_app(measure):
    app = service_app(
        title="wdtw",
        health=lambda: {
            "status": "ready",
            "ability": "wdtw",
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
    parser.add_argument("--f0-url", required=True)
    parser.add_argument("--f0-timeout-sec", type=float, default=300.0)
    args = parser.parse_args()
    measure = build(f0_url=args.f0_url, f0_timeout_sec=args.f0_timeout_sec)
    import uvicorn

    uvicorn.run(create_app(measure), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
