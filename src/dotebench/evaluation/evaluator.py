"""Concrete doteBench evaluator over frozen cases and metric abilities."""

import hashlib
import io
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from statistics import mean
from typing import Any

import soundfile as sf

from ..alignment import (
    WordSegment,
    alignment_to_segments,
    operation_pairs,
    query_intervals,
    validate_alignment,
)
from ..dataset import read_json, resources, safe_identifier
from ..services.backends.wdtw import compute_f0_preservation
from ..services.backends.wer import compute_error_rate
from .abilities import HTTPMetricAbilities, encoded
from .base import EvaluationProtocol, EvaluationResult
from .preservation import evaluate_preservation, select_preserved_segments
from .wer import aggregate_error_rate_results, compute_instruction_wer


def protocol_version() -> str:
    """Read the released scoring contract identity."""
    contract = read_json(resources() / "release/evaluation-protocol.json")
    if not isinstance(contract, dict) or set(contract) != {"version"}:
        raise ValueError("Invalid evaluation protocol identity")
    return safe_identifier(contract["version"])


FAMILY = {
    "ins": "text",
    "del": "text",
    "sub": "text",
    "emo": "emotion",
    "pitch": "prosody",
    "rate": "prosody",
    "pause": "pause",
}
OPERATIONS = {"ins": "insert", "del": "delete", "sub": "replace"}
MIN_PITCH_SHIFT_SEMITONES = 1.0
MIN_RATE_DURATION_CHANGE = 0.1
MIN_PAUSE_GAP_CHANGE_SECONDS = 0.16
PAUSE_GAP_TOLERANCE_SECONDS = 1e-6
THRESHOLD_EPSILON = 1e-9


def _pause_success(operation: str, source_gap: float, target_gap: float) -> bool:
    if operation == "ins":
        return (
            target_gap
            >= source_gap + MIN_PAUSE_GAP_CHANGE_SECONDS - PAUSE_GAP_TOLERANCE_SECONDS
        )
    if operation == "red":
        return (
            target_gap
            <= source_gap - MIN_PAUSE_GAP_CHANGE_SECONDS + PAUSE_GAP_TOLERANCE_SECONDS
        )
    return False


def _component_success(
    component: Mapping[str, Any], measurement: Mapping[str, Any]
) -> bool:
    family = component["family"]
    operation = component["operation"]
    if family == "text":
        local = measurement.get("local_wer_result")
        if isinstance(local, Mapping):
            edit_result = local.get("edit_region_wer")
            if isinstance(edit_result, Mapping) and isinstance(
                edit_result.get("error_rate"), (int, float)
            ):
                return float(edit_result["error_rate"]) == 0.0
        raise ValueError("Text component requires its XML-local recognition result")
    if family == "emotion":
        return measurement.get("predicted_emotion") == component["params"]["type"]
    if family == "prosody":
        if operation == "pitch":
            actual = measurement.get("pitch_shift_semitones")
            expected = float(component["params"]["semitones"])
            if actual is None:
                return False
            actual = float(actual)
            if expected > 0:
                return actual + THRESHOLD_EPSILON >= MIN_PITCH_SHIFT_SEMITONES
            if expected < 0:
                return actual - THRESHOLD_EPSILON <= -MIN_PITCH_SHIFT_SEMITONES
            return False
        actual = measurement.get("duration_ratio")
        expected = float(component["params"]["factor"])
        if actual is None:
            return False
        actual = float(actual)
        if expected > 1.0:
            return actual <= 1.0 - MIN_RATE_DURATION_CHANGE + THRESHOLD_EPSILON
        if expected < 1.0:
            return actual >= 1.0 + MIN_RATE_DURATION_CHANGE - THRESHOLD_EPSILON
        return False
    if family == "pause":
        source_gap = measurement.get("source_gap_sec")
        target_gap = measurement.get("target_gap_sec")
        if source_gap is None or target_gap is None:
            return False
        return _pause_success(operation, float(source_gap), float(target_gap))
    return False


def _expected_rate_duration(original_duration: float, factor: float) -> float:
    if factor <= 0:
        raise ValueError("Rate factor must be positive")
    return original_duration / factor


def _duration_l1(
    original_duration: float, generated_duration: float, factor: float
) -> float:
    return abs(generated_duration - _expected_rate_duration(original_duration, factor))


def _actual_pitch_shift_semitones(
    original_f0_hz: float | None, generated_f0_hz: float | None
) -> float | None:
    if original_f0_hz is None or generated_f0_hz is None:
        return None
    if not math.isfinite(original_f0_hz) or not math.isfinite(generated_f0_hz):
        return None
    if original_f0_hz <= 0.0 or generated_f0_hz <= 0.0:
        return None
    return 12.0 * math.log2(generated_f0_hz / original_f0_hz)


def _pitch_l1_semitones(
    original_f0_hz: float | None,
    generated_f0_hz: float | None,
    requested_semitones: float,
) -> float | None:
    actual = _actual_pitch_shift_semitones(original_f0_hz, generated_f0_hz)
    if actual is None:
        return None
    return abs(actual - requested_semitones)


def _measure_pitch_pairs(pairs, source_audio, target_audio, measure_f0, requested):
    pairs = [(s, t) for s, t in pairs if s.duration > 0 and t.duration > 0]
    shifts = []
    if pairs:
        source = measure_f0(
            source_audio, [{"start": s.start, "end": s.end} for s, _ in pairs]
        )
        target = measure_f0(
            target_audio, [{"start": t.start, "end": t.end} for _, t in pairs]
        )
        if len(source) != len(pairs) or len(target) != len(pairs):
            raise ValueError("F0 service did not return every requested word")
        for a, b in zip(source, target, strict=True):
            shift = _actual_pitch_shift_semitones(
                a.get("median_f0_hz"), b.get("median_f0_hz")
            )
            if shift is not None:
                shifts.append(shift)
    shift = sum(shifts) / len(shifts) if shifts else None
    return {
        "measurement_status": "ok" if shifts else "unmeasurable",
        "pitch_shift_semitones": shift,
        "pitch_l1_semitones": abs(shift - requested) if shift is not None else None,
    }


def _finite(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} returned a missing or non-finite value")
    return float(value)


def _mean(values):
    values = [_finite(v, "aggregate input") for v in values]
    return mean(values) if values else None


def _duration(audio):
    info = sf.info(io.BytesIO(audio))
    return info.frames / info.samplerate


class DoteBenchEvaluator(EvaluationProtocol):
    def __init__(self, abilities=None, *, urls=None, timeout=300, case_groups=None):
        self.case_groups = dict(case_groups or {})
        self.abilities = (
            abilities if abilities is not None else HTTPMetricAbilities(urls, timeout)
        )

    def identity(self):
        service_identity = self.abilities.identity()
        return {
            "protocol_version": protocol_version(),
            "services": service_identity,
        }

    def prepare(self):
        self.abilities.prepare()

    def close(self):
        self.abilities.close()

    def _category(self, case):
        group = self.case_groups.get(case.id)
        category = group.get("category") if isinstance(group, dict) else group
        families = {FAMILY[o.kind] for o in case.instruction.operations}
        return category or (
            "compositional" if len(families) > 1 else next(iter(families))
        )

    def _align(self, audio, text, language):
        result = self.abilities.measure(
            "qwen3_aligner", audio=encoded(audio), text=text, language=language
        )
        validate_alignment(text, result["segments"], _duration(audio))
        segments = tuple(alignment_to_segments(result["segments"]))
        for segment in segments:
            _finite(segment.start, "alignment start")
            _finite(segment.end, "alignment end")
        return segments

    def _f0(self, audio, spans):
        result = self.abilities.measure("f0", audio=encoded(audio), spans=spans)
        if len(result["spans"]) != len(spans):
            raise ValueError("F0 service did not return every requested span")
        return result["spans"]

    def _source_segments(self, case, source):
        frozen = case.annotations.source_alignment
        if frozen is None:
            raise ValueError("Missing or unsupported frozen source alignment")
        if (
            frozen.audio_sha256 != hashlib.sha256(source).hexdigest()
            or frozen.text_sha256
            != hashlib.sha256(case.source_text.encode()).hexdigest()
        ):
            raise ValueError("Frozen source alignment checksum mismatch")
        validate_alignment(case.source_text, frozen.segments, _duration(source))
        source_segments = tuple(
            alignment_to_segments([asdict(s) for s in frozen.segments])
        )
        return source_segments

    def evaluate(self, request):
        case = request.case
        if request.generation.status == "candidate_error":
            result = self._penalty(case, request)
            if case.id in self.case_groups:
                result["group"] = self.case_groups[case.id]
            return EvaluationResult(case.id, result)
        if request.generated_audio is None:
            raise ValueError("Successful generation must have decoded audio")
        source, target = request.source_audio.data, request.generated_audio.data
        source_segments = self._source_segments(case, source)
        hypothesis = self.abilities.measure(
            "qwen3_asr",
            audio=encoded(target),
            language=None if self._category(case) == "pause" else case.language,
        )["text"]
        if not isinstance(hypothesis, str):
            raise ValueError("ASR returned a non-text hypothesis")
        # The released protocol uses mixed units, including for local text WER.
        wer = self.abilities.measure(
            "wer",
            reference=case.target_text,
            hypothesis=hypothesis,
            language=case.language if self._category(case) == "emotion" else "auto",
        )
        counts = aggregate_error_rate_results([wer])
        if counts.ref_length > 0 and not math.isclose(
            _finite(wer["error_rate"], "WER"),
            counts.error_rate,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("WER value disagrees with sufficient statistics")
        local = compute_instruction_wer(
            case.instruction, hypothesis, context_k=3
        ).to_dict()
        target_segments = self._align(target, case.target_text, case.language)
        preservation = evaluate_preservation(
            {
                "instruction_xml": case.instruction_xml,
                "source_text": case.source_text,
                "target_text": case.target_text,
                "source_alignments": [asdict(s) for s in source_segments],
                "target_alignments": [asdict(s) for s in target_segments],
                "source_audio": encoded(source),
                "target_audio": encoded(target),
            },
            lambda payload: self.abilities.measure("wdtw", **payload),
        )
        dur, f0 = preservation["wdtw_dur"], preservation["wdtw_f0"]
        sim = self.abilities.measure(
            "speaker_similarity",
            source_audio=encoded(source),
            target_audio=encoded(target),
        )
        quality = self.abilities.measure("utmos", audio=encoded(target))
        components = self._components(
            case,
            source,
            target,
            hypothesis=hypothesis,
            source_segments=source_segments,
            target_segments=target_segments,
        )
        result = {
            "protocol_version": protocol_version(),
            "evaluation_status": "ok",
            "penalty_applied": False,
            "language": case.language,
            "hypothesis": hypothesis,
            "wer": wer,
            "local_wer": local,
            "wdtw_dur": dur,
            "wdtw_f0": f0,
            "speaker_similarity": _finite(sim["similarity"], "speaker similarity"),
            "utmos": _finite(quality["score"], "UTMOS"),
            "components": components,
            "alignment": {
                "target": [asdict(s) for s in target_segments],
                "selection": preservation["selection"],
            },
        }
        if case.id in self.case_groups:
            result["group"] = self.case_groups[case.id]
        return EvaluationResult(case.id, result)

    def _components(
        self, case, source, target, *, hypothesis, source_segments, target_segments
    ):
        rows = []
        category = self._category(case)
        composite = category == "compositional"
        whole_emotion = (
            not composite
            and len(case.instruction.operations) == 1
            and case.instruction.operations[0].kind == "emo"
            and not case.target_text[
                : case.instruction.operations[0].target.start
            ].strip()
            and not case.target_text[
                case.instruction.operations[0].target.end :
            ].strip()
        )
        targeted = [
            (i, o)
            for i, o in enumerate(case.instruction.operations)
            if o.kind not in {"ins", "del", "sub"} and not whole_emotion
        ]
        target_times = (
            query_intervals(
                target_segments,
                case.target_text,
                targeted,
                side="target",
                audio_duration=_duration(target),
            )
            if targeted
            else {}
        )
        source_times = (
            query_intervals(
                source_segments,
                case.source_text,
                targeted,
                side="source",
                audio_duration=_duration(source),
            )
            if targeted
            else {}
        )
        for index, operation in enumerate(case.instruction.operations):
            params = dict(operation.params)
            component = {
                "family": FAMILY[operation.kind],
                "operation": params["act"]
                if operation.kind == "pause"
                else OPERATIONS.get(operation.kind, operation.kind),
                "params": params,
            }
            measurement = {}
            if operation.kind in {"ins", "del", "sub"}:
                if hypothesis is None:
                    raise ValueError("Text components require the complete hypothesis")
                measurement["local_wer_result"] = compute_instruction_wer(
                    case.instruction,
                    hypothesis,
                    operation_index=index,
                ).to_dict()
            if operation.kind not in {"ins", "del", "sub"}:
                source_time = source_times.get(index)
                ts, te = (
                    (0.0, _duration(target)) if whole_emotion else target_times[index]
                )
                target_duration = max(0.0, te - ts)
                measurement["target_span"] = {"start": ts, "end": te}
                if operation.kind in {"pitch", "rate"}:
                    if source_time is None:
                        raise ValueError(
                            f"Missing frozen source operation reference: {index}"
                        )
                    start, end = source_time
                    source_duration = max(0.0, end - start)
                    measurement["source_span"] = {"start": start, "end": end}
                if operation.kind == "emo":
                    whole = whole_emotion
                    if whole:
                        ts, te = 0.0, _duration(target)
                    if te <= ts:
                        measurement.update(
                            predicted_emotion=None, measurement_status="empty_target"
                        )
                    else:
                        classification = self.abilities.measure(
                            "emotion",
                            audio=encoded(target),
                            start=ts,
                            end=te,
                            case_id=case.id,
                            span_index=0 if composite else index,
                            input_strategy="whole_audio"
                            if whole
                            else "padded_crop_250ms",
                        )
                        measurement.update(classification)
                        if classification.get("predicted_emotion") not in {
                            "happy",
                            "angry",
                            "sad",
                            "afraid",
                            "disgusted",
                            "melancholic",
                            "surprised",
                            "calm",
                        }:
                            raise ValueError(
                                "Emotion classifier did not return a canonical label"
                            )
                elif operation.kind == "pitch":
                    measurement.update(
                        _measure_pitch_pairs(
                            operation_pairs(
                                case.instruction,
                                operation,
                                source_segments,
                                target_segments,
                            ),
                            source,
                            target,
                            self._f0,
                            float(params["semitones"]),
                        )
                    )
                elif operation.kind == "rate":
                    if source_duration <= 0:
                        raise ValueError("Empty source rate span")
                    factor = float(params["factor"])
                    measurement.update(
                        duration_ratio=target_duration / source_duration,
                        duration_l1_sec=_duration_l1(
                            source_duration, target_duration, factor
                        ),
                    )
                elif operation.kind == "pause":
                    source_gap = (
                        max(0.0, source_time[1] - source_time[0])
                        if source_time
                        else None
                    )
                    if source_gap is None:
                        raise ValueError("Missing frozen source gap")
                    target_gap = target_duration
                    delta = target_gap - source_gap
                    direction = _pause_success(params["act"], source_gap, target_gap)
                    measurement.update(
                        source_gap_sec=source_gap,
                        target_gap_sec=target_gap,
                        pause_delta_sec=delta,
                        direction_correct=direction,
                    )
            rows.append(
                {
                    "operation_index": index,
                    **component,
                    "measurement": measurement,
                    "success": None
                    if measurement.get("measurement_status") == "unmeasurable"
                    else _component_success(component, measurement),
                }
            )
        return rows

    def _penalty(self, case, request):
        from ..text import tokenize_text

        source = request.source_audio.data
        source_segments = self._source_segments(case, source)
        target_segments = [
            WordSegment(word, 0, 0) for word in tokenize_text(case.target_text)
        ]
        ds, _, _ = select_preserved_segments(
            case.instruction,
            source_segments,
            target_segments,
            operation_kind="pitch",
        )
        fs, _, _ = select_preserved_segments(
            case.instruction,
            source_segments,
            target_segments,
            operation_kind="rate",
        )
        dur_normalizer = sum(s.duration for s in ds)
        f0_normalizer = 0.0
        if fs:
            reference = self._f0(source, [{"start": s.start, "end": s.end} for s in fs])
            f0_normalizer = compute_f0_preservation(
                source_segments=fs,
                target_segments=fs,
                source_f0_spans=reference,
                target_f0_spans=reference,
            ).normalizer
        penalty_language = (
            case.language if self._category(case) == "emotion" else "auto"
        )
        wer = compute_error_rate(
            case.target_text, "", language=penalty_language
        ).to_dict()
        wer["penalty_applied"] = True
        components = []
        for index, op in enumerate(case.instruction.operations):
            params = dict(op.params)
            measurement = {"penalty_applied": True}
            if op.kind == "pitch":
                measurement["pitch_l1_semitones"] = 8.0
            if op.kind == "rate":
                start, end = query_intervals(
                    source_segments,
                    case.source_text,
                    [(index, op)],
                    side="source",
                    audio_duration=_duration(source),
                )[index]
                if end <= start:
                    raise ValueError("Empty source rate span")
                measurement["duration_l1_sec"] = (end - start) / float(params["factor"])
            if op.kind == "pause":
                measurement.update(direction_correct=False)
            components.append(
                {
                    "operation_index": index,
                    "family": FAMILY[op.kind],
                    "operation": params["act"]
                    if op.kind == "pause"
                    else OPERATIONS.get(op.kind, op.kind),
                    "params": params,
                    "measurement": measurement,
                    "success": False,
                }
            )
        return {
            "protocol_version": protocol_version(),
            "evaluation_status": "penalized_generation_failed",
            "penalty_applied": True,
            "language": case.language,
            "penalty_reason": request.generation.error,
            "hypothesis": "",
            "wer": wer,
            "local_wer": compute_instruction_wer(
                case.instruction, "", context_k=3
            ).to_dict(),
            "wdtw_dur": {
                "wdtw_dur": 1.0 if dur_normalizer else None,
                "distance": dur_normalizer,
                "normalizer": dur_normalizer,
                "status": "penalized_empty_audio" if dur_normalizer else "no_segments",
            },
            "wdtw_f0": {
                "wdtw_f0": 8.0 if f0_normalizer else None,
                "distance": 8.0 * f0_normalizer,
                "normalizer": f0_normalizer,
                "status": "penalized_empty_audio" if f0_normalizer else "no_valid_f0",
            },
            "speaker_similarity": 0.0,
            "utmos": 1.0,
            "components": components,
        }

    def aggregate(self, results):
        ids = [r.id for r in results]
        if len(ids) != len(set(ids)):
            raise ValueError("Cannot aggregate duplicate evaluation IDs")
        rows = [dict(r.metrics) for r in results]
        if any(
            r.get("evaluation_status") not in {"ok", "penalized_generation_failed"}
            for r in rows
        ):
            raise ValueError("Incomplete evaluation cannot produce an official summary")
        version = protocol_version()
        if any(row.get("protocol_version") != version for row in rows):
            raise ValueError("Cannot aggregate a different evaluation protocol version")
        groups = defaultdict(list)
        for row in rows:
            families = sorted({c["family"] for c in row["components"]})
            group = row.get("group")
            category = group.get("category") if isinstance(group, dict) else group
            groups[
                category or ("compositional" if len(families) > 1 else families[0])
            ].append(row)
        return {
            "protocol_version": version,
            "evaluated": len(rows),
            "penalized": sum(r["penalty_applied"] for r in rows),
            "by_category": {
                name: summarize_category(items, name)
                for name, items in sorted(groups.items())
            },
        }


def _scored_components(row):
    components = []
    for component in row["components"]:
        if component["measurement"].get("measurement_status") == "unmeasurable":
            if component["operation"] != "pitch" or component["success"] is not None:
                raise ValueError("Only unmeasurable pitch operations may be excluded")
            continue
        if not isinstance(component["success"], bool):
            raise ValueError("Measured component requires boolean success")
        components.append(component)
    return components


def _corpus_percent(values):
    value = aggregate_error_rate_results(values).error_rate
    return 100 * value if value is not None else None


def summarize_group(rows, category):
    result = {
        "count": len(rows),
        "asr_error": _corpus_percent([r["wer"] for r in rows]),
        "speaker_similarity": _mean([r["speaker_similarity"] for r in rows]),
        "utmos": _mean([r["utmos"] for r in rows]),
    }
    for name, scale in (("wdtw_dur", 100), ("wdtw_f0", 1)):
        values = [r[name] for r in rows]
        distance = sum(_finite(v["distance"], name) for v in values)
        normalizer = sum(_finite(v["normalizer"], name) for v in values)
        if any(
            v["normalizer"] < 0
            or v["distance"] < 0
            or (v["normalizer"] == 0 and v["distance"] != 0)
            for v in values
        ):
            raise ValueError("Invalid WDTW sufficient statistics")
        result[name] = {
            "value": scale * distance / normalizer if normalizer else None,
            "distance": distance,
            "normalizer": normalizer,
        }
    per_case = [_scored_components(r) for r in rows]
    components = [c for cs in per_case for c in cs]
    by_family = defaultdict(list)
    for c in components:
        by_family[c["family"]].append(c)
    result["by_family"] = {
        f: {"count": len(cs), "success_rate": _mean([float(c["success"]) for c in cs])}
        for f, cs in sorted(by_family.items())
    }
    result["component_success_rate"] = _mean([float(c["success"]) for c in components])
    result["all_component_success_rate"] = _mean(
        [float(all(c["success"] for c in cs)) for cs in per_case if cs]
    )
    if category == "text":
        for name in ("edit_region_wer", "non_edit_region_wer"):
            result[name] = _corpus_percent([r["local_wer"][name] for r in rows])
    if category == "emotion":
        value = _mean([float(c["success"]) for c in components])
        result["emotion_accuracy"] = 100 * value if value is not None else None
    if category == "pause":
        value = _mean(
            [float(c["measurement"]["direction_correct"]) for c in components]
        )
        result["direction_accuracy"] = 100 * value if value is not None else None
    for key, operation in (
        ("duration_l1_sec", "rate"),
        ("pitch_l1_semitones", "pitch"),
    ):
        result[key] = _mean(
            [c["measurement"][key] for c in components if c["operation"] == operation]
        )
    return result


def summarize_category(rows, category):
    shards = defaultdict(list)
    for row in rows:
        group = row.get("group")
        shard = (
            group.get("shard", "selected") if isinstance(group, dict) else "selected"
        )
        shards[shard].append(row)
    result = summarize_group(rows, category)
    result["by_shard"] = {
        name: summarize_group(items, category) for name, items in sorted(shards.items())
    }
    result["aggregation"] = "pooled_sufficient_statistics"
    return result
