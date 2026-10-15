"""Operation scoring, measurability, and sufficient-statistic contracts."""

import json
import math
from types import SimpleNamespace

import pytest

from dotebench.alignment import WordSegment
from dotebench.domain import Annotations
from dotebench.evaluation.evaluator import (
    DoteBenchEvaluator,
    _component_success,
    _measure_pitch_pairs,
    summarize_category,
    summarize_group,
)
from dotebench.evaluation.preservation import evaluate_preservation
from dotebench.evaluation.wer import aggregate_error_rate_results
from dotebench.instructions import parse_instruction
from dotebench.services.backends.wdtw import compute_f0_preservation, measure_spans
from dotebench.services.backends.wer import compute_error_rate
from tests.evaluation.test_protocol import Abilities, audio, case, request


@pytest.mark.parametrize(
    "operation,delta,expected",
    [
        ("ins", 0.159, False),
        ("red", -0.159, False),
        ("ins", 0.16, True),
        ("red", -0.16, True),
        ("ins", 0.16 - 5e-7, True),
        ("red", -0.16 + 5e-7, True),
        ("ins", 0.16 - 2e-6, False),
        ("red", -0.16 + 2e-6, False),
        ("ins", 0.0, False),
        ("red", 0.0, False),
        ("ins", -0.1, False),
        ("red", 0.1, False),
    ],
)
def test_pause_success_requires_requested_160ms_change(operation, delta, expected):
    component = {"family": "pause", "operation": operation, "params": {}}
    measurement = {"source_gap_sec": 0.5, "target_gap_sec": 0.5 + delta}
    assert _component_success(component, measurement) is expected


def test_text_components_keep_independent_recognition_outcomes():
    instruction = parse_instruction(
        '<sub targ="red">blue</sub> one two three four five six seven '
        '<sub targ="green">yellow</sub>'
    )
    sample = SimpleNamespace(
        id="independent",
        language="en",
        instruction=instruction,
        source_text=instruction.source_text,
        target_text=instruction.target_text,
        annotations=Annotations(),
    )

    def unexpected_measure(*args, **kwargs):
        raise AssertionError("Text-only component scoring must not call a service")

    protocol = DoteBenchEvaluator(SimpleNamespace(measure=unexpected_measure))
    rows = protocol._components(
        sample,
        b"",
        b"",
        hypothesis="red one two three four five six seven yellow",
        source_segments=(),
        target_segments=(),
    )
    assert [row["success"] for row in rows] == [True, False]
    assert [row["operation_index"] for row in rows] == [0, 1]


def test_text_success_requires_xml_local_measurement():
    with pytest.raises(ValueError, match="XML-local"):
        _component_success(
            {"family": "text", "operation": "replace"},
            {"hypothesis": "a repeated replacement word"},
        )


def stat(value):
    return {
        key: value for key in ("min_f0_hz", "max_f0_hz", "mean_f0_hz", "median_f0_hz")
    }


def preservation(xml, source, target, values=None):
    inst = parse_instruction(xml)
    calls = []

    def probe(payload):
        calls.append(payload)
        assert all(s["end"] > s["start"] for s in payload["spans"])
        return {
            "spans": [stat(100) for _ in payload["spans"]]
            if values is None
            else values(payload)
        }

    result = evaluate_preservation(
        {
            "instruction_xml": xml,
            "source_text": inst.source_text,
            "target_text": inst.target_text,
            "source_alignments": [s.to_dict() for s in source],
            "target_alignments": [s.to_dict() for s in target],
            "source_audio": "source",
            "target_audio": "target",
        },
        lambda payload: measure_spans(payload, probe),
    )
    return result, calls


def test_zero_source_is_removed_from_both_branches_and_both_audio_probes():
    src = [WordSegment("a", 0, 0), WordSegment("b", 1, 2)]
    tgt = [WordSegment("a", 0, 1), WordSegment("b", 1, 2)]
    result, calls = preservation('a b<pause act="ins"/>', src, tgt)
    assert result["wdtw_dur"]["normalizer"] == 2
    assert result["wdtw_dur"]["distance"] == 0
    assert result["wdtw_f0"]["normalizer"] == 1
    assert all(p["spans"] == [{"start": 1, "end": 2}] for p in calls)


def test_zero_target_contributes_duration_and_f0_penalty():
    result, calls = preservation(
        'a<pause act="ins"/>',
        [WordSegment("a", 0, 1)],
        [WordSegment("a", 0, 0)],
    )
    assert result["wdtw_dur"]["wdtw_dur"] == 1
    assert result["wdtw_f0"]["wdtw_f0"] == 8
    assert result["wdtw_f0"]["normalizer"] == 1
    assert [p["audio"] for p in calls] == ["source"]


@pytest.mark.parametrize(
    "kind,attrs,dur,f0,expected_calls",
    [
        ("pitch", 'semitones="3"', 2, 0, 0),
        ("rate", 'factor="1.5"', 0, 1, 2),
        ("emo", 'type="happy" level="2"', 0, 0, 0),
    ],
)
def test_whole_acoustic_edit_has_metric_specific_preservation(
    kind, attrs, dur, f0, expected_calls
):
    seg = [WordSegment("a", 0, 1)]
    result, calls = preservation(f"<{kind} {attrs}>a</{kind}>", seg, seg)
    assert result["wdtw_dur"]["normalizer"] == dur
    assert result["wdtw_f0"]["normalizer"] == f0
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "source,target,expected,status",
    [
        ([100, None], [200, 100], 12, "ok"),
        ([100, 200], [200, 100], 0, "ok"),
        ([100, None], [None, 100], 0, "missing_target"),
        ([100, 100], [200, None], 6, "missing_target"),
    ],
)
def test_pitch_uses_source_valid_words(source, target, expected, status):
    seg = [WordSegment("a", 0, 1), WordSegment("b", 1, 2)]

    def probe(audio, spans):
        values = source if audio == "source" else target
        return [stat(values[int(s["start"])]) for s in spans]

    result = _measure_pitch_pairs(list(zip(seg, seg)), "source", "target", probe, 3)
    assert result["pitch_shift_semitones"] == expected
    assert result["pitch_l1_semitones"] == abs(expected - 3)
    assert result["measurement_status"] == status


@pytest.mark.parametrize("zero_duration", [False, True])
def test_pitch_without_valid_source_is_contract_error(zero_duration):
    source = WordSegment("a", 0, 0 if zero_duration else 1)

    def probe(*args):
        assert not zero_duration
        return [stat(None)]

    with pytest.raises(ValueError, match="no valid source F0"):
        _measure_pitch_pairs(
            [(source, WordSegment("a", 0, 1))], "source", "target", probe, 2
        )


def test_zero_emotion_fails_locally_and_other_metrics_continue():
    c = case('one <emo type="happy" level="1">two</emo> three')

    class Zero(Abilities):
        def measure(self, ability, **payload):
            result = super().measure(ability, **payload)
            if ability == "qwen3_aligner":
                result["segments"][1]["end"] = result["segments"][1]["start"]
            return result

    abilities = Zero(c.target_text, fail="emotion")
    result = DoteBenchEvaluator(abilities).evaluate(request(c)).metrics
    assert result["components"][0]["success"] is False
    assert (
        result["components"][0]["measurement"]["measurement_status"] == "empty_target"
    )
    assert sum(name == "qwen3_aligner" for name, _ in abilities.calls) == 1
    assert result["evaluation_status"] == "ok"


@pytest.mark.parametrize(
    "xml",
    [
        '<pitch semitones="3">one two</pitch>',
        '<pitch semitones="3">one</pitch> <emo type="happy" level="1">two</emo>',
    ],
)
def test_missing_pitch_retains_operation_and_sample_denominators(xml):
    c = case(xml)

    class Missing(Abilities):
        def measure(self, ability, **payload):
            result = super().measure(ability, **payload)
            if ability == "qwen3_aligner":
                result["segments"][0]["end"] = result["segments"][0]["start"]
            return result

    protocol = DoteBenchEvaluator(Missing(c.target_text))
    result = protocol.evaluate(request(c))
    component = result.metrics["components"][0]
    assert component["success"] is False
    assert component["measurement"]["measurement_status"] == "missing_target"
    summary = summarize_group([result.metrics], "compositional")
    assert summary["pitch_l1_semitones"] == 3
    assert summary["all_component_success_rate"] == 0
    assert summary["component_success_rate"] == (0.5 if "emo" in xml else 0)
    assert summary["by_family"]["prosody"]["count"] == 1


def test_rate_zero_target_retains_numeric_zero_and_fails():
    c = case('<rate factor="2">one</rate>')
    protocol = DoteBenchEvaluator(Abilities(c.target_text))
    rows = protocol._components(
        c,
        audio().data,
        audio().data,
        hypothesis="one",
        source_segments=[WordSegment("one", 0, 1)],
        target_segments=[WordSegment("one", 0, 0)],
    )
    assert rows[0]["measurement"]["duration_ratio"] == 0
    assert rows[0]["measurement"]["duration_l1_sec"] == 0.5
    assert rows[0]["success"] is False


def test_failure_weights_use_preserved_reference_and_real_local_partition():
    c = case('one <sub targ="four five">two</sub> three')
    protocol = DoteBenchEvaluator(Abilities(c.target_text))
    result = protocol.evaluate(request(c, "candidate_error")).metrics
    assert result["wdtw_dur"]["normalizer"] == pytest.approx(0.4)
    assert result["wdtw_f0"]["normalizer"] == 2
    assert result["wdtw_f0"]["distance"] == 16
    assert result["local_wer"]["edit_region_wer"]["ref_length"] == 4
    assert result["local_wer"]["non_edit_region_wer"]["ref_length"] == 0


def test_failure_whole_emotion_has_no_preservation_penalty():
    c = case('<emo type="happy" level="1">one two</emo>')
    abilities = Abilities(c.target_text)
    result = (
        DoteBenchEvaluator(abilities).evaluate(request(c, "candidate_error")).metrics
    )
    assert result["wdtw_dur"]["normalizer"] == result["wdtw_f0"]["normalizer"] == 0
    assert abilities.calls == []
    assert result["components"][0]["success"] is False


def test_empty_reference_insertions_remain_in_corpus_numerator():
    empty = compute_error_rate("", "extra", "en")
    regular = compute_error_rate("one two", "one two", "en")
    assert aggregate_error_rate_results([empty, regular]).error_rate == 0.5
    assert aggregate_error_rate_results([empty]).error_rate is None
    with pytest.raises(ValueError, match="statistics"):
        aggregate_error_rate_results([None])


def test_emotion_category_pools_shards_without_intermediate_rounding():
    c = case('<emo type="happy" level="1">one two</emo>')
    result = DoteBenchEvaluator(Abilities(c.target_text)).evaluate(request(c)).metrics
    first = {
        **result,
        "group": {"shard": "a"},
        "wer": compute_error_rate("one", "", "en").to_dict(),
    }
    second = {
        **result,
        "group": {"shard": "b"},
        "wer": compute_error_rate("one two three", "one two three", "en").to_dict(),
    }
    summary = summarize_category([first, second], "emotion")
    assert summary["asr_error"] == 25
    assert summary["wdtw_f0"]["value"] is None
    assert "valid_count" not in str(summary)
    with pytest.raises(ValueError, match="aggregate input"):
        summarize_group([{**first, "utmos": None}], "emotion")


def test_protocol_version_mixing_is_rejected():
    from dotebench.evaluation.base import EvaluationResult

    c = case('<pitch semitones="3">one</pitch>')
    protocol = DoteBenchEvaluator(Abilities(c.target_text))
    result = protocol.evaluate(request(c))
    old = EvaluationResult(c.id, {**result.metrics, "protocol_version": "v1_0"})
    with pytest.raises(ValueError, match="protocol version"):
        protocol.aggregate([old])


def test_protocol_identity_comes_from_release_contract(monkeypatch, tmp_path):
    from dotebench.evaluation import evaluator

    version = "fixture_next"
    (tmp_path / "release").mkdir()
    (tmp_path / "release/evaluation-protocol.json").write_text(
        json.dumps({"version": version})
    )
    monkeypatch.setattr(evaluator, "resources", lambda: tmp_path)
    c = case('<pitch semitones="3">one</pitch>')
    protocol = DoteBenchEvaluator(Abilities(c.target_text))
    assert protocol.identity()["protocol_version"] == version
    result = protocol.evaluate(request(c))
    assert result.metrics["protocol_version"] == version
    assert protocol.aggregate([result])["protocol_version"] == version


def acoustic_components(xml, source, target, source_f0=100, target_f0=100):
    c = case(xml)
    protocol = DoteBenchEvaluator(Abilities(c.target_text))
    raw = audio().data
    generated = raw + b"\0\0"
    protocol._f0 = lambda audio, spans: [
        stat(source_f0 if audio == raw else target_f0) for _ in spans
    ]
    return protocol._components(
        c,
        raw,
        generated,
        hypothesis=c.target_text,
        source_segments=source,
        target_segments=target,
    )


@pytest.mark.parametrize("shift", [-3, 3])
@pytest.mark.parametrize("missing", ["all", "partial", "unvoiced"])
@pytest.mark.parametrize("mixed", [False, True])
def test_pitch_missing_realization_has_finite_error_and_stable_denominators(
    shift, missing, mixed
):
    xml = f'<pitch semitones="{shift}">one two</pitch>'
    source = [WordSegment("one", 0, 0.4), WordSegment("two", 0.6, 1)]
    target = [WordSegment("one", 0, 0), WordSegment("two", 0.6, 1)]
    if missing == "all":
        target[1] = WordSegment("two", 0.6, 0.6)
    if missing == "unvoiced":
        target = source
    if mixed:
        xml += ' <emo type="happy" level="1">three</emo>'
        source = source + [WordSegment("three", 1.2, 1.5)]
        target = target + [source[-1]]
    rows = acoustic_components(
        xml,
        source,
        target,
        target_f0=None if missing == "unvoiced" else 100 * 2 ** (shift / 12),
    )
    measurement = rows[0]["measurement"]
    achieved = shift / 2 if missing == "partial" else 0
    assert measurement["pitch_shift_semitones"] == pytest.approx(achieved)
    assert measurement["pitch_l1_semitones"] == pytest.approx(abs(shift - achieved))
    assert measurement["measurement_status"] == "missing_target"
    assert rows[0]["success"] is False
    c = case('<emo type="happy" level="1">one</emo>')
    base = DoteBenchEvaluator(Abilities(c.target_text)).evaluate(request(c)).metrics
    summary = summarize_group([{**base, "components": rows}], "compositional")
    assert summary["by_family"]["prosody"]["count"] == 1
    assert summary["component_success_rate"] == (0.5 if mixed else 0)
    assert summary["all_component_success_rate"] == 0
    assert summary["pitch_l1_semitones"] == pytest.approx(abs(shift - achieved))


@pytest.mark.parametrize(
    "kind,attrs", [("pitch", 'semitones="3"'), ("rate", 'factor="2"')]
)
def test_source_zero_word_does_not_require_target_realization(kind, attrs):
    rows = acoustic_components(
        f"<{kind} {attrs}>one two</{kind}>",
        [WordSegment("one", 0, 0), WordSegment("two", 0.2, 1)],
        [WordSegment("one", 0, 0), WordSegment("two", 0.2, 0.5)],
        target_f0=200,
    )
    assert rows[0]["success"] is True


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("missing", ["all", "middle"])
def test_rate_requires_each_source_valid_word_even_with_positive_envelope(
    mixed, missing
):
    xml = '<rate factor="2">one two three</rate>'
    source = [
        WordSegment("one", 0, 0.3),
        WordSegment("two", 0.4, 0.7),
        WordSegment("three", 0.8, 1.2),
    ]
    target = [
        WordSegment("one", 0, 0.1),
        WordSegment("two", 0.2, 0.2),
        WordSegment("three", 0.3, 0.5),
    ]
    if missing == "all":
        target = [WordSegment(s.word, 0, 0) for s in source]
    if mixed:
        xml += ' <emo type="happy" level="1">four</emo>'
        source.append(WordSegment("four", 1.3, 1.6))
        target.append(source[-1])
    rows = acoustic_components(xml, source, target)
    assert rows[0]["success"] is False
    assert rows[0]["measurement"]["duration_ratio"] == pytest.approx(
        0 if missing == "all" else 0.5 / 1.2
    )
    assert rows[0]["measurement"]["duration_l1_sec"] == pytest.approx(
        0.6 if missing == "all" else 0.1
    )
    c = case('<emo type="happy" level="1">one</emo>')
    base = DoteBenchEvaluator(Abilities(c.target_text)).evaluate(request(c)).metrics
    summary = summarize_group([{**base, "components": rows}], "compositional")
    assert summary["component_success_rate"] == (0.5 if mixed else 0)
    assert summary["all_component_success_rate"] == 0


@pytest.mark.parametrize("act", ["ins", "red"])
@pytest.mark.parametrize("missing", ["left", "right", "both"])
@pytest.mark.parametrize("mixed", [False, True])
def test_pause_missing_anchors_fail_with_no_edit_delta(act, missing, mixed):
    source = [WordSegment("one", 0, 0.4), WordSegment("two", 0.9, 1.3)]
    target = [WordSegment("one", 0, 0.2), WordSegment("two", 1, 1.3)]
    for i, side in enumerate(("left", "right")):
        if missing in (side, "both"):
            target[i] = WordSegment(target[i].word, target[i].start, target[i].start)
    xml = f'one <pause act="{act}"/>two'
    if mixed:
        xml += ' <emo type="happy" level="1">three</emo>'
        source.append(WordSegment("three", 1.5, 1.8))
        target.append(source[-1])
    rows = acoustic_components(xml, source, target)
    row = rows[0]
    assert row["success"] is False
    assert row["measurement"]["direction_correct"] is False
    assert row["measurement"]["target_gap_sec"] is None
    assert row["measurement"]["pause_delta_sec"] == 0
    c = case('<emo type="happy" level="1">one</emo>')
    base = DoteBenchEvaluator(Abilities(c.target_text)).evaluate(request(c)).metrics
    summary = summarize_group([{**base, "components": [row]}], "pause")
    assert summary["direction_accuracy"] == 0
    assert (
        summary["component_success_rate"] == summary["all_component_success_rate"] == 0
    )


@pytest.mark.parametrize("overlap", [False, True])
def test_pause_valid_zero_gap_can_succeed(overlap):
    row = acoustic_components(
        'one <pause act="red"/>two',
        [WordSegment("one", 0, 0.4), WordSegment("two", 0.9, 1.3)],
        [WordSegment("one", 0, 0.4), WordSegment("two", 0.3 if overlap else 0.4, 0.8)],
    )[0]
    assert row["success"] is True
    assert row["measurement"]["target_gap_sec"] == 0
    assert row["measurement"]["direction_correct"] is True


@pytest.mark.parametrize("boundary", ["start", "end"])
@pytest.mark.parametrize("missing", [False, True])
def test_boundary_pause_requires_only_existing_anchor(boundary, missing):
    xml = '<pause act="red"/>one' if boundary == "start" else 'one<pause act="red"/>'
    target = (
        WordSegment("one", 0, 0.5)
        if boundary == "start"
        else WordSegment("one", 1.5, 2)
    )
    if missing:
        target = WordSegment("one", target.start, target.start)
    row = acoustic_components(xml, [WordSegment("one", 0.5, 1.5)], [target])[0]
    assert row["success"] is (not missing)
    assert row["measurement"]["direction_correct"] is (not missing)
    assert row["measurement"]["target_gap_sec"] == (None if missing else 0)


@pytest.mark.parametrize(
    "act,source_gap,target_gap,expected",
    [
        ("ins", 0.0, 0.2, True),
        ("ins", 0.0, 0.0, False),
        ("red", 0.0, 0.0, False),
        ("red", 0.0, 0.2, False),
        ("red", 0.4, 0.0, True),
        ("ins", 0.4, 0.6, True),
    ],
)
def test_pause_uses_frozen_source_gap_with_collapsed_source_neighbor(
    act, source_gap, target_gap, expected
):
    row = acoustic_components(
        f'one <pause act="{act}"/>two',
        [WordSegment("one", 0, 0), WordSegment("two", source_gap, source_gap + 0.3)],
        [
            WordSegment("one", 0, 0.3),
            WordSegment("two", 0.3 + target_gap, 0.6 + target_gap),
        ],
    )[0]
    assert row["measurement"]["source_gap_sec"] == source_gap
    assert row["measurement"]["target_gap_sec"] == pytest.approx(target_gap)
    assert row["success"] is expected
    assert row["measurement"]["direction_correct"] is expected


def test_failed_candidate_keeps_frozen_pause_reference_with_collapsed_neighbor():
    from dataclasses import replace
    from dotebench.domain import AlignmentUnit

    c = case('one <pause act="ins"/>two')
    frozen = c.annotations.source_alignment
    source = list(frozen.segments)
    source[0] = AlignmentUnit(source[0].word, source[0].start, source[0].start)
    c = replace(
        c,
        annotations=replace(
            c.annotations, source_alignment=replace(frozen, segments=tuple(source))
        ),
    )
    result = DoteBenchEvaluator(Abilities(c.target_text)).evaluate(
        request(c, "candidate_error")
    )
    row = result.metrics["components"][0]
    assert row["success"] is False
    assert row["measurement"]["source_gap_sec"] == pytest.approx(
        source[1].start - source[0].end
    )
    assert row["measurement"]["direction_correct"] is False


@pytest.mark.parametrize(
    "target_values,expected",
    [([100, 100, 100], 0), ([None, 100, 100], 4), ([None, None, 100], 8)],
)
def test_f0_preservation_source_denominator_and_fallback_provenance(
    target_values, expected
):
    segments = [WordSegment(w, i, i + 1) for i, w in enumerate(("one", "two", "three"))]
    result = compute_f0_preservation(
        source_segments=segments,
        target_segments=segments,
        source_f0_spans=[stat(100), stat(100), stat(None)],
        target_f0_spans=[stat(v) for v in target_values],
    )
    assert result.normalizer == 2
    assert result.wdtw_f0 == expected
    assert result.distance == expected * 2
    assert result.word_results[2]["status"] == "invalid_source_f0"
    for word, value in zip(result.word_results[:2], target_values):
        assert word["target_f0"] == stat(value)
        if value is None:
            assert word["status"] == "penalized_missing_target_f0"
            assert word["word_error_semitones"] == 8
            assert "mean_error_semitones" not in word


@pytest.mark.parametrize("bad", [{}, {"median_f0_hz": "bad"}, stat(math.nan), stat(-1)])
@pytest.mark.parametrize("side", ["source", "target"])
def test_pitch_malformed_f0_is_infrastructure_error(bad, side):
    segment = WordSegment("one", 0, 1)
    with pytest.raises(ValueError, match="F0 service"):
        _measure_pitch_pairs(
            [(segment, segment)],
            "source",
            "target",
            lambda audio, spans: [bad if audio == side else stat(100)],
            3,
        )


@pytest.mark.parametrize("bad", [{}, stat(math.inf), stat("100"), stat(0)])
def test_preservation_malformed_target_f0_is_infrastructure_error(bad):
    segment = WordSegment("one", 0, 1)
    with pytest.raises(ValueError, match="F0 service"):
        compute_f0_preservation(
            source_segments=[segment],
            target_segments=[segment],
            source_f0_spans=[stat(100)],
            target_f0_spans=[bad],
        )


def test_candidate_error_pitch_uses_same_zero_shift_fallback():
    c = case('<pitch semitones="-3">one</pitch>')
    result = DoteBenchEvaluator(Abilities(c.target_text)).evaluate(
        request(c, "candidate_error")
    )
    row = result.metrics["components"][0]
    assert row["success"] is False
    assert row["measurement"]["pitch_shift_semitones"] == 0
    assert row["measurement"]["pitch_l1_semitones"] == 3
