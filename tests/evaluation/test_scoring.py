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
from dotebench.services.backends.wdtw import measure_spans
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


def test_zero_target_contributes_duration_error_but_not_f0():
    result, calls = preservation(
        'a<pause act="ins"/>',
        [WordSegment("a", 0, 1)],
        [WordSegment("a", 0, 0)],
    )
    assert result["wdtw_dur"]["wdtw_dur"] == 1
    assert result["wdtw_f0"]["wdtw_f0"] is None
    assert result["wdtw_f0"]["normalizer"] == 0
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
    "source,target,expected",
    [
        ([100, None], [200, 100], 12),
        ([100, 200], [200, 100], 0),
        ([100, None], [None, 100], None),
        ([None, None], [100, 100], None),
        ([math.nan, 100], [200, math.inf], None),
    ],
)
def test_pitch_uses_only_colocated_measurable_words(source, target, expected):
    seg = [WordSegment("a", 0, 1), WordSegment("b", 1, 2)]

    def probe(audio, spans):
        return [stat(v) for v in (source if audio == "source" else target)]

    result = _measure_pitch_pairs(list(zip(seg, seg)), "source", "target", probe, 3)
    assert result["pitch_shift_semitones"] == expected
    assert result["pitch_l1_semitones"] == (
        abs(expected - 3) if expected is not None else None
    )
    assert result["measurement_status"] == (
        "ok" if expected is not None else "unmeasurable"
    )


def test_pitch_zero_duration_never_calls_f0():
    result = _measure_pitch_pairs(
        [(WordSegment("a", 0, 0), WordSegment("a", 0, 1))],
        b"",
        b"",
        lambda *_: pytest.fail("zero source was probed"),
        2,
    )
    assert result["measurement_status"] == "unmeasurable"


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
def test_unmeasurable_pitch_excluded_at_operation_and_sample_levels(xml):
    c = case(xml)

    class Unvoiced(Abilities):
        def measure(self, ability, **payload):
            if ability == "f0":
                return {"spans": [stat(None) for _ in payload["spans"]]}
            return super().measure(ability, **payload)

    protocol = DoteBenchEvaluator(Unvoiced(c.target_text))
    result = protocol.evaluate(request(c))
    assert result.metrics["components"][0]["success"] is None
    summary = summarize_group([result.metrics], "compositional")
    assert summary["pitch_l1_semitones"] is None
    assert summary["all_component_success_rate"] == (1 if "emo" in xml else None)
    assert "prosody" not in summary["by_family"]


def test_rate_zero_target_retains_numeric_zero_and_faster_success():
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
    assert rows[0]["success"] is True


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
    old = EvaluationResult(
        c.id, {**result.metrics, "protocol_version": "dotebench-formal-v4"}
    )
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
