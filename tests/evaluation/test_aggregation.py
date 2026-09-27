"""Frozen reference outcomes for the released scoring conventions."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dotebench.alignment import WordSegment
from dotebench.domain import (
    AlignmentUnit,
    Annotations,
    AudioAsset,
    Case,
    SourceAlignment,
)
from dotebench.evaluation.evaluator import DoteBenchEvaluator, summarize_group
from dotebench.services.backends.wer import compute_error_rate
from dotebench.instructions import parse_instruction
from dotebench.models.base import Audio, GenerationResult


def row(family, successes, errors, length, distance, normalizer):
    reference = " ".join(f"w{i}" for i in range(length))
    hypothesis = " ".join(f"w{i}" for i in range(errors, length))
    wer = compute_error_rate(reference, hypothesis, "en").to_dict()
    components = [
        {
            "family": family,
            "operation": family,
            "success": success,
            "measurement": {"direction_correct": success, "pause_error_ms": 100.0},
        }
        for success in successes
    ]
    return {
        "wer": wer,
        "local_wer": {"edit_region_wer": wer, "non_edit_region_wer": wer},
        "speaker_similarity": 0.8,
        "utmos": 3.0,
        "components": components,
        "wdtw_dur": {
            "wdtw_dur": distance / normalizer,
            "distance": distance,
            "normalizer": normalizer,
        },
        "wdtw_f0": {
            "wdtw_f0": distance / normalizer,
            "distance": distance,
            "normalizer": normalizer,
        },
    }


@pytest.mark.parametrize("category", ["text", "emotion", "pause", "prosody"])
def test_preservation_uses_corpus_denominators(category):
    result = summarize_group(
        [row(category, [True], 1, 2, 1, 2), row(category, [True], 0, 8, 0, 8)], category
    )
    assert result["wdtw_dur"]["value"] == pytest.approx(10.0)
    assert result["wdtw_f0"]["value"] == pytest.approx(0.1)


def test_compositional_preservation_uses_corpus_denominators():
    result = summarize_group(
        [row("text", [True], 1, 2, 1, 2), row("text", [True], 0, 8, 0, 8)],
        "compositional",
    )
    assert result["wdtw_dur"]["value"] == pytest.approx(10.0)
    assert result["wdtw_f0"]["value"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "category,expected",
    [
        ("text", 10.0),
        ("emotion", 10.0),
        ("pause", 10.0),
        ("prosody", 10.0),
        ("compositional", 10.0),
    ],
)
def test_category_content_weighting(category, expected):
    result = summarize_group(
        [row(category, [True], 1, 2, 1, 2), row(category, [True], 0, 8, 0, 8)], category
    )
    assert result["asr_error"] == pytest.approx(expected)
    if category == "text":
        assert result["edit_region_wer"] == pytest.approx(10.0)
        assert result["non_edit_region_wer"] == pytest.approx(10.0)


def test_prosody_effect_mean_retains_precision_and_equal_operation_weights():
    first = row("prosody", [False], 0, 2, 0, 2)
    second = row("prosody", [False, False], 0, 2, 0, 2)
    first["components"][0]["operation"] = "rate"
    first["components"][0]["measurement"]["duration_l1_sec"] = 0.1234567
    for component in second["components"]:
        component["operation"] = "rate"
        component["measurement"]["duration_l1_sec"] = 0.7654321
    result = summarize_group([first, second], "prosody")
    assert result["duration_l1_sec"] == pytest.approx((0.1234567 + 2 * 0.7654321) / 3)


@pytest.mark.parametrize(
    "category,key", [("emotion", "emotion_accuracy"), ("pause", "direction_accuracy")]
)
def test_region_accuracy_pools_regions(category, key):
    result = summarize_group(
        [row(category, [True], 0, 2, 0, 2), row(category, [False, False], 0, 8, 0, 8)],
        category,
    )
    assert result[key] == pytest.approx(100 / 3)


def make_case(xml, annotations=None, language="en"):
    if annotations is None:
        annotations = Annotations()
    parsed = parse_instruction(xml)
    import hashlib
    from dataclasses import replace

    from dotebench.text import tokenize_text

    segments = (
        annotations.source_alignment.segments
        if annotations.source_alignment
        else tuple(
            AlignmentUnit(word, i * 0.25, i * 0.25 + 0.2)
            for i, word in enumerate(tokenize_text(parsed.source_text))
        )
    )
    annotations = replace(
        annotations,
        source_alignment=SourceAlignment(
            hashlib.sha256(wav()).hexdigest(),
            hashlib.sha256(parsed.source_text.encode()).hexdigest(),
            segments,
        ),
    )
    return Case(
        "fixture",
        "en" if language == "en" else "zh",
        AudioAsset("audio.wav", "0" * 64),
        xml,
        annotations,
        parsed,
    )


def wav():
    output = io.BytesIO()
    sf.write(output, np.zeros(32000), 16000, format="WAV")
    return output.getvalue()


class Abilities:
    def identity(self):
        return {"fixture": "metrics"}

    def __init__(self, query_answers=()):
        self.calls = []
        self.query_answers = iter(query_answers)

    def measure(self, name, **payload):
        self.calls.append((name, payload))
        if name == "qwen3_aligner":
            answers = next(self.query_answers)
            assert len(payload["queries"]) == len(answers)
            assert all("<q>" in q and "</q>" in q for q in payload["queries"])
            return {"answers": [{"start": a, "end": b} for a, b in answers]}
        if name == "emotion":
            return {"predicted_emotion": "happy"}
        if name == "f0":
            return {"spans": [{"median_f0_hz": None} for _ in payload["spans"]]}
        raise AssertionError(name)


def test_pause_query_retains_the_selected_occurrence_of_a_repeated_word():
    case = make_case('Keep it, and pray to it,<pause act="ins"/> that it works.')
    from dotebench.alignment import query_intervals

    words = ["Keep", "it", "and", "pray", "to", "it", "that", "it", "works"]
    segments = [
        WordSegment(w.lower(), i * 0.5, i * 0.5 + 0.2) for i, w in enumerate(words)
    ]
    segments[5] = WordSegment("it", 4.24, 4.4)
    segments[6:] = [
        WordSegment("that", 4.64, 4.8),
        WordSegment("it", 5, 5.2),
        WordSegment("works", 5.4, 5.8),
    ]
    intervals = query_intervals(
        segments,
        case.target_text,
        list(enumerate(case.instruction.operations)),
        side="target",
        audio_duration=6,
    )
    assert intervals[0] == (4.4, 4.64)


def test_target_query_coordinates_follow_normalized_xml_projection():
    case = make_case(
        'For services <del>promptly rendered</del>, please <rate factor="0.8">speak softly</rate>.'
    )
    assert case.target_text == "For services, please speak softly."
    index, operation = next(
        (i, op) for i, op in enumerate(case.instruction.operations) if op.kind == "rate"
    )
    assert (
        case.target_text[operation.target.start : operation.target.end]
        == "speak softly"
    )
    from dotebench.alignment import query_intervals

    segments = [
        WordSegment(w, i * 0.25, (i + 1) * 0.25)
        for i, w in enumerate(["for", "services", "please", "speak", "softly"])
    ]
    intervals = query_intervals(
        segments,
        case.target_text,
        [(index, operation)],
        side="target",
        audio_duration=2,
    )
    assert intervals[index] == (0.75, 1.25)


def test_whole_emotion_uses_complete_audio():
    case = make_case('<emo type="happy" level="2">hello</emo>')
    ability = Abilities()
    protocol = DoteBenchEvaluator(ability)
    segments = [WordSegment("hello", 0.5, 1.0)]
    protocol._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=segments,
        target_segments=segments,
    )
    name, payload = ability.calls[-1]
    assert name == "emotion"
    assert payload["input_strategy"] == "whole_audio"
    assert (payload["start"], payload["end"]) == (0.0, 2.0)
    assert not any(name == "qwen3_aligner" for name, _ in ability.calls)


def test_local_emotion_uses_padded_crop():
    case = make_case('say <emo type="happy" level="2">hello</emo>')
    ability = Abilities([[(0.5, 1.0)]])
    protocol = DoteBenchEvaluator(ability)
    segments = [WordSegment("say", 0.1, 0.3), WordSegment("hello", 0.5, 1.0)]
    protocol._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=segments,
        target_segments=segments,
    )
    payload = ability.calls[-1][1]
    assert payload["input_strategy"] == "padded_crop_250ms"
    assert (payload["start"], payload["end"]) == (0.5, 1.0)


def test_compositional_unvoiced_pitch_is_unmeasurable():
    case = make_case(
        '<pitch semitones="3">hello</pitch> <emo type="happy" level="2">world</emo>',
        Annotations(
            source_alignment=SourceAlignment(
                "0" * 64, "0" * 64, (AlignmentUnit("hello", 0.1, 0.5),)
            )
        ),
    )
    segments = [WordSegment("hello", 0.1, 0.5), WordSegment("world", 0.7, 1.0)]
    rows = DoteBenchEvaluator(
        Abilities([[(0.1, 0.5), (0.7, 1.0)], [(0.1, 0.5)]])
    )._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=segments,
        target_segments=segments,
    )
    assert rows[0]["success"] is None
    assert rows[0]["measurement"]["pitch_shift_semitones"] is None
    assert rows[0]["measurement"]["pitch_l1_semitones"] is None


@pytest.mark.parametrize(
    "category,expected", [("prosody", 0.2), ("compositional", 0.2)]
)
def test_rate_failure_category_policy(category, expected):
    case = make_case(
        '<rate factor="2">hello</rate>',
        Annotations(
            source_alignment=SourceAlignment(
                "0" * 64, "0" * 64, (AlignmentUnit("hello", 0.1, 0.5),)
            )
        ),
    )
    ability = Abilities([[(0.1, 0.5)]] if category == "prosody" else [])
    protocol = DoteBenchEvaluator(
        ability, case_groups={case.id: {"category": category, "shard": "fixture"}}
    )
    request = SimpleNamespace(
        generation=GenerationResult(case.id, "candidate_error", error="failed"),
        source_audio=Audio(wav()),
    )
    result = protocol._penalty(case, request)
    assert result["components"][0]["measurement"]["duration_l1_sec"] == pytest.approx(
        expected
    )
    assert [name for name, _ in ability.calls] == ["f0"]


def test_standalone_pause_uses_measured_source_gap():
    case = make_case(
        'hello <pause act="ins"/>world',
        Annotations(),
    )
    source = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.6, 1.0)]
    target = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.9, 1.3)]
    answers = [[(s.start, s.end) for s in target], [(s.start, s.end) for s in source]]
    result = DoteBenchEvaluator(Abilities(answers))._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=source,
        target_segments=target,
    )[0]["measurement"]
    assert result["source_gap_sec"] == pytest.approx(0.2)
    assert result["target_gap_sec"] == pytest.approx(0.5)
    assert result["pause_delta_sec"] == pytest.approx(0.3)
    assert result["direction_correct"] is True


@pytest.mark.parametrize(
    "act,target_gap,expected",
    [
        ("ins", 0.559, False),
        ("ins", 0.56, True),
        ("red", 0.241, False),
        ("red", 0.24, True),
    ],
)
def test_pause_direction_requires_160ms_change(act, target_gap, expected):
    case = make_case(f'hello <pause act="{act}"/>world', Annotations())
    source = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.8, 1.0)]
    target = [
        WordSegment("hello", 0.1, 0.4),
        WordSegment("world", 0.4 + target_gap, 1.0 + target_gap),
    ]
    answers = [[(s.start, s.end) for s in target], [(s.start, s.end) for s in source]]
    component = DoteBenchEvaluator(Abilities(answers))._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=source,
        target_segments=target,
    )[0]
    assert component["measurement"]["direction_correct"] is expected
    assert component["success"] is expected


def test_pause_overlapping_alignment_clamps_gap():
    case = make_case(
        'hello <pause act="ins"/>world',
        Annotations(),
    )
    source = [WordSegment("hello", 0.1, 0.5), WordSegment("world", 0.4, 1.0)]
    target = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.6, 1.0)]
    answers = [[(s.start, s.end) for s in target], [(s.start, s.end) for s in source]]
    result = DoteBenchEvaluator(Abilities(answers))._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=source,
        target_segments=target,
    )[0]["measurement"]
    assert result["source_gap_sec"] == 0.0
    assert result["pause_delta_sec"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "category,language",
    [
        ("emotion", "zh"),
        ("text", "auto"),
        ("pause", "auto"),
        ("prosody", "auto"),
        ("compositional", "auto"),
    ],
)
def test_global_wer_language_policy(category, language):
    case = make_case('<emo type="happy" level="2">你好 world</emo>', language="zh")

    class StopAfterWER(Exception):
        pass

    class Recorder:
        def __init__(self):
            self.calls = []

        def measure(self, name, **payload):
            self.calls.append((name, payload))
            if name == "qwen3_asr":
                return {"text": "你好 world"}
            if name == "wer":
                raise StopAfterWER
            raise AssertionError(name)

    ability = Recorder()
    protocol = DoteBenchEvaluator(
        ability, case_groups={case.id: {"category": category, "shard": "fixture"}}
    )
    request = SimpleNamespace(
        case=case,
        generation=GenerationResult(case.id, "ok"),
        source_audio=Audio(wav()),
        generated_audio=Audio(wav()),
    )
    with pytest.raises(StopAfterWER):
        protocol.evaluate(request)
    assert ability.calls[-1][1]["language"] == language


def test_prosody_failure_uses_mixed_reference_units():
    case = make_case('<pitch semitones="3">你好 world</pitch>', language="zh")
    protocol = DoteBenchEvaluator(Abilities())
    request = SimpleNamespace(
        generation=GenerationResult(case.id, "candidate_error", error="failed"),
        source_audio=Audio(wav()),
    )
    result = protocol._penalty(case, request)
    assert result["wer"]["ref_length"] == 3
    assert result["wdtw_dur"]["normalizer"] == pytest.approx(0.6)
    assert result["wdtw_f0"]["normalizer"] == 0


def test_compositional_failure_uses_xml_partitioned_target_reference():
    case = make_case(
        '<sub targ="one two three">one</sub> <emo type="happy" level="2">four</emo>'
    )
    protocol = DoteBenchEvaluator(Abilities())
    request = SimpleNamespace(
        generation=GenerationResult(case.id, "candidate_error", error="failed"),
        source_audio=Audio(wav()),
    )
    result = protocol._penalty(case, request)
    assert result["local_wer"]["edit_region_wer"]["ref_length"] == 4
    assert result["local_wer"]["non_edit_region_wer"]["ref_length"] == 0


def test_compositional_effect_means_pool_components():
    first = row("prosody", [True], 0, 2, 0, 2)
    second = row("prosody", [False, False], 0, 8, 0, 8)
    for component in first["components"]:
        component["operation"] = "pitch"
        component["measurement"]["pitch_l1_semitones"] = 0.0
    for component in second["components"]:
        component["operation"] = "pitch"
        component["measurement"]["pitch_l1_semitones"] = 3.0
    result = summarize_group([first, second], "compositional")
    assert result["pitch_l1_semitones"] == 2.0


@pytest.mark.parametrize(
    "text,start,end,expected",
    [
        ("hello", 1, 4, {0}),
        ("rock-n-roll", 4, 5, {0, 1}),
        ("John's", 4, 5, {0, 1}),
        ("你好world", 1, 4, {1, 2}),
        ("hello", 2, 2, set()),
    ],
)
def test_character_edit_masks_preserve_reference_token_overlap(
    text, start, end, expected
):
    from dotebench.alignment import token_indices
    from dotebench.instructions import Span

    assert token_indices(text, Span(start, end)) == expected


def test_edit_boundaries_use_shared_alignment():
    case = make_case('say <emo type="happy" level="2">hello</emo>')
    ability = Abilities([[(0.6, 0.9)]])
    full = [WordSegment("say", 0.1, 0.3), WordSegment("hello", 0.5, 1.0)]
    DoteBenchEvaluator(ability)._components(
        case, wav(), wav(), hypothesis=None, source_segments=full, target_segments=full
    )
    assert not any(name == "qwen3_aligner" for name, _ in ability.calls)
    emotion_payload = next(
        payload for name, payload in ability.calls if name == "emotion"
    )
    assert (emotion_payload["start"], emotion_payload["end"]) == (0.5, 1.0)


def test_compositional_pause_uses_verified_reference_gap():
    case = make_case(
        'hello <pause act="ins"/>world',
        Annotations(),
    )
    ability = Abilities([[(0.1, 0.4), (0.9, 1.3)]])
    full = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.6, 1.0)]
    protocol = DoteBenchEvaluator(
        ability,
        case_groups={case.id: {"category": "compositional", "shard": "fixture"}},
    )
    target = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.9, 1.3)]
    result = protocol._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=full,
        target_segments=target,
    )[0]["measurement"]
    assert result["source_gap_sec"] == pytest.approx(0.2)
    assert result["target_gap_sec"] == 0.5
    assert len([name for name, _ in ability.calls if name == "qwen3_aligner"]) == 0


def test_pause_gap_is_derived_from_source_alignment():
    case = make_case(
        'hello <pause act="ins"/>world',
        Annotations(),
    )
    source = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.6, 1)]
    target = [WordSegment("hello", 0.1, 0.4), WordSegment("world", 0.9, 1.3)]
    ability = Abilities([[(s.start, s.end) for s in target]])
    result = DoteBenchEvaluator(ability)._components(
        case,
        wav(),
        wav(),
        hypothesis=None,
        source_segments=source,
        target_segments=target,
    )[0]["measurement"]
    assert result["source_gap_sec"] == pytest.approx(0.2)
    assert result["pause_delta_sec"] == pytest.approx(0.3)
    assert sum(a == "qwen3_aligner" for a, p in ability.calls) == 0
