"""XML identity, not repeated token similarity, defines preservation."""

import pytest

from dotebench.alignment import WordSegment
from dotebench.services.backends.wdtw import compute_duration_preservation
from dotebench.evaluation.preservation import select_preserved_segments
from dotebench.instructions import parse_instruction
from dotebench.text import tokenize_text


def segments(text, durations=None):
    words = tokenize_text(text)
    durations = durations or [1] * len(words)
    result, time = [], 0
    for word, duration in zip(words, durations, strict=True):
        result.append(WordSegment(word, time, time + duration))
        time += duration
    return result


def test_deleted_duplicate_does_not_steal_preserved_word():
    instruction = parse_instruction("<del>go </del>go home")
    source, target, count = select_preserved_segments(
        instruction,
        segments(instruction.source_text, [0.2, 0.4, 0.4]),
        segments(instruction.target_text, [0.8, 0.4]),
        operation_kind="pitch",
    )
    assert count == 2
    assert [s.word for s in source] == ["go", "home"]
    assert source[0].start == pytest.approx(0.2)
    result = compute_duration_preservation(source, target)
    assert result["distance"] == pytest.approx(0.4)
    assert result["wdtw_dur"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "xml",
    [
        "<ins>go </ins>go home",
        '<sub targ="go">stop</sub> go home',
        "<del>走</del>走回家",
        "go <del>go </del>home",
    ],
)
def test_repeated_edit_words_have_complete_preservation(xml):
    inst = parse_instruction(xml)
    source, target, count = select_preserved_segments(
        inst,
        segments(inst.source_text),
        segments(inst.target_text),
        operation_kind="rate",
    )
    assert count == (3 if "走" in xml else 2)
    assert [s.word for s in source] == [s.word for s in target]


def test_missing_alignment_unit_is_not_silently_dropped():
    inst = parse_instruction('go go <rate factor="2">home</rate>')
    with pytest.raises(ValueError, match="complete transcript"):
        select_preserved_segments(
            inst,
            segments("go home"),
            segments(inst.target_text),
            operation_kind="rate",
        )


def test_acoustic_preservation_scope_depends_on_operation():
    inst = parse_instruction(
        'go <pitch semitones="3">high</pitch> <rate factor="2">quickly</rate> home'
    )
    alignment = segments(inst.source_text)
    pitch_scope, _, _ = select_preserved_segments(
        inst, alignment, alignment, operation_kind="pitch"
    )
    rate_scope, _, _ = select_preserved_segments(
        inst, alignment, alignment, operation_kind="rate"
    )
    assert [segment.word for segment in pitch_scope] == ["go", "high", "home"]
    assert [segment.word for segment in rate_scope] == ["go", "quickly", "home"]


def test_unknown_preservation_operation_is_rejected():
    inst = parse_instruction("go home")
    alignment = segments(inst.source_text)
    with pytest.raises(ValueError, match="Unsupported preservation operation"):
        select_preserved_segments(inst, alignment, alignment, operation_kind="emo")


def test_timestamp_grid_is_bounded_without_dropping_units():
    from dotebench.alignment import bound_segments

    raw = [
        {"word": "out", "start": 12.48, "end": 12.64},
        {"word": "at", "start": 12.64, "end": 12.64},
    ]
    result = bound_segments(raw, 12.52)
    assert result == [
        {"word": "out", "start": 12.48, "end": 12.52},
        {"word": "at", "start": 12.52, "end": 12.52},
    ]
    assert raw[0]["end"] == 12.64
    with pytest.raises(ValueError, match="precedes"):
        bound_segments([{"word": "x", "start": 2, "end": 1}], 1)


def test_local_wer_evaluates_a_failed_deletion():
    from dotebench.evaluation.wer import compute_instruction_wer

    inst = parse_instruction("please <del>delete </del>this word now")
    failed = compute_instruction_wer(inst, inst.source_text)
    assert failed.edit_region_wer.insertions == 1
    assert failed.edit_region_wer.ref_length == 4
    assert failed.edit_region_wer.error_rate == 0.25
    assert (
        compute_instruction_wer(inst, inst.target_text).edit_region_wer.error_rate == 0
    )


def test_local_wer_insertion_is_owned_by_exactly_one_partition():
    import jiwer

    from dotebench.evaluation.wer import _compute_region_wer_from_alignment

    ref, hyp = ["one", "two"], ["one", "extra", "two"]
    output = jiwer.process_words(" ".join(ref), " ".join(hyp))
    left = _compute_region_wer_from_alignment(output, ref, hyp, {0})
    right = _compute_region_wer_from_alignment(output, ref, hyp, {1})
    assert left.insertions == 0
    assert right.insertions == 1
    assert left.insertions + right.insertions == output.insertions


def test_xml_deletion_boundary_defines_local_context():
    from dotebench.evaluation.wer import compute_instruction_wer

    inst = parse_instruction("a b c d <del>x </del>x e f g h")
    result = compute_instruction_wer(inst, inst.target_text, context_k=1)
    assert result.local_ref_text == "d x"
    assert result.local_window_indices == [3, 4]
