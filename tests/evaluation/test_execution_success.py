"""Directional execution and requested-magnitude success contracts."""

import pytest

from dotebench.alignment import WordSegment
from dotebench.evaluation.evaluator import (
    DoteBenchEvaluator,
    _component_execution,
    _component_success,
    summarize_category,
    summarize_group,
)
from tests.evaluation.test_aggregation import row
from tests.evaluation.test_protocol import Abilities, case, request
from tests.evaluation.test_scoring import acoustic_components


@pytest.mark.parametrize("requested", [-4, -1, 1, 4])
@pytest.mark.parametrize(
    "fraction,success",
    [
        (-1, False),
        (0, False),
        (0.5999, False),
        (0.6, True),
        (1, True),
        (1.6, True),
        (1.6001, False),
        (3, False),
    ],
)
def test_pitch_success_bounds_requested_signed_shift(requested, fraction, success):
    component = {
        "family": "prosody",
        "operation": "pitch",
        "params": {"semitones": requested},
    }
    measurement = {"pitch_shift_semitones": requested * fraction}
    assert _component_success(component, measurement) is success
    assert not success or _component_execution(component, measurement)


@pytest.mark.parametrize("factor", [1.04, 1.25, 2, 0.98, 0.8, 0.5])
@pytest.mark.parametrize(
    "fraction,success",
    [
        (-0.1, False),
        (0, False),
        (0.5999, False),
        (0.6, True),
        (1, True),
        (1.6, True),
        (1.6001, False),
    ],
)
def test_rate_success_bounds_speed_change(factor, fraction, success):
    component = {"family": "prosody", "operation": "rate", "params": {"factor": factor}}
    speed = 1 + (factor - 1) * fraction
    measurement = {"duration_ratio": 1 / speed}
    assert _component_success(component, measurement) is success
    assert not success or _component_execution(component, measurement)


@pytest.mark.parametrize(
    "factor,speed,success",
    [
        (1.25, 1.15, True),
        (1.25, 1.4, True),
        (1.25, 1.4001, False),
        (0.8, 0.68, True),
        (0.8, 0.88, True),
        (0.8, 0.67, False),
        (0.8, 0.89, False),
    ],
)
def test_rate_bounds_in_speed_units(factor, speed, success):
    component = {"family": "prosody", "operation": "rate", "params": {"factor": factor}}
    measurement = {"duration_ratio": 1 / speed}
    assert _component_success(component, measurement) is success
    assert not success or _component_execution(component, measurement)


@pytest.mark.parametrize(
    "operation,params,key,actual,executed",
    [
        ("pitch", {"semitones": 4}, "pitch_shift_semitones", 0.9999, False),
        ("pitch", {"semitones": 4}, "pitch_shift_semitones", 1, True),
        ("pitch", {"semitones": -4}, "pitch_shift_semitones", -0.9999, False),
        ("pitch", {"semitones": -4}, "pitch_shift_semitones", -1, True),
        ("rate", {"factor": 1.25}, "duration_ratio", 0.9001, False),
        ("rate", {"factor": 1.25}, "duration_ratio", 0.9, True),
        ("rate", {"factor": 0.8}, "duration_ratio", 1.0999, False),
        ("rate", {"factor": 0.8}, "duration_ratio", 1.1, True),
    ],
)
def test_execution_thresholds(operation, params, key, actual, executed):
    component = {"family": "prosody", "operation": operation, "params": params}
    assert _component_execution(component, {key: actual}) is executed


@pytest.mark.parametrize("requested", [-1, 1])
@pytest.mark.parametrize(
    "magnitude,executed", [(0.5999, False), (0.6, True), (2, True)]
)
def test_pitch_execution_minimum_follows_small_request(requested, magnitude, executed):
    component = {
        "family": "prosody",
        "operation": "pitch",
        "params": {"semitones": requested},
    }
    assert (
        _component_execution(
            component, {"pitch_shift_semitones": requested * magnitude}
        )
        is executed
    )


@pytest.mark.parametrize("factor", [1.04, 0.98])
@pytest.mark.parametrize(
    "fraction,executed", [(0.5999, False), (0.6, True), (1.7, True)]
)
def test_rate_execution_minimum_follows_small_request(factor, fraction, executed):
    component = {"family": "prosody", "operation": "rate", "params": {"factor": factor}}
    speed = 1 + fraction * (factor - 1)
    assert _component_execution(component, {"duration_ratio": 1 / speed}) is executed


@pytest.mark.parametrize("requested", [-1e10, 1e10])
@pytest.mark.parametrize(
    "effect,executed", [(-1, False), (0, False), (0.5, False), (1, True)]
)
def test_large_pitch_request_keeps_fixed_execution_minimum(requested, effect, executed):
    component = {
        "family": "prosody",
        "operation": "pitch",
        "params": {"semitones": requested},
    }
    actual = effect if requested > 0 else -effect
    assert (
        _component_execution(component, {"pitch_shift_semitones": actual}) is executed
    )


@pytest.mark.parametrize(
    "ratio,executed", [(1.1, False), (1, False), (0.95, False), (0.9, True)]
)
def test_large_rate_request_keeps_fixed_execution_minimum(ratio, executed):
    component = {"family": "prosody", "operation": "rate", "params": {"factor": 1e10}}
    assert _component_execution(component, {"duration_ratio": ratio}) is executed


@pytest.mark.parametrize("ratio", [None, 0, -1])
def test_invalid_target_duration_fails_both_outcomes(ratio):
    component = {"family": "prosody", "operation": "rate", "params": {"factor": 1.25}}
    measurement = {"duration_ratio": ratio}
    assert _component_execution(component, measurement) is False
    assert _component_success(component, measurement) is False


@pytest.mark.parametrize(
    "operation,params,measurement",
    [
        ("pitch", {"semitones": 0}, {"pitch_shift_semitones": 0}),
        ("rate", {"factor": 1}, {"duration_ratio": 1}),
    ],
)
def test_no_requested_change_has_no_edit_outcome(operation, params, measurement):
    component = {"family": "prosody", "operation": operation, "params": params}
    assert _component_execution(component, measurement) is False
    assert _component_success(component, measurement) is False


@pytest.mark.parametrize(
    "operation,params,measurement,executed,success",
    [
        ("pitch", {"semitones": 4}, {"pitch_shift_semitones": 1}, True, False),
        ("pitch", {"semitones": -4}, {"pitch_shift_semitones": -8}, True, False),
        ("rate", {"factor": 1.04}, {"duration_ratio": 1 / 1.04}, True, True),
        ("rate", {"factor": 0.98}, {"duration_ratio": 1 / 0.98}, True, True),
        ("rate", {"factor": 1.25}, {"duration_ratio": 0.5}, True, False),
        ("rate", {"factor": 0.8}, {"duration_ratio": 2}, True, False),
    ],
)
def test_execution_and_success_outcomes(
    operation, params, measurement, executed, success
):
    component = {"family": "prosody", "operation": operation, "params": params}
    assert _component_execution(component, measurement) is executed
    assert _component_success(component, measurement) is success
    assert not success or _component_execution(component, measurement)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("operation", ["pitch", "rate"])
@pytest.mark.parametrize("missing", ["none", "partial", "all"])
def test_realization_and_denominators_apply_to_both_rates(mixed, operation, missing):
    attrs = 'semitones="4"' if operation == "pitch" else 'factor="1.25"'
    xml = f"<{operation} {attrs}>one two</{operation}>"
    source = [WordSegment("one", 0, 0.4), WordSegment("two", 0.6, 1)]
    target = [WordSegment("one", 0, 0.32), WordSegment("two", 0.48, 0.8)]
    if missing != "none":
        target[0] = WordSegment("one", 0, 0)
    if missing == "all":
        target[1] = WordSegment("two", 0, 0)
    if mixed:
        xml += ' <emo type="happy">three</emo>'
        source.append(WordSegment("three", 1.2, 1.5))
        target.append(source[-1])
    # A partial pitch realization retains a numerically sufficient mean shift;
    # the incomplete target evidence must still fail the operation.
    components = acoustic_components(xml, source, target, target_f0=100 * 2 ** (6 / 12))
    base = row("prosody", [], 0, 2, 0, 2)
    base["components"] = components
    category = "compositional" if mixed else "prosody"
    result = summarize_group([base], category)
    expected = missing == "none"
    for metric, field in (("execution", "executed"), ("success", "success")):
        assert components[0][field] is expected
        assert result[f"all_component_{metric}_rate"] == float(expected)
        assert result[f"component_{metric}_rate"] == (float(expected) + mixed) / (
            1 + mixed
        )
        assert result["by_operation"][operation][f"{metric}_rate"] == float(expected)
    assert result["by_operation"][operation]["count"] == 1
    assert result["by_family"]["prosody"]["count"] == 1


def test_dual_outcomes_pool_operations_cases_and_shards():
    first = row("prosody", [False], 0, 2, 0, 2)
    second = row("prosody", [True, True], 0, 2, 0, 2)
    for record, shard in ((first, "a"), (second, "b")):
        record["group"] = {"category": "compositional", "shard": shard}
    first["components"][0].update(operation="pitch", executed=True)
    first["components"][0]["measurement"]["pitch_l1_semitones"] = 4
    for component in second["components"]:
        component.update(operation="rate", executed=True)
        component["measurement"]["duration_l1_sec"] = 0
    result = summarize_category([first, second], "compositional")
    assert result["component_execution_rate"] == 1
    assert result["component_success_rate"] == pytest.approx(2 / 3)
    assert result["all_component_execution_rate"] == 1
    assert result["all_component_success_rate"] == 0.5
    assert result["by_operation"]["pitch"] == {
        "count": 1,
        "execution_rate": 1,
        "success_rate": 0,
    }
    assert result["by_operation"]["rate"] == {
        "count": 2,
        "execution_rate": 1,
        "success_rate": 1,
    }
    assert result["by_family"]["prosody"]["execution_rate"] == 1
    assert result["by_family"]["prosody"]["success_rate"] == pytest.approx(2 / 3)
    assert result["by_shard"]["a"]["component_execution_rate"] == 1
    assert result["by_shard"]["b"]["component_success_rate"] == 1


@pytest.mark.parametrize("field", ["executed", "success"])
@pytest.mark.parametrize("value", [None, 1])
def test_aggregation_requires_both_boolean_outcomes(field, value):
    record = row("prosody", [False], 0, 2, 0, 2)
    record["components"][0][field] = value
    with pytest.raises(ValueError, match=f"boolean {field}"):
        summarize_group([record], "prosody")


def test_failed_candidate_retains_both_rate_denominators():
    sample = case(
        '<pitch semitones="4">one</pitch> <rate factor="1.25">two</rate> '
        '<emo type="happy">three</emo>'
    )
    evaluator = DoteBenchEvaluator(Abilities(sample.target_text))
    result = evaluator.evaluate(request(sample, "candidate_error"))
    assert all(
        c["executed"] is False and c["success"] is False
        for c in result.metrics["components"]
    )
    summary = evaluator.aggregate([result])["by_category"]["compositional"]
    assert summary["component_execution_rate"] == summary["component_success_rate"] == 0
    assert (
        summary["all_component_execution_rate"]
        == summary["all_component_success_rate"]
        == 0
    )
    assert summary["by_family"]["prosody"]["count"] == 2
