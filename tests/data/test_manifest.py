import dataclasses
import json

import pytest

from dotebench import dataset
from dotebench.dataset import load_cases, manifest_path, select_shards
from tests.manifest_fixture import write_manifest


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "resources", lambda: tmp_path)

    def create(category="text"):
        return write_manifest(tmp_path, category=category)

    return create


def test_release_version_comes_from_index(manifest, tmp_path):
    entry, _, path = manifest()
    index = dataset.release_index()
    entries = select_shards()

    assert dataset.data_version() == "fixture_v1"
    assert entries == [entry]
    assert manifest_path(entry) == path
    assert len(load_cases(entry, verify_audio=False)) == entry["case_count"]

    from dotebench.materialization.manifest import load_inventory
    from dotebench.materialization.verification import canonical_manifest_hashes

    inventory = load_inventory(data_root=tmp_path)
    assert inventory.entries == tuple(entries)
    assert set(canonical_manifest_hashes()) == set(index["manifests"])


def test_loaded_case_uses_xml_derived_transcripts_and_is_immutable(manifest):
    entry, _, _ = manifest()
    case = load_cases(entry, verify_audio=False)[0]
    assert (case.source_text, case.target_text) == ("hello", "world")
    assert case.instruction.source_text == case.source_text
    assert case.instruction.target_text == case.target_text
    assert len(case.instruction.operations) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        case.id = "changed"


@pytest.mark.parametrize(
    "mutation",
    [
        "renamed_instruction",
        "unknown",
        "duplicate",
        "annotation",
        "path",
        "checksum",
        "transcript",
    ],
)
def test_reject_invalid_manifest(manifest, mutation):
    entry, payload, path = manifest()
    case = payload["cases"][0]
    if mutation == "renamed_instruction":
        case["instruction_st"] = case.pop("instruction_xml")
    elif mutation == "unknown":
        payload["benchmark"] = "doteBench"
    elif mutation == "duplicate":
        payload["cases"].append(dict(case))
        entry["case_count"] = 2
    elif mutation == "annotation":
        case["annotations"]["operations"] = []
    elif mutation == "path":
        case["source_audio"]["path"] = "../secret.wav"
    elif mutation == "checksum":
        case["source_audio"]["sha256"] = "oops"
    else:
        case["target_text"] = "unexpected independent transcript"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(entry, verify_audio=False)


@pytest.mark.parametrize(
    "mutation", ["status", "boundary_type", "missing", "null", "unnecessary_interval"]
)
def test_annotation_whitelist_rejects_nonmetric_fields(manifest, mutation):
    entry, payload, path = manifest("compositional")
    annotation = payload["cases"][0]["annotations"]
    if mutation in {"status", "boundary_type"}:
        annotation[mutation] = "verified"
    elif mutation == "missing":
        del annotation["source_alignment"]
    elif mutation == "null":
        annotation["source_alignment"] = None
    else:
        annotation["temporal"] = [
            {"operation_index": 0, "start_sec": 0.0, "end_sec": 1.0}
        ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(entry, verify_audio=False)


@pytest.mark.parametrize("value", [None, "phrase", 1, [], {"label": "clause"}])
@pytest.mark.parametrize(
    "category,field",
    [("prosody", "span_granularity"), ("emotion", "source_emotion")],
)
def test_grouping_annotation_requires_declared_enum(manifest, value, category, field):
    entry, payload, path = manifest(category)
    payload["cases"][0]["annotations"][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=f"Invalid {field}"):
        load_cases(entry, verify_audio=False)


@pytest.mark.parametrize(
    "mutation", ["missing", "version", "audio", "text", "partial", "nan"]
)
def test_source_alignment_contract_rejects_invalid_annotations(manifest, mutation):
    entry, payload, path = manifest()
    annotations = payload["cases"][0]["annotations"]
    alignment = annotations["source_alignment"]
    if mutation == "missing":
        del annotations["source_alignment"]
    elif mutation == "version":
        alignment["version"] = "unknown"
    elif mutation in {"audio", "text"}:
        alignment[mutation + "_sha256"] = "f" * 64
    elif mutation == "partial":
        alignment["segments"].pop(0)
    else:
        alignment["segments"][0]["start"] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(entry, verify_audio=False)
