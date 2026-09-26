import dataclasses
import json

import pytest

from dotebench.dataset import load_cases, manifest_path, select_shards


def test_release_version_comes_from_index(monkeypatch, tmp_path):
    from dotebench import dataset

    original = dataset.resources()
    index = dataset.release_index()
    next_version = "fixture_next"
    (tmp_path / "release").mkdir()
    (tmp_path / "release/shards.json").write_text(
        json.dumps({**index, "version": next_version})
    )
    (tmp_path / "data").symlink_to(original / "data", target_is_directory=True)
    monkeypatch.setattr(dataset, "resources", lambda: tmp_path)
    entries = select_shards()
    assert dataset.data_version() == next_version
    assert sum(item["case_count"] for item in entries) == 2081
    assert all(manifest_path(item).is_file() for item in entries)
    assert sum(len(load_cases(item, verify_audio=False)) for item in entries) == 2081
    from dotebench.materialization.manifest import load_inventory
    from dotebench.materialization.verification import canonical_manifest_hashes

    inventory = load_inventory(data_root=tmp_path)
    assert len(inventory.entries) == 7
    assert set(canonical_manifest_hashes()) == set(index["manifests"])


def test_loaded_case_uses_xml_derived_transcripts_and_is_immutable():
    case = load_cases(select_shards()[0], verify_audio=False)[0]
    assert case.instruction.source_text == case.source_text
    assert case.instruction.target_text == case.target_text
    assert case.instruction.operations
    for operation in case.instruction.operations:
        assert (
            0 <= operation.source.start <= operation.source.end <= len(case.source_text)
        )
        assert (
            0 <= operation.target.start <= operation.target.end <= len(case.target_text)
        )
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
def test_reject_invalid_manifest(tmp_path, mutation):
    entry = dict(select_shards()[0])
    payload = json.loads(manifest_path(entry).read_text())
    payload["cases"] = payload["cases"][:1]
    entry["case_count"] = 1
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
    path = manifest_path(entry, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(entry, data_root=tmp_path, verify_audio=False)


@pytest.mark.parametrize(
    "mutation", ["status", "boundary_type", "missing", "null", "unnecessary_interval"]
)
def test_annotation_whitelist_rejects_nonmetric_fields(tmp_path, mutation):
    entry = dict(select_shards(category="compositional")[0])
    payload = json.loads(manifest_path(entry).read_text())
    payload["cases"] = payload["cases"][:1]
    entry["case_count"] = 1
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
    path = manifest_path(entry, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(entry, data_root=tmp_path, verify_audio=False)


@pytest.mark.parametrize("value", [None, "phrase", 1, [], {"label": "clause"}])
@pytest.mark.parametrize(
    "category,shard,field",
    [
        ("prosody", "default", "span_granularity"),
        ("emotion", "intense", "source_emotion"),
    ],
)
def test_grouping_annotation_requires_declared_enum(
    tmp_path, value, category, shard, field
):
    entry = dict(select_shards(category=category, shard=shard)[0], case_count=1)
    payload = json.loads(manifest_path(entry).read_text())
    payload["cases"] = payload["cases"][:1]
    payload["cases"][0]["annotations"][field] = value
    path = manifest_path(entry, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=f"Invalid {field}"):
        load_cases(entry, data_root=tmp_path, verify_audio=False)


@pytest.mark.parametrize(
    "mutation", ["missing", "version", "audio", "text", "partial", "nan"]
)
def test_source_alignment_contract_rejects_invalid_annotations(tmp_path, mutation):
    entry = dict(select_shards()[0], case_count=1)
    payload = json.loads(manifest_path(entry).read_text())
    payload["cases"] = payload["cases"][:1]
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
    path = manifest_path(entry, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(entry, data_root=tmp_path, verify_audio=False)
