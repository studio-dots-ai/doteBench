"""Model preparation verifies the lock; startup consumes its immutable receipt."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from dotebench.evaluation import models as assets
from dotebench.services.utils import model_assets


def fixture(monkeypatch, tmp_path):
    content = b"small model weights"
    (tmp_path / "weights.bin").write_bytes(content)
    lock = {
        "directory": "fixture",
        "filename": "weights.bin",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(assets, "model_locks", lambda: {"fixture": lock})
    return lock


def test_prepare_and_offline_check(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    identity = assets.prepare_model("fixture", tmp_path)
    receipt = (tmp_path / model_assets.RECEIPT).read_bytes()
    assert assets.prepared_identity("fixture", tmp_path) == identity
    assert assets.prepared_identity("fixture", tmp_path, full_check=True) == identity
    assert (tmp_path / model_assets.RECEIPT).read_bytes() == receipt


@pytest.mark.parametrize("change", ["weights", "lock", "missing", "extra", "receipt"])
def test_reject_changed_prepared_assets(monkeypatch, tmp_path, change):
    lock = fixture(monkeypatch, tmp_path)
    assets.prepare_model("fixture", tmp_path)
    if change == "weights":
        (tmp_path / "weights.bin").write_bytes(b"wrong weights with a different size")
    elif change == "lock":
        lock["sha256"] = "0" * 64
    elif change == "missing":
        (tmp_path / "weights.bin").unlink()
    elif change == "extra":
        (tmp_path / "config.json").write_text("{}")
    else:
        receipt = json.loads((tmp_path / model_assets.RECEIPT).read_text())
        receipt["files"]["weights.bin"]["sha256"] = "0" * 64
        (tmp_path / model_assets.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        assets.prepared_identity("fixture", tmp_path)


def test_bad_checkpoint_cannot_be_registered(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        assets.prepare_model("fixture", tmp_path)
    assert not (tmp_path / model_assets.RECEIPT).exists()


def test_full_check_detects_same_size_change_with_restored_mtime(monkeypatch, tmp_path):
    import os

    fixture(monkeypatch, tmp_path)
    assets.prepare_model("fixture", tmp_path)
    path = tmp_path / "weights.bin"
    stat = path.stat()
    path.write_bytes(b"x" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="checksum"):
        assets.prepared_identity("fixture", tmp_path, full_check=True)


@pytest.mark.parametrize("change", [None, "revision", "git_blob", "lfs", "missing"])
def test_pinned_hub_files_not_directory_name_determine_identity(
    monkeypatch, tmp_path, change
):
    lock = {"repository": "org/model", "revision": "a" * 40, "directory": "arbitrary"}
    monkeypatch.setattr(assets, "model_locks", lambda: {"fixture": lock})
    config, weights = b"{}", b"model"
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "weights.bin").write_bytes(weights)
    metadata = {
        "sha": lock["revision"],
        "siblings": [
            {
                "rfilename": "config.json",
                "size": 2,
                "blobId": hashlib.sha1(b"blob 2\0" + config).hexdigest(),
            },
            {
                "rfilename": "weights.bin",
                "size": 5,
                "lfs": {"sha256": hashlib.sha256(weights).hexdigest()},
            },
        ],
    }
    if change == "revision":
        metadata["sha"] = "b" * 40
    if change == "git_blob":
        metadata["siblings"][0]["blobId"] = "0" * 40
    if change == "lfs":
        metadata["siblings"][1]["lfs"]["sha256"] = "0" * 64
    if change == "missing":
        (tmp_path / "weights.bin").unlink()

    def get(url, **kwargs):
        assert url.endswith("/revision/" + lock["revision"])
        assert kwargs["params"] == {"blobs": "true"}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: metadata)

    monkeypatch.setattr(model_assets.requests, "get", get)
    if change:
        with pytest.raises(ValueError):
            assets.prepare_model("fixture", tmp_path)
        assert not (tmp_path / model_assets.RECEIPT).exists()
    else:
        expected = assets.prepare_model("fixture", tmp_path)
        assert assets.prepared_identity("fixture", tmp_path) == expected


def test_prepare_models_cli_and_offline_check(monkeypatch, tmp_path, capsys):
    from dotebench.cli import main

    lock = fixture(monkeypatch, tmp_path)
    model = tmp_path / lock["directory"]
    model.mkdir()
    (tmp_path / "weights.bin").rename(model / "weights.bin")
    args = [
        "prepare-evaluation-models",
        "--model-root",
        str(tmp_path),
        "--model",
        "fixture",
    ]
    assert main(args) == 0
    identity = json.loads(capsys.readouterr().out)
    assert main([*args, "--check"]) == 0
    assert json.loads(capsys.readouterr().out) == identity


@pytest.mark.parametrize("invalid", [False, True])
def test_metric_health_identity_checked_and_snapshotted(monkeypatch, invalid):
    from dotebench.evaluation.abilities import ABILITY_NAMES, HTTPMetricAbilities

    client = HTTPMetricAbilities({name: "http://" + name for name in ABILITY_NAMES})
    from dotebench.evaluation.models import MODELS

    bodies = {}
    for name in ABILITY_NAMES:
        identity = None
        if name in MODELS:
            identity = {
                "model": MODELS[name],
                "lock_sha256": model_assets.digest(assets.model_locks()[MODELS[name]]),
                "files_sha256": "a" * 64,
            }
        bodies["http://" + name + "/health"] = {
            "status": "ok",
            "ability": name,
            "model_identity": identity,
            "configuration": {"device": "cpu"},
        }
    if invalid:
        bodies["http://qwen3_asr/health"]["model_identity"]["lock_sha256"] = "0" * 64
    monkeypatch.setattr(
        client.session,
        "get",
        lambda url, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: bodies[url]
        ),
    )
    if invalid:
        with pytest.raises(ValueError, match="identity mismatch"):
            client.prepare()
    else:
        client.prepare()
        result = client.identity()
        result["qwen3_asr"]["configuration"]["device"] = "changed"
        assert client.identity()["qwen3_asr"]["configuration"]["device"] == "cpu"
    client.close()


def test_missing_model_receipt_stops_before_loading(monkeypatch, tmp_path):
    import sys

    from dotebench.services.backends import qwen3_asr as backend

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metric_service",
            "--model-path",
            str(tmp_path),
            "--port",
            "9999",
        ],
    )
    monkeypatch.setattr(
        backend,
        "build",
        lambda *args, **kwargs: pytest.fail("unprepared model must not load"),
    )
    with pytest.raises(FileNotFoundError):
        backend.main()


@pytest.mark.parametrize(
    "model_id", ["emotion", "speaker_similarity", "qwen3_asr", "qwen3_aligner"]
)
def test_capability_names_are_not_model_aliases(model_id, tmp_path):
    with pytest.raises(KeyError):
        assets.model_directory(tmp_path, model_id)


def test_service_cli_rejects_inference_overrides(monkeypatch):
    import sys

    from dotebench.services.backends import wer as backend

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metric_service",
            "--port",
            "9999",
            "--options-json",
            "{}",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        backend.main()
    assert exc.value.code == 2


def test_service_receipt_verification_is_independent_of_evaluator_lock(
    monkeypatch, tmp_path
):
    lock = fixture(monkeypatch, tmp_path)
    identity = assets.prepare_model("fixture", tmp_path)
    # Existing receipt format is sufficient for a service to start.
    assert model_assets.prepared_identity("fixture", tmp_path) == identity
    lock["sha256"] = "0" * 64
    assert model_assets.prepared_identity("fixture", tmp_path) == identity
    # The evaluator separately rejects a model that no longer matches its policy.
    with pytest.raises(ValueError, match="identity mismatch"):
        assets.prepared_identity("fixture", tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        model_assets.prepared_identity("different-model", tmp_path)
