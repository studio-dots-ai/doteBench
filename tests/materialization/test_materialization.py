"""Materialization recipes, identities, and content-addressed outputs."""

import importlib.util
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from dotebench.materialization import assets
from dotebench.materialization.backends.common import (
    implementation_identity,
    source_identity,
)
from dotebench.materialization.fingerprints import audio_digest
from dotebench.materialization.manifest import load_inventory
from dotebench.materialization.orchestrator import _validate_wav
from dotebench.materialization.providers import synthesize
from dotebench.materialization.verification import (
    _verify_manifest_contract,
    _verify_service_identities,
)
from dotebench.services import load_service_bundle


def _bundle_builder():
    path = Path(__file__).parents[2] / "scripts/build_audio_bundle.py"
    spec = importlib.util.spec_from_file_location("build_audio_bundle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_materialization_config_uses_isolated_official_services(monkeypatch, tmp_path):
    monkeypatch.setenv("DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path))
    root = Path(__file__).parents[2]
    graph = load_service_bundle(
        root / "configs/materialization.yaml",
        overrides=[
            f"runtime.source_root={root}",
            "runtime.model_root=/models",
            "runtime.env_root=/envs",
        ],
    )
    assert set(graph.roots) == {
        "qwen3_tts",
        "indextts2",
        "voxcpm2",
        "mimo_audio",
        "qwen3_aligner",
    }
    expected = json.loads((root / "release/materialization-sources.json").read_text())[
        "sources"
    ]
    for provider in ("qwen3_tts", "indextts2", "voxcpm2", "mimo_audio"):
        node = graph.roots[provider]
        upstream = Path(node.command[node.command.index("--upstream") + 1])
        assert upstream == root / expected[provider]["path"]
        assert node.command[0].startswith("/envs/dotebench-")
        assert set(graph.select_roots(provider).roots) == {provider}


def test_source_identity_rejects_wrong_remote_commit_and_dirty_checkout(tmp_path):
    repository = tmp_path / "source"
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(
        ["git", "-C", repository, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", repository, "config", "user.name", "Test"], check=True)
    (repository / "model.py").write_text("value = 1\n")
    subprocess.run(["git", "-C", repository, "add", "model.py"], check=True)
    subprocess.run(["git", "-C", repository, "commit", "-qm", "source"], check=True)
    source = "https://github.com/example/model.git"
    subprocess.run(
        ["git", "-C", repository, "remote", "add", "origin", source], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", repository, "rev-parse", "HEAD"], text=True
    ).strip()
    assert source_identity(repository, source=source, commit=commit) == {
        "repository": source,
        "commit": commit,
    }
    with pytest.raises(ValueError, match="commit mismatch"):
        source_identity(repository, source=source, commit="0" * 40)
    with pytest.raises(ValueError, match="remote mismatch"):
        source_identity(
            repository, source="https://github.com/other/model.git", commit=commit
        )
    (repository / "model.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="tracked modifications"):
        source_identity(repository, source=source, commit=commit)


def test_provider_request_is_standard_and_preserves_recipe(monkeypatch, tmp_path):
    spec = next(
        spec
        for spec in load_inventory(data_root=Path(__file__).parents[2]).assets.values()
        if spec.provider == "indextts2"
    )
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    response = Mock(
        content=b"wav",
        headers={"content-type": "audio/wav"},
    )
    monkeypatch.setattr(
        "dotebench.materialization.providers.requests.post",
        lambda url, **kwargs: setattr(response, "request", (url, kwargs)) or response,
    )
    assert synthesize("http://provider", spec, reference_audio=reference) == b"wav"
    response.raise_for_status.assert_called_once()
    url, kwargs = response.request
    assert url == "http://provider/synthesize"
    assert kwargs["json"]["text"] == spec.source_text
    assert kwargs["json"]["seed"] == spec.recipe["seed"]
    assert kwargs["json"]["parameters"] == spec.recipe["parameters"]
    assert "reference_audio" in kwargs["json"]


def test_wav_validation_and_digest_are_content_based():
    stream = io.BytesIO()
    sf.write(stream, np.zeros(160, dtype=np.float32), 16000, format="WAV")
    _validate_wav(stream.getvalue())
    with pytest.raises(sf.LibsndfileError):
        _validate_wav(b"not a wav")
    first = {
        "b": {"sha256": "2" * 64, "size_bytes": 2, "provider": None},
        "a": {"sha256": "1" * 64, "size_bytes": 1, "provider": None},
    }
    assert audio_digest(first) == audio_digest(dict(reversed(list(first.items()))))


def test_audio_bundle_rejects_stale_files(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "held.wav").write_bytes(b"stale")
    with pytest.raises(ValueError, match="Unexpected bundle entries"):
        _bundle_builder().reject_extras(audio, {"expected.wav"})


@pytest.mark.parametrize("mutation", ["instruction", "recipe", "alignment"])
def test_manifest_contract_rejects_non_audio_mutation(monkeypatch, tmp_path, mutation):
    from dotebench import dataset

    source_root = Path(__file__).parents[2]

    relative = Path("data/emotion/intense/manifest.json")
    canonical_root = tmp_path / "canonical"
    actual_root = tmp_path / "actual"
    for root in (canonical_root, actual_root):
        (root / relative.parent).mkdir(parents=True)
    shutil.copyfile(source_root / relative, canonical_root / relative)
    shutil.copyfile(source_root / relative, actual_root / relative)
    (canonical_root / "release").mkdir()
    (canonical_root / "release/shards.json").write_text(
        json.dumps(
            {"version": dataset.data_version(), "manifests": ["emotion/intense"]}
        )
    )
    monkeypatch.setattr(dataset, "resources", lambda: canonical_root)
    _verify_manifest_contract(actual_root)
    payload = json.loads((actual_root / relative).read_text())
    if mutation == "instruction":
        payload["cases"][0]["instruction_xml"] += " "
    elif mutation == "alignment":
        payload["cases"][0]["annotations"]["source_alignment"]["segments"][0][
            "start"
        ] += 0.001
    else:
        asset_id, recipe = next(
            (asset_id, recipe)
            for asset_id, recipe in payload["audio_assets"].items()
            if recipe["kind"] == "generate"
        )
        recipe["seed"] += 1
        payload["audio_assets"][asset_id] = recipe
    (actual_root / relative).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract|alignment"):
        _verify_manifest_contract(actual_root)


def test_model_preparation_requires_acceptance_and_detects_changes(
    monkeypatch, tmp_path
):
    model = tmp_path / "IndexTTS-2"
    model.mkdir()
    checkpoint = model / "weights.bin"
    checkpoint.write_bytes(b"weights")
    digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    lock = {
        "repository": "IndexTeam/IndexTTS-2",
        "revision": "a" * 40,
        "directory": "IndexTTS-2",
        "role": "generate",
        "license_acceptance": "indextts2",
    }
    monkeypatch.setattr(assets, "model_locks", lambda: {"indextts2": lock})
    monkeypatch.setattr(
        assets,
        "_hub_files",
        lambda value: {"weights.bin": {"size": 7, "sha256": digest}},
    )
    with pytest.raises(ValueError, match="requires --accept-license"):
        assets.prepare_model("indextts2", model, accepted=set())
    identity = assets.prepare_model("indextts2", model, accepted={"indextts2"})
    assert assets.prepared_identity("indextts2", model, full_check=True) == identity
    stat = checkpoint.stat()
    checkpoint.write_bytes(b"changed")
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed"):
        assets.prepared_identity("indextts2", model)


def test_backend_source_constants_match_release_lock():
    from dotebench.materialization.backends import (
        indextts2,
        mimo_audio,
        qwen3_tts,
        voxcpm2,
    )

    root = Path(__file__).parents[2]
    locks = json.loads((root / "release/materialization-sources.json").read_text())[
        "sources"
    ]
    for key, module in {
        "indextts2": indextts2,
        "mimo_audio": mimo_audio,
        "qwen3_tts": qwen3_tts,
        "voxcpm2": voxcpm2,
    }.items():
        assert module.SOURCE_URL == locks[key]["repository"]
        assert module.SOURCE_COMMIT == locks[key]["commit"]


def test_verifier_binds_service_identities_to_release_locks():
    from dotebench import RELEASE
    from dotebench.materialization.backends import (
        indextts2,
        mimo_audio,
        qwen3_tts,
        voxcpm2,
    )
    from dotebench.materialization.orchestrator import MATERIALIZER_IMPLEMENTATION

    root = Path(__file__).parents[2]
    sources = json.loads((root / "release/materialization-sources.json").read_text())[
        "sources"
    ]
    modules = {
        "indextts2": (indextts2, ("indextts2",)),
        "mimo_audio": (
            mimo_audio,
            ("mimo_audio_instruct", "mimo_audio_tokenizer"),
        ),
        "qwen3_tts": (qwen3_tts, ("qwen3_tts_custom_voice",)),
        "voxcpm2": (voxcpm2, ("voxcpm2",)),
    }
    services = {}
    for provider, (module, model_ids) in modules.items():
        services[provider] = {
            "adapter": implementation_identity(Path(module.__file__), module.__name__),
            "source": {key: sources[provider][key] for key in ("repository", "commit")},
            "models": {
                model_id: {
                    **assets.locked_identity(model_id),
                    "files_sha256": "0" * 64,
                }
                for model_id in model_ids
            },
            "inference": {"test": True},
        }
    receipt = {
        "materializer": {
            "release": RELEASE,
            "module": "dotebench.materialization.orchestrator",
            "sha256": MATERIALIZER_IMPLEMENTATION,
        },
        "services": services,
        "alignment": {"reused": 2081, "generated": 0},
    }
    _verify_service_identities(receipt)
    receipt["services"]["mimo_audio"]["source"]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="source identity mismatch"):
        _verify_service_identities(receipt)
