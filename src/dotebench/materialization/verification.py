"""Offline verification for materialized dataset roots."""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

import soundfile as sf

from dotebench import RELEASE, dataset
from dotebench.alignment import validate_alignment
from dotebench.instructions import parse_instruction

from .assets import locked_identity
from .fingerprints import audio_digest, file_sha256
from .manifest import PROVIDERS, load_inventory
from .orchestrator import MATERIALIZER_IMPLEMENTATION

_PROVIDERS = {
    "qwen3_tts": ("qwen3_tts.py", ("qwen3_tts_custom_voice",)),
    "indextts2": ("indextts2.py", ("indextts2",)),
    "voxcpm2": ("voxcpm2.py", ("voxcpm2",)),
    "mimo_audio": (
        "mimo_audio.py",
        ("mimo_audio_instruct", "mimo_audio_tokenizer"),
    ),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_manifest_hashes() -> dict[str, str]:
    """Identify the immutable public manifests used as the materialization contract."""
    root = dataset.resources()
    return {
        f"{entry['category']}/{entry['shard']}": file_sha256(
            dataset.manifest_path(entry, root)
        )
        for entry in dataset.select_shards(data_root=root)
    }


def _contract_case(canonical: dict, actual: dict, *, audio_root: Path) -> None:
    canonical_fixed = copy.deepcopy(canonical)
    actual_fixed = copy.deepcopy(actual)
    for payload in (canonical_fixed, actual_fixed):
        payload["source_audio"].pop("path")
        payload["source_audio"].pop("sha256")
        alignment = payload["annotations"]["source_alignment"]
        alignment.pop("segments")
        alignment.pop("audio_sha256")
    if actual_fixed != canonical_fixed:
        raise ValueError(f"Materialized case contract mismatch: {canonical['id']}")
    canonical_sha = canonical["source_audio"]["sha256"]
    actual_sha = actual["source_audio"]["sha256"]
    canonical_alignment = canonical["annotations"]["source_alignment"]
    actual_alignment = actual["annotations"]["source_alignment"]
    if actual_sha == canonical_sha:
        if actual_alignment != canonical_alignment:
            raise ValueError(
                f"Unchanged audio has changed alignment: {canonical['id']}"
            )
        return
    if actual_alignment["audio_sha256"] != actual_sha:
        raise ValueError(f"Materialized alignment audio mismatch: {canonical['id']}")
    path = dataset.relative_path(audio_root, actual["source_audio"]["path"])
    info = sf.info(path)
    validate_alignment(
        parse_instruction(actual["instruction_xml"]).source_text,
        actual_alignment["segments"],
        info.frames / info.samplerate,
    )


def _verify_manifest_contract(root: Path) -> dict[str, str]:
    canonical_root = dataset.resources()
    entries = dataset.select_shards(data_root=canonical_root)
    expected_paths = {
        str(dataset.manifest_path(entry, canonical_root).relative_to(canonical_root))
        for entry in entries
    }
    actual_paths = {
        str(path.relative_to(root)) for path in (root / "data").glob("**/manifest.json")
    }
    if actual_paths != expected_paths:
        raise ValueError("Materialized manifest set differs from the release")
    for entry in entries:
        canonical = dataset.read_json(dataset.manifest_path(entry, canonical_root))
        actual = dataset.read_json(dataset.manifest_path(entry, root))
        dataset.check_manifest_identity(actual, entry["category"], entry["shard"])
        for key in ("category", "shard", "audio_assets"):
            if actual.get(key) != canonical.get(key):
                raise ValueError(f"Materialized manifest contract mismatch: {key}")
        if len(actual["cases"]) != len(canonical["cases"]):
            raise ValueError("Materialized manifest structure mismatch")
        for expected_case, actual_case in zip(
            canonical["cases"], actual["cases"], strict=True
        ):
            _contract_case(expected_case, actual_case, audio_root=root)
    return {
        relative: file_sha256(root / relative) for relative in sorted(expected_paths)
    }


def _verify_service_identities(receipt: dict) -> None:
    expected_materializer = {
        "release": RELEASE,
        "module": "dotebench.materialization.orchestrator",
        "sha256": MATERIALIZER_IMPLEMENTATION,
    }
    if receipt.get("materializer") != expected_materializer:
        raise ValueError("Materializer implementation identity mismatch")
    services = receipt.get("services")
    if not isinstance(services, dict) or not set(PROVIDERS) <= set(services):
        raise ValueError("Materialization provider identities are incomplete")
    if set(services) - set(PROVIDERS) - {"qwen3_aligner"}:
        raise ValueError("Unknown materialization service identity")
    source_locks = dataset.read_json(
        dataset.resources() / "release/materialization-sources.json"
    )["sources"]
    backend_root = Path(__file__).resolve().parent / "backends"
    for provider, (filename, model_ids) in _PROVIDERS.items():
        identity = services[provider]
        if not isinstance(identity, dict) or set(identity) != {
            "adapter",
            "source",
            "models",
            "inference",
        }:
            raise ValueError(f"Invalid materialization provider identity: {provider}")
        expected_adapter = {
            "module": f"dotebench.materialization.backends.{filename.removesuffix('.py')}",
            "sha256": hashlib.sha256(
                (backend_root / filename).read_bytes()
            ).hexdigest(),
        }
        if identity["adapter"] != expected_adapter:
            raise ValueError(f"Materialization adapter identity mismatch: {provider}")
        expected_source = {
            key: source_locks[provider][key] for key in ("repository", "commit")
        }
        if identity["source"] != expected_source:
            raise ValueError(f"Materialization source identity mismatch: {provider}")
        if not isinstance(identity["inference"], dict) or not identity["inference"]:
            raise ValueError(f"Missing materialization inference identity: {provider}")
        if not isinstance(identity["models"], dict) or set(identity["models"]) != set(
            model_ids
        ):
            raise ValueError(f"Materialization model inventory mismatch: {provider}")
        for model_id in model_ids:
            model = identity["models"][model_id]
            expected_model = locked_identity(model_id)
            if (
                not isinstance(model, dict)
                or set(model) != set(expected_model) | {"files_sha256"}
                or any(model.get(key) != value for key, value in expected_model.items())
                or not _SHA256.fullmatch(str(model.get("files_sha256", "")))
            ):
                raise ValueError(
                    f"Materialization model identity mismatch: {provider}/{model_id}"
                )
    alignment = receipt.get("alignment")
    if not isinstance(alignment, dict) or set(alignment) != {"reused", "generated"}:
        raise ValueError("Invalid materialization alignment accounting")
    if alignment["generated"] and "qwen3_aligner" not in services:
        raise ValueError("Changed audio is missing its aligner identity")


def verify_materialization(root: Path, *, reference: Path | None = None) -> dict:
    root = Path(root).resolve()
    receipt = dataset.read_json(root / "materialization.json")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("version") != dataset.data_version()
    ):
        raise ValueError("Unsupported materialization receipt identity")
    _verify_service_identities(receipt)
    inventory = load_inventory(data_root=root)
    canonical_inventory = load_inventory(data_root=dataset.resources())
    if set(inventory.assets) != set(canonical_inventory.assets):
        raise ValueError("Materialized asset inventory differs from the release")
    records = receipt.get("audio")
    if not isinstance(records, dict) or set(records) != set(inventory.assets):
        raise ValueError("Materialization receipt asset inventory mismatch")
    expected_audio_files = {f"{record['sha256']}.wav" for record in records.values()}
    audio_directory = root / "audio"
    if (
        not audio_directory.is_dir()
        or {path.name for path in audio_directory.iterdir()} != expected_audio_files
    ):
        raise ValueError("Materialized audio directory is not content closed")
    for asset_id, spec in inventory.assets.items():
        record = records[asset_id]
        canonical_spec = canonical_inventory.assets[asset_id]
        if (
            not isinstance(record, dict)
            or set(record)
            != {"kind", "provider", "reference_sha256", "sha256", "size_bytes"}
            or record.get("kind") != spec.kind
            or record.get("provider") != spec.provider
            or record.get("reference_sha256") != canonical_spec.reference_sha256
            or record.get("sha256") != spec.reference_sha256
        ):
            raise ValueError(f"Materialization receipt recipe mismatch: {asset_id}")
        path = audio_directory / f"{record['sha256']}.wav"
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Materialized audio checksum mismatch: {asset_id}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Materialized audio size mismatch: {asset_id}")
    if audio_digest(records) != receipt.get("audio_digest"):
        raise ValueError("Materialization audio digest mismatch")
    manifest_hashes = _verify_manifest_contract(root)
    if manifest_hashes != receipt.get("manifests"):
        raise ValueError("Materialization manifest digest mismatch")
    for entry in inventory.entries:
        dataset.load_cases(entry, data_root=root, audio_root=root, verify_audio=True)
    if sum(receipt["alignment"].values()) != sum(
        entry["case_count"] for entry in inventory.entries
    ):
        raise ValueError("Materialization alignment accounting mismatch")
    comparison = {"strict_mismatches": [], "generated_mismatches": []}
    if reference is not None:
        expected = dataset.read_json(Path(reference))
        expected_audio = expected.get("assets")
        if (
            expected.get("schema_version") != 1
            or set(expected)
            != {"schema_version", "assets", "audio_digest", "canonical_manifests"}
            or expected.get("canonical_manifests") != canonical_manifest_hashes()
            or not isinstance(expected_audio, dict)
            or set(expected_audio) != set(records)
            or audio_digest(expected_audio) != expected.get("audio_digest")
        ):
            raise ValueError("Reference materialization identity mismatch")
        for asset_id, record in records.items():
            if any(
                expected_audio[asset_id].get(key) != record.get(key)
                for key in ("kind", "provider")
            ):
                raise ValueError(
                    f"Reference materialization recipe mismatch: {asset_id}"
                )
            if record["sha256"] == expected_audio[asset_id]["sha256"]:
                continue
            target = (
                "generated_mismatches"
                if record["kind"] == "generate"
                else "strict_mismatches"
            )
            comparison[target].append(asset_id)
        if comparison["strict_mismatches"]:
            raise ValueError(
                "Bundle or upstream audio differs from the reference materialization"
            )
    return {
        "status": "ok",
        "assets": len(records),
        "cases": sum(entry["case_count"] for entry in inventory.entries),
        "audio_digest": receipt["audio_digest"],
        "comparison": comparison,
    }
