"""Strict materialization recipes embedded in release shard manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotebench import dataset
from dotebench.instructions import parse_instruction

PROVIDERS = ("qwen3_tts", "indextts2", "voxcpm2", "mimo_audio")
KINDS = ("bundle", "upstream", "generate")


def _keys(value: object, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise TypeError("Expected materialization mapping")
    optional = optional or set()
    present = set(value)
    if not required <= present or present - required - optional:
        raise ValueError(
            f"Expected required fields {sorted(required)} and optional fields "
            f"{sorted(optional)}"
        )


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid {label} SHA-256")
    return value


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    reference_sha256: str
    source_text: str
    language: str
    recipe: dict[str, Any]
    case_ids: tuple[str, ...]

    @property
    def kind(self) -> str:
        return str(self.recipe["kind"])

    @property
    def provider(self) -> str | None:
        value = self.recipe.get("provider")
        return str(value) if value is not None else None


@dataclass(frozen=True)
class MaterializationInventory:
    entries: tuple[dict[str, Any], ...]
    manifests: dict[Path, dict[str, Any]]
    assets: dict[str, AssetSpec]


def validate_recipe(asset_id: str, raw: object) -> dict[str, Any]:
    dataset.safe_identifier(asset_id)
    if not isinstance(raw, dict):
        raise TypeError("Expected audio asset recipe")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(f"Invalid materialization kind: {kind!r}")
    if kind == "bundle":
        _keys(raw, {"kind", "path"})
        dataset.relative_path(Path("."), raw["path"])
    elif kind == "upstream":
        _keys(raw, {"kind", "dataset", "split", "clip_id", "path", "sha256"})
        if raw["dataset"] != "CommonVoice20.0":
            raise ValueError("Unsupported upstream dataset")
        if not all(
            isinstance(raw[key], str) and raw[key] for key in ("split", "clip_id")
        ):
            raise ValueError("Invalid upstream source identity")
        dataset.relative_path(Path("."), raw["path"])
        _sha(raw["sha256"], "upstream source")
    else:
        _keys(
            raw,
            {"kind", "provider", "model_id", "seed", "parameters"},
            {"reference_asset_id"},
        )
        if raw["provider"] not in PROVIDERS:
            raise ValueError("Unsupported materialization provider")
        if type(raw["seed"]) is not int or raw["seed"] < 0:
            raise ValueError("Invalid generation seed")
        if not isinstance(raw["model_id"], str) or not raw["model_id"]:
            raise ValueError("Invalid generation model ID")
        if not isinstance(raw["parameters"], dict):
            raise ValueError("Expected generation parameters")
        reference = raw.get("reference_asset_id")
        if reference is not None:
            dataset.safe_identifier(reference)
    return dict(raw)


def load_inventory(*, data_root: Path | None = None) -> MaterializationInventory:
    entries = tuple(dataset.select_shards(data_root=data_root))
    manifests: dict[Path, dict[str, Any]] = {}
    collected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = dataset.manifest_path(entry, data_root)
        payload = dataset.read_json(path)
        assets = payload.get("audio_assets")
        if not isinstance(assets, dict):
            raise TypeError(f"Manifest has no audio_assets mapping: {path}")
        manifests[path] = payload
        used: set[str] = set()
        for case in payload["cases"]:
            source = case["source_audio"]
            _keys(source, {"asset_id", "path", "sha256"})
            asset_id = dataset.safe_identifier(source["asset_id"])
            reference_sha = _sha(source["sha256"], "source audio")
            recipe = validate_recipe(asset_id, assets.get(asset_id))
            instruction = parse_instruction(case["instruction_xml"])
            record = {
                "reference_sha256": reference_sha,
                "source_text": instruction.source_text,
                "language": case["language"],
                "recipe": recipe,
            }
            prior = collected.setdefault(asset_id, {**record, "case_ids": []})
            if any(prior[key] != record[key] for key in record):
                raise ValueError(f"Conflicting materialization asset: {asset_id}")
            prior["case_ids"].append(case["id"])
            used.add(asset_id)
        if set(assets) != used:
            raise ValueError(f"Unused or missing audio asset recipe: {path}")
    for asset_id, record in collected.items():
        reference = record["recipe"].get("reference_asset_id")
        if reference is not None and reference not in collected:
            raise ValueError(
                f"Generation reference {reference!r} for {asset_id!r} is unavailable"
            )
    specs = {
        asset_id: AssetSpec(
            asset_id=asset_id,
            reference_sha256=record["reference_sha256"],
            source_text=record["source_text"],
            language=record["language"],
            recipe=record["recipe"],
            case_ids=tuple(sorted(set(record["case_ids"]))),
        )
        for asset_id, record in sorted(collected.items())
    }
    return MaterializationInventory(entries, manifests, specs)
