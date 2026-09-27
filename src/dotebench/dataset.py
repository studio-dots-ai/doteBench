"""Read the versioned public dataset with strict immutable records."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import get_args

from .domain import (
    AlignmentUnit,
    Annotations,
    AudioAsset,
    Case,
    SourceAlignment,
    SourceEmotion,
    SpanGranularity,
)
from .instructions import parse_instruction

CATEGORIES = ("text", "emotion", "prosody", "pause", "compositional")


def resources() -> Path:
    installed = Path(__file__).resolve().parent / "resources"
    return installed if installed.is_dir() else Path(__file__).resolve().parents[2]


def read_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def relative_path(root: Path, value: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or "\\" in value
    ):
        raise ValueError("Expected a relative asset path")
    return Path(root) / value


def release_index() -> dict:
    index = read_json(resources() / "release/shards.json")
    if set(index) != {"version", "manifests"}:
        raise ValueError("Invalid release index fields")
    version = safe_identifier(index["version"])
    paths = index["manifests"]
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise ValueError("Invalid release manifest list")
    for relative in paths:
        if not isinstance(relative, str) or len(Path(relative).parts) != 2:
            raise ValueError("Invalid release manifest path")
        category, shard = Path(relative).parts
        if category not in CATEGORIES:
            raise ValueError("Invalid release category")
        safe_identifier(shard)
    return {"version": version, "manifests": paths}


def data_version() -> str:
    return release_index()["version"]


def check_manifest_identity(manifest: dict, category: str, shard: str) -> None:
    expected = {"category": category, "shard": shard}
    fields = {"category", "shard", "audio_assets", "cases"}
    if set(manifest) != fields or any(
        manifest[key] != value for key, value in expected.items()
    ):
        raise ValueError("Manifest identity disagrees with its release path")


def select_shards(category=None, shard=None, *, data_root=None) -> list[dict]:
    root = Path(data_root or resources())
    selected = []
    index = release_index()
    for relative in index["manifests"]:
        entry_category, entry_shard = Path(relative).parts
        if (category and category != entry_category) or (
            shard and shard != entry_shard
        ):
            continue
        path = root / "data" / relative / "manifest.json"
        manifest = read_json(path)
        expected = {
            "category": entry_category,
            "shard": entry_shard,
        }
        check_manifest_identity(manifest, entry_category, entry_shard)
        selected.append(
            {
                **expected,
                "case_count": len(manifest["cases"]),
            }
        )
    if not selected:
        raise ValueError(f"Unknown dataset selection: {category}/{shard}")
    return selected


def manifest_path(entry: dict, data_root: Path | None = None) -> Path:
    return (
        Path(data_root or resources())
        / "data"
        / safe_identifier(entry["category"])
        / safe_identifier(entry["shard"])
        / "manifest.json"
    )


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Expected fields: {sorted(expected)}")


def _annotations(raw, category, asset, source_text):
    fields = {"source_alignment"}
    if category == "prosody":
        fields.add("span_granularity")
    if category == "emotion" and isinstance(raw, dict) and "source_emotion" in raw:
        fields.add("source_emotion")
    _keys(raw, fields)
    for name, annotation_type in (
        ("span_granularity", SpanGranularity),
        ("source_emotion", SourceEmotion),
    ):
        if name in raw and raw[name] not in get_args(annotation_type):
            raise ValueError(f"Invalid {name} annotation")
    return Annotations(
        raw.get("span_granularity"),
        raw.get("source_emotion"),
        _source_alignment(raw["source_alignment"], asset, source_text),
    )


def _source_alignment(raw, asset, text):
    _keys(raw, {"audio_sha256", "text_sha256", "segments"})
    if (
        raw["audio_sha256"] != asset.sha256
        or raw["text_sha256"] != hashlib.sha256(text.encode()).hexdigest()
    ):
        raise ValueError("Source alignment checksum mismatch")
    if not isinstance(raw["segments"], list):
        raise ValueError("Expected source alignment segments")
    units = []
    for item in raw["segments"]:
        _keys(item, {"word", "start", "end"})
        if not isinstance(item["word"], str) or not item["word"]:
            raise ValueError("Expected alignment word")
        if (
            any(
                type(item[k]) not in (int, float)
                or not math.isfinite(item[k])
                or item[k] < 0
                for k in ("start", "end")
            )
            or item["end"] < item["start"]
        ):
            raise ValueError("Invalid source alignment time")
        units.append(AlignmentUnit(**item))
    from .alignment import validate_alignment

    validate_alignment(text, units, max((u.end for u in units), default=0.0))
    return SourceAlignment(raw["audio_sha256"], raw["text_sha256"], tuple(units))


def load_cases(
    entry: dict,
    *,
    data_root: Path | None = None,
    audio_root: Path | None = None,
    verify_audio: bool = True,
) -> tuple[Case, ...]:
    manifest = read_json(manifest_path(entry, data_root))
    check_manifest_identity(manifest, entry["category"], entry["shard"])
    if not isinstance(manifest["audio_assets"], dict):
        raise TypeError("Expected audio_assets mapping")
    if (
        not isinstance(manifest["cases"], list)
        or len(manifest["cases"]) != entry["case_count"]
    ):
        raise ValueError("Manifest case count mismatch")
    seen, checked, result = set(), set(), []
    for raw in manifest["cases"]:
        _keys(
            raw,
            {
                "id",
                "language",
                "source_audio",
                "instruction_xml",
                "annotations",
            },
        )
        ident = safe_identifier(raw["id"])
        if ident in seen:
            raise ValueError(f"Duplicate case ID: {ident}")
        seen.add(ident)
        if raw["language"] not in {"en", "zh"}:
            raise ValueError("Unsupported language")
        for key in ("instruction_xml",):
            if not isinstance(raw[key], str):
                raise ValueError(f"Expected string: {key}")
        _keys(raw["source_audio"], {"asset_id", "path", "sha256"})
        safe_identifier(raw["source_audio"]["asset_id"])
        asset = AudioAsset(
            path=raw["source_audio"]["path"],
            sha256=raw["source_audio"]["sha256"],
        )
        path = relative_path(Path(audio_root or data_root or resources()), asset.path)
        if not isinstance(asset.sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", asset.sha256
        ):
            raise ValueError("Invalid audio checksum")
        check = (str(path), asset.sha256)
        if verify_audio and check not in checked:
            if sha256(path) != asset.sha256:
                raise ValueError(f"Audio checksum mismatch: {ident}")
            checked.add(check)
        instruction = parse_instruction(raw["instruction_xml"])
        result.append(
            Case(
                ident,
                raw["language"],
                asset,
                raw["instruction_xml"],
                _annotations(
                    raw["annotations"],
                    entry["category"],
                    asset,
                    instruction.source_text,
                ),
                instruction,
            )
        )
    return tuple(result)
