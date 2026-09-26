#!/usr/bin/env python3
"""Build embedded audio recipes from the reviewed public provenance ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GENERATED = {"indextts2", "voxcpm2", "mimo_audio_instruct", "qwen3_tts"}
PROVIDER = {
    "indextts2": "indextts2",
    "voxcpm2": "voxcpm2",
    "mimo_audio_instruct": "mimo_audio",
    "qwen3_tts": "qwen3_tts",
}


def load_ledger(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        sha = record["audio_sha256"]
        if sha in result:
            raise ValueError(f"Duplicate ledger audio: {sha}")
        result[sha] = record
    return result


def recipe(record: dict) -> dict:
    generation = record.get("generation") or {}
    generator = generation.get("generator")
    sha = record["audio_sha256"]
    if generator in GENERATED:
        parameters = {}
        for source, target in (
            ("control", "control"),
            ("preset_speaker", "speaker"),
            ("emotion_vector", "emotion_vector"),
            ("emotion_alpha", "emotion_alpha"),
            ("cfg_value", "cfg_value"),
            ("inference_timesteps", "inference_timesteps"),
            ("max_new_tokens", "max_new_tokens"),
        ):
            value = generation.get(source)
            if value is not None:
                parameters[target] = value
        result = {
            "kind": "generate",
            "provider": PROVIDER[generator],
            "model_id": generation["model_id"],
            "seed": generation["seed"],
            "parameters": parameters,
        }
        reference = generation.get("reference_voice")
        if reference:
            result["reference_asset_id"] = reference["audio_sha256"]
        return result
    if record["source_dataset"] == "CommonVoice20.0":
        return {
            "kind": "upstream",
            "dataset": "CommonVoice20.0",
            "split": record["source_split"],
            "clip_id": record["source_clip_id"],
            "path": record["source_path"],
            "sha256": record["source_raw_sha256"],
        }
    return {"kind": "bundle", "path": f"audio/{sha}.wav"}


def build(root: Path, ledger_path: Path) -> dict[str, int]:
    ledger = load_ledger(ledger_path)
    counts = {"bundle": 0, "upstream": 0, "generate": 0}
    seen = set()
    index = json.loads((root / "release/shards.json").read_text(encoding="utf-8"))
    for relative in index["manifests"]:
        path = root / "data" / relative / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assets = {}
        for case in payload["cases"]:
            source = case["source_audio"]
            sha = source["sha256"]
            if sha not in ledger:
                raise ValueError(f"Missing public provenance for {sha}")
            source["asset_id"] = sha
            item = recipe(ledger[sha])
            if sha in assets and assets[sha] != item:
                raise ValueError(f"Conflicting recipe for {sha}")
            assets[sha] = item
            seen.add(sha)
        payload = {
            "category": payload["category"],
            "shard": payload["shard"],
            "audio_assets": dict(sorted(assets.items())),
            "cases": payload["cases"],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if seen != set(ledger):
        raise ValueError("Public provenance and release manifests disagree")
    for sha in seen:
        counts[recipe(ledger[sha])["kind"]] += 1
    if counts != {"bundle": 1027, "upstream": 226, "generate": 100}:
        raise ValueError(f"Unexpected materialization counts: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--ledger", type=Path, default=Path("release/audio-provenance.jsonl")
    )
    args = parser.parse_args()
    ledger = args.ledger if args.ledger.is_absolute() else args.root / args.ledger
    print(json.dumps(build(args.root, ledger), sort_keys=True))


if __name__ == "__main__":
    main()
