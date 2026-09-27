"""Small synthetic release manifest for parser and materialization tests."""

import hashlib
import json


def write_manifest(root, *, category="text"):
    shard = "fixture"
    audio_sha = "a" * 64
    annotations = {
        "source_alignment": {
            "audio_sha256": audio_sha,
            "text_sha256": hashlib.sha256(b"hello").hexdigest(),
            "segments": [{"word": "hello", "start": 0.0, "end": 1.0}],
        }
    }
    if category == "prosody":
        annotations["span_granularity"] = "single_word"
    if category == "emotion":
        annotations["source_emotion"] = "happy"
    payload = {
        "category": category,
        "shard": shard,
        "audio_assets": {"source": {"kind": "bundle", "path": "audio/source.wav"}},
        "cases": [
            {
                "id": "fixture_case",
                "language": "en",
                "source_audio": {
                    "asset_id": "source",
                    "path": "audio/source.wav",
                    "sha256": audio_sha,
                },
                "instruction_xml": '<sub targ="world">hello</sub>',
                "annotations": annotations,
            }
        ],
    }
    index = {"version": "fixture_v1", "manifests": [f"{category}/{shard}"]}
    index_path = root / "release/shards.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))
    manifest_path = root / "data" / category / shard / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload))
    return {"category": category, "shard": shard, "case_count": 1}, payload, manifest_path
