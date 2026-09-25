"""Build the public transcript, XML, and annotation provenance ledger."""

import argparse
import hashlib
import json
from pathlib import Path

from dotebench.instructions import parse_instruction

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release" / "text-provenance.jsonl"

SOURCES = {
    "AISHELL-1": {
        "license": "Apache-2.0",
        "evidence_url": "https://www.openslr.org/33/",
        "redistribution": "allowed_with_notice",
    },
    "AISHELL3": {
        "license": "Apache-2.0",
        "evidence_url": "https://www.openslr.org/93/",
        "redistribution": "allowed_with_notice",
    },
    "CommonVoice20.0": {
        "license": "CC0-1.0",
        "evidence_url": "https://commonvoice.mozilla.org/en/datasets",
        "redistribution": "public_domain_dedication",
    },
    "Emo-Emilia": {
        "license": "CC-BY-NC-4.0",
        "evidence_url": "https://huggingface.co/datasets/ASLP-lab/Emo-Emilia",
        "redistribution": "noncommercial_with_attribution",
    },
    "FLEURS": {
        "license": "CC-BY-4.0",
        "evidence_url": "https://huggingface.co/datasets/google/fleurs",
        "redistribution": "allowed_with_attribution",
    },
    "LibriTTS-R": {
        "license": "CC-BY-4.0",
        "evidence_url": "https://www.openslr.org/141/",
        "redistribution": "allowed_with_attribution",
    },
    "MLS": {
        "license": "CC-BY-4.0",
        "evidence_url": "https://www.openslr.org/94/",
        "redistribution": "allowed_with_attribution",
    },
    "WenetSpeech": {
        "license": "CC-BY-4.0",
        "evidence_url": "https://wenet-e2e.github.io/WenetSpeech/",
        "redistribution": "noncommercial_with_attribution",
        "use_constraint": "The official dataset page limits downloads to non-commercial purposes.",
    },
    "doteBench": {
        "license": "Apache-2.0",
        "evidence_url": "../LICENSE",
        "redistribution": "allowed_with_notice",
    },
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audio_ledger(root: Path):
    result = {}
    path = root / "release" / "audio-provenance.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sha = row["audio_sha256"]
        if sha in result:
            raise ValueError(f"Duplicate audio provenance: {sha}")
        result[sha] = row
    return result


def normalized_origin(audio_row: dict):
    raw = audio_row["rights"]["transcript"]["source"]
    if isinstance(raw, str):
        origin = {
            "dataset": raw,
            "kind": "dataset_transcript",
            "split": audio_row.get("source_split"),
            "source_id": audio_row.get("source_clip_id"),
        }
    else:
        origin = {
            "dataset": raw.get("dataset") or "doteBench",
            "kind": raw["kind"],
            "split": raw.get("split"),
            "source_id": raw.get("uid"),
        }
    evidence = SOURCES[origin["dataset"]]
    return {**origin, **evidence}


def text_edits(parsed):
    rows = []
    for operation in parsed.operations:
        if operation.kind not in {"ins", "del", "sub"}:
            continue
        params = dict(operation.params)
        target = (
            parsed.target_text[operation.target.start : operation.target.end]
            if operation.kind == "ins"
            else params.get("targ", "")
            if operation.kind == "sub"
            else ""
        )
        rows.append(
            {
                "kind": operation.kind,
                "source_sha256": digest(
                    parsed.source_text[operation.source.start : operation.source.end]
                ),
                "target_sha256": digest(target),
            }
        )
    return rows


def build(root: Path):
    audio = audio_ledger(root)
    index = read_json(root / "release" / "shards.json")
    rows = []
    seen = set()
    for relative in index["manifests"]:
        manifest = read_json(root / "data" / relative / "manifest.json")
        for case in manifest["cases"]:
            if case["id"] in seen:
                raise ValueError(f"Duplicate case ID: {case['id']}")
            seen.add(case["id"])
            source_audio = case["source_audio"]["sha256"]
            if source_audio not in audio:
                raise ValueError(f"Missing audio provenance: {source_audio}")
            parsed = parse_instruction(case["instruction_xml"])
            recorded = case["annotations"]["source_alignment"]["text_sha256"]
            if recorded != digest(parsed.source_text):
                raise ValueError(f"Source text hash mismatch: {case['id']}")
            rows.append(
                {
                    "case_id": case["id"],
                    "category": manifest["category"],
                    "shard": manifest["shard"],
                    "language": case["language"],
                    "instruction_xml_sha256": digest(case["instruction_xml"]),
                    "source_text_sha256": recorded,
                    "target_text_sha256": digest(parsed.target_text),
                    "source_transcript": normalized_origin(audio[source_audio]),
                    "text_edits": text_edits(parsed),
                    "benchmark_contributions": {
                        "xml_markup": "Apache-2.0",
                        "text_edit_content": "Apache-2.0",
                        "annotations": "Apache-2.0",
                    },
                }
            )
    if len(rows) != 2081:
        raise ValueError(f"Expected 2081 cases, found {len(rows)}")
    return rows


def serialize(rows):
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / DEFAULT_OUTPUT.relative_to(ROOT)
    content = serialize(build(root))
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Text provenance is stale: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    print(json.dumps({"cases": content.count("\n"), "output": str(output)}))


if __name__ == "__main__":
    main()
