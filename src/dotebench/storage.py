"""Run persistence, integrity checks, and audio ownership."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from dotebench.models.base import Audio, GenerationResult


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RunStore:
    def __init__(self, root: Path, audio_root: Path):
        self.root = Path(root)
        self.audio_root = Path(audio_root)

    def write_json(self, name: str, value: object) -> None:
        path = self.root / name
        temp = path.with_suffix(path.suffix + ".pending")
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        temp.replace(path)

    def read_json(self, name: str):
        return json.loads((self.root / name).read_text())

    def append(self, name: str, value: dict) -> None:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with (self.root / name).open("a") as stream:
            stream.write(encoded + "\n")
            stream.flush()

    def rows(self, name: str) -> list[dict]:
        return [
            json.loads(line) for line in (self.root / name).read_text().splitlines()
        ]

    def initialize(self, config: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "audio").mkdir()
        self.write_json("run.json", config)
        (self.root / "generation.jsonl").touch()

    def source(self, case) -> Audio:
        relative = Path(case.source_audio.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source audio path must be relative")
        data = (self.audio_root / relative).read_bytes()
        if sha256(data) != case.source_audio.sha256:
            raise ValueError(f"Source audio checksum mismatch: {case.id}")
        audio = Audio(data)
        audio.validate()
        return audio

    def save_audio(self, id: str, audio: Audio) -> GenerationResult:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", id) or id in {".", ".."}:
            raise ValueError("Unsafe case ID")
        relative = f"audio/{id}.wav"
        (self.root / relative).write_bytes(audio.data)
        return GenerationResult(id, "ok", relative, sha256(audio.data))

    def generated(self, result: GenerationResult) -> Audio | None:
        if result.status == "candidate_error":
            if result.audio_path is not None or result.audio_sha256 is not None:
                raise ValueError("Failed generation has unexpected audio")
            return None
        if result.status != "ok" or result.audio_path != f"audio/{result.id}.wav":
            raise ValueError("Invalid generation record")
        audio = Audio((self.root / result.audio_path).read_bytes())
        if sha256(audio.data) != result.audio_sha256:
            raise ValueError("Generated audio checksum mismatch")
        audio.validate()
        return audio

    def record_generation(self, result: GenerationResult) -> None:
        self.append("generation.jsonl", asdict(result))
