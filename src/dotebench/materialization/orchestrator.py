"""Materialize complete benchmark audio and source alignments."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from dotebench import RELEASE, dataset
from dotebench.alignment import validate_alignment

from .fingerprints import audio_digest, file_sha256
from .manifest import AssetSpec, MaterializationInventory, load_inventory
from .providers import synthesize

MATERIALIZER_IMPLEMENTATION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


def _validate_wav(data: bytes) -> None:
    with sf.SoundFile(io.BytesIO(data)) as handle:
        if handle.format != "WAV" or handle.frames <= 0 or handle.samplerate <= 0:
            raise ValueError("Expected a non-empty WAV file")


def _decode_common_voice(source: Path) -> bytes:
    """Reproduce the frozen ffmpeg MP3 decode and PCM16 WAV serialization."""
    info = sf.info(source)
    if info.channels < 1 or info.samplerate < 1:
        raise ValueError(f"Invalid Common Voice audio: {source}")
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
    )
    samples = np.frombuffer(raw, dtype="<i2")
    if info.channels > 1:
        if len(samples) % info.channels:
            raise ValueError("Decoded Common Voice channel count mismatch")
        samples = samples.reshape(-1, info.channels)
    output = io.BytesIO()
    sf.write(output, samples, info.samplerate, format="WAV", subtype="PCM_16")
    return output.getvalue()


class Materializer:
    def __init__(
        self,
        *,
        data_root: Path,
        output_root: Path,
        bundle_root: Path,
        upstream_root: Path,
        timeout: float = 1800,
    ):
        self.data_root = Path(data_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.bundle_root = Path(bundle_root).resolve()
        self.upstream_root = Path(upstream_root).resolve()
        self.timeout = timeout
        self.inventory: MaterializationInventory = load_inventory(
            data_root=self.data_root
        )
        self.paths: dict[str, Path] = {}
        self.records: dict[str, dict[str, object]] = {}
        self.service_identities: dict[str, object] = {}

    def _store(self, spec: AssetSpec, data: bytes) -> Path:
        _validate_wav(data)
        digest = hashlib.sha256(data).hexdigest()
        path = self.output_root / "audio" / f"{digest}.wav"
        if path.exists():
            if file_sha256(path) != digest:
                raise ValueError(f"Corrupt materialized audio: {path}")
        else:
            _atomic_bytes(path, data)
        self.paths[spec.asset_id] = path
        self.records[spec.asset_id] = {
            "kind": spec.kind,
            "provider": spec.provider,
            "reference_sha256": spec.reference_sha256,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
        return path

    def materialize_static(self) -> None:
        for spec in self.inventory.assets.values():
            if spec.kind == "bundle":
                source = dataset.relative_path(
                    self.bundle_root, spec.recipe["path"]
                ).resolve()
                if file_sha256(source) != spec.reference_sha256:
                    raise ValueError(f"Bundle audio checksum mismatch: {spec.asset_id}")
                self._store(spec, source.read_bytes())
            elif spec.kind == "upstream":
                source = dataset.relative_path(
                    self.upstream_root, spec.recipe["path"]
                ).resolve()
                if file_sha256(source) != spec.recipe["sha256"]:
                    raise ValueError(
                        f"Upstream audio checksum mismatch: {spec.asset_id}"
                    )
                data = _decode_common_voice(source)
                if hashlib.sha256(data).hexdigest() != spec.reference_sha256:
                    raise ValueError(
                        f"Common Voice conversion differs from the release reference: {spec.asset_id}"
                    )
                self._store(spec, data)

    def materialize_provider(self, provider: str, url: str) -> None:
        health = requests.get(url.rstrip("/") + "/health", timeout=30)
        health.raise_for_status()
        payload = health.json()
        if (
            not isinstance(payload, dict)
            or payload.get("status") != "ready"
            or payload.get("provider") != provider
            or not isinstance(payload.get("identity"), dict)
        ):
            raise RuntimeError(
                f"Materialization provider identity mismatch: {provider}"
            )
        self.service_identities[provider] = payload["identity"]
        for spec in self.inventory.assets.values():
            if spec.provider != provider:
                continue
            reference = spec.recipe.get("reference_asset_id")
            reference_path = self.paths.get(reference) if reference else None
            if reference and reference_path is None:
                raise ValueError(
                    f"Reference asset {reference!r} is not materialized for {spec.asset_id}"
                )
            self._store(
                spec,
                synthesize(
                    url,
                    spec,
                    reference_audio=reference_path,
                    timeout=self.timeout,
                ),
            )

    def _align(self, url: str, spec: AssetSpec, path: Path) -> list[dict]:
        response = requests.post(
            url.rstrip("/") + "/measure",
            json={
                "audio": base64.b64encode(path.read_bytes()).decode("ascii"),
                "text": spec.source_text,
                "language": spec.language,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        segments = result.get("segments") if isinstance(result, dict) else None
        if not isinstance(segments, list):
            raise TypeError("Aligner returned no segments")
        info = sf.info(path)
        validate_alignment(spec.source_text, segments, info.frames / info.samplerate)
        return segments

    def register_aligner(self, url: str) -> None:
        health = requests.get(url.rstrip("/") + "/health", timeout=30)
        health.raise_for_status()
        payload = health.json()
        if not isinstance(payload, dict) or payload.get("status") not in {
            "ready",
            "healthy",
        }:
            raise RuntimeError("Materialization aligner is not ready")
        self.service_identities["qwen3_aligner"] = payload

    def finalize(self, *, aligner_url: str | None) -> dict[str, object]:
        if set(self.paths) != set(self.inventory.assets):
            missing = sorted(set(self.inventory.assets) - set(self.paths))
            raise ValueError(f"Materialization is incomplete: {missing[:5]}")
        alignments: dict[str, list[dict]] = {}
        alignment_counts = {"reused": 0, "generated": 0}
        for source_path, original in self.inventory.manifests.items():
            payload = copy.deepcopy(original)
            for case in payload["cases"]:
                asset_id = case["source_audio"]["asset_id"]
                spec = self.inventory.assets[asset_id]
                actual = self.records[asset_id]["sha256"]
                path = self.paths[asset_id]
                case["source_audio"]["path"] = f"audio/{actual}.wav"
                case["source_audio"]["sha256"] = actual
                alignment = case["annotations"]["source_alignment"]
                if alignment["audio_sha256"] == actual:
                    alignment_counts["reused"] += 1
                else:
                    if aligner_url is None:
                        raise ValueError("Changed audio requires a configured aligner")
                    if asset_id not in alignments:
                        alignments[asset_id] = self._align(aligner_url, spec, path)
                    alignment["audio_sha256"] = actual
                    alignment["segments"] = alignments[asset_id]
                    alignment_counts["generated"] += 1
            relative = source_path.relative_to(self.data_root)
            _atomic_json(self.output_root / relative, payload)
        manifests = {
            str(path.relative_to(self.output_root)): file_sha256(path)
            for path in sorted((self.output_root / "data").glob("**/manifest.json"))
        }
        receipt = {
            "schema_version": 1,
            "version": dataset.data_version(),
            "materializer": {
                "release": RELEASE,
                "module": __name__,
                "sha256": MATERIALIZER_IMPLEMENTATION,
            },
            "audio": dict(sorted(self.records.items())),
            "audio_digest": audio_digest(self.records),
            "manifests": manifests,
            "alignment": alignment_counts,
            "services": dict(sorted(self.service_identities.items())),
        }
        _atomic_json(self.output_root / "materialization.json", receipt)
        return receipt
