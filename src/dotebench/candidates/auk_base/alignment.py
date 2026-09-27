"""Candidate-owned online alignment and content-addressed inference records."""

import base64
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import requests
import soundfile as sf

from dotebench.alignment import validate_alignment
from dotebench.domain import (
    AlignmentUnit,
    InfrastructureError,
    SourceAlignment,
)


def validate_candidate_alignment(alignment, audio, text):
    if alignment is None:
        raise ValueError("AuK Base requires a supported candidate source alignment")
    if alignment.audio_sha256 != hashlib.sha256(audio.data).hexdigest():
        raise ValueError("Source alignment audio checksum mismatch")
    if alignment.text_sha256 != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError("Source alignment transcript checksum mismatch")
    return validate_alignment(
        text, alignment.segments, sf.info(io.BytesIO(audio.data)).duration
    )


class CandidateAligner:
    def __init__(self, url, cache_dir, timeout=600):
        self.url = url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = requests.Session()
        self.identity = None

    def prepare(self):
        try:
            response = self.session.get(self.url + "/health", timeout=self.timeout)
            response.raise_for_status()
            health = response.json()
            if (
                health.get("status") != "ready"
                or health.get("role") != "candidate-aligner"
            ):
                raise ValueError("Unexpected aligner service")
            identity = health["identity"]
            if not isinstance(identity, dict) or not identity.get("weights_sha256"):
                raise ValueError("Missing alignment model identity")
            if self.identity is not None and identity != self.identity:
                raise ValueError("Candidate aligner identity changed")
            self.identity = identity
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise InfrastructureError(
                "Candidate aligner unavailable or identity changed"
            ) from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def align(self, audio, text, language):
        try:
            return self._align(audio, text, language)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InfrastructureError(
                "Candidate alignment storage or response is invalid"
            ) from exc

    def _align(self, audio, text, language):
        self.prepare()
        inputs = dict(
            audio_sha256=hashlib.sha256(audio.data).hexdigest(),
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            language=language,
            aligner=self.identity,
        )
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / (key + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record["inputs"] != inputs:
                raise InfrastructureError("Candidate alignment cache identity mismatch")
            raw = record["alignment"]
            if (
                record["alignment_sha256"]
                != hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            ):
                raise InfrastructureError("Candidate alignment cache checksum mismatch")
        else:
            try:
                response = self.session.post(
                    self.url + "/align",
                    json={
                        "audio": base64.b64encode(audio.data).decode(),
                        "text": text,
                        "language": language,
                        "identity": self.identity,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                if result.get("identity") != self.identity:
                    raise ValueError("Alignment response identity mismatch")
            except (requests.RequestException, ValueError) as exc:
                raise InfrastructureError("Candidate alignment service failed") from exc
            # A valid inference response that cannot locate the words is a candidate failure.
            units = tuple(
                AlignmentUnit(s["word"], s["start"], s["end"])
                for s in result["segments"]
            )
            alignment = SourceAlignment(
                inputs["audio_sha256"],
                inputs["text_sha256"],
                units,
            )
            validate_candidate_alignment(alignment, audio, text)
            raw = json.loads(json.dumps(asdict(alignment)))
            record = {
                "inputs": inputs,
                "alignment": raw,
                "alignment_sha256": hashlib.sha256(
                    json.dumps(raw, sort_keys=True).encode()
                ).hexdigest(),
            }
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(
                mode="w", dir=self.cache_dir, delete=False
            ) as stream:
                json.dump(record, stream, ensure_ascii=False)
                temp = Path(stream.name)
            temp.replace(path)
        if set(raw) != {"audio_sha256", "text_sha256", "segments"}:
            raise ValueError("Unexpected candidate alignment fields")
        alignment = SourceAlignment(
            raw["audio_sha256"],
            raw["text_sha256"],
            tuple(AlignmentUnit(**s) for s in raw["segments"]),
        )
        validate_candidate_alignment(alignment, audio, text)
        return alignment, {"cache_key": key, **record}

    def close(self):
        self.session.close()
