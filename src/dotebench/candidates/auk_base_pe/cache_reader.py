"""Read terminal PE instructions and retain their on-disk provenance."""

import hashlib
import json
from pathlib import Path

from dotebench.domain import InfrastructureError

from .instruction_cache import digest


class InstructionCache:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        manifest, self.provenance = self._read(self.directory / "manifest.json")
        self.identity = manifest["identity"]
        if self.identity["schema"] != 1 or self.identity["scope"] != "instruction_only":
            raise InfrastructureError("Unsupported PE instruction cache schema")

    @staticmethod
    def _read(path):
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise InfrastructureError(f"Cannot read PE cache: {path}") from exc
        return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def lookup(self, item):
        # Use the build's recorded identity, including its original source hashes.
        key = digest({"identity": self.identity, "input": item})
        record, provenance = self._read(self.directory / "records" / f"{key}.json")
        if (
            record["key"] != key
            or record["identity"] != self.identity
            or record["input"] != item
        ):
            raise InfrastructureError(f"PE cache record identity mismatch: {key}")
        if record["status"] not in {"ok", "unsupported", "error"}:
            raise InfrastructureError(f"PE cache record is incomplete: {key}")
        return record, {**provenance, "key": key, "manifest": self.provenance}


def require_instruction(record):
    if record["status"] == "ok":
        return record["output"]
    attempt = record["attempts"][-1]
    message = f"PE {record['status']}: {attempt['error']}"
    if record["status"] == "error":
        raise InfrastructureError(message)
    # An official refusal is an outcome of this candidate.
    raise RuntimeError(message)
