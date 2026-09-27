"""Native request value type and deterministic audio transport mapping."""

import base64
from dataclasses import dataclass
from pathlib import Path

from dotebench.domain import Audio


@dataclass(frozen=True)
class NativeRequest:
    endpoint: str
    fields: tuple[tuple[str, object], ...]
    audio_field: str
    audio_encoding: str
    response_encoding: str
    selected_operation_index: int | None
    scope: str
    transport: str = "http"
    case_id: str | None = None

    def payload(self, audio: Audio, path: Path) -> dict:
        payload = dict(self.fields)
        if self.audio_encoding == "file":
            payload[self.audio_field] = str(path)
        elif self.audio_encoding == "base64":
            payload[self.audio_field] = {
                "data": base64.b64encode(audio.data).decode("ascii"),
                "format": "wav",
            }
        else:
            raise ValueError("Unsupported native audio encoding")
        return payload
