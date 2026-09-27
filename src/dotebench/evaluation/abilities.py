"""Transport for shared metric abilities; service faults propagate to the runner."""

import base64
from copy import deepcopy
from typing import Protocol

import requests

from dotebench.evaluation.models import validate_service_model
from dotebench.services.utils.http import request_json

ABILITY_NAMES = (
    "qwen3_asr",
    "qwen3_aligner",
    "wer",
    "f0",
    "wdtw",
    "utmos",
    "speaker_similarity",
    "emotion",
)


class MetricAbilities(Protocol):
    def identity(self) -> dict: ...
    def prepare(self): ...
    def measure(self, ability: str, **payload) -> dict: ...
    def close(self): ...


def encoded(audio):
    return base64.b64encode(audio).decode("ascii")


class HTTPMetricAbilities:
    def __init__(self, urls: dict[str, str], timeout: float = 300):
        missing = set(ABILITY_NAMES) - set(urls)
        if missing:
            raise ValueError(f"Missing metric abilities: {sorted(missing)}")
        self.urls = {k: v.rstrip("/") for k, v in urls.items()}
        self.timeout = timeout
        self.session = requests.Session()
        self._identity = {}

    def prepare(self):
        identities = {}
        for name in ABILITY_NAMES:
            data = request_json(
                self.session, "GET", self.urls[name] + "/health", timeout=self.timeout
            )
            if data.get("status") not in {"ok", "healthy", "ready"}:
                raise RuntimeError(f"Metric ability {name} is not ready: {data}")
            if data.get("ability") != name:
                raise RuntimeError(f"Metric ability identity mismatch for {name}")
            validate_service_model(name, data.get("model_identity"))
            if not isinstance(data.get("configuration"), dict):
                raise TypeError(f"Missing effective metric configuration: {name}")
            identities[name] = {
                "model_identity": data.get("model_identity"),
                "configuration": data.get("configuration", {}),
            }
        self._identity = identities

    def identity(self):
        return deepcopy(self._identity)

    def measure(self, ability, **payload):
        return request_json(
            self.session,
            "POST",
            self.urls[ability] + "/measure",
            payload=payload,
            timeout=self.timeout,
        )

    def close(self):
        self.session.close()
