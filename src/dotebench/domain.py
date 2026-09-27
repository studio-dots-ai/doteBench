"""Immutable benchmark records shared by every editing category."""

import io
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import soundfile as sf

from .instructions.types import ParsedInstruction


@dataclass(frozen=True)
class AudioAsset:
    path: str
    sha256: str


@dataclass(frozen=True)
class AlignmentUnit:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class SourceAlignment:
    audio_sha256: str
    text_sha256: str
    segments: tuple[AlignmentUnit, ...]


SpanGranularity = Literal["single_word", "short_phrase", "clause", "sentence"]
SourceEmotion = Literal["afraid", "angry", "happy", "melancholic", "sad", "surprised"]


@dataclass(frozen=True)
class Annotations:
    span_granularity: SpanGranularity | None = None
    source_emotion: SourceEmotion | None = None
    source_alignment: SourceAlignment | None = None


@dataclass(frozen=True)
class Case:
    id: str
    language: str
    source_audio: AudioAsset
    instruction_xml: str
    annotations: Annotations
    instruction: ParsedInstruction

    @property
    def source_text(self) -> str:
        return self.instruction.source_text

    @property
    def target_text(self) -> str:
        return self.instruction.target_text


class InfrastructureError(RuntimeError):
    """A transport or runtime failure that is not a candidate prediction."""


@dataclass(frozen=True)
class Audio:
    data: bytes

    def validate(self) -> None:
        if not isinstance(self.data, bytes):
            raise ValueError("Audio must contain immutable WAV bytes")
        with sf.SoundFile(io.BytesIO(self.data)) as stream:
            if stream.format not in {"WAV", "WAVEX", "RF64"} or stream.frames <= 0:
                raise ValueError("Expected nonempty WAV audio")
            samples = stream.read(dtype="float32")
            if not np.isfinite(samples).all():
                raise ValueError("Audio contains nonfinite samples")


@dataclass(frozen=True)
class GenerationRequest:
    id: str
    language: str
    source_audio: Audio
    instruction_xml: str


@dataclass(frozen=True)
class GenerationResult:
    id: str
    status: str
    audio_path: str | None = None
    audio_sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationRequest:
    case: Case
    generation: GenerationResult
    source_audio: Audio
    generated_audio: Audio | None


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    metrics: Mapping[str, object]
