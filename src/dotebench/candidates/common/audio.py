"""Audio payload encoding for native candidate services."""

import base64
import logging
import os
import uuid
from typing import Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EncodedAudio(BaseModel):
    """Base64-encoded audio bytes with format metadata."""

    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format: 'wav', 'mp3', 'flac', etc.")


class EncodedTensor(BaseModel):
    """Base64-encoded numpy array with metadata."""

    data: str = Field(..., description="Base64-encoded tensor bytes")
    shape: list[int] = Field(..., description="Tensor shape")
    dtype: str = Field("float32", description="Numpy dtype string")


def decode_audio_input(
    audio_input: Union["EncodedAudio", str],
    temp_dir: str,
    prefix: str = "audio",
) -> str:
    """Decode audio input to a file path.

    If *audio_input* is already a path string the path is returned unchanged
    (after verifying it exists).  If it is an ``EncodedAudio`` object the
    base64 payload is written to a temporary file in *temp_dir*.

    Args:
        audio_input: Either a file-path string or an ``EncodedAudio`` object.
        temp_dir: Directory for writing temporary decoded files.
        prefix: Filename prefix for the temp file.

    Returns:
        Absolute path to the audio file (either the original or the temp file).

    Raises:
        ValueError: If a path string does not exist or base64 decoding fails.
    """
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        return audio_input

    try:
        audio_bytes = base64.b64decode(audio_input.data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 audio: {e}")

    file_ext = audio_input.format.lower()
    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"
    temp_path = os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{file_ext}")

    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    return temp_path


def encode_numpy(arr: np.ndarray) -> EncodedTensor:
    """Encode a numpy array to a base64-encoded ``EncodedTensor``."""
    tensor_bytes = arr.tobytes()
    data_b64 = base64.b64encode(tensor_bytes).decode("utf-8")
    return EncodedTensor(
        data=data_b64,
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def cleanup_temp_files(
    paths: list[str],
    temp_dir: str | None = None,
) -> None:
    """Remove temporary files created during request handling.

    Only files whose path starts with *temp_dir* (when provided) are deleted.
    Errors are logged but never raised.

    Args:
        paths: List of file paths to consider for deletion.
        temp_dir: If given, only delete files under this directory.
    """
    logger = logging.getLogger(__name__)
    for path in paths:
        try:
            if not path or not os.path.exists(path):
                continue
            if temp_dir and not path.startswith(temp_dir):
                continue
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {path}: {e}")
