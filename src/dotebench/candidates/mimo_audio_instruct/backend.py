"""MiMo-Audio-Instruct native TTS service."""

from dotebench import RELEASE

import argparse
import io
import logging
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

from dotebench.candidates.common.audio import (
    EncodedAudio,
    EncodedTensor,
    cleanup_temp_files,
    decode_audio_input,
    encode_numpy,
)
from dotebench.candidates.common.runtime import activate_runtime, runtime_identity

TOOL_ROOT = activate_runtime("mimo_audio_instruct")
RUNTIME_IDENTITY = runtime_identity("mimo_audio_instruct")
logger = logging.getLogger(__name__)
app = FastAPI(version=RELEASE, title="MiMo-Audio-Instruct")
TEMP_DIR = os.environ["DOTEBENCH_SERVICE_OUTPUTS"]
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
MODEL = None
MODEL_INFO = {
    "model_repo_id": "XiaomiMiMo/MiMo-Audio-7B-Instruct",
    "variant": "instruct",
    "model_path": None,
    "tokenizer_path": None,
}


class TTSRequest(BaseModel):
    """Request for instruct TTS synthesis."""

    text: str = Field(
        ...,
        description="Text to synthesize into speech.",
    )
    instruct: str | None = Field(
        None,
        description="Style instruction for the speech (e.g. 'Say it happily in a child voice'). "
        "When set, enables instruct-TTS mode.",
    )
    read_text_only: bool = Field(
        True,
        description="If True (default), treat `text` as pure text to read aloud. "
        "If False, `text` may contain embedded natural instructions "
        "(e.g. '用气喘吁吁的声音说：我跑不动了').",
    )
    prompt_speech: EncodedAudio | None = Field(
        None,
        description="Voice cloning prompt audio. When provided, the model generates speech "
        "with the same timbre as this prompt.",
    )
    output_format: str = Field(
        "wav",
        description="Output format: 'wav' (binary WAV response) or 'numpy' (JSON with encoded tensor).",
    )
    max_new_tokens: int = Field(
        8192,
        ge=256,
        le=32768,
        description="Maximum number of new tokens to generate.",
    )
    seed: int | None = Field(
        None,
        description="Random seed for inference. When set, fixes all random states "
        "(Python, NumPy, PyTorch, CUDA) with cudNN determinism enabled. "
        "Reduces but does not fully eliminate GPU non-determinism.",
    )

    @model_validator(mode="after")
    def check_output_format(self):
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(
                f"output_format must be 'wav' or 'numpy', got '{self.output_format}'"
            )
        return self


class TTSResponse(BaseModel):
    """Response for TTS (numpy output format)."""

    status: str = Field("success")
    text_channel: str = Field(
        ..., description="Decoded text from the model's text channel."
    )
    audio: EncodedTensor = Field(
        ..., description="Generated audio waveform (24kHz, float32)."
    )
    sample_rate: int = Field(24000, description="Audio sample rate.")
    max_new_tokens: int
    generated_step_count: int
    decoded_audio_frame_count: int
    output_duration_seconds: float
    truncated_by_token_limit: bool


def _set_seed(seed: int) -> None:
    """Set all random seeds and enable CUDA determinism for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # cudNN determinism — may slightly reduce throughput but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _wav_tensor_to_bytes(wav: torch.Tensor, sample_rate: int = 24000) -> bytes:
    """Convert a 1-D waveform tensor to broadly compatible PCM16 WAV bytes."""
    buf = io.BytesIO()
    arr = wav.reshape(-1).detach().cpu().numpy()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _generation_headers(metadata: dict[str, object]) -> dict[str, str]:
    """Expose bounded-generation provenance without changing binary WAV bodies."""

    return {
        "X-Max-New-Tokens": str(metadata.get("max_new_tokens", "")),
        "X-Generated-Step-Count": str(metadata.get("generated_step_count", "")),
        "X-Decoded-Audio-Frame-Count": str(
            metadata.get("decoded_audio_frame_count", "")
        ),
        "X-Audio-Token-Frame-Rate-Hz": str(
            metadata.get("audio_token_frame_rate_hz", "")
        ),
        "X-Output-Duration-Seconds": str(metadata.get("output_duration_seconds", "")),
        "X-Truncated-By-Token-Limit": str(
            bool(metadata.get("truncated_by_token_limit", False))
        ).lower(),
    }


@app.get("/health")
async def health():
    return {
        "runtime": RUNTIME_IDENTITY,
        "model_loaded": MODEL is not None,
        "status": "healthy" if MODEL is not None else "loading",
        "service": "MiMo-Audio",
        "model": MODEL_INFO["model_repo_id"],
        "model_variant": MODEL_INFO["variant"],
        "model_path": MODEL_INFO["model_path"],
        "tokenizer_path": MODEL_INFO["tokenizer_path"],
        "device": str(MODEL.device) if MODEL is not None else "unknown",
    }


@app.post("/tts")
def tts(request: TTSRequest):
    """
    Instruct TTS: generate speech from text with optional style control and voice cloning.

    Behavior modes (determined by parameter combinations):
      - text only: basic TTS (model picks voice/style)
      - text + instruct: instruct TTS (style-controlled)
      - text + instruct + prompt_speech: instruct TTS with voice cloning
      - text + instruct="" + prompt_speech: voice cloning TTS
      - read_text_only=False: natural instruction TTS (text contains embedded style info)

    Returns:
      - If output_format='wav': Binary WAV file response (audio/wav)
      - If output_format='numpy': JSON TTSResponse with encoded waveform
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    temp_files = []
    prompt_speech_path = None

    try:
        # Decode prompt speech if provided
        if request.prompt_speech is not None:
            prompt_speech_path = decode_audio_input(
                request.prompt_speech, TEMP_DIR, prefix="prompt"
            )
            temp_files.append(prompt_speech_path)

        logger.info(
            f"TTS request: text='{request.text[:80]}...', "
            f"instruct={'yes' if request.instruct else 'no'}, "
            f"prompt_speech={'yes' if prompt_speech_path else 'no'}, "
            f"read_text_only={request.read_text_only}, "
            f"seed={request.seed}"
        )

        from src.mimo_audio.modeling_mimo_audio import MiMoStopper

        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[
                    MODEL.tokenizer.eos_token_id,
                    MODEL.eostm_idx,
                    MODEL.im_end_idx,
                ],
                group_size=MODEL.group_size,
                audio_channels=MODEL.audio_channels,
            )
        ]

        # The official prompt builder selects a chat template with Python's
        # random module. Seed before prompt construction so a request-local
        # seed also fixes the actual prompt, not only token sampling.
        if request.seed is not None:
            _set_seed(request.seed)
        input_ids = MODEL.get_tts_sft_prompt(
            request.text,
            instruct=request.instruct,
            read_text_only=request.read_text_only,
            prompt_speech=prompt_speech_path,
        )

        # Generate — use official sampling params; seed enables CUDA determinism
        with torch.no_grad():
            wav, generation_metadata = MODEL.forward(
                input_ids,
                return_audio=True,
                output_audio_path=None,
                stopping_criteria=stopping_criteria,
                max_new_tokens=request.max_new_tokens,
                task_name="tts",
                return_generation_metadata=True,
            )

        # forward() returns text_channel string when output_audio_path is set,
        # but returns wav tensor when output_audio_path is None
        if isinstance(wav, str):
            # Model produced no audio (text-only output)
            raise HTTPException(
                status_code=500,
                detail=f"Model did not generate audio. Text channel: {wav}",
            )

        wav_1d = wav.reshape(-1).detach().cpu()
        generation_metadata["output_duration_seconds"] = float(wav_1d.numel()) / 24000.0

        if request.output_format == "wav":
            wav_bytes = _wav_tensor_to_bytes(wav_1d)
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=tts_output.wav",
                    "X-Sample-Rate": "24000",
                    **_generation_headers(generation_metadata),
                },
            )
        else:
            return TTSResponse(
                status="success",
                text_channel="",
                audio=encode_numpy(wav_1d.numpy().astype(np.float32)),
                sample_rate=24000,
                max_new_tokens=int(generation_metadata["max_new_tokens"]),
                generated_step_count=int(generation_metadata["generated_step_count"]),
                decoded_audio_frame_count=int(
                    generation_metadata.get("decoded_audio_frame_count", 0)
                ),
                output_duration_seconds=float(
                    generation_metadata["output_duration_seconds"]
                ),
                truncated_by_token_limit=bool(
                    generation_metadata["truncated_by_token_limit"]
                ),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
    finally:
        cleanup_temp_files(temp_files, temp_dir=TEMP_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for value in (args.model_path, args.tokenizer_path):
        if not Path(value).is_dir():
            raise ValueError(f"Existing checkpoint directory required: {value}")
    from src.mimo_audio.mimo_audio import MimoAudio

    MODEL = MimoAudio(args.model_path, args.tokenizer_path, device=args.device)
    MODEL_INFO.update(model_path=args.model_path, tokenizer_path=args.tokenizer_path)
    uvicorn.run(app, host=args.host, port=args.port)
