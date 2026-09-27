"""Step-Audio-EditX FastAPI service.

Exposes zero-shot TTS/voice cloning and speech editing endpoints for use as a
baseline tool. Model loading is explicit at startup because the checkpoints are
large and must be downloaded separately.
"""

from dotebench import RELEASE

import argparse
import base64
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dotebench.candidates.common.runtime import activate_runtime, runtime_identity

TOOL_ROOT = activate_runtime("step_audio_editx")
RUNTIME_IDENTITY = runtime_identity("step_audio_editx")
PROJECT_ROOT = Path(os.environ.get("CHECKPOINTS_DIR", ".")).resolve().parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_ARGS = {}
OUTPUT_ROOT = Path(os.environ["DOTEBENCH_SERVICE_OUTPUTS"])


class TTSRequest(BaseModel):
    prompt_audio_path: str = Field(..., description="Reference speech wav path.")
    prompt_text: str = Field(..., description="Transcript of prompt_audio_path.")
    text: str = Field(
        ..., description="Target text to synthesize with the prompt voice."
    )
    max_new_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description="Optional generation-token cap. Omit to retain the model default.",
    )


class EditRequest(BaseModel):
    audio_path: str = Field(..., description="Speech audio path to edit.")
    transcript: str = Field(
        "", description="Transcript of audio_path. Not required for denoise/vad."
    )
    edit_type: Literal["emotion", "style", "vad", "denoise", "paralinguistic", "speed"]
    edit_info: str | None = Field(
        None, description="Subtype such as happy, whisper, faster."
    )
    target_text: str | None = Field(
        None, description="Target text for paralinguistic editing."
    )
    max_new_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description="Optional generation-token cap. Omit to retain the model default.",
    )


class FreeformEditRequest(BaseModel):
    audio_path: str = Field(..., description="Speech audio path to edit.")
    instruction: str = Field(
        ...,
        min_length=1,
        description="Natural-language instruction for the model-native edit slot.",
    )
    max_new_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description="Optional generation-token cap. Omit to retain the model default.",
    )


class AudioResponse(BaseModel):
    status: str
    audio_base64: str
    sample_rate: int
    output_path: str
    max_new_tokens: int | None = None
    generated_token_count: int | None = None
    decoded_audio_token_count: int | None = None
    finish_reason: str | None = None
    truncated_by_token_limit: bool = False


def _load_model(args) -> None:
    global MODEL, MODEL_ARGS
    if MODEL is not None:
        return
    if not args.model_path or not args.tokenizer_path:
        logger.info(
            "Step-Audio-EditX model paths not provided; service starts in not_loaded mode"
        )
        return
    from tokenizer import StepAudioTokenizer
    from tts import StepAudioTTS

    tokenizer = StepAudioTokenizer(args.tokenizer_path, model_source=args.model_source)
    MODEL = StepAudioTTS(
        args.model_path,
        tokenizer,
        model_source=args.model_source,
        tts_model_id=args.tts_model_id,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        attention_backend=args.attention_backend,
        cosyvoice_dtype=args.cosyvoice_dtype,
        cosyvoice_cuda_graph=not args.no_cosyvoice_cuda_graph,
    )
    MODEL_ARGS = vars(args)


def _save_response(
    audio: Any,
    sample_rate: int,
    prefix: str,
    generation_metadata: dict[str, Any] | None = None,
) -> AudioResponse:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fd, out_path = tempfile.mkstemp(
        prefix=f"{prefix}_", suffix=".wav", dir=str(OUTPUT_ROOT)
    )
    os.close(fd)
    array = audio.detach().cpu().float().numpy().squeeze()
    sf.write(out_path, array, sample_rate)
    with open(out_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    metadata = generation_metadata or {}
    return AudioResponse(
        status="success",
        audio_base64=audio_base64,
        sample_rate=sample_rate,
        output_path=out_path,
        max_new_tokens=metadata.get("max_new_tokens"),
        generated_token_count=metadata.get("generated_token_count"),
        decoded_audio_token_count=metadata.get("decoded_audio_token_count"),
        finish_reason=metadata.get("finish_reason"),
        truncated_by_token_limit=bool(metadata.get("truncated_by_token_limit", False)),
    )


def _prepare_runtime_cache_dirs(port: int) -> None:
    cache_root = Path(os.environ["DOTEBENCH_SERVICE_OUTPUTS"]) / "cache" / str(port)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    for key in ("XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _warmup_audio_runtime() -> None:
    import librosa

    audio = np.zeros(16000, dtype=np.float32)
    librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=128)
    logger.info("Audio runtime warmup completed")


app = FastAPI(title="Step-Audio-EditX Service", version=RELEASE)


@app.get("/health")
@app.post("/health")
def health():
    return {
        "runtime": RUNTIME_IDENTITY,
        "status": "healthy" if MODEL is not None else "not_loaded",
        "service": "step-audio-editx",
        "model_loaded": MODEL is not None,
        "default_port": 8051,
        "model_args": {
            k: v
            for k, v in MODEL_ARGS.items()
            if k in {"model_path", "tokenizer_path", "model_source"}
        },
    }


@app.get("/info")
def info():
    return {
        "service": "step-audio-editx",
        "supports": {
            "tts": True,
            "text_edit": False,
            "emotion_edit": True,
            "pause_edit": "via vad silence removal only; no targeted pause insertion",
            "prosody_edit": True,
            "speech_rate_edit": True,
            "paralinguistic_edit": True,
            "freeform_edit": True,
        },
        "endpoints": ["/tts", "/edit", "/edit_freeform"],
    }


@app.post("/tts", response_model=AudioResponse)
def tts(req: TTSRequest):
    if MODEL is None:
        raise HTTPException(
            503, "Model is not loaded. Start with --model-path and --tokenizer-path."
        )
    with MODEL_LOCK:
        audio, sr, metadata = MODEL.clone(
            req.prompt_audio_path,
            req.prompt_text,
            req.text,
            max_new_tokens=req.max_new_tokens,
            return_generation_metadata=True,
        )
    return _save_response(audio, sr, "tts", metadata)


@app.post("/edit", response_model=AudioResponse)
def edit(req: EditRequest):
    if MODEL is None:
        raise HTTPException(
            503, "Model is not loaded. Start with --model-path and --tokenizer-path."
        )
    with MODEL_LOCK:
        audio, sr, metadata = MODEL.edit(
            prompt_wav_path=req.audio_path,
            prompt_text=req.transcript,
            edit_type=req.edit_type,
            edit_info=req.edit_info,
            target_text=req.target_text,
            max_new_tokens=req.max_new_tokens,
            return_generation_metadata=True,
        )
    return _save_response(audio, sr, "edit", metadata)


@app.post("/edit_freeform", response_model=AudioResponse)
def edit_freeform(req: FreeformEditRequest):
    if MODEL is None:
        raise HTTPException(
            503, "Model is not loaded. Start with --model-path and --tokenizer-path."
        )
    with MODEL_LOCK:
        audio, sr, metadata = MODEL.edit_freeform(
            prompt_wav_path=req.audio_path,
            instruction=req.instruction,
            max_new_tokens=req.max_new_tokens,
            return_generation_metadata=True,
        )
    return _save_response(audio, sr, "edit_freeform", metadata)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "STEP_AUDIO_EDITX_MODEL_PATH",
            str(PROJECT_ROOT / "checkpoints/Step-Audio-EditX"),
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get(
            "STEP_AUDIO_TOKENIZER_PATH",
            str(PROJECT_ROOT / "checkpoints/Step-Audio-Tokenizer"),
        ),
    )
    parser.add_argument(
        "--model-source",
        default="local",
        choices=["auto", "local", "modelscope", "huggingface"],
    )
    parser.add_argument("--tts-model-id", default=None)
    parser.add_argument("--quantization", default=None, choices=["awq", "gptq", "fp8"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
    )
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument(
        "--cosyvoice-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--no-cosyvoice-cuda-graph", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _prepare_runtime_cache_dirs(args.port)
    _warmup_audio_runtime()
    _load_model(args)
    uvicorn.run(app, host=args.host, port=args.port)
