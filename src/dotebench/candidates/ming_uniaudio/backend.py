"""FastAPI service for Ming-UniAudio free-form speech editing.

The service intentionally preserves the upstream message contract: one HUMAN
message containing the source audio followed by a ``<prompt>`` text part. It
does not add a system prompt or infer task-specific transcript fields.
"""

from dotebench import RELEASE

import argparse
import base64
import logging
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from dotebench.candidates.common.runtime import activate_runtime, runtime_identity

TOOL_ROOT = activate_runtime("ming_uniaudio")
RUNTIME_IDENTITY = runtime_identity("ming_uniaudio")

from generation_errors import (
    NoAudioTokensError,
)

PROJECT_ROOT = Path(os.environ.get("CHECKPOINTS_DIR", ".")).resolve().parent
DEFAULT_MODEL_ID = "inclusionAI/Ming-UniAudio-16B-A3B-Edit"
DEFAULT_REVISION = "c7a9a8ba6816fd85b5bbb591efe248ce8d5e2ecb"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "Ming-UniAudio-16B-A3B-Edit"
OUTPUT_ROOT = Path(os.environ["DOTEBENCH_SERVICE_OUTPUTS"])

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL: Any | None = None
MODEL_LOCK = threading.Lock()
MODEL_INFO: dict[str, Any] = {
    "model_id": DEFAULT_MODEL_ID,
    "revision": DEFAULT_REVISION,
    "model_path": str(DEFAULT_MODEL_PATH),
}


class EditRequest(BaseModel):
    audio_path: str = Field(..., description="Path to the source speech audio.")
    instruction: str = Field(
        ..., description="Natural-language free-form edit instruction."
    )
    target_text: str | None = Field(
        None,
        description="Optional benchmark provenance only; never appended to the model instruction.",
    )
    seed: int = Field(1895, ge=0, le=2**32 - 1)
    use_cot: bool = True
    max_audio_seconds: float | None = Field(
        None,
        gt=0,
        description="Optional output-duration cap; omitted requests keep upstream behavior.",
    )

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be empty")
        if "<prompt>" in value or "</prompt>" in value:
            raise ValueError("instruction must not contain <prompt> tags")
        return value.strip()


class EditResponse(BaseModel):
    status: str
    audio_base64: str
    output_path: str
    edited_text: str
    sample_rate: int
    model_id: str
    revision: str
    decoding: dict[str, Any]
    generation: dict[str, Any]


def build_edit_messages(
    audio_path: str, instruction: str, *, target_sample_rate: int = 16000
) -> list[dict]:
    """Build the exact upstream single-HUMAN edit message."""
    return [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "audio",
                    "audio": audio_path,
                    "target_sample_rate": target_sample_rate,
                },
                {
                    "type": "text",
                    "text": f"<prompt>{instruction}\n</prompt>",
                },
            ],
        }
    ]


def seed_everything(seed: int) -> None:
    """Match the upstream cookbook seeding without importing torch at module load."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extend_audio_rope_cache(model, length: int = 8192) -> None:
    """Extend audio rotary tables while preserving their existing entries."""
    import torch
    from torchtune.modules import RotaryPositionalEmbeddings

    for module in model.audio.decoder.semantic_model.modules():
        if not isinstance(module, RotaryPositionalEmbeddings):
            continue
        previous = module.cache
        if previous.shape[0] >= length:
            continue
        # Recreate frequencies in FP32: loaded buffers may already be BF16.
        with torch.device("cpu"):
            fresh = RotaryPositionalEmbeddings(
                dim=module.dim, max_seq_len=length, base=module.base
            )
        expanded = fresh.cache.to(device=previous.device, dtype=previous.dtype)
        expanded[: previous.shape[0]].copy_(previous)
        module.register_buffer("cache", expanded, persistent=False)
        module.max_seq_len = length


class MingEditModel:
    """Minimal inference wrapper around the upstream ``generate_edit`` core."""

    def __init__(
        self, model_path: str, device: str = "cuda:0", use_grouped_gemm: bool = False
    ):
        import torch
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        from transformers import AutoProcessor

        self.device = device
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        extend_audio_rope_cache(self.model)
        self.model = self.model.to(device)
        if use_grouped_gemm and not self.model.config.llm_config.use_grouped_gemm:
            self.model.model.fuse_experts()
        self.model = self.model.eval().to(torch.bfloat16).to(device)
        self.processor = AutoProcessor.from_pretrained(
            str(TOOL_ROOT), trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.sample_rate = self.processor.audio_processor.sample_rate

    def speech_edit(
        self,
        messages: list[dict],
        output_wav_path: str,
        use_cot: bool = True,
        max_audio_seconds: float | None = None,
    ):
        import torch

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(
            messages
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(self.device)

        if use_cot:
            answer = torch.tensor(
                [self.tokenizer.encode("<answer>")], device=inputs["input_ids"].device
            )
            inputs["input_ids"] = torch.cat([inputs["input_ids"], answer], dim=1)
            inputs["attention_mask"] = torch.ones(
                inputs["input_ids"].shape,
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            )
        for key in inputs:
            if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
                inputs[key] = inputs[key].to(dtype=torch.bfloat16)

        result = self.model.generate_edit(
            **inputs,
            tokenizer=self.tokenizer,
            output_wav_path=output_wav_path,
            max_audio_seconds=max_audio_seconds,
        )
        self.last_edit_metadata = dict(getattr(self.model, "last_edit_metadata", {}))
        return result


def load_model(
    model_path: str,
    *,
    device: str,
    use_grouped_gemm: bool,
    model_id: str,
    revision: str,
) -> None:
    global MODEL, MODEL_INFO
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        logger.warning(
            "Ming-UniAudio checkpoint is missing; service starts in not_loaded mode: %s",
            path,
        )
        MODEL = None
        MODEL_INFO = {
            "model_id": model_id,
            "revision": revision,
            "model_path": str(path),
        }
        return
    MODEL = MingEditModel(str(path), device=device, use_grouped_gemm=use_grouped_gemm)
    MODEL_INFO = {
        "model_id": model_id,
        "revision": revision,
        "model_path": str(path),
        "device": device,
        "use_grouped_gemm": use_grouped_gemm,
    }


app = FastAPI(title="Ming-UniAudio Edit Service", version=RELEASE)


@app.get("/health")
@app.post("/health")
def health() -> dict[str, Any]:
    return {
        "runtime": RUNTIME_IDENTITY,
        "status": "healthy" if MODEL is not None else "not_loaded",
        "service": "ming-uniaudio",
        "model_loaded": MODEL is not None,
        "default_port": 8056,
        **MODEL_INFO,
    }


@app.post("/edit", response_model=EditResponse)
def edit(req: EditRequest) -> EditResponse:
    if MODEL is None:
        raise HTTPException(
            503, "Model is not loaded. Start the service with a valid --model-path."
        )
    source_path = Path(req.audio_path).expanduser().resolve()
    if not source_path.is_file():
        raise HTTPException(400, f"Source audio does not exist: {source_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fd, output_path = tempfile.mkstemp(
        prefix="edit_", suffix=".wav", dir=str(OUTPUT_ROOT)
    )
    os.close(fd)
    try:
        messages = build_edit_messages(str(source_path), req.instruction)
        with MODEL_LOCK:
            seed_everything(req.seed)
            _edited_speech, edited_text = MODEL.speech_edit(
                messages=messages,
                output_wav_path=output_path,
                use_cot=req.use_cot,
                max_audio_seconds=req.max_audio_seconds,
            )
        output = Path(output_path)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Ming-UniAudio did not produce a non-empty WAV file")
        audio_base64 = base64.b64encode(output.read_bytes()).decode("ascii")
        sample_rate = int(getattr(MODEL, "sample_rate", 16000))
        return EditResponse(
            status="success",
            audio_base64=audio_base64,
            output_path=str(output),
            edited_text=str(edited_text),
            sample_rate=sample_rate,
            model_id=str(MODEL_INFO["model_id"]),
            revision=str(MODEL_INFO["revision"]),
            decoding={
                "seed": req.seed,
                "use_cot": req.use_cot,
                "text_decoding": "greedy",
                "audio_cfg": 2,
                "system_prompt": None,
                "target_text_forwarded": False,
            },
            generation=dict(getattr(MODEL, "last_edit_metadata", {})),
        )
    except HTTPException:
        raise
    except NoAudioTokensError as exc:
        logger.warning(
            "Ming-UniAudio produced no audio tokens: %s",
            exc.diagnostics,
        )
        Path(output_path).unlink(missing_ok=True)
        raise HTTPException(
            500,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
                **exc.diagnostics,
            },
        ) from exc
    except Exception as exc:
        logger.exception("Ming-UniAudio edit failed")
        Path(output_path).unlink(missing_ok=True)
        raise HTTPException(500, f"Ming-UniAudio edit failed: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8056)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MING_UNIAUDIO_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision", default=os.environ.get("MING_UNIAUDIO_REVISION", DEFAULT_REVISION)
    )
    parser.add_argument(
        "--device", default=os.environ.get("MING_UNIAUDIO_DEVICE", "cuda:0")
    )
    parser.add_argument("--use-grouped-gemm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(
        args.model_path,
        device=args.device,
        use_grouped_gemm=args.use_grouped_gemm,
        model_id=args.model_id,
        revision=args.revision,
    )
    uvicorn.run(app, host=args.host, port=args.port)
