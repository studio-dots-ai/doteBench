"""Serial, single-call native FireRedTTS3-Instruct editing service."""

from dotebench import RELEASE

import argparse
import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_REVISION = "dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610"


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    audio_path: str
    instruction: str = Field(min_length=1)
    seed: int = Field(1234, ge=0, le=2**32 - 1)
    n_timesteps: int = Field(10, gt=0)
    inference_cfg: float = Field(1.2, ge=0, allow_inf_nan=False)


def create_app(model, identity, output_root):
    app = FastAPI(version=RELEASE, title="FireRedTTS3-Instruct")
    lock = threading.Lock()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model_loaded": True,
            "model": "FireRedTTS3-Instruct",
            "revision": MODEL_REVISION,
            "runtime": identity,
        }

    def edit(req, semantic):
        import torch

        path = Path(req.audio_path)
        try:
            samples, sr = sf.read(path, dtype="float32", always_2d=True)
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError("Invalid source audio")
        except (OSError, ValueError, sf.LibsndfileError) as exc:
            raise HTTPException(400, str(exc)) from exc
        # Preserve the full recording; the official backend handles resampling.
        audio = torch.from_numpy(samples.T.copy())
        started = time.time()
        with lock, torch.inference_mode():
            # Also seed before encoding, so future stochastic encoder changes remain reproducible.
            from fireredtts3.utils.utils import fix_seed

            fix_seed(req.seed)
            method = (
                model.generate_semantic_edit
                if semantic
                else model.generate_acoustic_edit
            )
            try:
                result = method(
                    instruction=req.instruction,
                    audio_in=audio,
                    audio_in_sr=sr,
                    n_timesteps=req.n_timesteps,
                    inference_cfg=req.inference_cfg,
                    seed=req.seed,
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise HTTPException(503, "GPU out of memory") from exc
            wave, sample_rate = result[:2]
            array = wave.detach().float().cpu().numpy()
            if array.ndim != 2 or not array.shape[1] or not np.isfinite(array).all():
                raise HTTPException(
                    500,
                    detail={
                        "error_code": "generation_failed",
                        "message": "Model returned empty or nonfinite audio",
                    },
                )
            buffer = io.BytesIO()
            sf.write(buffer, array.T, int(sample_rate), format="WAV", subtype="FLOAT")
            identifier = uuid.uuid4().hex
            metadata = {
                "case_id": req.case_id,
                "instruction": req.instruction,
                "method": method.__name__,
                "seed": req.seed,
                "n_timesteps": req.n_timesteps,
                "inference_cfg": req.inference_cfg,
                "input_samples": len(samples),
                "input_sample_rate": sr,
                "output_samples": array.shape[1],
                "sample_rate": int(sample_rate),
                "generated_text": result[2] if semantic else None,
                "elapsed_sec": time.time() - started,
            }
            (output_root / (identifier + ".json")).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            )
            return {
                "audio_base64": base64.b64encode(buffer.getvalue()).decode(),
                "metadata": metadata,
            }

    @app.post("/semantic_edit")
    def semantic_edit(req: EditRequest):
        return edit(req, True)

    @app.post("/acoustic_edit")
    def acoustic_edit(req: EditRequest):
        return edit(req, False)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    from dotebench.candidates.common.runtime import activate_runtime, runtime_identity

    activate_runtime("fireredtts3_instruct")
    from fireredtts3.llm.fireredtts3_instruct import FireRedTTS3Instruct
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    model = FireRedTTS3Instruct(args.model_path)
    model.redae.eval()
    model.tts_core.eval()
    app = create_app(
        model,
        runtime_identity("fireredtts3_instruct"),
        os.environ["DOTEBENCH_SERVICE_OUTPUTS"],
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
