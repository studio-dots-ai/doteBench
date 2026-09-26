"""AuK Base whole-utterance and local generation."""

from dotebench import RELEASE

import argparse
import base64
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from dotebench.candidates.common.runtime import activate_runtime, runtime_identity

app = FastAPI(version=RELEASE, title="AuK Base")
engine = None
identity = None
weights = None
lock = Lock()


class LocalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    audio: str
    instruction: str = Field(min_length=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    target_frames: int = Field(gt=0)
    context_mode: Literal["clean", "bridge"] = "clean"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    nfe: int = Field(default=32, ge=1)
    cfg_strength: float = Field(default=2.0, ge=0)
    metadata: dict


@torch.inference_mode()
def generate_local(model, source, sample_rate, req):
    if source.ndim != 2 or source.shape[0] != 1 or not torch.isfinite(source).all():
        raise ValueError("Expected finite mono source")
    sr, hop = model.target_sample_rate, model.downsample_rate
    if (sr, hop) != (24000, 480) or model.is_flash:
        raise ValueError("Expected AuK Base with 24 kHz / 20 ms latent frames")
    a, b, count = req.start_frame, req.end_frame, req.target_frames
    if not 0 <= a < b <= ((source.shape[-1] * 50 + sample_rate - 1) // sample_rate):
        raise ValueError("Invalid source frame interval")
    mono = source[0]
    audio16 = torchaudio.functional.resample(mono, sample_rate, 16000).numpy()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": req.instruction},
                {"type": "audio", "audio": audio16},
            ],
        }
    ]
    audio24 = torchaudio.functional.resample(source, sample_rate, sr)
    torch.manual_seed(req.seed)
    padded = F.pad(audio24, (0, (-audio24.shape[-1]) % hop)).to(model.device)
    latents, lengths = model.vae_model.encoding_and_normalization(padded.unsqueeze(0))
    latents = latents[:, : int(lengths[0])]
    if b > latents.shape[1]:
        raise ValueError("Source window exceeds encoded latent length")
    known = torch.cat(
        (latents[:, :a], latents.new_zeros(1, count, model.latent_dim), latents[:, b:]),
        dim=1,
    )
    mask = torch.zeros(known.shape[:2], dtype=torch.bool, device=model.device)
    mask[:, a : a + count] = True
    total = latents.shape[1] + known.shape[1]
    if total > 65536:
        raise ValueError("Reference and target exceed AuK sampler capacity")
    with torch.autocast(
        "cuda", dtype=model.dtype, enabled=str(model.device).startswith("cuda")
    ):
        cond = model.model.build_cond_inputs([messages], model.model.text_processor)
        whole = req.metadata.get("generation_scope") == "whole"
        if whole and (a != 0 or b != latents.shape[1] or not mask.all()):
            raise ValueError("Whole-utterance requests must cover every target frame")
        context = (
            {}
            if whole
            else dict(known_target=known, edit_mask=mask, context_mode=req.context_mode)
        )
        generated, _ = model.model.sample(
            cond=latents,
            text=cond,
            duration=total,
            seed=req.seed,
            **context,
            steps=req.nfe,
            cfg_strength=req.cfg_strength,
            sway_sampling_coef=-1.0,
        )
    target = generated[:, latents.shape[1] :]
    if not torch.equal(target[~mask], known[~mask]):
        raise RuntimeError("Sampler changed fixed context latents")
    decoded = (
        model.vae_model.inference_from_latents(
            model.vae_model.denormalize(target).transpose(1, 2)
        )[0]
        .cpu()
        .float()
    )
    segment = decoded[:, a * hop : (a + count) * hop]
    if segment.shape[-1] != count * hop or not torch.isfinite(segment).all():
        raise RuntimeError("Invalid generated segment")
    segment = torchaudio.functional.resample(segment, sr, sample_rate)
    n = round(count * sample_rate / 50)
    # resample returns ceil(length * ratio); select the nearest exact sample count.
    segment = segment[:, :n]
    if segment.shape[-1] != n:
        raise RuntimeError("Resampled segment length mismatch")
    start = min(source.shape[-1], round(a * sample_rate / 50))
    end = min(source.shape[-1], round(b * sample_rate / 50))
    output = torch.cat((source[:, :start], segment, source[:, end:]), dim=-1)
    metadata = dict(
        req.metadata,
        context_mode=req.context_mode,
        instruction=req.instruction,
        start_sample=start,
        end_sample=end,
        generated_samples=n,
        suffix_shift_samples=n - (end - start),
        sample_rate=sample_rate,
        output_samples=output.shape[-1],
        context_latents_equal=True,
        sampler_calls=1,
        total_latent_frames=total,
        seed=req.seed,
        nfe=req.nfe,
        cfg_strength=req.cfg_strength,
    )
    return output, metadata


@app.get("/health")
def health():
    return dict(
        status="healthy" if engine is not None else "loading",
        model_loaded=engine is not None,
        model="AuK Base",
        runtime=identity,
        weights=weights,
    )


@app.post("/edit/local")
def edit_local(req: LocalRequest):
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    try:
        raw = Path(req.audio).read_bytes()
        if hashlib.sha256(raw).hexdigest() != req.metadata["source_audio_sha256"]:
            raise ValueError("Source bytes changed after compilation")
        source, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        if not len(source) or not np.isfinite(source).all():
            raise ValueError("Invalid source audio")
        with lock:
            output, metadata = generate_local(
                engine, torch.from_numpy(source.T.copy()), sr, req
            )
        buffer = io.BytesIO()
        sf.write(buffer, output.T.numpy(), sr, format="WAV", subtype="FLOAT")
        metadata["output_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
        logdir = Path(os.environ["DOTEBENCH_SERVICE_OUTPUTS"]) / "requests"
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir / (uuid.uuid4().hex + ".json")).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        return dict(
            audio_base64=base64.b64encode(buffer.getvalue()).decode(), metadata=metadata
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            422, detail=dict(error_code="candidate_error", message=str(exc))
        ) from exc


def main():
    global engine, identity, weights
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--qwen-path", required=True, type=Path)
    parser.add_argument("--weights-manifest", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=18000, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    runtime = activate_runtime("auk_base")
    import sys

    sys.path.insert(0, str(runtime / "src"))
    from auk.infer.infer_auk import AukInfer

    weights = json.loads(args.weights_manifest.read_text())
    for group, root in [("auk", args.checkpoint_dir), ("qwen", args.qwen_path)]:
        actual_files = {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and ".cache" not in p.parts
        }
        if actual_files != set(weights[group]):
            raise ValueError(f"Weight directory file set mismatch: {group}")
        for relative, expected in weights[group].items():
            digest = hashlib.sha256()
            with (root / relative).open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Weight checksum mismatch: {group}/{relative}")
    identity = runtime_identity("auk_base")
    engine = AukInfer(
        str(args.checkpoint_dir / "config.yaml"),
        str(args.checkpoint_dir / "auk_base.safetensors"),
        qwen_path=str(args.qwen_path),
        device=args.device,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
