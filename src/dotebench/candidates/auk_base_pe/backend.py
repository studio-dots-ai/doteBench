"""Whole-utterance AuK generation from cached official PE instructions."""

import base64
import hashlib
import io
import json
import os
import uuid
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from dotebench import RELEASE
from dotebench.candidates.auk_base import backend as base

app = FastAPI(version=RELEASE, title="AuK Base + PE")
app.get("/health")(base.health)


class PERequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str
    prepared: dict
    source_text: str
    language: str
    metadata: dict


def prepare_cached(enhancer, req):
    from auk.infer import pe

    cached = req.prepared
    if cached["task_type"] == "instruct_tts":
        raise pe.UnsupportedRequestError("instruct_tts requires PE instruction retry")
    classified = pe._Classified(
        supported=True,
        task_type=cached["task_type"],
        operation_subtype=cached["operation_subtype"],
        params_extracted=cached["params_extracted"],
        needs_text=False,
        reasoning="Cached official PE instruction",
        language=cached["language"],
        text_language=cached["text_language"],
    )
    return enhancer._prepare_generation(
        classified,
        dict(cached["prepared_params"]),
        audio_path=req.audio,
        asr=pe.ASRCall(
            model="dotebench-provided-transcript-v1",
            text=req.source_text,
            language=req.language,
            raw_response={"origin": "provided_transcript"},
        ),
        instruction_language=cached["language"],
        llm_calls=[],
        model_instruction=cached["instruction"],
    )


@torch.inference_mode()
def generate(model, req, enhancer, directory):
    from auk.infer import infer_gradio

    raw = Path(req.audio).read_bytes()
    if hashlib.sha256(raw).hexdigest() != req.metadata["source_audio_sha256"]:
        raise ValueError("Source bytes changed after compilation")
    # Use the verified, already loaded model with the official generation wrapper.
    infer_gradio.ENGINES[infer_gradio.BASE_LABEL] = model
    with prepare_cached(enhancer, req) as prepared:
        metadata = dict(
            req.metadata, pe=prepared.debug_dict(include_raw_responses=False)
        )
        if prepared.audio:
            model_input = Path(prepared.audio).read_bytes()
            input_path = directory / "model-input.wav"
            input_path.write_bytes(model_input)
            metadata["model_input"] = {
                "path": str(input_path),
                "sha256": hashlib.sha256(model_input).hexdigest(),
            }
        else:
            metadata["model_input"] = None
        sr, waveform = infer_gradio.run_generate(
            **prepared.to_run_generate_kwargs(
                variant=infer_gradio.BASE_LABEL, nfe=32, cfg=2.0, seed=44
            ),
            task_type=prepared.task_type,
        )
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sr, format="WAV", subtype="PCM_16")
        output = buffer.getvalue()
        metadata.update(
            seed=44,
            nfe=32,
            cfg_strength=2.0,
            sampler_calls=1,
            output_sha256=hashlib.sha256(output).hexdigest(),
        )
        return output, metadata


@app.post("/edit/pe")
def edit_pe(req: PERequest):
    if base.engine is None:
        raise HTTPException(503, "Model not loaded")
    import gradio as gr
    from auk.infer import pe

    # Audio-only PE stages need neither ASR nor LLM clients.
    enhancer = pe.PromptEnhancer.__new__(pe.PromptEnhancer)
    directory = (
        Path(os.environ["DOTEBENCH_SERVICE_OUTPUTS"]) / "requests" / uuid.uuid4().hex
    )
    directory.mkdir(parents=True)
    try:
        with base.lock:
            output, metadata = generate(base.engine, req, enhancer, directory)
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        return {"audio_base64": base64.b64encode(output).decode(), "metadata": metadata}
    except torch.cuda.OutOfMemoryError:
        raise
    except (ValueError, RuntimeError, gr.Error) as exc:
        raise HTTPException(
            422, detail={"error_code": "candidate_error", "message": str(exc)}
        ) from exc


def main():
    base.main("auk_base_pe", application=app)


if __name__ == "__main__":
    main()
