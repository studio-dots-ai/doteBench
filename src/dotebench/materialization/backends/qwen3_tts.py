"""Qwen3-TTS-12Hz-1.7B-CustomVoice materialization service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotebench.materialization.assets import prepared_identity

from .common import (
    SynthesisRequest,
    create_app,
    implementation_identity,
    set_seed,
    source_identity,
    wav_bytes,
)

SOURCE_COMMIT = "1ab0dd75353392f28a0d05d9ca960c9954b13c83"
SOURCE_URL = "https://github.com/QwenLM/Qwen3-TTS.git"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_KEY = "qwen3_tts_custom_voice"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18101)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="flash_attention_2")
    args = parser.parse_args()
    source = source_identity(
        args.upstream, source=SOURCE_URL, commit=SOURCE_COMMIT
    )
    model_identity = prepared_identity(MODEL_KEY, args.model_path)
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = getattr(torch, args.dtype)
    model = Qwen3TTSModel.from_pretrained(
        str(args.model_path),
        device_map=args.device,
        dtype=dtype,
        attn_implementation=args.attention,
        local_files_only=True,
    )

    def synthesize(request: SynthesisRequest) -> bytes:
        if request.model_id != MODEL_ID or request.reference_audio is not None:
            raise ValueError("Qwen3-TTS materialization request identity mismatch")
        parameters = request.parameters
        allowed = {"speaker", "control"}
        if set(parameters) - allowed or "speaker" not in parameters:
            raise ValueError("Invalid Qwen3-TTS materialization parameters")
        set_seed(request.seed)
        language = "Chinese" if request.language == "zh" else "English"
        wavs, rate = model.generate_custom_voice(
            text=request.text,
            language=language,
            speaker=parameters["speaker"],
            instruct=parameters.get("control") or "",
        )
        return wav_bytes(wavs[0], rate)

    identity = {
        "adapter": implementation_identity(Path(__file__), __name__),
        "source": source,
        "models": {MODEL_KEY: model_identity},
        "inference": {"dtype": args.dtype, "attention": args.attention},
    }
    import uvicorn

    uvicorn.run(
        create_app(provider="qwen3_tts", identity=identity, synthesize=synthesize),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
