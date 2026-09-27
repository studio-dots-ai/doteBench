"""qwen3_asr service and local model inference."""

import argparse
import base64
import io
from copy import deepcopy
from pathlib import Path

import soundfile as sf

from dotebench.services.utils.http import json_endpoint, serialized, service_app
from dotebench.services.utils.inference import (
    configured,
    language_name,
    transformer_options,
)
from dotebench.services.utils.model_assets import prepared_identity

MODEL_ID = "qwen3_asr_1_7b"
DEFAULTS = {
    "model_options": {
        "dtype": "bfloat16",
        "max_inference_batch_size": 1,
        "max_new_tokens": 512,
    }
}


def build(model_path: Path, device="cuda"):
    model_options = deepcopy(DEFAULTS).get("model_options", {})
    import torch
    from qwen_asr import Qwen3ASRModel

    parameters = transformer_options(model_options, device)
    model = Qwen3ASRModel.from_pretrained(str(model_path), **parameters)

    def measure(payload):
        wav, sr = sf.read(
            io.BytesIO(base64.b64decode(payload["audio"])), dtype="float32"
        )
        if wav.size and not wav.any():
            return {"text": ""}
        with torch.inference_mode():
            results = model.transcribe(
                audio=(wav, sr), language=language_name(payload["language"])
            )
        return {"text": results[0].text}

    return configured(measure, parameters)


def create_app(measure, *, model_identity=None):
    app = service_app(
        title="qwen3_asr",
        health=lambda: {
            "status": "ready",
            "ability": "qwen3_asr",
            "model_identity": model_identity,
            "configuration": getattr(measure, "configuration", {}),
        },
        close=getattr(measure, "close", None),
    )
    app.post("/measure")(json_endpoint(serialized(measure)))
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    identity = prepared_identity(MODEL_ID, args.model_path)
    measure = build(args.model_path, args.device)
    import uvicorn

    uvicorn.run(
        create_app(measure, model_identity=identity), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
