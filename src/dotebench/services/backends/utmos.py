"""utmos service and local model inference."""

import argparse
from pathlib import Path

from dotebench.services.utils.http import json_endpoint, serialized, service_app
from dotebench.services.utils.inference import (
    audio_array,
    configured,
)
from dotebench.services.utils.model_assets import prepared_identity

MODEL_ID = "utmos"
DEFAULTS = {}


def build(model_path: Path, device="cuda"):
    import torch

    if not (model_path / "hubconf.py").is_file():
        raise FileNotFoundError(
            "UTMOS model path must contain local SpeechMOS v1.2.0 hubconf.py and weights"
        )
    checkpoint = (
        model_path / "utmos22_strong_step7459_v1.pt"
        if model_path.is_dir()
        else model_path
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected local UTMOS checkpoint {checkpoint}")
    model = torch.hub.load(
        str(model_path), "utmos22_strong", source="local", pretrained=False
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True
    )
    model = model.to(device).eval()

    def measure(payload):
        wav, sr = audio_array(payload["audio"])
        with torch.inference_mode():
            value = model(torch.from_numpy(wav).unsqueeze(0).to(device), sr).item()
        return {"score": value}

    return configured(measure, {"device": device})


def create_app(measure, *, model_identity=None):
    app = service_app(
        title="utmos",
        health=lambda: {
            "status": "ready",
            "ability": "utmos",
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
