"""speaker_similarity service and local model inference."""

import argparse
from pathlib import Path

from dotebench.services.utils.http import json_endpoint, serialized, service_app
from dotebench.services.utils.inference import (
    audio_array,
    configured,
)
from dotebench.services.utils.model_assets import prepared_identity

MODEL_ID = "wavlm_sv"
DEFAULTS = {}


def build(model_path: Path, device="cuda"):
    import torch

    from dotebench.services.backends.resources.speaker_models.ecapa_tdnn import (
        ECAPA_TDNN_SMALL,
    )

    checkpoint = (
        model_path / "wavlm_large_finetune.pth" if model_path.is_dir() else model_path
    )
    if checkpoint.name != "wavlm_large_finetune.pth":
        raise ValueError("Expected the WavLM ECAPA checkpoint filename")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected WavLM ECAPA checkpoint {checkpoint}")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model = model.to(device).eval()

    def embedding(encoded):
        wav, sr = audio_array(encoded)
        # Frozen service bounds oversized inputs at 59.9s after 60s.
        if len(wav) / sr > 60:
            wav = wav[: int(59.9 * sr)]
        import librosa

        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return model(torch.from_numpy(wav).unsqueeze(0).float().to(device))

    def measure(payload):
        with torch.inference_mode():
            a, b = (
                embedding(payload["source_audio"]),
                embedding(payload["target_audio"]),
            )
            return {"similarity": torch.nn.functional.cosine_similarity(a, b).item()}

    return configured(measure, {"device": device})


def create_app(measure, *, model_identity=None):
    app = service_app(
        title="speaker_similarity",
        health=lambda: {
            "status": "ready",
            "ability": "speaker_similarity",
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
