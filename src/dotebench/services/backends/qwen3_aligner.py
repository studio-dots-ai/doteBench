"""qwen3_aligner service and local model inference."""

import argparse
from copy import deepcopy
from pathlib import Path

from dotebench.services.utils.http import json_endpoint, serialized, service_app
from dotebench.services.utils.inference import (
    audio_array,
    configured,
    transformer_options,
)
from dotebench.services.utils.model_assets import prepared_identity

MODEL_ID = "qwen3_forced_aligner_0_6b"
DEFAULTS = {"model_options": {"dtype": "bfloat16"}}


def build(model_path: Path, device="cuda"):
    model_options = deepcopy(DEFAULTS).get("model_options", {})

    from qwen_asr import Qwen3ForcedAligner

    parameters = transformer_options(model_options, device)
    model = Qwen3ForcedAligner.from_pretrained(str(model_path), **parameters)

    def measure(payload):
        if "queries" in payload:
            raise ValueError("Query extraction requires existing alignment")
        wav, sr = audio_array(payload["audio"])
        from dotebench.text import token_spans

        text = payload["text"]
        units = [text[start:end] for start, end in token_spans(text)]
        return {"segments": align_units(model, (wav, sr), units)}

    return configured(measure, parameters)


def align_units(aligner, audio, units):
    """Run the original align_with_units path with explicitly owned units."""
    import torch
    from qwen_asr.inference.utils import normalize_audios

    encoded = (
        "<|audio_start|><|audio_pad|><|audio_end|>"
        + "<timestamp><timestamp>".join(units)
        + "<timestamp><timestamp>"
    )
    inputs = aligner.processor(
        text=[encoded],
        audio=normalize_audios([audio]),
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(aligner.model.device).to(aligner.model.dtype)
    with torch.inference_mode():
        logits = aligner.model.thinker(**inputs).logits
        output = logits.argmax(dim=-1)[0]
    selected = output[inputs["input_ids"][0] == aligner.timestamp_token_id]
    timestamp = (selected * aligner.timestamp_segment_time).to("cpu").numpy()
    items = aligner.aligner_processor.parse_timestamp(units, timestamp)
    from dotebench.alignment import bound_segments

    segments = [
        {
            "word": str(item["text"]),
            "start": round(float(item["start_time"]) / 1000, 3),
            "end": round(float(item["end_time"]) / 1000, 3),
        }
        for item in items
    ]
    return bound_segments(segments, len(audio[0]) / audio[1])


def create_app(measure, *, model_identity=None):
    infer = serialized(measure)

    def handle(payload):
        if "queries" in payload:
            from dotebench.alignment import query_alignment

            if "audio" in payload:
                raise ValueError("Query extraction accepts alignment, not audio")
            return {
                "answers": query_alignment(
                    payload["text"],
                    payload["alignment"],
                    payload["queries"],
                    payload["audio_duration"],
                )
            }
        return infer(payload)

    app = service_app(
        title="qwen3_aligner",
        health=lambda: {
            "status": "ready",
            "ability": "qwen3_aligner",
            "model_identity": model_identity,
            "configuration": getattr(measure, "configuration", {}),
        },
        close=getattr(measure, "close", None),
    )
    app.post("/measure")(json_endpoint(handle))
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
