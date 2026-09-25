"""emotion service and local model inference."""

import argparse
import base64
import io
from copy import deepcopy
from pathlib import Path

import soundfile as sf

from dotebench.services.utils.http import json_endpoint, serialized, service_app
from dotebench.services.utils.inference import (
    configured,
)
from dotebench.services.utils.model_assets import prepared_identity

MODEL_ID = "qwen3_omni_30b_a3b_instruct"
DEFAULTS = {
    "model_options": {
        "tensor_parallel_size": None,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 16384,
        "trust_remote_code": True,
    },
    "sampling_options": {"temperature": 0, "max_tokens": 4096},
}

import hashlib
import random
import re
from collections.abc import Sequence

_EMOTION_LABELS = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
_INPUT_PADDED_CROP = "padded_crop_250ms"
_INPUT_WHOLE_AUDIO = "whole_audio"
_LABEL_ORDER_SEED_NAMESPACE = "gemini-emotion-audio-v1"


def _emotion_candidate_order(*, case_id: str, span_index: int) -> tuple[str, ...]:
    digest = hashlib.sha256(
        f"{_LABEL_ORDER_SEED_NAMESPACE}\x00{case_id}\x00{span_index}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    labels = list(_EMOTION_LABELS)
    rng.shuffle(labels)
    return tuple(labels)


def _parse_emotion_label(response: str) -> str | None:
    normalized = str(response).strip().casefold()
    canonical_pattern = re.compile(
        "(?<![a-z0-9_])(?:"
        + "|".join(map(re.escape, (*_EMOTION_LABELS, "neutral")))
        + ")(?![a-z0-9_])"
    )
    matches = {
        "calm" if label == "neutral" else label
        for label in canonical_pattern.findall(normalized)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _build_emotion_prompt(*, candidates: Sequence[str], input_strategy: str) -> str:
    if set(candidates) != set(_EMOTION_LABELS) or len(candidates) != len(
        _EMOTION_LABELS
    ):
        raise ValueError("Candidates must contain every canonical emotion exactly once")
    candidate_text = ", ".join(candidates)
    if input_strategy == _INPUT_PADDED_CROP:
        focus = "The audio is a short excerpt padded around one target interval. Judge the vocal emotion of the central target speech."
    elif input_strategy == _INPUT_WHOLE_AUDIO:
        focus = "Judge the dominant vocal emotion of the entire utterance."
    else:
        raise ValueError(f"Unsupported emotion input strategy: {input_strategy}")
    prompt = f"Classify perceived vocal emotion from the attached audio. Use acoustic delivery only. Do not infer emotion from words you may hear. {focus}\nChoose exactly one label from: {candidate_text}\nReturn only that lowercase label, with no explanation or punctuation."
    return prompt + "\nThe emotion is (one single word):"


def build(model_path, device="cuda"):
    defaults = deepcopy(DEFAULTS)
    model_options = defaults["model_options"]
    sampling_options = defaults["sampling_options"]
    import torch
    from vllm import LLM, SamplingParams

    parameters = model_options
    if parameters["tensor_parallel_size"] is None:
        parameters["tensor_parallel_size"] = torch.cuda.device_count()
    model = LLM(model=str(model_path), **parameters)
    sampling_parameters = sampling_options
    sampling = SamplingParams(**sampling_parameters)

    def measure(payload):
        # Crop at the original sampling rate and encode PCM16 as the frozen
        # evaluator did; the Omni processor owns subsequent resampling.
        wav, sr = sf.read(
            io.BytesIO(base64.b64decode(payload["audio"])), dtype="float32"
        )
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        start, end = float(payload["start"]), float(payload["end"])
        if end <= start or start >= len(wav) / sr:
            raise ValueError("Empty aligned emotion interval")
        if payload["input_strategy"] != "whole_audio":
            wav = wav[
                int(max(0, start - 0.25) * sr) : int(
                    min(len(wav) / sr, end + 0.25) * sr
                )
            ]
            output = io.BytesIO()
            sf.write(output, wav, sr, format="WAV", subtype="PCM_16")
            audio = base64.b64encode(output.getvalue()).decode("ascii")
        else:
            audio = payload["audio"]
        candidates = _emotion_candidate_order(
            case_id=payload["case_id"], span_index=payload["span_index"]
        )
        prompt = _build_emotion_prompt(
            candidates=candidates,
            input_strategy=payload["input_strategy"],
        )
        responses = []
        for attempt in range(2):
            current = prompt + (
                "\nYour previous response did not match the required format. Return exactly one allowed lowercase label."
                if attempt
                else ""
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "data:audio/wav;base64," + audio},
                        },
                        {"type": "text", "text": current},
                    ],
                }
            ]
            generated = model.chat(
                messages,
                sampling_params=sampling,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=False,
            )
            response = generated[0].outputs[0].text
            responses.append(response)
            label = _parse_emotion_label(response)
            if label is not None:
                return {
                    "predicted_emotion": label,
                    "raw_responses": responses,
                    "attempts": attempt + 1,
                }
        raise ValueError(
            "Emotion classifier returned no unique canonical label after format retry"
        )

    return configured(measure, parameters, sampling_parameters)


def create_app(measure, *, model_identity=None):
    app = service_app(
        title="emotion",
        health=lambda: {
            "status": "ready",
            "ability": "emotion",
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
