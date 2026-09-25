"""Audio and model runtime helpers shared by concrete backends."""

import base64
import io
import json


def audio_array(value, *, sr=None):
    import librosa

    return librosa.load(
        io.BytesIO(base64.b64decode(value, validate=True)), sr=sr, mono=True
    )


def language_name(value):
    return {"en": "English", "zh": "Chinese", "eng": "English", "zho": "Chinese"}.get(
        value, value
    )


def transformer_options(options, device):
    import torch

    parameters = {"device_map": device, **options}
    dtype = parameters["dtype"]
    if isinstance(dtype, str) and dtype != "auto":
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported transformer dtype")
        parameters["dtype"] = getattr(torch, dtype)
    return parameters


def configured(measure, model=None, sampling=None):
    # Torch dtype/device values are represented as strings in execution records.
    measure.configuration = json.loads(
        json.dumps(
            {"model_options": model or {}, "sampling_options": sampling or {}},
            default=str,
        )
    )
    return measure
