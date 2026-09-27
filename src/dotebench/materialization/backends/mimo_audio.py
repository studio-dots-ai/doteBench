"""MiMo-Audio-7B-Instruct cloning materialization service."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from dotebench.materialization.assets import prepared_identity

from .common import (
    SynthesisRequest,
    create_app,
    decoded_audio,
    implementation_identity,
    set_seed,
    source_identity,
    wav_bytes,
)

SOURCE_COMMIT = "62d956b4a1a45419bee5e41f477078c3684dbbcc"
SOURCE_URL = "https://github.com/XiaomiMiMo/MiMo-Audio.git"
MODEL_ID = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
MODEL_KEYS = ("mimo_audio_instruct", "mimo_audio_tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18104)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source = source_identity(
        args.upstream, source=SOURCE_URL, commit=SOURCE_COMMIT
    )
    model_identities = {
        MODEL_KEYS[0]: prepared_identity(MODEL_KEYS[0], args.model_path),
        MODEL_KEYS[1]: prepared_identity(MODEL_KEYS[1], args.tokenizer_path),
    }
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    from src.mimo_audio.mimo_audio import MimoAudio
    from src.mimo_audio.modeling_mimo_audio import MiMoStopper

    model = MimoAudio(str(args.model_path), str(args.tokenizer_path), args.device)

    def synthesize(request: SynthesisRequest) -> bytes:
        if request.model_id != MODEL_ID or request.reference_audio is None:
            raise ValueError("MiMo-Audio materialization request identity mismatch")
        allowed = {"control", "max_new_tokens"}
        if set(request.parameters) - allowed:
            raise ValueError("Invalid MiMo-Audio materialization parameters")
        reference = decoded_audio(request.reference_audio)
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(reference)
            handle.flush()
            set_seed(request.seed)
            inputs = model.get_tts_sft_prompt(
                request.text,
                instruct=request.parameters.get("control"),
                read_text_only=True,
                prompt_speech=handle.name,
            )
            stopping = [
                MiMoStopper(
                    stop_tokens=[
                        model.tokenizer.eos_token_id,
                        model.eostm_idx,
                        model.im_end_idx,
                    ],
                    group_size=model.group_size,
                    audio_channels=model.audio_channels,
                )
            ]
            with torch.no_grad():
                samples = model.forward(
                    inputs,
                    return_audio=True,
                    output_audio_path=None,
                    stopping_criteria=stopping,
                    max_new_tokens=int(
                        request.parameters.get("max_new_tokens", 8192)
                    ),
                    task_name="tts",
                )
        if isinstance(samples, str):
            raise TypeError("MiMo-Audio produced no audio tokens")
        return wav_bytes(samples.reshape(-1).detach().cpu().numpy(), 24000)

    identity = {
        "adapter": implementation_identity(Path(__file__), __name__),
        "source": source,
        "models": model_identities,
        "inference": {
            "read_text_only": True,
            "task_name": "tts",
            "sample_rate": 24000,
        },
    }
    import uvicorn

    uvicorn.run(
        create_app(provider="mimo_audio", identity=identity, synthesize=synthesize),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
