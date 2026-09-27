"""IndexTTS-2 emotion-controlled cloning materialization service."""

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

SOURCE_COMMIT = "1eafa935e8887f9ba877f82cf7319fceddfc86cb"
SOURCE_URL = "https://github.com/index-tts/index-tts.git"
MODEL_ID = "IndexTeam/IndexTTS-2"
MODEL_KEY = "indextts2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18102)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    source = source_identity(
        args.upstream, source=SOURCE_URL, commit=SOURCE_COMMIT
    )
    model_identity = prepared_identity(MODEL_KEY, args.model_path)
    sys.path.insert(0, str(args.upstream.resolve()))
    from indextts.infer_v2 import IndexTTS2

    config = args.config_path or args.model_path / "config.yaml"
    model = IndexTTS2(
        cfg_path=str(config),
        model_dir=str(args.model_path),
        use_fp16=args.fp16,
        device=args.device,
    )

    def synthesize(request: SynthesisRequest) -> bytes:
        if request.model_id != MODEL_ID or request.reference_audio is None:
            raise ValueError("IndexTTS-2 materialization request identity mismatch")
        allowed = {"emotion_vector", "emotion_alpha"}
        if set(request.parameters) - allowed or "emotion_vector" not in request.parameters:
            raise ValueError("Invalid IndexTTS-2 materialization parameters")
        vector = request.parameters["emotion_vector"]
        if not isinstance(vector, list) or len(vector) != 8:
            raise ValueError("IndexTTS-2 requires an eight-value emotion vector")
        reference = decoded_audio(request.reference_audio)
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(reference)
            handle.flush()
            set_seed(request.seed)
            result = model.infer(
                spk_audio_prompt=handle.name,
                text=request.text,
                output_path=None,
                emo_vector=vector,
                emo_alpha=float(request.parameters.get("emotion_alpha", 1.0)),
                interval_silence=200,
                verbose=False,
                top_p=0.8,
                top_k=30,
                temperature=0.8,
                repetition_penalty=10.0,
                max_mel_tokens=1500,
                max_text_tokens_per_segment=120,
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("IndexTTS-2 returned an unexpected result")
        rate, samples = result
        return wav_bytes(samples, int(rate))

    identity = {
        "adapter": implementation_identity(Path(__file__), __name__),
        "source": source,
        "models": {MODEL_KEY: model_identity},
        "inference": {
            "fp16": args.fp16,
            "interval_silence": 200,
            "top_p": 0.8,
            "top_k": 30,
            "temperature": 0.8,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 1500,
            "max_text_tokens_per_segment": 120,
        },
    }
    import uvicorn

    uvicorn.run(
        create_app(provider="indextts2", identity=identity, synthesize=synthesize),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
