"""VoxCPM2 controllable-cloning materialization service."""

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

SOURCE_COMMIT = "5510503182e405fb78b6765d24d2e3db5d987a5c"
SOURCE_URL = "https://github.com/openbmb/VoxCPM.git"
MODEL_ID = "openbmb/VoxCPM2"
MODEL_KEY = "voxcpm2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18103)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()
    source = source_identity(
        args.upstream, source=SOURCE_URL, commit=SOURCE_COMMIT
    )
    model_identity = prepared_identity(MODEL_KEY, args.model_path)
    sys.path.insert(0, str((args.upstream / "src").resolve()))
    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained(
        hf_model_id=str(args.model_path),
        load_denoiser=False,
        local_files_only=True,
        optimize=args.optimize,
        device=args.device,
    )

    def synthesize(request: SynthesisRequest) -> bytes:
        if request.model_id != MODEL_ID or request.reference_audio is None:
            raise ValueError("VoxCPM2 materialization request identity mismatch")
        allowed = {"control", "cfg_value", "inference_timesteps"}
        if set(request.parameters) - allowed:
            raise ValueError("Invalid VoxCPM2 materialization parameters")
        reference = decoded_audio(request.reference_audio)
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(reference)
            handle.flush()
            text = request.text
            control = request.parameters.get("control")
            if control:
                text = f"({control}){text}"
            set_seed(request.seed)
            samples = model.generate(
                text=text,
                reference_wav_path=handle.name,
                cfg_value=float(request.parameters.get("cfg_value", 2.0)),
                inference_timesteps=int(
                    request.parameters.get("inference_timesteps", 10)
                ),
                normalize=False,
                denoise=False,
            )
        rate = int(getattr(model.tts_model, "sample_rate", 48000))
        return wav_bytes(samples, rate)

    identity = {
        "adapter": implementation_identity(Path(__file__), __name__),
        "source": source,
        "models": {MODEL_KEY: model_identity},
        "inference": {
            "optimize": args.optimize,
            "normalize": False,
            "denoise": False,
        },
    }
    import uvicorn

    uvicorn.run(
        create_app(provider="voxcpm2", identity=identity, synthesize=synthesize),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
