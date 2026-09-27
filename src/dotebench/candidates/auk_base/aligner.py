"""Independent source alignment inference for the AuK Base candidate system."""

import argparse
import hashlib
import json
import threading
from pathlib import Path

from dotebench.services.utils.http import service_app


def create_app(measure, identity):
    from fastapi import HTTPException

    app = service_app(
        title="AuK Base candidate aligner",
        health=lambda: {
            "status": "ready",
            "role": "candidate-aligner",
            "identity": identity,
        },
        close=getattr(measure, "close", None),
    )
    lock = threading.Lock()

    @app.post("/align")
    def align(payload: dict):
        if (
            set(payload) != {"audio", "text", "language", "identity"}
            or payload["identity"] != identity
        ):
            raise HTTPException(400, "Unexpected alignment input or identity")
        with lock:
            result = measure({k: payload[k] for k in ("audio", "text", "language")})
        return {**result, "identity": identity}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from dotebench.compilation import source_path
    from dotebench.services.backends.qwen3_aligner import build

    files = {
        str(p.relative_to(args.model_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.model_path.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}
    }
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Missing aligner model weights")
    modules = [
        "dotebench.candidates.auk_base.aligner",
        "dotebench.services.utils.inference",
        "dotebench.services.utils.http",
        "dotebench.services.backends.qwen3_aligner",
        "dotebench.alignment",
        "dotebench.text",
    ]
    implementation = {
        name: hashlib.sha256(source_path(name).read_bytes()).hexdigest()
        for name in modules
    }
    import importlib.metadata

    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "transformers", "qwen-asr"]
    }
    identity = {
        "model": "Qwen3-ForcedAligner-0.6B",
        "files": files,
        "weights_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            json.dumps(implementation, sort_keys=True).encode()
        ).hexdigest(),
        "implementation": implementation,
        "packages": versions,
        "device": args.device,
        "options": {},
    }
    measure = build(args.model_path, args.device)
    import uvicorn

    uvicorn.run(create_app(measure, identity), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
