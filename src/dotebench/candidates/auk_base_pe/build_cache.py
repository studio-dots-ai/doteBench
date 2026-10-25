"""Build the full one-take/sequential instruction cache without audio inference."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotebench.compilers.sequential.planner import plan_operations
from dotebench.dataset import data_version, load_cases, select_shards
from dotebench.instructions import parse_instruction

from .instruction_cache import (
    MAX_ATTEMPTS,
    OfficialPE,
    RequestPacer,
    digest,
    write_json,
)


def plan_requests():
    requests, references = {}, []
    for entry in select_shards():
        for case in load_cases(entry, verify_audio=False):
            steps = [("one_take", 0, None, case.instruction_xml)]
            steps.extend(
                ("sequential", index, step.operation_index, step.instruction_xml)
                for index, step in enumerate(plan_operations(case.instruction_xml))
            )
            for mode, index, operation, xml in steps:
                item = {
                    "instruction_xml": xml,
                    "source_text": parse_instruction(xml).source_text,
                    "language": case.language,
                }
                input_key = digest(item)
                requests[input_key] = item
                references.append(
                    {
                        "case_id": case.id,
                        "compiler": mode,
                        "step_index": index,
                        "operation_index": operation,
                        "input_key": input_key,
                    }
                )
    return requests, references


def identity(auk_source, base_url, model):
    from . import instruction_cache

    paths = [
        Path(instruction_cache.__file__),
        Path(__file__),
        auk_source / "src/auk/infer/pe.py",
        auk_source / "src/auk/infer/pe.config.yaml",
    ]
    return {
        "schema": 1,
        "dataset_version": data_version(),
        "scope": "instruction_only",
        "transcript_provider": "dotebench-provided-transcript-v1",
        "model": model,
        "base_url": base_url.rstrip("/"),
        "max_attempts": MAX_ATTEMPTS,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auk-source", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL_NAME"))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--request-interval", type=float, default=0.75)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_url or not args.model:
        parser.error("Supply --base-url and --model, or LLM_BASE_URL/LLM_MODEL_NAME")
    if args.request_interval <= 0:
        parser.error("--request-interval must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return build(args)


def build(args):
    from .instruction_cache import prepare_record

    inputs, references = plan_requests()
    config = identity(args.auk_source, args.base_url, args.model)
    records = {
        digest({"identity": config, "input": item}): item for item in inputs.values()
    }
    for ref in references:
        item = inputs[ref.pop("input_key")]
        ref["key"] = digest({"identity": config, "input": item})
    manifest = {"identity": config, "references": references}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Cache identity changed; use a different output directory")
    write_json(manifest_path, manifest)
    counts = Counter(ref["compiler"] for ref in references)
    print(
        json.dumps({"unique_requests": len(records), "references": counts}), flush=True
    )
    if args.plan_only:
        return 0
    api_key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ.get("LLM_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError("Supply LLM_API_KEY or --api-key-file")
    sys.path.insert(0, str(args.auk_source / "src"))
    directory = args.output / "records"
    directory.mkdir(exist_ok=True)
    local = threading.local()
    backends = []
    pacer = RequestPacer(args.request_interval)

    def process(key, item):
        path = directory / f"{key}.json"
        old = json.loads(path.read_text()) if path.exists() else None
        if old is not None:
            if old["key"] != key or old["input"] != item or old["identity"] != config:
                raise ValueError("Cached record disagrees with request identity")
            if old["status"] in {"ok", "unsupported", "error"}:
                return old, True
        if not hasattr(local, "backend"):
            local.backend = OfficialPE(
                api_key=api_key, base_url=args.base_url, model=args.model, pacer=pacer
            )
            backends.append(local.backend)
        result = prepare_record(
            item, config, local.backend, old, save=lambda value: write_json(path, value)
        )
        return result, False

    started = time.monotonic()
    outcomes, attempts, cached, fresh = Counter(), 0, 0, 0

    def progress():
        elapsed = time.monotonic() - started
        remaining = len(records) - sum(outcomes.values())
        return {
            "total": len(records),
            "completed": sum(outcomes.values()),
            "cached": cached,
            "fresh": fresh,
            "remaining": remaining,
            "outcomes": dict(outcomes),
            "attempts_in_completed_records": attempts,
            "elapsed_seconds": round(elapsed, 1),
            "fresh_per_minute": round(fresh * 60 / elapsed, 1),
            "eta_seconds": round(remaining * elapsed / fresh) if fresh else None,
            "workers": args.workers,
            "request_interval_seconds": args.request_interval,
        }

    write_json(args.output / "progress.json", progress())
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, key, item) for key, item in records.items()]
            for future in as_completed(futures):
                record, reused = future.result()
                outcomes[record["status"]] += 1
                attempts += len(record["attempts"])
                cached += int(reused)
                fresh += int(not reused)
                status = progress()
                write_json(args.output / "progress.json", status)
                if status["completed"] % 50 == 0 or not status["remaining"]:
                    print(json.dumps(status), flush=True)
    finally:
        for backend in backends:
            backend.close()
    write_json(args.output / "summary.json", {**progress(), "status": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
