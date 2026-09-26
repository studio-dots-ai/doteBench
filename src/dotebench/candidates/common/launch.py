"""Launch a configured candidate service from its verified upstream runtime."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotebench.candidates.common.runtime import (
    candidate_directory,
    check_source_registration,
    prepare_runtime,
)
from dotebench.candidates.registry import native_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=native_candidates(),
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("service_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    name = args.candidate.replace("-", "_")
    manifest = json.loads((candidate_directory(name) / "runtime.json").read_text())
    check_source_registration(name, args.source_root)
    runtime = prepare_runtime(
        name,
        upstream=args.source_root / "third_party" / manifest["upstream"]["directory"],
        destination=args.runtime_root,
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["DOTEBENCH_NATIVE_RUNTIME"] = str(runtime)
    os.environ["DOTEBENCH_SERVICE_OUTPUTS"] = str(output)
    for key, directory in [
        ("TMPDIR", "temporary"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("XDG_CACHE_HOME", "cache"),
    ]:
        os.environ.setdefault(key, str(output / directory))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    arguments = args.service_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    os.execv(
        sys.executable,
        [sys.executable, "-m", f"dotebench.candidates.{name}.backend", *arguments],
    )


if __name__ == "__main__":
    main()
