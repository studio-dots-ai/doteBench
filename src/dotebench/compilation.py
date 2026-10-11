"""Registered versions for candidate compilers."""

import argparse
import json
from pathlib import Path

from dotebench.candidates.registry import CANDIDATES


def source_path(module):
    """Locate packaged source for runtime and alignment provenance."""
    base = Path(__file__).resolve().parent
    relative = (
        ""
        if module == "dotebench"
        else module.removeprefix("dotebench.").replace(".", "/")
    )
    path = base / (relative + ".py")
    if not path.is_file():
        path = base / relative / "__init__.py"
    if not path.is_file():
        raise ValueError(f"Source module missing: {module}")
    return path


def compiler_keys(name):
    return CANDIDATES[name].compilers


def manifest_path(name, compiler):
    if name not in CANDIDATES or compiler not in compiler_keys(name):
        raise ValueError(f"Unknown compiler: {name}/{compiler}")
    return (
        Path(__file__).parent / "candidates" / name / "compilers" / f"{compiler}.json"
    )


def compiler_identity(name, compiler="one_take"):
    manifest = json.loads(manifest_path(name, compiler).read_text(encoding="utf-8"))
    version = manifest.get("version")
    if type(version) is not int or version < 1:
        raise ValueError(
            f"Compiler version must be a positive integer: {name}/{compiler}"
        )
    return {"name": name.replace("_", "-"), "compiler": compiler, "version": version}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check",))
    parser.add_argument(
        "--candidate", choices=tuple(n.replace("_", "-") for n in CANDIDATES)
    )
    parser.add_argument("--compiler", choices=("one_take", "sequential"))
    args = parser.parse_args(argv)
    if args.compiler and not args.candidate:
        parser.error("--compiler requires --candidate")
    names = [args.candidate.replace("-", "_")] if args.candidate else CANDIDATES
    results = [
        compiler_identity(name, compiler)
        for name in names
        for compiler in ((args.compiler,) if args.compiler else compiler_keys(name))
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
