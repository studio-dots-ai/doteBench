"""Content-addressed identities for deterministic candidate compilers."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from dotebench.candidates.registry import CANDIDATES

ALGORITHM = "compiler-source-sha256-v2"


def source_path(module):
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
        raise ValueError(f"Compiler dependency source missing: {module}")
    return path


def dependencies(module):
    tree = ast.parse(source_path(module).read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                package = (
                    module
                    if source_path(module).name == "__init__.py"
                    else module.rsplit(".", 1)[0]
                )
                name = ".".join(
                    package.split(".")[: len(package.split(".")) - node.level + 1]
                    + ([name] if name else [])
                )
            names = [name]
            for alias in node.names:
                child = name + "." + alias.name
                if name == "dotebench" or name.startswith("dotebench."):
                    try:
                        source_path(child)
                    except ValueError:
                        pass
                    else:
                        names.append(child)
        else:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"__import__", "eval", "exec"}
            ):
                raise ValueError(
                    f"Dynamic compiler dependency is not allowed: {module}"
                )
            continue
        for name in names:
            if name == "dotebench" or name.startswith("dotebench."):
                result.add(name)
            if name == "importlib" or name.startswith("importlib."):
                raise ValueError(f"Dynamic compiler imports are not allowed: {module}")
    return result


def compiler_keys(name):
    return CANDIDATES[name].compilers


def entry_module(name, compiler):
    if name not in CANDIDATES or compiler not in compiler_keys(name):
        raise ValueError(f"Unknown compiler: {name}/{compiler}")
    return f"dotebench.candidates.{name}.compilers.{compiler}"


def manifest_path(name, compiler):
    return source_path(entry_module(name, compiler)).with_suffix(".json")


def compiler_dependencies(name, compiler):
    tree = ast.parse(source_path(entry_module(name, compiler)).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEPENDENCIES"
            for target in node.targets
        ):
            values = ast.literal_eval(node.value)
            if not isinstance(values, tuple) or not all(
                isinstance(v, str) for v in values
            ):
                raise ValueError(
                    "Compiler dependencies must be a tuple of compiler names"
                )
            return values
    return ()


def fingerprint(name, compiler="one_take", *, _parents=()):
    key = (name, compiler)
    if key in _parents:
        raise ValueError(f"Compiler dependency cycle: {name}/{compiler}")
    entry = entry_module(name, compiler)
    child_fingerprints = {
        child: fingerprint(name, child, _parents=(*_parents, key))
        for child in compiler_dependencies(name, compiler)
    }
    pending = [entry, "dotebench.instructions"]
    modules = set()
    while pending:
        module = pending.pop()
        if module in modules:
            continue
        modules.add(module)
        parts = module.split(".")
        pending.extend(
            ".".join(parts[:i])
            for i in range(1, len(parts))
            if ".".join(parts[:i]) not in modules
        )
        pending.extend(dependencies(module) - modules)
    for child in child_fingerprints.values():
        modules.update(child["sources"])
    if name != "identity":
        modules.update(
            {
                "dotebench.models.base",
                "dotebench.models.executor",
                "dotebench.models.native",
                f"dotebench.candidates.{name}.adapter",
            }
        )
    digest = hashlib.sha256(ALGORITHM.encode())
    digest.update(f"{name}/{compiler}".encode())
    files = {}
    for module in sorted(modules):
        content = source_path(module).read_bytes()
        files[module] = hashlib.sha256(content).hexdigest()
        for part in (module.encode(), content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    children = {child: value["sha256"] for child, value in child_fingerprints.items()}
    digest.update(json.dumps(children, sort_keys=True).encode())
    symbol = (
        "compile_request"
        if name == "identity"
        else ("OneTakeCompiler" if compiler == "one_take" else "SequentialCompiler")
    )
    return {
        "entrypoint": entry + ":" + symbol,
        "dependencies": children,
        "sources": files,
        "sha256": digest.hexdigest(),
    }


def compiler_identity(name, compiler="one_take"):
    expected = json.loads(manifest_path(name, compiler).read_text())
    actual = fingerprint(name, compiler)
    if expected != actual:
        raise ValueError(
            f"Compiler source changed for {name}/{compiler}; refresh its manifest after review"
        )
    for child in actual["dependencies"]:
        compiler_identity(name, child)
    return {
        "algorithm": ALGORITHM,
        **actual,
        "name": name.replace("_", "-"),
        "compiler": compiler,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "fingerprint", "refresh"))
    parser.add_argument(
        "--candidate", choices=tuple(n.replace("_", "-") for n in CANDIDATES)
    )
    parser.add_argument("--compiler", choices=("one_take", "sequential"))
    args = parser.parse_args(argv)
    if args.compiler and not args.candidate:
        parser.error("--compiler requires --candidate")
    names = [args.candidate.replace("-", "_")] if args.candidate else CANDIDATES
    results = []
    for name in names:
        for compiler in (args.compiler,) if args.compiler else compiler_keys(name):
            if args.action == "refresh":
                manifest_path(name, compiler).write_text(
                    json.dumps(fingerprint(name, compiler), indent=2) + "\n"
                )
            results.append(
                fingerprint(name, compiler)
                if args.action == "fingerprint"
                else compiler_identity(name, compiler)
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
