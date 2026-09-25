"""Candidate compilers consume only public benchmark interfaces."""

import ast
import importlib.util
from pathlib import Path

import dotebench

PACKAGE = Path(dotebench.__file__).parent


def imports(path):
    relative = path.relative_to(PACKAGE).with_suffix("")
    parts = relative.parts
    module = ".".join(("dotebench", *parts))
    package = module.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = importlib.util.resolve_name("." * node.level + name, package)
            for alias in node.names:
                yield name, alias.name


def test_candidate_compilation_uses_public_shared_interfaces():
    candidates = PACKAGE / "candidates"
    paths = [
        *candidates.glob("*/compile.py"),
        *candidates.glob("*/compilers/*.py"),
        candidates / "common/selection.py",
        candidates / "auk_base/alignment.py",
    ]
    for path in paths:
        for module, symbol in imports(path):
            assert not module.startswith("dotebench.evaluation"), (path, module)
            if module.startswith("dotebench.") and not module.startswith(
                "dotebench.candidates."
            ):
                assert symbol is None or not symbol.startswith("_"), (
                    path,
                    module,
                    symbol,
                )
