"""Keep instruction semantics and shared utilities independent of consumers."""

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


def test_instruction_semantics_have_one_way_dependencies():
    allowed = {
        "__init__.py": {"parser", "rendering", "types"},
        "parser.py": {"rendering", "types"},
        "rendering.py": {"types"},
        "types.py": set(),
    }
    for path in (PACKAGE / "instructions").glob("*.py"):
        for module, _ in imports(path):
            if module.startswith("dotebench"):
                assert module in {
                    "dotebench.instructions." + name for name in allowed[path.name]
                }, (path, module)


def test_shared_text_and_alignment_do_not_depend_on_consumers():
    for filename in ("text.py", "alignment.py"):
        for module, _ in imports(PACKAGE / filename):
            if module.startswith("dotebench"):
                assert module == "dotebench.text", (filename, module)
