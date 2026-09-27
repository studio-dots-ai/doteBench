"""Public services stay usable without consumer domain implementations."""

import ast
import importlib.util
from pathlib import Path


def test_services_have_no_transitive_consumer_imports():
    root = Path(__file__).parents[2] / "src"
    modules = {}
    for path in (root / "dotebench").rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[name] = path
    forbidden = (
        "dotebench.evaluation",
        "dotebench.candidates",
        "dotebench.materialization",
    )
    pending = [
        (name, [name]) for name in modules if name.startswith("dotebench.services")
    ]
    visited = set()
    while pending:
        name, chain = pending.pop()
        assert not any(name == f or name.startswith(f + ".") for f in forbidden), chain
        if name in visited or name not in modules:
            continue
        visited.add(name)
        path = modules[name]
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                dependencies.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = (
                    importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                    if node.level
                    else node.module
                )
                dependencies.add(module)
                dependencies.update(module + "." + alias.name for alias in node.names)
            elif isinstance(node, ast.Call) and node.args:
                # Include literal dynamic imports used by lazy model loaders.
                func = node.func
                if (isinstance(func, ast.Name) and func.id == "__import__") or (
                    isinstance(func, ast.Attribute) and func.attr == "import_module"
                ):
                    if isinstance(node.args[0], ast.Constant) and isinstance(
                        node.args[0].value, str
                    ):
                        module = node.args[0].value
                        dependencies.add(importlib.util.resolve_name(module, package))
        for dependency in dependencies:
            if not dependency or not dependency.startswith("dotebench"):
                continue
            # Importing a module also executes each parent package initializer.
            parts = dependency.split(".")
            for i in range(1, len(parts) + 1):
                target = ".".join(parts[:i])
                pending.append((target, chain + [target]))
