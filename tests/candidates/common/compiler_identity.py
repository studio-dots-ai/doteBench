"""Isolated source trees for candidate compiler identity tests."""

import shutil

from dotebench import compilation


def isolate_candidate_sources(monkeypatch, tmp_path, candidate, compilers):
    original = compilation.source_path
    keys = tuple((candidate, compiler) for compiler in compilers)
    modules = {
        module for key in keys for module in compilation.fingerprint(*key)["sources"]
    }
    for module in modules:
        path = original(module)
        target = tmp_path / (
            module.replace(".", "/")
            + ("/__init__.py" if path.name == "__init__.py" else ".py")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name, compiler in keys:
        manifest = compilation.manifest_path(name, compiler)
        target = tmp_path / "dotebench/candidates" / name / "compilers" / manifest.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest, target)

    def resolve(module):
        path = tmp_path / (module.replace(".", "/") + ".py")
        if not path.exists():
            path = tmp_path / module.replace(".", "/") / "__init__.py"
        if not path.exists():
            raise ValueError(module)
        return path

    monkeypatch.setattr(compilation, "source_path", resolve)
    return resolve
