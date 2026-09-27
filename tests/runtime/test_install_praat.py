"""Installer integrity and environment-local deployment contracts."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def installer(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "install_praat", Path(__file__).parents[2] / "scripts/install_praat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("fixture")
    (environment / "bin").mkdir()
    data = b"fixture executable"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as package:
        member = tarfile.TarInfo("praat_barren")
        member.size = len(data)
        package.addfile(member, io.BytesIO(data))
    payload = archive.getvalue()
    monkeypatch.setattr(module, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module, "BINARY_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: io.BytesIO(payload))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="Praat 6.1.38 (January 2 2021)\n"),
    )
    return module, environment


def test_install_and_offline_reuse(monkeypatch, tmp_path):
    module, environment = installer(monkeypatch, tmp_path)
    result = module.install(environment)
    assert result == environment / "bin/praat"
    assert result.stat().st_mode & 0o111
    monkeypatch.setattr(
        module, "urlopen", lambda *a, **k: pytest.fail("download on reuse")
    )
    assert module.install(environment) == result
    result.write_bytes(b"different installation")
    with pytest.raises(FileExistsError):
        module.install(environment)
    assert result.read_bytes() == b"different installation"


@pytest.mark.parametrize("failure", ["archive", "binary", "version", "network"])
def test_failed_install_leaves_no_executable_or_scratch(monkeypatch, tmp_path, failure):
    module, environment = installer(monkeypatch, tmp_path)
    if failure in {"archive", "binary"}:
        monkeypatch.setattr(module, failure.upper() + "_SHA256", "0" * 64)
    elif failure == "version":
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout="Praat 7.0"),
        )
    else:

        def fail(*a, **k):
            raise OSError("network unavailable")

        monkeypatch.setattr(module, "urlopen", fail)
    with pytest.raises((ValueError, OSError)):
        module.install(environment)
    assert list((environment / "bin").iterdir()) == []


def test_default_executable_uses_service_environment(monkeypatch):
    import sys
    from dotebench.services.backends import f0

    monkeypatch.setenv("PRAAT_BIN", "/another/environment/praat")
    received = []

    def pool(binary, workers, timeout):
        received.append(binary)
        return SimpleNamespace(binary=binary, version="fixture", close=lambda: None)

    monkeypatch.setattr(f0, "PraatPool", pool)
    f0.build()
    assert received == [str(Path(sys.prefix) / "bin/praat")]
