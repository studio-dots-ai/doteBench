"""Runtime preparation preserves upstream source and detects tampering."""

import hashlib
import json
import subprocess

import pytest

from dotebench.candidates.common import runtime


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def fixture(monkeypatch, tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-q")
    (upstream / "model.py").write_text("value = 1\n")
    git(upstream, "add", ".")
    git(
        upstream,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "Fixture",
    )
    directory = tmp_path / "candidate"
    directory.mkdir()
    patch = "--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    (directory / "inference.patch").write_text(patch)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    manifest = {
        "upstream": {"commit": git(upstream, "rev-parse", "HEAD")},
        "patches": [{"path": "inference.patch", "sha256": digest(patch)}],
        "files": {
            "model.py": {
                "before": digest("value = 1\n"),
                "after": digest("value = 2\n"),
            }
        },
    }
    (directory / "runtime.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "candidate_directory", lambda name: directory)
    monkeypatch.setattr(
        runtime,
        "runtime_identity",
        lambda name: {"sha256": "test", "compiler": "fixed"},
    )
    return upstream, tmp_path / "prepared"


def test_prepare_reuse_and_tamper_detection(monkeypatch, tmp_path):
    upstream, destination = fixture(monkeypatch, tmp_path)
    prepared = runtime.prepare_runtime(
        "test", upstream=upstream, destination=destination
    )
    assert (prepared / "model.py").read_text() == "value = 2\n"
    assert (upstream / "model.py").read_text() == "value = 1\n"
    assert git(upstream, "status", "--porcelain") == ""
    assert (
        runtime.prepare_runtime("test", upstream=upstream, destination=destination)
        == prepared
    )
    (prepared / "model.py").write_text("value = 3\n")
    with pytest.raises(ValueError, match="Prepared runtime changed"):
        runtime.prepare_runtime("test", upstream=upstream, destination=destination)


def test_dirty_upstream_rejected(monkeypatch, tmp_path):
    upstream, destination = fixture(monkeypatch, tmp_path)
    (upstream / "extra.py").write_text("pass\n")
    with pytest.raises(ValueError, match="must be clean"):
        runtime.prepare_runtime("test", upstream=upstream, destination=destination)
    assert not destination.exists()


def test_concurrent_runtime_initialization(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    upstream, destination = fixture(monkeypatch, tmp_path)

    def prepare(_):
        return runtime.prepare_runtime(
            "test", upstream=upstream, destination=destination
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(prepare, range(4)))
    assert len(set(paths)) == 1
    assert (paths[0] / "model.py").read_text() == "value = 2\n"
    assert len(list(destination.glob("test-*"))) == 1
    assert not list(destination.glob("prepare-*"))


@pytest.mark.parametrize('change', [None, 'commit', 'url', 'patch'])
def test_source_registration_rejects_drift(monkeypatch, tmp_path, change):
    _upstream, _ = fixture(monkeypatch, tmp_path)
    directory = tmp_path / 'candidate'
    path = directory / 'runtime.json'
    manifest = json.loads(path.read_text())
    manifest['upstream'].update(directory='fixture', url='https://example.org/model')
    path.write_text(json.dumps(manifest))
    commit = manifest['upstream']['commit']
    def checked(command):
        if 'ls-tree' in command:
            return '160000 commit ' + ('0' * 40 if change == 'commit' else commit) + '\tthird_party/fixture'
        return 'wrong-url' if change == 'url' else manifest['upstream']['url']
    monkeypatch.setattr(runtime, 'checked', checked)
    if change == 'patch':
        (directory / 'inference.patch').write_text('changed')
    if change:
        with pytest.raises(ValueError):
            runtime.check_source_registration('test', tmp_path)
    else:
        runtime.check_source_registration('test', tmp_path)
