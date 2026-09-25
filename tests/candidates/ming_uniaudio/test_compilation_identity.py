"""Ming-UniAudio compiler identities cover their dependency graph."""

import pytest

from dotebench import compilation
from tests.candidates.common.compiler_identity import isolate_candidate_sources


def isolated_sources(monkeypatch, tmp_path):
    return isolate_candidate_sources(
        monkeypatch, tmp_path, "ming_uniaudio", ("one_take", "sequential")
    )


def test_relocation_preserves_compiler_hash(monkeypatch, tmp_path):
    expected = compilation.compiler_identity("ming_uniaudio", "sequential")
    isolated_sources(monkeypatch, tmp_path)
    assert compilation.compiler_identity("ming_uniaudio", "sequential") == expected


@pytest.mark.parametrize(
    "module",
    [
        "dotebench.instructions.rendering",
        "dotebench.candidates.ming_uniaudio.compile",
    ],
)
def test_changed_dependency_rejects_registration(monkeypatch, tmp_path, module):
    resolve = isolated_sources(monkeypatch, tmp_path)
    path = resolve(module)
    path.write_text(path.read_text() + "\n# source change\n")
    with pytest.raises(ValueError, match="Compiler source changed"):
        compilation.compiler_identity("ming_uniaudio", "sequential")


def test_refresh_selected_manifest_does_not_rewrite_sibling(monkeypatch, tmp_path):
    resolve = isolated_sources(monkeypatch, tmp_path)
    path = resolve("dotebench.candidates.ming_uniaudio.compilers.one_take")
    path.write_text(path.read_text() + "\n# reviewed change\n")
    sibling = compilation.manifest_path("ming_uniaudio", "sequential")
    before = sibling.read_bytes()
    compilation.main(
        ["refresh", "--candidate", "ming-uniaudio", "--compiler", "one_take"]
    )
    compilation.compiler_identity("ming_uniaudio", "one_take")
    assert sibling.read_bytes() == before
    with pytest.raises(ValueError, match="Compiler source changed"):
        compilation.compiler_identity("ming_uniaudio", "sequential")


def test_dependency_cycle_is_rejected(monkeypatch, tmp_path):
    resolve = isolated_sources(monkeypatch, tmp_path)
    path = resolve("dotebench.candidates.ming_uniaudio.compilers.one_take")
    path.write_text(
        path.read_text().replace("DEPENDENCIES = ()", 'DEPENDENCIES = ("sequential",)')
    )
    with pytest.raises(ValueError, match="dependency cycle"):
        compilation.fingerprint("ming_uniaudio", "sequential")


def test_dynamic_import_rejected(monkeypatch, tmp_path):
    resolve = isolated_sources(monkeypatch, tmp_path)
    path = resolve("dotebench.candidates.ming_uniaudio.compile")
    path.write_text(path.read_text() + "\nimport importlib\n")
    with pytest.raises(ValueError, match="Dynamic compiler imports"):
        compilation.fingerprint("ming_uniaudio")


def test_manifest_records_one_take_dependency():
    root = compilation.compiler_identity("ming_uniaudio", "sequential")
    child = compilation.compiler_identity("ming_uniaudio", "one_take")
    assert root["dependencies"] == {"one_take": child["sha256"]}
