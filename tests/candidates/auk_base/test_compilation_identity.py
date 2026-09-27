"""AuK Base compiler identity includes shared alignment behavior."""

import pytest

from dotebench import compilation
from tests.candidates.common.compiler_identity import isolate_candidate_sources


def test_shared_alignment_change_rejects_registration(monkeypatch, tmp_path):
    resolve = isolate_candidate_sources(
        monkeypatch, tmp_path, "auk_base", ("one_take",)
    )
    path = resolve("dotebench.alignment")
    path.write_text(path.read_text() + "\n# reviewed alignment change\n")
    with pytest.raises(ValueError, match="Compiler source changed"):
        compilation.compiler_identity("auk_base", "one_take")
