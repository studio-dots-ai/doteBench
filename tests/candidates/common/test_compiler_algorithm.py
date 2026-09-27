"""Compiler identity uses one algorithm definition for all candidates."""

import json

from dotebench import compilation
from tests.candidates.common.compiler_identity import isolate_candidate_sources


def test_compiler_algorithm_flows_to_manifest_and_identity(monkeypatch, tmp_path):
    isolate_candidate_sources(monkeypatch, tmp_path, "identity", ("one_take",))
    monkeypatch.setattr(compilation, "ALGORITHM", "fixture-next")
    compilation.main(["refresh", "--candidate", "identity", "--compiler", "one_take"])
    fingerprint = compilation.fingerprint("identity")
    assert (
        json.loads(compilation.manifest_path("identity", "one_take").read_text())
        == fingerprint
    )
    assert compilation.compiler_identity("identity") == {
        "algorithm": "fixture-next",
        **fingerprint,
        "name": "identity",
        "compiler": "one_take",
    }
