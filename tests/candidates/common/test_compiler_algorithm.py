"""Compiler identity uses one algorithm definition for all candidates."""

import json

import pytest

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


@pytest.mark.parametrize(
    ("candidate", "compiler"),
    [
        (name, key)
        for name in compilation.CANDIDATES
        for key in compilation.compiler_keys(name)
    ],
)
def test_compiler_identity_is_stable_across_distribution_versions(
    monkeypatch, tmp_path, candidate, compiler
):
    original = compilation.source_path
    expected = compilation.fingerprint(candidate, compiler)
    version_file = tmp_path / "__init__.py"
    version_file.write_text('RELEASE = "99.0.0"\n')
    monkeypatch.setattr(
        compilation,
        "source_path",
        lambda module: version_file if module == "dotebench" else original(module),
    )
    assert compilation.fingerprint(candidate, compiler) == expected
