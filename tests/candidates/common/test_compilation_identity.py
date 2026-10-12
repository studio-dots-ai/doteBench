"""Registered compiler versions and candidate selection."""

import json

import pytest

from dotebench import compilation

COMPILERS = [
    (name, compiler)
    for name in compilation.CANDIDATES
    for compiler in compilation.compiler_keys(name)
]


@pytest.mark.parametrize(("candidate", "compiler"), COMPILERS)
def test_registered_compiler_version(candidate, compiler):
    identity = compilation.compiler_identity(candidate, compiler)
    assert set(identity) == {"name", "compiler", "version"}
    assert identity["name"] == candidate.replace("_", "-")
    assert identity["compiler"] == compiler
    assert type(identity["version"]) is int
    assert identity["version"] >= 1


def test_default_compiler_version():
    assert compilation.compiler_identity("identity") == compilation.compiler_identity(
        "identity", "one_take"
    )


@pytest.mark.parametrize(("candidate", "compiler"), COMPILERS)
def test_individual_compiler_version_update(monkeypatch, tmp_path, candidate, compiler):
    original = compilation.manifest_path
    manifest = tmp_path / "compiler.json"
    expected = {
        (name, key): compilation.compiler_identity(name, key) for name, key in COMPILERS
    }
    expected[candidate, compiler]["version"] += 1
    manifest.write_text(
        json.dumps({"version": expected[candidate, compiler]["version"]})
    )
    monkeypatch.setattr(
        compilation,
        "manifest_path",
        lambda name, key: (
            manifest if (name, key) == (candidate, compiler) else original(name, key)
        ),
    )
    for name, key in COMPILERS:
        assert compilation.compiler_identity(name, key) == expected[name, key]


@pytest.mark.parametrize(
    ("options", "selected"),
    [
        ([], COMPILERS),
        (["--candidate", "identity"], [("identity", "one_take")]),
        (
            ["--candidate", "ming-uniaudio"],
            [("ming_uniaudio", "one_take"), ("ming_uniaudio", "sequential")],
        ),
        (
            ["--candidate", "auk-base", "--compiler", "sequential"],
            [("auk_base", "sequential")],
        ),
    ],
)
def test_check_selected_versions(capsys, options, selected):
    compilation.main(["check", *options])
    assert json.loads(capsys.readouterr().out) == [
        compilation.compiler_identity(name, compiler) for name, compiler in selected
    ]
