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
    assert compilation.compiler_identity(candidate, compiler) == {
        "name": candidate.replace("_", "-"),
        "compiler": compiler,
        "version": 1,
    }


def test_default_compiler_version():
    assert compilation.compiler_identity("identity") == {
        "name": "identity",
        "compiler": "one_take",
        "version": 1,
    }


@pytest.mark.parametrize(("candidate", "compiler"), COMPILERS)
def test_individual_compiler_version_update(monkeypatch, tmp_path, candidate, compiler):
    original = compilation.manifest_path
    manifest = tmp_path / "compiler.json"
    manifest.write_text('{"version": 2}\n')
    monkeypatch.setattr(
        compilation,
        "manifest_path",
        lambda name, key: (
            manifest if (name, key) == (candidate, compiler) else original(name, key)
        ),
    )
    for name, key in COMPILERS:
        assert compilation.compiler_identity(name, key) == {
            "name": name.replace("_", "-"),
            "compiler": key,
            "version": 2 if (name, key) == (candidate, compiler) else 1,
        }


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
