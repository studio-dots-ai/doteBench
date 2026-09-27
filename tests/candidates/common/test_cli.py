"""Candidate selection fails before execution when preflight contracts fail."""

import pytest


def test_preflight_precedes_service_start(monkeypatch, tmp_path):
    from unittest.mock import Mock

    import pytest

    from dotebench import cli
    from dotebench.candidates.identity.adapter import Identity
    from dotebench.services.utils import processes

    monkeypatch.setattr(cli.dataset, "select_shards", lambda *a, **k: [])
    monkeypatch.setattr(
        Identity,
        "compilation_identity",
        Mock(side_effect=ValueError("changed compiler")),
    )
    start = Mock()
    monkeypatch.setattr(processes, "start_services", start)
    with pytest.raises(ValueError, match="changed compiler"):
        cli.main(
            ["generate", "--audio-root", str(tmp_path), "--run", str(tmp_path / "run")]
        )
    start.assert_not_called()


def test_candidate_mismatch_fails_before_start(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from dotebench import cli
    from dotebench.services.utils import processes

    monkeypatch.setattr(cli.dataset, "select_shards", lambda *a, **k: [])
    start = Mock()
    monkeypatch.setattr(processes, "start_services", start)
    config = tmp_path / "config.yaml"
    config.write_text("candidate:\n  name: different-candidate\nservices: {}\n")
    with pytest.raises(SystemExit):
        cli.main(
            [
                "generate",
                "--candidate",
                "identity",
                "--audio-root",
                str(tmp_path),
                "--run",
                str(tmp_path / "run"),
                "--config",
                str(config),
            ]
        )
    start.assert_not_called()
