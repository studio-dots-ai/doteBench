"""dots.tts.edit CLI paths preserve explicit artifact configuration."""

from pathlib import Path

from dotebench.services import load_service_bundle


def test_service_host_override_updates_bind_arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path))
    config = Path(__file__).parents[3] / "configs" / "dots_tts_edit.yaml"
    graph = load_service_bundle(config, overrides=["runtime.host=127.0.0.2"])
    for node in graph.roots.values():
        assert node.url.startswith("http://127.0.0.2:")
        assert node.command[node.command.index("--host") + 1] == "127.0.0.2"


def test_explicit_artifact_root_is_preserved(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from dotebench import cli
    from dotebench.candidates.dots_tts_edit.adapter import DotsTtsEdit
    from dotebench.services.utils import processes

    monkeypatch.setenv(
        "DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path / "explicit-artifacts")
    )
    monkeypatch.setattr(cli.dataset, "select_shards", lambda *a, **k: [])
    monkeypatch.setattr(DotsTtsEdit, "compilation_identity", lambda self: {})
    monkeypatch.setattr(processes, "start_services", lambda **k: nullcontext())
    runner = Mock()
    runner.return_value.run.return_value = {}
    monkeypatch.setattr(cli, "GenerationRunner", runner)
    cli.main(
        [
            "generate",
            "--candidate",
            "dots-tts-edit",
            "--audio-root",
            str(tmp_path),
            "--run",
            str(tmp_path / "run"),
        ]
    )
    options = runner.call_args.kwargs["config"]["generation"]["options"]
    assert options["scratch_dir"] == str(
        tmp_path / "explicit-artifacts/candidate-scratch"
    )
    runner.call_args.kwargs["candidate"].close()
