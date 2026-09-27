import json
import subprocess
import sys

from dotebench.cli import main


def test_validate_data_command(monkeypatch, tmp_path, capsys):
    from dotebench import cli

    monkeypatch.setattr(
        cli.dataset,
        "select_shards",
        lambda *a, **k: [{"category": "text", "shard": "easy"}],
    )
    monkeypatch.setattr(cli.dataset, "manifest_path", lambda *a, **k: tmp_path / "m")
    monkeypatch.setattr(cli.dataset, "sha256", lambda *a, **k: "manifest-sha")
    monkeypatch.setattr(cli.dataset, "load_cases", lambda *a, **k: ())

    assert main(["validate-data", "--audio-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"shards": 1, "cases": 0}


def test_report_requires_complete_evaluation(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"evaluation_status": "incomplete"}))
    result = subprocess.run(
        [sys.executable, "-m", "dotebench", "report", "--run", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "incomplete" in result.stderr


def test_completed_report(monkeypatch, tmp_path, capsys):
    from dotebench.evaluation import evaluator

    contract_root = tmp_path / "contract"
    (contract_root / "release").mkdir(parents=True)
    (contract_root / "release/evaluation-protocol.json").write_text(
        json.dumps({"version": "fixture_next"})
    )
    monkeypatch.setattr(evaluator, "resources", lambda: contract_root)
    (tmp_path / "run.json").write_text(json.dumps({"evaluation_status": "complete"}))
    summary = {
        "status": "complete",
        "metrics": {"protocol_version": "fixture_next", "score": 1},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert main(["report", "--run", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == summary


def test_report_rejects_incomplete_summary(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"evaluation_status": "complete"}))
    (tmp_path / "summary.json").write_text(json.dumps({"status": "incomplete"}))
    result = subprocess.run(
        [sys.executable, "-m", "dotebench", "report", "--run", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0


def test_report_rejects_mismatched_protocol(tmp_path):
    import pytest

    (tmp_path / "run.json").write_text(json.dumps({"evaluation_status": "complete"}))
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "metrics": {"protocol_version": "dotebench-formal-v5"},
            }
        )
    )
    with pytest.raises(SystemExit):
        main(["report", "--run", str(tmp_path)])


def test_evaluate_config_supplies_connection_and_timeout(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from dotebench import cli
    from dotebench.evaluation import evaluator
    from dotebench.services.utils import processes

    monkeypatch.setattr(cli.dataset, "select_shards", lambda *a, **k: [])
    monkeypatch.setattr(processes, "start_services", lambda **k: nullcontext())
    protocol = Mock()
    monkeypatch.setattr(evaluator, "DoteBenchEvaluator", protocol)
    runner = Mock()
    runner.return_value.run.return_value = {}
    monkeypatch.setattr(cli, "EvaluationRunner", runner)
    main(
        [
            "evaluate",
            "--audio-root",
            str(tmp_path),
            "--run",
            str(tmp_path / "run"),
            "--override",
            "evaluation.timeout=42",
            "--override",
            "services.emotion.port=19011",
        ]
    )
    assert protocol.call_args.kwargs["timeout"] == 42
    assert protocol.call_args.kwargs["urls"]["emotion"].endswith(":19011")
    assert "plugin" not in runner.call_args.kwargs["config"]["evaluation"]
