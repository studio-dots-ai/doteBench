"""CPU subprocess tests for service graph and lifecycle invariants."""

import socket
import sys
from pathlib import Path

import pytest

from dotebench.services import (
    load_service_bundle,
    resolve_service_graph,
    start_services,
)
from dotebench.services.utils.resolver import combine_resolved_service_graphs


@pytest.fixture(autouse=True)
def artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DOTEBENCH_SERVICE_OUTPUTS", str(tmp_path))


def declaration(key="a", port=None, **kwargs):
    return dict(
        instance_key=key,
        config_key=key,
        url=f"http://127.0.0.1:{port or (19890 + sum(map(ord, key)) % 100)}",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        **kwargs,
    )


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server(key, port):
    node = declaration(key, port, health_path="/", startup_timeout_sec=5)
    node.update(
        command=[sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
    )
    return node


def test_scoped_dependencies_deduplicate():
    child = declaration("shared")
    graph = resolve_service_graph(
        {
            "first": declaration("first", dependencies={"private": child}),
            "second": declaration("second", dependencies={"private": child}),
        }
    )
    assert len(graph.instances) == 3
    assert (
        graph.roots["first"].dependencies["private"]
        is graph.roots["second"].dependencies["private"]
    )
    assert len(graph.instances["shared"].callers) == 2
    assert graph.direct_services.optional_url("private") is None
    with pytest.raises(KeyError, match="direct service dependency"):
        graph.direct_services.url("private")


def test_conflicting_instance_and_cycle():
    with pytest.raises(ValueError, match="conflicting url"):
        resolve_service_graph({"a": declaration(), "b": declaration(port=19001)})
    node = declaration()
    node["dependencies"] = {"self": declaration()}
    with pytest.raises(ValueError, match="cycle"):
        resolve_service_graph({"a": node})


def test_port_collision_and_missing_command():
    with pytest.raises(ValueError, match="multiple service"):
        resolve_service_graph({"a": server("a", 19000), "b": server("b", 19000)})
    node = declaration()
    node["command"] = []
    with pytest.raises(ValueError, match="requires a command"):
        resolve_service_graph({"a": node})


def test_combine_validates_and_deduplicates():
    first = resolve_service_graph({"a": declaration()})
    combined = combine_resolved_service_graphs([first, first])
    assert len(combined.instances) == 1
    assert len(combined.roots) == 2
    with pytest.raises(ValueError, match="conflicting url"):
        combine_resolved_service_graphs(
            [first, resolve_service_graph({"b": declaration(port=19001)})]
        )


def test_hydra_composition_interpolation_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "services:\n  asr:\n    instance_key: asr\n    config_key: asr\n    command: [python, -m, example]\n    port: 18001\n    url: http://127.0.0.1:${.port}\n"
    )
    (tmp_path / "bundle.yaml").write_text("defaults:\n  - base\n  - _self_\n")
    graph = load_service_bundle(
        tmp_path / "bundle.yaml", overrides=["services.asr.port=19001"]
    )
    assert graph.direct_services.url("asr") == "http://127.0.0.1:19001"


def test_managed_readiness_and_cleanup(tmp_path):
    graph = resolve_service_graph({"web": server("web", free_port())})
    with start_services(
        graph=graph, project_root=tmp_path, log_dir=tmp_path / "logs"
    ) as group:
        assert len(group.processes) == 1
        assert group.processes[0].poll() is None
        assert all(path.exists() for path in group.log_paths.values())
    assert all(process.poll() is not None for process in group.processes)
    group.stop()


def test_rejects_external_mode():
    node = declaration()
    node["managed"] = False
    with pytest.raises(ValueError, match="All declared services"):
        resolve_service_graph({"a": node})


def test_timeout_and_early_exit_diagnostics(tmp_path):
    node = declaration(startup_timeout_sec=0.15)
    with pytest.raises(TimeoutError, match="timed out"):
        start_services(
            graph=resolve_service_graph({"web": node}),
            project_root=tmp_path,
            log_dir=tmp_path / "timeout",
        )
    node.update(
        startup_timeout_sec=5,
        command=[
            sys.executable,
            "-c",
            'print("failure-sentinel", flush=True); raise SystemExit(3)',
        ],
    )
    with pytest.raises(RuntimeError, match="failure-sentinel"):
        start_services(
            graph=resolve_service_graph({"web": node}),
            project_root=tmp_path,
            log_dir=tmp_path / "exit",
        )


def test_dependency_is_ready_before_consumer_and_failure_cleans_up(tmp_path):
    dependency = server("dependency", free_port())
    marker = tmp_path / "dependency_ready"
    consumer = server("consumer", free_port())
    consumer["dependencies"] = {"dependency": dependency}
    consumer["command"] = [
        sys.executable,
        "-c",
        f"from urllib.request import urlopen; from pathlib import Path; urlopen({dependency['url']!r}); Path({str(marker)!r}).touch(); raise SystemExit(7)",
    ]
    with pytest.raises(RuntimeError, match="returncode=7"):
        start_services(
            graph=resolve_service_graph({"consumer": consumer}),
            project_root=tmp_path,
            log_dir=tmp_path / "logs",
        )
    assert marker.exists()
    with socket.socket() as sock:
        assert (
            sock.connect_ex(("127.0.0.1", int(dependency["url"].rsplit(":", 1)[1])))
            != 0
        )


def test_public_evaluation_bundle():
    config_dir = Path(__file__).parents[2] / "configs"
    expected = {
        "qwen3_asr",
        "qwen3_aligner",
        "wer",
        "f0",
        "wdtw",
        "utmos",
        "speaker_similarity",
        "emotion",
    }
    graph = load_service_bundle(
        config_dir / "evaluation_bundle.yaml",
        overrides=["runtime.model_root=/models", "services.qwen3_asr.port=19033"],
    )
    assert set(graph.roots) == expected
    assert graph.roots["qwen3_asr"].url.endswith(":19033")
    assert "/models/Qwen3-ASR-1.7B" in graph.roots["qwen3_asr"].command
    assert len(graph.command_specs()) == 8
    assert graph.roots["wdtw"].dependencies["f0"] is graph.roots["f0"]
    assert "--f0-url" in graph.roots["wdtw"].command
    assert "dotebench.services.backends.f0" in graph.roots["f0"].command
    assert "dotebench.services.backends.wdtw" in graph.roots["wdtw"].command
    assert graph.command_specs().index(
        graph.roots["f0"].command_spec()
    ) < graph.command_specs().index(graph.roots["wdtw"].command_spec())


def test_invalid_timing():
    with pytest.raises(ValueError, match="invalid startup timing"):
        resolve_service_graph({"a": declaration(startup_timeout_sec=0)})


def test_argv_environment_and_unique_logs(tmp_path):
    port = free_port()
    node = server("web", port)
    node["env"] = {"SERVICE_TEST_VALUE": "literal $HOME value"}
    node["gpu_ids"] = "4,5"
    node["command"] = [
        sys.executable,
        "-c",
        (
            "import os; from http.server import HTTPServer, SimpleHTTPRequestHandler; "
            'print(os.environ["SERVICE_TEST_VALUE"], os.environ["CUDA_VISIBLE_DEVICES"], '
            'os.environ["BASE_ONLY"], flush=True); '
            f'HTTPServer(("127.0.0.1", {port}), SimpleHTTPRequestHandler).serve_forever()'
        ),
    ]
    paths = []
    for _ in range(2):
        with start_services(
            graph=resolve_service_graph({"web": node}),
            project_root=tmp_path,
            log_dir=tmp_path / "logs",
            base_env={"BASE_ONLY": "base"},
        ) as group:
            path = next(iter(group.log_paths.values()))
            paths.append(path)
            assert "literal $HOME value 4,5 base" in path.read_text()
    assert paths[0] != paths[1]


def test_metric_services_expose_deployment_settings_only():
    config = Path(__file__).parents[2] / "configs/evaluation_bundle.yaml"
    graph = load_service_bundle(
        config,
        overrides=[
            "services.qwen3_asr.port=19033",
            "services.qwen3_asr.env.HF_HUB_OFFLINE=0",
        ],
    )
    asr = graph.roots["qwen3_asr"]
    assert asr.command[asr.command.index("--port") + 1] == "19033"
    assert asr.env["HF_HUB_OFFLINE"] == "0"
    for service in graph.roots.values():
        assert "--options-json" not in service.command
    from hydra.errors import ConfigCompositionException

    with pytest.raises(ConfigCompositionException):
        load_service_bundle(
            config,
            overrides=["services.qwen3_asr.params.model_options.max_new_tokens=128"],
        )


def test_deployment_arguments_reach_real_process(tmp_path):
    import json

    port = free_port()
    executable = tmp_path / "fixture.py"
    executable.write_text("""import json, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
Path('received.json').write_text(json.dumps({'port': int(sys.argv[1]), 'literal': sys.argv[2], 'env': os.environ['FIXTURE_ENV']}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
""")
    literal = 'spaces "quotes" $HOME; $(touch sentinel)'
    config = tmp_path / "bundle.yaml"
    config.write_text(
        json.dumps(
            {
                "services": {
                    "fixture": {
                        "instance_key": "fixture",
                        "config_key": "fixture",
                        "port": 19000,
                        "startup_timeout_sec": 5,
                        "env": {"FIXTURE_ENV": "original"},
                        "command": [
                            sys.executable,
                            str(executable),
                            "${..port}",
                            literal,
                        ],
                    }
                }
            }
        )
    )
    graph = load_service_bundle(
        config,
        overrides=[
            f"services.fixture.port={port}",
            "services.fixture.env.FIXTURE_ENV=overridden",
        ],
    )
    with start_services(graph=graph, project_root=tmp_path, log_dir=tmp_path / "logs"):
        assert json.loads((tmp_path / "received.json").read_text()) == {
            "port": port,
            "env": "overridden",
            "literal": literal,
        }
        assert not (tmp_path / "sentinel").exists()


def test_service_command_requires_argument_array():
    node = declaration()
    node["command"] = "python -m service"
    with pytest.raises(ValueError, match="argument array"):
        resolve_service_graph({"example": node})


@pytest.mark.parametrize(
    "filename",
    [
        p.name
        for p in (Path(__file__).parents[2] / "configs").glob("*.yaml")
        if p.name != "base.yaml"
    ],
)
def test_every_public_config_is_standalone(filename, monkeypatch, tmp_path):
    for key in [
        "MODEL_PATH",
        "TOKENIZER_PATH",
        "SOURCE_ROOT",
        "RUNTIME_ROOT",
        "WEIGHTS_MANIFEST",
    ]:
        monkeypatch.setenv("DOTEBENCH_" + key, str(tmp_path / key))
    graph = load_service_bundle(Path(__file__).parents[2] / "configs" / filename)
    assert graph.roots
    for node in graph.instances.values():
        assert node.command
        assert node.url
        assert not hasattr(node, "managed")


def test_interruption_closes_process_group(tmp_path):
    graph = resolve_service_graph({"web": server("web", free_port())})
    with pytest.raises(KeyboardInterrupt), start_services(
        graph=graph, project_root=tmp_path, log_dir=tmp_path / "logs"
    ) as group:
        raise KeyboardInterrupt
    assert all(p.poll() is not None for p in group.processes)
