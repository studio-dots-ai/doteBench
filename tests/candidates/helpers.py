"""Shared fixtures and assertions for independently runnable candidate suites."""

import base64
import importlib
import io
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from dotebench.domain import Audio, GenerationRequest
from dotebench.instructions import parse_instruction


def audio_request(xml, language="en"):
    stream = io.BytesIO()
    sf.write(stream, np.zeros(100), 16000, format="WAV")
    return GenerationRequest("case", language, Audio(stream.getvalue()), xml)


def wav(value):
    stream = io.BytesIO()
    sf.write(
        stream,
        np.full(160, value, dtype="float32"),
        16000,
        format="WAV",
        subtype="FLOAT",
    )
    return Audio(stream.getvalue())


def session_with(responses):
    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    session.post.side_effect = responses
    return session


def assert_first_acoustic_uses_complete_target(adapter_class, tmp_path):
    request = audio_request(
        '<sub targ="Goodbye">Hello</sub> <pitch semitones="3">world</pitch> '
        '<rate factor="0.8">today</rate>'
    )
    adapter = adapter_class(url="http://native", scratch_dir=str(tmp_path))
    actual = adapter.compiler.bind_request(resolved={}, request=request)
    assert actual.selected_operation_index == 1
    assert actual.scope == "global"
    assert actual == adapter.compiler.bind_request(resolved={}, request=request)
    fields = dict(actual.fields)
    assert parse_instruction(request.instruction_xml).target_text in str(fields)
    assert "0.8" not in str(fields)
    assert "<pitch" not in str(fields) and "<rate" not in str(fields)
    assert 'only "world"' not in str(fields)


def assert_same_position_keeps_xml_order(adapter_class, tmp_path):
    request = audio_request('<pause act="ins"/><pitch semitones="3">Hello</pitch>')
    compiled = adapter_class(
        url="http://native", scratch_dir=str(tmp_path)
    ).compiler.bind_request(resolved={}, request=request)
    assert compiled.selected_operation_index == 0
    assert "semitone" not in str(compiled.fields)


def assert_native_transport_one_request(adapter_class, tmp_path):
    request = audio_request('<pitch semitones="-3">Hello</pitch>')
    session = Mock()
    session.get.return_value.json.return_value = {"status": "healthy"}
    response = session.post.return_value
    response.ok = True
    response.headers = {}
    response.json.return_value = {
        "audio_base64": base64.b64encode(request.source_audio.data).decode()
    }
    response.content = request.source_audio.data
    captured = []

    def observe(case_id, compiled, payload):
        captured.append((case_id, compiled, payload))
        assert "instruction_xml" not in payload
        if compiled.audio_encoding == "file":
            assert (
                Path(payload[compiled.audio_field]).read_bytes()
                == request.source_audio.data
            )
        else:
            assert (
                base64.b64decode(payload[compiled.audio_field]["data"])
                == request.source_audio.data
            )

    adapter = adapter_class(
        url="http://native",
        scratch_dir=str(tmp_path),
        session=session,
        request_observer=observe,
    )
    adapter.prepare()
    assert adapter.generate(request).data == request.source_audio.data
    adapter.close()
    session.post.assert_called_once()
    session.close.assert_called_once()
    assert len(captured) == 1
    assert not list(tmp_path.glob("source-*"))
    assert adapter.last_execution_trace.is_file()


def assert_cli_wiring(monkeypatch, tmp_path, candidate_name, *, has_aligner=False):
    from dotebench import cli
    from dotebench.candidates.registry import CANDIDATES
    from dotebench.services.utils import processes

    for key in (
        "MODEL_PATH",
        "TOKENIZER_PATH",
        "SOURCE_ROOT",
        "RUNTIME_ROOT",
        "WEIGHTS_MANIFEST",
    ):
        monkeypatch.setenv("DOTEBENCH_" + key, str(tmp_path / key))
    evaluator = ModuleType("dotebench.evaluation.evaluator")
    evaluator.DoteBenchEvaluator = object
    monkeypatch.setitem(sys.modules, evaluator.__name__, evaluator)
    monkeypatch.setattr(cli.dataset, "select_shards", lambda *args, **kwargs: [])
    candidate_class = getattr(
        importlib.import_module(f"dotebench.candidates.{candidate_name}.adapter"),
        CANDIDATES[candidate_name].adapter,
    )
    monkeypatch.setattr(
        candidate_class, "compilation_identity", lambda self: {"sha256": "test"}
    )
    start = Mock(side_effect=lambda **kwargs: nullcontext())
    monkeypatch.setattr(processes, "start_services", start)
    runner = Mock()
    runner.return_value.run.return_value = {}
    monkeypatch.setattr(cli, "GenerationRunner", runner)
    cli.main(
        [
            "generate",
            "--candidate",
            candidate_name.replace("_", "-"),
            "--compiler",
            "sequential",
            "--audio-root",
            str(tmp_path),
            "--run",
            str(tmp_path / "run"),
            "--override",
            "services.candidate.port=19017",
        ]
    )
    kwargs = runner.call_args.kwargs
    assert kwargs["candidate"].url == "http://127.0.0.1:19017"
    assert kwargs["config"]["generation"]["candidate"] == candidate_name.replace(
        "_", "-"
    )
    assert kwargs["config"]["generation"]["compiler"] == "sequential"
    node = start.call_args.kwargs["graph"].roots["candidate"]
    assert node.command[node.command.index("--port") + 1] == "19017"
    if has_aligner:
        assert kwargs["config"]["generation"]["options"]["aligner_url"].endswith(
            ":18001"
        )
    kwargs["candidate"].close()
