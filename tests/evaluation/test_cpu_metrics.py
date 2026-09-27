"""Deterministic CPU ability tests using known counts and generated waveforms."""

import base64
import io
import shutil

import numpy as np
import pytest
import soundfile as sf

from dotebench.services.backends import f0, wdtw, wer


def wav_payload(audio, sr=16000):
    stream = io.BytesIO()
    sf.write(stream, audio, sr, format="WAV", subtype="FLOAT")
    return base64.b64encode(stream.getvalue()).decode()


def test_wer_counts_and_language_units():
    measure = wer.build()
    assert (
        measure({"reference": "Hello, WORLD!", "hypothesis": "hello word"})[
            "error_rate"
        ]
        == 0.5
    )
    result = measure({"reference": "你好 world", "hypothesis": "你 world"})
    assert result["ref_length"] == 3
    assert result["deletions"] == 1
    assert result["error_rate"] == pytest.approx(1 / 3)
    result = measure({"reference": "", "hypothesis": "word"})
    assert result["error_rate"] is None
    assert result["insertions"] == 1
    assert result["status"] == "empty_reference"


def preservation_payload(source, target):
    return {
        "duration_source": source,
        "duration_target": target,
        "f0_source": source,
        "f0_target": target,
        "source_audio": wav_payload(np.zeros(32000, dtype=np.float32)),
        "target_audio": wav_payload(np.zeros(32000, dtype=np.float32)),
    }


def test_wdtw_duration_and_empty_denominator(monkeypatch):
    def f0_response(_session, url, *, json, timeout):
        assert url == "http://f0.test/measure"
        assert timeout == 300
        return type(
            "Response",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"spans": [{} for _ in json["spans"]]},
            },
        )()

    monkeypatch.setattr("requests.Session.post", f0_response)
    measure = wdtw.build(f0_url="http://f0.test")
    source = [{"word": "a", "start": 0, "end": 1}]
    assert measure(preservation_payload(source, source))["wdtw_dur"]["wdtw_dur"] == 0
    result = measure(
        preservation_payload(source, [{"word": "a", "start": 0, "end": 2}])
    )["wdtw_dur"]
    assert result["distance"] == 1
    assert result["normalizer"] == 3
    assert result["wdtw_dur"] == pytest.approx(1 / 3)
    assert measure(preservation_payload([], []))["wdtw_dur"]["wdtw_dur"] is None
    with pytest.raises(ValueError, match="finite 0 <= start <= end"):
        measure(preservation_payload([{"word": "a", "start": 1, "end": 0}], source))
    with pytest.raises(ValueError, match="duration_source"):
        measure({"source_alignments": source, "target_alignments": source})


def test_f0_known_tone_silence_and_short_span():
    if not shutil.which(f0.DEFAULT_PRAAT_BIN):
        pytest.skip("Praat CLI is unavailable")
    measure = f0.build()
    sr = 16000
    audio = (0.5 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    result = measure(
        {
            "audio": wav_payload(audio),
            "spans": [{"start": 0, "end": 1}, {"start": 0, "end": 0.01}],
        }
    )
    assert result["spans"][0]["median_f0_hz"] == pytest.approx(200, abs=0.01)
    assert result["spans"][0]["voiced_fraction"] == 1
    assert result["spans"][1]["mean_f0_hz"] is None
    assert result["spans"][1]["num_frames"] == 0
    silence = measure({"audio": wav_payload(np.zeros(sr, dtype=np.float32))})
    assert silence["spans"][0]["mean_f0_hz"] is None
    assert silence["spans"][0]["voiced_fraction"] == 0
    assert measure({"audio": wav_payload(audio), "spans": []})["spans"] == []
    with pytest.raises(ValueError, match="f0_min_hz"):
        measure({"audio": wav_payload(audio), "f0_min_hz": 700})
    with pytest.raises(ValueError, match="span"):
        measure({"audio": wav_payload(audio), "spans": [{"start": 1, "end": 0}]})


@pytest.mark.parametrize(
    "ability,payload,expected_key",
    [
        ("wer", {"reference": "hello world", "hypothesis": "hello word"}, "error_rate"),
        (
            "wdtw",
            preservation_payload(
                [{"word": "a", "start": 0, "end": 1}],
                [{"word": "a", "start": 0, "end": 1}],
            ),
            "wdtw_dur",
        ),
        ("f0", {"audio": wav_payload(np.zeros(1600, dtype=np.float32))}, "spans"),
    ],
)
def test_real_cpu_service_http_lifecycle(tmp_path, ability, payload, expected_key):
    import json
    import socket
    import sys
    from pathlib import Path
    from urllib.request import Request, urlopen

    from dotebench.services import resolve_service_graph, start_services

    if ability in {"f0", "wdtw"} and not shutil.which(f0.DEFAULT_PRAAT_BIN):
        pytest.skip("Praat CLI is unavailable")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = Path(__file__).parents[2]
    url = f"http://127.0.0.1:{port}"
    module = f"dotebench.services.backends.{ability}"
    node = {
        "instance_key": ability,
        "config_key": ability,
        "url": url,
        "startup_timeout_sec": 30,
        "command": [
            sys.executable,
            "-m",
            module,
            "--port",
            str(port),
        ],
    }
    if ability == "wdtw":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            f0_port = sock.getsockname()[1]
        f0_url = f"http://127.0.0.1:{f0_port}"
        node["dependencies"] = {
            "f0": {
                "instance_key": "f0",
                "config_key": "f0",
                "url": f0_url,
                "startup_timeout_sec": 30,
                "command": [
                    sys.executable,
                    "-m",
                    "dotebench.services.backends.f0",
                    "--port",
                    str(f0_port),
                ],
            }
        }
        node["command"].extend(["--f0-url", f0_url])
        node["env"] = {"PATH": "/unavailable-in-wdtw"}
    with start_services(
        graph=resolve_service_graph({ability: node}),
        project_root=root,
        log_dir=tmp_path / "logs",
        base_env={"PYTHONPATH": str(root / "src")},
    ) as group:
        if ability == "wdtw":
            assert len(group.processes) == 2
            assert "--praat-bin" not in node["command"]
        with urlopen(url + "/health") as response:
            health = json.load(response)
            assert health["status"] == "ready"
            assert health["ability"] == ability
            assert health["model_identity"] is None
        request = Request(
            url + "/measure",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.load(response)
        assert expected_key in result
        if ability == "wer":
            assert result["error_rate"] == 0.5
        elif ability == "wdtw":
            assert result["wdtw_dur"]["wdtw_dur"] == 0
        else:
            assert result["spans"][0]["voiced_fraction"] == 0
    assert all(process.poll() is not None for process in group.processes)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_wdtw_rejects_invalid_selected_spans_before_f0(bad):
    source = [{"word": "a", "start": bad, "end": 1}]
    with pytest.raises(ValueError, match="finite"):
        wdtw.measure_spans(
            preservation_payload(source, source),
            lambda payload: pytest.fail("invalid segments must not reach F0"),
        )


def test_wdtw_preserves_raw_word_and_requires_complete_f0_response():
    segment = {"word": "a", "raw_word": "A", "start": 0, "end": 1}
    payload = preservation_payload([segment], [segment])
    with pytest.raises(ValueError, match="every requested span"):
        wdtw.measure_spans(payload, lambda p: {"spans": []})
    result = wdtw.measure_spans(payload, lambda p: {"spans": [{}]})
    assert result["wdtw_f0"]["word_results"][0]["source_span"]["raw_word"] == "A"
    payload["f0_target"] = []
    with pytest.raises(ValueError, match="counts must match"):
        wdtw.measure_spans(payload, lambda p: pytest.fail("unpaired segments"))
