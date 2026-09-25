"""Entrypoint and protocol regression checks without neural checkpoints."""

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from dotebench.evaluation.models import MODELS


@pytest.mark.parametrize("name", ["wer", "f0", "wdtw", *MODELS])
def test_entrypoint_builds_own_backend_and_closes_resources(
    monkeypatch, tmp_path, name
):
    backend = importlib.import_module(f"dotebench.services.backends.{name}")
    events = []

    def measure(payload):
        return {"received": payload}

    measure.configuration = {"test": name}
    measure.close = lambda: events.append("closed")

    def build(*args, **kwargs):
        events.append(("build", args, kwargs))
        return measure

    monkeypatch.setattr(backend, "build", build)
    argv = [name, "--port", "12345"]
    identity = None
    if name in MODELS:
        identity = {"model_id": MODELS[name]}
        assert backend.MODEL_ID == MODELS[name]

        def prepare(model_id, path):
            assert model_id == MODELS[name]
            assert path == tmp_path
            assert not events
            events.append("verified")
            return identity

        monkeypatch.setattr(backend, "prepared_identity", prepare)
        argv += ["--model-path", str(tmp_path), "--device", "cpu"]
    elif name == "wdtw":
        argv += ["--f0-url", "http://f0.test"]

    def run(app, *, host, port):
        assert (host, port) == ("127.0.0.1", 12345)
        endpoints = {route.path: route.endpoint for route in app.routes}

        async def exercise():
            async with app.router.lifespan_context(app):
                assert endpoints["/health"]() == {
                    "status": "ready",
                    "ability": name,
                    "model_identity": identity,
                    "configuration": {"test": name},
                }
                assert endpoints["/measure"]({"test": 1}) == {"received": {"test": 1}}
                assert "closed" not in events

        asyncio.run(exercise())

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.setattr(sys, "argv", argv)
    backend.main()
    assert events.count("closed") == 1
    built = next(event for event in events if isinstance(event, tuple))
    if name in MODELS:
        assert built[1:] == ((tmp_path, "cpu"), {})
    elif name == "wdtw":
        assert built[2] == {"f0_url": "http://f0.test", "f0_timeout_sec": 300.0}


@pytest.mark.parametrize("name", list(MODELS))
def test_unprepared_model_fails_before_loading(monkeypatch, tmp_path, name):
    backend = importlib.import_module(f"dotebench.services.backends.{name}")
    monkeypatch.setattr(backend, "build", lambda *a, **k: pytest.fail("model loaded"))
    monkeypatch.setattr(
        sys, "argv", [name, "--port", "12345", "--model-path", str(tmp_path)]
    )
    with pytest.raises(FileNotFoundError):
        backend.main()


@pytest.mark.parametrize("failure", ["timeout", "unavailable"])
def test_wdtw_propagates_f0_transport_faults_and_releases_session(monkeypatch, failure):
    import requests

    from dotebench.services.backends import wdtw

    closed = []

    def post(session, *args, **kwargs):
        if failure == "timeout":
            raise requests.Timeout("F0 deadline exceeded")
        raise requests.ConnectionError("F0 unavailable")

    monkeypatch.setattr(requests.Session, "post", post)
    monkeypatch.setattr(
        requests.Session, "close", lambda session: closed.append(session)
    )
    measure = wdtw.build(f0_url="http://f0.test", f0_timeout_sec=0.05)
    payload = {
        "duration_source": [{"word": "a", "start": 0, "end": 1}],
        "duration_target": [{"word": "a", "start": 0, "end": 1}],
        "f0_source": [{"word": "a", "start": 0, "end": 1}],
        "f0_target": [{"word": "a", "start": 0, "end": 1}],
        "source_audio": "unused",
        "target_audio": "unused",
    }
    app = wdtw.create_app(measure)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/measure")
    with pytest.raises(HTTPException) as error:
        endpoint(payload)
    assert error.value.status_code == 500
    assert "F0" in error.value.detail
    assert len(closed) == 1


def test_concurrent_wdtw_requests_isolate_and_close_f0_connections(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import requests

    from dotebench.services.backends import wdtw

    barrier = Barrier(4)
    closed = []

    def post(session, url, *, json, timeout):
        barrier.wait(timeout=10)
        assert url == "http://f0.test/measure"
        stats = {key: 200.0 for key in ("min_f0_hz", "max_f0_hz", "mean_f0_hz")}
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"spans": [dict(stats) for _ in json["spans"]]},
        )

    monkeypatch.setattr(requests.Session, "post", post)
    monkeypatch.setattr(
        requests.Session, "close", lambda session: closed.append(session)
    )
    payload = {
        "duration_source": [{"word": "a", "start": 0, "end": 1}],
        "duration_target": [{"word": "a", "start": 0, "end": 1}],
        "f0_source": [{"word": "a", "start": 0, "end": 1}],
        "f0_target": [{"word": "a", "start": 0, "end": 1}],
        "source_audio": "unused",
        "target_audio": "unused",
    }
    measure = wdtw.build(f0_url="http://f0.test")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(measure, [payload] * 8))
    assert all(result["wdtw_f0"]["wdtw_f0"] == 0 for result in results)
    assert len(closed) == len({id(session) for session in closed}) == 16
