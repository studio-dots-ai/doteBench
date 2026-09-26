"""A single fixed alignment supplies all positional queries and consumers."""

import pytest

from dotebench.alignment import query_alignment
from dotebench.text import token_spans


def alignment(text):
    return [
        {"word": text[a:b], "start": i / 10, "end": (i + 1) / 10}
        for i, (a, b) in enumerate(token_spans(text))
    ]


@pytest.mark.parametrize(
    "text,query,expected",
    [
        ("Annie's eyes", "<q>Annie's</q> eyes", (0, 0.2)),
        ("follow—the", "follow—<q>the</q>", (0.1, 0.2)),
        ("你好world", "你<q>好world</q>", (0.1, 0.3)),
        ("it it it", "it <q>it</q> it", (0.1, 0.2)),
        ("a & b", "a & <q>b</q>", (0.1, 0.2)),
        ("a b", "<q></q>a b", (0, 0)),
        ("a b", "a b<q></q>", (2, 2)),
    ],
)
def test_fixed_queries(text, query, expected):
    result = query_alignment(text, alignment(text), [query], 2)[0]
    assert (result["start"], result["end"]) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "reordered",
        "wrong_word",
        "split",
        "nan",
        "negative",
        "outside",
    ],
)
def test_alignment_contract(mutation):
    units = alignment("one two")
    if mutation == "missing":
        units.pop()
    elif mutation == "extra":
        units.append(units[-1])
    elif mutation == "reordered":
        units.reverse()
    elif mutation == "wrong_word":
        units[0]["word"] = "three"
    elif mutation == "split":
        units[0]["word"] = "o-ne"
    elif mutation == "nan":
        units[0]["end"] = float("nan")
    elif mutation == "negative":
        units[0]["start"] = -1
    else:
        units[-1]["end"] = 3
    with pytest.raises(ValueError):
        query_alignment("one two", units, ["<q>one</q> two"], 2)


@pytest.mark.parametrize(
    "query", ["o<q>ne</q> two", "<q>other</q> two", "one two", "<q>one <q>two</q></q>"]
)
def test_invalid_queries(query):
    with pytest.raises(ValueError):
        query_alignment("one two", alignment("one two"), [query], 2)


def test_zero_duration_is_preserved():
    units = [{"word": "one", "start": 0, "end": 0}]
    assert query_alignment("one", units, ["<q>one</q>"], 1) == [{"start": 0, "end": 0}]


def test_query_service_never_invokes_model():
    from dotebench.services.backends.qwen3_aligner import create_app

    def forbidden(payload):
        raise AssertionError("Query extraction must not invoke the model")

    app = create_app(forbidden)
    route = next(r for r in app.routes if r.path == "/measure")
    result = route.endpoint(
        {
            "text": "one",
            "alignment": alignment("one"),
            "queries": ["<q>one</q>"],
            "audio_duration": 1,
        }
    )
    assert result == {"answers": [{"start": 0, "end": 0.1}]}


def test_many_operations_share_target_alignment():
    from dotebench.evaluation.evaluator import DoteBenchEvaluator
    from tests.evaluation.test_protocol import Abilities, case, request

    c = case(
        '<pitch semitones="2">one</pitch> <rate factor="2">two</rate> <pause act="ins"/><emo type="happy" level="1">three</emo>'
    )
    abilities = Abilities(c.target_text)
    result = DoteBenchEvaluator(abilities).evaluate(request(c))
    calls = [p for name, p in abilities.calls if name == "qwen3_aligner"]
    assert len(calls) == 1 and "queries" not in calls[0]
    sent = next(p for name, p in abilities.calls if name == "wdtw")
    target = result.metrics["alignment"]["target"]
    assert sent["duration_target"] == [target[0]]
    assert sent["f0_target"] == [target[1]]
    assert result.metrics["components"][0]["measurement"]["target_span"] == {
        "start": 0,
        "end": 0.2,
    }
    assert result.metrics["components"][1]["measurement"]["target_span"] == {
        "start": 0.25,
        "end": 0.45,
    }
    assert result.metrics["components"][2]["measurement"][
        "target_gap_sec"
    ] == pytest.approx(0.05)
    emotion = next(p for name, p in abilities.calls if name == "emotion")
    assert (emotion["start"], emotion["end"]) == (0.5, 0.7)


def test_zero_target_rate_keeps_numeric_success_rule():
    from dotebench.evaluation.evaluator import DoteBenchEvaluator
    from tests.evaluation.test_protocol import Abilities, case, request

    class Zero(Abilities):
        def measure(self, name, **payload):
            result = super().measure(name, **payload)
            if name == "qwen3_aligner":
                result["segments"][1]["end"] = result["segments"][1]["start"]
            return result

    c = case('one <rate factor="2">two</rate> three')
    result = DoteBenchEvaluator(Zero(c.target_text)).evaluate(request(c))
    row = result.metrics["components"][0]
    assert row["measurement"]["target_span"] == {"start": 0.25, "end": 0.25}
    assert row["success"] is True


def test_query_service_live_http(tmp_path):
    import json
    import socket
    import sys
    from pathlib import Path
    from urllib.request import Request, urlopen

    from dotebench.services import resolve_service_graph, start_services

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    code = (
        "import uvicorn; from dotebench.services.backends.qwen3_aligner import create_app; "
        "app = create_app(lambda p: (_ for _ in ()).throw(AssertionError('model invoked'))); "
        f"uvicorn.run(app, host='127.0.0.1', port={port})"
    )
    root = Path(__file__).parents[2]
    node = {
        "instance_key": "aligner",
        "config_key": "aligner",
        "url": url,
        "startup_timeout_sec": 30,
        "command": [sys.executable, "-c", code],
    }
    with start_services(
        graph=resolve_service_graph({"aligner": node}),
        project_root=root,
        log_dir=tmp_path / "logs",
        base_env={"PYTHONPATH": str(root / "src")},
    ):
        with urlopen(url + "/health") as response:
            assert json.load(response) == {
                "status": "ready",
                "ability": "qwen3_aligner",
                "model_identity": None,
                "configuration": {},
            }
        payload = {
            "text": "it it",
            "alignment": alignment("it it"),
            "queries": ["it <q>it</q>"],
            "audio_duration": 1,
        }
        req = Request(
            url + "/measure",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req) as response:
            assert json.load(response) == {"answers": [{"start": 0.1, "end": 0.2}]}
