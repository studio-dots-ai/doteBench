import json
from dataclasses import dataclass

from dotebench.candidates.auk_base_pe.instruction_cache import (
    LEGEND,
    prepare_record,
)

ITEM = {
    "instruction_xml": '<emo type="happy">hello</emo>',
    "source_text": "hello",
    "language": "en",
}


class Unsupported(RuntimeError):
    pass


class Backend:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.inputs = []

    def prepare(self, item, instruction, attempt, on_update):
        self.inputs.append(instruction)
        value = next(self.outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    def is_unsupported(self, exc):
        return isinstance(exc, Unsupported)

    def error_message(self, exc):
        return str(exc)


def test_format_and_refusal_feedback_preserves_explanation_until_success():
    backend = Backend(
        [
            json.JSONDecodeError("Expecting ',' delimiter", "{}", 1),
            Unsupported("pause is not supported"),
            {"instruction": "add a pause"},
        ]
    )
    record = prepare_record(ITEM, {}, backend)
    assert record["status"] == "ok"
    assert [a["variant"] for a in record["attempts"]] == [
        "explained_xml_zh",
        "explained_xml_zh",
        "explained_xml_zh",
    ]
    assert backend.inputs[0] == LEGEND + ITEM["instruction_xml"]
    assert backend.inputs[1].startswith(LEGEND)
    assert "Expecting ',' delimiter" in backend.inputs[1]
    assert backend.inputs[2].startswith(LEGEND + ITEM["instruction_xml"])
    assert "pause is not supported" in backend.inputs[2]
    assert "理由很可能不成立" in backend.inputs[2]
    assert len(backend.inputs) == 3


def test_exhausted_refusals_stay_terminal_on_resume():
    backend = Backend([Unsupported("reason")] * 5)
    record = prepare_record(ITEM, {}, backend)
    assert record["status"] == "unsupported"
    assert len(record["attempts"]) == 5
    assert prepare_record(ITEM, {}, Backend([]), record) == record


def test_exhausted_format_errors_are_not_semantic_refusals():
    record = prepare_record(ITEM, {}, Backend([ValueError("invalid JSON")] * 5))
    assert record["status"] == "error"
    assert len(record["attempts"]) == 5


def test_inflight_attempt_counts_toward_budget_after_restart():
    record = {
        "status": "pending",
        "attempts": [{"status": "running"}] * 4,
        "output": None,
    }
    backend = Backend([Unsupported("still unsupported")])
    result = prepare_record(ITEM, {}, backend, record)
    assert len(backend.inputs) == 1
    assert len(result["attempts"]) == 5
    assert result["status"] == "unsupported"


def test_plan_preserves_one_take_and_uses_sequential_source_text(monkeypatch):
    from dotebench.candidates.auk_base_pe import build_cache

    @dataclass
    class Case:
        id: str
        language: str
        instruction_xml: str

    xml = '<sub targ="new">old</sub> <rate factor="1.2">word</rate>'
    monkeypatch.setattr(build_cache, "select_shards", lambda: [{}])
    monkeypatch.setattr(
        build_cache, "load_cases", lambda *a, **k: [Case("id", "en", xml)]
    )
    inputs, refs = build_cache.plan_requests()
    assert [r["compiler"] for r in refs] == ["one_take", "sequential", "sequential"]
    assert inputs[refs[0]["input_key"]]["instruction_xml"] == xml
    assert inputs[refs[1]["input_key"]]["source_text"] == "old word"
    assert inputs[refs[2]["input_key"]]["source_text"] == "new word"
    assert all(
        set(item) == {"instruction_xml", "source_text", "language"}
        for item in inputs.values()
    )


def test_gateway_rate_limit_waits_without_another_pe_attempt():
    from types import SimpleNamespace

    from dotebench.candidates.auk_base_pe.instruction_cache import OfficialPE

    class Response:
        def __init__(self, status, value):
            self.status_code = status
            self.text = json.dumps(value)

        def raise_for_status(self):
            assert self.status_code == 200

    responses = iter(
        [
            Response(429, {"detail": "TPM limit"}),
            Response(
                200, {"model": "test", "choices": [{"message": {"content": "{}"}}]}
            ),
        ]
    )
    waits, cooldowns, updates = [], [], []
    backend = OfficialPE.__new__(OfficialPE)
    backend.api_key = "test-secret"
    backend.base_url = "https://example.test/v1"
    backend.model = "test"
    backend.pe = SimpleNamespace(
        LLM_MAX_TOKENS=4096, LLM_TEMPERATURE=0, LLMCall=lambda **kw: kw
    )
    backend.session = SimpleNamespace(post=lambda *a, **kw: next(responses))
    backend.pacer = SimpleNamespace(
        wait=lambda: waits.append(1), cooldown=lambda: cooldowns.append(1)
    )
    backend.attempt = {"calls": []}
    backend.on_update = lambda: updates.append(1)
    result = backend.call("classify", [{"role": "user", "content": "XML"}])
    assert result["content"] == "{}"
    assert len(backend.attempt["calls"]) == 1
    assert len(backend.attempt["calls"][0]["rate_limits"]) == 1
    assert len(waits) == 2
    assert len(cooldowns) == 1
    assert len(updates) == 2
    assert "test-secret" not in json.dumps(backend.attempt)
