"""Protocol contract, preservation boundaries, and failure-integrity tests."""

import io

import numpy as np
import pytest
import soundfile as sf

from dotebench.alignment import WordSegment
from dotebench.domain import Annotations, AudioAsset, Case
from dotebench.evaluation.base import EvaluationRequest, EvaluationResult
from dotebench.evaluation.evaluator import DoteBenchEvaluator
from dotebench.services.backends.wdtw import (
    compute_duration_preservation,
    compute_f0_preservation,
)
from dotebench.services.backends.wer import compute_error_rate
from dotebench.instructions import parse_instruction
from dotebench.models.base import Audio, GenerationResult


def case(xml, id="case"):
    inst = parse_instruction(xml)
    return Case(
        id,
        "en",
        AudioAsset("source.wav", "a" * 64),
        xml,
        source_annotations(inst),
        inst,
    )


def source_annotations(inst):
    import hashlib

    from dotebench.domain import AlignmentUnit, SourceAlignment

    abilities = Abilities(inst.source_text)
    raw = audio().data
    response = abilities.measure("qwen3_aligner", text=inst.source_text)
    return Annotations(
        source_alignment=SourceAlignment(
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(inst.source_text.encode()).hexdigest(),
            tuple(AlignmentUnit(**r) for r in response["segments"]),
        ),
    )


def audio():
    output = io.BytesIO()
    sf.write(
        output,
        np.ones(32000, dtype=np.float32) * 0.1,
        16000,
        format="WAV",
        subtype="PCM_16",
    )
    return Audio(output.getvalue())


class Abilities:
    def identity(self):
        return {"fixture": "metrics"}

    def __init__(self, hypothesis, *, fail=None):
        self.hypothesis, self.fail, self.calls = hypothesis, fail, []

    def prepare(self):
        pass

    def close(self):
        pass

    def measure(self, ability, **payload):
        self.calls.append((ability, payload))
        if ability == self.fail:
            raise RuntimeError("backend unavailable")
        if ability == "qwen3_asr":
            return {"text": self.hypothesis}
        if ability == "wer":
            return compute_error_rate(**payload).to_dict()
        if ability == "qwen3_aligner":
            if "queries" in payload:
                answers = []
                for q in payload["queries"]:
                    before, rest = q.split("<q>")
                    selected = rest.split("</q>")[0]
                    start = len(before.split()) * 0.25
                    end = start + len(selected.split()) * 0.25 - 0.05
                    answers.append({"start": start, "end": end})
                return {"answers": answers}
            return {
                "segments": [
                    {"word": w, "start": i * 0.25, "end": i * 0.25 + 0.2}
                    for i, w in enumerate(payload["text"].split())
                ]
            }
        if ability == "wdtw":
            from dotebench.services.backends.wdtw import measure_spans

            return measure_spans(payload, lambda p: self.measure("f0", **p))
        if ability == "f0":
            return {
                "spans": [
                    {
                        "min_f0_hz": 100.0,
                        "max_f0_hz": 100.0,
                        "mean_f0_hz": 100.0,
                        "median_f0_hz": 100.0,
                    }
                    for s in payload["spans"]
                ]
            }
        if ability == "utmos":
            return {"score": 4.0}
        if ability == "speaker_similarity":
            return {"similarity": 1.0}
        if ability == "emotion":
            return {"predicted_emotion": "happy"}
        raise AssertionError(ability)


def request(c, status="ok"):
    a = audio()
    return EvaluationRequest(
        c,
        GenerationResult(c.id, status, error="failed" if status != "ok" else None),
        a,
        a if status == "ok" else None,
    )


def test_identity_measures_requested_effect_and_preservation_separately():
    c = case('one <pitch semitones="2">two</pitch> three')
    abilities = Abilities(c.target_text)
    protocol = DoteBenchEvaluator(abilities)
    result = protocol.evaluate(request(c))
    assert result.metrics["components"][0]["measurement"]["pitch_l1_semitones"] == 2
    assert result.metrics["components"][0]["success"] is False
    assert result.metrics["wdtw_dur"]["wdtw_dur"] == 0
    assert result.metrics["wdtw_f0"]["wdtw_f0"] == 0
    assert result.metrics["wdtw_f0"]["normalizer"] == 2
    sent = next(p for a, p in abilities.calls if a == "wdtw")
    assert set(sent) == {
        "source_audio",
        "target_audio",
        "duration_source",
        "duration_target",
        "f0_source",
        "f0_target",
    }
    assert [s["word"] for s in sent["f0_source"]] == ["one", "three"]
    assert (
        len(result.metrics["alignment"]["selection"]["duration_source"]) == 3
    )  # Pitch edits preserve word durations.


def test_candidate_failure_only_measures_source_f0():
    c = case('one <sub targ="four">two</sub> three')
    abilities = Abilities(c.target_text, fail="qwen3_asr")
    protocol = DoteBenchEvaluator(abilities)
    result = protocol.evaluate(request(c, "candidate_error"))
    assert [name for name, _ in abilities.calls] == ["f0"]
    assert len(abilities.calls[0][1]["spans"]) == 2
    assert result.metrics["wer"]["error_rate"] == 1
    assert result.metrics["wdtw_dur"]["wdtw_dur"] == 1
    assert result.metrics["wdtw_f0"]["wdtw_f0"] == 8
    assert result.metrics["utmos"] == 1
    assert result.metrics["speaker_similarity"] == 0
    assert protocol.aggregate([result])["penalized"] == 1


def test_failed_whole_rate_uses_source_words_for_f0_penalty():
    c = case('<rate factor="2">one two</rate>')
    abilities = Abilities(c.target_text)
    result = DoteBenchEvaluator(abilities).evaluate(request(c, "candidate_error"))
    assert [name for name, _ in abilities.calls] == ["f0"]
    assert len(abilities.calls[0][1]["spans"]) == 2
    assert result.metrics["wdtw_dur"]["normalizer"] == 0
    assert result.metrics["wdtw_f0"]["wdtw_f0"] == 8
    assert result.metrics["wdtw_f0"]["normalizer"] == 2


@pytest.mark.parametrize(
    "ability",
    ["qwen3_asr", "wer", "qwen3_aligner", "wdtw", "f0", "utmos", "speaker_similarity"],
)
def test_infrastructure_failure_never_becomes_candidate_penalty(ability):
    c = case('one <sub targ="four">two</sub> three')
    with pytest.raises(RuntimeError, match="backend unavailable"):
        DoteBenchEvaluator(Abilities(c.target_text, fail=ability)).evaluate(request(c))


def test_emotion_metric_failure_propagates():
    c = case('one <emo type="happy" level="1">two</emo> three')
    with pytest.raises(RuntimeError):
        DoteBenchEvaluator(Abilities(c.target_text, fail="emotion")).evaluate(
            request(c)
        )


def test_nonfinite_metric_is_rejected():
    class Invalid(Abilities):
        def measure(self, ability, **payload):
            if ability == "utmos":
                return {"score": float("nan")}
            return super().measure(ability, **payload)

    c = case('one <sub targ="four">two</sub> three')
    with pytest.raises(ValueError, match="non-finite"):
        DoteBenchEvaluator(Invalid(c.target_text)).evaluate(request(c))


def test_duplicate_or_incomplete_results_cannot_be_official():
    protocol = DoteBenchEvaluator(Abilities(""))
    bad = EvaluationResult("a", {"evaluation_status": "error"})
    with pytest.raises(ValueError, match="Incomplete"):
        protocol.aggregate([bad])
    with pytest.raises(ValueError, match="duplicate"):
        protocol.aggregate([bad, bad])


def test_acoustic_kernels_frozen_numeric_examples():
    source = [WordSegment("a", 0, 1), WordSegment("b", 1, 2)]
    target = [WordSegment("a", 0, 0.5), WordSegment("b", 0.5, 2)]
    assert compute_duration_preservation(source, target)["wdtw_dur"] == 0.25
    src = [{"min_f0_hz": 100, "max_f0_hz": 200, "mean_f0_hz": 150}] * 2
    tgt = [
        {"min_f0_hz": 200, "max_f0_hz": 400, "mean_f0_hz": 300},
        {"min_f0_hz": None, "max_f0_hz": None, "mean_f0_hz": None},
    ]
    result = compute_f0_preservation(
        source_segments=source,
        target_segments=target,
        source_f0_spans=src,
        target_f0_spans=tgt,
    )
    assert result.wdtw_f0 == 12
    assert result.normalizer == 1
    assert "skipped_word_count" not in result.to_dict()


def test_whole_sentence_emotion_uses_whole_audio():
    c = case('<emo type="happy" level="1">one two three</emo>')
    abilities = Abilities(c.target_text)
    DoteBenchEvaluator(abilities).evaluate(request(c))
    emotion = next(p for a, p in abilities.calls if a == "emotion")
    assert emotion["input_strategy"] == "whole_audio"
    assert emotion["start"] == 0 and emotion["end"] == 2


def test_query_backend_forwards_normalized_arrays_and_timestamp_scale(monkeypatch):
    import contextlib
    import sys
    from types import ModuleType, SimpleNamespace

    from dotebench.alignment import query_alignment
    from dotebench.services.backends.qwen3_aligner import align_units

    observed = {}
    normalized = [np.ones(16000, dtype=np.float32)]
    utils = ModuleType("qwen_asr.inference.utils")

    def normalize(value):
        observed["normalizer_input"] = value
        return normalized

    utils.normalize_audios = normalize
    monkeypatch.setitem(sys.modules, "qwen_asr", ModuleType("qwen_asr"))
    monkeypatch.setitem(
        sys.modules, "qwen_asr.inference", ModuleType("qwen_asr.inference")
    )
    monkeypatch.setitem(sys.modules, "qwen_asr.inference.utils", utils)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=contextlib.nullcontext)
    )

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def argmax(self, dim):
            return Tensor(self.array.argmax(axis=dim))

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def __eq__(self, value):
            return self.array == value

        def __mul__(self, value):
            return Tensor(self.array * value)

        def to(self, *args):
            return self

        def numpy(self):
            return self.array

    class Inputs(dict):
        def to(self, *args):
            return self

    def processor(**kwargs):
        observed["processor"] = kwargs
        return Inputs(input_ids=Tensor([[99, 7, 8, 7]]))

    logits = np.zeros((1, 4, 5))
    logits[0, 1, 2] = 1
    logits[0, 3, 4] = 1

    def parse(units, timestamps):
        observed["units"] = units
        observed["timestamps"] = timestamps.tolist()
        return [
            {"text": "alpha", "start_time": timestamps[0], "end_time": timestamps[1]}
        ]

    aligner = SimpleNamespace(
        processor=processor,
        timestamp_token_id=7,
        timestamp_segment_time=100,
        model=SimpleNamespace(
            device="cpu",
            dtype="float32",
            thinker=lambda **kw: SimpleNamespace(logits=Tensor(logits)),
        ),
        aligner_processor=SimpleNamespace(
            encode_timestamp=lambda text, lang: (text.split(), ""),
            parse_timestamp=parse,
        ),
    )
    original = (np.zeros(8000, dtype=np.float32), 8000)
    result = {"segments": align_units(aligner, original, ["alpha"])}
    result["answers"] = query_alignment(
        "alpha", result["segments"], ["<q>alpha</q>"], 1.0
    )
    assert observed["normalizer_input"][0] is original
    assert observed["processor"]["audio"] is normalized
    assert observed["processor"]["text"] == [
        "<|audio_start|><|audio_pad|><|audio_end|>alpha<timestamp><timestamp>"
    ]
    assert observed["units"] == ["alpha"]
    assert observed["timestamps"] == [200, 400]
    assert result["segments"] == [{"word": "alpha", "start": 0.2, "end": 0.4}]
    assert result["answers"][0]["start"] == 0.2 and result["answers"][0]["end"] == 0.4


def test_emotion_prompt_hash_and_candidate_order_match_frozen_reference():
    import hashlib

    from dotebench.services.backends.emotion import (
        _build_emotion_prompt,
        _emotion_candidate_order,
        _parse_emotion_label,
    )

    candidates = _emotion_candidate_order(
        case_id="emo_c312_clean_v1_001_whole", span_index=0
    )
    assert candidates == (
        "afraid",
        "sad",
        "melancholic",
        "disgusted",
        "happy",
        "surprised",
        "angry",
        "calm",
    )
    prompt = _build_emotion_prompt(candidates=candidates, input_strategy="whole_audio")
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == "6e5350f776987f563eedc68d0b5a5261341f6706c08e2bf265c1bce65b0c4975"
    )
    assert _parse_emotion_label("The emotion is happy. Happy.") == "happy"
    assert _parse_emotion_label("happy or sad") is None
    assert _parse_emotion_label("happiness") is None
    assert _parse_emotion_label("neutral") == "calm"
    assert _parse_emotion_label("NEUTRAL") == "calm"
    assert _parse_emotion_label("neutral and calm") == "calm"
    assert _parse_emotion_label("neutral or sad") is None
    assert _parse_emotion_label("neutrality") is None


@pytest.mark.parametrize("ability", ["qwen3_asr", "qwen3_aligner", "emotion"])
def test_neural_loader_uses_fixed_defaults(monkeypatch, ability, tmp_path):
    import sys
    from types import SimpleNamespace

    from dotebench.services.backends import emotion, qwen3_aligner, qwen3_asr

    observed = {}

    def loader(*args, **kwargs):
        observed["args"], observed["kwargs"] = args, kwargs
        return SimpleNamespace()

    def sampling(**kwargs):
        observed["sampling"] = kwargs
        return SimpleNamespace()

    fake_torch = SimpleNamespace(
        bfloat16="torch.bfloat16",
        float16="torch.float16",
        cuda=SimpleNamespace(device_count=lambda: 2),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        SimpleNamespace(
            Qwen3ASRModel=SimpleNamespace(from_pretrained=loader),
            Qwen3ForcedAligner=SimpleNamespace(from_pretrained=loader),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=loader, SamplingParams=sampling)
    )
    backend = {
        "qwen3_asr": qwen3_asr,
        "qwen3_aligner": qwen3_aligner,
        "emotion": emotion,
    }[ability]
    measure = backend.build(tmp_path, "cuda")
    assert callable(measure)
    if ability in {"qwen3_asr", "qwen3_aligner"}:
        expected = {"dtype": "torch.bfloat16", "device_map": "cuda"}
        if ability == "qwen3_asr":
            expected.update(max_inference_batch_size=1, max_new_tokens=512)
        assert observed["args"] == (str(tmp_path),)
        assert observed["kwargs"] == expected
    else:
        assert observed["kwargs"] == {
            "model": str(tmp_path),
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 16384,
            "trust_remote_code": True,
        }
        assert observed["sampling"] == {"temperature": 0, "max_tokens": 4096}
    configuration = measure.configuration
    assert configuration["model_options"] == {
        key: value for key, value in observed["kwargs"].items() if key != "model"
    }


def test_asr_transformers_preserves_pcm_values_and_handles_exact_silence(
    monkeypatch, tmp_path
):
    import base64
    import io
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np
    import soundfile as sf

    from dotebench.services.backends import qwen3_asr

    calls = []

    def transcribe(**request):
        calls.append(request)
        return [SimpleNamespace(text="hello")]

    def load(path, **options):
        assert path == str(tmp_path)
        assert options == {
            "dtype": "bfloat16",
            "device_map": "cuda:0",
            "max_inference_batch_size": 1,
            "max_new_tokens": 512,
        }
        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="bfloat16", inference_mode=nullcontext),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        SimpleNamespace(Qwen3ASRModel=SimpleNamespace(from_pretrained=load)),
    )
    measure = qwen3_asr.build(tmp_path, "cuda:0")

    def encode(samples):
        buffer = io.BytesIO()
        sf.write(buffer, samples, 24000, format="WAV", subtype="FLOAT")
        return base64.b64encode(buffer.getvalue()).decode()

    samples = np.array([0.00001, -0.00001, 0.123456], dtype=np.float32)
    assert measure({"audio": encode(samples), "language": "en"}) == {"text": "hello"}
    np.testing.assert_array_equal(calls[0]["audio"][0], samples)
    assert calls[0]["audio"][1] == 24000
    assert calls[0]["language"] == "English"
    assert measure({"audio": encode(np.zeros(24000)), "language": None}) == {"text": ""}
    assert len(calls) == 1


def test_neural_loader_rejects_parameter_overrides(tmp_path):
    from dotebench.services.backends import qwen3_asr

    with pytest.raises(TypeError, match="options"):
        qwen3_asr.build(tmp_path, "cuda", options={})


def test_metric_service_binds_default_model(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from dotebench.services.backends import qwen3_asr as backend

    observed = {}

    def build(path, device):
        observed.update(path=path, device=device)
        return lambda payload: payload

    monkeypatch.setattr(backend, "build", build)
    monkeypatch.setattr(backend, "prepared_identity", lambda *args: {"verified": True})
    monkeypatch.setattr(backend, "create_app", lambda measure, **kwargs: "app")
    monkeypatch.setitem(
        sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **kwargs: None)
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "metric_service",
            "--model-path",
            str(tmp_path),
            "--port",
            "9999",
        ],
    )
    backend.main()
    assert observed == {
        "path": tmp_path,
        "device": "cuda",
    }


def test_source_alignment_is_consumed_without_source_requests():
    c = case('one <rate factor="2">two</rate> three')
    abilities = Abilities(c.target_text)
    result = DoteBenchEvaluator(abilities).evaluate(request(c))
    alignments = [p for a, p in abilities.calls if a == "qwen3_aligner"]
    # One complete target alignment serves every consumer.
    assert len(alignments) == 1
    assert sum("queries" not in p for p in alignments) == 1
    assert result.metrics["components"][0]["measurement"]["source_span"] == {
        "start": 0.25,
        "end": 0.45,
    }


def test_bad_generated_content_still_uses_frozen_target_text():
    c = case('one <sub targ="four">two</sub> three')
    abilities = Abilities("completely unrelated speech")
    result = DoteBenchEvaluator(abilities).evaluate(request(c))
    assert result.metrics["evaluation_status"] == "ok"
    assert result.metrics["penalty_applied"] is False
    assert all(
        p["text"] == c.target_text for a, p in abilities.calls if a == "qwen3_aligner"
    )
    assert result.metrics["wer"]["error_rate"] > 0


def test_missing_source_annotation_never_falls_back_to_alignment():
    from dataclasses import replace

    c = case('one <rate factor="2">two</rate> three')
    c = replace(c, annotations=replace(c.annotations, source_alignment=None))
    abilities = Abilities(c.target_text)
    with pytest.raises(ValueError, match="frozen source alignment"):
        DoteBenchEvaluator(abilities).evaluate(request(c))
    assert not any(a == "qwen3_aligner" for a, p in abilities.calls)
