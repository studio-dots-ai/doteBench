"""XML-local recognition scoring and evaluation count aggregation."""

from dataclasses import asdict, dataclass

import jiwer

from dotebench.services.backends.wer import WERResult, mixed_language_tokens


@dataclass
class AggregateWERResult:
    """Sequence-level aggregate over multiple WER/CER result dictionaries."""

    error_rate: float | None
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int
    hyp_length: int
    metric: str
    count: int

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorRateAccumulator:
    """Incrementally aggregate WER/CER counts using total reference length."""

    def __init__(self) -> None:
        self.count = 0
        self.substitutions = 0
        self.deletions = 0
        self.insertions = 0
        self.hits = 0
        self.ref_length = 0
        self.hyp_length = 0
        self.metrics: set[str] = set()

    def add(self, result: object) -> None:
        self.count += 1
        required = (
            "substitutions",
            "deletions",
            "insertions",
            "hits",
            "ref_length",
            "hyp_length",
        )
        try:
            data = _result_to_dict(result)
            values = {key: int(data[key]) for key in required}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing or invalid ER sufficient statistics") from exc
        if (
            any(value < 0 for value in values.values())
            or values["hits"] + values["substitutions"] + values["deletions"]
            != values["ref_length"]
        ):
            raise ValueError("Inconsistent ER sufficient statistics")
        self.substitutions += values["substitutions"]
        self.deletions += values["deletions"]
        self.insertions += values["insertions"]
        self.hits += values["hits"]
        self.ref_length += values["ref_length"]
        self.hyp_length += values["hyp_length"]
        metric = data.get("metric")
        if isinstance(metric, str) and metric:
            self.metrics.add(metric)

    def to_result(self) -> AggregateWERResult:
        errors = self.substitutions + self.deletions + self.insertions
        if self.count == 0:
            error_rate = None
        elif self.ref_length > 0:
            error_rate: float | None = errors / self.ref_length
        else:
            error_rate = None
        if not self.metrics:
            metric = "unknown"
        elif len(self.metrics) == 1:
            metric = next(iter(self.metrics))
        else:
            metric = "mixed"
        return AggregateWERResult(
            error_rate=error_rate,
            substitutions=self.substitutions,
            deletions=self.deletions,
            insertions=self.insertions,
            hits=self.hits,
            ref_length=self.ref_length,
            hyp_length=self.hyp_length,
            metric=metric,
            count=self.count,
        )


def _result_to_dict(result: object) -> dict:
    if isinstance(result, WERResult):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported WER result type: {type(result).__name__}")


def aggregate_error_rate_results(results: list[object]) -> AggregateWERResult:
    """Aggregate WER/CER results as if references were concatenated."""
    accumulator = ErrorRateAccumulator()
    for result in results:
        accumulator.add(result)
    return accumulator.to_result()


@dataclass
class LocalWERResult:
    """Result of a local WER/CER computation around edit regions."""

    edit_region_wer: WERResult
    non_edit_region_wer: WERResult
    edit_word_indices: list[int]
    local_window_indices: list[int]
    context_words: int
    local_ref_text: str
    local_hyp_text: str

    def to_dict(self) -> dict:
        return {
            "edit_region_wer": self.edit_region_wer.to_dict(),
            "non_edit_region_wer": self.non_edit_region_wer.to_dict(),
            "edit_word_indices": self.edit_word_indices,
            "local_window_indices": self.local_window_indices,
            "context_words": self.context_words,
            "local_ref_text": self.local_ref_text,
            "local_hyp_text": self.local_hyp_text,
        }


def _compute_region_wer_from_alignment(
    output, ref_words: list[str], hyp_words: list[str], region_ref_indices: set
) -> WERResult:
    """Count WER errors only within a set of reference word indices.

    Uses jiwer alignment chunks to attribute errors to specific ref positions.
    """
    local_s = local_d = local_i = local_h = 0
    local_ref_positions = []
    local_hyp_positions = []
    for chunk in output.alignments[0]:
        chunk_type = chunk.type
        ref_start = chunk.ref_start_idx
        ref_end = chunk.ref_end_idx
        hyp_start = chunk.hyp_start_idx
        hyp_end = chunk.hyp_end_idx
        ref_range = set(range(ref_start, ref_end))
        overlap = ref_range & region_ref_indices
        if not overlap and chunk_type != "insert":
            continue
        if chunk_type == "equal":
            n = len(overlap)
            local_h += n
            local_ref_positions.extend(sorted(overlap))
            for ri in sorted(overlap):
                offset = ri - ref_start
                if hyp_start + offset < hyp_end:
                    local_hyp_positions.append(hyp_start + offset)
        elif chunk_type == "substitute":
            n = len(overlap)
            local_s += n
            local_ref_positions.extend(sorted(overlap))
            for ri in sorted(overlap):
                offset = ri - ref_start
                if hyp_start + offset < hyp_end:
                    local_hyp_positions.append(hyp_start + offset)
        elif chunk_type == "delete":
            n = len(overlap)
            local_d += n
            local_ref_positions.extend(sorted(overlap))
        elif chunk_type == "insert":
            # Each insertion belongs to one reference boundary, so the edit
            # and non-edit partitions cannot count the same error twice.
            anchor = min(ref_start, len(ref_words) - 1)
            if anchor in region_ref_indices:
                local_i += hyp_end - hyp_start
                for hi in range(hyp_start, hyp_end):
                    local_hyp_positions.append(hi)
    local_ref_len = local_h + local_s + local_d
    error_rate = (
        (local_s + local_d + local_i) / local_ref_len if local_ref_len > 0 else 0.0
    )
    local_ref_text = " ".join(
        ref_words[i] for i in sorted(set(local_ref_positions)) if i < len(ref_words)
    )
    local_hyp_text = " ".join(
        hyp_words[i] for i in sorted(set(local_hyp_positions)) if i < len(hyp_words)
    )
    return WERResult(
        error_rate=error_rate,
        substitutions=local_s,
        deletions=local_d,
        insertions=local_i,
        hits=local_h,
        ref_length=local_ref_len,
        hyp_length=len(set(local_hyp_positions)),
        reference_normalized=local_ref_text,
        hypothesis_normalized=local_hyp_text,
        metric="wer",
    )


def compute_instruction_wer(
    instruction,
    hypothesis: str,
    context_k: int = 3,
    *,
    operation_index: int | None = None,
) -> LocalWERResult:
    """Use XML-owned text edits, including deletion boundaries, for local WER."""
    ref_tokens = mixed_language_tokens(instruction.target_text)
    hyp_tokens = mixed_language_tokens(hypothesis)
    edited, window = set(), set()
    for index, operation in enumerate(instruction.operations):
        if operation_index is not None and index != operation_index:
            continue
        if operation.kind not in {"ins", "del", "sub"}:
            continue
        start = len(
            mixed_language_tokens(instruction.target_text[: operation.target.start])
        )
        end = len(
            mixed_language_tokens(instruction.target_text[: operation.target.end])
        )
        edited.update(range(start, end))
        window.update(
            range(max(0, start - context_k), min(len(ref_tokens), end + context_k))
        )
    output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    edit = _compute_region_wer_from_alignment(output, ref_tokens, hyp_tokens, window)
    other = _compute_region_wer_from_alignment(
        output, ref_tokens, hyp_tokens, set(range(len(ref_tokens))) - window
    )
    edit = WERResult(**{**asdict(edit), "metric": "mixed"})
    other = WERResult(**{**asdict(other), "metric": "mixed"})
    return LocalWERResult(
        edit,
        other,
        sorted(edited),
        sorted(window),
        context_k,
        " ".join(ref_tokens[i] for i in sorted(window)),
        edit.hypothesis_normalized,
    )
