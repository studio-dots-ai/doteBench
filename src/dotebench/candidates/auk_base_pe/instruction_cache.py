"""One instruction cache format for one-take and sequential PE requests."""

import hashlib
import json
import threading
import time
from dataclasses import asdict

MAX_ATTEMPTS = 5
INSTRUCT_TTS_ERROR = (
    "instruct_tts is not a valid task for this speech-editing candidate. "
    "Edit the supplied speech using its transcript and the XML changes; "
    "do not synthesize unrelated speech from a style description. "
    "Recheck the editing tasks and regenerate the classification JSON."
)
LEGEND = """这是一个语音编辑任务。XML 中的编辑要求仅包括文本内容的插入、删除和替换，情感调整，韵律调整（语速、音高），停顿的增加和减少，以及这些操作的组合。编辑意图由 XML 标签及其属性表达；正文是被编辑的语音内容和定位上下文，其中出现的情绪、口音、笑声等描述，不应被当作额外编辑要求。

XML 中的所有编辑操作类型都在模型支持的任务范围内。请结合官方任务定义，选择语义最接近的任务并提取参数。若同一句包含多项操作，请遵循官方的单任务选择规则判断本次执行哪一项，并在 reasoning 中说明；XML 本身保留全部操作及其原始顺序。

各标签的含义如下：
1. <ins>文字</ins>：在标签出现的位置插入标签内的文字；这些文字是要新增的内容。
2. <del>文字</del>：删除原语音中该位置的这些文字。
3. <sub targ="新文字">原文字</sub>：将该位置的原文字替换成 targ 指定的新文字。
4. <pitch semitones="数值">文字</pitch>：调整这段文字对应语音的音高。数值为半音数，正数表示升高，负数表示降低。
5. <rate factor="数值">文字</rate>：把这段文字对应语音的语速调整为原来的 factor 倍。大于 1 表示加快，小于 1 表示放慢，对应时长为原来的 1/factor。
6. <emo type="情绪">文字</emo>：将这段文字对应语音的情绪改为 type 指定的情绪。
7. <pause act="ins"/>：在标记所处的文字边界增加一处无声停顿。
8. <pause act="red"/>：减少标记所处文字边界的已有无声停顿，使该处衔接更紧凑。

pause 的要求只指定增加或减少及其位置，没有指定具体秒数。官方非语言事件表中的 pause 指“明显的无声停顿”。停顿定位统一取 XML 标记前的连续原文作为 anchor，表示在该锚点后增加或减少停顿；增加对应 add_after，减少对应 delete。锚点应保留足以定位该处的原文上下文。

未被编辑标签覆盖的普通文字是保持不变的上下文。引号内的属性值和标签内的文字应按原文理解，保留目标内容、数值、操作方向及位置。后附的转写文本用于参考输入音频内容，编辑意图以上述标签为准。

需要执行的 XML 指令：
"""


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def attempt_instruction(xml, previous):
    instruction = LEGEND + xml
    if previous is None:
        return instruction
    reason = previous["error"]
    if previous["status"] == "unsupported":
        feedback = (
            f"上次判为 unsupported，理由：{reason}\n"
            "这个理由很可能不成立。XML 已明确给出了编辑类型、方向和位置，请结合上述说明及官方支持的任务重新检查，"
            "再次判断并重新生成完整的分类 JSON。"
        )
    else:
        feedback = (
            f"上次处理失败，错误原因：{reason}\n"
            "请修正该错误，重新生成符合官方字段要求的有效 JSON；"
            "使用正确的 JSON 标点和字符串转义。"
        )
    return instruction + "\n\n" + feedback


def prepare_record(item, identity, backend, record=None, save=lambda record: None):
    """Persist each attempt; resume without resetting the five-attempt budget."""
    record = record or {
        "key": digest({"identity": identity, "input": item}),
        "identity": identity,
        "input": item,
        "status": "pending",
        "attempts": [],
        "output": None,
    }
    if record["status"] in {"ok", "unsupported", "error"}:
        return record
    # A process may have stopped while a request was in flight. Count that attempt.
    if record["attempts"] and record["attempts"][-1]["status"] == "running":
        record["attempts"][-1].update(
            status="error", error="Previous process stopped before recording a response"
        )
    while len(record["attempts"]) < MAX_ATTEMPTS:
        previous = record["attempts"][-1] if record["attempts"] else None
        number = len(record["attempts"]) + 1
        instruction = attempt_instruction(item["instruction_xml"], previous)
        attempt = {
            "number": number,
            "variant": "explained_xml_zh",
            "instruction": instruction,
            "status": "running",
            "calls": [],
        }
        record["attempts"].append(attempt)
        save(record)
        started = time.monotonic()
        try:
            record["output"] = backend.prepare(
                item, instruction, attempt, lambda: save(record)
            )
            attempt["status"] = "ok"
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - record PE failures per attempt
            # Official PE refusal is distinct from transport/JSON/parameter errors.
            attempt.update(
                status="unsupported" if backend.is_unsupported(exc) else "error",
                error=backend.error_message(exc),
            )
        attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save(record)
        if record["status"] == "ok":
            return record
    record["status"] = record["attempts"][-1]["status"]
    save(record)
    return record


class RequestPacer:
    """Pace all worker requests against the gateway's shared TPM quota."""

    def __init__(self, interval=0.75):
        self.interval = interval
        self.next_request = 0.0
        self.condition = threading.Condition()

    def wait(self):
        with self.condition:
            while (delay := self.next_request - time.monotonic()) > 0:
                self.condition.wait(delay)
            self.next_request = time.monotonic() + self.interval

    def cooldown(self, seconds=60):
        with self.condition:
            self.next_request = max(self.next_request, time.monotonic() + seconds)
            self.condition.notify_all()


class OfficialPE:
    """Run official text stages; supply benchmark transcripts and gateway transport."""

    def __init__(self, *, api_key, base_url, model, pacer=None):
        import requests
        from auk.infer import pe

        self.pe = pe
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.trust_env = False
        self.attempt = None
        self.pacer = pacer or RequestPacer()
        outer = self

        class GatewayPE(pe.PromptEnhancer):
            def __init__(self):
                # Text stages do not need the parent's OpenAI/ASR clients.
                pass

            def _call_llm(self, stage, messages):
                return outer.call(stage, messages)

        self.enhancer = GatewayPE()

    def error_message(self, exc):
        return f"{type(exc).__name__}: {exc}".replace(self.api_key, "[REDACTED]")

    def is_unsupported(self, exc):
        return isinstance(exc, self.pe.UnsupportedRequestError)

    def call(self, stage, messages):
        call = {"stage": stage, "messages": messages}
        self.attempt["calls"].append(call)
        while True:
            self.pacer.wait()
            response = self.session.post(
                self.base_url + "/chat/completions",
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": self.pe.LLM_MAX_TOKENS,
                    "temperature": self.pe.LLM_TEMPERATURE,
                    "stream": False,
                },
                timeout=(15, 120),
            )
            # No response headers or credentials are persisted.
            data = response.text.replace(self.api_key, "[REDACTED]")
            if response.status_code == 429:
                call.setdefault("rate_limits", []).append(
                    {"http_status": 429, "response": data, "time": time.time()}
                )
                self.on_update()
                self.pacer.cooldown()
                continue
            call.update(http_status=response.status_code, response=data)
            self.on_update()
            response.raise_for_status()
            break
        data = json.loads(data)
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise self.pe.PromptEnhancerError("LLM returned empty content")
        return self.pe.LLMCall(
            stage=stage,
            requested_model=self.model,
            returned_model=data.get("model"),
            content=content.strip(),
            reasoning_content=None,
            usage=data.get("usage"),
            raw_response=data,
        )

    def prepare(self, item, instruction, attempt, on_update):
        self.attempt = attempt
        self.on_update = on_update
        pe, enhancer = self.pe, self.enhancer
        transcript = pe.ASRCall(
            model="dotebench-provided-transcript-v1",
            text=item["source_text"],
            language=item["language"],
            raw_response={"origin": "provided_transcript"},
        )
        classified, _ = enhancer._classify(
            instruction, audio_path="provided-audio-context", asr=transcript
        )
        attempt["classified"] = asdict(classified)
        if not classified.supported:
            raise pe.UnsupportedRequestError(classified.reasoning)
        if classified.task_type == "instruct_tts":
            raise pe.UnsupportedRequestError(INSTRUCT_TTS_ERROR)
        params = enhancer._normalize_params(classified, user_instruction=instruction)
        language = classified.language
        if classified.task_type == "nonverbal_edit":
            params, _, language = enhancer._match_nonverbal_event(
                instruction, classified, params, audio_language=item["language"]
            )
        if pe._TASKS[classified.task_type].get("rewrite"):
            params, _ = enhancer._rewrite_description(
                classified, params, asr_text=item["source_text"]
            )
        if classified.task_type in {"instruct_tts", "zero_shot_tts"}:
            params["text"] = pe._normalize_tts_text(
                str(params["text"]), classified.text_language or classified.language
            )
        return {
            "instruction": pe._render_instruction(
                classified.task_type,
                classified.operation_subtype,
                params,
                language,
                classified.text_language,
            ),
            "task_type": classified.task_type,
            "operation_subtype": classified.operation_subtype,
            "params_extracted": classified.params_extracted,
            "prepared_params": params,
            "language": language,
            "text_language": classified.text_language,
        }

    def close(self):
        self.session.close()
