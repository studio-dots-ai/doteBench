"""Compile AuK Base whole-utterance and local editing requests."""

import hashlib
import io
import math

import soundfile as sf

from dotebench.alignment import query_intervals, validate_alignment
from dotebench.instructions import parse_instruction, render_replacement
from dotebench.models.requests import NativeRequest

from .alignment import validate_candidate_alignment

ZH_EMOTIONS = dict(
    afraid="害怕",
    angry="生气",
    happy="开心",
    melancholic="忧郁",
    sad="悲伤",
    surprised="惊讶",
    calm="平静",
    disgusted="厌恶",
    excited="兴奋",
)


def compile_request(request, context_mode="clean", *, alignment=None):
    parsed = parse_instruction(request.instruction_xml)
    if context_mode not in {"clean", "bridge"}:
        raise ValueError("Expected clean or bridge context mode")
    text = parsed.source_text
    index, op = min(
        enumerate(parsed.operations),
        key=lambda pair: (pair[1].source.start, pair[0]),
    )
    outside = text[: op.source.start] + text[op.source.end :]
    whole = op.kind in {"sub", "ins", "del"} or (
        op.kind in {"emo", "rate", "pitch"} and not any(ch.isalnum() for ch in outside)
    )
    if whole:
        return compile_whole(request, context_mode, index, op)
    text = parsed.source_text
    validate_candidate_alignment(alignment, request.source_audio, text)
    info = sf.info(io.BytesIO(request.source_audio.data))
    if info.channels != 1:
        raise ValueError("AuK Base local editing requires mono source audio")
    duration = info.frames / info.samplerate
    units = alignment.segments
    spans = validate_alignment(text, units, duration)
    ops = parsed.operations
    index, op = min(enumerate(ops), key=lambda pair: (pair[1].source.start, pair[0]))
    start, raw_end = query_intervals(
        units, text, [(index, op)], side="source", audio_duration=duration
    )[index]
    end = max(start, raw_end)
    original = end - start
    left = [j for j, (_, b) in enumerate(spans) if b <= op.source.start]
    right = [j for j, (a, _) in enumerate(spans) if a >= op.source.end]
    adjacent = ([left[-1]] if left else []) + ([right[0]] if right else [])
    window_start, window_end = start, end
    if op.kind == "pause":
        window_start = window_start if left else 0.0
        window_end = window_end if right else duration
        for j in adjacent:
            window_start = min(window_start, units[j].start)
            window_end = max(window_end, units[j].end)
    p = dict(op.params)
    zh = request.language == "zh"
    anchor = (
        text[slice(*spans[right[0]])]
        if right
        else (text[slice(*spans[left[-1]])] if left else "")
    )
    before = bool(right)
    position = (
        (
            f"‘{anchor}’{'前面' if before else '后面'}"
            if zh
            else f"{'before' if before else 'after'} '{anchor}'"
        )
        if anchor
        else ("语音开头" if zh else "the beginning of the speech")
    )
    target = original
    if op.kind == "emo":
        emotion = p["type"]
        command = (
            f"将情绪改为{ZH_EMOTIONS[emotion]}。"
            if zh
            else f"Change the emotion to {'fearful' if emotion == 'afraid' else emotion}."
        )
    elif op.kind == "pitch":
        value = float(p["semitones"])
        command = (
            f"将音高{'提高' if value > 0 else '降低'}{abs(value):g}个半音。"
            if zh
            else f"{'Raise' if value > 0 else 'Lower'} the pitch by {abs(value):g} semitones."
        )
    elif op.kind == "rate":
        target = original / float(p["factor"])
        command = (
            f"将语速调整为{p['factor']}倍。"
            if zh
            else f"Adjust the speech speed to {p['factor']}x."
        )
    elif op.kind == "pause":
        adding = p["act"] == "ins"
        target = original + 0.2 if adding else max(0.0, original - 0.2)
        command = (
            (f"在{position}增加停顿。" if adding else f"删除{position}的停顿。")
            if zh
            else (
                f"Add a pause {position}."
                if adding
                else f"Remove the pause {position}."
            )
        )
    else:
        raise ValueError(f"Unsupported operation: {op.kind}")
    target_window = window_end - window_start + target - original
    if window_end <= window_start:
        return identity_request(request, index, window_start, window_end)
    if target_window < -1e-8:
        raise ValueError("Invalid generation window")
    a, b = round(window_start * 50), round(window_end * 50)
    b = min(b, math.ceil(duration * 50))
    if b <= a:
        return identity_request(request, index, window_start, window_end)
    count = max(1, round(target_window * 50))
    metadata = dict(
        request_id=request.id,
        selected_operation_index=index,
        kind=op.kind,
        source_text=text,
        source_span=[op.source.start, op.source.end],
        operation_params=p,
        source_interval=[start, end],
        source_window=[window_start, window_end],
        target_edit_seconds=target,
        target_window_seconds=target_window,
        source_sample_rate=info.samplerate,
        source_samples=info.frames,
        source_audio_sha256=alignment.audio_sha256,
        source_frames=[a, b],
        target_frames=count,
        start_sample=min(info.frames, round(a * info.samplerate / 50)),
        end_sample=min(info.frames, round(b * info.samplerate / 50)),
        generated_samples=round(count * info.samplerate / 50),
    )
    return NativeRequest(
        "/edit/local",
        tuple(
            dict(
                instruction=command,
                start_frame=a,
                end_frame=b,
                target_frames=count,
                context_mode=context_mode,
                seed=0,
                nfe=32,
                cfg_strength=2.0,
                metadata=metadata,
            ).items()
        ),
        "audio",
        "file",
        "audio_base64",
        index,
        "local",
    )


def compile_whole(request, context_mode, index, op):
    parsed = parse_instruction(request.instruction_xml)
    text = parsed.source_text
    info = sf.info(io.BytesIO(request.source_audio.data))
    if info.channels != 1:
        raise ValueError("AuK Base requires mono source audio")
    duration = info.frames / info.samplerate
    if duration <= 0:
        raise ValueError("Source audio must be nonempty")
    params = dict(op.params)
    zh = request.language == "zh"
    target_text = text
    target = duration
    if op.kind in {"sub", "ins", "del"}:
        replacement = (
            params["targ"]
            if op.kind == "sub"
            else op.content
            if op.kind == "ins"
            else ""
        )
        target_text = render_replacement(text, op.source, replacement)
        target = (
            duration
            * len(target_text.encode("utf-8"))
            / max(1, len(text.encode("utf-8")))
        )
        if any(ch.isalnum() for ch in target_text):
            command = (
                f"将语音内容改为：{target_text}"
                if zh
                else f"Change the spoken content to: {target_text}"
            )
        else:
            command = "删除全部语音内容。" if zh else "Remove all spoken content."
    elif op.kind == "emo":
        emotion = params["type"]
        command = (
            f"将情绪改为{ZH_EMOTIONS[emotion]}。"
            if zh
            else f"Change the emotion to {'fearful' if emotion == 'afraid' else emotion}."
        )
    elif op.kind == "pitch":
        value = float(params["semitones"])
        command = (
            f"将音高{'提高' if value > 0 else '降低'}{abs(value):g}个半音。"
            if zh
            else f"{'Raise' if value > 0 else 'Lower'} the pitch by {abs(value):g} semitones."
        )
    elif op.kind == "rate":
        target = duration / float(params["factor"])
        command = (
            f"将语速调整为{params['factor']}倍。"
            if zh
            else f"Adjust the speech speed to {params['factor']}x."
        )
    else:
        raise ValueError(f"Unsupported whole-utterance operation: {op.kind}")
    count = max(1, round(target * 50))
    end = (info.frames * 50 + info.samplerate - 1) // info.samplerate
    metadata = dict(
        request_id=request.id,
        selected_operation_index=index,
        kind=op.kind,
        source_text=text,
        source_span=[op.source.start, op.source.end],
        operation_params=params,
        source_interval=[0.0, duration],
        source_window=[0.0, duration],
        target_edit_seconds=target,
        target_window_seconds=target,
        source_sample_rate=info.samplerate,
        source_samples=info.frames,
        source_audio_sha256=hashlib.sha256(request.source_audio.data).hexdigest(),
        source_frames=[0, end],
        target_frames=count,
        start_sample=0,
        end_sample=info.frames,
        generated_samples=round(count * info.samplerate / 50),
        generation_scope="whole",
        target_text=target_text,
    )
    return NativeRequest(
        "/edit/local",
        tuple(
            dict(
                instruction=command,
                start_frame=0,
                end_frame=end,
                target_frames=count,
                context_mode=context_mode,
                seed=0,
                nfe=32,
                cfg_strength=2.0,
                metadata=metadata,
            ).items()
        ),
        "audio",
        "file",
        "audio_base64",
        index,
        "whole",
    )


def identity_request(request, index, start, end):
    """Preserve the current audio when a local window contains no latent frame."""
    return NativeRequest(
        endpoint="",
        fields=(
            (
                "metadata",
                {
                    "reason": "empty_generation_window",
                    "source_window": [start, end],
                },
            ),
        ),
        audio_field="audio",
        audio_encoding="base64",
        response_encoding="wav",
        selected_operation_index=index,
        scope="identity",
        transport="identity",
    )
