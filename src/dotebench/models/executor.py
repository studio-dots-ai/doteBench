"""Execute compiler plans while feeding each output audio to the next call."""

import fcntl
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import soundfile as sf

from dotebench.compilers.base import (
    CompilationError,
    CompilePlan,
    CompileStep,
    Requirement,
)
from dotebench.domain import InfrastructureError
from dotebench.models.requests import NativeRequest


def _write(path, data):
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise InfrastructureError("Cannot persist compiler execution audio") from exc


def _audio_record(audio, path):
    _write(path, audio.data)
    record = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(audio.data).hexdigest(),
    }
    with sf.SoundFile(io.BytesIO(audio.data)) as stream:
        record.update(
            duration=stream.frames / stream.samplerate,
            frames=stream.frames,
            sample_rate=stream.samplerate,
        )
    return record


def _requirement_record(requirement, value):
    return requirement.record(value)


def execute_plan(candidate, compiler, request):
    """Run a light Python plan through one candidate backend."""

    try:
        plan = compiler.compile(request)
        if not isinstance(plan, CompilePlan) or not plan.steps:
            raise TypeError("Compiler must produce a nonempty CompilePlan")
        binders = []
        for step in plan.steps:
            if not isinstance(step, CompileStep):
                raise TypeError("Compiler plan contains an invalid step")
            if step.case_id != request.id or step.language != request.language:
                raise ValueError("Compiler step changed the case identity")
            binders.append(compiler.resolve(step.binder))
            if not all(
                isinstance(requirement, Requirement)
                for requirement in step.requirements
            ):
                raise TypeError("Compiler requirements must implement Requirement")
            names = [requirement.name for requirement in step.requirements]
            if len(names) != len(set(names)):
                raise ValueError("Compiler step has duplicate runtime requirements")
    except InfrastructureError:
        raise
    except Exception as exc:
        raise CompilationError(f"Compiler planning failed: {exc}") from exc
    identity = candidate.compilation_identity()
    directory = Path(candidate.trace_dir) / (
        hashlib.sha256(request.id.encode()).hexdigest()[:16] + "-" + uuid4().hex
    )
    try:
        directory.mkdir(parents=True)
    except OSError as exc:
        raise InfrastructureError("Cannot create compiler trace directory") from exc
    trace_path = directory / "trace.json"
    candidate.last_execution_trace = trace_path
    trace = {
        "case_id": request.id,
        "compiler": identity,
        "instruction_xml": request.instruction_xml,
        "steps": [],
        "status": "running",
    }

    def save():
        temporary = directory / "trace.tmp"
        try:
            temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(trace_path)
        except OSError as exc:
            raise InfrastructureError("Cannot persist compiler trace") from exc

    def index_trace():
        entry = {
            "case_id": request.id,
            "trace": str(trace_path.resolve()),
            "status": trace["status"],
        }
        try:
            with (Path(candidate.trace_dir) / "index.jsonl").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.write(json.dumps(entry) + "\n")
                handle.flush()
                fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError as exc:
            raise InfrastructureError("Cannot index compiler trace") from exc

    audio = request.source_audio
    save()
    for index, (step, binder) in enumerate(zip(plan.steps, binders)):
        record = {
            "execution_index": index,
            "operation_index": step.operation_index,
            "instruction_xml": step.instruction_xml,
            "source_text": step.source_text,
            "target_text": step.target_text,
            "binder": step.binder,
            "status": "running",
        }
        trace["steps"].append(record)
        candidate.last_call_metadata = {}
        started = time.monotonic()
        try:
            record["input_audio"] = _audio_record(
                audio, directory / f"{index:03d}-input.wav"
            )
            resolved = {}
            requirement_records = []
            for requirement in step.requirements:
                value = requirement.resolve(candidate, audio)
                resolved[requirement.name] = value
                requirement_records.append(_requirement_record(requirement, value))
            if requirement_records:
                record["requirements"] = requirement_records
            try:
                bound = binder.bind(step, audio, resolved)
                if not isinstance(bound, NativeRequest):
                    raise TypeError("Compiler bind() must return NativeRequest")
            except InfrastructureError:
                raise
            except Exception as exc:
                raise CompilationError(f"Compiler binding failed: {exc}") from exc
            bound = replace(
                bound,
                case_id=request.id,
            )
            output = candidate.invoke(bound, audio)
            output.validate()
            record["output_audio"] = _audio_record(
                output, directory / f"{index:03d}-output.wav"
            )
            audio = output
            record["status"] = "ok"
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            trace["status"] = "failed"
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["call"] = candidate.last_call_metadata
            save()
            if trace["status"] == "failed":
                index_trace()
    trace["status"] = "ok"
    save()
    index_trace()
    return audio
