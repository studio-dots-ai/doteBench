"""Full-recording F0 analysis through resident Praat CLI workers."""

import argparse
import atexit
import base64
import io
import math
import sys
import queue
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from dotebench.services.utils.http import json_endpoint, service_app

DEFAULT_PRAAT_BIN = str(Path(sys.prefix) / "bin" / "praat")

_SCRIPT = Path(__file__).with_name("resources") / "praat_worker.praat"


class F0Span(BaseModel):
    start: float = Field(0.0, ge=0.0)
    end: float | None = Field(None, ge=0.0)
    label: str | None = None


class F0SpanResult(BaseModel):
    start: float
    end: float
    duration_sec: float
    label: str | None = None
    min_f0_hz: float | None
    max_f0_hz: float | None
    median_f0_hz: float | None
    mean_f0_hz: float | None
    voiced_fraction: float
    num_frames: int
    num_voiced_frames: int


class PraatWorker:
    def __init__(self, binary: str, timeout_sec: float):
        self.binary = binary
        self.timeout_sec = timeout_sec
        self._temp = tempfile.TemporaryDirectory(prefix="dotebench-praat-")
        self.root = Path(self._temp.name)
        self.process: subprocess.Popen | None = None
        self._stderr = None
        self._start()

    def _start(self) -> None:
        self._stderr = (self.root / "stderr.log").open("ab")
        self.process = subprocess.Popen(
            [
                self.binary,
                "--no-pref-files",
                "--no-plugins",
                "--run",
                str(_SCRIPT),
                str(self.root),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            start_new_session=True,
        )

    def _stop_process(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            (self.root / "stop.ready").touch()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def restart(self) -> None:
        self._stop_process()
        for name in (
            "stop.ready",
            "request.ready",
            "request.tmp",
            "response.txt",
            "done.txt",
        ):
            (self.root / name).unlink(missing_ok=True)
        self._start()

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
        floor: float,
        ceiling: float,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.process is None or self.process.poll() is not None:
            self.restart()
        wav = self.root / f"{uuid4().hex}.wav"
        sf.write(wav, audio, sample_rate, format="WAV", subtype="FLOAT")
        request_id = uuid4().hex
        try:
            for name in ("response.txt", "done.txt"):
                (self.root / name).unlink(missing_ok=True)
            (self.root / "request.tmp").write_text(
                f"{request_id}\n{wav}\n{step}\n{floor}\n{ceiling}\n", encoding="utf-8"
            )
            (self.root / "request.tmp").replace(self.root / "request.ready")
            deadline = time.monotonic() + self.timeout_sec
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Praat worker exited with code {self.process.returncode}"
                    )
                done = self.root / "done.txt"
                if (
                    done.exists()
                    and done.read_text(encoding="utf-8").strip() == request_id
                ):
                    lines = (
                        (self.root / "response.txt")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    )
                    if not lines or not lines[0].startswith("frames "):
                        raise RuntimeError("Malformed Praat worker response")
                    count = int(lines[0].split()[1])
                    if len(lines) - 1 != count:
                        raise RuntimeError("Praat worker frame count mismatch")
                    frames = np.asarray(
                        [[float(x) for x in line.split()] for line in lines[1:]],
                        dtype=np.float64,
                    ).reshape(-1, 2)
                    if not np.all(np.isfinite(frames)):
                        raise RuntimeError("Non-finite Praat worker response")
                    return frames[:, 0], frames[:, 1]
                time.sleep(0.01)
            raise TimeoutError(f"Praat worker timed out after {self.timeout_sec}s")
        finally:
            wav.unlink(missing_ok=True)

    def close(self) -> None:
        self._stop_process()
        self._temp.cleanup()


class PraatPool:
    def __init__(
        self, binary: str = DEFAULT_PRAAT_BIN, workers: int = 4, timeout_sec: float = 60
    ):
        if workers < 1 or timeout_sec <= 0:
            raise ValueError("Praat workers and timeout must be positive")
        resolved = shutil.which(binary)
        if resolved is None:
            raise FileNotFoundError(f"Praat executable not found: {binary}")
        self.binary = resolved
        self.version = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        self.workers = workers
        self.timeout_sec = timeout_sec
        self._queue: queue.Queue[PraatWorker] = queue.Queue()
        self._all: list[PraatWorker] = []
        try:
            for _ in range(workers):
                worker = PraatWorker(resolved, timeout_sec)
                self._all.append(worker)
                self._queue.put(worker)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
        floor: float,
        ceiling: float,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        worker = self._queue.get()
        try:
            for attempt in range(2):
                try:
                    return worker.analyze(audio, sample_rate, floor, ceiling, step)
                except (RuntimeError, TimeoutError):
                    worker.restart()
                    if attempt:
                        raise
            raise AssertionError("unreachable")
        finally:
            self._queue.put(worker)

    def close(self) -> None:
        for worker in self._all:
            worker.close()
        self._all.clear()


def analyze_spans(
    audio: np.ndarray,
    sample_rate: int,
    spans: list[F0Span],
    *,
    pool: PraatPool,
    f0_min_hz: float = 50.0,
    f0_max_hz: float = 600.0,
    time_step_sec: float | None = None,
) -> list[F0SpanResult]:
    duration = len(audio) / sample_rate
    bounds = []
    for span in spans:
        start = min(max(float(span.start), 0.0), duration)
        end = (
            duration if span.end is None else min(max(float(span.end), start), duration)
        )
        bounds.append((start, end, span.label))
    if not spans:
        return []
    if len(audio) < math.ceil(3 * sample_rate / f0_min_hz):
        times = freqs = np.empty(0)
    else:
        times, freqs = pool.analyze(
            audio, sample_rate, f0_min_hz, f0_max_hz, time_step_sec or 0.0
        )
    results = []
    for start, end, label in bounds:
        # Frame centres use a half-open interval except at the recording end.
        mask = (times >= start - 1e-10) & (times < end - 1e-10)
        if end >= duration - 1e-10:
            mask = (times >= start - 1e-10) & (times <= end + 1e-10)
        selected = freqs[mask]
        voiced = selected[selected > 0]
        count = len(selected)
        results.append(
            F0SpanResult(
                start=start,
                end=end,
                duration_sec=max(0.0, end - start),
                label=label,
                min_f0_hz=float(np.min(voiced)) if len(voiced) else None,
                max_f0_hz=float(np.max(voiced)) if len(voiced) else None,
                median_f0_hz=float(np.median(voiced)) if len(voiced) else None,
                mean_f0_hz=float(np.mean(voiced)) if len(voiced) else None,
                voiced_fraction=len(voiced) / count if count else 0.0,
                num_frames=count,
                num_voiced_frames=len(voiced),
            )
        )
    return results


def build(*, praat_bin=None, praat_workers=4, praat_timeout_sec=60.0):
    praat_bin = praat_bin or DEFAULT_PRAAT_BIN
    pool = PraatPool(praat_bin, praat_workers, praat_timeout_sec)

    def measure(payload):
        raw = base64.b64decode(payload["audio"], validate=True)
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains non-finite samples")
        floor = float(payload.get("f0_min_hz", 50.0))
        ceiling = float(payload.get("f0_max_hz", 600.0))
        step = payload.get("time_step_sec")
        if (
            not math.isfinite(floor)
            or not math.isfinite(ceiling)
            or not 0 < floor < ceiling
        ):
            raise ValueError("Require finite 0 < f0_min_hz < f0_max_hz")
        if step is not None and (not math.isfinite(float(step)) or float(step) <= 0):
            raise ValueError("time_step_sec must be finite and positive")
        spans = payload.get("spans")
        if spans is None:
            spans = [{"start": 0.0, "end": len(audio) / sample_rate}]
        parsed_spans = []
        for item in spans:
            span = F0Span(**item)
            if not math.isfinite(span.start) or (
                span.end is not None
                and (not math.isfinite(span.end) or span.end < span.start)
            ):
                raise ValueError("F0 span requires finite start <= end")
            parsed_spans.append(span)
        results = [
            x.model_dump()
            for x in analyze_spans(
                audio,
                sample_rate,
                parsed_spans,
                pool=pool,
                f0_min_hz=floor,
                f0_max_hz=ceiling,
                time_step_sec=None if step is None else float(step),
            )
        ]
        return {
            "status": "success",
            "sample_rate": sample_rate,
            "duration_sec": len(audio) / sample_rate,
            "spans": results,
        }

    measure.close = pool.close
    measure.configuration = {
        "praat_bin": pool.binary,
        "praat_version": pool.version,
        "praat_workers": praat_workers,
        "praat_timeout_sec": praat_timeout_sec,
    }
    return measure


def create_app(measure):
    app = service_app(
        title="f0",
        health=lambda: {
            "status": "ready",
            "ability": "f0",
            "model_identity": None,
            "configuration": getattr(measure, "configuration", {}),
        },
        close=getattr(measure, "close", None),
    )
    app.post("/measure")(json_endpoint(measure))
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--praat-bin", default=DEFAULT_PRAAT_BIN)
    parser.add_argument("--praat-workers", type=int, default=4)
    parser.add_argument("--praat-timeout-sec", type=float, default=60.0)
    args = parser.parse_args()
    measure = build(
        praat_bin=args.praat_bin,
        praat_workers=args.praat_workers,
        praat_timeout_sec=args.praat_timeout_sec,
    )
    import uvicorn

    uvicorn.run(create_app(measure), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
