"""Resident Praat worker contract and recovery tests."""

import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from dotebench.services.backends.f0 import (
    DEFAULT_PRAAT_BIN,
    F0Span,
    PraatPool,
    analyze_spans,
)


def praat_binary():
    binary = DEFAULT_PRAAT_BIN
    if not shutil.which(binary):
        pytest.skip("Praat CLI is unavailable")
    return binary


def tone(hz, seconds=1.0, sr=16000):
    t = np.arange(round(seconds * sr)) / sr
    return np.asarray(0.5 * np.sin(2 * np.pi * hz * t), dtype=np.float32)


def test_multi_span_and_short_recording():
    with_pool = PraatPool(praat_binary(), workers=2)
    try:
        audio = tone(200)
        spans = [
            F0Span(start=0, end=1),
            F0Span(start=0, end=0.01),
            F0Span(start=0.3, end=0.5),
            F0Span(start=0.5, end=0.7),
        ]
        result = analyze_spans(audio, 16000, spans, pool=with_pool)
        assert len(result) == 4
        assert result[0].median_f0_hz == pytest.approx(200, abs=0.02)
        assert result[1].num_frames == 0
        assert result[2].num_frames > 0
        assert result[3].num_frames > 0
        assert result[2].num_frames + result[3].num_frames <= result[0].num_frames
        short = analyze_spans(
            tone(200, 0.02), 16000, [F0Span(start=0, end=0.02)], pool=with_pool
        )
        assert short[0].num_frames == 0
    finally:
        with_pool.close()


def test_concurrent_requests_crash_recovery_timeout_and_cleanup():
    pool = PraatPool(praat_binary(), workers=2, timeout_sec=10)
    roots = [w.root for w in pool._all]
    try:

        def measure(hz):
            return analyze_spans(tone(hz), 16000, [F0Span(start=0, end=1)], pool=pool)[
                0
            ].median_f0_hz

        with ThreadPoolExecutor(max_workers=8) as executor:
            observed = list(
                executor.map(measure, [120, 200, 250, 120, 200, 250, 120, 200])
            )
        assert observed == pytest.approx(
            [120, 200, 250, 120, 200, 250, 120, 200], abs=0.1
        )
        worker = pool._all[0]
        worker.process.kill()
        worker.process.wait()
        assert measure(200) == pytest.approx(200, abs=0.02)
        original_timeout = worker.timeout_sec
        for member in pool._all:
            member.timeout_sec = 1e-9
        with pytest.raises(TimeoutError):
            measure(200)
        for member in pool._all:
            member.timeout_sec = original_timeout
        assert measure(200) == pytest.approx(200, abs=0.02)
    finally:
        pool.close()
    assert all(not root.exists() for root in roots)
