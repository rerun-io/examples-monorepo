"""MultiprocessDecoder must fail fast when a worker can't initialize.

Regression guard: a decode worker opens its CUDA TorchCodec reader before it
signals ``ready``. If that init raises (corrupt/missing video, NVDEC or CUDA
unavailable, OOM) the worker exits without ever sending ``ready`` — and the
parent used to block forever on a bare ``out_q.get()`` during construction.
Construction must now surface a RuntimeError promptly instead of hanging.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mamma.engine.mp_decode import MultiprocessDecoder


def test_construction_raises_on_worker_init_failure() -> None:
    """A bad video path makes every worker fail init; build must raise, not hang."""
    bad_path: Path = Path("/nonexistent/__mamma_no_such_video__.mp4")
    started: float = time.perf_counter()
    with pytest.raises(RuntimeError):
        MultiprocessDecoder([bad_path], resize_hw=(64, 64))
    elapsed: float = time.perf_counter() - started
    # The worker fails init immediately; the parent should report within seconds.
    # A regression (the old unbounded get()) would instead hang until the suite's
    # faulthandler timeout — so anything well under a minute proves no hang.
    assert elapsed < 90.0, f"construction took {elapsed:.1f}s — likely hung on a dead worker"
