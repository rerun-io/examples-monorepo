"""Performance regression guard for ``log_video`` using Rerun ``Mp4Reader``."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import rerun as rr

from simplecv.rerun_log_utils import log_video

_HOCAP_BASE = Path("data/hocap/sample")
_MAX_MP4_READER_TIME_S: float = 1.5
_TRIALS: int = 5


def _find_hololens_mp4() -> Path | None:
    if not _HOCAP_BASE.exists():
        return None
    candidates: list[Path] = sorted(_HOCAP_BASE.rglob("hololens*/output.mp4"))
    return candidates[0] if candidates else None


def _time_log(mp4: Path, tmp_path: Path, trial: int) -> float:
    rrd_path: Path = tmp_path / f"stream-{trial}.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(
        application_id="speed-video-stream",
        recording_id=f"speed-{trial}",
    )
    rec.save(str(rrd_path))
    t0: float = time.perf_counter()
    log_video(mp4, Path("/v"), recording=rec)
    del rec  # finalize before timing stops
    return time.perf_counter() - t0


def test_log_video_mp4_reader_ingestion_under_budget(tmp_path: Path) -> None:
    """Mp4Reader ingestion stays below its measured regression ceiling."""
    mp4: Path | None = _find_hololens_mp4()
    if mp4 is None:
        pytest.skip("hocap sample not downloaded (run pixi _download-hocap-sample)")

    _time_log(mp4, tmp_path, -1)  # warm OS file cache

    stream_times: list[float] = [_time_log(mp4, tmp_path, t) for t in range(_TRIALS)]
    stream_median: float = sorted(stream_times)[_TRIALS // 2]
    print(
        f"Mp4Reader median: {stream_median * 1000:.1f} ms (trials: "
        f"{[f'{x * 1000:.0f}ms' for x in stream_times]}, budget: "
        f"{_MAX_MP4_READER_TIME_S * 1000:.0f} ms)"
    )

    assert stream_median <= _MAX_MP4_READER_TIME_S, (
        f"Mp4Reader median {stream_median * 1000:.1f} ms exceeds "
        f"{_MAX_MP4_READER_TIME_S * 1000:.0f} ms budget."
    )
