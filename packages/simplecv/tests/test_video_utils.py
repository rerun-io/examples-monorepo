"""Tests for shared video timeline helpers."""

import numpy as np

from simplecv.video_utils import frame_at


def test_frame_at_matches_viewer_latest_at_semantics() -> None:
    timestamps_ns = (np.arange(10) * 33_333_333).astype(np.int64)

    assert frame_at(timestamps_ns, 0.0) == 0
    assert frame_at(timestamps_ns, 10_000_000.0) == 0
    assert frame_at(timestamps_ns, 30_000_000.0) == 0
    assert frame_at(timestamps_ns, 33_333_333.0) == 1
    assert frame_at(timestamps_ns, 10_833_333_333.0) == 9
    assert frame_at(timestamps_ns, -5.0) == 0
