"""Synthetic association tests for the vendored LAMP tracker."""

import numpy as np

from lamptrack.third_party.lamp.core.types import Detection2D
from lamptrack.third_party.lamp.tracking.tracker import LampTracker


def _detection(box: tuple[float, float, float, float], timestamp_ns: int) -> Detection2D:
    """Build a bare-box synthetic detection."""
    return Detection2D(
        box_xyxy=np.asarray(box, dtype=np.float32),
        box_score=0.9,
        keypoints=np.zeros((17, 3), dtype=np.float32),
        cam_idx=0,
        timestamp_ns=timestamp_ns,
        has_keypoints=False,
    )


def test_tracker_keeps_id_for_overlapping_detection() -> None:
    """Hungarian matching retains an ID for a strongly overlapping box."""
    tracker = LampTracker(num_cameras=1)
    identity = np.eye(4, dtype=np.float32)
    first = _detection((10.0, 20.0, 110.0, 220.0), 1_000_000_000)
    second = _detection((12.0, 22.0, 112.0, 222.0), 1_100_000_000)

    tracker.track_frameset({0: [first]}, {0: identity}, {0: None}, first.timestamp_ns)
    tracker.track_frameset({0: [second]}, {0: identity}, {0: None}, second.timestamp_ns)

    assert first.track_id == 1
    assert second.track_id == 1
    assert set(tracker.people) == {1}


def test_tracker_creates_id_for_disjoint_detection() -> None:
    """A disjoint box starts another track."""
    tracker = LampTracker(num_cameras=1)
    identity = np.eye(4, dtype=np.float32)
    first = _detection((10.0, 20.0, 110.0, 220.0), 1_000_000_000)
    second = _detection((400.0, 100.0, 500.0, 300.0), 1_100_000_000)

    tracker.track_frameset({0: [first]}, {0: identity}, {0: None}, first.timestamp_ns)
    tracker.track_frameset({0: [second]}, {0: identity}, {0: None}, second.timestamp_ns)

    assert second.track_id == 2
    assert set(tracker.people) == {1, 2}
