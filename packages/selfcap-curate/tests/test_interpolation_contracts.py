"""Each production reader keeps its gap policy and endpoint behavior."""

import numpy as np
import pytest

from selfcap_curate.head_anchor import HeadTrack
from selfcap_curate.keypoints import interpolate
from selfcap_curate.quest_fit import hand_neighbors
from selfcap_curate.records import HandTrack


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0], ids=["first", "middle", "last"])
@pytest.mark.parametrize("limit", [0.04, 0.1], ids=["wrist", "head"])
@pytest.mark.parametrize("scale,accepted", [(0.0, False), (0.99, True), (1.0, True), (1.01, False)])
def test_anchor_interpolation_keeps_inclusive_gap(limit: float, scale: float, accepted: bool, fraction: float) -> None:
    gap = limit * scale
    values = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    actual = interpolate(np.array([gap * fraction]), np.array([0.0, gap]), values, np.ones(2, dtype=bool), limit)
    if accepted:
        np.testing.assert_allclose(actual, [np.array([2.0, 4.0, 6.0]) * fraction])
    else:
        assert np.isnan(actual).all()


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0], ids=["first", "middle", "last"])
@pytest.mark.parametrize("gap,accepted", [(0.0, False), (0.099, True), (0.1, True), (0.101, False)])
def test_head_track_keeps_inclusive_gap_and_invalid_identity(gap: float, accepted: bool, fraction: float) -> None:
    track = HeadTrack(np.array([0.0, gap]), np.tile(np.eye(3), (2, 1, 1)), np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]))
    rotations, positions, valid = track.at(np.array([gap * fraction]))
    assert valid.tolist() == [accepted]
    np.testing.assert_allclose(rotations, [np.eye(3)])
    np.testing.assert_allclose(positions, [np.array([2.0, 4.0, 6.0]) * fraction if accepted else np.zeros(3)])


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0], ids=["first", "middle", "last"])
@pytest.mark.parametrize("gap,accepted", [(0.0, False), (0.039, True), (0.04, False), (0.041, False)])
def test_hand_neighbors_keeps_strict_gap(gap: float, accepted: bool, fraction: float) -> None:
    hand = HandTrack(np.array([0.0, gap]), np.zeros((2, 21, 3)), np.ones(2, dtype=bool), np.zeros(2))
    right, alpha, valid = hand_neighbors(hand, np.array([gap * fraction]))
    assert right.tolist() == [1]
    assert valid.tolist() == [accepted]
    np.testing.assert_allclose(alpha, [fraction if gap > 0.0 else 0.0])


@pytest.mark.parametrize("endpoint", [0, 1])
def test_anchor_and_hand_interpolation_require_both_tracked_endpoints(endpoint: int) -> None:
    flags = np.ones(2, dtype=bool)
    flags[endpoint] = False
    stamps = np.array([0.0, 0.02])
    times = np.array([0.01])
    assert np.isnan(interpolate(times, stamps, np.zeros((2, 3)), flags, 0.04)).all()
    hand = HandTrack(stamps, np.zeros((2, 21, 3)), flags, np.zeros(2))
    assert not hand_neighbors(hand, times)[2].any()
