"""Detector PTS stays on the presented container clock."""

import numpy as np
import pytest

from selfcap_curate.quest_fit import load_quest_hands, pairs
from selfcap_curate.records import CatalogHands, HandTrack, PhoneDetections


@pytest.mark.parametrize("rule", ["jump", "freeze", "cone"])
def test_catalog_hand_filter_keeps_the_legacy_tracking_guards(rule: str) -> None:

    times = np.arange(30, dtype=np.float64) * 0.01
    points = np.full((30, 133, 3), np.nan)
    for base in (91, 112):
        points[:, base : base + 21, 0] = 0.1 + np.arange(30)[:, None] * 0.001
        points[:, base : base + 21, 1] = np.arange(21)[None, :] * 0.001
        points[:, base : base + 21, 2] = 1.0
    if rule == "jump":
        points[15, 91:112, 0] += 3.0
        rejected = [14, 15, 16, 17]
    elif rule == "freeze":
        points[5:8, 91:112] = points[4, 91:112]
        rejected = [5, 6]
    else:
        points[:10, 91:112, 2] = -1.0
        rejected = list(range(8))
    source = CatalogHands(times, points, times, np.zeros((30, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (30, 1)))
    result = load_quest_hands(source)
    assert not result["left"].valid[rejected].any()
    assert result["right"].valid.all()


def test_pairs_use_presented_phone_pts_against_raw_quest_stamps() -> None:

    times = np.array([0.04, 0.06, 0.08])
    joints = np.zeros((3, 21, 3))
    joints[:, :, 0] = times[:, None]
    joints[:, :, 2] = 1.0
    track = HandTrack(times, joints, np.ones(3, dtype=bool), np.ones(3))
    detections = PhoneDetections(np.array([0.04, 0.06]), np.zeros((2, 133, 2)), np.ones((2, 133)))
    result = pairs(detections, {"left": track, "right": track}, 0.0)
    assert len(result.world) == 60
    np.testing.assert_allclose(np.unique(result.world[:, 0]), [0.04, 0.06])
