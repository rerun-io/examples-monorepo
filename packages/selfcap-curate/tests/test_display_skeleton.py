"""The display skeleton chooses one source and connects elbows to hand roots."""

import numpy as np

from selfcap_curate.keypoints import DISPLAY_CLASS_IDS, DISPLAY_CLASSES, DISPLAY_KEYPOINT_IDS, DISPLAY_LINKS, display_skeleton


def test_display_selection_and_hand_fallback() -> None:
    times = np.array([0.01, 0.1])
    xyz = np.ones((2, 133, 3), dtype=np.float32)
    confidence = np.ones((2, 133), dtype=np.float32)
    confidence[:, 120] = 0.2
    quest = np.full((3, 133, 3), np.nan, dtype=np.float32)
    quest[:2, 91:112] = 3.0
    quest[1, 91:112] = 5.0
    actual = display_skeleton(times, xyz, confidence, (np.array([0.0, 0.02, 0.1]), quest))
    np.testing.assert_allclose(actual[0, 91:112], 4.0)
    np.testing.assert_allclose(actual[1, 91:112], 1.0)  # no bridging the 80 ms gap
    np.testing.assert_allclose(actual[:, 112:120], 1.0)  # absent right Quest hand
    assert np.isnan(actual[:, 120]).all()  # low-confidence fallback is missing
    assert np.isnan(actual[:, np.r_[0:3, 9:11, 23:91]]).all()
    np.testing.assert_allclose(actual[:, np.r_[3:9, 11:23]], 1.0)


def test_display_links_use_hand_roots_and_exclude_hidden_joints() -> None:
    links = {tuple(link) for link in DISPLAY_LINKS}
    assert (7, 91) in links and (8, 112) in links
    assert (91, 92) in links and (112, 113) in links
    assert not {0, 1, 2, 9, 10, *range(23, 91)} & {joint for link in links for joint in link}


def test_every_bone_has_two_points_in_its_side_colour_class() -> None:
    assert DISPLAY_CLASS_IDS[91] == 0 and DISPLAY_CLASS_IDS[112] == 1
    for side, description in enumerate(DISPLAY_CLASSES):
        ids = set(DISPLAY_KEYPOINT_IDS[side == DISPLAY_CLASS_IDS].tolist())
        assert all(int(pair.keypoint0) in ids and int(pair.keypoint1) in ids for pair in description.keypoint_connections)
    assert DISPLAY_CLASSES[0].info.color != DISPLAY_CLASSES[1].info.color
