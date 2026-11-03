"""The display skeleton chooses one source and connects elbows to hand roots."""

import numpy as np

from selfcap_curate.display_style import DISPLAY_LINKS
from selfcap_curate.keypoints import display_skeleton


def test_display_selection_and_hand_fallback() -> None:
    times = np.array([0.01, 0.1])
    xyz = np.ones((2, 133, 3), dtype=np.float32)
    confidence = np.ones((2, 133), dtype=np.float32)
    confidence[:, 120] = 0.2
    quest = np.full((3, 133, 3), np.nan, dtype=np.float32)
    quest[:2, 91:112] = 3.0
    quest[1, 91:112] = 5.0
    fusion = display_skeleton(times, xyz, confidence, (np.array([0.0, 0.02, 0.1]), quest))
    actual = fusion.positions
    assert fusion.quest_tracked[0, 91] and not fusion.quest_tracked[1, 91]
    np.testing.assert_allclose(actual[0, 91:112], 4.0)
    np.testing.assert_allclose(actual[1, 91:112], 1.0)  # no bridging the 80 ms gap
    np.testing.assert_allclose(actual[:, 112:120], 1.0)  # absent right Quest hand
    np.testing.assert_allclose(actual[:, 120], 1.0)  # Low confidence is red, not hidden.
    assert fusion.confidence[0, 91] == 1.0  # Quest has no per-joint score.
    assert np.isnan(actual[:, np.r_[0:3, 9:11, 13:91]]).all()
    np.testing.assert_allclose(actual[:, np.r_[3:9, 11:13]], 1.0)


def test_display_links_use_hand_roots_and_exclude_hidden_joints() -> None:
    links = {tuple(link) for link in DISPLAY_LINKS}
    assert (7, 91) in links and (8, 112) in links
    assert (91, 92) in links and (112, 113) in links
    assert {(5, 6), (5, 11), (6, 12), (11, 12)} <= links
    assert not {0, 1, 2, 9, 10, *range(13, 91)} & {joint for link in links for joint in link}
