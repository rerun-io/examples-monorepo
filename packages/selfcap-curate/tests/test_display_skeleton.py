"""The display skeleton chooses one source and connects elbows to hand roots."""

import numpy as np

from selfcap_curate.display_style import DISPLAY_LINKS, skeleton_style
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
    assert np.isnan(actual[:, 120]).all()  # low-confidence fallback is missing
    assert np.isnan(actual[:, np.r_[0:3, 9:11, 13:91]]).all()
    np.testing.assert_allclose(actual[:, np.r_[3:9, 11:13]], 1.0)


def test_display_links_use_hand_roots_and_exclude_hidden_joints() -> None:
    links = {tuple(link) for link in DISPLAY_LINKS}
    assert (7, 91) in links and (8, 112) in links
    assert (91, 92) in links and (112, 113) in links
    assert {(5, 6), (5, 11), (6, 12), (11, 12)} <= links
    assert not {0, 1, 2, 9, 10, *range(13, 91)} & {joint for link in links for joint in link}


def test_an_inferred_endpoint_dims_the_whole_link() -> None:
    observed = np.ones((1, 133), dtype=bool)
    observed[0, 7] = False
    style = skeleton_style(observed)
    assert style.colors[0, 7].tolist() == [125, 135, 145, 190]
    assert style.colors[0, 91].tolist() == [255, 64, 255, 255]
    assert style.colors[0, 112].tolist() == [255, 255, 255, 255]
    for edge, (a, b) in enumerate(DISPLAY_LINKS):
        assert style.observed_edges[0, edge] == (a != 7 and b != 7)
        if a == 7 or b == 7:
            assert style.edge_colors[0, edge].tolist() == [105, 115, 125, 150]


def test_quest_hand_observation_uses_raw_camera_detection() -> None:
    from selfcap_curate.keypoints import SelectedView, exo_display_observed

    pixels = np.zeros((1, 133, 2), dtype=np.float32)
    detected = pixels.copy()
    detected[0, 91, 0] = 14.0
    detected[0, 92, 0] = 16.0
    scores = np.ones((1, 133), dtype=np.float64)
    scores[0, 93] = 0.69
    inliers = np.zeros((1, 133), dtype=bool)
    inliers[0, 5] = True
    tracked = np.zeros((1, 133), dtype=bool)
    tracked[:, 91:112] = True
    observed = exo_display_observed(pixels, SelectedView(detected, scores), inliers, tracked)
    assert observed[0, 91]  # raw detection supports Quest despite absent triangulation support
    assert not observed[0, 92]  # beyond 15 px
    assert not observed[0, 93]  # low raw confidence
    assert observed[0, 94]
    assert observed[0, 5]  # body still uses geometric support
    assert not observed[0, 6]
