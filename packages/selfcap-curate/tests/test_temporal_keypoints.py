"""Centred body filtering preserves phase, gaps, and the separate hand treatment."""

import numpy as np


def test_body_filter_leaves_hand_treatment_unchanged() -> None:
    from selfcap_curate.keypoints import Keypoints3D
    from selfcap_curate.temporal_keypoints import stable_body

    xyz = np.zeros((61, 133, 3), dtype=np.float32)
    xyz[:, 91:] = np.arange(61, dtype=np.float32)[:, None, None]
    result = Keypoints3D(np.arange(61) / 24.0, xyz, np.ones((61,133),np.uint8), np.ones((61,133),np.float32), {}, {}, 24.0)
    np.testing.assert_array_equal(stable_body(result, xyz)[:, 91:], xyz[:, 91:])


def test_short_gaps_are_bridged_with_reduced_confidence_but_long_gaps_stay_missing() -> None:
    from selfcap_curate.temporal_keypoints import bridge_short_gaps

    positions = np.zeros((30, 1, 3), dtype=np.float32)
    positions[:, 0, 0] = np.arange(30)
    confidence = np.full((30, 1), 0.9, dtype=np.float32)
    positions[2:5] = np.nan
    positions[10:24] = np.nan
    filled, scores, bridged = bridge_short_gaps(positions, confidence, 24.0)
    np.testing.assert_allclose(filled[2:5, 0, 0], [2, 3, 4])
    assert bridged[2:5].all()
    assert (scores[2:5] < 0.9).all()
    assert np.isnan(filled[10:24]).all()
    assert not bridged[10:24].any()


def test_body_filter_is_time_symmetric_and_preserves_fast_motion_phase() -> None:
    from selfcap_curate.temporal_keypoints import responsive_body

    t = np.arange(121, dtype=np.float32) / 24.0
    signal = np.zeros((121, 1, 3), dtype=np.float32)
    signal[:, 0, 0] = 0.4 * np.sin(2 * np.pi * t)
    signal[30, 0, 1] = 0.2  # isolated tracking jump, including during fast motion
    filtered = responsive_body(signal, 24.0)
    reverse = responsive_body(signal[::-1].copy(), 24.0)[::-1]
    np.testing.assert_allclose(filtered, reverse, atol=1e-6)
    assert abs(filtered[30, 0, 1]) < 0.02
    # The central periodic peaks must not shift by a frame.
    reference = signal[24:96, 0, 0]
    correlations = [np.dot(reference, filtered[24 + lag:96 + lag, 0, 0]) for lag in range(-3, 4)]
    assert int(np.argmax(correlations)) - 3 == 0
