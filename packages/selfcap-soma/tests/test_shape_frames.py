import numpy as np

from selfcap_soma.shape_frames import select_shape_frames


def test_lowered_confidence_with_two_view_support_is_eligible() -> None:
    times = np.arange(800, dtype=np.float64) / 24.0
    xyz = np.zeros((800, 133, 3), dtype=np.float64)
    confidence = np.full((800, 133), 0.45)
    support = np.full((800, 133), 2, dtype=np.int64)
    result = select_shape_frames(times, xyz, confidence, support)
    assert len(result.indices) >= 10
    assert not result.used_fallback


def test_uses_best_supported_frames_across_full_segment_when_early_window_is_bad() -> None:
    times = np.arange(120, dtype=np.float64)
    xyz = np.zeros((120, 133, 3), dtype=np.float64)
    xyz[:30] = np.nan
    confidence = np.full((120, 133), 0.5)
    support = np.full((120, 133), 2, dtype=np.int64)
    support[30:50] = 0
    result = select_shape_frames(times, xyz, confidence, support)
    assert result.used_fallback
    assert len(result.indices) >= 10
    assert np.all(result.indices >= 50)


def test_fallback_keeps_low_confidence_bridged_evidence_when_direct_support_is_sparse() -> None:
    times = np.arange(12, dtype=np.float64)
    xyz = np.zeros((12, 133, 3), dtype=np.float64)
    confidence = np.full((12, 133), .2)
    support = np.zeros((12, 133), dtype=np.int64)
    result = select_shape_frames(times, xyz, confidence, support)
    assert result.used_fallback
    np.testing.assert_array_equal(result.indices, np.arange(12))


def test_one_missing_hip_does_not_discard_observed_upper_body() -> None:
    times = np.arange(120, dtype=np.float64)
    xyz = np.zeros((120, 133, 3), dtype=np.float64)
    confidence = np.full((120, 133), .6)
    support = np.full((120, 133), 2, dtype=np.int64)
    xyz[:, 11] = np.nan
    confidence[:, 11] = 0.
    support[:, 11] = 0
    assert len(select_shape_frames(times, xyz, confidence, support).indices) >= 3


def test_absent_torso_has_distinct_input_limitation() -> None:
    import pytest

    from selfcap_soma.shape_frames import InsufficientBodyObservation

    with pytest.raises(InsufficientBodyObservation):
        select_shape_frames(np.arange(10, dtype=np.float64), np.full((10,133,3),np.nan), np.zeros((10,133)), None)
