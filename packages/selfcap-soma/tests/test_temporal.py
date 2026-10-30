import numpy as np

from selfcap_soma.temporal import second_difference_mm


def test_second_difference_reports_24_hz_motion_in_mm_and_masks_gaps() -> None:
    times = np.arange(4, dtype=np.float64) / 24.0
    points = np.zeros((4, 1, 3), dtype=np.float32)
    points[:, 0, 0] = [0., .001, 0., .001]
    np.testing.assert_allclose(second_difference_mm(points, times), [[2.], [2.]], atol=1e-6)
    times[2:] += 1.0
    assert np.isnan(second_difference_mm(points, times)).all()
