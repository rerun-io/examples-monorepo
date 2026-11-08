"""Settings the uncalibrated (rays) pose solve cannot honour fail at tracker construction, not silently."""

import pytest

pytest.importorskip("torch", reason="MASt3R-SLAM requires PyTorch")


def test_kernel_values_pass() -> None:
    from mast3r_slam.tracker import check_rays_tracking_settings

    check_rays_tracking_settings(1.345, 1e-3)


@pytest.mark.parametrize(("huber", "rel_error"), [(2.0, 1e-3), (1.345, 1e-4)])
def test_other_values_are_rejected(huber: float, rel_error: float) -> None:
    from mast3r_slam.tracker import check_rays_tracking_settings

    with pytest.raises(ValueError, match="no effect without calibration"):
        check_rays_tracking_settings(huber, rel_error)
