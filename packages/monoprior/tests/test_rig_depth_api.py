"""Fast tests for the ETH3D rig-depth demo boundary."""

from pathlib import Path

import numpy as np
from jaxtyping import Float32, UInt8

from monopriors.apis.rig_depth import eth3d_rig_metrics, metric_depth_to_disparity
from monopriors.apis.stereo_depth import MiddleburyCalibration, read_middlebury_calib


def test_middlebury_calibration_includes_disparity_offset(tmp_path: Path) -> None:
    calib_path: Path = tmp_path / "calib.txt"
    calib_path.write_text("cam0=[100 0 20; 0 101 10; 0 0 1]\ndoffs=3.5\nbaseline=120\n")

    calibration: MiddleburyCalibration = read_middlebury_calib(calib_path)

    assert calibration.doffs_px == 3.5
    assert calibration.baseline_m == 0.12


def test_eth3d_metric_depth_metrics_use_middlebury_doffs() -> None:
    calibration = MiddleburyCalibration(K_33=np.diag([100.0, 100.0, 1.0]).astype(np.float32), baseline_m=0.1, doffs_px=2.0)
    depth_hw: Float32[np.ndarray, "2 2"] = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
    expected_disparity: Float32[np.ndarray, "2 2"] = np.array([[12.0, 7.0], [4.5, 4.0]], dtype=np.float32)
    ground_truth: Float32[np.ndarray, "2 2"] = expected_disparity + np.array([[0.0, 0.5], [-1.5, 0.0]], dtype=np.float32)
    nonoccluded: UInt8[np.ndarray, "2 2"] = np.full((2, 2), 255, dtype=np.uint8)

    disparity_hw: Float32[np.ndarray, "2 2"] = metric_depth_to_disparity(depth_hw, calibration)
    epe_px, bad1_percent, abs_rel = eth3d_rig_metrics(depth_hw, ground_truth, nonoccluded, calibration)

    assert np.allclose(disparity_hw, expected_disparity)
    assert np.isclose(epe_px, 0.5)
    assert np.isclose(bad1_percent, 25.0)
    assert abs_rel >= 0.0
