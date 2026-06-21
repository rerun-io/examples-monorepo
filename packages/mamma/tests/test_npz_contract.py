"""Round-trip guard for the ma_cap per-camera NPZ contract.

dump_camera_npz must be the exact inverse of load_camera_npz — the field<->key
mapping (cam_int<->k_matrix, cam_ext<->world_to_cam, cam_img_w/h<->width/height)
lives in one place, so a write-then-read reproduces the calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from mamma.calibration.npz_contract import CameraCalibration, dump_camera_npz, load_camera_npz


def test_camera_npz_round_trip(tmp_path: Path) -> None:
    """load(dump(cam)) reproduces name, K, extrinsics (meters), and image size."""
    k_matrix: Float64[ndarray, "3 3"] = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    world_to_cam: Float64[ndarray, "4 4"] = np.eye(4)
    world_to_cam[:3, 3] = [0.5, -0.2, 2.0]  # meters: |t| <= 200, so load does NOT mm->m re-normalize
    cam: CameraCalibration = CameraCalibration(name="A001", k_matrix=k_matrix, world_to_cam=world_to_cam, width=1920, height=1080)

    path: Path = tmp_path / "A001.npz"
    dump_camera_npz(cam, path)
    loaded: CameraCalibration = load_camera_npz(path)

    assert loaded.name == cam.name
    assert loaded.width == cam.width
    assert loaded.height == cam.height
    assert loaded.k_matrix.dtype == np.float64
    assert loaded.world_to_cam.dtype == np.float64
    np.testing.assert_allclose(loaded.k_matrix, cam.k_matrix)
    np.testing.assert_allclose(loaded.world_to_cam, cam.world_to_cam)
