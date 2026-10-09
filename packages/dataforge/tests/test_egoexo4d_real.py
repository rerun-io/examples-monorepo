"""Ego-Exo4D on a real converted take: the GoPro calibration base logs against the cameras inside the HM fit.

SLAHMR fit each take from the localized GoPros in gopro_calibs.csv order, undistorted with OpenCV's fisheye model
(balance 0.8, 4K, then halved to 1080p), and saved those pinhole cameras and its reprojected keypoints in the npz.
So the camera nodes base logs must reproduce the fit's extrinsics, and our KB4 projection of the sidecar's calibration
(what the projections layer uses), undistorted the same way, must land on the shipped joints2d. Needs
`dataforge-convert egoexo4d --sequences cmu_bike02_4` first (licence keys).
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import raw_asset, read_chunks
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge import paths, schema
from dataforge.datasets.egoexo4d_download import fit_path
from dataforge.datasets.egoexo4d_layers import SIDECAR, read_sidecar
from dataforge.datasets.egoexo4d_source import stored_size
from dataforge.identity import SequenceIdentity

TAKE: str = os.environ.get("DATAFORGE_EGOEXO4D_TAKE", "cmu_bike02_4")
RAW_ROOT: Path = Path(os.environ.get("DATAFORGE_EGOEXO4D_ROOT", str(paths.raw_root() / "egoexo4d")))


@pytest.mark.golden
def test_gopro_calibration_reproduces_the_fit_cameras() -> None:
    identity = SequenceIdentity("egoexo4d", (TAKE,))
    sidecar: Path = raw_asset(f"egoexo4d sidecar of {TAKE} (dataforge-convert egoexo4d)", paths.sidecar_path(paths.output_root(), identity, SIDECAR))
    fit: Path = raw_asset(f"Ego-Exo4D-HM fit of {TAKE} (dataforge-download egoexo4d)", fit_path(RAW_ROOT, TAKE))
    _, cameras = read_sidecar(sidecar)
    base: dict[tuple[str, str], object] = {  # every static component base logs, by (entity, column): its one row
        (str(chunk.entity_path), name): chunk.to_record_batch().column(name)[0].as_py()
        for chunk in read_chunks(paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity))
        if chunk.is_static
        for name in chunk.to_record_batch().schema.names
    }
    with np.load(fit) as npz:
        cam_R, cam_t, intrins = npz["cam_R"][1:, 0], npz["cam_t"][1:, 0], npz["intrins"][1:, 0]  # view 0 copies view 1
        joints3d, joints2d, valid = npz["joints3d"][0], npz["joints2d"], npz["valid"][0] == 1
    assert len(cameras.gopros) >= len(cam_R)
    points: np.ndarray = joints3d[valid].reshape(-1, 3).astype(np.float64)
    for view, calib in enumerate(cameras.gopros[: len(cam_R)]):
        node: str = schema.cam_path(view + 1, 0)
        assert base[(node, "name")] == [calib.cam_uid]
        logged_R: np.ndarray = np.reshape(np.asarray(base[(node, "Transform3D:mat3x3")]), (3, 3)).T  # Rerun stores it column-major
        np.testing.assert_allclose(logged_R, cam_R[view], atol=1e-5, err_msg=calib.cam_uid)
        np.testing.assert_allclose(np.reshape(np.asarray(base[(node, "Transform3D:translation")]), 3), cam_t[view], atol=1e-4, err_msg=calib.cam_uid)
        logged_K: np.ndarray = np.reshape(np.asarray(base[(f"{node}/pinhole", "Pinhole:image_from_camera")]), (3, 3)).T
        stored = calib.camera(*stored_size(calib))
        assert stored.intrinsics.k_matrix is not None
        np.testing.assert_allclose(logged_K, stored.intrinsics.k_matrix, rtol=1e-6, err_msg=calib.cam_uid)
        np.testing.assert_allclose(
            np.reshape(np.asarray(base[(f"{node}/pinhole", "Pinhole:resolution")]), 2), stored_size(calib), err_msg=calib.cam_uid
        )
        cam_T_world: np.ndarray = np.linalg.inv(calib.world_T_cam)
        native = calib.camera(calib.image_width, calib.image_height)
        lens = native.distortion
        assert lens is not None and native.intrinsics.k_matrix is not None
        distortion: np.ndarray = np.array([lens.k1, lens.k2, lens.k3, lens.k4])
        rectified_K: np.ndarray = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            native.intrinsics.k_matrix, distortion, (calib.image_width, calib.image_height), np.eye(3), balance=0.8
        )
        halved: np.ndarray = rectified_K * np.array([[0.5], [0.5], [1.0]])
        np.testing.assert_allclose(halved, intrins[view], rtol=1e-4, err_msg=calib.cam_uid)
        fisheye_px: np.ndarray = project_fisheye62(points @ cam_T_world[:3, :3].T + cam_T_world[:3, 3], native)
        seen: np.ndarray = np.isfinite(fisheye_px).all(axis=1)
        rectified_px: np.ndarray = cv2.fisheye.undistortPoints(fisheye_px[seen].reshape(-1, 1, 2), native.intrinsics.k_matrix, distortion, P=halved)
        shipped: np.ndarray = joints2d[view][valid].reshape(-1, 2)[seen]
        assert seen.mean() > 0.5, f"{calib.cam_uid}: only {seen.mean():.0%} of the keypoints land in the fisheye image"
        np.testing.assert_allclose(rectified_px.reshape(-1, 2), shipped, atol=0.5, err_msg=calib.cam_uid)
