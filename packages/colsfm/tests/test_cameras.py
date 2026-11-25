"""cuSFM camera parameters must convert to the same COLMAP cameras the blob wrote."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from mapping_helpers import build_synthetic_rig

from colsfm.cameras import (
    camera_params_with_colmap_intrinsics,
    colmap_camera,
    colmap_cameras,
    fit_distortion_coefficients,
    with_colmap_intrinsics,
)
from colsfm.export import write_colmap_model
from colsfm.frames_meta import FRAMES_META_NAME, CameraParams, FramesMeta, read_frames_meta, write_frames_meta
from colsfm.geometry import rigid3d_from_axis_angle_degrees
from colsfm.rerun_log import camera_k_matrix
from colsfm.run_report import FocalChange, focal_changes


def read_blob_cameras(path: Path) -> dict[int, pycolmap.Camera]:
    """Load the blob's own `cameras.txt` with COLMAP's reader.

    Args:
        path: Directory holding `cameras.txt`, `images.txt` and `points3D.txt`.

    Returns:
        Cameras keyed by `camera_id`.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction(str(path))
    return dict(reconstruction.cameras)


def assert_cameras_match(ours: dict[int, pycolmap.Camera], theirs: dict[int, pycolmap.Camera]) -> None:
    """Assert two camera sets agree on ids, models, sizes and parameters.

    Args:
        ours: Cameras produced by `colsfm.cameras`.
        theirs: Cameras read from a blob-written model.
    """
    assert sorted(ours) == sorted(theirs)
    for camera_id, expected in theirs.items():
        actual: pycolmap.Camera = ours[camera_id]
        assert actual.model.name == expected.model.name
        assert (actual.width, actual.height) == (expected.width, expected.height)
        assert np.allclose(actual.params, expected.params, atol=1e-9)


def distorted_pinhole_camera(distortion_coefficients: np.ndarray) -> CameraParams:
    """Build a `DISTORTED_PINHOLE` camera; no shipped dataset uses one.

    Args:
        distortion_coefficients: Coefficients to attach, of any length.

    Returns:
        Camera parameters with a known 3x3 intrinsic matrix.
    """
    return CameraParams(
        camera_params_id=3,
        sensor_name="synthetic",
        model_name="",
        sensor_id=3,
        frequency_hz=30.0,
        vehicle_T_cam=rigid3d_from_axis_angle_degrees(
            axis_xyz=np.array([0.0, 0.0, 1.0]), angle_degrees=0.0, translation_xyz=np.zeros(3)
        ),
        image_width=1280,
        image_height=720,
        projection_matrix=None,
        camera_matrix=np.array([[500.0, 0.0, 640.0], [0.0, 501.0, 360.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=distortion_coefficients,
        projection_model="DISTORTED_PINHOLE",
        rolling_shutter_delay_microseconds=0,
    )


def test_distorted_pinhole_becomes_full_opencv_with_twelve_parameters() -> None:
    """`DISTORTED_PINHOLE` maps to `FULL_OPENCV`: `fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6`."""
    coefficients: np.ndarray = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    camera: pycolmap.Camera = colmap_camera(distorted_pinhole_camera(coefficients))
    assert camera.model.name == "FULL_OPENCV"
    assert np.allclose(camera.params, np.concatenate([[500.0, 501.0, 640.0, 360.0], coefficients]))


def test_too_few_distortion_coefficients_are_zero_padded() -> None:
    """export.md §3.3: cuSFM warns and pads rather than failing."""
    fitted: np.ndarray = fit_distortion_coefficients(distorted_pinhole_camera(np.array([0.1, 0.2])), 8)
    assert np.allclose(fitted, np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_too_many_distortion_coefficients_are_truncated() -> None:
    """Only the first `wanted` coefficients survive, matching the blob's warning path."""
    fitted: np.ndarray = fit_distortion_coefficients(distorted_pinhole_camera(np.arange(12.0)), 8)
    assert np.allclose(fitted, np.arange(8.0))


def test_absent_distortion_coefficients_become_zeros() -> None:
    """An empty `distortion_coefficients` yields an undistorted `FULL_OPENCV` camera."""
    camera: pycolmap.Camera = colmap_camera(distorted_pinhole_camera(np.array([])))
    assert np.allclose(camera.params[4:], np.zeros(8))


def _refined(camera: pycolmap.Camera, focal_scale: float, extra_shift: float) -> pycolmap.Camera:
    """A copy of one COLMAP camera with its focal scaled and every distortion term shifted, as a refining solve leaves it.

    Args:
        camera: The nominal camera.
        focal_scale: Factor on fx and fy.
        extra_shift: Added to every parameter after cx, cy.

    Returns:
        The moved camera.
    """
    params: np.ndarray = np.asarray(camera.params, dtype=np.float64).copy()
    params[:2] *= focal_scale
    params[4:] += extra_shift
    return pycolmap.Camera(camera_id=camera.camera_id, model=camera.model, width=camera.width, height=camera.height, params=params)


@pytest.mark.parametrize("fisheye", [False, True], ids=["pinhole", "fisheye"])
def test_refined_intrinsics_reach_the_written_metadata_as_the_sparse_model_holds_them(fisheye: bool, tmp_path: Path) -> None:
    """`with_colmap_intrinsics` writes a solve's cameras back, so `frames_meta.json`, `sparse/` and the frusta agree.

    With `--refine-camera-model` the mapper moves every focal (and, on a distorted model, the distortion block), and
    `sparse/cameras.txt` is written from that reconstruction. The exported `frames_meta.json` and the Rerun layer read
    the metadata, so it has to carry the same numbers: PINHOLE through the rectified `projection_matrix` (whose 4th
    column, the stereo baseline term, keeps its geometry), the distorted models through `camera_matrix` and the
    coefficients. The focal report says how far each camera moved.
    """
    nominal: FramesMeta = build_synthetic_rig(num_frames=2, num_points=10, fisheye=fisheye).frames_meta
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    for camera in colmap_cameras(nominal).values():
        reconstruction.add_camera(_refined(camera, 1.012, 0.001 if fisheye else 0.0))
    write_colmap_model(tmp_path, reconstruction)

    refined: FramesMeta = with_colmap_intrinsics(nominal, reconstruction)
    write_frames_meta(tmp_path / FRAMES_META_NAME, refined)
    written: FramesMeta = read_frames_meta(tmp_path / FRAMES_META_NAME)

    sparse: dict[int, pycolmap.Camera] = read_blob_cameras(tmp_path / "sparse")
    assert_cameras_match(colmap_cameras(written), sparse)
    for camera_params_id, camera in written.cameras.items():
        assert np.allclose(camera_k_matrix(camera), sparse[camera_params_id].calibration_matrix(), atol=1e-9)
        assert camera.vehicle_T_cam.matrix() == pytest.approx(nominal.cameras[camera_params_id].vehicle_T_cam.matrix())
    changes: tuple[FocalChange, ...] = focal_changes(nominal, refined)
    assert [change.camera_params_id for change in changes] == sorted(nominal.cameras)
    assert all(change.focal_change_percent == pytest.approx(1.2) for change in changes)


def test_unmoved_intrinsics_leave_the_metadata_byte_identical() -> None:
    """A camera the solve did not move keeps its matrices exactly, so a held model rewrites nothing."""
    nominal: FramesMeta = build_synthetic_rig(num_frames=2, num_points=10).frames_meta
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    for camera in colmap_cameras(nominal).values():
        reconstruction.add_camera(camera)
    assert with_colmap_intrinsics(nominal, reconstruction).to_json() == nominal.to_json()


def test_a_refined_stereo_projection_keeps_its_baseline() -> None:
    """The right camera of a rectified pair, `P = K [I | (-B, 0, 0)]`, keeps the baseline `B` when its K is refined.

    So the 4th column follows the new focal (`-fx_new * B`) rather than keeping the nominal `-fx_old * B`.
    """
    baseline_m: float = 0.12
    k_old: np.ndarray = np.array([[500.0, 0.0, 640.0], [0.0, 501.0, 360.0], [0.0, 0.0, 1.0]])
    right: CameraParams = replace(
        distorted_pinhole_camera(np.array([])),
        projection_model="PINHOLE",
        camera_matrix=None,
        projection_matrix=k_old @ np.hstack([np.eye(3), np.array([[-baseline_m], [0.0], [0.0]])]),
    )
    k_new: np.ndarray = np.array([[506.0, 0.0, 641.5], [0.0, 507.0, 359.0], [0.0, 0.0, 1.0]])
    solved: pycolmap.Camera = pycolmap.Camera(
        camera_id=right.camera_params_id, model="PINHOLE", width=right.image_width, height=right.image_height, params=[506.0, 507.0, 641.5, 359.0]
    )
    refined: CameraParams = camera_params_with_colmap_intrinsics(right, solved)
    assert refined.projection_matrix is not None
    assert np.allclose(refined.projection_matrix, k_new @ np.hstack([np.eye(3), np.array([[-baseline_m], [0.0], [0.0]])]))
