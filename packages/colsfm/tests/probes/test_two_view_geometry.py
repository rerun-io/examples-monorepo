"""Probe (b): two-view geometry from given matches, for PINHOLE and OPENCV_FISHEYE."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float, UInt32
from numpy import ndarray

from probes.synthetic import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SyntheticScene,
    matches_between,
    write_database,
)

CameraModelName: TypeAlias = str
"""A COLMAP camera model name accepted by `Camera.create_from_model_name`."""


def _two_view_projections(
    camera: pycolmap.Camera, cam2_from_cam1: pycolmap.Rigid3d, num_points: int = 200, seed: int = 3
) -> tuple[Float[ndarray, "n 2"], Float[ndarray, "n 2"], UInt32[ndarray, "n 2"]]:
    """Project a random point cloud into two views of the same camera model.

    Args:
        camera: The (shared) camera intrinsics.
        cam2_from_cam1: The relative pose between the two views.
        num_points: How many points to draw.
        seed: PRNG seed.

    Returns:
        `(points1, points2, matches)` where the points are pixel coordinates and
        `matches` is the identity correspondence over the surviving points.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    points_in_cam1: Float[ndarray, "n 3"] = np.stack(
        [
            rng.uniform(-4.0, 4.0, num_points),
            rng.uniform(-3.0, 3.0, num_points),
            rng.uniform(3.0, 9.0, num_points),
        ],
        axis=1,
    )
    points_in_cam2: Float[ndarray, "n 3"] = (
        points_in_cam1 @ cam2_from_cam1.rotation.matrix().T + cam2_from_cam1.translation
    )
    pixels1: Float[ndarray, "n 2"] = camera.img_from_cam(points_in_cam1)
    pixels2: Float[ndarray, "n 2"] = camera.img_from_cam(points_in_cam2)

    def _in_frame(pixels: Float[ndarray, "n 2"], points_in_cam: Float[ndarray, "n 3"]) -> ndarray:
        return (
            np.isfinite(pixels).all(axis=1)
            & (points_in_cam[:, 2] > 0.0)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < IMAGE_WIDTH)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < IMAGE_HEIGHT)
        )

    keep: ndarray = _in_frame(pixels1, points_in_cam1) & _in_frame(pixels2, points_in_cam2)
    points1: Float[ndarray, "n 2"] = pixels1[keep]
    points2: Float[ndarray, "n 2"] = pixels2[keep]
    identity: UInt32[ndarray, "n 2"] = np.tile(
        np.arange(len(points1), dtype=np.uint32)[:, None], (1, 2)
    )
    return points1, points2, identity


def _make_camera(model_name: CameraModelName, calibrated: bool = True) -> pycolmap.Camera:
    """Create a camera of the given model with plausible distortion.

    Args:
        model_name: `"PINHOLE"` or `"OPENCV_FISHEYE"`.
        calibrated: Sets `has_prior_focal_length`, which is what makes
            `estimate_two_view_geometry` take the calibrated (essential-matrix)
            path.  It defaults to False on a freshly created camera.

    Returns:
        The configured camera.
    """
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_name(
        1, model_name, 300.0, IMAGE_WIDTH, IMAGE_HEIGHT
    )
    if model_name == "PINHOLE":
        camera.params = [300.0, 300.0, IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0]
    else:
        # k1..k4 of the Kannala-Brandt polynomial, in the range RoboCap's
        # Fisheye62 calibration lands in.
        camera.params = [
            300.0,
            300.0,
            IMAGE_WIDTH / 2.0,
            IMAGE_HEIGHT / 2.0,
            -0.02,
            0.004,
            -0.0008,
            0.0001,
        ]
    camera.has_prior_focal_length = calibrated
    assert camera.verify_params()
    return camera


@pytest.mark.parametrize("model_name", ["PINHOLE", "OPENCV_FISHEYE"])
def test_estimate_two_view_geometry_recovers_a_calibrated_pose(
    model_name: CameraModelName,
) -> None:
    """`estimate_two_view_geometry` takes pixel points plus matches and returns a pose.

    The signature is
    `estimate_two_view_geometry(camera1, points1, camera2, points2, matches, options)`
    with float64 Nx2 points and uint32 Nx2 matches.  With
    `compute_relative_pose = True` the result carries `cam2_from_cam1`, up to the
    usual translation scale ambiguity, and `config == CALIBRATED`.

    The calibrated path is selected by `Camera.has_prior_focal_length`, not by
    the camera model — see `test_uncalibrated_path_is_chosen_without_prior_focal`.
    """
    camera: pycolmap.Camera = _make_camera(model_name)
    truth: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.0, np.deg2rad(6.0), 0.0])),
        np.array([-0.6, 0.05, 0.1]),
    )
    points1, points2, matches = _two_view_projections(camera, truth)
    assert len(points1) >= 50, f"{model_name}: too few in-frame projections"

    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    options.ransac.max_error = 2.0
    options.ransac.min_inlier_ratio = 0.25
    options.min_num_inliers = 20
    options.compute_relative_pose = True

    geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        camera, points1, camera, points2, matches, options
    )
    assert geometry.config == int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
    assert len(geometry.inlier_matches) >= 0.9 * len(matches)
    assert geometry.cam2_from_cam1 is not None

    estimated_direction: Float[ndarray, "3"] = geometry.cam2_from_cam1.translation / np.linalg.norm(
        geometry.cam2_from_cam1.translation
    )
    true_direction: Float[ndarray, "3"] = truth.translation / np.linalg.norm(truth.translation)
    assert np.dot(estimated_direction, true_direction) > 0.999
    angle_deg: float = float(
        np.rad2deg(geometry.cam2_from_cam1.rotation.angle_to(truth.rotation))
    )
    assert angle_deg < 0.5, f"{model_name}: rotation off by {angle_deg:.3f} deg"


def test_uncalibrated_path_is_chosen_without_prior_focal() -> None:
    """`has_prior_focal_length = False` silently downgrades the estimate.

    A PINHOLE pair then comes back UNCALIBRATED (a fundamental matrix, no pose),
    and an OPENCV_FISHEYE pair comes back DEGENERATE with zero inliers, because
    a fundamental matrix cannot model the fisheye projection.  Neither raises.
    This is the single easiest way to get a silently wrong pipeline: cameras
    imported from `frames_meta.json` default to `has_prior_focal_length = False`.
    """
    truth: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.0, np.deg2rad(6.0), 0.0])),
        np.array([-0.6, 0.05, 0.1]),
    )
    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    options.compute_relative_pose = True

    pinhole: pycolmap.Camera = _make_camera("PINHOLE", calibrated=False)
    assert pinhole.has_prior_focal_length is False
    points1, points2, matches = _two_view_projections(pinhole, truth)
    pinhole_geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        pinhole, points1, pinhole, points2, matches, options
    )
    assert pinhole_geometry.config == int(pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED)

    fisheye: pycolmap.Camera = _make_camera("OPENCV_FISHEYE", calibrated=False)
    points1, points2, matches = _two_view_projections(fisheye, truth)
    fisheye_geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        fisheye, points1, fisheye, points2, matches, options
    )
    assert fisheye_geometry.config == int(pycolmap.TwoViewGeometryConfiguration.DEGENERATE)
    assert len(fisheye_geometry.inlier_matches) == 0


def test_min_num_inliers_rejects_a_weak_pair() -> None:
    """`TwoViewGeometryOptions.min_num_inliers` marks a thin pair DEGENERATE.

    Note the value: COLMAP reports DEGENERATE (1), not UNDEFINED (0), when the
    inlier count falls under the threshold.
    """
    camera: pycolmap.Camera = _make_camera("PINHOLE")
    truth: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(), np.array([-0.5, 0.0, 0.0])
    )
    points1, points2, matches = _two_view_projections(camera, truth, num_points=40)

    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    options.min_num_inliers = 10_000
    geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        camera, points1, camera, points2, matches, options
    )
    assert geometry.config == int(pycolmap.TwoViewGeometryConfiguration.DEGENERATE)
    assert len(geometry.inlier_matches) == 0


def test_ransac_max_error_controls_outlier_rejection() -> None:
    """A tight `ransac.max_error` drops injected outliers; a loose one keeps them."""
    camera: pycolmap.Camera = _make_camera("PINHOLE")
    truth: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.0, np.deg2rad(5.0), 0.0])), np.array([-0.7, 0.0, 0.0])
    )
    points1, points2, matches = _two_view_projections(camera, truth)
    rng: np.random.Generator = np.random.default_rng(11)
    corrupted: Float[ndarray, "n 2"] = points2.copy()
    num_outliers: int = len(points2) // 4
    corrupted[:num_outliers] += rng.uniform(30.0, 80.0, (num_outliers, 2))

    tight: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    tight.ransac.max_error = 1.0
    tight.compute_relative_pose = True
    strict_geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        camera, points1, camera, corrupted, matches, tight
    )

    loose: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    loose.ransac.max_error = 200.0
    loose.compute_relative_pose = True
    loose_geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
        camera, points1, camera, corrupted, matches, loose
    )

    print(
        f"[probe] inliers: max_error=1.0 -> {len(strict_geometry.inlier_matches)}, "
        f"max_error=200.0 -> {len(loose_geometry.inlier_matches)} of {len(matches)}"
    )
    assert len(strict_geometry.inlier_matches) < len(loose_geometry.inlier_matches)
    assert len(strict_geometry.inlier_matches) >= len(matches) - num_outliers - 5


def test_verify_matches_runs_over_a_database(
    scene_on_disk: tuple[SyntheticScene, Path, Path], tmp_path: Path
) -> None:
    """`verify_matches(database_path, pairs_path, options)` fills the two-view table.

    It takes a text file of `image_name1 image_name2` pairs, reads the raw
    matches from the database, and writes back verified `TwoViewGeometry` rows.
    Unlike `estimate_two_view_geometry` it does not return anything.
    """
    scene, _, _ = scene_on_disk
    database_path: Path = tmp_path / "verify.db"
    write_database(scene, database_path)

    # Clear the geometries so the call has something to do.
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    database.clear_two_view_geometries()
    assert database.num_verified_image_pairs() == 0
    image_ids: list[int] = sorted(scene.reconstruction.images)
    database.close()

    pairs_path: Path = tmp_path / "pairs.txt"
    pairs: list[str] = []
    for image_id1 in image_ids:
        for image_id2 in image_ids:
            if image_id2 <= image_id1:
                continue
            if len(matches_between(scene, image_id1, image_id2)) >= 20:
                pairs.append(f"{scene.image_names[image_id1]} {scene.image_names[image_id2]}")
    pairs_path.write_text("\n".join(pairs) + "\n")

    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    options.ransac.max_error = 4.0
    options.min_num_inliers = 15
    pycolmap.verify_matches(database_path, pairs_path, options)

    database = pycolmap.Database.open(database_path)
    verified: int = database.num_verified_image_pairs()
    database.close()
    print(f"[probe] verify_matches verified {verified} of {len(pairs)} pairs")
    assert verified == len(pairs)


def test_two_view_geometry_options_defaults() -> None:
    """Pin the option defaults the pipeline will rely on."""
    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    assert options.min_num_inliers == 15
    assert options.ransac.max_error == pytest.approx(4.0)
    assert options.ransac.min_inlier_ratio == pytest.approx(0.25)
    assert options.ransac.confidence == pytest.approx(0.999)
    assert options.ransac.max_num_trials == 10_000
    assert options.compute_relative_pose is False
    assert options.multiple_models is False
    assert options.use_sampson_refinement is True
    assert options.detect_watermark is True

    # The nested RANSACOptions inside TwoViewGeometryOptions is NOT the same as a
    # bare RANSACOptions(); copying defaults from the wrong one changes the
    # trial budget by an order of magnitude.
    bare: pycolmap.RANSACOptions = pycolmap.RANSACOptions()
    assert bare.confidence == pytest.approx(0.9999)
    assert bare.min_inlier_ratio == pytest.approx(0.01)
    assert bare.min_num_trials == 1_000
    assert bare.max_num_trials == 100_000
