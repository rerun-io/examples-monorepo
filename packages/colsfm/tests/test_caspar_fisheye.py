"""CASPAR's OPENCV_FISHEYE factor near 90 deg off-axis, against Ceres, as recipes/colmap-src patches it.

The factor projects through the off-axis angle of the ray, atan2(|XY|, Z). The X/Z form it replaced jumps at Z = 0
(the point is mirrored to the far side of the image), so a step that carried a rim observation across the image
plane raised the cost by orders of magnitude; every LM trial was rejected and the solve returned its start values.
"""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64
from numpy import ndarray

from colsfm.ba_backend import apply_caspar_options
from colsfm.reconstruction import RIG_ID, camera_sensor_id

SIZE_PX: int = 1000
FOCAL_PX: float = 294.0
BASELINE_M: float = 0.25
NUM_FRAMES: int = 6
NUM_POINTS: int = 400


@pytest.fixture(autouse=True)
def _needs_the_patched_caspar(caspar_enabled: bool) -> None:
    if not caspar_enabled:
        pytest.skip("pycolmap is built without CASPAR")


def _rim_rig(seed: int = 0) -> tuple[pycolmap.Reconstruction, dict[int, Float64[ndarray, " 3"]]]:
    """A two-camera OPENCV_FISHEYE rig moving forward past points out to 89.5 deg off-axis, then knocked off.

    Returns the perturbed reconstruction and each frame's true rig centre.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    for camera_id in (1, 2):
        reconstruction.add_camera(pycolmap.Camera.create_from_model_id(camera_id, pycolmap.CameraModelId.OPENCV_FISHEYE, FOCAL_PX, SIZE_PX, SIZE_PX))
    rig.add_ref_sensor(camera_sensor_id(1))
    rig.add_sensor(camera_sensor_id(2), pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([-BASELINE_M, 0.0, 0.0])))
    reconstruction.add_rig(rig)
    truth: dict[int, Float64[ndarray, " 3"]] = {}
    for index in range(NUM_FRAMES):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = index + 1
        frame.rig_id = RIG_ID
        frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.0, 0.01 * index, 0.0])), np.array([0.0, 0.0, -0.3 * index]))
        truth[frame.frame_id] = np.asarray(frame.rig_from_world.inverse().translation)
        for camera_id in (1, 2):
            frame.add_data_id(pycolmap.data_t(camera_sensor_id(camera_id), 2 * index + camera_id))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        for camera_id in (1, 2):
            image: pycolmap.Image = pycolmap.Image(name=f"{camera_id}/{index}.png", camera_id=camera_id, image_id=2 * index + camera_id)
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)
    # Rays from the middle of the path, off-axis angles up to 89.5 deg: a quarter of them within 2 deg of the plane.
    theta: Float64[ndarray, " n"] = np.radians(
        np.concatenate([rng.uniform(0.0, 87.5, 3 * NUM_POINTS // 4), rng.uniform(87.5, 89.5, NUM_POINTS // 4)])
    )
    azimuth: Float64[ndarray, " n"] = rng.uniform(0.0, 2.0 * np.pi, NUM_POINTS)
    distance: Float64[ndarray, " n"] = rng.uniform(3.0, 8.0, NUM_POINTS)
    rays: Float64[ndarray, "n 3"] = np.column_stack([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])
    points: Float64[ndarray, "n 3"] = rays * distance[:, None] + np.array([0.0, 0.0, 0.15 * NUM_FRAMES])
    observations: dict[int, list[tuple[int, int]]] = {index: [] for index in range(NUM_POINTS)}
    for image_id in sorted(reconstruction.images):
        image = reconstruction.image(image_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        in_camera: Float64[ndarray, "n 3"] = points @ np.asarray(cam_from_world.rotation.matrix()).T + np.asarray(cam_from_world.translation)
        seen: list[int] = [index for index in range(NUM_POINTS) if in_camera[index, 2] > 0.0]
        pixels = np.asarray(reconstruction.camera(image.camera_id).img_from_cam(in_camera[seen])) + rng.normal(0.0, 0.5, (len(seen), 2))
        image.points2D = pycolmap.Point2DList([pycolmap.Point2D(pixel) for pixel in pixels])
        for point2D_index, index in enumerate(seen):
            observations[index].append((image_id, point2D_index))
    for index, observed in observations.items():
        if len(observed) >= 2:
            track: pycolmap.Track = pycolmap.Track()
            for image_id, point2D_index in observed:
                track.add_element(image_id, point2D_index)
            reconstruction.add_point3D(points[index] + rng.normal(0.0, 0.02, 3), track)
    for frame_id in sorted(reconstruction.frames)[1:]:
        frame = reconstruction.frame(frame_id)
        start: pycolmap.Rigid3d | None = frame.rig_from_world
        assert start is not None
        knock: pycolmap.Rotation3d = pycolmap.Rotation3d(rng.normal(0.0, np.radians(0.2), 3))
        frame.rig_from_world = pycolmap.Rigid3d(knock * start.rotation, np.asarray(start.translation) + rng.normal(0.0, 0.02, 3))
    return reconstruction, truth


def _solve(reconstruction: pycolmap.Reconstruction, backend: str) -> None:
    """colsfm's mapping solve: intrinsics and sensor_from_rig fixed, rig poses and points free, frame 1 the gauge."""
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(1)
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.print_summary = False
    options.refine_focal_length = options.refine_extra_params = options.refine_principal_point = False
    options.refine_sensor_from_rig = False
    if backend == "caspar":
        options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
        options.caspar.gpu_index = "-1"
        apply_caspar_options(options.caspar, {"reduced_camera_system": True})
    else:
        for tolerance in ("function_tolerance", "gradient_tolerance", "parameter_tolerance"):
            setattr(options.ceres.solver_options, tolerance, 1e-12)
        options.ceres.solver_options.max_num_iterations = 200
    pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()


def _squared_error(reconstruction: pycolmap.Reconstruction) -> float:
    """Sum of squared reprojection errors, px^2, over the observations in front of their camera."""
    total: float = 0.0
    for image in reconstruction.images.values():
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        for point2D in image.points2D:
            if point2D.has_point3D():
                xyz: Float64[ndarray, " 3"] = np.asarray(cam_from_world * reconstruction.points3D[point2D.point3D_id].xyz)
                projected = camera.img_from_cam(xyz)
                if projected is not None:
                    total += float(((np.asarray(projected) - point2D.xy) ** 2).sum())
    return total


def _centre_error_m(reconstruction: pycolmap.Reconstruction, truth: dict[int, Float64[ndarray, " 3"]]) -> float:
    """Largest rig-centre distance to the truth, metres."""
    return max(
        float(np.linalg.norm(np.asarray(frame.rig_from_world.inverse().translation) - truth[frame_id]))
        for frame_id, frame in reconstruction.frames.items()
    )


def test_caspar_moves_a_fisheye_rig_with_rim_observations_to_the_ceres_minimum() -> None:
    """From the same knocked-off start as Ceres, CASPAR ends at Ceres' cost and brings the rig back to the truth.

    Stock (X/Z) factor: every LM trial uphill, the start returned unchanged (cost and centre error as knocked off).
    """
    start, truth = _rim_rig()
    start_error: float = _squared_error(start)
    reference, _ = _rim_rig()
    _solve(reference, "ceres")
    reconstruction, _ = _rim_rig()
    _solve(reconstruction, "caspar")
    error, expected = _squared_error(reconstruction), _squared_error(reference)
    # f32 kernels: 4e-4 above Ceres measured (the stock factor stays at the start, 23x above).
    assert error <= expected * (1.0 + 1e-3), f"CASPAR {error:.3f} px^2, Ceres {expected:.3f}, start {start_error:.3f}"
    assert _centre_error_m(reconstruction, truth) < 0.5 * _centre_error_m(start, truth)
