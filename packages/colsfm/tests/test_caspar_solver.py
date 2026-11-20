"""CASPAR's PCG as recipes/colmap-src patches it, against Ceres on small pinhole problems.

`caspar-pcg-beta-recurrence.patch` (beta_k = rho_k / rho_{k-1}) and `caspar-reduced-camera-system.patch` (Point blocks
eliminated inside the PCG, Q stopping rule, guards for exact convergence and breakdown, validated parameters). Each
problem is solved by CASPAR and by Ceres from the same start; the sum of squared reprojection errors must agree.
"""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64
from numpy import ndarray

from colsfm.reconstruction import RIG_ID, camera_sensor_id

WIDTH_PX, HEIGHT_PX = 640, 480


@pytest.fixture(autouse=True)
def _needs_the_patched_caspar(caspar_enabled: bool) -> None:
    if not caspar_enabled:
        pytest.skip("pycolmap is built without CASPAR")
    if not hasattr(pycolmap.BundleAdjustmentOptions().caspar, "reduced_camera_system"):
        pytest.skip("this pycolmap predates recipes/colmap-src caspar-reduced-camera-system.patch")


def _problem(num_frames: int, *, noise_px: float = 0.5, perturb: bool = True, seed: int = 0) -> pycolmap.Reconstruction:
    """One pinhole camera translating along x, 200 points; noisy observations, points and non-gauge poses knocked off."""
    rng: np.random.Generator = np.random.default_rng(seed)
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(pycolmap.Camera.create_from_model_id(1, pycolmap.CameraModelId.PINHOLE, 500.0, WIDTH_PX, HEIGHT_PX))
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(camera_sensor_id(1))
    reconstruction.add_rig(rig)
    for index in range(num_frames):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = index + 1
        frame.rig_id = RIG_ID
        frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.0, 0.02 * index, 0.0])), np.array([-0.25 * index, 0.0, 0.0]))
        frame.add_data_id(pycolmap.data_t(camera_sensor_id(1), index + 1))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        image: pycolmap.Image = pycolmap.Image(name=f"i{index}.png", camera_id=1, image_id=index + 1)
        image.frame_id = frame.frame_id
        reconstruction.add_image(image)
    points: Float64[ndarray, "200 3"] = np.column_stack(
        [rng.uniform(-2.0, 2.0 + 0.25 * num_frames, 200), rng.uniform(-1.5, 1.5, 200), rng.uniform(4.0, 8.0, 200)]
    )
    camera: pycolmap.Camera = reconstruction.camera(1)
    observations: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(points))}
    for image_id in sorted(reconstruction.images):
        image = reconstruction.image(image_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        in_camera: Float64[ndarray, "200 3"] = points @ np.asarray(cam_from_world.rotation.matrix()).T + np.asarray(cam_from_world.translation)
        pixels: Float64[ndarray, "200 2"] = np.asarray(camera.img_from_cam(in_camera)) + rng.normal(0.0, noise_px, (len(points), 2))
        seen: list[int] = [
            index for index in range(len(points))
            if in_camera[index, 2] > 0.1 and (pixels[index] >= 0.0).all() and (pixels[index] < [WIDTH_PX, HEIGHT_PX]).all()
        ]
        image.points2D = pycolmap.Point2DList([pycolmap.Point2D(pixels[index]) for index in seen])
        for point2D_index, index in enumerate(seen):
            observations[index].append((image_id, point2D_index))
    for index, observed in observations.items():
        if len(observed) >= 2:
            track: pycolmap.Track = pycolmap.Track()
            for image_id, point2D_index in observed:
                track.add_element(image_id, point2D_index)
            reconstruction.add_point3D(points[index] + (rng.normal(0.0, 0.05, 3) if perturb else 0.0), track)
    for frame_id in sorted(reconstruction.frames)[1:] if perturb else []:
        frame = reconstruction.frame(frame_id)
        start: pycolmap.Rigid3d | None = frame.rig_from_world
        assert start is not None
        frame.rig_from_world = pycolmap.Rigid3d(start.rotation, np.asarray(start.translation) + rng.normal(0.0, 0.02, 3))
    return reconstruction


def _solve(reconstruction: pycolmap.Reconstruction, backend: str, constant_frames: list[int], caspar: dict[str, int | float | bool]) -> None:
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    for frame_id in constant_frames:
        config.set_constant_rig_from_world_pose(frame_id)
    config.set_constant_cam_intrinsics(1)
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.print_summary = False
    options.refine_focal_length = options.refine_extra_params = options.refine_principal_point = options.refine_sensor_from_rig = False
    if backend == "caspar":
        options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
        options.caspar.gpu_index = "-1"
        for name, value in caspar.items():
            setattr(options.caspar, name, value)
    else:
        for tolerance in ("function_tolerance", "gradient_tolerance", "parameter_tolerance"):
            setattr(options.ceres.solver_options, tolerance, 1e-12)
        options.ceres.solver_options.max_num_iterations = 200
    pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()


def _squared_error(reconstruction: pycolmap.Reconstruction) -> float:
    """Sum of squared reprojection errors, px^2; NaN when any pose or point went non-finite."""
    total: float = 0.0
    for image in reconstruction.images.values():
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        for point2D in image.points2D:
            if point2D.has_point3D():
                xyz: Float64[ndarray, " 3"] = reconstruction.points3D[point2D.point3D_id].xyz
                projected = camera.img_from_cam((np.asarray(cam_from_world.rotation.matrix()) @ xyz + np.asarray(cam_from_world.translation))[None])
                if projected is None:
                    return float("nan")
                total += float(((np.asarray(projected)[0] - point2D.xy) ** 2).sum())
    return total


CASES: dict[str, tuple[int, bool, list[int]]] = {
    # (frames, exact = no noise and no perturbation, constant frames)
    # Zero gradient: the reduced right-hand side is exactly zero, alpha_0 would be 0/0.
    "exact-gauge-only": (6, True, [1]),
    # No camera block at all: only the point back-substitution moves anything.
    "exact-point-only": (6, True, [1, 2, 3, 4, 5, 6]),
    "noisy-point-only": (6, False, [1, 2, 3, 4, 5, 6]),
    # One 6-dof camera block: the PCG converges in a handful of iterations, then rho_k reaches 0.
    "one-variable-frame": (3, False, [1, 2]),
    "eight-frames": (8, False, [1]),
}


@pytest.mark.parametrize("mode", ["reduced", "full"])
@pytest.mark.parametrize("case", list(CASES))
def test_caspar_reaches_the_ceres_minimum(case: str, mode: str) -> None:
    """Both PCG modes, from the same start as Ceres, end at Ceres' cost: finite, within 1e-6 relative + 1e-4 px^2."""
    num_frames, exact, constant = CASES[case]
    reference: pycolmap.Reconstruction = _problem(num_frames, noise_px=0.0 if exact else 0.5, perturb=not exact)
    _solve(reference, "ceres", constant, {})
    reconstruction: pycolmap.Reconstruction = _problem(num_frames, noise_px=0.0 if exact else 0.5, perturb=not exact)
    caspar: dict[str, int | float | bool] = {} if mode == "reduced" else {"reduced_camera_system": False, "pcg_iter_max": 500, "pcg_rel_error_exit": 1e-12}
    _solve(reconstruction, "caspar", constant, caspar)
    error, expected = _squared_error(reconstruction), _squared_error(reference)
    assert np.isfinite(error)
    assert error <= expected * (1.0 + 1e-6) + 1e-4, f"{case} {mode}: {error:.6f} px^2 vs Ceres {expected:.6f}"


def test_the_pcg_beta_recurrence_keeps_the_directions_conjugate() -> None:
    """A full-system PCG capped at 10 iterations per step, 10 steps: with beta_k = rho_k / rho_{k-1} it ends 2.0e-5 above
    Ceres' minimum; Symforce 0.12's beta_k = rho_k / rho_0 ends 8.4e-5 above it (measured on both builds)."""
    reference: pycolmap.Reconstruction = _problem(8)
    _solve(reference, "ceres", [1], {})
    reconstruction: pycolmap.Reconstruction = _problem(8)
    _solve(reconstruction, "caspar", [1], {"reduced_camera_system": False, "pcg_iter_max": 10, "pcg_rel_error_exit": 1e-12, "solver_iter_max": 10})
    excess: float = _squared_error(reconstruction) / _squared_error(reference) - 1.0
    assert excess < 4e-5, f"{excess:.3e} above Ceres: the PCG lost conjugacy"


@pytest.mark.parametrize(
    "caspar",
    [{"reduced_pcg_iter_max": 0}, {"reduced_pcg_eta": float("nan")}, {"reduced_pcg_eta": 1.0}, {"pcg_rel_score_exit": 0.5}],
    ids=["zero-cap", "nan-eta", "eta-one", "score-exit-with-reduced"],
)
def test_the_solver_refuses_invalid_reduced_parameters(caspar: dict[str, int | float | bool]) -> None:
    """The native boundary checks what colsfm's parser also checks, so a direct pycolmap user gets the same refusal."""
    with pytest.raises(ValueError, match="Check failed"):
        _solve(_problem(3), "caspar", [1], caspar)
