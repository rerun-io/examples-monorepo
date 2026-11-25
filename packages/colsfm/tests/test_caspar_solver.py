"""CASPAR as recipes/colmap-src patches it, against Ceres on small pinhole problems and one fisheye rig.

`caspar-pcg-beta-recurrence.patch` (beta_k = rho_k / rho_{k-1}) and `caspar-reduced-camera-system.patch` (Point blocks
eliminated inside the PCG, Q stopping rule, guards for exact convergence and breakdown, validated parameters). Each
problem is solved by CASPAR and by Ceres from the same start; the sum of squared reprojection errors must agree.
CASPAR's options go through `apply_caspar_options`, the production cast path.

The OPENCV_FISHEYE factor (`add-caspar-adapters*.patch`) projects through the off-axis angle of the ray,
atan2(|XY|, Z). The X/Z form it replaced jumps at Z = 0 (the point is mirrored to the far side of the image), so a step
that carried a rim observation across the image plane raised the cost by orders of magnitude; every LM trial was
rejected and the solve returned its start values. The rim-rig test below covers it.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64
from numpy import ndarray

from colsfm.ba_backend import apply_caspar_options
from colsfm.reconstruction import RIG_ID, camera_sensor_id

WIDTH_PX, HEIGHT_PX = 640, 480


@pytest.fixture(autouse=True)
def _needs_the_patched_caspar(caspar_enabled: bool) -> None:
    if not caspar_enabled:
        pytest.skip("pycolmap is built without CASPAR")


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


def _solve(
    reconstruction: pycolmap.Reconstruction,
    backend: str,
    constant_frames: list[int],
    caspar: dict[str, int | float | bool],
    *,
    camera_only: bool = False,
) -> None:
    """One bundle adjustment. `camera_only`: the points fixed, the shared focal length free (CASPAR's PinholeFocal)."""
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    for frame_id in constant_frames:
        config.set_constant_rig_from_world_pose(frame_id)
    if not camera_only:
        config.set_constant_cam_intrinsics(1)
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.print_summary = False
    options.refine_principal_point = options.refine_sensor_from_rig = False
    # CASPAR refuses a free focal length with a fixed distortion block; PINHOLE has none, so the two move together.
    options.refine_focal_length = options.refine_extra_params = camera_only
    options.refine_points3D = not camera_only
    if backend == "caspar":
        options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
        options.caspar.gpu_index = "-1"
        apply_caspar_options(options.caspar, caspar)
    else:
        for tolerance in ("function_tolerance", "gradient_tolerance", "parameter_tolerance"):
            setattr(options.ceres.solver_options, tolerance, 1e-12)
        options.ceres.solver_options.max_num_iterations = 200
    pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()


def _squared_error(reconstruction: pycolmap.Reconstruction, *, skip_behind: bool = False) -> float:
    """Sum of squared reprojection errors, px^2; NaN when any pose or point went non-finite.

    `skip_behind`: leave out observations whose point sits behind their camera (a fisheye rim point can end up there).
    """
    total: float = 0.0
    for image in reconstruction.images.values():
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        for point2D in image.points2D:
            if point2D.has_point3D():
                xyz: Float64[ndarray, " 3"] = reconstruction.points3D[point2D.point3D_id].xyz
                in_camera: Float64[ndarray, " 3"] = np.asarray(cam_from_world.rotation.matrix()) @ xyz + np.asarray(cam_from_world.translation)
                if skip_behind and in_camera[2] <= 0.0:
                    continue
                projected = camera.img_from_cam(in_camera[None])
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
    caspar: dict[str, int | float | bool] = (
        {"reduced_camera_system": True} if mode == "reduced" else {"reduced_camera_system": False, "pcg_iter_max": 500, "pcg_rel_error_exit": 1e-12}
    )
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
    ("caspar", "message"),
    [
        ({"reduced_pcg_iter_max": 0}, "Check failed"),
        ({"reduced_pcg_eta": float("nan")}, "Check failed"),
        ({"reduced_pcg_eta": 1.0}, "Check failed"),
        ({"pcg_rel_score_exit": 0.5}, "Check failed"),
        # Passes the adapter's double check and rounds to 1.0f: the solver's own f32 check refuses it.
        ({"reduced_pcg_eta": 0.99999999}, r"reduced_pcg_eta in \(0, 1\)"),
    ],
    ids=["zero-cap", "nan-eta", "eta-one", "score-exit-with-reduced", "eta-rounds-to-one"],
)
def test_the_solver_refuses_invalid_reduced_parameters(caspar: dict[str, int | float | bool], message: str) -> None:
    """The native boundary checks what colsfm's parser also checks, so a direct pycolmap user gets the same refusal."""
    with pytest.raises(ValueError, match=message):
        _solve(_problem(3), "caspar", [1], {"reduced_camera_system": True, **caspar})


def _pcg_iterations(capfd: pytest.CaptureFixture[str], reconstruction: pycolmap.Reconstruction, caspar: dict[str, int | float | bool]) -> list[int]:
    """Solve camera-only on CASPAR and return the PCG iteration index of each LM step, from the solver's progress lines."""
    level: int = pycolmap.logging.verbose_level
    pycolmap.logging.verbose_level = 2  # VLOG(2) turns on CASPAR's per-step progress print
    try:
        capfd.readouterr()
        _solve(reconstruction, "caspar", [], caspar, camera_only=True)
        ctypes.CDLL(None).fflush(None)  # the solver prints with printf, block-buffered into the capture file
        out: str = capfd.readouterr().out
    finally:
        pycolmap.logging.verbose_level = level
    return [int(line.split("pcg_iter:")[1].split()[0]) for line in out.splitlines() if "pcg_iter:" in line]


def test_a_camera_only_solve_keeps_the_reduced_pcg_policy(capfd: pytest.CaptureFixture[str]) -> None:
    """Fixed points, free poses, a free shared focal length: the requested reduced mode sets the PCG's cap and stop rule.

    COLMAP's adapter keeps constant points as (factor-free) Point nodes, so this exercises the elimination path; the
    solver keeps the same policy with no Point node at all, which only a direct GraphSolver caller can reach."""
    one_step: dict[str, int | float | bool] = {"solver_iter_max": 1}
    capped: pycolmap.Reconstruction = _problem(8)
    capped_iterations: list[int] = _pcg_iterations(capfd, capped, {**one_step, "reduced_camera_system": True, "reduced_pcg_iter_max": 1})
    uncapped: pycolmap.Reconstruction = _problem(8)
    uncapped_iterations: list[int] = _pcg_iterations(capfd, uncapped, {**one_step, "reduced_camera_system": True, "reduced_pcg_eta": 1e-6})
    full_one: pycolmap.Reconstruction = _problem(8)
    _pcg_iterations(capfd, full_one, {**one_step, "reduced_camera_system": False, "pcg_iter_max": 1})
    assert capped_iterations == [0], f"cap 1 ran PCG iterations {capped_iterations}"
    assert uncapped_iterations and uncapped_iterations[0] >= 3, f"uncapped: {uncapped_iterations}; the case must need several iterations"
    start: float = _squared_error(_problem(8))
    capped_error, uncapped_error, full_error = _squared_error(capped), _squared_error(uncapped), _squared_error(full_one)
    # One PCG iteration without points is the full system's first iteration: the same step.
    assert capped_error == pytest.approx(full_error, rel=1e-5)
    assert uncapped_error < capped_error < start, f"start {start:.3f}, cap 1 {capped_error:.3f}, uncapped {uncapped_error:.3f} px^2"


RIM_SIZE_PX: int = 1000
RIM_FOCAL_PX: float = 294.0
RIM_BASELINE_M: float = 0.25
RIM_NUM_FRAMES: int = 6
RIM_NUM_POINTS: int = 400


def _rim_rig(seed: int = 0) -> tuple[pycolmap.Reconstruction, dict[int, Float64[ndarray, " 3"]]]:
    """A two-camera OPENCV_FISHEYE rig moving forward past points out to 89.5 deg off-axis, then knocked off.

    Returns the perturbed reconstruction and each frame's true rig centre.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    for camera_id in (1, 2):
        reconstruction.add_camera(pycolmap.Camera.create_from_model_id(camera_id, pycolmap.CameraModelId.OPENCV_FISHEYE, RIM_FOCAL_PX, RIM_SIZE_PX, RIM_SIZE_PX))
    rig.add_ref_sensor(camera_sensor_id(1))
    rig.add_sensor(camera_sensor_id(2), pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([-RIM_BASELINE_M, 0.0, 0.0])))
    reconstruction.add_rig(rig)
    truth: dict[int, Float64[ndarray, " 3"]] = {}
    for index in range(RIM_NUM_FRAMES):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = index + 1
        frame.rig_id = RIG_ID
        rig_from_world: pycolmap.Rigid3d = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.0, 0.01 * index, 0.0])), np.array([0.0, 0.0, -0.3 * index]))
        frame.rig_from_world = rig_from_world
        truth[frame.frame_id] = np.asarray(rig_from_world.inverse().translation)
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
        np.concatenate([rng.uniform(0.0, 87.5, 3 * RIM_NUM_POINTS // 4), rng.uniform(87.5, 89.5, RIM_NUM_POINTS // 4)])
    )
    azimuth: Float64[ndarray, " n"] = rng.uniform(0.0, 2.0 * np.pi, RIM_NUM_POINTS)
    distance: Float64[ndarray, " n"] = rng.uniform(3.0, 8.0, RIM_NUM_POINTS)
    rays: Float64[ndarray, "n 3"] = np.column_stack([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])
    points: Float64[ndarray, "n 3"] = rays * distance[:, None] + np.array([0.0, 0.0, 0.15 * RIM_NUM_FRAMES])
    observations: dict[int, list[tuple[int, int]]] = {index: [] for index in range(RIM_NUM_POINTS)}
    for image_id in sorted(reconstruction.images):
        image = reconstruction.image(image_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        in_camera: Float64[ndarray, "n 3"] = points @ np.asarray(cam_from_world.rotation.matrix()).T + np.asarray(cam_from_world.translation)
        seen: list[int] = [index for index in range(RIM_NUM_POINTS) if in_camera[index, 2] > 0.0]
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
    start_error: float = _squared_error(start, skip_behind=True)
    reference, _ = _rim_rig()
    _solve(reference, "ceres", [1], {})
    reconstruction, _ = _rim_rig()
    _solve(reconstruction, "caspar", [1], {"reduced_camera_system": True})
    error, expected = _squared_error(reconstruction, skip_behind=True), _squared_error(reference, skip_behind=True)
    # f32 kernels: 4e-4 above Ceres measured (the stock factor stays at the start, 23x above).
    assert error <= expected * (1.0 + 1e-3), f"CASPAR {error:.3f} px^2, Ceres {expected:.3f}, start {start_error:.3f}"
    assert _centre_error_m(reconstruction, truth) < 0.5 * _centre_error_m(start, truth)
