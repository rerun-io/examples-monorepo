"""CASPAR's PCG as recipes/colmap-src patches it, against Ceres on small pinhole problems.

`caspar-pcg-beta-recurrence.patch` (beta_k = rho_k / rho_{k-1}) and `caspar-reduced-camera-system.patch` (Point blocks
eliminated inside the PCG, Q stopping rule, guards for exact convergence and breakdown, validated parameters). Each
problem is solved by CASPAR and by Ceres from the same start; the sum of squared reprojection errors must agree.
CASPAR's options go through `apply_caspar_options`, the production cast path.

The rig factor (`caspar-rig-refine`, free `sensor_from_rig`) is checked the same way on a synthetic two-camera rig,
on builds whose probe (`CasparCapability.refines_sensor_from_rig`) says they have it.
"""

from __future__ import annotations

import copy
import ctypes
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64
from numpy import ndarray

from colsfm.ba_backend import apply_caspar_options, detect_caspar_capability
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


# ══════════════════════════════════════════════════════════════════════════════════════
# the rig factor (recipes/colmap-src caspar-rig-refine): sensor_from_rig on CASPAR against Ceres
# ══════════════════════════════════════════════════════════════════════════════════════
# Ported from the Mission office scratch (caspar-rig/tests/test_rig_synthetic.py and rig_compare.py, named in
# recipes/colmap-src/patches/PROVENANCE.md). Skipped on a build whose probe does not refine a rig.

RigIntrinsics: TypeAlias = Literal["none", "fae", "all"]
"""What a rig solve frees besides the rig: nothing, focal and distortion block (`fae`), or those and the principal point."""

RIG_CAMERA_PARAMS: Final[dict[str, list[float]]] = {
    "PINHOLE": [800.0, 780.0, 512.0, 384.0],
    "SIMPLE_RADIAL": [800.0, 512.0, 384.0, 0.05],
    "OPENCV_FISHEYE": [500.0, 505.0, 512.0, 384.0, 0.05, -0.01, 0.002, -0.0005],
}
"""Camera parameters of the synthetic rig, per COLMAP model."""


@dataclass(frozen=True, slots=True)
class RigDiff:
    """How far two reconstructions' rigs and frame poses are apart, the second aligned to the first by Sim(3)."""

    rig_rot_deg: float
    """Largest rotation between the non-reference `sensor_from_rig`s."""
    rig_trans: float
    """Largest distance between them, after scaling by the alignment."""
    pose_rot_deg_max: float
    """Largest rotation between the aligned `rig_from_world`s."""
    pose_center_max: float
    """Largest distance between the aligned rig centres."""


@pytest.fixture
def _needs_the_rig_factor() -> None:
    if not detect_caspar_capability().refines_sensor_from_rig:
        pytest.skip("this CASPAR build does not refine sensor_from_rig (_caspar_refines_sensor_from_rig); build recipes/colmap-src caspar-rig-refine")


def _rig_problem(model: str, seed: int = 1) -> tuple[pycolmap.Reconstruction, pycolmap.Reconstruction]:
    """A 2-camera rig, 12 frames, 400 points: the truth, and a start with noise and the rig knocked 2 deg / 3 cm off."""
    pycolmap.set_random_seed(seed)
    options: pycolmap.SyntheticDatasetOptions = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 2
    options.num_frames_per_rig = 12
    options.num_points3D = 400
    options.sensor_from_rig_rotation_stddev = 30.0
    options.sensor_from_rig_translation_stddev = 0.3
    options.camera_model_id = getattr(pycolmap.CameraModelId, model)
    options.camera_params = RIG_CAMERA_PARAMS[model]
    truth: pycolmap.Reconstruction = pycolmap.synthesize_dataset(options)
    start: pycolmap.Reconstruction = copy.deepcopy(truth)
    noise: pycolmap.SyntheticNoiseOptions = pycolmap.SyntheticNoiseOptions()
    noise.point2D_stddev = 0.5
    noise.point3D_stddev = 0.05
    noise.rig_from_world_rotation_stddev = 0.3
    noise.rig_from_world_translation_stddev = 0.05
    pycolmap.synthesize_noise(noise, start)
    rng: np.random.Generator = np.random.default_rng(seed)
    for rig in start.rigs.values():
        for sensor in list(rig.non_ref_sensors):
            sensor_from_rig: pycolmap.Rigid3d = _known(rig.sensor_from_rig(sensor))
            axis: Float64[ndarray, " 3"] = rng.normal(size=3)
            shift: Float64[ndarray, " 3"] = rng.normal(size=3)
            turn: pycolmap.Rotation3d = pycolmap.Rotation3d(axis / np.linalg.norm(axis) * np.radians(2.0))
            rig.set_sensor_from_rig(
                sensor, pycolmap.Rigid3d(turn * sensor_from_rig.rotation, sensor_from_rig.translation + 0.03 * shift / np.linalg.norm(shift))
            )
    return truth, start


def _rig_options(backend: str, intrinsics: RigIntrinsics, **caspar: int | float | bool) -> pycolmap.BundleAdjustmentOptions:
    """Rig, poses and points free on either backend; Ceres on the TRIVIAL loss, the L2 cost CASPAR minimises."""
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.refine_rig_from_world = options.refine_sensor_from_rig = options.refine_points3D = True
    options.refine_focal_length = options.refine_extra_params = intrinsics in ("fae", "all")
    options.refine_principal_point = intrinsics == "all"
    options.print_summary = False
    if backend == "ceres":
        options.ceres.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
        options.ceres.solver_options.max_num_iterations = 100
        options.ceres.solver_options.function_tolerance = 1e-10
        options.ceres.solver_options.gradient_tolerance = 1e-12
        options.ceres.solver_options.parameter_tolerance = 1e-10
    else:
        options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
        options.caspar.gpu_index = "-1"
        apply_caspar_options(options.caspar, caspar)
    return options


def _rig_config(reconstruction: pycolmap.Reconstruction) -> pycolmap.BundleAdjustmentConfig:
    """Every registered image, gauge on three points (each solver then sits in its own gauge; `_rig_diff` aligns)."""
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.add_image(image_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
    return config


def _rig_solve(start: pycolmap.Reconstruction, options: pycolmap.BundleAdjustmentOptions) -> tuple[pycolmap.Reconstruction, pycolmap.BundleAdjustmentSummary]:
    """Solve a copy of `start`."""
    solved: pycolmap.Reconstruction = copy.deepcopy(start)
    summary: pycolmap.BundleAdjustmentSummary = pycolmap.create_default_bundle_adjuster(options, _rig_config(solved), solved).solve()
    return solved, summary


def _known(pose: pycolmap.Rigid3d | None) -> pycolmap.Rigid3d:
    """A pose the synthetic rig always has (pycolmap types frame and sensor poses as optional)."""
    assert pose is not None
    return pose


def _rotation_deg(a: Float64[ndarray, "3 3"], b: Float64[ndarray, "3 3"]) -> float:
    """Angle of `a @ b.T`, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0))))


def _rig_centres(reconstruction: pycolmap.Reconstruction, frame_ids: list[int]) -> Float64[ndarray, "n 3"]:
    """World positions of the given frames' rigs."""
    return np.array([np.asarray(_known(reconstruction.frames[frame_id].rig_from_world).inverse().translation) for frame_id in frame_ids])


def _rig_diff(a: pycolmap.Reconstruction, b: pycolmap.Reconstruction) -> RigDiff:
    """Compare `b` with `a` after aligning `b`'s rig centres onto `a`'s by Sim(3) (Umeyama); rig rotations are gauge-free."""
    frame_ids: list[int] = sorted(frame_id for frame_id, frame in a.frames.items() if frame.has_pose and b.frames[frame_id].has_pose)
    centres_a: Float64[ndarray, "n 3"] = _rig_centres(a, frame_ids)
    centres_b: Float64[ndarray, "n 3"] = _rig_centres(b, frame_ids)
    mean_a: Float64[ndarray, " 3"] = centres_a.mean(axis=0)
    mean_b: Float64[ndarray, " 3"] = centres_b.mean(axis=0)
    u, singular, vt = np.linalg.svd((centres_a - mean_a).T @ (centres_b - mean_b) / len(frame_ids))
    sign: Float64[ndarray, "3 3"] = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    a_R_b: Float64[ndarray, "3 3"] = u @ sign @ vt
    scale: float = float(np.trace(np.diag(singular) @ sign) / ((centres_b - mean_b) ** 2).sum(axis=1).mean())
    aligned_b: Float64[ndarray, "n 3"] = scale * centres_b @ a_R_b.T + (mean_a - scale * a_R_b @ mean_b)
    rig_rot_deg, rig_trans = 0.0, 0.0
    for rig_id, rig_a in a.rigs.items():
        for sensor in rig_a.non_ref_sensors:
            sensor_a: pycolmap.Rigid3d = _known(rig_a.sensor_from_rig(sensor))
            sensor_b: pycolmap.Rigid3d = _known(b.rigs[rig_id].sensor_from_rig(sensor))
            rig_rot_deg = max(rig_rot_deg, _rotation_deg(sensor_a.rotation.matrix(), sensor_b.rotation.matrix()))
            rig_trans = max(rig_trans, float(np.linalg.norm(np.asarray(sensor_a.translation) - scale * np.asarray(sensor_b.translation))))
    pose_rot_deg: list[float] = [
        _rotation_deg(_known(a.frames[frame_id].rig_from_world).rotation.matrix(), _known(b.frames[frame_id].rig_from_world).rotation.matrix() @ a_R_b.T)
        for frame_id in frame_ids
    ]
    return RigDiff(
        rig_rot_deg=rig_rot_deg,
        rig_trans=rig_trans,
        pose_rot_deg_max=max(pose_rot_deg),
        pose_center_max=float(np.linalg.norm(centres_a - aligned_b, axis=1).max()),
    )


def _l2_cost(reconstruction: pycolmap.Reconstruction) -> float:
    """0.5 * the sum of squared reprojection errors over the observations in front of their camera, px^2."""
    squared: list[float] = []
    for point in reconstruction.points3D.values():
        for element in point.track.elements:
            image: pycolmap.Image = reconstruction.images[element.image_id]
            projected = reconstruction.cameras[image.camera_id].img_from_cam(image.cam_from_world() * point.xyz)
            if projected is not None:
                squared.append(float(np.sum((np.asarray(projected) - image.points2D[element.point2D_idx].xy) ** 2)))
    return 0.5 * float(np.sum(squared))


@pytest.mark.usefixtures("_needs_the_rig_factor")
@pytest.mark.parametrize(
    ("model", "intrinsics"),
    [
        ("PINHOLE", "none"),
        ("PINHOLE", "fae"),
        ("PINHOLE", "all"),
        ("OPENCV_FISHEYE", "none"),
        ("OPENCV_FISHEYE", "fae"),
        ("OPENCV_FISHEYE", "all"),
        ("SIMPLE_RADIAL", "fae"),
    ],
)
def test_caspar_recovers_a_perturbed_rig_where_ceres_does(model: str, intrinsics: RigIntrinsics) -> None:
    """From the same start, CASPAR (full PCG and the reduced camera system) reaches Ceres' cost and Ceres' rig.

    With every intrinsic of a fisheye free, a principal-point shift and a small rotation trade almost freely (~1 px per
    0.15 deg); f32 CASPAR stops a hair above Ceres' cost along that mode. So every CASPAR run reaches Ceres' cost
    (within 1e-4), the reduced camera system (colsfm's mode) matches the rig to 0.01 deg (0.05 deg on that weak mode),
    and the full PCG's rig is checked where the principal point is held. The full PCG's stop along the weak mode varies
    from run to run (1.2e-4 to 1.6e-4 above Ceres over 8 runs on an RTX 5090, on two builds), so its cost bound there
    is 5e-4.
    """
    truth, start = _rig_problem(model)
    start_rot_deg: float = _rig_diff(truth, start).rig_rot_deg
    ceres, _ = _rig_solve(start, _rig_options("ceres", intrinsics))
    ceres_cost: float = _l2_cost(ceres)
    weak: bool = model == "OPENCV_FISHEYE" and intrinsics == "all"
    for mode, reduced in (("full", False), ("reduced", True)):
        caspar, summary = _rig_solve(start, _rig_options("caspar", intrinsics, reduced_camera_system=reduced))
        assert summary.termination_type != pycolmap.BundleAdjustmentTerminationType.FAILURE, mode
        cost_tolerance: float = 5e-4 if weak and not reduced else 1e-4
        assert _l2_cost(caspar) <= ceres_cost * (1.0 + cost_tolerance), (mode, _l2_cost(caspar), ceres_cost)
        if not reduced and intrinsics == "all":
            continue
        diff: RigDiff = _rig_diff(ceres, caspar)
        tolerance_deg: float = 0.05 if weak else 0.01
        assert diff.rig_rot_deg < tolerance_deg and diff.rig_trans < 1e-3, (mode, diff)
        assert diff.pose_rot_deg_max < tolerance_deg and diff.pose_center_max < 1e-3, (mode, diff)
        if reduced:
            assert _rig_diff(truth, caspar).rig_rot_deg < 0.1 * start_rot_deg, "the perturbation is undone"


@pytest.mark.usefixtures("_needs_the_rig_factor")
def test_a_sensor_held_constant_stays_where_it_was() -> None:
    """A sensor set constant in the config keeps its `sensor_from_rig` on CASPAR while the rig is otherwise free."""
    _, start = _rig_problem("PINHOLE")
    config: pycolmap.BundleAdjustmentConfig = _rig_config(start)
    rig: pycolmap.Rig = start.rigs[next(iter(start.rigs))]
    sensor: pycolmap.sensor_t = next(iter(rig.non_ref_sensors))
    config.set_constant_sensor_from_rig_pose(sensor)
    before: pycolmap.Rigid3d = _known(rig.sensor_from_rig(sensor))
    pycolmap.create_default_bundle_adjuster(_rig_options("caspar", "none"), config, start).solve()
    after: pycolmap.Rigid3d = _known(start.rigs[rig.rig_id].sensor_from_rig(sensor))
    assert np.allclose(before.translation, after.translation)
    assert np.allclose(before.rotation.quat, after.rotation.quat)


@pytest.mark.usefixtures("_needs_the_rig_factor")
def test_a_free_rig_with_only_the_principal_point_free_is_refused() -> None:
    """The generated factors have no principal-point-only block for a free rig: the build fails loudly, not silently."""
    _, start = _rig_problem("PINHOLE")
    options: pycolmap.BundleAdjustmentOptions = _rig_options("caspar", "none")
    options.refine_principal_point = True
    with pytest.raises((ValueError, RuntimeError), match="principal point"):
        pycolmap.create_default_bundle_adjuster(options, _rig_config(start), start)


@pytest.mark.usefixtures("_needs_the_rig_factor")
def test_a_rig_that_mixes_camera_models_is_refused() -> None:
    """Two lenses on two camera models would get unlinked per-model poses, so a free rig there is refused."""
    _, start = _rig_problem("PINHOLE")
    camera: pycolmap.Camera = start.cameras[2]
    fx, _, cx, cy = (float(value) for value in camera.params)
    camera.model = pycolmap.CameraModelId.SIMPLE_RADIAL
    camera.params = [fx, cx, cy, 0.0]
    with pytest.raises((ValueError, RuntimeError), match="mixes camera models"):
        pycolmap.create_default_bundle_adjuster(_rig_options("caspar", "fae"), _rig_config(start), start)
