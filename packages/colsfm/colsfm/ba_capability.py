"""What this build's CASPAR and the Mojo solver can do, measured by solving: the capability half of `colsfm.ba_backend`.

There is no pycolmap call that lists CASPAR's camera adapters, and no Mojo call that lists the ported models, so the
only way to know is to hand each solver a sixteen-point problem per camera model and read the summary. One sweep
(`_measure`) asks both solvers the same questions in the same order, with a backend-specific solve:

* **projection**: CASPAR silently drops the observations of a model it has no adapter for and reports
  `num_residuals == 0`, so a residual count is the signal; the Mojo solver raises for a model it has not ported.
* **refinement**: one two-camera PINHOLE rig with free `sensor_from_rig`, and one solve per projected model with free
  intrinsics; a refinement is supported when the solve leaves a usable solution.

Every probe that raises, or that ran on residuals and still left no usable solution, is recorded in
`CasparCapability.probe_errors`, so a refused request can say why. The answer distinguishes *unavailable* (no CASPAR
build, or no Mojo extension) from *probe failure* (the solver is there but nothing solved) from *available*.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import cache
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64
from numpy import ndarray

from colsfm.reconstruction import RIG_ID, camera_sensor_id

CASPAR_STOCK_CAMERA_MODELS: Final[frozenset[str]] = frozenset({"PINHOLE", "SIMPLE_RADIAL"})
"""The two models stock CASPAR (COLMAP 4.2.0) projects, and the floor every build meets.

Observations of a model the build has no adapter for are dropped with a `LOG(WARNING)`
and the solve still reports success, so the pre-flight is a check, not a guard. Which
models a build actually carries is a property of that build — the `colsfm` environment (recipes/colmap-src)
adds OPENCV_FISHEYE and fixed-calibration FULL_OPENCV (`recipes/colmap-src/patches/PROVENANCE.md`) — so this constant is the
reference point, not the answer; `caspar_supported_camera_models` measures the answer."""

CASPAR_PROBE_CAMERA_MODELS: Final[tuple[str, ...]] = (
    "PINHOLE",
    "SIMPLE_RADIAL",
    "OPENCV_FISHEYE",
    "OPENCV",
    "FULL_OPENCV",
)
"""Camera models `caspar_supported_camera_models` asks the build about, in probe order.

The two stock adapters, the OPENCV_FISHEYE and FULL_OPENCV ones this repo adds, and OPENCV,
the obvious next port — probing them costs a few milliseconds each and means a
future adapter is picked up without editing this module."""

CASPAR_DISABLED_MARKER: Final[str] = "CASPAR_ENABLED"
"""What COLMAP's "built without CASPAR_ENABLED" `ValueError` says; the capability test."""

CASPAR_PROBE_FOCAL_LENGTH_PX: Final[float] = 300.0
"""Focal length of the probe camera; wide enough that a 640x480 image sees the whole grid."""

CASPAR_PROBE_IMAGE_SIZE_PX: Final[tuple[int, int]] = (640, 480)
"""Width and height of the probe camera, in pixels."""

CASPAR_PROBE_POINT_OFFSET_M: Final[float] = 0.02
"""How far the probe knocks its points off their exact positions, so the solve has work."""


def _caspar_probe_reconstruction(model_name: str, num_sensors: int = 1) -> pycolmap.Reconstruction | None:
    """A minimal two-frame model of one camera model, knocked off its own solution.

    One rig of `num_sensors` cameras side by side (one sensor: `refine_sensor_from_rig`
    never comes up); two frames, the first of which is the gauge; and a grid of points
    whose observations are exact projections, displaced afterwards so that a solver with
    anything to do has something to do. Small enough to build and solve in a few milliseconds.

    Args:
        model_name: A `pycolmap.CameraModelId` member name, e.g. `OPENCV_FISHEYE`.
        num_sensors: Cameras in the rig, 0.1 m apart along x; camera 1 is the reference.

    Returns:
        The model, or None when this pycolmap has no such camera model.
    """
    model: pycolmap.CameraModelId | None = getattr(pycolmap.CameraModelId, model_name, None)
    if model is None:
        return None
    width_px, height_px = CASPAR_PROBE_IMAGE_SIZE_PX
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    for camera_id in range(1, num_sensors + 1):
        reconstruction.add_camera(
            pycolmap.Camera.create_from_model_id(camera_id, model, CASPAR_PROBE_FOCAL_LENGTH_PX, width_px, height_px)
        )
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(camera_sensor_id(1))
    for camera_id in range(2, num_sensors + 1):
        rig.add_sensor(camera_sensor_id(camera_id), pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([-0.1 * (camera_id - 1), 0.0, 0.0])))
    reconstruction.add_rig(rig)

    baselines_m: tuple[float, float] = (0.0, -0.3)
    for frame_index, baseline_m in enumerate(baselines_m):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = frame_index + 1
        frame.rig_id = RIG_ID
        frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([baseline_m, 0.0, 0.0]))
        image_ids: list[int] = [frame_index * num_sensors + camera_id for camera_id in range(1, num_sensors + 1)]
        for camera_id, image_id in enumerate(image_ids, start=1):
            frame.add_data_id(pycolmap.data_t(camera_sensor_id(camera_id), image_id))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        for camera_id, image_id in enumerate(image_ids, start=1):
            image: pycolmap.Image = pycolmap.Image(name=f"probe{image_id}.png", camera_id=camera_id, image_id=image_id)
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)

    grid: Float64[ndarray, "n_points 3"] = np.array(
        [[x, y, z] for x in np.linspace(-1.0, 1.0, 4) for y in np.linspace(-0.8, 0.8, 4) for z in (3.0, 5.0)],
        dtype=np.float64,
    )
    observations: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(grid))}
    for image_id in sorted(reconstruction.images):
        image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        points_in_cam: Float64[ndarray, "n_points 3"] = (
            grid @ np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64).T
            + np.asarray(cam_from_world.translation, dtype=np.float64)
        )
        projected: Float64[ndarray, "n_points 2"] = np.asarray(camera.img_from_cam(points_in_cam), dtype=np.float64)
        visible: Bool[ndarray, " n_points"] = (
            np.isfinite(projected).all(axis=1)
            & (points_in_cam[:, 2] > 0.1)
            & (projected >= 0.0).all(axis=1)
            & (projected < np.array([width_px, height_px], dtype=np.float64)).all(axis=1)
        )
        image.points2D = pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in projected[visible]])
        for observation_index, point_index in enumerate(np.flatnonzero(visible)):
            observations[int(point_index)].append((image_id, observation_index))

    rng: np.random.Generator = np.random.default_rng(0)
    for point_index, point_xyz in enumerate(grid):
        elements: list[tuple[int, int]] = observations[point_index]
        # A one-view track is gauge-free and would tell the solver nothing.
        if len(elements) < 2:
            continue
        reconstruction.add_point3D(
            point_xyz + rng.normal(0.0, CASPAR_PROBE_POINT_OFFSET_M, 3),
            pycolmap.Track([pycolmap.TrackElement(image_id, index) for image_id, index in elements]),
            np.array([128, 128, 128], dtype=np.uint8),
        )
    return reconstruction


def _probe_problem(
    model_name: str,
    *,
    num_sensors: int = 1,
    refine_intrinsics: bool = False,
    refine_sensor_from_rig: bool = False,
    constant_intrinsics: bool = False,
) -> tuple[pycolmap.Reconstruction, pycolmap.BundleAdjustmentConfig, pycolmap.BundleAdjustmentOptions] | None:
    """The probe model with the config and options every capability probe solves, on both GPU backends.

    Args:
        model_name: A `pycolmap.CameraModelId` member name.
        num_sensors: Cameras in the probe rig (`_caspar_probe_reconstruction`).
        refine_intrinsics: Free the focal and the distortion block (CASPAR takes them together).
        refine_sensor_from_rig: Free the rig's `sensor_from_rig`.
        constant_intrinsics: Hold camera 1's intrinsics constant in the config as well.

    Returns:
        Model, config and CASPAR options, or None when this pycolmap has no such camera model.
    """
    reconstruction: pycolmap.Reconstruction | None = _caspar_probe_reconstruction(model_name, num_sensors=num_sensors)
    if reconstruction is None:
        return None
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(min(reconstruction.frames))
    if constant_intrinsics:
        config.set_constant_cam_intrinsics(1)

    ba_options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    ba_options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
    ba_options.print_summary = False
    ba_options.refine_sensor_from_rig = refine_sensor_from_rig
    ba_options.refine_focal_length = refine_intrinsics
    ba_options.refine_extra_params = refine_intrinsics
    ba_options.refine_principal_point = False
    ba_options.refine_rig_from_world = True
    ba_options.refine_points3D = True
    # `-1` lets COLMAP pick the device, which is what this module's `use_gpu=False`
    # means everywhere else; the probe asks about adapters, not about a GPU.
    ba_options.caspar.gpu_index = "-1"
    return reconstruction, config, ba_options


def _caspar_probe_solve(
    model_name: str,
    *,
    num_sensors: int = 1,
    refine_intrinsics: bool = False,
    refine_sensor_from_rig: bool = False,
    constant_intrinsics: bool = False,
) -> pycolmap.BundleAdjustmentSummary | None:
    """One CASPAR bundle adjustment of the probe model: the solve every CASPAR capability probe reads.

    Args:
        model_name: A `pycolmap.CameraModelId` member name.
        num_sensors: Cameras in the probe rig (`_caspar_probe_reconstruction`).
        refine_intrinsics: Free the focal and the distortion block (CASPAR takes them together).
        refine_sensor_from_rig: Free the rig's `sensor_from_rig`.
        constant_intrinsics: Hold camera 1's intrinsics constant in the config as well.

    Returns:
        The solve's summary, or None when this pycolmap has no such camera model.

    Raises:
        ValueError: When pycolmap is built without CASPAR_ENABLED, or CASPAR refuses the problem.
        RuntimeError: When CASPAR throws mid-solve.
    """
    problem = _probe_problem(
        model_name,
        num_sensors=num_sensors,
        refine_intrinsics=refine_intrinsics,
        refine_sensor_from_rig=refine_sensor_from_rig,
        constant_intrinsics=constant_intrinsics,
    )
    if problem is None:
        return None
    reconstruction, config, ba_options = problem
    return pycolmap.create_default_bundle_adjuster(ba_options, config, reconstruction).solve()


# ══════════════════════════════════════════════════════════════════════════════════════
# what this build can do
# ══════════════════════════════════════════════════════════════════════════════════════


CasparAvailability: TypeAlias = Literal["available", "unavailable", "probe_failed"]
"""The three answers `detect_caspar_capability` can give, which used to be one empty set.

* **`available`** — the build has CASPAR and it projected at least one probe model.
* **`unavailable`** — pycolmap was built without `CASPAR_ENABLED`, or the build has
  it and projected nothing at all. Either way there is no half-problem to protect
  against, so the camera guard has nothing to say and the solve-time fallback
  reports the missing build in the words that name the fix.
* **`probe_failed`** — the build has CASPAR, but every probe raised. That is a
  broken environment rather than a statement about camera models, so the guard
  again stays quiet and the solve decides.
"""


@dataclass(frozen=True, slots=True)
class CasparCapability:
    """What this pycolmap's CASPAR actually does, measured once per process."""

    availability: CasparAvailability
    """Whether CASPAR is usable at all, and how the probe found out."""
    supported_camera_models: frozenset[str]
    """`pycolmap.CameraModelId` member names whose observations CASPAR parameterises.

    Empty on `unavailable` and `probe_failed`; a subset of
    `CASPAR_PROBE_CAMERA_MODELS` on `available`."""
    probe_errors: tuple[str, ...]
    """One message per probe that raised, in probe order; empty when none did.

    Kept rather than swallowed: a build whose OPENCV_FISHEYE adapter throws is a
    different fact from one that has no adapter, and only this field says which."""
    refines_sensor_from_rig: bool = False
    """Whether this build's CASPAR refined `sensor_from_rig` on a two-sensor probe rig (`_measure`).

    Stock CASPAR throws on a multi-sensor frame with free extrinsics; recipes/colmap-src with the rig factor solves it.
    False keeps every solve that frees the rig on Ceres."""
    intrinsics_refinable_camera_models: frozenset[str] = frozenset()
    """The projected models whose focal (and distortion block) CASPAR also refines (`_measure`).

    A subset of `supported_camera_models`: recipes/colmap-src's FULL_OPENCV adapter projects with fixed intrinsics only
    and throws mid-run when they are freed. Empty when not measured, so a stated capability refines nothing on CASPAR."""

    def fixed_intrinsics_only(self, camera_model_names: Iterable[str]) -> frozenset[str]:
        """The models whose intrinsics this build's CASPAR cannot refine.

        Args:
            camera_model_names: `pycolmap.CameraModelId` member names.

        Returns:
            The names outside `intrinsics_refinable_camera_models`; empty when nothing was measured.
        """
        if not self.guards_camera_models:
            return frozenset()
        return frozenset(camera_model_names) - self.intrinsics_refinable_camera_models

    @property
    def guards_camera_models(self) -> bool:
        """Whether the pre-flight camera check has anything to check against.

        Returns:
            True only when the probe found real support, so a model outside
            `supported_camera_models` is genuinely one CASPAR would drop.
        """
        return self.availability == "available" and bool(self.supported_camera_models)

    def unsupported(self, camera_model_names: Iterable[str]) -> frozenset[str]:
        """The models among these that this build would silently drop.

        Args:
            camera_model_names: `pycolmap.CameraModelId` member names in a model.

        Returns:
            Those outside `supported_camera_models`, or an empty set when the probe
            has nothing to guard with.
        """
        if not self.guards_camera_models:
            return frozenset()
        return frozenset(name for name in camera_model_names if name not in self.supported_camera_models)


CASPAR_NOT_MEASURED: Final[CasparCapability] = CasparCapability(
    availability="unavailable", supported_camera_models=frozenset(), probe_errors=()
)
"""What a plan carries when nothing asked the build anything.

A `--ba-backend ceres` run never probes: the probe costs five GPU solves and no
decision in the plan would read the answer. `unavailable` is the honest value —
nothing has been shown to work — and it is unreachable, because every path that
reads a capability first checks that the backend is `caspar`."""


ProbeSolve: TypeAlias = Callable[..., "pycolmap.BundleAdjustmentSummary | None"]
"""`_caspar_probe_solve` or `_mojo_probe_solve`: one probe problem with the given freedoms, solved; None for a model
this pycolmap does not have."""

BeartypeException: tuple[type[Exception], ...] = (sys.modules["beartype.roar"].BeartypeException,) if "beartype.roar" in sys.modules else ()
"""Re-raised from the probes' `except`: a test-time type violation is a bug, not a capability."""


def _measure(
    probe_solve: ProbeSolve, refusals: tuple[type[Exception], ...], projects: Callable[[pycolmap.BundleAdjustmentSummary], bool]
) -> CasparCapability:
    """The probe sweep both backends share: projection per model, then the rig and the free intrinsics.

    Args:
        probe_solve: The backend's probe solve.
        refusals: What the backend raises when it refuses a probe; anything else propagates.
        projects: Whether a projection probe's summary shows the model is carried (CASPAR: residuals; Mojo: a usable
            solve, since the Mojo solver refuses by raising).

    Returns:
        The capability; `unavailable` as soon as an error names `CASPAR_DISABLED_MARKER`.
    """
    errors: list[str] = []

    def solved(label: str, model_name: str, **freedoms: int | bool) -> pycolmap.BundleAdjustmentSummary | None:
        try:
            summary: pycolmap.BundleAdjustmentSummary | None = probe_solve(model_name, **freedoms)
        except BeartypeException:
            raise
        except refusals as error:
            if CASPAR_DISABLED_MARKER in str(error):
                raise _BuildWithoutCaspar(str(error)) from error
            errors.append(f"{label}: {error}")
            return None
        # Residuals and no usable solution: the solver took the problem and failed it, which is a fact worth keeping;
        # no residuals is CASPAR's own "no adapter" answer.
        if summary is not None and summary.num_residuals > 0 and not summary.is_solution_usable():
            errors.append(f"{label}: ended {summary.termination_type.name} on {summary.num_residuals} residuals")
        return summary

    def usable(summary: pycolmap.BundleAdjustmentSummary | None) -> bool:
        return summary is not None and summary.num_residuals > 0 and summary.is_solution_usable()

    try:
        supported: frozenset[str] = frozenset(
            model_name
            for model_name in CASPAR_PROBE_CAMERA_MODELS
            if (summary := solved(model_name, model_name, constant_intrinsics=True)) is not None and projects(summary)
        )
        if not supported:
            return CasparCapability(
                availability="probe_failed" if errors else "unavailable", supported_camera_models=frozenset(), probe_errors=tuple(errors)
            )
        refines_sensor_from_rig: bool = "PINHOLE" in supported and usable(
            solved("PINHOLE +sensor_from_rig", "PINHOLE", num_sensors=2, refine_sensor_from_rig=True)
        )
        refinable: frozenset[str] = frozenset(
            model_name for model_name in sorted(supported) if usable(solved(f"{model_name} +intrinsics", model_name, refine_intrinsics=True))
        )
    except _BuildWithoutCaspar as error:
        return CasparCapability(availability="unavailable", supported_camera_models=frozenset(), probe_errors=(str(error),))
    return CasparCapability(
        availability="available",
        supported_camera_models=supported,
        probe_errors=tuple(errors),
        refines_sensor_from_rig=refines_sensor_from_rig,
        intrinsics_refinable_camera_models=refinable,
    )


class _BuildWithoutCaspar(Exception):
    """A probe met a pycolmap built without CASPAR: the sweep stops, as no later probe can say anything else."""


@cache
def detect_caspar_capability() -> CasparCapability:
    """Measure what this build's CASPAR projects and refines, once for the life of the process.

    Every probe runs through `_caspar_probe_solve` and only from here, so the `functools.cache` above bounds the whole
    sweep to once per process; `colsfm.ba_backend.resolve_ba_plan` only reaches it when `caspar` was asked for.

    Returns:
        The typed outcome: which models are projected, and — when none are —
        whether that is a build without CASPAR or a probe that could not run.
    """
    return _measure(_caspar_probe_solve, (ValueError, RuntimeError), lambda summary: summary.num_residuals > 0)


def caspar_supported_camera_models() -> frozenset[str]:
    """The camera models this build's CASPAR actually projects.

    Kept as the name the audit tools and the fisheye property suite already use;
    `detect_caspar_capability` is the answer with its provenance attached.

    Returns:
        The subset of `CASPAR_PROBE_CAMERA_MODELS` this build projects; empty when
        pycolmap is built without CASPAR_ENABLED or when no probe succeeded.
    """
    return detect_caspar_capability().supported_camera_models


def _mojo_probe_solve(
    model_name: str,
    *,
    num_sensors: int = 1,
    refine_intrinsics: bool = False,
    refine_sensor_from_rig: bool = False,
    constant_intrinsics: bool = False,
) -> pycolmap.BundleAdjustmentSummary | None:
    """The probe model of `_caspar_probe_solve`, solved by the Mojo solver through `colsfm.ba_mojo`.

    Args:
        model_name: A `pycolmap.CameraModelId` member name.
        num_sensors: Cameras in the probe rig.
        refine_intrinsics: Free the focal and the distortion block.
        refine_sensor_from_rig: Free the rig's `sensor_from_rig`.
        constant_intrinsics: Hold camera 1's intrinsics constant in the config as well.

    Returns:
        The solve's summary, or None when this pycolmap has no such camera model.

    Raises:
        ValueError: When the bridge's array contract does not carry the model.
        Exception: Whatever the extension raises: a model it has not ported, or no GPU.
    """
    from colsfm.ba_mojo import solve_with_mojo

    problem = _probe_problem(
        model_name,
        num_sensors=num_sensors,
        refine_intrinsics=refine_intrinsics,
        refine_sensor_from_rig=refine_sensor_from_rig,
        constant_intrinsics=constant_intrinsics,
    )
    if problem is None:
        return None
    reconstruction, config, ba_options = problem
    # COLMAP's default is the full system, which the Mojo solver does not have; the mapper sets the same switch.
    ba_options.caspar.reduced_camera_system = True
    return solve_with_mojo(reconstruction, config, ba_options).bundle_adjustment_summary()


@cache
def detect_mojo_capability() -> CasparCapability:
    """Measure what the Mojo solver solves, once for the life of the process, with CASPAR's probes.

    `unavailable` only when the `caspar_mojo` extension does not import (not built for this checkout:
    `pixi run caspar-mojo-build`); once it imports, a sweep that solved nothing is `probe_failed` with each probe's
    reason. A model it has not ported raises in the extension rather than being dropped, so its message lands in
    `probe_errors`.

    Returns:
        The typed outcome, in the same record CASPAR's probe fills.
    """
    try:
        # By name: the compiled module exists only once `caspar-mojo-build` has run, so no type checker can see it.
        importlib.import_module("caspar_mojo._caspar_mojo")
    except ImportError as error:
        return CasparCapability(availability="unavailable", supported_camera_models=frozenset(), probe_errors=(f"caspar_mojo: {error}",))
    # The extension raises its own errors as plain `Exception`, so every one of them is a refusal here.
    capability: CasparCapability = _measure(
        _mojo_probe_solve, (Exception,), lambda summary: summary.num_residuals > 0 and summary.is_solution_usable()
    )
    return replace(capability, availability="probe_failed") if capability.availability == "unavailable" else capability
