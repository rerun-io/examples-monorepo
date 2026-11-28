"""Which bundle-adjustment backend a run gets, and what this build's CASPAR can do.

One owner for a policy that used to be spread over three files: `colsfm.pipeline`
parsed the `--caspar-option` strings after feature extraction, matching and loop
closure had already run; `colsfm.mapping` collapsed a probe error into "this
camera model is unsupported", let an empty support set bypass the camera guard,
and switched backend at solve time by rewriting the options object the caller
handed it. Nothing wrote down which backend actually ran, or why.

The module answers three questions, in order:

1. **What can this build do?** `detect_caspar_capability` probes it — there is no
   pycolmap call that lists CASPAR's camera adapters, so the only way to know is
   to hand it a sixteen-point problem per model and read `num_residuals`. The
   answer is a `CasparCapability`, which distinguishes *unavailable* (pycolmap
   built without `CASPAR_ENABLED`) from *probe failure* (the build has CASPAR but
   the probe could not run, e.g. no GPU answered) from *available*, and carries
   the models it projects. The three used to be one empty `frozenset`.
2. **What did the caller ask for?** `parse_caspar_options` turns
   `--caspar-option name=value` into a validated `CasparOptions`, rejecting an
   unknown name and rejecting `solver_iter_max=2.9` rather than truncating it to
   2 the way the old `int(...)` cast did.
3. **What will actually run?** `resolve_ba_plan` folds the two together into a
   `BaExecutionPlan`: the requested backend, the effective one, the reason for any
   fallback, the validated solver overrides and whether a Ceres polish closes the
   run. It is computed once, before a run writes anything, and the effective
   backend and the reason reach `summary.json`.

The fallbacks themselves are unchanged and deliberate. CASPAR *silently drops*
the observations of a camera model it has no adapter for, *throws* when asked to
refine `sensor_from_rig`, and cannot solve at all in a build without it; all three
make Ceres take the solve, loudly. All three are decided here, before the run
writes anything: the capability probe finds out about the missing build by
solving, so `colsfm.mapping` never has to discover it mid-run. The camera models
are the one fact that arrives late — the reconstruction does not exist at
resolution — and `BaExecutionPlan.for_camera_models` narrows the plan with them
exactly once, which is the plan the mapper then executes and reports.

The third backend, `mojo`, is the same CASPAR algorithm (LM + reduced-camera-system PCG) on generic Mojo GPU kernels
(`packages/caspar-mojo`), reached through `colsfm.ba_mojo`. It reads its solver parameters from the same CASPAR
block, so `--caspar-option` applies to it unchanged, and it is probed the way CASPAR is: `detect_mojo_capability`
solves the same sixteen-point model per camera model through the bridge. Where it differs it says so before the
run starts: it refuses a camera model it has not ported (instead of dropping its observations), it takes one camera
model per solve, and it has no full-system PCG, so `reduced_camera_system=false` is refused outright.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cache
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64
from numpy import ndarray

from colsfm.reconstruction import RIG_ID, camera_sensor_id

BaBackend: TypeAlias = Literal["ceres", "caspar", "mojo"]
"""Which implementation solves the global bundle adjustment; `BaExecutionPlan.backend`."""

BUNDLE_ADJUSTMENT_BACKEND: Final[dict[BaBackend, pycolmap.BundleAdjustmentBackend]] = {
    "ceres": pycolmap.BundleAdjustmentBackend.CERES,
    "caspar": pycolmap.BundleAdjustmentBackend.CASPAR,
    "mojo": pycolmap.BundleAdjustmentBackend.CASPAR,
}
"""The name to pycolmap's enum. The enum is bound unconditionally, so its presence proves
nothing about the build — only a solve does. `mojo` selects CASPAR because the Mojo solver reads its parameters
from the CASPAR block; `colsfm.mapping.solve_bundle_adjustment` takes the backend name and branches on it."""

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

CasparOptions: TypeAlias = dict[str, float | int | bool]
"""Overrides for `pycolmap.BundleAdjustmentOptions.caspar`, by attribute name.

The stopping and damping knobs of CASPAR's Levenberg-Marquardt / PCG loop —
`solver_iter_max`, `solver_rel_decrease_min`, `score_exit_value`, `diag_*`; the PCG mode
`reduced_camera_system` (the one bool) with the reduced system's `reduced_pcg_iter_max` and
`reduced_pcg_eta`, or the full system's `pcg_iter_max`, `pcg_rel_error_exit`, `pcg_rel_score_exit`
and `pcg_rel_decrease_min` (`resolve_caspar_mode`). Values are cast to the attribute's own type on the way in, so the integer counters
take integers whichever way they were written. `gpu_index` is a string and is not
settable here; `MappingOptions.use_gpu` owns it."""


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
        point3D_id: int = reconstruction.add_point3D(
            point_xyz + rng.normal(0.0, CASPAR_PROBE_POINT_OFFSET_M, 3),
            pycolmap.Track([pycolmap.TrackElement(image_id, index) for image_id, index in elements]),
            np.array([128, 128, 128], dtype=np.uint8),
        )
        for image_id, observation_index in elements:
            reconstruction.image(image_id).set_point3D_for_point2D(observation_index, point3D_id)
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
    ba_options.backend = BUNDLE_ADJUSTMENT_BACKEND["caspar"]
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


def _caspar_projects(model_name: str) -> bool:
    """Whether this build's CASPAR has an adapter for one camera model.

    CASPAR skips the images of a model it cannot project — "Skipping image ... with
    unsupported camera model" — and then reports `USER_FAILURE` with
    `num_residuals == 0` on the empty problem it is left with, while a model it does
    project yields two residuals per observation. That count is the signal: it is
    bound by pycolmap and it distinguishes the two cases exactly, where the
    termination type alone would not survive a solver that fails for another reason.

    Args:
        model_name: A `pycolmap.CameraModelId` member name.

    Returns:
        True when the solve parameterised the probe's observations.

    Raises:
        ValueError: When pycolmap is built without CASPAR_ENABLED.
    """
    summary: pycolmap.BundleAdjustmentSummary | None = _caspar_probe_solve(model_name, constant_intrinsics=True)
    return summary is not None and summary.num_residuals > 0


def _caspar_solves_usably(model_name: str, *, num_sensors: int = 1, refine_intrinsics: bool = False, refine_sensor_from_rig: bool = False) -> bool:
    """Whether the probe solve with these freedoms ran and left a usable solution; any exception or an empty problem is a no.

    Args:
        model_name: A `pycolmap.CameraModelId` member name.
        num_sensors: Cameras in the probe rig.
        refine_intrinsics: Free the focal and the distortion block.
        refine_sensor_from_rig: Free the rig's `sensor_from_rig`.

    Returns:
        True when the solve took the problem and left a usable solution.
    """
    try:
        summary: pycolmap.BundleAdjustmentSummary | None = _caspar_probe_solve(
            model_name, num_sensors=num_sensors, refine_intrinsics=refine_intrinsics, refine_sensor_from_rig=refine_sensor_from_rig
        )
    except (ValueError, RuntimeError):
        return False
    return summary is not None and summary.num_residuals > 0 and summary.is_solution_usable()


def _caspar_refines_intrinsics(model_name: str) -> bool:
    """Whether this build's CASPAR refines one camera model's focal and distortion block.

    recipes/colmap-src's FULL_OPENCV adapter throws here ("supported by CASPAR only for fixed intrinsics").

    Args:
        model_name: A `pycolmap.CameraModelId` member name.

    Returns:
        True when the solve took the free intrinsics and left a usable solution.
    """
    return _caspar_solves_usably(model_name, refine_intrinsics=True)


def _caspar_refines_sensor_from_rig() -> bool:
    """Whether this build's CASPAR refines `sensor_from_rig`, measured on a two-camera PINHOLE probe rig.

    Stock CASPAR throws "refine_sensor_from_rig=true is not supported by CASPAR" on a multi-sensor frame with free
    extrinsics; recipes/colmap-src with the rig factor solves it (Mission office: rig + per-face focals free in every
    mapping round, 46 s against 383 s on Ceres, poses 0.018 deg apart).

    Returns:
        True when the solve took the rig problem and left a usable solution.
    """
    return _caspar_solves_usably("PINHOLE", num_sensors=2, refine_sensor_from_rig=True)


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
    """Whether this build's CASPAR refined `sensor_from_rig` on a two-sensor probe rig (`_caspar_refines_sensor_from_rig`).

    Stock CASPAR throws on a multi-sensor frame with free extrinsics; recipes/colmap-src with the rig factor solves it.
    False keeps every solve that frees the rig on Ceres."""
    intrinsics_refinable_camera_models: frozenset[str] = frozenset()
    """The projected models whose focal (and distortion block) CASPAR also refines (`_caspar_refines_intrinsics`).

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


@cache
def detect_caspar_capability() -> CasparCapability:
    """Measure what this build's CASPAR projects, once for the life of the process.

    There is no pycolmap call that lists CASPAR's adapters, so the only way to know
    is to hand it a problem and see whether it took it: `_caspar_projects` solves a
    sixteen-point model per candidate and reads `num_residuals`; on a build that projects
    something, one more solve asks about the rig factor and one per projected model about
    free intrinsics. Every probe runs through `_caspar_probe_solve` and only from here, so
    the `functools.cache` above bounds the whole sweep to once per process; `resolve_ba_plan`
    only reaches it when `caspar` was actually asked for.

    Returns:
        The typed outcome: which models are projected, and — when none are —
        whether that is a build without CASPAR or a probe that could not run.
    """
    supported: set[str] = set()
    errors: list[str] = []
    for model_name in CASPAR_PROBE_CAMERA_MODELS:
        try:
            if _caspar_projects(model_name):
                supported.add(model_name)
        except (ValueError, RuntimeError) as error:
            if CASPAR_DISABLED_MARKER in str(error):
                return CasparCapability(
                    availability="unavailable", supported_camera_models=frozenset(), probe_errors=(str(error),)
                )
            # One model that throws says nothing about the next; a build without
            # CASPAR at all has already returned above.
            errors.append(f"{model_name}: {error}")
    if supported:
        return CasparCapability(
            availability="available",
            supported_camera_models=frozenset(supported),
            probe_errors=tuple(errors),
            refines_sensor_from_rig="PINHOLE" in supported and _caspar_refines_sensor_from_rig(),
            intrinsics_refinable_camera_models=frozenset(model_name for model_name in sorted(supported) if _caspar_refines_intrinsics(model_name)),
        )
    availability: CasparAvailability = "probe_failed" if errors else "unavailable"
    return CasparCapability(availability=availability, supported_camera_models=frozenset(), probe_errors=tuple(errors))


def caspar_supported_camera_models() -> frozenset[str]:
    """The camera models this build's CASPAR actually projects.

    Kept as the name the audit tools and the fisheye property suite already use;
    `detect_caspar_capability` is the answer with its provenance attached.

    Returns:
        The subset of `CASPAR_PROBE_CAMERA_MODELS` this build projects; empty when
        pycolmap is built without CASPAR_ENABLED or when no probe succeeded.
    """
    return detect_caspar_capability().supported_camera_models


BeartypeException: tuple[type[Exception], ...] = (sys.modules["beartype.roar"].BeartypeException,) if "beartype.roar" in sys.modules else ()
"""What the Mojo probe re-raises from its broad `except`: a test-time type violation is a bug, not a capability."""


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


def _mojo_solves_usably(model_name: str, *, num_sensors: int = 1, refine_intrinsics: bool = False, refine_sensor_from_rig: bool = False) -> bool:
    """Whether the Mojo probe solve with these freedoms ran and left a usable solution; any exception is a no.

    Args:
        model_name: A `pycolmap.CameraModelId` member name.
        num_sensors: Cameras in the probe rig.
        refine_intrinsics: Free the focal and the distortion block.
        refine_sensor_from_rig: Free the rig's `sensor_from_rig`.

    Returns:
        True when the solve took the problem and left a usable solution.
    """
    try:
        summary: pycolmap.BundleAdjustmentSummary | None = _mojo_probe_solve(
            model_name, num_sensors=num_sensors, refine_intrinsics=refine_intrinsics, refine_sensor_from_rig=refine_sensor_from_rig
        )
    except BeartypeException:
        raise
    except Exception:
        return False
    return summary is not None and summary.num_residuals > 0 and summary.is_solution_usable()


@cache
def detect_mojo_capability() -> CasparCapability:
    """Measure what the Mojo solver solves, once for the life of the process, with CASPAR's probes.

    `unavailable` when the `caspar_mojo` extension does not import (not built for this checkout:
    `pixi run caspar-mojo-build`); `probe_failed` when it imports but no probe model solved (no GPU answered);
    `available` with the models it solved otherwise. A model it has not ported raises in the extension
    rather than being dropped, so its message lands in `probe_errors`.

    Returns:
        The typed outcome, in the same record CASPAR's probe fills.
    """
    try:
        from caspar_mojo import _caspar_mojo  # noqa: F401
    except ImportError as error:
        return CasparCapability(availability="unavailable", supported_camera_models=frozenset(), probe_errors=(f"caspar_mojo: {error}",))
    supported: set[str] = set()
    errors: list[str] = []
    for model_name in CASPAR_PROBE_CAMERA_MODELS:
        try:
            summary: pycolmap.BundleAdjustmentSummary | None = _mojo_probe_solve(model_name, constant_intrinsics=True)
        except BeartypeException:
            raise
        except Exception as error:
            errors.append(f"{model_name}: {error}")
            continue
        if summary is not None and summary.num_residuals > 0 and summary.is_solution_usable():
            supported.add(model_name)
    if not supported:
        return CasparCapability(
            availability="probe_failed" if errors else "unavailable", supported_camera_models=frozenset(), probe_errors=tuple(errors)
        )
    return CasparCapability(
        availability="available",
        supported_camera_models=frozenset(supported),
        probe_errors=tuple(errors),
        refines_sensor_from_rig="PINHOLE" in supported and _mojo_solves_usably("PINHOLE", num_sensors=2, refine_sensor_from_rig=True),
        intrinsics_refinable_camera_models=frozenset(
            model_name for model_name in sorted(supported) if _mojo_solves_usably(model_name, refine_intrinsics=True)
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# what the caller asked for
# ══════════════════════════════════════════════════════════════════════════════════════


def backend_name(ba_options: pycolmap.BundleAdjustmentOptions) -> BaBackend:
    """Which backend a set of options selects, as this module's name for it.

    Args:
        ba_options: The options a solve ran with.

    Returns:
        `caspar` when the options select CASPAR, `ceres` otherwise.
    """
    return "caspar" if ba_options.backend == BUNDLE_ADJUSTMENT_BACKEND["caspar"] else "ceres"




CASPAR_BOOL_OPTIONS: Final[frozenset[str]] = frozenset({"reduced_camera_system"})
"""The bool solver controls `--caspar-option` takes; the binding's other bools are not solver knobs."""

FULL_SYSTEM_OPTIONS: Final[frozenset[str]] = frozenset({"pcg_iter_max", "pcg_rel_error_exit", "pcg_rel_score_exit", "pcg_rel_decrease_min"})
"""Knobs of the full-system PCG (recipes/colmap-src `caspar-reduced-camera-system.patch`): ignored or refused by the
reduced camera system."""

REDUCED_SYSTEM_OPTIONS: Final[frozenset[str]] = frozenset({"reduced_pcg_iter_max", "reduced_pcg_eta"})
"""Knobs of the reduced camera system's PCG."""


def caspar_option_types() -> dict[str, type[int | float | bool]]:
    """The Python type each settable CASPAR option is bound as.

    `CasparBundleAdjustmentOptions` binds `solver_iter_max` and `pcg_iter_max` as
    C++ ints, which pybind11 refuses to take a Python float for; the rest are
    doubles, plus the bools in `CASPAR_BOOL_OPTIONS`. Validation needs the distinction
    before a run creates any artifact.

    Returns:
        Option name to `int`, `float` or `bool`.
    """
    defaults: dict[str, object] = pycolmap.BundleAdjustmentOptions().caspar.todict()
    return {
        name: type(value)
        for name, value in defaults.items()
        if (isinstance(value, (int, float)) and not isinstance(value, bool)) or (isinstance(value, bool) and name in CASPAR_BOOL_OPTIONS)
    }


def resolve_caspar_mode(options: CasparOptions) -> CasparOptions:
    """Make the PCG mode explicit when the overrides imply it, and refuse overrides the chosen mode would ignore.

    With no `reduced_camera_system` given, a full-system knob (`FULL_SYSTEM_OPTIONS`) selects the full system: the
    reduced default would otherwise run while silently ignoring it. A mode given explicitly must not come with the other
    mode's knobs, and the reduced knobs are range-checked here, before the run creates anything. `reduced_pcg_eta` is
    checked as the f32 the solver uses, as the solver checks it (0.99999999 rounds to 1.0).

    Args:
        options: Parsed overrides.

    Returns:
        The overrides, with `reduced_camera_system=False` added when full-system knobs imply it.

    Raises:
        ValueError: On contradictory modes or an out-of-range reduced knob.
    """
    full: list[str] = sorted(FULL_SYSTEM_OPTIONS & options.keys())
    reduced: list[str] = sorted(REDUCED_SYSTEM_OPTIONS & options.keys())
    mode: float | int | bool | None = options.get("reduced_camera_system")
    if mode is None and full and reduced:
        raise ValueError(f"[colsfm] --caspar-option sets both the full-system {full} and the reduced {reduced} PCG knobs; pick one with reduced_camera_system")
    if mode is True and full:
        raise ValueError(f"[colsfm] --caspar-option {full} apply to the full-system PCG only; drop them or pass reduced_camera_system=false")
    if mode is False and reduced:
        raise ValueError(f"[colsfm] --caspar-option {reduced} apply to the reduced camera system only; drop them or pass reduced_camera_system=true")
    if "reduced_pcg_iter_max" in options and options["reduced_pcg_iter_max"] < 1:
        raise ValueError(f"[colsfm] --caspar-option reduced_pcg_iter_max must be at least 1, got {options['reduced_pcg_iter_max']}")
    if "reduced_pcg_eta" in options and not 0.0 < np.float32(options["reduced_pcg_eta"]) < 1.0:
        raise ValueError(f"[colsfm] --caspar-option reduced_pcg_eta must be in (0, 1) as an f32, got {options['reduced_pcg_eta']}")
    return {**options, "reduced_camera_system": False} if mode is None and full else options


def parse_caspar_options(items: tuple[str, ...]) -> CasparOptions:
    """Turn `--caspar-option name=value` strings into validated solver overrides.

    Validation happens here rather than at the solve, so a typo costs nothing:
    `run_pipeline` parses before it creates a workspace, and an unknown name or a
    fractional iteration count stops the run instead of being discovered after
    feature extraction, matching and loop closure have been paid for.

    `solver_iter_max=2.9` is rejected rather than truncated to 2. The old path cast
    with `int(...)` and ran a solve nobody asked for.

    Args:
        items: One `name=value` per repetition of the flag.

    Returns:
        The parsed overrides, empty when nothing was passed — which is what keeps
        COLMAP's own CASPAR defaults.

    Raises:
        ValueError: When an item has no `=`, its value is not a number, its name is
            not a settable numeric CASPAR option, or an integer option is given a
            fractional value.
    """
    types: dict[str, type[int | float | bool]] = caspar_option_types()
    options: CasparOptions = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"[colsfm] --caspar-option takes `name=value`, got `{item}`")
        key: str = name.strip()
        expected: type[int | float | bool] | None = types.get(key)
        if expected is bool:
            flag: bool | None = {"true": True, "1": True, "false": False, "0": False}.get(raw.strip().lower())
            if flag is None:
                raise ValueError(f"[colsfm] --caspar-option `{key}` takes true or false (1 or 0), got `{raw}`")
            options[key] = flag
            continue
        try:
            value: float = float(raw)
        except ValueError as error:
            raise ValueError(f"[colsfm] --caspar-option `{key}` needs a number, got `{raw}`") from error
        if expected is None:
            raise ValueError(
                f"[colsfm] `{key}` is not a numeric CASPAR solver option; the settable ones are {sorted(types)}"
            )
        if expected is int and value != int(value):
            raise ValueError(
                f"[colsfm] --caspar-option `{key}` is an integer solver knob and cannot take `{raw}`; "
                f"write {int(value)} if that is what you meant"
            )
        options[key] = int(value) if expected is int else value
    return resolve_caspar_mode(options)


def apply_caspar_options(caspar: pycolmap.CasparBundleAdjustmentOptions, overrides: CasparOptions) -> None:
    """Set CASPAR solver knobs by name, casting each value to the attribute's own type.

    `CasparBundleAdjustmentOptions` binds `solver_iter_max` and `pcg_iter_max` as
    C++ ints, which pybind11 refuses to take a Python float for, so every value
    goes through the type the default carries. `gpu_index` is a string knob owned
    by `MappingOptions.use_gpu` and is rejected here.

    Args:
        caspar: The `BundleAdjustmentOptions.caspar` block, modified in place.
        overrides: Attribute name to value.

    Raises:
        KeyError: When a name is not a numeric CASPAR option.
    """
    types: dict[str, type[int | float | bool]] = caspar_option_types()
    for name, value in overrides.items():
        expected: type[int | float | bool] | None = types.get(name)
        if expected is None:
            raise KeyError(
                f"[colsfm] `{name}` is not a numeric CASPAR solver option; the settable ones are {sorted(types)}"
            )
        setattr(caspar, name, expected(value))


# ══════════════════════════════════════════════════════════════════════════════════════
# what will actually run
# ══════════════════════════════════════════════════════════════════════════════════════


EXTRINSICS_FALLBACK_REASON: Final[str] = (
    "this CASPAR build holds `sensor_from_rig` fixed and throws when asked to refine it, so a bundle "
    "adjustment that frees it (`--no-regularised-extrinsics`, `--refine-camera-model`) "
    "solves on Ceres; recipes/colmap-src with the rig factor keeps it on CASPAR"
)
"""Why a solve that refines `sensor_from_rig` ends up on Ceres.

Not the default configuration's refinement. `--optimize-extrinsics` is on by default
and stays on CASPAR, because the regularised path never asks a bundle adjuster to
free the extrinsics: `colsfm.extrinsic_refinement` moves them in pyceres and holds
them fixed in every pycolmap solve. What reaches this sentence is
`--no-regularised-extrinsics`, whose refinement *is*
`run_mapping(..., optimize_extrinsics=True)`."""

MIXED_MODEL_RIG_FALLBACK_REASON: Final[str] = (
    "a rig of this model holds cameras of more than one camera model, and CASPAR keys a frame's pose by camera "
    "model (one pose per model instead of one rig pose; a free `sensor_from_rig` stops the solve)"
)
"""Why a `caspar` request ends up on Ceres when one rig mixes camera models.

Every model may project on this build and the problem is still wrong: the rig
constraint between a PINHOLE and an OPENCV_FISHEYE camera of the same frame is
lost. `build_rig` gives each cuSFM camera its own model, so a mixed rig is a real
input, not a corner case."""

CASPAR_BUILD_FALLBACK_REASON: Final[str] = (
    "this pycolmap is built without CASPAR_ENABLED, so the GPU backend cannot solve "
    "(the `colsfm` environment, ai-demos colmap-caspar, has the build that can)"
)
"""Why a `caspar` request ends up on Ceres when the build has no CASPAR.

`detect_caspar_capability` already discovers this -- its probe *is* a CASPAR solve --
so the run knows it before it creates anything, and no bundle adjustment ever has to
catch `CASPAR_DISABLED_MARKER` mid-run to find out."""

CASPAR_PROBE_FAILED_REASON: Final[str] = (
    "this build has CASPAR but every probe model raised, so it cannot be trusted with "
    "the run; `CasparCapability.probe_errors` carries what it said"
)
"""Why a `caspar` request ends up on Ceres when the probe could not run a single model."""


MOJO_BUILD_FALLBACK_REASON: Final[str] = (
    "the caspar_mojo extension does not import in this checkout (build it for the local GPU with "
    "`pixi run -e colsfm caspar-mojo-build`), so the Mojo solver cannot solve"
)
"""Why a `mojo` request ends up on Ceres when `detect_mojo_capability` cannot import the extension."""

MOJO_PROBE_FAILED_REASON: Final[str] = (
    "the caspar_mojo extension imports but no probe model solved (no GPU answered?); "
    "`CasparCapability.probe_errors` carries what it said"
)
"""Why a `mojo` request ends up on Ceres when the extension imports and every probe raised."""

MOJO_EXTRINSICS_FALLBACK_REASON: Final[str] = (
    "the Mojo solver did not refine `sensor_from_rig` on the two-camera probe rig, so a bundle adjustment that frees it solves on Ceres"
)
"""Why a `mojo` solve that refines `sensor_from_rig` ends up on Ceres."""

MOJO_MIXED_MODELS_FALLBACK_REASON: Final[str] = "the Mojo solver takes one camera model per solve, and this model holds several"
"""Why a `mojo` request ends up on Ceres when the reconstruction holds more than one camera model."""


def unsupported_models_reason(supported: Iterable[str], unsupported: Iterable[str], backend: BaBackend = "caspar") -> str:
    """The sentence a camera-model fallback reports and records.

    Args:
        supported: Model names this build projects.
        unsupported: Model names in the reconstruction that it does not.
        backend: The GPU backend that was asked for.

    Returns:
        One line naming both sets.
    """
    if backend == "mojo":
        return f"the Mojo solver solves {sorted(supported)}, and this model has {sorted(unsupported)}, which it refuses"
    return (
        f"this CASPAR build projects {sorted(supported)}, and this model has {sorted(unsupported)}, "
        f"whose observations it would silently drop"
    )


def resolve_backend(
    requested: BaBackend,
    *,
    optimize_extrinsics: bool,
    camera_model_names: Iterable[str] | None = None,
    capability: CasparCapability | None = None,
) -> tuple[BaBackend, str | None]:
    """The backend that will actually run, and why it is not the requested one.

    No fallback is an error: CASPAR would *silently* drop the observations of a camera
    model this build has no adapter for, would hold `sensor_from_rig` fixed where the
    caller asked for it to move, and cannot solve at all in a build without it. All
    three hand the solve to Ceres, and all three are decided here — there is no
    mid-run discovery left, because the capability probe finds out by solving.

    Args:
        requested: The backend the caller asked for.
        optimize_extrinsics: Whether this solve refines `sensor_from_rig`, i.e.
            whether `refine_sensor_from_rig` will be set on the bundle adjuster.
        camera_model_names: The models in the reconstruction about to be adjusted.
            None skips the camera check — the caller has no model to offer.
        capability: What this build can do; measured when None.

    Returns:
        The effective backend and the fallback reason, or None when the request stands.
    """
    if requested == "ceres":
        return requested, None
    mojo: bool = requested == "mojo"
    measured: CasparCapability = capability if capability is not None else detect_mojo_capability() if mojo else detect_caspar_capability()
    if measured.availability == "unavailable":
        return "ceres", MOJO_BUILD_FALLBACK_REASON if mojo else CASPAR_BUILD_FALLBACK_REASON
    if measured.availability == "probe_failed":
        return "ceres", MOJO_PROBE_FAILED_REASON if mojo else CASPAR_PROBE_FAILED_REASON
    if optimize_extrinsics and not measured.refines_sensor_from_rig:
        return "ceres", MOJO_EXTRINSICS_FALLBACK_REASON if mojo else EXTRINSICS_FALLBACK_REASON
    if camera_model_names is not None:
        unsupported: frozenset[str] = measured.unsupported(camera_model_names)
        if unsupported:
            return "ceres", unsupported_models_reason(measured.supported_camera_models, unsupported, requested)
    return requested, None


@dataclass(frozen=True, slots=True)
class BaExecutionPlan:
    """How this run's bundle adjustments will be solved, decided once and written down."""

    requested_backend: BaBackend
    """What the command line asked for, kept so the summary can say what was wanted."""
    backend: BaBackend
    """What will run, after the pre-flight fallbacks. `ceres` unless CASPAR is usable."""
    caspar_options: CasparOptions
    """Validated solver overrides; empty keeps COLMAP's own CASPAR defaults.

    Parsed and checked before the run creates anything, so an unknown name or a
    fractional integer knob costs a message rather than a whole extraction pass.
    Inert on `ceres`, as the CASPAR block is never consulted there."""
    ceres_polish: bool
    """Whether one closing Ceres global bundle adjustment finishes a CASPAR run.

    Already False whenever `backend` is `ceres`: every fallback lands on a solver
    that has converged, so there is nothing left to polish."""
    fallback_reason: str | None
    """Why `backend` is not `requested_backend`, or None when it is.

    The record the run keeps of a decision that used to exist only as a line on
    stdout, and the one `summary.json` reports."""
    capability: CasparCapability
    """What the build was measured to do when this plan was made.

    Carried so that `for_camera_models` narrows against the same measurement the rest
    of the plan was decided on, and so that a caller — a test, a sweep — can state a
    capability once and have every later decision honour it."""

    @property
    def fell_back(self) -> bool:
        """Whether the requested backend was refused.

        Returns:
            True when `backend` differs from `requested_backend`.
        """
        return self.backend != self.requested_backend

    def for_camera_models(
        self,
        camera_model_names: Iterable[str],
        *,
        refine_intrinsics: bool = False,
        rig_camera_models: Iterable[frozenset[str]] = (),
    ) -> BaExecutionPlan:
        """The same plan, narrowed by the camera models the reconstruction turned out to hold.

        The one fact resolution cannot have: the reconstruction is built in stage 7,
        long after the plan is made. Everything else is already decided, so this can
        only move `caspar` to `ceres`, never the other way.

        Args:
            camera_model_names: `pycolmap.CameraModelId` member names in the model
                about to be adjusted.
            refine_intrinsics: Whether the solves free the focal length; CASPAR throws
                mid-run for a model it projects with fixed intrinsics only (FULL_OPENCV).

        Returns:
            This plan when nothing changes, or the Ceres plan with the camera-model
            reason when CASPAR would drop a model's observations or cannot free its intrinsics.
        """
        if self.backend == "ceres":
            return self
        names: frozenset[str] = frozenset(camera_model_names)
        unsupported: frozenset[str] = self.capability.unsupported(names)
        if unsupported:
            return replace(
                self,
                backend="ceres",
                ceres_polish=False,
                fallback_reason=unsupported_models_reason(self.capability.supported_camera_models, unsupported, self.backend),
            )
        if self.backend == "mojo" and len(names) > 1:
            return replace(self, backend="ceres", ceres_polish=False, fallback_reason=MOJO_MIXED_MODELS_FALLBACK_REASON)
        if any(len(models) > 1 for models in rig_camera_models):
            return replace(self, backend="ceres", ceres_polish=False, fallback_reason=MIXED_MODEL_RIG_FALLBACK_REASON)
        fixed: frozenset[str] = self.capability.fixed_intrinsics_only(names) if refine_intrinsics else frozenset()
        if not fixed:
            return self
        return replace(
            self,
            backend="ceres",
            ceres_polish=False,
            fallback_reason=f"this {'Mojo solver' if self.backend == 'mojo' else 'CASPAR build'} refines the intrinsics of {sorted(self.capability.intrinsics_refinable_camera_models)} only, and this model frees them for {sorted(fixed)}",
        )

    def announce(self) -> None:
        """Print the fallback, once, wherever it was decided.

        A `--ba-backend caspar` run that solves on Ceres has to say so: the reason is
        recorded in `summary.json`, but somebody watching the run needs it now.
        """
        if self.fallback_reason is not None:
            print(f"[colsfm] {self.fallback_reason}; using Ceres")


CERES_ONLY_PLAN: Final[BaExecutionPlan] = BaExecutionPlan(
    requested_backend="ceres",
    backend="ceres",
    caspar_options={},
    ceres_polish=False,
    fallback_reason=None,
    capability=CASPAR_NOT_MEASURED,
)
"""The plan of a caller who said nothing about bundle adjustment: Ceres, no polish.

`MappingOptions.ba_plan`'s default, and what every solve outside a `--ba-backend
caspar` run executes. Nothing fell back, so there is no reason to report."""


def resolve_ba_plan(
    requested: BaBackend,
    caspar_option_items: tuple[str, ...],
    *,
    ceres_polish: bool,
    optimize_extrinsics: bool,
    camera_model_names: Iterable[str] | None = None,
    capability: CasparCapability | None = None,
) -> BaExecutionPlan:
    """Fold the command line's bundle-adjustment knobs into one plan.

    Called at the start of a run, before any artifact is created: the CASPAR
    strings are validated here, so `--caspar-option solver_iter_max=2.9` stops the
    run instead of being truncated after feature extraction has been paid for.

    Args:
        requested: `--ba-backend`.
        caspar_option_items: `--caspar-option name=value`, one per repetition.
        ceres_polish: `--caspar-ceres-polish`; inert once the backend is Ceres.
        optimize_extrinsics: Whether a *bundle adjustment* of this run frees
            `sensor_from_rig`, which CASPAR cannot honour. That is not the same
            question as `--optimize-extrinsics`: the regularised refinement, which
            is the default, refines the extrinsics outside the bundle adjuster and
            leaves this False. See `colsfm.run_config.resolve_run`.
        camera_model_names: The models the run will adjust, when they are known.
            None at the start of a run — the reconstruction does not exist yet —
            so the camera check happens at the first solve instead.
        capability: What this build can do; measured when None.

    Returns:
        The plan the run will execute and report.

    Raises:
        ValueError: When a `--caspar-option` item is malformed, names an option
            that is not a settable numeric CASPAR knob, or gives an integer knob a
            fractional value.
    """
    options: CasparOptions = parse_caspar_options(caspar_option_items)
    if requested == "mojo" and options.get("reduced_camera_system") is False:
        raise ValueError(
            "[colsfm] --ba-backend mojo solves the reduced camera system only; drop reduced_camera_system=false and the full-system knobs"
        )
    # Measured here rather than inside `resolve_backend`, so the plan carries the same
    # answer into `for_camera_models` instead of asking the build a second time. A
    # `ceres` request never probes: nothing in the plan will consult the answer.
    measured: CasparCapability = CASPAR_NOT_MEASURED
    if capability is not None:
        measured = capability
    elif requested == "caspar":
        measured = detect_caspar_capability()
    elif requested == "mojo":
        measured = detect_mojo_capability()
    backend, reason = resolve_backend(
        requested,
        optimize_extrinsics=optimize_extrinsics,
        camera_model_names=camera_model_names,
        capability=measured,
    )
    return BaExecutionPlan(
        requested_backend=requested,
        backend=backend,
        caspar_options=options,
        ceres_polish=ceres_polish and backend != "ceres",
        fallback_reason=reason,
        capability=measured,
    )
