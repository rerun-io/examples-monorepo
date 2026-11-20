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
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cache
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64
from numpy import ndarray

from colsfm.reconstruction import RIG_ID, camera_sensor_id

BaBackend: TypeAlias = Literal["ceres", "caspar"]
"""Which implementation solves the global bundle adjustment; `BaExecutionPlan.backend`."""

BUNDLE_ADJUSTMENT_BACKEND: Final[dict[BaBackend, pycolmap.BundleAdjustmentBackend]] = {
    "ceres": pycolmap.BundleAdjustmentBackend.CERES,
    "caspar": pycolmap.BundleAdjustmentBackend.CASPAR,
}
"""The name to pycolmap's enum. The enum is bound unconditionally, so its presence proves
nothing about the build — only a solve does."""

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

CasparOptions: TypeAlias = dict[str, float | int]
"""Numeric overrides for `pycolmap.BundleAdjustmentOptions.caspar`, by attribute name.

The stopping and damping knobs of CASPAR's Levenberg-Marquardt / PCG loop —
`solver_iter_max`, `pcg_iter_max`, `pcg_rel_error_exit`, `pcg_rel_score_exit`,
`pcg_rel_decrease_min`, `solver_rel_decrease_min`, `score_exit_value`, `diag_*`.
Values are cast to the attribute's own type on the way in, so the integer counters
take integers whichever way they were written. `gpu_index` is a string and is not
settable here; `MappingOptions.use_gpu` owns it."""


CASPAR_PROBE_FOCAL_LENGTH_PX: Final[float] = 300.0
"""Focal length of the probe camera; wide enough that a 640x480 image sees the whole grid."""

CASPAR_PROBE_IMAGE_SIZE_PX: Final[tuple[int, int]] = (640, 480)
"""Width and height of the probe camera, in pixels."""

CASPAR_PROBE_POINT_OFFSET_M: Final[float] = 0.02
"""How far the probe knocks its points off their exact positions, so the solve has work."""


def _caspar_probe_reconstruction(model_name: str) -> pycolmap.Reconstruction | None:
    """A minimal two-frame model of one camera model, knocked off its own solution.

    One single-sensor rig, so `refine_sensor_from_rig` never comes up; two frames, the
    first of which is the gauge; and a grid of points whose observations are exact
    projections, displaced afterwards so that a solver with anything to do has
    something to do. Small enough to build and solve in a few milliseconds.

    Args:
        model_name: A `pycolmap.CameraModelId` member name, e.g. `OPENCV_FISHEYE`.

    Returns:
        The model, or None when this pycolmap has no such camera model.
    """
    model: pycolmap.CameraModelId | None = getattr(pycolmap.CameraModelId, model_name, None)
    if model is None:
        return None
    width_px, height_px = CASPAR_PROBE_IMAGE_SIZE_PX
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(
        pycolmap.Camera.create_from_model_id(1, model, CASPAR_PROBE_FOCAL_LENGTH_PX, width_px, height_px)
    )
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(camera_sensor_id(1))
    reconstruction.add_rig(rig)

    baselines_m: tuple[float, float] = (0.0, -0.3)
    for frame_index, baseline_m in enumerate(baselines_m):
        image_id: int = frame_index + 1
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = image_id
        frame.rig_id = RIG_ID
        frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([baseline_m, 0.0, 0.0]))
        frame.add_data_id(pycolmap.data_t(camera_sensor_id(1), image_id))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        image: pycolmap.Image = pycolmap.Image(name=f"probe{image_id}.png", camera_id=1, image_id=image_id)
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
    reconstruction: pycolmap.Reconstruction | None = _caspar_probe_reconstruction(model_name)
    if reconstruction is None:
        return False
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(min(reconstruction.frames))
    config.set_constant_cam_intrinsics(1)

    ba_options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    ba_options.backend = BUNDLE_ADJUSTMENT_BACKEND["caspar"]
    ba_options.print_summary = False
    ba_options.refine_sensor_from_rig = False
    ba_options.refine_focal_length = False
    ba_options.refine_extra_params = False
    ba_options.refine_principal_point = False
    ba_options.refine_rig_from_world = True
    ba_options.refine_points3D = True
    # `-1` lets COLMAP pick the device, which is what this module's `use_gpu=False`
    # means everywhere else; the probe asks about adapters, not about a GPU.
    ba_options.caspar.gpu_index = "-1"
    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(ba_options, config, reconstruction)
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    return summary.num_residuals > 0


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
    sixteen-point model per candidate and reads `num_residuals`. The whole sweep is
    five sub-millisecond GPU solves, and `resolve_ba_plan` only reaches it when
    `caspar` was actually asked for.

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
            availability="available", supported_camera_models=frozenset(supported), probe_errors=tuple(errors)
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




def caspar_option_types() -> dict[str, type[int | float]]:
    """The Python type each numeric CASPAR option is bound as.

    `CasparBundleAdjustmentOptions` binds `solver_iter_max` and `pcg_iter_max` as
    C++ ints, which pybind11 refuses to take a Python float for; the rest are
    doubles. Validation needs the distinction before a run creates any artifact.

    Returns:
        Option name to `int` or `float`.
    """
    defaults: dict[str, object] = pycolmap.BundleAdjustmentOptions().caspar.todict()
    return {
        name: type(value)
        for name, value in defaults.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


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
    types: dict[str, type[int | float]] = caspar_option_types()
    options: CasparOptions = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"[colsfm] --caspar-option takes `name=value`, got `{item}`")
        key: str = name.strip()
        try:
            value: float = float(raw)
        except ValueError as error:
            raise ValueError(f"[colsfm] --caspar-option `{key}` needs a number, got `{raw}`") from error
        expected: type[int | float] | None = types.get(key)
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
    return options


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
    types: dict[str, type[int | float]] = caspar_option_types()
    for name, value in overrides.items():
        expected: type[int | float] | None = types.get(name)
        if expected is None:
            raise KeyError(
                f"[colsfm] `{name}` is not a numeric CASPAR solver option; the settable ones are {sorted(types)}"
            )
        setattr(caspar, name, expected(value))


# ══════════════════════════════════════════════════════════════════════════════════════
# what will actually run
# ══════════════════════════════════════════════════════════════════════════════════════


EXTRINSICS_FALLBACK_REASON: Final[str] = (
    "CASPAR holds `sensor_from_rig` fixed and throws when asked to refine it, so a bundle "
    "adjustment that frees it -- which is what `--no-regularised-extrinsics` asks for -- "
    "solves on Ceres"
)
"""Why a solve that refines `sensor_from_rig` ends up on Ceres.

Not the default configuration's refinement. `--optimize-extrinsics` is on by default
and stays on CASPAR, because the regularised path never asks a bundle adjuster to
free the extrinsics: `colsfm.extrinsic_refinement` moves them in pyceres and holds
them fixed in every pycolmap solve. What reaches this sentence is
`--no-regularised-extrinsics`, whose refinement *is*
`run_mapping(..., optimize_extrinsics=True)`."""

CASPAR_BUILD_FALLBACK_REASON: Final[str] = (
    "this pycolmap is built without CASPAR_ENABLED, so the GPU backend cannot solve "
    "(the `colsfm` environment, recipes/colmap-src, has the build that can)"
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


def unsupported_models_reason(supported: Iterable[str], unsupported: Iterable[str]) -> str:
    """The sentence a camera-model fallback reports and records.

    Args:
        supported: Model names this build projects.
        unsupported: Model names in the reconstruction that it does not.

    Returns:
        One line naming both sets.
    """
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
    if requested != "caspar":
        return requested, None
    if optimize_extrinsics:
        return "ceres", EXTRINSICS_FALLBACK_REASON
    measured: CasparCapability = detect_caspar_capability() if capability is None else capability
    if measured.availability == "unavailable":
        return "ceres", CASPAR_BUILD_FALLBACK_REASON
    if measured.availability == "probe_failed":
        return "ceres", CASPAR_PROBE_FAILED_REASON
    if camera_model_names is not None:
        unsupported: frozenset[str] = measured.unsupported(camera_model_names)
        if unsupported:
            return "ceres", unsupported_models_reason(measured.supported_camera_models, unsupported)
    return "caspar", None


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

    def for_camera_models(self, camera_model_names: Iterable[str]) -> BaExecutionPlan:
        """The same plan, narrowed by the camera models the reconstruction turned out to hold.

        The one fact resolution cannot have: the reconstruction is built in stage 7,
        long after the plan is made. Everything else is already decided, so this can
        only move `caspar` to `ceres`, never the other way.

        Args:
            camera_model_names: `pycolmap.CameraModelId` member names in the model
                about to be adjusted.

        Returns:
            This plan when nothing changes, or the Ceres plan with the camera-model
            reason when CASPAR would drop a model's observations.
        """
        if self.backend != "caspar":
            return self
        unsupported: frozenset[str] = self.capability.unsupported(camera_model_names)
        if not unsupported:
            return self
        return replace(
            self,
            backend="ceres",
            ceres_polish=False,
            fallback_reason=unsupported_models_reason(self.capability.supported_camera_models, unsupported),
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
    # Measured here rather than inside `resolve_backend`, so the plan carries the same
    # answer into `for_camera_models` instead of asking the build a second time. A
    # `ceres` request never probes: nothing in the plan will consult the answer.
    measured: CasparCapability = CASPAR_NOT_MEASURED
    if capability is not None:
        measured = capability
    elif requested == "caspar":
        measured = detect_caspar_capability()
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
        ceres_polish=ceres_polish and backend == "caspar",
        fallback_reason=reason,
        capability=measured,
    )
