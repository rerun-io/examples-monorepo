"""Which bundle-adjustment backend a run gets, and what this build's CASPAR can do.

One owner for a policy that used to be spread over three files: `colsfm.pipeline`
parsed the `--caspar-option` strings after feature extraction, matching and loop
closure had already run; `colsfm.mapping` collapsed a probe error into "this
camera model is unsupported", let an empty support set bypass the camera guard,
and switched backend at solve time by rewriting the options object the caller
handed it. Nothing wrote down which backend actually ran, or why.

The module answers three questions, in order:

1. **What can this build do?** `colsfm.ba_capability` probes it by solving (there is
   no pycolmap call that lists CASPAR's camera adapters). The answer is a
   `CasparCapability`, which distinguishes *unavailable* (pycolmap built without
   `CASPAR_ENABLED`) from *probe failure* (the build has CASPAR but the probe could
   not run, e.g. no GPU answered) from *available*, and carries the models it
   projects. The three used to be one empty `frozenset`.
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
runs the same probe sweep through the bridge. Where it differs it says so before the
run starts: it refuses a camera model it has not ported (instead of dropping its observations), it takes one camera
model per solve, and it has no full-system PCG, so `reduced_camera_system=false` is refused outright.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap

from colsfm.ba_capability import CASPAR_NOT_MEASURED, CasparCapability, detect_caspar_capability, detect_mojo_capability

BaBackend: TypeAlias = Literal["ceres", "caspar", "mojo"]
"""Which implementation solves the global bundle adjustment; `BaExecutionPlan.backend`."""

BUNDLE_ADJUSTMENT_BACKEND: Final[dict[BaBackend, pycolmap.BundleAdjustmentBackend]] = {
    "ceres": pycolmap.BundleAdjustmentBackend.CERES,
    "caspar": pycolmap.BundleAdjustmentBackend.CASPAR,
    "mojo": pycolmap.BundleAdjustmentBackend.CASPAR,
}
"""The name to pycolmap's enum. The enum is bound unconditionally, so its presence proves
nothing about the build — only a solve does. `mojo` selects CASPAR because the Mojo solver reads its parameters
from the CASPAR block; `PreparedSolve` carries the name with the options, so the enum never has to tell them apart."""


CasparOptions: TypeAlias = dict[str, float | int | bool]
"""Overrides for `pycolmap.BundleAdjustmentOptions.caspar`, by attribute name.

The stopping and damping knobs of CASPAR's Levenberg-Marquardt / PCG loop —
`solver_iter_max`, `solver_rel_decrease_min`, `score_exit_value`, `diag_*`; the PCG mode
`reduced_camera_system` (the one bool) with the reduced system's `reduced_pcg_iter_max` and
`reduced_pcg_eta`, or the full system's `pcg_iter_max`, `pcg_rel_error_exit`, `pcg_rel_score_exit`
and `pcg_rel_decrease_min` (`resolve_caspar_mode`). Values are cast to the attribute's own type on the way in, so the integer counters
take integers whichever way they were written. `gpu_index` is a string and is not
settable here; `MappingOptions.use_gpu` owns it."""


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
        reason: str | None = camera_model_fallback(requested, measured, camera_model_names)
        if reason is not None:
            return "ceres", reason
    return requested, None


def camera_model_fallback(
    backend: BaBackend,
    capability: CasparCapability,
    camera_model_names: Iterable[str],
    *,
    refine_intrinsics: bool = False,
    rig_camera_models: Iterable[frozenset[str]] = (),
) -> str | None:
    """Why a GPU backend cannot take a reconstruction of these camera models: the one eligibility rule.

    `resolve_backend` and `BaExecutionPlan.for_camera_models` both ask here, so a request that one of them keeps on the
    GPU the other cannot send to Ceres.

    Args:
        backend: `caspar` or `mojo`.
        capability: What the backend was measured to do.
        camera_model_names: `pycolmap.CameraModelId` member names in the model about to be adjusted.
        refine_intrinsics: Whether the solves free the focal length; CASPAR throws mid-run for a model it projects
            with fixed intrinsics only (FULL_OPENCV).
        rig_camera_models: The camera model names of each rig.

    Returns:
        The fallback reason, or None when the backend takes the model.
    """
    names: frozenset[str] = frozenset(camera_model_names)
    unsupported: frozenset[str] = capability.unsupported(names)
    if unsupported:
        return unsupported_models_reason(capability.supported_camera_models, unsupported, backend)
    if backend == "mojo" and len(names) > 1:
        return MOJO_MIXED_MODELS_FALLBACK_REASON
    if any(len(models) > 1 for models in rig_camera_models):
        return MIXED_MODEL_RIG_FALLBACK_REASON
    fixed: frozenset[str] = capability.fixed_intrinsics_only(names) if refine_intrinsics else frozenset()
    if fixed:
        solver: str = "Mojo solver" if backend == "mojo" else "CASPAR build"
        return f"this {solver} refines the intrinsics of {sorted(capability.intrinsics_refinable_camera_models)} only, and this model frees them for {sorted(fixed)}"
    return None


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
            rig_camera_models: The camera model names of each rig.

        Returns:
            This plan when nothing changes, or the Ceres plan with `camera_model_fallback`'s reason.
        """
        if self.backend == "ceres":
            return self
        reason: str | None = camera_model_fallback(
            self.backend, self.capability, camera_model_names, refine_intrinsics=refine_intrinsics, rig_camera_models=rig_camera_models
        )
        return self if reason is None else replace(self, backend="ceres", ceres_polish=False, fallback_reason=reason)

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


@dataclass(frozen=True, slots=True)
class PreparedSolve:
    """One bundle adjustment ready to run: the solver and the pycolmap options it reads, built together.

    pycolmap's enum cannot name `mojo` (its options select the CASPAR block the Mojo solver reads), so the backend
    travels with the options from construction to `solve` instead of being recovered from them.
    """

    backend: BaBackend
    options: pycolmap.BundleAdjustmentOptions

    def __post_init__(self) -> None:
        if self.options.backend != BUNDLE_ADJUSTMENT_BACKEND[self.backend]:
            raise ValueError(f"[colsfm] a {self.backend} solve needs options on {BUNDLE_ADJUSTMENT_BACKEND[self.backend]}, got {self.options.backend}")

    def solve(
        self, reconstruction: pycolmap.Reconstruction, config: pycolmap.BundleAdjustmentConfig
    ) -> tuple[pycolmap.BundleAdjustmentSummary, float]:
        """Adjust the reconstruction in place.

        Args:
            reconstruction: The model.
            config: As for `pycolmap.create_default_bundle_adjuster`.

        Returns:
            The summary and the seconds the run pays for it: the solve, plus the bridge and write-back on `mojo`.
        """
        if self.backend == "mojo":
            from colsfm.ba_mojo import solve_with_mojo

            started: float = time.perf_counter()
            summary: pycolmap.BundleAdjustmentSummary = solve_with_mojo(reconstruction, config, self.options).bundle_adjustment_summary()
            return summary, time.perf_counter() - started
        adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(self.options, config, reconstruction)
        started = time.perf_counter()
        summary = adjuster.solve()
        return summary, time.perf_counter() - started
