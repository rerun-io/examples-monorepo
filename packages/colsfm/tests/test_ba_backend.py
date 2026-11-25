"""The bundle-adjustment backend policy: capability, validation and the execution plan.

Pure contracts — nothing here needs a GPU, a CASPAR build or a dataset. The one
function that does measure the build, `detect_caspar_capability`, is exercised
through a stated `CasparCapability` instead, which every caller can hand to
`resolve_backend` and `resolve_ba_plan`.

These are the review's confirmed defects, pinned:

* `--caspar-option solver_iter_max=2.9` used to be cast to 2 and solved with,
  because the value only met its C++ `int` binding at the solve.
* An unknown option name used to be discovered at the solve too, i.e. after
  feature extraction, matching and loop closure had been paid for.
* A probe error, a build without CASPAR and a build that projects nothing were
  one empty `frozenset`, so the camera guard could not tell them apart.
"""

from __future__ import annotations

import pytest

from colsfm.ba_backend import (
    CASPAR_BUILD_FALLBACK_REASON,
    CASPAR_PROBE_FAILED_REASON,
    EXTRINSICS_FALLBACK_REASON,
    BaExecutionPlan,
    CasparCapability,
    CasparOptions,
    caspar_option_types,
    parse_caspar_options,
    resolve_ba_plan,
    resolve_backend,
    resolve_caspar_mode,
)

STOCK_CAPABILITY: CasparCapability = CasparCapability(
    availability="available", supported_camera_models=frozenset({"PINHOLE", "SIMPLE_RADIAL"}), probe_errors=()
)
"""What stock CASPAR reports: two adapters, no probe error."""

RIG_CAPABILITY: CasparCapability = CasparCapability(
    availability="available",
    supported_camera_models=frozenset({"PINHOLE", "SIMPLE_RADIAL", "OPENCV_FISHEYE"}),
    probe_errors=(),
    refines_sensor_from_rig=True,
)
"""A build whose CASPAR also refines `sensor_from_rig` (recipes/colmap-src with the rig factor)."""

NO_BUILD_CAPABILITY: CasparCapability = CasparCapability(
    availability="unavailable", supported_camera_models=frozenset(), probe_errors=("built without CASPAR_ENABLED",)
)
"""A pycolmap with no CASPAR at all; the probe finds this out by solving, so the plan refuses it."""

BROKEN_PROBE_CAPABILITY: CasparCapability = CasparCapability(
    availability="probe_failed", supported_camera_models=frozenset(), probe_errors=("PINHOLE: no CUDA device",)
)
"""A build that has CASPAR but could not be measured; nothing has been shown to solve, so Ceres takes it."""


# ─────────────────────────────────────────────────────────────────────────────
# `--caspar-option` validation
# ─────────────────────────────────────────────────────────────────────────────


def test_no_flags_means_colmaps_own_defaults() -> None:
    """An empty tuple parses to an empty override set, which changes nothing."""
    assert parse_caspar_options(()) == {}


def test_an_integer_knob_keeps_its_integer_type() -> None:
    """`solver_iter_max` is a C++ int, so the parsed value is an `int`, not a float."""
    parsed: CasparOptions = parse_caspar_options(("solver_iter_max=1000",))
    assert parsed == {"solver_iter_max": 1000}
    assert isinstance(parsed["solver_iter_max"], int)


def test_a_double_knob_keeps_its_fractional_value() -> None:
    """The damping and exit knobs are doubles and are not rounded."""
    assert parse_caspar_options(("diag_scaling_up=2.5",)) == {"diag_scaling_up": 2.5}


def test_a_fractional_value_for_an_integer_knob_is_refused() -> None:
    """`solver_iter_max=2.9` is a mistake, not a request for two iterations."""
    with pytest.raises(ValueError, match="integer solver knob"):
        parse_caspar_options(("solver_iter_max=2.9",))


def test_an_unknown_option_name_is_refused_at_parse_time() -> None:
    """The name is checked against the binding before the run creates anything."""
    with pytest.raises(ValueError, match="not a numeric CASPAR solver option"):
        parse_caspar_options(("solver_iterations=10",))


def test_gpu_index_is_not_settable_here() -> None:
    """`gpu_index` is a string owned by `MappingOptions.use_gpu`, so it is not an override."""
    assert "gpu_index" not in caspar_option_types()
    with pytest.raises(ValueError, match="not a numeric CASPAR solver option"):
        parse_caspar_options(("gpu_index=0",))


def test_an_item_without_an_equals_sign_is_refused() -> None:
    """The flag takes `name=value`; anything else is a typo."""
    with pytest.raises(ValueError, match="takes `name=value`"):
        parse_caspar_options(("solver_iter_max",))


def test_a_non_numeric_value_is_refused() -> None:
    """Every settable CASPAR knob is a number."""
    with pytest.raises(ValueError, match="needs a number"):
        parse_caspar_options(("solver_iter_max=many",))


def test_without_a_mode_the_reduced_camera_system_keeps_its_own_knobs() -> None:
    """No mode and no full-system knob: the build's default (the reduced camera system) runs, with its knobs as given."""
    assert resolve_caspar_mode({"reduced_pcg_iter_max": 200, "solver_iter_max": 50}) == {"reduced_pcg_iter_max": 200, "solver_iter_max": 50}


@pytest.mark.parametrize("knob", ["pcg_iter_max", "pcg_rel_error_exit", "pcg_rel_score_exit", "pcg_rel_decrease_min"])
def test_a_full_system_knob_without_a_mode_selects_the_full_system(knob: str) -> None:
    """`pcg_iter_max=80` used to be accepted and then ignored by the reduced default; it now means the full system."""
    assert resolve_caspar_mode({knob: 1})["reduced_camera_system"] is False


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"reduced_camera_system": True, "pcg_iter_max": 80}, "apply to the full-system PCG only"),
        ({"reduced_camera_system": False, "reduced_pcg_eta": 0.2}, "apply to the reduced camera system only"),
        ({"pcg_iter_max": 80, "reduced_pcg_iter_max": 200}, "sets both the full-system"),
        ({"reduced_pcg_iter_max": 0}, "reduced_pcg_iter_max must be at least 1"),
        ({"reduced_pcg_eta": 1.0}, r"reduced_pcg_eta must be in \(0, 1\)"),
        ({"reduced_pcg_eta": float("nan")}, r"reduced_pcg_eta must be in \(0, 1\)"),
        ({"reduced_pcg_eta": 0.99999999}, r"reduced_pcg_eta must be in \(0, 1\)"),
    ],
    ids=["reduced-with-full-knob", "full-with-reduced-knob", "both-families", "zero-cap", "eta-one", "eta-nan", "eta-rounds-to-one-in-f32"],
)
def test_contradictory_or_invalid_modes_are_refused_before_the_run(options: CasparOptions, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_caspar_mode(options)


def test_the_reduced_camera_system_switch_parses_as_a_bool() -> None:
    """The one bool solver control `--caspar-option` takes: true/false/1/0, anything else refused."""
    for raw, expected in (("false", False), ("0", False), ("True", True), ("1", True)):
        parsed: CasparOptions = parse_caspar_options((f"reduced_camera_system={raw}",))
        assert parsed == {"reduced_camera_system": expected} and isinstance(parsed["reduced_camera_system"], bool)
    with pytest.raises(ValueError, match="takes true or false"):
        parse_caspar_options(("reduced_camera_system=0.5",))
    assert parse_caspar_options(("pcg_iter_max=80",)) == {"pcg_iter_max": 80, "reduced_camera_system": False}


# ─────────────────────────────────────────────────────────────────────────────
# capability, as a typed outcome
# ─────────────────────────────────────────────────────────────────────────────


def test_a_measured_build_guards_on_its_camera_models() -> None:
    """Only a probe that found real support may claim a model is unsupported."""
    assert STOCK_CAPABILITY.guards_camera_models
    assert STOCK_CAPABILITY.unsupported(["PINHOLE", "OPENCV_FISHEYE"]) == frozenset({"OPENCV_FISHEYE"})


@pytest.mark.parametrize("capability", [NO_BUILD_CAPABILITY, BROKEN_PROBE_CAPABILITY])
def test_an_unmeasurable_build_guards_nothing(capability: CasparCapability) -> None:
    """No CASPAR and a failed probe both mean the guard has nothing to say."""
    assert not capability.guards_camera_models
    assert capability.unsupported(["OPENCV_FISHEYE"]) == frozenset()


def test_the_three_outcomes_are_distinguishable() -> None:
    """The empty set used to be all three; the availability field separates them."""
    assert NO_BUILD_CAPABILITY.availability != BROKEN_PROBE_CAPABILITY.availability
    assert BROKEN_PROBE_CAPABILITY.probe_errors, "a failed probe keeps the error that failed it"


# ─────────────────────────────────────────────────────────────────────────────
# resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_a_ceres_request_is_never_second_guessed() -> None:
    """Nothing about CASPAR is consulted when Ceres was asked for."""
    assert resolve_backend("ceres", optimize_extrinsics=True) == ("ceres", None)


def test_free_extrinsics_fall_back_to_ceres_with_a_reason() -> None:
    """A CASPAR that holds `sensor_from_rig` throws when asked to refine it, so Ceres takes the solve."""
    assert resolve_backend("caspar", optimize_extrinsics=True, capability=STOCK_CAPABILITY) == ("ceres", EXTRINSICS_FALLBACK_REASON)


def test_free_extrinsics_stay_on_caspar_when_the_build_refines_the_rig() -> None:
    """A build whose probe refined a two-sensor rig keeps the solve on the GPU.

    Measured on Mission office (rig + per-face focals free in every round): 46 s of mapping on CASPAR against 383 s on
    Ceres, poses 0.018 deg apart.
    """
    assert resolve_backend("caspar", optimize_extrinsics=True, camera_model_names=["PINHOLE"], capability=RIG_CAPABILITY) == ("caspar", None)
    assert resolve_backend("caspar", optimize_extrinsics=True, camera_model_names=["FULL_OPENCV"], capability=RIG_CAPABILITY)[0] == "ceres"


def test_an_unsupported_camera_model_falls_back_and_names_both_sets() -> None:
    """The observations CASPAR would silently drop are the reason, and it says which."""
    backend, reason = resolve_backend(
        "caspar",
        optimize_extrinsics=False,
        camera_model_names=["PINHOLE", "OPENCV_FISHEYE"],
        capability=STOCK_CAPABILITY,
    )
    assert backend == "ceres"
    assert reason is not None
    assert "OPENCV_FISHEYE" in reason and "PINHOLE" in reason


def test_a_supported_model_keeps_the_gpu_backend() -> None:
    """Nothing falls back when the build projects every model in the reconstruction."""
    assert resolve_backend(
        "caspar", optimize_extrinsics=False, camera_model_names=["PINHOLE"], capability=STOCK_CAPABILITY
    ) == ("caspar", None)


@pytest.mark.parametrize(
    ("capability", "expected_reason"),
    [(NO_BUILD_CAPABILITY, CASPAR_BUILD_FALLBACK_REASON), (BROKEN_PROBE_CAPABILITY, CASPAR_PROBE_FAILED_REASON)],
)
def test_a_build_that_cannot_be_used_falls_back_before_the_run_starts(
    capability: CasparCapability, expected_reason: str
) -> None:
    """A CASPAR that has not been shown to solve is refused here, not by a mid-run throw.

    The probe finds this out *by solving*, so the run knows before it writes anything.
    `colsfm.mapping` used to let the request stand, catch COLMAP's "built without
    CASPAR_ENABLED" at the first bundle adjustment, rewrite the caller's options object
    in place and then infer the reason by comparing backends.
    """
    assert resolve_backend(
        "caspar", optimize_extrinsics=False, camera_model_names=["OPENCV_FISHEYE"], capability=capability
    ) == ("ceres", expected_reason)


# ─────────────────────────────────────────────────────────────────────────────
# the execution plan
# ─────────────────────────────────────────────────────────────────────────────


def test_the_plan_validates_before_it_resolves() -> None:
    """A bad override stops the run whatever the backend would have been."""
    with pytest.raises(ValueError, match="integer solver knob"):
        resolve_ba_plan("caspar", ("solver_iter_max=2.9",), ceres_polish=True, optimize_extrinsics=False)


def test_a_ceres_plan_carries_no_polish() -> None:
    """Every fallback lands on a solver that has converged, so there is nothing to polish."""
    plan: BaExecutionPlan = resolve_ba_plan("ceres", (), ceres_polish=True, optimize_extrinsics=False)
    assert plan.backend == "ceres"
    assert not plan.ceres_polish
    assert not plan.fell_back


def test_a_fallback_plan_records_what_was_asked_for_and_why_it_was_refused() -> None:
    """The request, the effective backend and the reason all survive into the plan."""
    plan: BaExecutionPlan = resolve_ba_plan("caspar", (), ceres_polish=True, optimize_extrinsics=True, capability=STOCK_CAPABILITY)
    assert plan.requested_backend == "caspar"
    assert plan.backend == "ceres"
    assert plan.fell_back
    assert plan.fallback_reason == EXTRINSICS_FALLBACK_REASON
    assert not plan.ceres_polish


def test_the_build_fallback_reason_names_the_environment_that_fixes_it() -> None:
    """A run that lands on Ceres for want of a build must say how to get one."""
    assert "recipes/colmap-src" in CASPAR_BUILD_FALLBACK_REASON
