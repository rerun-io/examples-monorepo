"""The `mojo` bundle-adjustment backend: its plan logic without a GPU, and one mapping run against CASPAR's with one.

The capability is stated (or the probe solve monkeypatched), as `test_ba_backend.py` does for CASPAR, so the unit
lane needs neither the extension nor a GPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pycolmap
import pytest

if TYPE_CHECKING:
    from mapping_helpers import SyntheticRig  # pyrefly: ignore[missing-import]

    from colsfm.ba_backend import BaExecutionPlan, CasparCapability
    from colsfm.config import CusfmConfig


def _pinhole_only() -> CasparCapability:
    from colsfm.ba_backend import CasparCapability

    return CasparCapability(
        availability="available",
        supported_camera_models=frozenset({"PINHOLE"}),
        probe_errors=("SIMPLE_RADIAL: camera model 1 is not ported yet",),
        refines_sensor_from_rig=True,
        intrinsics_refinable_camera_models=frozenset({"PINHOLE"}),
    )


def test_a_mojo_plan_measures_the_mojo_solver_not_caspar(monkeypatch: pytest.MonkeyPatch) -> None:
    from colsfm import ba_backend

    def no_caspar() -> CasparCapability:
        raise AssertionError("a mojo plan must not probe CASPAR")

    monkeypatch.setattr(ba_backend, "detect_caspar_capability", no_caspar)
    monkeypatch.setattr(ba_backend, "detect_mojo_capability", _pinhole_only)
    plan: BaExecutionPlan = ba_backend.resolve_ba_plan("mojo", ("solver_iter_max=7",), ceres_polish=True, optimize_extrinsics=True)
    assert (plan.backend, plan.fallback_reason, plan.ceres_polish) == ("mojo", None, True)
    assert plan.caspar_options == {"solver_iter_max": 7}


@pytest.mark.parametrize(
    ("availability", "reason_name"), [("unavailable", "MOJO_BUILD_FALLBACK_REASON"), ("probe_failed", "MOJO_PROBE_FAILED_REASON")]
)
def test_a_mojo_solver_that_cannot_solve_falls_back_with_its_own_reason(availability: str, reason_name: str) -> None:
    from colsfm import ba_backend

    capability = ba_backend.CasparCapability(availability=availability, supported_camera_models=frozenset(), probe_errors=("x",))  # type: ignore[arg-type]
    plan: BaExecutionPlan = ba_backend.resolve_ba_plan("mojo", (), ceres_polish=True, optimize_extrinsics=False, capability=capability)
    assert (plan.requested_backend, plan.backend, plan.ceres_polish) == ("mojo", "ceres", False)
    assert plan.fallback_reason == getattr(ba_backend, reason_name)


def test_free_extrinsics_fall_back_when_the_mojo_probe_did_not_refine_a_rig() -> None:
    from dataclasses import replace

    from colsfm.ba_backend import MOJO_EXTRINSICS_FALLBACK_REASON, resolve_backend

    no_rig = replace(_pinhole_only(), refines_sensor_from_rig=False)
    assert resolve_backend("mojo", optimize_extrinsics=True, capability=no_rig) == ("ceres", MOJO_EXTRINSICS_FALLBACK_REASON)
    assert resolve_backend("mojo", optimize_extrinsics=True, capability=_pinhole_only()) == ("mojo", None)


def test_the_camera_models_narrow_a_mojo_plan() -> None:
    from colsfm.ba_backend import MOJO_MIXED_MODELS_FALLBACK_REASON, CasparCapability, resolve_ba_plan

    plan: BaExecutionPlan = resolve_ba_plan("mojo", (), ceres_polish=False, optimize_extrinsics=False, capability=_pinhole_only())
    assert plan.for_camera_models(["PINHOLE"], refine_intrinsics=True) is plan
    unported: BaExecutionPlan = plan.for_camera_models(["SIMPLE_RADIAL"])
    assert unported.backend == "ceres" and unported.fallback_reason is not None
    assert "Mojo" in unported.fallback_reason and "SIMPLE_RADIAL" in unported.fallback_reason

    two_models = CasparCapability(availability="available", supported_camera_models=frozenset({"PINHOLE", "SIMPLE_RADIAL"}), probe_errors=())
    mixed = resolve_ba_plan("mojo", (), ceres_polish=False, optimize_extrinsics=False, capability=two_models)
    # Two rigs, one model each: CASPAR would take it, the Mojo solver takes one model per solve.
    narrowed = mixed.for_camera_models(["PINHOLE", "SIMPLE_RADIAL"], rig_camera_models=[frozenset({"PINHOLE"}), frozenset({"SIMPLE_RADIAL"})])
    assert narrowed.fallback_reason == MOJO_MIXED_MODELS_FALLBACK_REASON


@pytest.mark.parametrize("items", [("reduced_camera_system=false",), ("pcg_iter_max=40",)])
def test_the_full_system_pcg_is_refused_for_mojo(items: tuple[str, ...]) -> None:
    from colsfm.ba_backend import resolve_ba_plan

    with pytest.raises(ValueError, match="reduced camera system only"):
        resolve_ba_plan("mojo", items, ceres_polish=False, optimize_extrinsics=False, capability=_pinhole_only())


def test_the_mojo_probe_reads_which_models_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    from colsfm import ba_capability

    def probe(model_name: str, *, num_sensors: int = 1, refine_intrinsics: bool = False, **_: bool) -> pycolmap.BundleAdjustmentSummary:
        if model_name != "PINHOLE":
            raise RuntimeError(f"{model_name} is not ported yet")
        summary = pycolmap.BundleAdjustmentSummary()
        summary.num_residuals = 64 * num_sensors
        summary.termination_type = pycolmap.BundleAdjustmentTerminationType.NO_CONVERGENCE
        return summary

    monkeypatch.setitem(sys.modules, "caspar_mojo._caspar_mojo", types.ModuleType("caspar_mojo._caspar_mojo"))
    monkeypatch.setattr(ba_capability, "_mojo_probe_solve", probe)
    ba_capability.detect_mojo_capability.cache_clear()
    try:
        capability: CasparCapability = ba_capability.detect_mojo_capability()
    finally:
        ba_capability.detect_mojo_capability.cache_clear()
    assert capability.availability == "available"
    assert capability.supported_camera_models == frozenset({"PINHOLE"})
    assert capability.refines_sensor_from_rig and capability.intrinsics_refinable_camera_models == frozenset({"PINHOLE"})
    assert len(capability.probe_errors) == len(ba_capability.CASPAR_PROBE_CAMERA_MODELS) - 1


def _summary(num_residuals: int, termination: str) -> pycolmap.BundleAdjustmentSummary:
    summary = pycolmap.BundleAdjustmentSummary()
    summary.num_residuals = num_residuals
    summary.termination_type = getattr(pycolmap.BundleAdjustmentTerminationType, termination)
    return summary


def test_a_mojo_probe_whose_solves_all_fail_is_a_probe_failure_with_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from colsfm import ba_capability

    def probe(model_name: str, **_: object) -> pycolmap.BundleAdjustmentSummary:
        return _summary(64, "FAILURE")

    monkeypatch.setitem(sys.modules, "caspar_mojo._caspar_mojo", types.ModuleType("caspar_mojo._caspar_mojo"))
    monkeypatch.setattr(ba_capability, "_mojo_probe_solve", probe)
    ba_capability.detect_mojo_capability.cache_clear()
    try:
        capability: CasparCapability = ba_capability.detect_mojo_capability()
    finally:
        ba_capability.detect_mojo_capability.cache_clear()
    assert capability.availability == "probe_failed"
    assert len(capability.probe_errors) == len(ba_capability.CASPAR_PROBE_CAMERA_MODELS)
    assert all("FAILURE" in error for error in capability.probe_errors)


def test_a_failed_refinement_probe_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    from colsfm import ba_capability

    def probe(model_name: str, *, num_sensors: int = 1, refine_intrinsics: bool = False, **_: object) -> pycolmap.BundleAdjustmentSummary:
        if model_name != "PINHOLE":
            raise RuntimeError(f"{model_name} is not ported yet")
        if num_sensors > 1:
            raise RuntimeError("rig factor missing")
        return _summary(64, "FAILURE" if refine_intrinsics else "CONVERGENCE")

    monkeypatch.setitem(sys.modules, "caspar_mojo._caspar_mojo", types.ModuleType("caspar_mojo._caspar_mojo"))
    monkeypatch.setattr(ba_capability, "_mojo_probe_solve", probe)
    ba_capability.detect_mojo_capability.cache_clear()
    try:
        capability: CasparCapability = ba_capability.detect_mojo_capability()
    finally:
        ba_capability.detect_mojo_capability.cache_clear()
    assert capability.availability == "available" and not capability.refines_sensor_from_rig
    assert not capability.intrinsics_refinable_camera_models
    assert any("rig factor missing" in error for error in capability.probe_errors)
    assert any("PINHOLE" in error and "intrinsics" in error and "FAILURE" in error for error in capability.probe_errors)


def test_resolve_backend_and_the_plan_agree_on_a_two_model_reconstruction() -> None:
    from colsfm.ba_backend import MOJO_MIXED_MODELS_FALLBACK_REASON, CasparCapability, resolve_ba_plan, resolve_backend

    both = CasparCapability(
        availability="available",
        supported_camera_models=frozenset({"PINHOLE", "SIMPLE_RADIAL"}),
        probe_errors=(),
        refines_sensor_from_rig=True,
        intrinsics_refinable_camera_models=frozenset({"PINHOLE", "SIMPLE_RADIAL"}),
    )
    models: tuple[str, ...] = ("PINHOLE", "SIMPLE_RADIAL")
    plan: BaExecutionPlan = resolve_ba_plan("mojo", (), ceres_polish=False, optimize_extrinsics=False, capability=both).for_camera_models(models)
    assert (plan.backend, plan.fallback_reason) == ("ceres", MOJO_MIXED_MODELS_FALLBACK_REASON)
    assert resolve_backend("mojo", optimize_extrinsics=False, camera_model_names=models, capability=both) == ("ceres", MOJO_MIXED_MODELS_FALLBACK_REASON)


@pytest.mark.parametrize(
    ("exit_reason", "final_cost", "termination"),
    [("CONVERGED_DIAG_EXIT", 1.0, "NO_CONVERGENCE"), ("CONVERGED_SCORE_THRESHOLD", 1.0, "CONVERGENCE"), ("MAX_ITERATIONS", 3.0, "FAILURE")],
)
def test_a_mojo_solve_reports_colmaps_termination(exit_reason: str, final_cost: float, termination: str) -> None:
    from caspar_mojo.problem import BaResult

    from colsfm.ba_mojo import MojoSolveSummary

    result = BaResult(
        intrinsics=np.zeros((1, 12)),
        sensor_from_rig=np.zeros((1, 7)),
        rig_from_world=np.zeros((1, 7)),
        points=np.zeros((1, 3)),
        initial_cost=2.0,
        final_cost=final_cost,
        iterations=3,
        pcg_iterations=9,
        exit_reason=exit_reason,
        solve_seconds=0.0,
        total_seconds=0.0,
    )
    summary = MojoSolveSummary(result=result, num_residuals=8, extract_seconds=0.0, solve_seconds=0.0, apply_seconds=0.0).bundle_adjustment_summary()
    assert summary.termination_type.name == termination and summary.num_residuals == 8


def test_a_mojo_plan_builds_caspars_options_with_the_overrides(isaac_config: CusfmConfig) -> None:
    from mapping_helpers import quiet_options  # pyrefly: ignore[missing-import]

    from colsfm.ba_backend import backend_name, resolve_ba_plan
    from colsfm.mapping import bundle_adjustment_options

    plan: BaExecutionPlan = resolve_ba_plan(
        "mojo", ("reduced_pcg_eta=0.25",), ceres_polish=False, optimize_extrinsics=False, capability=_pinhole_only()
    )
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options(ba_plan=plan)
    )
    assert backend_name(ba_options) == "caspar"
    assert ba_options.caspar.reduced_camera_system and ba_options.caspar.reduced_pcg_eta == pytest.approx(0.25)


@pytest.mark.integration
def test_the_mapper_on_mojo_matches_the_mapper_on_caspar(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig, caspar_enabled: bool
) -> None:
    from mapping_helpers import quiet_options  # pyrefly: ignore[missing-import]

    from colsfm.ba_backend import resolve_ba_plan
    from colsfm.mapping import MappingResult, run_mapping
    from colsfm.reconstruction import build_reconstruction
    from colsfm.run_report import MappingStats

    if not caspar_enabled:
        pytest.skip("this pycolmap has no CASPAR to compare with")
    mojo_plan: BaExecutionPlan = resolve_ba_plan("mojo", (), ceres_polish=False, optimize_extrinsics=False)
    # Only a missing extension skips; a probe that ran and refused the PINHOLE rig is the regression this test is for.
    if mojo_plan.capability.availability == "unavailable":
        pytest.skip(f"the caspar_mojo extension is not built here: {mojo_plan.capability.probe_errors}")
    assert mojo_plan.backend == "mojo", f"{mojo_plan.fallback_reason} {mojo_plan.capability.probe_errors}"
    caspar_plan: BaExecutionPlan = resolve_ba_plan("caspar", (), ceres_polish=False, optimize_extrinsics=False)
    mapping_config = isaac_config.vision_mapping
    caspar: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta), synthetic_database, mapping_config, quiet_options(ba_plan=caspar_plan)
    )
    mojo: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta), synthetic_database, mapping_config, quiet_options(ba_plan=mojo_plan)
    )
    stats: MappingStats = MappingStats.of(mojo)
    assert (mojo.ba_backend, stats.ba_backend, stats.ba_fallback_reason) == ("mojo", "mojo", None)
    assert mojo.num_registered_images == caspar.num_registered_images
    assert mojo.num_points3D == pytest.approx(caspar.num_points3D, rel=0.01)
    assert mojo.mean_reprojection_error_px == pytest.approx(caspar.mean_reprojection_error_px, rel=0.02)
