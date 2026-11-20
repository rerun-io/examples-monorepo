"""Which bundle adjuster a mapping run gets, and the Ceres polish that ends it.

The policy `test_ba_backend.py` states as a pure contract, observed here where it
is actually applied: `bundle_adjustment_options` reading a real
`pycolmap.Reconstruction`, and `run_mapping` finishing a run on whatever backend
survived the pre-flight. That needs a rig, a database and a CASPAR build to be
present or absent, which is why it is not in the pure-contract file — and why it
is not in `test_mapping.py` either, whose subject is the map rather than the solver.

Covered here: the three CASPAR fallbacks (camera model, extrinsics refinement, a
build without CASPAR_ENABLED), the capability probe, the solver-option cast, and
the single Ceres bundle adjustment that finishes a CASPAR run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from mapping_helpers import (
    SyntheticRig,
    build_synthetic_rig,
    caspar_plan,
    match_tracks_to_truth,
    quiet_options,
    synthetic_matches,
    write_database,
)

from colsfm import ba_backend
from colsfm.ba_backend import (
    CASPAR_BUILD_FALLBACK_REASON,
    CASPAR_PROBE_CAMERA_MODELS,
    CASPAR_STOCK_CAMERA_MODELS,
    BaExecutionPlan,
    apply_caspar_options,
    backend_name,
    caspar_supported_camera_models,
    detect_caspar_capability,
    resolve_ba_plan,
)
from colsfm.config import BundleAdjustmentConfig, CusfmConfig, VisionMappingConfig
from colsfm.mapping import MappingOptions, MappingResult, bundle_adjustment_options, ceres_polish_options, run_mapping
from colsfm.reconstruction import build_reconstruction
from colsfm.run_config import PipelineOptions
from colsfm.run_report import MappingStats


@pytest.fixture(scope="module")
def fisheye_rig() -> SyntheticRig:
    """A 6-frame, two-camera OPENCV_FISHEYE rig: the model stock CASPAR cannot project."""
    return build_synthetic_rig(num_frames=6, num_points=200, fisheye=True, seed=3)


@pytest.fixture(scope="module")
def fisheye_database(fisheye_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fisheye rig's observations as a COLMAP database."""
    database_path: Path = tmp_path_factory.mktemp("fisheye") / "database.db"
    write_database(database_path, fisheye_rig.frames_meta, fisheye_rig.keypoints_px, synthetic_matches(fisheye_rig))
    return database_path


def test_the_caspar_backend_reaches_the_options_the_solver_reads(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """A PINHOLE rig under a CASPAR plan gets CASPAR and the settings it demands.

    CASPAR throws on a free `sensor_from_rig` and on a focal length refined apart
    from the distortion block, so the options builder owes all three: the backend,
    the frozen extrinsics and the two refinement flags in agreement.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    assert {camera.model.name for camera in reconstruction.cameras.values()} == {"PINHOLE"}

    plan: BaExecutionPlan = caspar_plan(supported_camera_models=CASPAR_STOCK_CAMERA_MODELS)
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options(ba_plan=plan)
    )
    assert backend_name(ba_options) == "caspar"
    assert ba_options.refine_sensor_from_rig is False
    assert ba_options.refine_extra_params == ba_options.refine_focal_length
    assert ba_options.caspar.gpu_index == "-1"
    on_gpu: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options(ba_plan=plan, use_gpu=True)
    )
    assert on_gpu.caspar.gpu_index == "0"


def test_caspar_solver_options_reach_the_options_object(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """`BaExecutionPlan.caspar_options` lands on `BundleAdjustmentOptions.caspar`, cast.

    `--caspar-option` moves CASPAR's stopping rules from the command line, so the dict has to arrive at the solver intact — and the
    two iteration counters are C++ ints, which pybind11 will not take a float for.
    """
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment,
        quiet_options(
            ba_plan=caspar_plan(
                caspar_option_items=("solver_iter_max=1000", "pcg_iter_max=80.0", "pcg_rel_error_exit=1e-6"),
                supported_camera_models=CASPAR_STOCK_CAMERA_MODELS,
            )
        ),
    )
    assert backend_name(ba_options) == "caspar"
    assert ba_options.caspar.solver_iter_max == 1000
    assert isinstance(ba_options.caspar.solver_iter_max, int)
    assert ba_options.caspar.pcg_iter_max == 80
    assert isinstance(ba_options.caspar.pcg_iter_max, int)
    assert ba_options.caspar.pcg_rel_error_exit == pytest.approx(1e-6)
    # Untouched knobs keep COLMAP's defaults, and `use_gpu` still owns `gpu_index`.
    defaults: dict[str, object] = pycolmap.BundleAdjustmentOptions().caspar.todict()
    assert ba_options.caspar.diag_init == defaults["diag_init"]
    assert ba_options.caspar.gpu_index == "-1"


def test_caspar_solver_options_are_inert_on_ceres_and_reject_unknown_names(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """Ceres never reads the CASPAR block, and a misspelt knob is an error, not a no-op."""
    inert: BaExecutionPlan = resolve_ba_plan(
        "ceres", ("solver_iter_max=1000",), ceres_polish=True, optimize_extrinsics=False
    )
    assert inert.caspar_options == {"solver_iter_max": 1000}
    on_ceres: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options(ba_plan=inert)
    )
    assert backend_name(on_ceres) == "ceres"
    assert on_ceres.caspar.solver_iter_max == pycolmap.BundleAdjustmentOptions().caspar.solver_iter_max

    with pytest.raises(KeyError, match="solver_iter_maxx"):
        apply_caspar_options(pycolmap.BundleAdjustmentOptions().caspar, {"solver_iter_maxx": 1.0})
    with pytest.raises(KeyError, match="gpu_index"):
        apply_caspar_options(pycolmap.BundleAdjustmentOptions().caspar, {"gpu_index": 0.0})


def test_the_default_backend_is_ceres_and_leaves_caspar_alone(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """Without the flag nothing about the solve changes: Ceres, as every run before."""
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options()
    )
    assert backend_name(ba_options) == "ceres"
    assert ba_options.caspar.gpu_index == "-1"


def test_a_camera_model_this_caspar_build_cannot_project_falls_back_to_ceres(
    isaac_config: CusfmConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """An OPENCV_FISHEYE rig goes to Ceres on a build whose adapters stop at SIMPLE_RADIAL.

    CASPAR skips the observations of a model it has no adapter for with a log line
    and still reports success, so a silent half-problem is what the fallback exists
    to prevent. The supported set is stated rather than probed, so the test states
    one rule — model not in the set means Ceres — on every environment.

    This is the one decision resolution cannot make: the camera models arrive with
    the reconstruction. `for_camera_models` is where it is made, once, and the plan
    it returns is what the mapper executes and what `summary.json` reports.
    """
    rig: SyntheticRig = build_synthetic_rig(num_frames=4, num_points=50, fisheye=True, seed=3)
    reconstruction: pycolmap.Reconstruction = build_reconstruction(rig.frames_meta).reconstruction

    planned: BaExecutionPlan = caspar_plan(supported_camera_models=CASPAR_STOCK_CAMERA_MODELS)
    assert planned.backend == "caspar", "nothing is known about the cameras until the model exists"
    narrowed: BaExecutionPlan = planned.for_camera_models(
        camera.model.name for camera in reconstruction.cameras.values()
    )
    assert narrowed.backend == "ceres"
    assert not narrowed.ceres_polish, "a Ceres run has converged; there is nothing to polish"
    assert narrowed.fallback_reason is not None
    assert "OPENCV_FISHEYE" in narrowed.fallback_reason
    # The reason has to say what this build *can* do, or the reader cannot tell a
    # missing adapter from a missing build.
    assert "SIMPLE_RADIAL" in narrowed.fallback_reason
    narrowed.announce()
    assert "OPENCV_FISHEYE" in capsys.readouterr().out

    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options(ba_plan=narrowed)
    )
    assert backend_name(ba_options) == "ceres"


def test_a_fisheye_rig_stays_on_caspar_when_the_build_has_the_adapter(
    isaac_config: CusfmConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same rig keeps CASPAR once OPENCV_FISHEYE is in the supported set.

    This is the whole point of making the pre-flight capability-aware: the
    `colsfm` build (recipes/colmap-src) projects OPENCV_FISHEYE, so a fisheye rig must
    not be sent to Ceres there.
    """
    rig: SyntheticRig = build_synthetic_rig(num_frames=4, num_points=50, fisheye=True, seed=3)
    reconstruction: pycolmap.Reconstruction = build_reconstruction(rig.frames_meta).reconstruction

    narrowed: BaExecutionPlan = caspar_plan(
        supported_camera_models=CASPAR_STOCK_CAMERA_MODELS | {"OPENCV_FISHEYE"}
    ).for_camera_models(camera.model.name for camera in reconstruction.cameras.values())
    assert narrowed.backend == "caspar"
    assert narrowed.fallback_reason is None
    narrowed.announce()
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options(ba_plan=narrowed)
    )
    assert backend_name(ba_options) == "caspar"
    assert "OPENCV_FISHEYE" not in capsys.readouterr().out


def test_the_probe_reports_the_camera_models_this_caspar_build_projects(
    caspar_enabled: bool, pixi_environment_name: str
) -> None:
    """`caspar_supported_camera_models` reads the build, not a hard-coded list.

    Stock CASPAR (COLMAP 4.2.0) ships adapters for PINHOLE and SIMPLE_RADIAL only;
    the `colsfm` environments (recipes/colmap-src) add OPENCV_FISHEYE.
    The expectation comes from the environment name, so it is independent of the
    solve the probe runs to find out.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-dev`")
    supported: frozenset[str] = caspar_supported_camera_models()
    print(f"[colsfm] {pixi_environment_name or 'unknown env'} CASPAR projects {sorted(supported)}")

    assert supported >= CASPAR_STOCK_CAMERA_MODELS
    if pixi_environment_name in {"colsfm", "colsfm-dev"}:
        assert "OPENCV_FISHEYE" in supported
    else:
        assert supported == CASPAR_STOCK_CAMERA_MODELS


def test_the_probe_solves_once_however_often_it_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is cached, so the mapper pays for its CASPAR solves once per process.

    The expensive thing is `_caspar_projects`: one real bundle adjustment per
    candidate camera model. The contract is that the whole sweep runs once and
    every later caller reads the answer, so the test counts the solves. Timing
    the second call instead — which is what this used to do, against a 10 ms
    budget — measures the machine, and asking whether two calls returned the
    same object measures `functools.cache` rather than the mapper's cost.

    Counting also frees the test from needing a CASPAR build: the stand-in
    answers for the solve, so the caching contract is checked on every
    environment rather than only where the GPU backend exists.
    """
    asked: list[str] = []

    def counting_probe(model_name: str) -> bool:
        """Stand in for one CASPAR solve, recording which model it was asked about."""
        asked.append(model_name)
        return model_name in CASPAR_STOCK_CAMERA_MODELS

    monkeypatch.setattr(ba_backend, "_caspar_projects", counting_probe)
    # The real answer is memoised for the life of the process, so the cache has
    # to be empty going in and empty again coming out — otherwise this test
    # either reads a real probe or leaves the stand-in's answer behind.
    detect_caspar_capability.cache_clear()
    try:
        first: frozenset[str] = caspar_supported_camera_models()
        repeats: list[frozenset[str]] = [caspar_supported_camera_models() for _ in range(3)]
    finally:
        detect_caspar_capability.cache_clear()

    assert asked == list(CASPAR_PROBE_CAMERA_MODELS), "the sweep runs once, in probe order"
    assert first == CASPAR_STOCK_CAMERA_MODELS
    assert repeats == [first, first, first]


def test_the_probe_reports_nothing_without_a_caspar_build(caspar_enabled: bool) -> None:
    """A pycolmap without CASPAR_ENABLED supports no model, so every rig falls back."""
    if caspar_enabled:
        pytest.skip("this pycolmap has CASPAR; the empty answer only exists on the stock build")
    assert caspar_supported_camera_models() == frozenset()


def test_refining_extrinsics_falls_back_to_ceres(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mapper asked to free `sensor_from_rig` and CASPAR are exclusive; the request wins.

    CASPAR hard-throws on a free `sensor_from_rig` in a multi-sensor frame. Forcing
    the flag off instead would turn an asked-for refinement into a silent no-op.

    This is `MappingOptions.optimize_extrinsics`, not `--optimize-extrinsics`: a
    default run refines its extrinsics *and* keeps CASPAR, because the regularised
    refinement moves them outside the bundle adjuster. What lands here is the
    `--no-regularised-extrinsics` ablation, and the message says so.
    """
    plan: BaExecutionPlan = caspar_plan(
        optimize_extrinsics=True, supported_camera_models=CASPAR_STOCK_CAMERA_MODELS
    )
    assert plan.backend == "ceres"
    ba_options: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(
        isaac_config.vision_mapping.bundle_adjustment, quiet_options(ba_plan=plan, optimize_extrinsics=True)
    )
    assert backend_name(ba_options) == "ceres"
    assert ba_options.refine_sensor_from_rig is True
    plan.announce()
    assert "no-regularised-extrinsics" in capsys.readouterr().out


def test_the_mapper_runs_on_caspar_or_reports_the_build_that_cannot(
    synthetic_rig: SyntheticRig,
    synthetic_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--ba-backend caspar` maps the synthetic rig, in either environment.

    On a CASPAR-enabled pycolmap the GPU backend solves and the map matches the
    Ceres one; on the stock one the missing capability is reported once and Ceres
    finishes the run, so the flag never costs a result. The map itself is checked
    against the ground-truth points, not against the solver's own opinion.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    ceres: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta), synthetic_database, mapping_config, quiet_options()
    )
    capsys.readouterr()
    caspar: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        mapping_config,
        quiet_options(ba_plan=caspar_plan()),
    )
    printed: str = capsys.readouterr().out

    # What a finished run *says* about the backend, which is the only record left
    # once stdout is gone: `--ba-backend caspar` is the default since 2026-09-06, so
    # in an environment without the build every summary carries this sentence.
    stats: MappingStats = MappingStats.of(caspar)

    if not caspar_enabled:
        assert caspar.ba_backend == "ceres"
        assert caspar.num_points3D == ceres.num_points3D
        assert stats.ba_backend == "ceres"
        assert stats.ba_fallback_reason == CASPAR_BUILD_FALLBACK_REASON
        assert MappingStats.of(ceres).ba_fallback_reason is None
        return

    assert caspar.ba_backend == "caspar"
    assert "CASPAR_ENABLED" not in printed, "nothing fell back, so nothing is announced"
    assert stats.ba_backend == "caspar"
    assert stats.ba_fallback_reason is None
    recovered, mixed_tracks, relative_errors = match_tracks_to_truth(caspar.reconstruction, synthetic_rig)
    print(
        f"[colsfm] caspar synthetic: {caspar.num_points3D} points against ceres' {ceres.num_points3D}, "
        f"reprojection {caspar.mean_reprojection_error_px:.4f} px against {ceres.mean_reprojection_error_px:.4f} px"
    )
    assert mixed_tracks == 0
    assert len(recovered) / len(synthetic_rig.points_xyz) >= 0.95
    assert float(np.median(relative_errors)) < 0.01
    assert caspar.num_registered_images == ceres.num_registered_images
    assert caspar.num_points3D == pytest.approx(ceres.num_points3D, rel=0.05)
    assert caspar.mean_reprojection_error_px == pytest.approx(ceres.mean_reprojection_error_px, abs=0.05)



def test_the_mapping_result_carries_the_plan_its_solves_ran_on(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig, caspar_enabled: bool
) -> None:
    """What `summary.json` reports is the object the solves executed, not a re-derivation.

    `run_mapping` used to resolve the backend three times — once to record it, once
    per `bundle_adjustment_options` call, and once more by *comparing* the two to
    guess a fallback reason — while the solve path could still rewrite the backend
    underneath all three. There is one plan now, and every reported field reads off it.
    """
    result: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        isaac_config.vision_mapping,
        quiet_options(ba_plan=caspar_plan()),
    )
    assert result.ba_plan.requested_backend == "caspar"
    assert result.ba_backend == result.ba_plan.backend
    assert result.ba_fallback_reason == result.ba_plan.fallback_reason
    assert (result.polish is not None) is result.ba_plan.ceres_polish

    stats: MappingStats = MappingStats.of(result)
    assert stats.ba_backend == result.ba_plan.backend
    assert stats.ba_fallback_reason == result.ba_plan.fallback_reason
    if caspar_enabled:
        assert result.ba_plan.backend == "caspar"
        assert result.ba_plan.fallback_reason is None
    else:
        assert result.ba_plan.backend == "ceres"
        assert result.ba_plan.fallback_reason == CASPAR_BUILD_FALLBACK_REASON


def test_a_fisheye_rig_maps_on_caspar_only_where_the_adapter_exists(
    fisheye_rig: SyntheticRig,
    fisheye_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
    pixi_environment_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end on an OPENCV_FISHEYE rig, the run the capability probe exists for.

    the `colsfm` build carries the adapter, so the rounds run on the GPU and the
    Ceres polish finishes them; a stock CASPAR build does not, so the pre-flight names the
    model and the whole run is Ceres, which has already converged and needs no polish.
    Either way the map is checked against the rig's own points, not against the solver.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-dev`")
    result: MappingResult = run_mapping(
        build_reconstruction(fisheye_rig.frames_meta),
        fisheye_database,
        isaac_config.vision_mapping,
        quiet_options(ba_plan=caspar_plan()),
    )
    printed: str = capsys.readouterr().out
    recovered, mixed_tracks, relative_errors = match_tracks_to_truth(result.reconstruction, fisheye_rig)
    print(
        f"[colsfm] fisheye rig on {pixi_environment_name or 'unknown env'}: backend {result.ba_backend}, "
        f"{result.num_points3D} points, {result.mean_reprojection_error_px:.4f} px"
    )
    assert mixed_tracks == 0
    assert len(recovered) / len(fisheye_rig.points_xyz) >= 0.95
    assert float(np.median(relative_errors)) < 0.01

    if pixi_environment_name in {"colsfm", "colsfm-dev"}:
        assert result.ba_backend == "caspar"
        assert "OPENCV_FISHEYE" not in printed
        assert result.polish is not None
        assert result.polish.num_observations > 0
        assert result.polish.mean_reprojection_error_after_px <= result.polish.mean_reprojection_error_before_px
    else:
        assert result.ba_backend == "ceres"
        assert "OPENCV_FISHEYE" in printed
        assert result.polish is None


def test_the_polish_option_set_is_the_ceres_one_built_from_the_same_config(
    synthetic_rig: SyntheticRig, isaac_config: CusfmConfig
) -> None:
    """`ceres_polish_options` differs from the CASPAR round options in the backend only.

    The polish was measured with colsfm's own solve — the config's loss and scale, its iteration cap, SPARSE_SCHUR
    and the same frozen extrinsics — so the polish has to be that solve and not a
    fresh set of defaults.
    """
    ba_config: BundleAdjustmentConfig = isaac_config.vision_mapping.bundle_adjustment
    options: MappingOptions = quiet_options(
        ba_plan=caspar_plan(supported_camera_models=CASPAR_STOCK_CAMERA_MODELS)
    )
    rounds: pycolmap.BundleAdjustmentOptions = bundle_adjustment_options(ba_config, options)
    polish: pycolmap.BundleAdjustmentOptions = ceres_polish_options(ba_config, options)

    assert backend_name(rounds) == "caspar"
    assert backend_name(polish) == "ceres"
    assert polish.ceres.loss_function_type == rounds.ceres.loss_function_type
    assert polish.ceres.loss_function_scale == pytest.approx(
        ba_config.reprojection_error_standard_deviation * ba_config.loss_function_scale
    )
    assert polish.ceres.solver_options.max_num_iterations == ba_config.max_num_iterations
    assert polish.ceres.solver_options.linear_solver_type == rounds.ceres.solver_options.linear_solver_type
    assert polish.refine_sensor_from_rig is False
    assert polish.refine_focal_length is False
    assert polish.refine_extra_params is False


def test_the_polish_is_off_by_default_and_switches_on() -> None:
    """The switch reaches the plan, and the default is the shipped behaviour: CASPAR alone (reduced camera system).

    `ceres_polish` is only ever True on a plan that ended on CASPAR: a Ceres run has
    already converged, so the plan zeroes the request rather than leaving the mapper
    to test the backend a second time.
    """
    assert PipelineOptions(input_dir=Path(), output_dir=Path()).caspar_ceres_polish is False
    assert caspar_plan(supported_camera_models=CASPAR_STOCK_CAMERA_MODELS).ceres_polish is True
    assert not caspar_plan(
        ceres_polish=False, supported_camera_models=CASPAR_STOCK_CAMERA_MODELS
    ).ceres_polish
    assert not MappingOptions().ba_plan.ceres_polish


def test_the_ceres_backend_never_polishes(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A Ceres mapping run has already converged, so nothing is appended to it.

    This also covers every CASPAR fallback: they all end with `ba_backend == "ceres"`,
    which is the condition the polish is keyed on.
    """
    result: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        isaac_config.vision_mapping,
        quiet_options(),
    )
    assert result.ba_backend == "ceres"
    assert result.polish is None


def test_the_caspar_run_finishes_with_one_ceres_bundle_adjustment(
    synthetic_rig: SyntheticRig,
    synthetic_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--ba-backend caspar` maps on the GPU and ends on one Ceres solve.

    The polish moves the model without touching the observation set, so the point
    and observation counts are the ones the rounds left behind and only the
    reprojection error may change.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-dev`")
    result: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        isaac_config.vision_mapping,
        quiet_options(ba_plan=caspar_plan(), verbose=True),
    )
    printed: str = capsys.readouterr().out

    assert result.ba_backend == "caspar"
    assert result.polish is not None
    assert result.polish.seconds > 0.0
    assert result.polish.mean_reprojection_error_after_px == pytest.approx(result.mean_reprojection_error_px)
    assert result.polish.num_observations == result.num_observations
    assert result.polish.ba_termination in {"CONVERGENCE", "NO_CONVERGENCE"}
    assert result.bundle_adjustment_seconds >= result.polish.seconds
    assert "polish" in printed
    print(
        f"[colsfm] caspar polish: {result.polish.report.num_iterations} iterations, "
        f"{result.polish.mean_reprojection_error_before_px:.4f} px -> "
        f"{result.polish.mean_reprojection_error_after_px:.4f} px in {result.polish.seconds:.2f} s"
    )


def test_the_polish_can_be_switched_off_for_an_ablation(
    synthetic_rig: SyntheticRig,
    synthetic_database: Path,
    isaac_config: CusfmConfig,
    caspar_enabled: bool,
) -> None:
    """A plan with `ceres_polish=False` leaves a CASPAR run exactly where CASPAR stopped."""
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-dev`")
    result: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        isaac_config.vision_mapping,
        quiet_options(ba_plan=caspar_plan(ceres_polish=False)),
    )
    assert result.ba_backend == "caspar"
    assert result.polish is None
