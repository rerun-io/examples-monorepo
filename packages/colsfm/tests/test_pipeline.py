"""The end-to-end pipeline, run once on Galileo at the isaac default keyframe spacing.

The seam is `colsfm.pipeline.run_pipeline`: one call in, one `PipelineSummary`
out plus a workspace on disk. Everything asserted here is observed through that
public surface — the files a cuSFM run leaves behind, the COLMAP model as
`pycolmap` reads it back, `runtime.csv` as `colsfm.export` reads it back, and
`summary.json` through pyserde — never through the pipeline's internals.

The fixture uses the stock `feature_extractor_main` gates (0.5 m / 5 deg), which
keep 34 of Galileo's 226 keyframes: enough for a real reconstruction, fast
enough for a smoke test. The 226-frame run is the benchmark, not a unit test.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
import pytest
from serde.json import from_json, to_json

from colsfm.ba_backend import CASPAR_BUILD_FALLBACK_REASON, EXTRINSICS_FALLBACK_REASON, CasparCapability, detect_caspar_capability
from colsfm.benchmark import read_run, rig_rigidity_spread_millimeters
from colsfm.config import CusfmConfig, read_config_directory
from colsfm.export import KEYFRAME_METADATA_SUBPATH, RUNTIME_CSV_NAME, RuntimeRecord, read_runtime_records
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, read_frames_meta
from colsfm.geometry import MILLIMETRES_PER_METRE
from colsfm.matching import MatchingOptions, MatchLimitPolicy
from colsfm.model_assets import (
    RACO_LIGHTGLUE_ONNX_PATH,
    RACO_ONNX_PATH,
    missing_raco_graphs_message,
    required_raco_graphs,
)
from colsfm.pipeline import StageResult, run_pipeline, run_stage
from colsfm.run_config import DEFAULT_CONFIG_DIR, PipelineOptions, ResolvedRun, SelectionOptions, plan_runs_extrinsic_stage, resolve_run
from colsfm.run_lifecycle import (
    ALL_STAGE_NAMES,
    LOOP_EDGES_NAME,
    RUN_STATE_NAME,
    STAGE_NAMES,
    SUMMARY_NAME,
    RunState,
    StageLedger,
    read_run_state,
    stage_names,
    timed_stage,
)
from colsfm.run_report import ExtrinsicChange, LoopEdgeRecord, PipelineSummary
from colsfm.stages import LoopClosureStageResult, PoseGraphStageResult, run_loop_closure_stage, run_pose_graph_stage

SMOKE_MIN_INTER_FRAME_DISTANCE_M: float = 0.5
"""`feature_extractor_main`'s own default gates, 0.5 m with `SMOKE_MIN_INTER_FRAME_ROTATION_DEGREES`; keep 34 of Galileo's 226 keyframes."""

SMOKE_MIN_INTER_FRAME_ROTATION_DEGREES: float = 5.0
"""Stated, not inherited: colsfm's own default is 15 deg (`DEFAULT_MIN_INTER_FRAME_ROTATION_DEGREES`), which keeps 14 keyframes."""

MIN_REGISTERED_IMAGES: int = 30
"""Of the 34 selected keyframes, at least this many must end up registered."""

RIG_RIGIDITY_TOLERANCE_MM: float = 1e-6
"""How far the eight cameras of one rig instant may disagree on `world_T_vehicle`.

Pure algebra once the export is self-consistent: every pose is derived from the
same rig, so the only spread is float rounding."""

MIN_OBSERVING_FRACTION: float = 0.7
"""Fraction of the selected keyframes that must observe at least one point.

Deliberately below the registered count: 0.5 m spacing leaves five rig samples
across Galileo's 0.66 m sweep, and the cameras facing away from the overlap
legitimately see nothing. Measured 26 of 34."""

PYCOLMAP_ABLATION: Final[PipelineOptions] = PipelineOptions(
    input_dir=Path(),
    output_dir=Path(),
    min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
    min_inter_frame_rotation_degrees=SMOKE_MIN_INTER_FRAME_ROTATION_DEGREES,
    features_backend="pycolmap",
    matching_backend="pycolmap",
    ba_backend="ceres",
    optimize_extrinsics=False,
)
"""The no-GPU-engine ablation every run in this module takes, minus its two paths.

`python -m colsfm run` defaults to RaCo, LightGlue+ and CASPAR since 2026-09-06
(NOTES.md decision 17), which needs the uncommitted graphs under
`data/cusfm_models/` and a `CASPAR_ENABLED` pycolmap. This module is the *pipeline*
suite, not a backend suite: what it asserts — the artifacts, the publication
boundary, the summary — is the same on every backend, so it names the ablation that
runs on a bare checkout and leaves the engines to `test_raco_backend.py` and
`test_mapping_ba_backend.py`. Every fixture below is `replace(PYCOLMAP_ABLATION,
input_dir=..., output_dir=...)`, so the ablation is stated once."""


@pytest.fixture(scope="module")
def galileo_run(galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineSummary:
    """Run the whole pipeline once on Galileo into a temporary workspace.

    Args:
        galileo_input_dir: The r2b_galileo input directory, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The summary the run produced.
    """
    input_dir: Path = galileo_input_dir
    if not (input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {input_dir / FRAMES_META_NAME}")
    options: PipelineOptions = replace(
        PYCOLMAP_ABLATION, input_dir=input_dir, output_dir=tmp_path_factory.mktemp("galileo_colsfm")
    )
    return run_pipeline(options)


def test_the_run_writes_every_artifact_a_cusfm_run_leaves_behind(galileo_run: PipelineSummary) -> None:
    """Each of cuSFM's output paths exists after the run."""
    output_dir: Path = galileo_run.options.output_dir
    expected: list[Path] = [
        output_dir / "keyframes" / FRAMES_META_NAME,
        output_dir / "database.db",
        output_dir / "pose_graph" / FRAMES_META_NAME,
        output_dir / "pose_graph" / "vehicle_pose.tum",
        output_dir / "pose_graph" / LOOP_EDGES_NAME,
        output_dir / "sparse" / "cameras.txt",
        output_dir / "sparse" / "images.txt",
        output_dir / "sparse" / "points3D.txt",
        output_dir / KEYFRAME_METADATA_SUBPATH,
        output_dir / "output_poses" / "merged_pose_file.tum",
        output_dir / RUNTIME_CSV_NAME,
        output_dir / SUMMARY_NAME,
    ]
    missing: list[str] = [str(path) for path in expected if not path.exists()]
    assert not missing, f"pipeline did not write {missing}"


def test_the_sparse_model_loads_and_registers_the_selected_keyframes(galileo_run: PipelineSummary) -> None:
    """`sparse/` is a real COLMAP text model with most of the selected images in it."""
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(galileo_run.options.output_dir / "sparse"))
    print(
        f"[colsfm] galileo smoke | registered {len(reconstruction.reg_image_ids())} "
        f"| points {reconstruction.num_points3D()} "
        f"| reprojection {reconstruction.compute_mean_reprojection_error():.4f} px"
    )
    assert len(reconstruction.reg_image_ids()) >= MIN_REGISTERED_IMAGES
    assert reconstruction.num_points3D() > 0
    assert reconstruction.num_points3D() == galileo_run.mapping.num_points3D
    assert galileo_run.mapping.num_registered_images == galileo_run.num_selected_keyframes
    assert galileo_run.mapping.num_images_with_observations >= MIN_OBSERVING_FRACTION * galileo_run.num_selected_keyframes


def test_the_runtime_log_holds_one_row_per_stage(galileo_run: PipelineSummary) -> None:
    """`runtime.csv` is cuSFM's two-column log, one row per stage, in stage order."""
    records: list[RuntimeRecord] = read_runtime_records(galileo_run.options.output_dir / RUNTIME_CSV_NAME)
    assert [record.command for record in records] == list(STAGE_NAMES)
    assert list(galileo_run.stage_seconds) == list(STAGE_NAMES)
    assert all(record.runtime_seconds >= 0.0 for record in records)


def test_the_summary_round_trips_through_pyserde(galileo_run: PipelineSummary) -> None:
    """`summary.json` deserialises back into the very summary the run returned."""
    restored: PipelineSummary = from_json(PipelineSummary, (galileo_run.options.output_dir / SUMMARY_NAME).read_text())
    assert restored == galileo_run


def test_the_pose_graph_stage_writes_a_pose_per_rig_frame(galileo_run: PipelineSummary) -> None:
    """`pose_graph/vehicle_pose.tum` carries one TUM line per rig frame."""
    lines: list[str] = (galileo_run.options.output_dir / "pose_graph" / "vehicle_pose.tum").read_text().splitlines()
    assert len(lines) == galileo_run.num_rig_frames
    assert all(len(line.split()) == 8 for line in lines)


# ── `--ba-backend caspar`: COLMAP's GPU bundle adjustment ────────────────────────────

MAX_REGISTERED_DIFFERENCE: int = 2
"""How far the CASPAR run's registered count may sit from the Ceres run's.

The two solvers move the same poses by fractions of a millimetre, but they are a
double and a float32 solver with different stopping rules, so a track that sits
exactly on a round's pixel gate can fall either side of it and take its image's
last observation with it."""


@pytest.fixture(scope="module")
def galileo_caspar_run(
    galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory, caspar_enabled: bool
) -> PipelineSummary:
    """The same smoke run with `--ba-backend caspar`, into its own workspace.

    Args:
        galileo_input_dir: The r2b_galileo input directory, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.
        caspar_enabled: Whether this pycolmap has the GPU backend compiled in.

    Returns:
        The summary the run produced.
    """
    if not caspar_enabled:
        pytest.skip("this pycolmap is built without CASPAR_ENABLED; run under `pixi run -e colsfm-dev`")
    input_dir: Path = galileo_input_dir
    if not (input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {input_dir / FRAMES_META_NAME}")
    options: PipelineOptions = replace(
        PYCOLMAP_ABLATION,
        input_dir=input_dir,
        output_dir=tmp_path_factory.mktemp("galileo_colsfm_caspar"),
        ba_backend="caspar",
    )
    return run_pipeline(options)


def test_the_caspar_run_reconstructs_what_the_ceres_run_reconstructs(
    galileo_caspar_run: PipelineSummary, galileo_run: PipelineSummary
) -> None:
    """The GPU backend finishes the pipeline and lands on the same map.

    `summary.json` records the backend the solves *ran* on, not the one asked for,
    so this also proves no fallback fired.
    """
    print(
        f"[colsfm] galileo smoke caspar | {galileo_caspar_run.mapping.num_points3D} points against "
        f"{galileo_run.mapping.num_points3D}, reprojection "
        f"{galileo_caspar_run.mapping.mean_reprojection_error_px:.4f} px against "
        f"{galileo_run.mapping.mean_reprojection_error_px:.4f} px, mapping stage "
        f"{galileo_caspar_run.stage_seconds['reconstruction']:.2f} s against "
        f"{galileo_run.stage_seconds['reconstruction']:.2f} s"
    )
    assert galileo_caspar_run.options.ba_backend == "caspar"
    assert galileo_caspar_run.mapping.ba_backend == "caspar"
    assert galileo_run.mapping.ba_backend == "ceres"
    registered_difference: int = abs(
        galileo_caspar_run.mapping.num_registered_images - galileo_run.mapping.num_registered_images
    )
    assert registered_difference <= MAX_REGISTERED_DIFFERENCE
    assert galileo_caspar_run.mapping.num_registered_images >= MIN_REGISTERED_IMAGES


def test_the_caspar_run_ends_on_caspar_alone_by_default(galileo_caspar_run: PipelineSummary, galileo_run: PipelineSummary) -> None:
    """`summary.json` carries no polish: CASPAR's reduced camera system reaches its own least-squares minimum.

    The polish (`--caspar-ceres-polish`) is off by default since recipes/colmap-src
    `caspar-reduced-camera-system.patch`; the mapping-level tests still run it explicitly.
    """
    assert galileo_caspar_run.mapping.polish_seconds is None
    assert galileo_caspar_run.mapping.polish_iterations is None
    assert galileo_run.mapping.polish_seconds is None
    assert galileo_run.mapping.polish_iterations is None


# ── `--optimize-extrinsics`: the blob's second mapping pass ──────────────────────────


@pytest.fixture(scope="module")
def galileo_refined_run(galileo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineSummary:
    """The same smoke run with `--optimize-extrinsics`, into its own workspace.

    Args:
        galileo_input_dir: The r2b_galileo input directory, from the shared conftest.
        tmp_path_factory: pytest's per-module temporary directory factory.

    Returns:
        The summary the run produced.
    """
    input_dir: Path = galileo_input_dir
    if not (input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {input_dir / FRAMES_META_NAME}")
    options: PipelineOptions = replace(
        PYCOLMAP_ABLATION,
        input_dir=input_dir,
        output_dir=tmp_path_factory.mktemp("galileo_colsfm_ext"),
        optimize_extrinsics=True,
    )
    return run_pipeline(options)


def test_the_refined_run_records_a_second_mapping_pass(galileo_refined_run: PipelineSummary) -> None:
    """`runtime.csv` gains one row, `extrinsic_refinement`, between mapping and export."""
    expected: list[str] = list(stage_names(optimize_extrinsics=True))
    assert expected == list(ALL_STAGE_NAMES) == [*STAGE_NAMES[:-1], "extrinsic_refinement", "export"]
    records: list[RuntimeRecord] = read_runtime_records(galileo_refined_run.options.output_dir / RUNTIME_CSV_NAME)
    assert [record.command for record in records] == expected
    assert list(galileo_refined_run.stage_seconds) == expected


def test_the_refinement_moves_the_extrinsics_and_says_by_how_much(galileo_refined_run: PipelineSummary) -> None:
    """`summary.json` carries one movement record per camera, and they are not all zero."""
    assert galileo_refined_run.options.optimize_extrinsics is True
    changes: tuple[ExtrinsicChange, ...] = galileo_refined_run.extrinsic_changes
    assert len(changes) > 1
    print(
        "[colsfm] galileo smoke refinement: "
        + ", ".join(
            f"{change.sensor_name}={change.translation_change_mm:.2f} mm/{change.rotation_change_deg:.3f} deg"
            for change in changes
        )
    )
    references: list[ExtrinsicChange] = [change for change in changes if change.is_reference]
    assert len(references) == 1, "exactly one camera is the rig origin"
    assert references[0].translation_change_mm == pytest.approx(0.0, abs=1e-9)
    assert references[0].rotation_change_deg == pytest.approx(0.0, abs=1e-9)
    moved: list[ExtrinsicChange] = [change for change in changes if not change.is_reference]
    assert all(change.translation_change_mm > 0.0 for change in moved), "every free extrinsic moved"


def test_the_refined_extrinsics_reach_the_exported_metadata(galileo_refined_run: PipelineSummary) -> None:
    """`kpmap/keyframes/frames_meta.json` carries the refined `sensor_to_vehicle_transform`.

    Read back through `colsfm.frames_meta` rather than out of the summary, so
    this is the file a downstream consumer would see.
    """
    exported: FramesMeta = read_frames_meta(galileo_refined_run.options.output_dir / KEYFRAME_METADATA_SUBPATH)
    source: FramesMeta = read_frames_meta(galileo_refined_run.options.output_dir / "pose_graph" / FRAMES_META_NAME)
    by_camera_params_id: dict[int, ExtrinsicChange] = {
        change.camera_params_id: change for change in galileo_refined_run.extrinsic_changes
    }
    for camera_params_id, camera in exported.cameras.items():
        written_mm: float = MILLIMETRES_PER_METRE * float(
            np.linalg.norm(
                np.asarray(camera.vehicle_T_cam.translation)
                - np.asarray(source.cameras[camera_params_id].vehicle_T_cam.translation)
            )
        )
        assert written_mm == pytest.approx(by_camera_params_id[camera_params_id].translation_change_mm, abs=1e-9)


def test_the_exported_camera_poses_agree_with_the_refined_extrinsics(
    galileo_refined_run: PipelineSummary,
) -> None:
    """`camera_to_world` and `sensor_to_vehicle_transform` in the export describe one rig.

    Every camera of a `synced_sample_id` must imply the same `world_T_vehicle`;
    if the export wrote refined extrinsics next to poses derived from the input
    ones, that spread would open up.
    """
    exported: FramesMeta = read_frames_meta(galileo_refined_run.options.output_dir / KEYFRAME_METADATA_SUBPATH)
    spread_mm: float = rig_rigidity_spread_millimeters(exported)
    print(f"[colsfm] galileo smoke refinement: rig rigidity spread {spread_mm:.4f} mm")
    assert spread_mm < RIG_RIGIDITY_TOLERANCE_MM


def test_the_refined_summary_round_trips_through_pyserde(galileo_refined_run: PipelineSummary) -> None:
    """The nested `ExtrinsicChange` records survive `summary.json` and come back."""
    restored: PipelineSummary = from_json(
        PipelineSummary, (galileo_refined_run.options.output_dir / SUMMARY_NAME).read_text()
    )
    assert restored == galileo_refined_run
    assert restored.extrinsic_changes == galileo_refined_run.extrinsic_changes


def test_a_run_without_the_flag_records_nothing_about_extrinsics(galileo_run: PipelineSummary) -> None:
    """With the flag off the summary and the stage list are exactly what they were."""
    assert galileo_run.options.optimize_extrinsics is False
    assert galileo_run.extrinsic_changes == ()
    assert stage_names(optimize_extrinsics=False) == STAGE_NAMES


# ── What a viewer reads back: the loop edges, the diagnostics, the solver summary ─────


def test_the_pose_graph_stage_always_writes_a_loop_edge_file(galileo_run: PipelineSummary) -> None:
    """`pose_graph/loop_edges.json` exists on every run, holding `[]` when loops are off.

    An absent file cannot say "the stage ran and found nothing", which is the
    normal outcome and the one a viewer has to draw differently from "not yet".
    """
    written: str = (galileo_run.options.output_dir / "pose_graph" / LOOP_EDGES_NAME).read_text()
    records: list[LoopEdgeRecord] = from_json(list[LoopEdgeRecord], written)
    assert records == []
    assert galileo_run.num_loop_edges == len(records)


def test_the_disabled_loop_closure_stage_reports_no_diagnostics(
    galileo_input: FramesMeta, tmp_path: Path
) -> None:
    """With the stage off there is nothing to explain, so `diagnostics` is None.

    The field exists so a caller can tell "searched and rejected everything" from
    "never searched"; None is how the second says so.
    """
    result: LoopClosureStageResult = run_loop_closure_stage(
        galileo_input,
        tmp_path / "absent.db",
        read_config_directory(DEFAULT_CONFIG_DIR).pose_graph,
        MatchingOptions(),
        enabled=False,
    )
    assert result.diagnostics is None
    assert result.edges == []


def test_the_pose_graph_stage_carries_its_solver_summary(
    galileo_input: FramesMeta, three_samples: FramesMeta, galileo_input_dir: Path, tmp_path: Path
) -> None:
    """`PoseGraphStageResult.solve` holds the Ceres summary, and is None with no edges.

    Three rig frames give sequential edges and therefore a solve; one rig frame
    gives none, and then the input poses stand untouched.
    """
    config: CusfmConfig = read_config_directory(DEFAULT_CONFIG_DIR)
    options: PipelineOptions = PipelineOptions(input_dir=galileo_input_dir, output_dir=tmp_path / "many")
    solved: PoseGraphStageResult = run_pose_graph_stage(options, three_samples, config, [])
    assert solved.solve is not None
    assert solved.solve.iterations >= 0
    assert set(solved.solve.world_T_rig) == {node.rig_id for node in solved.nodes}

    one_frame: FramesMeta = galileo_input.filtered(list(galileo_input.rig_frames()[0].keyframe_ids))
    single: PoseGraphStageResult = run_pose_graph_stage(
        PipelineOptions(input_dir=galileo_input_dir, output_dir=tmp_path / "one"), one_frame, config, []
    )
    assert single.edges == []
    assert single.solve is None


def test_a_stage_that_raises_records_no_timing_and_claims_no_finish(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A failing stage leaves `runtime.csv` alone and never claims it finished.

    `runtime.csv` is the run's evidence of what completed, so a stage that raised
    belongs nowhere in it. The exception still reaches the caller.
    """
    ledger: StageLedger = StageLedger(output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="stage body failed"), timed_stage(ledger, "matching"):
        raise RuntimeError("stage body failed")

    assert ledger.seconds_by_stage == {}
    assert not (tmp_path / RUNTIME_CSV_NAME).exists()
    assert "finished" not in capsys.readouterr().out


def test_a_stage_that_returns_records_its_timing_and_says_it_finished(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The success path is untouched: one `runtime.csv` row, one "finished" line."""
    ledger: StageLedger = StageLedger(output_dir=tmp_path)
    with timed_stage(ledger, "matching"):
        pass

    assert set(ledger.seconds_by_stage) == {"matching"}
    records: list[RuntimeRecord] = read_runtime_records(tmp_path / RUNTIME_CSV_NAME)
    assert [record.command for record in records] == ["matching"]
    assert "matching finished in" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# Configuration is resolved and validated once, before anything is created
# ─────────────────────────────────────────────────────────────────────────────


def test_the_default_run_is_the_fast_full_pipeline() -> None:
    """`python -m colsfm run` with no other flag asks for the configuration of record.

    RaCo-ALIKED, FlexGlue (since 2026-11-15; LightGlue+ before), CASPAR with its Ceres polish,
    the regularised extrinsic refinement and loop closure — NOTES.md decision 17, 2026-09-06. The defaults are
    pinned here because they are a *product* decision: a machine without the RaCo
    graphs or without a CASPAR build must be told so, not quietly given another
    pipeline, and this test is what a change of mind has to walk past.
    """
    default: PipelineOptions = PipelineOptions(input_dir=Path("in"), output_dir=Path("out"))
    assert default.features_backend == "raco"
    assert default.matching_backend == "flexglue"
    assert default.ba_backend == "caspar"
    assert default.caspar_ceres_polish is False
    assert default.optimize_extrinsics is True
    assert default.regularised_extrinsics is True
    assert default.loop_closure is True


def test_the_default_maps_on_caspar_and_refines_the_extrinsics_afterwards(
    galileo_input_dir: Path, tmp_path: Path, caspar_enabled: bool
) -> None:
    """The default resolves to a CASPAR mapping pass with `sensor_from_rig` held fixed.

    The two defaults look mutually exclusive — CASPAR throws when asked to refine
    `sensor_from_rig` — and they are not: stage 7 maps with the extrinsics fixed, so
    the GPU solve is legal, and stage 7b then moves them in pyceres with the priors
    (`colsfm.extrinsic_refinement`). The plan must therefore say `caspar` with no
    fallback; it said `ceres` while the pipeline was in fact solving on the GPU.

    On a pycolmap without CASPAR the same resolution answers `ceres` and names the
    missing build, before the run creates anything. That fallback used to be found by
    a bundle adjustment throwing halfway through stage 7.
    """
    # The two backend fields are the one part of the default this assertion does not
    # want: they would make a bare checkout raise for a missing graph, and they have
    # nothing to do with which bundle adjuster the plan names.
    # `test_the_default_resolves_to_the_raco_backends` covers them, behind the graphs.
    default: PipelineOptions = PipelineOptions(input_dir=galileo_input_dir, output_dir=tmp_path / "cusfm")
    resolved: ResolvedRun = resolve_run(
        replace(default, features_backend="pycolmap", matching_backend="pycolmap")
    )
    # One decision, carried: the mapper is handed the plan, not the flags it came from.
    assert resolved.mapping.ba_plan is resolved.ba_plan
    assert resolved.ba_plan.requested_backend == "caspar"
    assert resolved.mapping.optimize_extrinsics is False, "stage 7 maps with the rig fixed"
    if not caspar_enabled:
        assert resolved.ba_plan.backend == "ceres"
        assert resolved.ba_plan.fallback_reason == CASPAR_BUILD_FALLBACK_REASON
        assert resolved.ba_plan.ceres_polish is False, "Ceres has converged; nothing to polish"
        return
    assert resolved.ba_plan.backend == "caspar"
    assert resolved.ba_plan.fallback_reason is None
    assert resolved.ba_plan.ceres_polish is False, "CASPAR alone by default; --caspar-ceres-polish turns it on"


def test_the_unregularised_refinement_stays_on_caspar_only_where_the_rig_probe_says(
    galileo_input_dir: Path, tmp_path: Path
) -> None:
    """`--no-regularised-extrinsics` needs CASPAR's rig factor, and the plan says whether this build has it.

    That path is `run_mapping(..., optimize_extrinsics=True)`, i.e. pycolmap's own rig
    bundle adjustment, which stock CASPAR throws on and recipes/colmap-src's rig factor
    solves. The regularised default never asks the GPU solver for that, so the
    fallback belongs to the ablation alone.
    """
    plan: ResolvedRun = resolve_run(
        replace(
            PYCOLMAP_ABLATION,
            input_dir=galileo_input_dir,
            output_dir=tmp_path / "cusfm",
            ba_backend="caspar",
            optimize_extrinsics=True,
            regularised_extrinsics=False,
        )
    )
    capability: CasparCapability = detect_caspar_capability()
    if capability.availability != "available":
        pytest.skip(f"CASPAR is {capability.availability} in this pycolmap, so the plan falls back for the build, not the rig")
    expected: tuple[str, str | None] = ("caspar", None) if capability.refines_sensor_from_rig else ("ceres", EXTRINSICS_FALLBACK_REASON)
    assert (plan.ba_plan.backend, plan.ba_plan.fallback_reason) == expected
    assert plan.ba_plan.ceres_polish is False


def test_refining_the_camera_model_frees_focals_and_the_rig_in_the_mapper(galileo_input_dir: Path, tmp_path: Path) -> None:
    """`--refine-camera-model` frees fx, fy and `sensor_from_rig` in every mapping round, on CASPAR where the rig probe says, and drops stage 7b.

    Measured on Mission office with Spirula's own matches: colsfm's mapper with the nominal cube model held ends 0.2645
    deg from Spirula's poses; with these freedoms 0.034 deg, the per-face focals within 0.5 px of Spirula's.
    """
    plan: ResolvedRun = resolve_run(
        replace(PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=tmp_path / "cusfm", ba_backend="caspar", refine_camera_model=True)
    )
    assert plan.mapping.refine_focal_length is True
    assert plan.mapping.optimize_extrinsics is True
    capability: CasparCapability = detect_caspar_capability()
    if capability.availability != "available":
        pytest.skip(f"CASPAR is {capability.availability} in this pycolmap, so the plan falls back for the build, not the rig")
    expected: tuple[str, str | None] = ("caspar", None) if capability.refines_sensor_from_rig else ("ceres", EXTRINSICS_FALLBACK_REASON)
    assert (plan.ba_plan.backend, plan.ba_plan.fallback_reason) == expected
    assert stage_names(optimize_extrinsics=plan_runs_extrinsic_stage(replace(PYCOLMAP_ABLATION, refine_camera_model=True))) == STAGE_NAMES


@pytest.mark.skipif(
    any(not path.is_file() for path in required_raco_graphs("raco", "flexglue")),
    reason="the RaCo or FlexGlue graphs are missing; `colsfm-raco-export` and `colsfm-download-flexglue` provide them",
)
def test_the_default_resolves_to_raco_features_and_flexglue_matching(galileo_input_dir: Path, tmp_path: Path) -> None:
    """RaCo extraction and FlexGlue matching when nothing on the command line says otherwise.

    Separate from the plan assertion above, and skipped rather than failed where the
    graphs are absent, because `resolve_run` refuses a `raco` run it cannot perform —
    which is the behaviour `test_a_missing_raco_graph_names_the_export_command_and_the_ablation`
    is about.
    """
    resolved: ResolvedRun = resolve_run(PipelineOptions(input_dir=galileo_input_dir, output_dir=tmp_path / "cusfm"))
    assert resolved.features.backend == "raco"
    assert resolved.matching_backend == "flexglue"


def test_the_raco_default_needs_two_graphs_and_the_ablation_needs_none() -> None:
    """Which model files each backend choice requires, before a run creates anything."""
    assert required_raco_graphs("pycolmap", "pycolmap") == ()
    assert required_raco_graphs("raco", "pycolmap") == (RACO_ONNX_PATH,)
    assert required_raco_graphs("pycolmap", "raco") == (RACO_LIGHTGLUE_ONNX_PATH,)
    assert required_raco_graphs("raco", "raco") == (RACO_ONNX_PATH, RACO_LIGHTGLUE_ONNX_PATH)


def test_a_missing_raco_graph_names_the_export_command_and_the_ablation() -> None:
    """A fresh clone is told how to build the graphs and how to run without them.

    Falling back to `pycolmap` here would be the worst of the options: it changes
    every number a run reports without saying so. The run stops instead.
    """
    message: str = missing_raco_graphs_message((Path("data/cusfm_models/raco-aliked-b1-16.onnx"),))
    assert "raco-aliked-b1-16.onnx" in message
    assert "pixi run -e colsfm colsfm-raco-export" in message
    assert "--features-backend pycolmap --matching-backend pycolmap" in message



def test_a_bad_caspar_override_stops_the_run_before_the_workspace_exists(
    galileo_input_dir: Path, tmp_path: Path
) -> None:
    """`solver_iter_max=2.9` is refused, and nothing is created on the way to refusing it.

    The strings used to be parsed after keyframe selection, feature extraction,
    matching, loop closure and the pose graph — and the value was then cast with
    `int(...)`, so 2.9 silently became 2 and a solve nobody asked for ran.
    """
    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")
    output_dir: Path = tmp_path / "never_created"
    options: PipelineOptions = replace(
        PYCOLMAP_ABLATION,
        input_dir=galileo_input_dir,
        output_dir=output_dir,
        ba_backend="caspar",
        caspar_option=("solver_iter_max=2.9",),
    )
    with pytest.raises(ValueError, match="integer solver knob"):
        run_pipeline(options)
    assert not output_dir.exists(), "a refused run must not leave a workspace behind"


def test_an_unknown_caspar_override_stops_the_run_too(galileo_input_dir: Path, tmp_path: Path) -> None:
    """A typo in an option name is cheap to catch and expensive to discover at the solve."""
    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")
    output_dir: Path = tmp_path / "never_created"
    options: PipelineOptions = replace(
        PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=output_dir, caspar_option=("solver_iterations=10",)
    )
    with pytest.raises(ValueError, match="not a numeric CASPAR solver option"):
        run_pipeline(options)
    assert not output_dir.exists()


def test_the_resolved_run_gives_both_match_stages_one_matcher(galileo_input_dir: Path, tmp_path: Path) -> None:
    """Stage 4 and stage 5's batch match take the same resolved `MatchingOptions`."""
    options: PipelineOptions = replace(
        PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=tmp_path / "cusfm", match_cap_mode="image_area"
    )
    resolved: ResolvedRun = resolve_run(options)
    config: CusfmConfig = read_config_directory(DEFAULT_CONFIG_DIR)
    assert resolved.matching_options(config) == resolved.matching_options(config)
    assert resolved.match_limit == MatchLimitPolicy.of("image_area", options.max_matches_per_pair)


def test_the_two_spellings_of_an_uncapped_run_resolve_to_the_same_policy(
    galileo_input_dir: Path, tmp_path: Path
) -> None:
    """`--match-cap-mode off` and `--max-matches-per-pair None` are one state, not two."""
    by_mode: ResolvedRun = resolve_run(
        replace(PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=tmp_path / "a", match_cap_mode="off")
    )
    by_budget: ResolvedRun = resolve_run(
        replace(PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=tmp_path / "b", max_matches_per_pair=None)
    )
    assert by_mode.match_limit == by_budget.match_limit
    assert not by_mode.match_limit.limits


def test_the_metadata_stages_need_no_output_directory(galileo_input_dir: Path) -> None:
    """`python -m colsfm stage` takes `SelectionOptions`, which has no workspace in it.

    The two cheap stages read metadata, touch no image and write nothing; asking
    them for an output directory asked for a workspace nobody would fill.
    """
    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")
    fields: set[str] = {field.name for field in dataclasses.fields(SelectionOptions)}
    assert "output_dir" not in fields
    result: StageResult = run_stage(
        SelectionOptions(
            input_dir=galileo_input_dir,
            min_inter_frame_distance=SMOKE_MIN_INTER_FRAME_DISTANCE_M,
            min_inter_frame_rotation_degrees=SMOKE_MIN_INTER_FRAME_ROTATION_DEGREES,
        ),
        "pair_selection",
    )
    assert result.num_selected_keyframes > 0
    assert result.num_pairs > 0


# ─────────────────────────────────────────────────────────────────────────────
# The publication boundary, end to end
# ─────────────────────────────────────────────────────────────────────────────


def test_a_published_run_says_it_succeeded(galileo_run: PipelineSummary) -> None:
    """A run directory is only published once it is whole, and it says so."""
    state: RunState | None = read_run_state(galileo_run.options.output_dir)
    assert state is not None, f"a colsfm run writes {RUN_STATE_NAME}"
    assert state.status == "succeeded"
    assert state.stages_completed == tuple(STAGE_NAMES)
    assert state.stages_planned == tuple(STAGE_NAMES)
    assert state.failure is None


def test_a_failed_rerun_leaves_the_previous_run_whole(galileo_input_dir: Path, tmp_path: Path) -> None:
    """The defect the staging workspace exists for: no directory of two runs.

    A first run publishes. A second run over the same destination fails at stage 1
    — its config directory does not exist — and the destination must still hold the
    first run, byte for byte, rather than a mixture with `runtime.csv` cleared and
    the database deleted.
    """
    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")
    output_dir: Path = tmp_path / "cusfm"
    good: PipelineOptions = replace(
        PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=output_dir, loop_closure=False
    )
    run_pipeline(good)
    before: dict[str, int] = {
        str(path.relative_to(output_dir)): path.stat().st_size for path in sorted(output_dir.rglob("*")) if path.is_file()
    }
    assert before, "the first run wrote something"

    doomed: PipelineOptions = replace(good, config_dir=tmp_path / "no_such_config_profile")
    with pytest.raises(FileNotFoundError):
        run_pipeline(doomed)

    after: dict[str, int] = {
        str(path.relative_to(output_dir)): path.stat().st_size for path in sorted(output_dir.rglob("*")) if path.is_file()
    }
    assert after == before, "a failed rerun must not touch the destination"
    assert read_run_state(output_dir) is not None
    published: RunState | None = read_run_state(output_dir)
    assert published is not None and published.status == "succeeded"


def test_a_benchmark_reader_refuses_a_run_that_did_not_finish(galileo_run: PipelineSummary, tmp_path: Path) -> None:
    """`colsfm.benchmark.read_run` checks provenance, not just file presence."""
    published: Path = galileo_run.options.output_dir
    unfinished: Path = tmp_path / "unfinished"
    shutil.copytree(published, unfinished)
    state: RunState = read_run_state(unfinished)  # type: ignore[assignment]
    (unfinished / RUN_STATE_NAME).write_text(to_json(replace(state, status="failed", failure="boom")) + "\n")

    with pytest.raises(ValueError, match="holds a run that failed"):
        read_run(unfinished, "unfinished")
    # The same directory reads fine while it says it succeeded.
    read_run(published, "published")


def test_a_reference_run_without_a_state_file_still_reads(galileo_run: PipelineSummary, tmp_path: Path) -> None:
    """A `pycusfm` run has no `run_state.json`; the reader must not demand one."""
    foreign: Path = tmp_path / "blob_style"
    shutil.copytree(galileo_run.options.output_dir, foreign)
    (foreign / RUN_STATE_NAME).unlink()
    read_run(foreign, "blob_style")


def test_a_library_run_aligns_to_its_prior_before_it_publishes(galileo_input_dir: Path, tmp_path: Path) -> None:
    """`align_to_prior` is applied by `run_pipeline` itself and lands in the typed, published summary."""
    from colsfm.run_lifecycle import SUMMARY_NAME

    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")
    options: PipelineOptions = replace(PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=tmp_path / "run", align_to_prior=True)

    summary: PipelineSummary = run_pipeline(options)

    assert summary.prior_alignment is not None and summary.prior_alignment.num_cameras >= 3
    published: PipelineSummary = from_json(PipelineSummary, (tmp_path / "run" / SUMMARY_NAME).read_text())
    assert published.prior_alignment == summary.prior_alignment


def test_a_failed_prior_alignment_publishes_nothing(galileo_input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The alignment is part of the run: when it fails, the destination keeps what it held (here: nothing)."""
    from colsfm import prior_alignment

    if not (galileo_input_dir / FRAMES_META_NAME).is_file():
        pytest.skip(f"missing {galileo_input_dir / FRAMES_META_NAME}")

    def refuse(*_: object) -> prior_alignment.PriorAlignment:
        raise ValueError("no prior")

    monkeypatch.setattr(prior_alignment, "fit_prior_alignment", refuse)
    options: PipelineOptions = replace(PYCOLMAP_ABLATION, input_dir=galileo_input_dir, output_dir=tmp_path / "run", align_to_prior=True)

    with pytest.raises(ValueError, match="no prior"):
        run_pipeline(options)
    assert not (tmp_path / "run").exists()


@pytest.mark.integration
def test_the_run_layer_carries_the_run_record_it_is_given(galileo_run: PipelineSummary, tmp_path: Path) -> None:
    """`colsfm catalog` hands `write_run_layer` its run record (`simplecv.run_record`); the layer holds it as `run_colsfm`."""
    import rerun.chunk as rrc
    from simplecv.run_record import start_run

    from colsfm.rerun_layer import DEFAULT_SOURCE, write_run_layer

    layer: Path = tmp_path / "colsfm.rrd"
    write_run_layer(galileo_run.options.output_dir, layer, segment="seg", record=start_run(None, source=Path(__file__)).finish(DEFAULT_SOURCE, "seg"))
    assert "/__properties/run_colsfm" in {str(chunk.entity_path) for chunk in rrc.RrdReader(layer).stream()}
