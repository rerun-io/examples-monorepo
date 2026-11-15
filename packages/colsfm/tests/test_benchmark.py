"""Benchmark metrics, checked against synthetic runs with known answers.

Every synthetic fixture is written to disk in the **real** cuSFM layout, through
the real writers (`pycolmap.Reconstruction.write_text` via
`colsfm.export.write_colmap_model`, `write_optimised_frames_meta`,
`append_runtime_record`), so `read_run` is exercised end to end rather than
handed an in-memory shortcut. The builder itself lives in `conftest`, because
`test_rerun_log` scores the same runs.

Expected values come from an independent source: a closed-form residual for a
scaled trajectory, the pixel offset the observations were built with, the
`(1 - 1/k) d` a single displaced camera puts into a k-camera mean, and — for
the real-data test — the numbers `demo_rerun.py` printed for the same blob run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TypeAlias, get_args

import numpy as np
import pycolmap
import pytest
from conftest import (
    BLOB_ERROR_PLACEHOLDER,
    BLOB_RUNTIMES,
    COLSFM_PIPELINE_RUNTIMES,
    COLSFM_RUNTIMES,
    RIGID_TOLERANCE_MM,
    SYNTHETIC_ERROR_PX,
    transform_rig_trajectory,
    write_synthetic_run,
)
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde.json import from_json, to_json

from colsfm.bench_report import render_markdown_report
from colsfm.benchmark import (
    STAGES,
    AcceptanceBounds,
    Comparison,
    RigidAlignment,
    RigTrack,
    RunArtifacts,
    RunMetrics,
    RuntimeRecord,
    Stage,
    TrajectoryMetrics,
    align_rigid,
    check_acceptance,
    classify_stage,
    compare_runs,
    latest_run_records,
    match_timestamps,
    read_ground_truth,
    read_run,
    reconstruction_metrics,
    rig_rigidity_spread_millimeters,
    rig_track_from_frames_meta,
    stage_runtime_seconds,
)
from colsfm.export import append_runtime_record, read_runtime_records
from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.run_lifecycle import StageName
from colsfm.runtime import STAGE_BY_PIPELINE_STAGE

RunPair: TypeAlias = tuple[RunArtifacts, RunArtifacts]
"""Run A (the blob's vocabulary) and run B (colsfm's), read back off disk."""


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_pair(three_samples: FramesMeta, tmp_path_factory: pytest.TempPathFactory) -> RunPair:
    """Two synthetic runs of the same three samples, so B reproduces A exactly.

    Only the `runtime.csv` vocabularies differ — A writes the blob's binary names
    and B writes `colsfm`'s stage names — which is what makes the pair the right
    fixture for the runtime table, the acceptance bounds and the JSON round trip
    all at once.
    """
    root: Path = tmp_path_factory.mktemp("synthetic_pair")
    run_a: RunArtifacts = read_run(
        write_synthetic_run(root / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=BLOB_RUNTIMES), "blob"
    )
    run_b: RunArtifacts = read_run(
        write_synthetic_run(root / "b", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )
    return run_a, run_b


def comparison_for(
    pair: RunPair, galileo_input_dir: Path, *, dataset: str = "galileo", bounds: AcceptanceBounds | None = None
) -> Comparison:
    """Compare a pair of synthetic runs against the Galileo input trajectory.

    Args:
        pair: The two runs, A then B.
        galileo_input_dir: Directory holding `frames_meta.json` and `ground_truth.txt`.
        dataset: The label written into the report; no behaviour depends on it.
        bounds: Acceptance bounds, or None to check none.

    Returns:
        The comparison.
    """
    return compare_runs(pair[0], pair[1], galileo_input_dir, dataset, bounds)


# ─────────────────────────────────────────────────────────────────────────────
# Alignment
# ─────────────────────────────────────────────────────────────────────────────


def test_a_rigidly_moved_trajectory_aligns_back_exactly() -> None:
    """A rotation plus a translation is undone by the fit, leaving unit scale."""
    generator: np.random.Generator = np.random.default_rng(3)
    source_xyz: Float64[ndarray, "n 3"] = generator.normal(0.0, 1.0, (40, 3))
    quaternion_xyzw: Float64[ndarray, "4"] = np.array([0.1, -0.2, 0.3, 0.9])
    rotation: Float64[ndarray, "3 3"] = pycolmap.Rotation3d(quaternion_xyzw / np.linalg.norm(quaternion_xyzw)).matrix()
    target_xyz: Float64[ndarray, "n 3"] = source_xyz @ np.asarray(rotation).T + np.array([2.0, -3.0, 0.5])

    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)

    assert alignment.rmse_meters == pytest.approx(0.0, abs=1e-12)
    assert alignment.max_error_meters == pytest.approx(0.0, abs=1e-12)
    assert alignment.would_be_scale == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(alignment.apply(source_xyz), target_xyz, atol=1e-12)


def test_a_scaled_trajectory_reports_the_scale_the_fit_refused_to_absorb() -> None:
    """Holding scale at 1 turns a 5 % shrink into a residual with a closed form.

    For `source = s * target` about the centroid, the scale-1 fit leaves
    `|s - 1| * rms_radius` of residual, and the similarity fit that was declined
    would have chosen `1 / s`.
    """
    generator: np.random.Generator = np.random.default_rng(11)
    target_xyz: Float64[ndarray, "n 3"] = generator.normal(0.0, 2.0, (60, 3))
    scale: float = 0.95
    source_xyz: Float64[ndarray, "n 3"] = scale * target_xyz

    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)

    centred: Float64[ndarray, "n 3"] = target_xyz - target_xyz.mean(axis=0)
    rms_radius: float = float(np.sqrt((centred**2).sum(axis=1).mean()))
    assert alignment.would_be_scale == pytest.approx(1.0 / scale, rel=1e-9)
    assert alignment.rmse_meters == pytest.approx(abs(scale - 1.0) * rms_radius, rel=1e-9)


def test_a_scale_estimating_fit_absorbs_the_scale_the_rigid_fit_refused() -> None:
    """`estimate_scale=True` is the same fit with `s` let in, and `apply` honours it.

    On a trajectory scaled by a known factor the SIM(3) fit reports that factor as
    `scale` and leaves no residual, where the default rigid fit reports the same
    factor as `would_be_scale` but keeps the residual it refused to absorb.
    """
    generator: np.random.Generator = np.random.default_rng(23)
    source_xyz: Float64[ndarray, "n 3"] = generator.normal(0.0, 1.0, (30, 3))
    scale: float = 2.5
    quaternion_xyzw: Float64[ndarray, "4"] = np.array([0.1, -0.2, 0.3, 0.9])
    rotation: pycolmap.Rotation3d = pycolmap.Rotation3d(quaternion_xyzw / np.linalg.norm(quaternion_xyzw))
    translation_xyz: Float64[ndarray, "3"] = np.array([1.0, 2.0, 3.0])
    target_xyz: Float64[ndarray, "n 3"] = scale * (source_xyz @ np.asarray(rotation.matrix()).T) + translation_xyz

    similarity: RigidAlignment = align_rigid(source_xyz, target_xyz, estimate_scale=True)
    rigid: RigidAlignment = align_rigid(source_xyz, target_xyz)

    assert similarity.scale == pytest.approx(scale, rel=1e-12)
    assert similarity.rmse_meters == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(similarity.apply(source_xyz), target_xyz, atol=1e-12)
    assert rigid.scale == 1.0
    assert rigid.would_be_scale == pytest.approx(scale, rel=1e-12)
    assert rigid.rmse_meters > 1.0

    moved: pycolmap.Rigid3d = similarity.apply_pose(pycolmap.Rigid3d())
    assert np.allclose(np.asarray(moved.translation), translation_xyz, atol=1e-12)
    assert np.allclose(np.asarray(moved.rotation.matrix()), np.asarray(rotation.matrix()), atol=1e-12)


def test_alignment_refuses_mismatched_trajectories() -> None:
    """Two trajectories of different lengths cannot be aligned."""
    with pytest.raises(ValueError, match="matched trajectories"):
        align_rigid(np.zeros((3, 3)), np.zeros((4, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp joins
# ─────────────────────────────────────────────────────────────────────────────


def test_the_timestamp_join_keeps_only_pairs_inside_the_tolerance() -> None:
    """A 20 us window accepts a 2 us gap and rejects a 50 us one."""
    query: Int64[ndarray, "n"] = np.array([1000, 2000, 3000], dtype=np.int64)
    reference: Int64[ndarray, "k"] = np.array([1002, 2050, 3000], dtype=np.int64)

    query_indices, reference_indices = match_timestamps(query, reference, 20)

    assert query_indices.tolist() == [0, 2]
    assert reference_indices.tolist() == [0, 2]


def test_an_exact_join_demands_an_exact_timestamp() -> None:
    """Tolerance 0 is what the input-trajectory comparison uses."""
    query: Int64[ndarray, "n"] = np.array([10, 20], dtype=np.int64)
    reference: Int64[ndarray, "k"] = np.array([10, 21], dtype=np.int64)

    query_indices, _ = match_timestamps(query, reference, 0)

    assert query_indices.tolist() == [0]


# ─────────────────────────────────────────────────────────────────────────────
# Runtime mapping
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("feature_extractor_main --input_image_directory x", "extraction"),
        ("generate_bow_vocabulary_main --keyframe_directory x", "retrieval"),
        ("generate_bow_index_main --keyframe_directory x", "retrieval"),
        ("generate_association_main --keyframe_dir x", "association"),
        ("pose_graph_main --keyframe_directory x", "pose_graph"),
        ("pose_graph_association_main --keyframe_directory x", "pose_graph"),
        ("feature_matcher_task_builder_main --current_keyframe_directory x", "pair_selection"),
        ("feature_matcher_main --current_keyframe_directory x", "matching"),
        ("keypoints_mapper_main --keyframe_dir x", "mapping"),
        ("kpmap_to_colmap --map_dir x", "export"),
        ("extract_pose_from_map_main --output_pose_dir x", "export"),
        ("update_keyframe_pose_main --input x", "export"),
        ("colsfm.features --input x", "extraction"),
        ("colsfm.retrieval --keyframes x", "retrieval"),
        ("colsfm.loop_closure --keyframes x", "association"),
        ("colsfm.pose_graph --keyframes x", "pose_graph"),
        ("colsfm.pairs --keyframes x", "pair_selection"),
        ("colsfm.matching --keyframes x", "matching"),
        ("colsfm.mapping --keyframes x", "mapping"),
        ("colsfm.export --output x", "export"),
        # The names `colsfm.pipeline` actually writes.
        ("keyframe_selection --input x", "extraction"),
        ("feature_extraction --input x", "extraction"),
        ("pair_selection --keyframes x", "pair_selection"),
        ("matching --keyframes x", "matching"),
        ("loop_closure --keyframes x", "association"),
        ("pose_graph --keyframes x", "pose_graph"),
        ("reconstruction --keyframes x", "mapping"),
        ("extrinsic_refinement --keyframes x", "mapping"),
        ("export --output x", "export"),
    ],
)
def test_every_stage_command_maps_onto_a_plan_stage(command: str, expected: Stage) -> None:
    """Both vocabularies — blob binaries and colsfm stage names — land on the plan's rows."""
    assert classify_stage(command) == expected


@pytest.mark.parametrize("stage_name", get_args(StageName))
def test_every_pipeline_stage_name_is_classified(stage_name: StageName) -> None:
    """A stage `colsfm.pipeline` can time must reach the runtime table.

    The regression this guards: `extrinsic_refinement` matched no substring rule,
    so `classify_stage` returned None for it and the stage vanished from the
    per-stage table, from `total_runtime_seconds`, and from the runtime
    acceptance bound — silently, because an unrecognised row is dropped by design.
    """
    assert classify_stage(f"{stage_name} --config x") in STAGES


def test_the_pipeline_stage_table_is_exhaustive() -> None:
    """`colsfm.pipeline` owns the vocabulary; every member needs a row to land in."""
    assert set(get_args(StageName)) == set(STAGE_BY_PIPELINE_STAGE)


def test_an_unknown_command_is_not_forced_into_a_stage() -> None:
    """A row the harness does not recognise is dropped, not silently mis-attributed."""
    assert classify_stage("cuvslam_main --input x") is None


def test_an_appended_runtime_log_reports_only_its_last_run(tmp_path: Path) -> None:
    """`runtime.csv` is appended to across runs; the newest block is the run."""
    for command, seconds in BLOB_RUNTIMES.items():
        append_runtime_record(tmp_path, RuntimeRecord(command=command, runtime_seconds=seconds))
    for command, seconds in BLOB_RUNTIMES.items():
        append_runtime_record(tmp_path, RuntimeRecord(command=command, runtime_seconds=2.0 * seconds))

    records: list[RuntimeRecord] = read_runtime_records(tmp_path / "runtime.csv")
    assert len(records) == 2 * len(BLOB_RUNTIMES)
    assert len(latest_run_records(records)) == len(BLOB_RUNTIMES)

    totals: dict[Stage, float] = stage_runtime_seconds(records)
    assert totals["extraction"] == pytest.approx(16.0)
    assert totals["retrieval"] == pytest.approx(20.0)
    assert totals["export"] == pytest.approx(5.0)
    assert sum(totals.values()) == pytest.approx(2.0 * sum(BLOB_RUNTIMES.values()))


def test_a_run_whose_stages_are_out_of_pipeline_order_is_not_split() -> None:
    """`colsfm` matches before it closes loops; that is one run, not two."""
    rows: list[RuntimeRecord] = [
        RuntimeRecord(command=command, runtime_seconds=seconds) for command, seconds in COLSFM_PIPELINE_RUNTIMES.items()
    ]

    assert len(latest_run_records(rows)) == len(rows)
    assert sum(stage_runtime_seconds(rows).values()) == pytest.approx(sum(COLSFM_PIPELINE_RUNTIMES.values()))


def test_a_row_repeated_verbatim_mid_run_is_not_a_run_boundary() -> None:
    """`kpmap_to_colmap` takes no config path, so its row repeats across runs.

    Keying the block boundary on the whole row would cut the second run in half
    at that line; keying it on the command *name* does not.
    """
    rows: list[RuntimeRecord] = [
        RuntimeRecord(command="feature_extractor_main --config isaac", runtime_seconds=1.0),
        RuntimeRecord(command="kpmap_to_colmap --map_dir kpmap", runtime_seconds=2.0),
        RuntimeRecord(command="feature_extractor_main --config fixed", runtime_seconds=3.0),
        RuntimeRecord(command="kpmap_to_colmap --map_dir kpmap", runtime_seconds=4.0),
    ]

    kept: list[RuntimeRecord] = latest_run_records(rows)

    assert [record.runtime_seconds for record in kept] == [3.0, 4.0]


def test_a_sharded_stage_stays_inside_one_run() -> None:
    """Several `feature_matcher_main` task files are one stage, not two runs."""
    rows: list[RuntimeRecord] = [
        RuntimeRecord(command="feature_extractor_main --input x", runtime_seconds=1.0),
        RuntimeRecord(command="feature_matcher_main --task_file a", runtime_seconds=2.0),
        RuntimeRecord(command="feature_matcher_main --task_file b", runtime_seconds=3.0),
    ]

    totals: dict[Stage, float] = stage_runtime_seconds(rows)

    assert totals["extraction"] == pytest.approx(1.0)
    assert totals["matching"] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rig geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_the_shipped_galileo_rig_is_rigid(galileo_input: FramesMeta) -> None:
    """Every camera of a sample agrees on the rig pose to within 10 um.

    Not exactly zero: `camera_to_world` is stored as an axis and an angle in
    *degrees*, so each camera's pose carries its own float64 round-trip noise.
    The demo prints this as `0.00 mm`; `RIGID_TOLERANCE_MM` is that, resolved.
    """
    assert rig_rigidity_spread_millimeters(galileo_input) < RIGID_TOLERANCE_MM


def test_displacing_one_camera_moves_the_rigidity_spread_by_one_minus_one_over_k(galileo_input: FramesMeta) -> None:
    """With `k` cameras, an offset `d` on one leaves `(1 - 1/k) * d` after the mean.

    The mean of the `k` per-camera estimates moves by `d/k`, so the displaced
    camera sits `(1 - 1/k) d` from it — the largest deviation in the sample, and
    hence the reported spread.
    """
    sample = galileo_input.rig_frames()[0]
    num_cameras: int = len(sample.keyframe_ids)
    victim: int = sample.keyframe_ids[-1]
    original: pycolmap.Rigid3d = galileo_input.keyframe_by_id()[victim].world_T_cam
    displacement_meters: float = 0.008
    moved: pycolmap.Rigid3d = pycolmap.Rigid3d(
        original.rotation, np.asarray(original.translation, dtype=np.float64) + np.array([displacement_meters, 0.0, 0.0])
    )

    spread: float = rig_rigidity_spread_millimeters(galileo_input.with_camera_to_world({victim: moved}))

    expected_millimeters: float = 1000.0 * (1.0 - 1.0 / num_cameras) * displacement_meters
    assert spread == pytest.approx(expected_millimeters, abs=RIGID_TOLERANCE_MM)


def test_the_rig_track_follows_the_synchronised_samples(galileo_input: FramesMeta) -> None:
    """One rig pose per `synced_sample_id`, in ascending timestamp order."""
    track: RigTrack = rig_track_from_frames_meta(galileo_input)

    assert len(track) == len(galileo_input.rig_frames())
    assert np.all(np.diff(track.timestamps_microseconds) > 0)
    assert track.world_t_rig.shape == (len(track), 3)
    assert track.world_R_rig.shape == (len(track), 3, 3)


def test_the_rig_track_drops_samples_no_image_registered(galileo_input: FramesMeta) -> None:
    """Passing a registered-image filter keeps only the samples it covers."""
    first = galileo_input.rig_frames()[0]
    by_id: dict[int, KeyframeMeta] = galileo_input.keyframe_by_id()
    names: set[str] = {by_id[keyframe_id].image_name for keyframe_id in first.keyframe_ids}

    track: RigTrack = rig_track_from_frames_meta(galileo_input, keep_image_names=names)

    assert len(track) == 1
    assert int(track.timestamps_microseconds[0]) == first.timestamp_microseconds


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────


def test_galileo_ground_truth_joins_onto_every_rig_sample(galileo_input_dir: Path, galileo_input: FramesMeta) -> None:
    """`ground_truth.txt` matches all 29 rig samples inside the 0.02 ms window."""
    ground_truth: RigTrack | None = read_ground_truth(galileo_input_dir)
    assert ground_truth is not None

    track: RigTrack = rig_track_from_frames_meta(galileo_input)
    query_indices, _ = match_timestamps(track.timestamps_microseconds, ground_truth.timestamps_microseconds, 20)

    assert len(query_indices) == len(track)


def test_a_dataset_without_ground_truth_reports_none(tmp_path: Path) -> None:
    """RoboCap ships none; the metric must be absent, not zero."""
    assert read_ground_truth(tmp_path) is None


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end on synthetic runs
# ─────────────────────────────────────────────────────────────────────────────


def test_a_synthetic_run_reports_the_error_it_was_built_with(synthetic_pair: RunPair) -> None:
    """The recomputed reprojection error is the pixel offset, not the stored 2.0."""
    run_a: RunArtifacts = synthetic_pair[0]

    stored: pycolmap.Reconstruction = pycolmap.Reconstruction(str(run_a.run_dir / "sparse"))
    assert stored.compute_mean_reprojection_error() == pytest.approx(BLOB_ERROR_PLACEHOLDER)

    assert reconstruction_metrics(run_a.reconstruction).mean_reprojection_error_px == pytest.approx(SYNTHETIC_ERROR_PX, abs=1e-6)


def test_a_rigidly_offset_run_scores_zero_against_the_reference(
    synthetic_pair: RunPair, three_samples: FramesMeta, tmp_path: Path, galileo_input_dir: Path
) -> None:
    """Moving the whole rig trajectory rigidly is invisible to the ATE, by design."""
    offset: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.0, 0.0, np.sin(0.15), np.cos(0.15)])), np.array([1.5, -0.25, 0.75])
    )
    moved: FramesMeta = transform_rig_trajectory(three_samples, rig_transform=offset)
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", moved, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )

    comparison: Comparison = comparison_for((synthetic_pair[0], run_b), galileo_input_dir)

    assert comparison.run_b.vs_input.num_matched == 3
    assert comparison.run_b.vs_input.rmse_millimeters == pytest.approx(0.0, abs=1e-6)
    assert comparison.run_b.vs_input.would_be_scale == pytest.approx(1.0, abs=1e-6)
    assert comparison.pose_delta.position_rmse_millimeters == pytest.approx(0.0, abs=1e-6)
    assert comparison.pose_delta.rotation_rmse_degrees == pytest.approx(0.0, abs=1e-6)


def test_a_shrunken_run_reports_the_scale_and_the_residual(
    synthetic_pair: RunPair, three_samples: FramesMeta, tmp_path: Path, galileo_input_dir: Path
) -> None:
    """A 2 % shrink shows up as a would-be scale and a closed-form residual."""
    scale: float = 0.98
    shrunk: FramesMeta = transform_rig_trajectory(three_samples, rig_transform=pycolmap.Rigid3d(), scale=scale)
    run_a: RunArtifacts = synthetic_pair[0]
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", shrunk, error_px=SYNTHETIC_ERROR_PX, runtimes=COLSFM_RUNTIMES), "colsfm"
    )

    delta = comparison_for((run_a, run_b), galileo_input_dir).pose_delta

    reference: Float64[ndarray, "n 3"] = run_a.track.world_t_rig
    centred: Float64[ndarray, "n 3"] = reference - reference.mean(axis=0)
    rms_radius: float = float(np.sqrt((centred**2).sum(axis=1).mean()))
    assert delta.num_matched == 3
    assert delta.position_rmse_millimeters == pytest.approx(1000.0 * abs(scale - 1.0) * rms_radius, rel=1e-6)
    assert delta.rotation_rmse_degrees == pytest.approx(0.0, abs=1e-9)


def test_the_runtime_table_maps_both_vocabularies_onto_one_set_of_rows(
    synthetic_pair: RunPair, galileo_input_dir: Path
) -> None:
    """Blob binaries and colsfm stage names share the plan's eight rows, with a B/A ratio."""
    comparison: Comparison = comparison_for(synthetic_pair, galileo_input_dir)

    assert [row.stage for row in comparison.stages] == list(STAGES)
    by_stage: dict[Stage, float | None] = {row.stage: row.seconds_a for row in comparison.stages}
    assert by_stage["retrieval"] == pytest.approx(10.0)
    assert by_stage["export"] == pytest.approx(2.5)
    assert comparison.run_a.total_runtime_seconds == pytest.approx(sum(BLOB_RUNTIMES.values()))
    assert comparison.run_b.total_runtime_seconds == pytest.approx(sum(COLSFM_RUNTIMES.values()))
    assert comparison.total_runtime_ratio == pytest.approx(2.0)
    mapping_row = next(row for row in comparison.stages if row.stage == "mapping")
    assert mapping_row.ratio_b_over_a == pytest.approx(5.0)


def test_a_run_that_matches_the_reference_meets_every_bound(synthetic_pair: RunPair, galileo_input_dir: Path) -> None:
    """The bounds are relative, so reproducing run A exactly passes all four.

    Under the plan's old absolute figures this same pair failed: 24 registered
    images is far below 220, however faithfully B reproduces A.
    """
    comparison: Comparison = comparison_for(synthetic_pair, galileo_input_dir, bounds=AcceptanceBounds())

    results: dict[str, bool | None] = {check.name: check.passed for check in comparison.acceptance}
    assert set(results) == {
        "registered images",
        "mean reprojection error (px)",
        "ATE vs ground truth (mm)",
        "total runtime ratio",
    }
    assert all(results.values())


def test_acceptance_flags_a_candidate_that_falls_behind_the_reference(
    synthetic_pair: RunPair, three_samples: FramesMeta, tmp_path: Path, galileo_input_dir: Path
) -> None:
    """Twice A's reprojection error and twice A's runtime budget both fail."""
    slow_runtimes: dict[str, float] = {command: 3.0 * seconds for command, seconds in COLSFM_RUNTIMES.items()}
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", three_samples, error_px=2.0 * SYNTHETIC_ERROR_PX, runtimes=slow_runtimes),
        "colsfm",
    )

    comparison: Comparison = comparison_for((synthetic_pair[0], run_b), galileo_input_dir, bounds=AcceptanceBounds())

    results: dict[str, bool | None] = {check.name: check.passed for check in comparison.acceptance}
    assert results["mean reprojection error (px)"] is False
    assert results["total runtime ratio"] is False
    assert results["registered images"] is True


@pytest.mark.parametrize(("extra_shortfall", "expected"), [(0, True), (1, False)])
def test_the_registered_image_bound_allows_a_small_shortfall(
    synthetic_pair: RunPair, galileo_input_dir: Path, extra_shortfall: int, expected: bool
) -> None:
    """B may register up to `registered_images_allowance` fewer images than A, and no more."""
    bounds: AcceptanceBounds = AcceptanceBounds()
    metrics_a: RunMetrics = comparison_for(synthetic_pair, galileo_input_dir).run_a
    shortfall: int = bounds.registered_images_allowance + extra_shortfall
    metrics_b: RunMetrics = dataclasses.replace(
        metrics_a,
        name="colsfm",
        reconstruction=dataclasses.replace(
            metrics_a.reconstruction, registered_images=metrics_a.reconstruction.registered_images - shortfall
        ),
    )

    checks: dict[str, bool | None] = {
        check.name: check.passed for check in check_acceptance(metrics_a, metrics_b, 1.0, bounds)
    }

    assert checks["registered images"] is expected


def test_a_comparison_without_bounds_checks_nothing(synthetic_pair: RunPair, galileo_input_dir: Path) -> None:
    """Acceptance is data-driven: no bounds in, no checks out, whatever the dataset."""
    assert comparison_for(synthetic_pair, galileo_input_dir, dataset="robocap").acceptance == ()


def test_bounds_are_not_a_galileo_privilege(synthetic_pair: RunPair, galileo_input_dir: Path) -> None:
    """The dataset label decides nothing; the bounds and the data decide which rows exist."""
    comparison: Comparison = comparison_for(synthetic_pair, galileo_input_dir, dataset="robocap", bounds=AcceptanceBounds())

    assert {check.name for check in comparison.acceptance} == {
        "registered images",
        "mean reprojection error (px)",
        "ATE vs ground truth (mm)",
        "total runtime ratio",
    }


def test_a_bound_that_cannot_be_measured_is_not_a_failure(synthetic_pair: RunPair, galileo_input_dir: Path) -> None:
    """A reference run that matched fewer than three samples has a NaN ATE.

    That makes the *bound* NaN too, and `value <= nan` is False — which used to
    fail the check and, with it, the whole run. Not measurable is a third state:
    the check reports `passed is None`, the report prints `n/a`, and the tally
    counts only the bounds that could be evaluated.
    """
    comparison: Comparison = comparison_for(synthetic_pair, galileo_input_dir, bounds=AcceptanceBounds())
    unmatched: TrajectoryMetrics = TrajectoryMetrics(
        reference="ground_truth",
        num_matched=1,
        rmse_millimeters=float("nan"),
        max_millimeters=float("nan"),
        would_be_scale=float("nan"),
        reference_path_length_meters=0.0,
    )
    metrics_a: RunMetrics = dataclasses.replace(comparison.run_a, vs_ground_truth=unmatched)

    checks = check_acceptance(metrics_a, comparison.run_b, 1.0, AcceptanceBounds())

    verdicts: dict[str, bool | None] = {check.name: check.passed for check in checks}
    assert verdicts["ATE vs ground truth (mm)"] is None
    assert verdicts["registered images"] is True
    report: str = render_markdown_report(dataclasses.replace(comparison, acceptance=checks))
    assert "| ATE vs ground truth (mm) |" in report
    assert "| n/a |" in report
    assert "**3/3 bounds met.** 1 not measurable." in report


def test_a_candidate_that_timed_nothing_is_not_a_free_pass(
    synthetic_pair: RunPair, three_samples: FramesMeta, tmp_path: Path, galileo_input_dir: Path
) -> None:
    """A run with no `runtime.csv` has no total, and no total is not a fast run.

    Summing an empty log gave 0.0 s, so the candidate looked infinitely fast and
    met the runtime-ratio bound. A missing measurement stays missing: the total is
    None, the ratio is None, and the bound reports "not measurable" instead of
    PASS.
    """
    run_b: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "b", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes={}), "colsfm"
    )

    comparison: Comparison = comparison_for((synthetic_pair[0], run_b), galileo_input_dir, bounds=AcceptanceBounds())

    assert comparison.run_b.total_runtime_seconds is None
    assert comparison.run_a.total_runtime_seconds == pytest.approx(sum(BLOB_RUNTIMES.values()))
    assert comparison.total_runtime_ratio is None
    verdicts: dict[str, bool | None] = {check.name: check.passed for check in comparison.acceptance}
    assert verdicts["total runtime ratio"] is None
    report: str = render_markdown_report(comparison)
    assert "| total runtime ratio |" in report
    assert "not measurable" in report


def test_a_reference_that_timed_nothing_leaves_the_ratio_unmeasurable(
    synthetic_pair: RunPair, three_samples: FramesMeta, tmp_path: Path, galileo_input_dir: Path
) -> None:
    """The other side is symmetric: no reference total, no ratio, no verdict."""
    run_a: RunArtifacts = read_run(
        write_synthetic_run(tmp_path / "a", three_samples, error_px=SYNTHETIC_ERROR_PX, runtimes={}), "blob"
    )

    comparison: Comparison = comparison_for((run_a, synthetic_pair[1]), galileo_input_dir, bounds=AcceptanceBounds())

    assert comparison.run_a.total_runtime_seconds is None
    assert comparison.total_runtime_ratio is None
    assert {check.name: check.passed for check in comparison.acceptance}["total runtime ratio"] is None


def test_the_comparison_round_trips_through_pyserde(synthetic_pair: RunPair, galileo_input_dir: Path) -> None:
    """The JSON dump is lossless, so the report and the machine-readable form agree."""
    comparison: Comparison = comparison_for(synthetic_pair, galileo_input_dir, bounds=AcceptanceBounds())

    restored: Comparison = from_json(Comparison, to_json(comparison))

    assert restored == comparison
    assert "| Stage | A (s) | B (s) | B/A |" in render_markdown_report(restored)


# ─────────────────────────────────────────────────────────────────────────────
# Real data
# ─────────────────────────────────────────────────────────────────────────────


