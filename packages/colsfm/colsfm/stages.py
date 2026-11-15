"""One function per stage, and the typed result each one returns.

The stages themselves, with no timing, no `runtime.csv` row and no idea what runs
next: `colsfm.pipeline` composes them and `colsfm.run_lifecycle` times them. That
separation is what lets a caller that wants to step through a dataset one stage
at a time -- the Rerun walkthrough, `run_stage`, a notebook -- reuse exactly the
code a full run executes rather than a parallel copy of it.

Each `run_*_stage` takes what it needs and returns the stage's own result. A
stage that produced one object returns that object: there is no wrapper carrying
a single field, because a wrapper with one field states no contract the field
does not state itself.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from jaxtyping import Int
from numpy import ndarray

from colsfm.config import CusfmConfig, KeyframeSelectionConfig, PoseGraphConfig, read_config_directory
from colsfm.database import (
    Descriptors,
    ImagePair,
    Keypoints,
    create_database,
    pairs_with_matches,
    read_descriptors_from_database,
    read_keypoints_batch,
)
from colsfm.export import (
    colour_points_from_images,
    write_colmap_model,
    write_optimised_frames_meta,
    write_pose_files,
    write_tum_file,
)
from colsfm.extrinsic_refinement import ExtrinsicRefinementOptions, ExtrinsicRefinementResult, refine_extrinsics
from colsfm.features import ExtractionReport, FeatureOptions, extract_features
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, RigFrame, read_frames_meta, write_frames_meta
from colsfm.geometry import TumPose
from colsfm.keyframe_selection import KeyframeSelection, apply_selection, select_keyframes
from colsfm.loop_closure import (
    LoopClosureConfig,
    LoopClosureDiagnostics,
    LoopClosureResult,
    LoopSearchPlan,
    plan_loop_search,
    verify_loop_plan,
)
from colsfm.mapping import MappingOptions, MappingResult, run_mapping
from colsfm.matching import (
    MatchingOptions,
    MatchReport,
    match_pairs,
)
from colsfm.pairs import select_pairs
from colsfm.pose_graph import PoseGraphEdge, PoseGraphResult, RigNode, sequential_edges, solve_pose_graph
from colsfm.reconstruction import PosedModel, build_reconstruction, gauge_camera_params_id
from colsfm.retrieval import RetrievalIndex, build_retrieval_index
from colsfm.run_config import PipelineOptions, SelectionOptions
from colsfm.run_lifecycle import (
    LOOP_EDGES_NAME,
    POSE_GRAPH_DIR_NAME,
    VEHICLE_POSE_TUM_NAME,
)
from colsfm.run_report import (
    ExtrinsicChange,
    extrinsic_changes,
    write_loop_edges,
)


def rig_nodes(frames_meta: FramesMeta) -> list[RigNode]:
    """Turn a collection's rig frames into pose-graph nodes.

    Args:
        frames_meta: The selected collection.

    Returns:
        One node per `synced_sample_id`, in rig-frame order, carrying `world_T_rig`
        as `FramesMeta.rig_frames` derives it (from the lowest keyframe id).
    """
    return [
        RigNode(
            rig_id=rig_frame.synced_sample_id,
            world_T_rig=rig_frame.world_T_vehicle,
            timestamp_us=rig_frame.timestamp_microseconds,
        )
        for rig_frame in frames_meta.rig_frames()
    ]


def camera_poses_from_rig_poses(
    frames_meta: FramesMeta,
    world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d],
) -> dict[int, pycolmap.Rigid3d]:
    """Derive every camera pose from the optimised rig poses.

    `world_T_cam = world_T_rig * rig_T_cam`, where `rig_T_cam` is the metadata's
    `sensor_to_vehicle_transform` — the composition `pose_graph_main` writes back
    into its own `frames_meta.json` (docs/spec/pose_graph_main.md §4).

    Args:
        frames_meta: The collection whose keyframes to re-pose.
        world_T_rig_by_rig_id: Optimised rig pose per `synced_sample_id`.

    Returns:
        `world_T_cam` per keyframe id; keyframes whose rig frame is missing from
        the mapping are left out, so the caller keeps their original pose.
    """
    world_T_cam_by_keyframe_id: dict[int, pycolmap.Rigid3d] = {}
    for keyframe in frames_meta.keyframes:
        world_T_rig: pycolmap.Rigid3d | None = world_T_rig_by_rig_id.get(keyframe.synced_sample_id)
        if world_T_rig is None:
            continue
        rig_T_cam: pycolmap.Rigid3d = frames_meta.cameras[keyframe.camera_params_id].vehicle_T_cam
        world_T_cam_by_keyframe_id[keyframe.keyframe_id] = world_T_rig * rig_T_cam
    return world_T_cam_by_keyframe_id


def optimised_camera_poses(reconstruction: pycolmap.Reconstruction) -> dict[int, pycolmap.Rigid3d]:
    """Read `world_T_cam` per image out of an adjusted reconstruction.

    Args:
        reconstruction: The model bundle adjustment left behind.

    Returns:
        `world_T_cam` per image id, for every image of a registered frame.
    """
    poses: dict[int, pycolmap.Rigid3d] = {}
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        poses[image_id] = image.cam_from_world().inverse()
    return poses


def _write_vehicle_pose_file(path: Path, nodes: Sequence[RigNode], world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d]) -> None:
    """Write the rig trajectory `pose_graph_main` leaves in its output directory.

    Args:
        path: Destination `.tum` file; parent directories are created.
        nodes: The pose-graph nodes, for their timestamps.
        world_T_rig_by_rig_id: Optimised rig pose per `synced_sample_id`.
    """
    poses: list[TumPose] = [
        TumPose(timestamp_microseconds=node.timestamp_us, world_T_body=world_T_rig_by_rig_id[node.rig_id])
        for node in sorted(nodes, key=lambda entry: (entry.timestamp_us, entry.rig_id))
        if node.rig_id in world_T_rig_by_rig_id
    ]
    write_tum_file(path, poses)


def _largest_translation_change_m(nodes: Sequence[RigNode], world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d]) -> float:
    """Largest rig position change between the input poses and the solved ones.

    Args:
        nodes: The pose-graph nodes as they entered the solve.
        world_T_rig_by_rig_id: Optimised rig pose per `synced_sample_id`.

    Returns:
        The maximum displacement in metres, or 0.0 for an empty graph.
    """
    changes: list[float] = [
        float(np.linalg.norm(world_T_rig_by_rig_id[node.rig_id].translation - node.world_T_rig.translation))
        for node in nodes
        if node.rig_id in world_T_rig_by_rig_id
    ]
    return max(changes) if changes else 0.0


def _print_mapping(mapping: MappingResult) -> None:
    """Print one mapping pass's headline counts.

    Args:
        mapping: What `run_mapping` returned.
    """
    print(
        f"[colsfm] mapping: {mapping.num_images_with_observations}/{mapping.num_registered_images} images with points, "
        f"{mapping.num_points3D} points, {mapping.mean_reprojection_error_px:.4f} px, "
        f"track {mapping.mean_track_length:.3f}"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# the stages
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class KeyframeSelectionStageResult:
    """Stage 1: the configuration read and the keyframes that survived selection."""

    config: CusfmConfig
    """The whole config profile, with the command line's selection gates folded in."""
    frames_meta: FramesMeta
    """The input collection, unfiltered."""
    selected: FramesMeta
    """The kept keyframes, renumbered into the dense 1-based rig numbering."""
    rig_frames: tuple[RigFrame, ...]
    """Rig frames among the selected keyframes, one per `synced_sample_id`, in time
    order. Bound once here: `FramesMeta.rig_frames` regroups the whole collection on
    every call, and the stage's own print line and the summary both want it."""


def run_keyframe_selection_stage(options: SelectionOptions) -> KeyframeSelectionStageResult:
    """Read the config and the input metadata and apply cuSFM's keyframe selection.

    Writes nothing; `run_pipeline` is what saves `keyframes/frames_meta.json`, so
    `run_stage` can inspect a dataset without creating a workspace.

    Args:
        options: The metadata-only options: the config directory and the three gates.

    Returns:
        The configuration, the input collection and the selected one.

    Raises:
        FileNotFoundError: When the input metadata or a config file is missing.
        ValueError: When no keyframe survives selection.
    """
    if not options.frames_meta_path.is_file():
        raise FileNotFoundError(f"No {FRAMES_META_NAME} at {options.frames_meta_path}")
    config: CusfmConfig = read_config_directory(
        options.config_dir,
        KeyframeSelectionConfig(
            min_inter_frame_distance_m=options.min_inter_frame_distance,
            min_inter_frame_rotation_degrees=options.min_inter_frame_rotation_degrees,
            sample_sync_threshold_microseconds=options.sample_sync_threshold_microseconds,
        ),
    )
    frames_meta: FramesMeta = read_frames_meta(options.frames_meta_path)
    selection: KeyframeSelection = select_keyframes(
        frames_meta,
        min_inter_frame_distance_m=config.keyframe_selection.min_inter_frame_distance_m,
        min_inter_frame_rotation_degrees=config.keyframe_selection.min_inter_frame_rotation_degrees,
        sample_sync_threshold_microseconds=config.keyframe_selection.sample_sync_threshold_microseconds,
    )
    selected: FramesMeta = apply_selection(frames_meta, selection)
    if not selected.keyframes:
        raise ValueError("keyframe selection kept no frames; loosen the distance and rotation gates")
    return KeyframeSelectionStageResult(
        config=config, frames_meta=frames_meta, selected=selected, rig_frames=selected.rig_frames()
    )


def run_feature_extraction_stage(
    options: PipelineOptions, selected: FramesMeta, feature_options: FeatureOptions
) -> ExtractionReport:
    """Create the COLMAP database and run ALIKED over the selected keyframes.

    Args:
        options: The run's options, for the database path and the image root.
        selected: The collection keyframe selection kept.
        feature_options: The extractor's resolved settings, as `resolve_run` produced
            them; `PipelineOptions.resolve()` is the one place that derives them.

    Returns:
        The extraction report: per-image keypoint counts, the device used and the
        wall time.

    Raises:
        FileNotFoundError: When the image root is missing.
    """
    create_database(options.database_path, selected, overwrite=True)
    return extract_features(
        options.database_path,
        options.input_dir,
        [keyframe.image_name for keyframe in selected.keyframes],
        feature_options,
    )


def run_pair_selection_stage(selected: FramesMeta, config: CusfmConfig) -> list[ImagePair]:
    """Enumerate the pairs `feature_matcher_task_builder_main` would have written.

    Args:
        selected: The collection keyframe selection kept.
        config: The config profile, for `connected_keyframe_num`.

    Returns:
        The normalised, deduplicated, sorted image pairs; no loop pair is here yet.

    Raises:
        ValueError: When `connected_keyframe_num` is below 1.
    """
    pairs: list[ImagePair] = select_pairs(selected, config.pose_graph.connected_keyframe_num)
    print(
        f"[colsfm] {len(pairs)} pairs from connected_keyframe_num="
        f"{config.pose_graph.connected_keyframe_num} and {len(selected.stereo_pairs)} stereo declarations"
    )
    return pairs


def run_matching_stage(
    options: PipelineOptions, pairs: Sequence[ImagePair], matching_options: MatchingOptions
) -> MatchReport:
    """Match and verify every selected pair into the database.

    Args:
        options: The run's options, for the database path.
        pairs: The pairs stage 3 selected.
        matching_options: Matching and verification settings.

    Returns:
        The match report: per-pair raw and inlier counts, the device used and the
        wall time.
    """
    return match_pairs(options.database_path, list(pairs), matching_options)


@dataclass(frozen=True, slots=True)
class LoopClosureStageResult:
    """Stage 5: what the loop-closure search produced, whether it ran or not."""

    edges: list[PoseGraphEdge]
    """Verified, gated loop constraints; empty when the stage is off or found nothing."""
    num_pairs_matched: int
    """Image pairs this stage had to match itself, i.e. the candidates stage 4 had not
    already matched. The candidates it reused are not counted: `summary.json` reports
    the *extra* matching work loop closure cost."""
    diagnostics: LoopClosureDiagnostics | None = None
    """`find_loop_edges`' own rejection breakdown, or None when the stage did not run.

    Zero loop edges is the normal outcome on a short or low-recall sequence, and only
    these counters say which it was. The stage already prints them; carrying the object
    means a caller — a viewer, a notebook — can read them instead of parsing stdout.
    None on the two paths that return before the search: loop closure switched off, and
    a database with no descriptors."""


def _warn_about_closed_loop_gates(pose_graph_config: PoseGraphConfig) -> None:
    """Say so, loudly, when the config's loop gates will reject every edge.

    `gate_loop_edges` accepts a candidate whose relative pose is within
    `max_translation_m` and `max_rotation_deg` of the retrieval estimate, so a
    threshold of exactly 0.0 accepts nothing. Both are proto3 scalars, so the
    stock `pycusfm/configs/isaac` and `pycusfm/configs/av` profiles — which set
    neither — read back as 0.0 and silently discard 100 % of the edges after
    paying for the whole stage. Only `data/cusfm_configs/loop-closure-fixed`
    sets them (1.0 m / 10 deg). The gates keep their meaning; this only refuses
    to be quiet about the outcome.

    Args:
        pose_graph_config: `pose_graph_config.pb.txt` as it was parsed.
    """
    if pose_graph_config.loop_edge_translation_threshold_meters > 0.0:
        return
    if pose_graph_config.loop_edge_rotation_threshold_degrees > 0.0:
        return
    print(
        "[colsfm] WARNING: loop closure is on but both loop gates are 0.0 "
        "(loop_edge_translation_threshold_meters and loop_edge_rotation_threshold_degrees), "
        "so every candidate edge will be rejected and the stage will produce nothing. "
        "The stock isaac and av profiles leave both unset; use "
        "--config-dir data/cusfm_configs/loop-closure-fixed, or set the two thresholds."
    )


def run_loop_closure_stage(
    frames_meta: FramesMeta,
    database_path: Path,
    pose_graph_config: PoseGraphConfig,
    matching_options: MatchingOptions,
    *,
    enabled: bool,
) -> LoopClosureStageResult:
    """Retrieve, match and verify loop candidates.

    Three steps, none of which matches anything it does not have to: `plan_loop_search`
    decides which rig pairs to measure and which image pairs that needs, one batch
    `match_pairs` fills in whatever the database does not already hold, and
    `verify_loop_plan` measures and gates the plan.

    Args:
        frames_meta: The selected collection, in the pose graph's frame conventions.
        database_path: The database holding the keypoints and descriptors.
        pose_graph_config: `pose_graph_config.pb.txt`, whose loop gates this honours.
        matching_options: Settings for the batch match over the missing pairs.
        enabled: Whether to run at all.

    Returns:
        The loop edges and how many extra pairs were matched to find them.
    """
    if not enabled:
        print("[colsfm] loop closure disabled (--loop-closure to enable); no loop edges")
        return LoopClosureStageResult(edges=[], num_pairs_matched=0)
    _warn_about_closed_loop_gates(pose_graph_config)

    descriptors: dict[int, Descriptors] = read_descriptors_from_database(database_path)
    if not descriptors:
        print("[colsfm] loop closure: the database holds no descriptors; no loop edges")
        return LoopClosureStageResult(edges=[], num_pairs_matched=0)
    index: RetrievalIndex = build_retrieval_index(descriptors)
    # The index is built; the raw descriptors are not read again. On RoboCap they are
    # 4.7 GB, which would otherwise stay resident for the whole stage -- through the
    # keypoint read, the batch match and the measurement.
    del descriptors
    print(f"[colsfm] loop closure: retrieval index over {len(index.image_ids)} images in {index.build_seconds:.2f}s")

    config: LoopClosureConfig = LoopClosureConfig.from_pose_graph(pose_graph_config)
    # Read once and handed to the plan: the keypoints are the same file either side of
    # the batch match, and on RoboCap they are 148 MB.
    keypoints: dict[int, Keypoints] = read_keypoints_batch(database_path, index.image_ids, dtype=np.float64)

    plan: LoopSearchPlan = plan_loop_search(frames_meta, database_path, index, config, keypoints)
    # Presence, not count: a pair stage 4 matched and found nothing in is finished work,
    # and matching it again would cost a matcher call to produce the same nothing. Most
    # of the plan is stage 4's own consecutive and stereo pairs (8 400 of 14 443 on
    # RoboCap), which is what makes this check worth making at all.
    present: set[ImagePair] = pairs_with_matches(database_path, plan.image_pairs)
    missing_pairs: list[ImagePair] = [pair for pair in plan.image_pairs if pair not in present]
    print(
        f"[colsfm] loop closure: {len(plan.rig_pairs)} shortlisted rig pairs need "
        f"{len(plan.image_pairs)} image pairs, {len(present)} already matched, {len(missing_pairs)} to match"
    )
    if missing_pairs:
        match_pairs(database_path, missing_pairs, matching_options)

    # One handle for the whole verification pass: a plan of tens of thousands of pairs
    # would otherwise pay a SQLite open and close per `match_fn` call.
    with pycolmap.Database.open(database_path) as database:

        def read_matches(image_id_a: int, image_id_b: int) -> Int[ndarray, "num_matches 2"]:
            """Read one pair's raw matches back out of the database."""
            return np.asarray(database.read_matches(image_id_a, image_id_b), dtype=np.int64)

        result: LoopClosureResult = verify_loop_plan(plan, read_matches)
    edges: list[PoseGraphEdge] = list(result.edges)
    # Print the counters, not just the edge count: zero edges is the normal outcome on a
    # short or a low-recall sequence, and only the rejection breakdown says which it was.
    print(
        f"[colsfm] loop closure: {len(edges)} loop edges | {result.diagnostics.counters.queries} queries, "
        f"{result.diagnostics.counters.candidates_retrieved} retrieved, {result.diagnostics.counters.rejected_by_score} below score, "
        f"{result.diagnostics.counters.rejected_by_time} inside the {result.diagnostics.min_time_gap_seconds:.2f}s gap, "
        f"{result.diagnostics.counters.rejected_by_geometry} failed geometry, {result.diagnostics.counters.rejected_by_is_good} not good, "
        f"{result.diagnostics.counters.verified} verified over {len(plan.image_pairs)} image pairs"
    )
    return LoopClosureStageResult(
        edges=edges, num_pairs_matched=len(missing_pairs), diagnostics=result.diagnostics
    )


@dataclass(frozen=True, slots=True)
class PoseGraphStageResult:
    """Stage 6: the solved rig trajectory and the collection re-posed through it."""

    nodes: list[RigNode]
    """The rig nodes as they entered the solve, carrying the input poses."""
    edges: list[PoseGraphEdge]
    """Sequential and loop constraints together, in the order they were solved."""
    frames_meta: FramesMeta
    """The selected collection with every camera pose re-derived through the rig."""
    translation_change_m: float
    """Largest rig position change the solve produced, in metres."""
    solve: PoseGraphResult | None = None
    """The Ceres summary and the solved poses, or None when there was nothing to solve.

    None exactly when `edges` is empty, which happens only on a single-rig-frame
    collection; the input poses stand and no solver ran."""


def run_pose_graph_stage(
    options: PipelineOptions, selected: FramesMeta, config: CusfmConfig, loop_edges: Sequence[PoseGraphEdge]
) -> PoseGraphStageResult:
    """Solve the rig pose graph and write `pose_graph/`.

    Args:
        options: The run's options, for the output directory.
        selected: The collection keyframe selection kept.
        config: The config profile, for `connected_keyframe_num`.
        loop_edges: Extra constraints from stage 5; may be empty.

    Returns:
        The nodes, the edges, the re-posed collection and the largest move.
    """
    nodes: list[RigNode] = rig_nodes(selected)
    edges: list[PoseGraphEdge] = sequential_edges(nodes, config.pose_graph.connected_keyframe_num)
    edges.extend(loop_edges)
    world_T_rig_by_rig_id: dict[int, pycolmap.Rigid3d] = {node.rig_id: node.world_T_rig for node in nodes}
    solved: PoseGraphResult | None = None
    if edges:
        solved = solve_pose_graph(nodes, edges)
        world_T_rig_by_rig_id = solved.world_T_rig
        print(
            f"[colsfm] pose graph: {len(nodes)} nodes, {len(edges)} edges "
            f"({len(loop_edges)} loop), {solved.termination} after {solved.iterations} iterations, "
            f"cost {solved.initial_cost:.6g} -> {solved.final_cost:.6g}"
        )
    else:
        print("[colsfm] pose graph: a single rig frame has no edges; the input poses stand")
    # Written unconditionally, even with the solve a no-op: the camera poses are
    # re-derived through the rig, which is what makes the collection rigid and what
    # `pose_graph_main` itself writes out.
    pose_graph_meta: FramesMeta = selected.with_camera_to_world(camera_poses_from_rig_poses(selected, world_T_rig_by_rig_id))
    pose_graph_dir: Path = options.output_dir / POSE_GRAPH_DIR_NAME
    write_frames_meta(pose_graph_dir / FRAMES_META_NAME, pose_graph_meta)
    _write_vehicle_pose_file(pose_graph_dir / VEHICLE_POSE_TUM_NAME, nodes, world_T_rig_by_rig_id)
    write_loop_edges(pose_graph_dir / LOOP_EDGES_NAME, loop_edges)
    return PoseGraphStageResult(
        nodes=nodes,
        edges=edges,
        frames_meta=pose_graph_meta,
        translation_change_m=_largest_translation_change_m(nodes, world_T_rig_by_rig_id),
        solve=solved,
    )


def run_reconstruction_stage(
    options: PipelineOptions, pose_graph_meta: FramesMeta, config: CusfmConfig, mapping_options: MappingOptions
) -> MappingResult:
    """Triangulate and bundle-adjust the pose-graph poses.

    With `--optimize-extrinsics` the reconstruction is built on the **gauge
    camera** rather than the vehicle body, because COLMAP freezes every
    `sensor_from_rig` of a rig whose reference sensor owns no images
    (`colsfm.reconstruction`). Every `cam_T_world` is identical either way, so
    this pass produces the same model and stage 7b can refine the extrinsics of
    the very model it left behind instead of mapping the scene a second time.

    Args:
        options: The run's options; `optimize_extrinsics` chooses the rig origin.
        pose_graph_meta: The collection stage 6 re-posed.
        config: The config profile, for `vision_mapping_config`.
        mapping_options: Command-line style mapper knobs.

    Returns:
        The adjusted reconstruction, its rig reference and the per-round statistics.

    Raises:
        FileNotFoundError: When the database does not exist.
        ValueError: When no frame is registered.
    """
    reference_camera_params_id: int | None = (
        gauge_camera_params_id(pose_graph_meta) if options.optimize_extrinsics else None
    )
    model: PosedModel = build_reconstruction(pose_graph_meta, reference_camera_params_id=reference_camera_params_id)
    mapping: MappingResult = run_mapping(model, options.database_path, config.vision_mapping, mapping_options)
    _print_mapping(mapping)
    return mapping


@dataclass(frozen=True, slots=True)
class ExtrinsicRefinementStageResult:
    """Stage 7b: the alternating extrinsic refinement's model and its movement."""

    mapping: MappingResult
    """The refined mapping result; the same reconstruction, further adjusted."""
    frames_meta: FramesMeta
    """The pose-graph collection carrying the refined `sensor_to_vehicle_transform`."""
    changes: tuple[ExtrinsicChange, ...]
    """Per-camera movement, ordered by `camera_params_id`."""
    rounds_run: int
    """Alternation rounds the refinement took before it converged."""


def run_extrinsic_refinement_stage(
    options: PipelineOptions,
    pose_graph_meta: FramesMeta,
    config: CusfmConfig,
    mapping_options: MappingOptions,
    mapping: MappingResult,
) -> ExtrinsicRefinementStageResult:
    """Refine the rig extrinsics of the model stage 7 produced.

    The blob's second `keypoints_mapper_main` pass: the same matches and the same
    pose-graph poses, this time with `--optimize_extrinsics=True`, overwriting
    `kpmap/` (`pycusfm.cusfm_runner.refine_extrinsics`).

    Args:
        options: The run's options; the refinement's ceiling and the prior switch.
        pose_graph_meta: The collection stage 6 re-posed — what both priors pull towards.
        config: The config profile, for the sigmas and the solver settings.
        mapping_options: Command-line style mapper knobs, for the (B) half of the alternation.
        mapping: What stage 7 produced, on a camera-referenced rig.

    Returns:
        The refined mapping result, the re-calibrated collection and the movement.

    Raises:
        ValueError: When stage 7 built a vehicle-referenced rig, whose extrinsics
            COLMAP would silently hold constant.
    """
    refinement: ExtrinsicRefinementResult = refine_extrinsics(
        mapping,
        options.database_path,
        config.vision_mapping,
        mapping_options,
        {camera_params_id: camera.vehicle_T_cam for camera_params_id, camera in pose_graph_meta.cameras.items()},
        ExtrinsicRefinementOptions(
            regularised=options.regularised_extrinsics,
            num_rounds=options.extrinsic_refinement_rounds,
            use_absolute_prior=config.vision_mapping.use_camera_extrinsic_constraint,
            use_relative_prior=config.vision_mapping.use_camera_extrinsic_constraint,
        ),
    )
    reference_camera_params_id: int = gauge_camera_params_id(pose_graph_meta)
    changes: tuple[ExtrinsicChange, ...] = extrinsic_changes(
        pose_graph_meta, refinement.refined_extrinsics, reference_camera_params_id
    )
    _print_mapping(refinement.mapping)
    print(
        f"[colsfm] extrinsic refinement "
        f"({'regularised' if refinement.regularised else 'unregularised'}): rig origin on camera "
        f"{reference_camera_params_id} "
        f"({pose_graph_meta.cameras[reference_camera_params_id].sensor_name}), moves "
        + ", ".join(f"{change.camera_params_id}={change.translation_change_mm:.2f} mm" for change in changes)
    )
    return ExtrinsicRefinementStageResult(
        mapping=refinement.mapping,
        frames_meta=pose_graph_meta.with_extrinsics(refinement.refined_extrinsics),
        changes=changes,
        rounds_run=len(refinement.rounds),
    )


@dataclass(frozen=True, slots=True)
class ExportStageResult:
    """Stage 8: the four artifacts a cuSFM run leaves behind."""

    optimised: FramesMeta
    """The collection written to `kpmap/keyframes/frames_meta.json`."""
    num_coloured: int
    """Points that got a colour from the source imagery."""
    pose_files: list[Path]
    """Every TUM trajectory written, per-camera files first."""


def run_export_stage(options: PipelineOptions, mapped_meta: FramesMeta, mapping: MappingResult) -> ExportStageResult:
    """Colour the points and write `sparse/`, `kpmap/` and `output_poses/`.

    Args:
        options: The run's options, for the output directory and the image root.
        mapped_meta: The collection carrying whatever extrinsics were just solved.
        mapping: The final mapping result.

    Returns:
        The exported collection and what was written.

    Raises:
        FileNotFoundError: When the image root is missing.
    """
    num_coloured: int = colour_points_from_images(mapping.reconstruction, options.input_dir, options.num_threads)
    print(f"[colsfm] export: coloured {num_coloured}/{mapping.num_points3D} points from the source imagery")
    write_colmap_model(options.output_dir, mapping.reconstruction)
    # `camera_to_world` comes straight off the adjusted reconstruction, so it stays
    # consistent with whatever extrinsics were just written next to it.
    optimised: FramesMeta = write_optimised_frames_meta(
        options.output_dir, mapped_meta, optimised_camera_poses(mapping.reconstruction), "ALIGNMENT"
    )
    written: list[Path] = write_pose_files(options.output_dir, optimised)
    print(f"[colsfm] export: sparse/ model, kpmap/keyframes/{FRAMES_META_NAME}, {len(written)} TUM files")
    return ExportStageResult(optimised=optimised, num_coloured=num_coloured, pose_files=written)


