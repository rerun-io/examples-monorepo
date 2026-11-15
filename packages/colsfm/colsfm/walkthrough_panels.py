"""One function per stage: what that stage produced, drawn into the recording.

The visualisation half of `colsfm.walkthrough`. Each `log_*` takes the stage's
own typed result — the very object `colsfm.pipeline` handed the observer — plus
whatever the panel needs to read back out of the run (the database, the image
root, a reference run), and logs the spatial, tabular and textual entities its
tab shows. Every panel closes with `log_notes`, so the prose, this run's numbers
and the stage's own console output land in one document.

Nothing here runs a stage or decides what runs next: `colsfm.walkthrough`'s
observer calls these after each stage's timing window has closed, which is what
keeps the visualisation outside the run's measurements.
"""


from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pycolmap
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from simplecv.rerun_rig_logger import log_rig_pose_stream, log_rig_static
from simplecv.rig import Rig, entity_id

from colsfm.alignment import RigTrack, rig_track_from_frames_meta
from colsfm.bench_report import render_markdown_report
from colsfm.benchmark import AcceptanceBounds, Comparison, RunArtifacts, compare_runs, read_run
from colsfm.config import CusfmConfig
from colsfm.database import ImagePair, Keypoints, read_keypoints_batch, read_two_view_geometry
from colsfm.features import ExtractionReport
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, KeyframeMeta, RigFrame, read_frames_meta
from colsfm.mapping import MappingResult, RoundStats
from colsfm.matching import MatchReport, PairMatchStats
from colsfm.pairs import normalise_pair, stereo_pairs
from colsfm.pose_graph import PoseGraphEdge, RigNode
from colsfm.rerun_log import IMAGE_PLANE_DISTANCE, POINT_RADIUS, TIMELINE, Rgb, build_rig
from colsfm.stages import (
    ExportStageResult,
    ExtrinsicChange,
    ExtrinsicRefinementStageResult,
    KeyframeSelectionStageResult,
    LoopClosureStageResult,
    PipelineOptions,
    PoseGraphStageResult,
)
from colsfm.walkthrough_draw import (
    arc_strip,
    camera_centres,
    camera_frusta,
    image_path_of,
    load_rgb,
    log_notes,
    polyline_segments,
    rerun_keypoints,
    scene_extent,
    spread_indices,
)
from colsfm.walkthrough_stages import (
    EDGE_RADIUS_FRACTION,
    FRUSTUM_RADIUS_FRACTION,
    INLIER_HISTOGRAM_BINS,
    INPUT_COLOR,
    INPUT_FRUSTUM_FRACTION,
    KEYPOINT_RADIUS,
    LOOP_COLOR,
    MATCH_LINE_RADIUS,
    MAX_LOOP_ARCS,
    MAX_MATCH_LINES,
    MIN_LOOP_RIG_GAP,
    NODE_RADIUS_FRACTION,
    REFERENCE_COLOR,
    ROUND_TIMELINE,
    SELECTED_COLOR,
    SELECTED_FRUSTUM_FRACTION,
    SOLVED_COLOR,
    SOLVED_NODE_SHRINK,
    STACKED_MAX_WIDTH,
    STAGE_TIMELINE,
    STEREO_COLOR,
    MarkdownRows,
    SamplePane,
    WalkthroughStage,
)

# ══════════════════════════════════════════════════════════════════════════════════════
# per-stage logging
# ══════════════════════════════════════════════════════════════════════════════════════


def _selection_verdict(num_input: int, num_kept: int) -> str:
    """Say in one line whether the selection gates actually removed anything.

    Args:
        num_input: Keyframes in the input collection.
        num_kept: Keyframes that survived.

    Returns:
        A sentence for the stage's notes table.
    """
    if num_kept == num_input:
        return (
            f"nothing was dropped — all {num_input} keyframes cleared the gates, so the grey and the blue frusta "
            "coincide. Pass `--min-inter-frame-distance 0.5` (the stock `feature_extractor_main` value) to watch the "
            "rule bite: on Galileo it keeps 34"
        )
    return f"dropped {num_input - num_kept} of {num_input} keyframes; the grey frusta with no blue frustum inside them are the ones that went"


def log_keyframe_selection(stage: WalkthroughStage, selection: KeyframeSelectionStageResult, console: str) -> None:
    """Stage 1: every input pose in grey, the kept ones in blue, and the rig track.

    Args:
        stage: This stage's timeline slot and prose.
        selection: What `run_keyframe_selection_stage` returned.
        console: What it printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    all_keyframes: tuple[KeyframeMeta, ...] = selection.frames_meta.keyframes
    kept: tuple[KeyframeMeta, ...] = selection.selected.keyframes
    input_centres: Float64[ndarray, "n 3"] = camera_centres(all_keyframes)
    extent: float = scene_extent(input_centres)
    rr.log(
        f"{stage.entity_root}/input/frusta",
        rr.LineStrips3D(
            camera_frusta(selection.frames_meta, all_keyframes, INPUT_FRUSTUM_FRACTION * extent),
            colors=INPUT_COLOR,
            radii=FRUSTUM_RADIUS_FRACTION * extent,
        ),
    )
    rr.log(
        f"{stage.entity_root}/input/centres",
        rr.Points3D(input_centres, colors=INPUT_COLOR, radii=0.5 * NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/selected/frusta",
        rr.LineStrips3D(
            camera_frusta(selection.selected, kept, SELECTED_FRUSTUM_FRACTION * extent),
            colors=SELECTED_COLOR,
            radii=FRUSTUM_RADIUS_FRACTION * extent,
        ),
    )
    rr.log(
        f"{stage.entity_root}/selected/centres",
        rr.Points3D(camera_centres(kept), colors=SELECTED_COLOR, radii=SOLVED_NODE_SHRINK * NODE_RADIUS_FRACTION * extent),
    )
    rig_positions: Float64[ndarray, "n 3"] = np.asarray(
        [rig_frame.world_T_vehicle.translation for rig_frame in selection.rig_frames], dtype=np.float64
    ).reshape(-1, 3)
    rr.log(
        f"{stage.entity_root}/selected/rig_track",
        rr.LineStrips3D(polyline_segments(rig_positions), colors=SOLVED_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    gates = selection.config.keyframe_selection
    log_notes(
        stage,
        [
            ("input keyframes", str(len(all_keyframes))),
            ("kept keyframes", str(len(kept))),
            ("rig frames kept", str(len(selection.rig_frames))),
            ("cameras", str(len(selection.selected.cameras))),
            ("distance gate", f"{gates.min_inter_frame_distance_m:.3f} m"),
            ("rotation gate", f"{gates.min_inter_frame_rotation_degrees:.3f} deg"),
            ("rig sync window", f"{gates.sample_sync_threshold_microseconds} us"),
            ("selection rule", "keep a rig sample when it moved past the distance gate **or** turned past the rotation gate"),
            ("what the rule did here", _selection_verdict(len(all_keyframes), len(kept))),
        ],
        console,
    )


def sample_keyframes(selected: FramesMeta, wanted: int) -> list[KeyframeMeta]:
    """Pick a few keyframes spread over the sequence and over the cameras.

    Args:
        selected: The collection keyframe selection kept.
        wanted: How many samples to return.

    Returns:
        One keyframe per sampled rig frame, cycling the camera so the samples do
        not all show the same view.
    """
    rig_frames: tuple[RigFrame, ...] = selected.rig_frames()
    by_id: dict[int, KeyframeMeta] = selected.keyframe_by_id()
    samples: list[KeyframeMeta] = []
    for offset, position in enumerate(spread_indices(len(rig_frames), wanted)):
        members: tuple[int, ...] = rig_frames[position].keyframe_ids
        samples.append(by_id[members[offset % len(members)]])
    return samples


def log_feature_extraction(
    stage: WalkthroughStage,
    selected: FramesMeta,
    extraction: ExtractionReport,
    database_path: Path,
    input_dir: Path,
    num_samples: int,
    console: str,
    published_database_path: Path | None = None,
) -> list[SamplePane]:
    """Stage 2: sample keyframes with their ALIKED keypoints, plus the count histogram.

    The keypoints are read back out of the database rather than kept from the
    extractor, so what is drawn is what the matcher will see.

    Args:
        stage: This stage's timeline slot and prose.
        selected: The collection keyframe selection kept.
        extraction: The report `run_feature_extraction_stage` returned.
        database_path: The database the stage wrote, read from here and now.
        input_dir: The dataset's raw image root.
        num_samples: How many keyframes to show.
        console: What the stage printed.
        published_database_path: Where that database will live once the run publishes,
            for the notes row. Mid-run the stage writes into the run's staging
            directory (`colsfm.run_lifecycle`), and naming that in a finished
            recording would name a directory that no longer exists.

    Returns:
        One pane per keyframe that was drawn, so the blueprint can give each its
        own 2D view — stacked in one view, only the last image would be visible.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    panes: list[SamplePane] = []
    drawable: list[KeyframeMeta] = [
        keyframe for keyframe in sample_keyframes(selected, num_samples) if image_path_of(input_dir, keyframe).is_file()
    ]
    # One database handle for the panel, not one per sample: `read_keypoints` opens and
    # closes the file every call.
    keypoints_by_id: dict[int, Keypoints] = read_keypoints_batch(
        database_path, [keyframe.keyframe_id for keyframe in drawable]
    )
    for keyframe in drawable:
        sensor_name: str = selected.cameras[keyframe.camera_params_id].sensor_name
        sample_path: str = f"{stage.entity_root}/samples/{sensor_name}_{keyframe.keyframe_id}"
        image_path: Path = image_path_of(input_dir, keyframe)
        keypoints_xy: Keypoints = keypoints_by_id[keyframe.keyframe_id]
        rr.log(f"{sample_path}/image", rr.EncodedImage(path=image_path))
        rr.log(
            f"{sample_path}/keypoints",
            rr.Points2D(rerun_keypoints(keypoints_xy), colors=SELECTED_COLOR, radii=KEYPOINT_RADIUS),
        )
        panes.append(SamplePane(entity_path=sample_path, name=f"{sensor_name} #{keyframe.keyframe_id} ({len(keypoints_xy)} kp)"))
    counts: dict[int, int] = extraction.keypoint_counts
    ordered: Int64[ndarray, "n"] = np.asarray([counts[key] for key in sorted(counts)], dtype=np.int64)
    rr.log(f"{stage.entity_root}/keypoint_counts", rr.BarChart(ordered, color=SELECTED_COLOR))
    log_notes(
        stage,
        [
            ("images extracted", str(len(counts))),
            ("keypoints total", f"{int(ordered.sum()):,}"),
            ("keypoints per image (min / median / max)", f"{int(ordered.min())} / {int(np.median(ordered))} / {int(ordered.max())}"),
            ("device", extraction.device),
            ("seconds", f"{extraction.elapsed_seconds:.2f}"),
            ("database", f"`{database_path if published_database_path is None else published_database_path}`"),
            ("samples drawn", ", ".join(pane.name for pane in panes) if panes else "none; the imagery is not on disk"),
        ],
        console,
    )
    return panes


def log_pair_selection(
    stage: WalkthroughStage,
    selected: FramesMeta,
    config: CusfmConfig,
    pairs: Sequence[ImagePair],
    console: str,
) -> None:
    """Stage 3: the consecutive and stereo pairs as lines between camera centres.

    The two colours partition the stage's own `pairs` rather than re-deriving them:
    `select_pairs` is the union of the two rules, so one `stereo_pairs` set says which
    side each pair came from, and what is drawn is by construction what was selected.

    Args:
        stage: This stage's timeline slot and prose.
        selected: The collection keyframe selection kept.
        config: The config profile, for `connected_keyframe_num` in the notes.
        pairs: The pairs `run_pair_selection_stage` returned.
        console: What it printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    by_id: dict[int, KeyframeMeta] = selected.keyframe_by_id()
    declared_stereo: set[ImagePair] = set(stereo_pairs(selected))
    consecutive: list[ImagePair] = [pair for pair in pairs if pair not in declared_stereo]
    stereo: list[ImagePair] = [pair for pair in pairs if pair in declared_stereo]

    def segments(pairs: Sequence[ImagePair]) -> Float64[ndarray, "n 2 3"]:
        """Two-point segments joining each pair's camera centres."""
        if not pairs:
            return np.zeros((0, 2, 3), dtype=np.float64)
        return np.asarray(
            [[by_id[first].world_T_cam.translation, by_id[second].world_T_cam.translation] for first, second in pairs],
            dtype=np.float64,
        ).reshape(-1, 2, 3)

    centres: Float64[ndarray, "n 3"] = camera_centres(selected.keyframes)
    extent: float = scene_extent(centres)
    rr.log(f"{stage.entity_root}/centres", rr.Points3D(centres, colors=INPUT_COLOR, radii=0.5 * NODE_RADIUS_FRACTION * extent))
    rr.log(
        f"{stage.entity_root}/pairs/consecutive",
        rr.LineStrips3D(segments(consecutive), colors=SELECTED_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/pairs/stereo",
        rr.LineStrips3D(segments(stereo), colors=STEREO_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    log_notes(
        stage,
        [
            ("pairs total", str(len(pairs))),
            ("consecutive pairs (blue)", str(len(consecutive))),
            ("stereo pairs (orange)", str(len(stereo))),
            ("connected_keyframe_num", str(config.pose_graph.connected_keyframe_num)),
            ("stereo declarations", str(len(selected.stereo_pairs))),
            ("loop pairs", "0 — they cannot exist before stage 5 has run"),
        ],
        console,
    )


def _pair_for_display(report_pairs: Mapping[ImagePair, PairMatchStats]) -> ImagePair | None:
    """Pick the pair whose inlier count is the median, so the panel shows a typical pair.

    Args:
        report_pairs: The matcher's per-pair statistics.

    Returns:
        The chosen pair, or None when nothing verified.
    """
    verified: list[ImagePair] = sorted(
        (pair for pair, stats in report_pairs.items() if stats.inlier_matches > 0),
        key=lambda pair: report_pairs[pair].inlier_matches,
    )
    return verified[len(verified) // 2] if verified else None


def log_matching(
    stage: WalkthroughStage,
    selected: FramesMeta,
    matching: MatchReport,
    database_path: Path,
    input_dir: Path,
    console: str,
) -> None:
    """Stage 4: one sample pair with its verified matches, plus the inlier distribution.

    Args:
        stage: This stage's timeline slot and prose.
        selected: The collection keyframe selection kept.
        matching: The report `run_matching_stage` returned.
        database_path: The database holding the keypoints and geometries.
        input_dir: The dataset's raw image root.
        console: What the stage printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    stats: dict[ImagePair, PairMatchStats] = matching.pair_stats
    inliers: Int64[ndarray, "n"] = np.asarray([stats[pair].inlier_matches for pair in sorted(stats)], dtype=np.int64)
    rr.log(f"{stage.entity_root}/inliers_per_pair", rr.BarChart(inliers, color=SELECTED_COLOR))
    histogram: Int64[ndarray, "bins"] = np.histogram(inliers, bins=INLIER_HISTOGRAM_BINS)[0].astype(np.int64)
    rr.log(f"{stage.entity_root}/inlier_histogram", rr.BarChart(histogram, color=STEREO_COLOR))

    rows: MarkdownRows = [
        ("pairs matched", str(len(stats))),
        ("pairs with no verified geometry", str(matching.empty_pairs)),
        ("median inliers per pair", f"{matching.median_inliers:.1f}"),
        ("median inlier retention", f"{100.0 * matching.median_retention:.1f} %"),
        ("device", matching.device),
        ("seconds", f"{matching.elapsed_seconds:.2f}"),
    ]
    chosen: ImagePair | None = _pair_for_display(stats)
    if chosen is not None:
        rows.extend(_log_match_pair(stage, selected, chosen, database_path, input_dir, stats[chosen]))
    log_notes(stage, rows, console)


def _log_match_pair(
    stage: WalkthroughStage,
    selected: FramesMeta,
    pair: ImagePair,
    database_path: Path,
    input_dir: Path,
    stats: PairMatchStats,
) -> MarkdownRows:
    """Draw one verified pair: each image with its keypoints, then the two concatenated.

    Args:
        stage: This stage's timeline slot.
        selected: The collection, for the keyframes' image names.
        pair: The image pair to draw.
        database_path: The database holding the keypoints and the geometry.
        input_dir: The dataset's raw image root.
        stats: The matcher's counters for this pair.

    Returns:
        Rows describing the pair, for the stage's notes table.
    """
    by_id: dict[int, KeyframeMeta] = selected.keyframe_by_id()
    left: KeyframeMeta = by_id[pair[0]]
    right: KeyframeMeta = by_id[pair[1]]
    left_path: Path = image_path_of(input_dir, left)
    right_path: Path = image_path_of(input_dir, right)
    if not left_path.is_file() or not right_path.is_file():
        return [("sample pair", "imagery missing on disk")]

    geometry: pycolmap.TwoViewGeometry = read_two_view_geometry(database_path, pair[0], pair[1])
    matches: Int64[ndarray, "m 2"] = np.asarray(geometry.inlier_matches, dtype=np.int64).reshape(-1, 2)
    # Both sides through one handle; `read_keypoints` would open the database twice more.
    keypoints_by_id: dict[int, Keypoints] = read_keypoints_batch(database_path, pair)
    left_keypoints: Keypoints = keypoints_by_id[pair[0]]
    right_keypoints: Keypoints = keypoints_by_id[pair[1]]
    left_matched: Float64[ndarray, "m 2"] = rerun_keypoints(left_keypoints[matches[:, 0]]) if len(matches) else np.zeros((0, 2))
    right_matched: Float64[ndarray, "m 2"] = rerun_keypoints(right_keypoints[matches[:, 1]]) if len(matches) else np.zeros((0, 2))

    for side, keyframe, path, matched in (("left", left, left_path, left_matched), ("right", right, right_path, right_matched)):
        side_path: str = f"{stage.entity_root}/pair/{side}"
        rr.log(f"{side_path}/image", rr.EncodedImage(path=path))
        rr.log(f"{side_path}/keypoints", rr.Points2D(matched, colors=SOLVED_COLOR, radii=KEYPOINT_RADIUS))
        rr.log(f"{side_path}/name", rr.TextDocument(f"{selected.cameras[keyframe.camera_params_id].sensor_name}\n{keyframe.image_name}"))

    left_rgb, scale = load_rgb(left_path, STACKED_MAX_WIDTH // 2)
    right_rgb, _ = load_rgb(right_path, STACKED_MAX_WIDTH // 2)
    height: int = min(left_rgb.shape[0], right_rgb.shape[0])
    stacked: UInt8[ndarray, "h w 3"] = np.hstack([left_rgb[:height], right_rgb[:height]])
    offset_x: float = float(left_rgb.shape[1])
    scaled_left: Float64[ndarray, "m 2"] = scale * left_matched
    scaled_right: Float64[ndarray, "m 2"] = scale * right_matched + np.array([offset_x, 0.0])
    drawn: list[int] = spread_indices(len(matches), MAX_MATCH_LINES)
    lines: Float64[ndarray, "k 2 2"] = np.stack([scaled_left[drawn], scaled_right[drawn]], axis=1) if drawn else np.zeros((0, 2, 2))
    rr.log(f"{stage.entity_root}/pair/stacked/image", rr.Image(stacked))
    rr.log(
        f"{stage.entity_root}/pair/stacked/keypoints",
        rr.Points2D(np.vstack([scaled_left, scaled_right]), colors=SOLVED_COLOR, radii=KEYPOINT_RADIUS),
    )
    rr.log(f"{stage.entity_root}/pair/stacked/matches", rr.LineStrips2D(lines, colors=STEREO_COLOR, radii=MATCH_LINE_RADIUS))
    return [
        ("sample pair", f"`{left.image_name}` ↔ `{right.image_name}`"),
        ("sample pair raw matches", str(stats.raw_matches)),
        ("sample pair verified inliers", str(stats.inlier_matches)),
        ("match lines drawn", f"{len(drawn)} of {len(matches)}"),
    ]


def _rig_id_by_keyframe(frames_meta: FramesMeta) -> dict[int, int]:
    """Index each keyframe's rig-frame position, for turning image pairs into rig pairs.

    Args:
        frames_meta: The collection to index.

    Returns:
        Rig-frame position (0-based, time ordered) keyed by keyframe id.
    """
    positions: dict[int, int] = {}
    for position, rig_frame in enumerate(frames_meta.rig_frames()):
        for keyframe_id in rig_frame.keyframe_ids:
            positions[keyframe_id] = position
    return positions


def _verified_revisit_pairs(database_path: Path, rig_position: Mapping[int, int]) -> dict[tuple[int, int], int]:
    """Every verified pair in a database that spans more than one rig frame.

    A pair whose two images sit in adjacent rig frames is a sequence link, which
    stage 3 already produced; anything further apart in time only reaches the
    database because the loop-closure stage matched it there.

    Args:
        database_path: The run's COLMAP database.
        rig_position: Rig-frame position per keyframe id.

    Returns:
        Summed inlier count keyed by the ordered rig-frame position pair.
    """
    with pycolmap.Database.open(database_path) as database:
        pair_ids, counts = database.read_two_view_geometry_num_inliers()
    revisits: dict[tuple[int, int], int] = {}
    for pair_id, count in zip(pair_ids, counts, strict=True):
        image_id_a, image_id_b = pycolmap.pair_id_to_image_pair(pair_id)
        first: int | None = rig_position.get(int(image_id_a))
        second: int | None = rig_position.get(int(image_id_b))
        if first is None or second is None or abs(first - second) < MIN_LOOP_RIG_GAP:
            continue
        key: tuple[int, int] = normalise_pair(first, second)
        revisits[key] = revisits.get(key, 0) + int(count)
    return revisits


def log_loop_closure(stage: WalkthroughStage, loops: LoopClosureStageResult, robocap_run: Path | None, console: str) -> None:
    """Stage 5: the rejection funnel, and a dataset that does have revisits.

    Galileo is a 0.66 m sweep, so every candidate the retrieval index proposes is
    rejected — the funnel in the notes is the whole result. To keep the stage from
    being an empty panel, a RoboCap run that *did* close loops is drawn beside it,
    in its own entity subtree and labelled as the other dataset.

    Args:
        stage: This stage's timeline slot and prose.
        loops: What `run_loop_closure_stage` returned.
        robocap_run: A RoboCap `.../cusfm` directory, or None to skip that half.
        console: What the stage printed — the funnel counters live here.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    rows: MarkdownRows = [
        ("loop edges kept on this dataset", str(len(loops.edges))),
        ("extra pairs this stage had to match", str(loops.num_pairs_matched)),
        (
            "which gate rejected what",
            (
                "read the funnel below. On Galileo every retrieved candidate falls to the **time** gate, not the "
                "metric one: the whole sweep lasts 0.93 s and a revisit has to be at least `min_time_gap_seconds` "
                "old, so no pair can qualify however well it matches"
            ),
        ),
    ]
    rows.extend(_log_robocap_loops(stage, robocap_run))
    log_notes(stage, rows, console)


def _log_robocap_loops(stage: WalkthroughStage, robocap_run: Path | None) -> MarkdownRows:
    """Draw a second dataset's revisit graph under `<stage>/robocap`.

    Args:
        stage: This stage's timeline slot.
        robocap_run: A RoboCap `.../cusfm` directory, or None.

    Returns:
        Rows describing the borrowed dataset, for the notes table.
    """
    if robocap_run is None:
        return [("second dataset", "not requested")]
    pose_graph_meta: Path = robocap_run / "pose_graph" / FRAMES_META_NAME
    database_path: Path = robocap_run / "database.db"
    if not pose_graph_meta.is_file() or not database_path.is_file():
        return [("second dataset", f"`{robocap_run}` is not on disk; nothing drawn")]

    frames_meta: FramesMeta = read_frames_meta(pose_graph_meta)
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    positions: Float64[ndarray, "n 3"] = np.asarray(
        [rig_frame.world_T_vehicle.translation for rig_frame in rig_frames], dtype=np.float64
    ).reshape(-1, 3)
    revisits: dict[tuple[int, int], int] = _verified_revisit_pairs(database_path, _rig_id_by_keyframe(frames_meta))
    strongest: list[tuple[int, int]] = sorted(revisits, key=lambda key: revisits[key], reverse=True)[:MAX_LOOP_ARCS]

    extent: float = scene_extent(positions)
    rr.log(
        f"{stage.entity_root}/robocap/nodes",
        rr.Points3D(positions, colors=INPUT_COLOR, radii=0.5 * NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/robocap/trajectory",
        rr.LineStrips3D(polyline_segments(positions), colors=SELECTED_COLOR, radii=EDGE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/robocap/loop_edges",
        rr.LineStrips3D(
            [arc_strip(positions[first], positions[second]) for first, second in strongest],
            colors=LOOP_COLOR,
            radii=0.6 * EDGE_RADIUS_FRACTION * extent,
        ),
    )
    return [
        ("second dataset", f"RoboCap, `{robocap_run}` — **not** the dataset the other stages show"),
        ("RoboCap rig frames", str(len(rig_frames))),
        ("RoboCap verified revisit pairs", f"{len(revisits)} rig-frame pairs at least {MIN_LOOP_RIG_GAP} frames apart"),
        ("RoboCap arcs drawn", f"{len(strongest)}, the strongest by inlier count"),
        (
            "why the arcs are pairs and not edges",
            (
                "the run's `summary.json` counts the gated loop *edges* it kept, but `run_pipeline` never writes the "
                "edge list to disk, so what is drawn is the verified revisit pair graph its database still holds"
            ),
        ),
    ]


def log_pose_graph(stage: WalkthroughStage, pose_graph: PoseGraphStageResult, console: str) -> None:
    """Stage 6: rig nodes and constraints before and after the solve.

    Args:
        stage: This stage's timeline slot and prose.
        pose_graph: What `run_pose_graph_stage` returned.
        console: What it printed — the Ceres termination and costs live here.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    nodes: list[RigNode] = pose_graph.nodes
    before_by_id: dict[int, Float64[ndarray, "3"]] = {
        node.rig_id: np.asarray(node.world_T_rig.translation, dtype=np.float64) for node in nodes
    }
    after_by_id: dict[int, Float64[ndarray, "3"]] = {
        rig_frame.synced_sample_id: np.asarray(rig_frame.world_T_vehicle.translation, dtype=np.float64)
        for rig_frame in pose_graph.frames_meta.rig_frames()
    }

    def edge_segments(by_id: Mapping[int, Float64[ndarray, "3"]], kind: str) -> Float64[ndarray, "n 2 3"]:
        """Two-point segments for every constraint of one kind."""
        drawn: list[PoseGraphEdge] = [
            edge for edge in pose_graph.edges if edge.kind == kind and edge.source in by_id and edge.target in by_id
        ]
        if not drawn:
            return np.zeros((0, 2, 3), dtype=np.float64)
        return np.asarray([[by_id[edge.source], by_id[edge.target]] for edge in drawn], dtype=np.float64).reshape(-1, 2, 3)

    extent: float = scene_extent(np.asarray(list(before_by_id.values()), dtype=np.float64).reshape(-1, 3))
    # The solved nodes are drawn smaller, not just greener: on a graph with no loop
    # constraint the solve moves nothing, and equal radii would hide the input nodes
    # entirely behind them — which reads as "the input was not logged".
    for label, by_id, colour, shrink in (
        ("before", before_by_id, INPUT_COLOR, 1.0),
        ("after", after_by_id, SOLVED_COLOR, SOLVED_NODE_SHRINK),
    ):
        node_positions: Float64[ndarray, "n 3"] = np.asarray([by_id[node.rig_id] for node in nodes if node.rig_id in by_id]).reshape(-1, 3)
        rr.log(
            f"{stage.entity_root}/{label}/nodes",
            rr.Points3D(node_positions, colors=colour, radii=shrink * NODE_RADIUS_FRACTION * extent),
        )
        rr.log(
            f"{stage.entity_root}/{label}/edges",
            rr.LineStrips3D(edge_segments(by_id, "consecutive"), colors=colour, radii=shrink * EDGE_RADIUS_FRACTION * extent),
        )
        rr.log(
            f"{stage.entity_root}/{label}/loop_edges",
            rr.LineStrips3D(edge_segments(by_id, "loop"), colors=LOOP_COLOR, radii=shrink * EDGE_RADIUS_FRACTION * extent),
        )

    num_loop: int = sum(1 for edge in pose_graph.edges if edge.kind == "loop")
    log_notes(
        stage,
        [
            ("rig nodes", str(len(nodes))),
            ("constraints", str(len(pose_graph.edges))),
            ("of which loop constraints", str(num_loop)),
            ("largest rig position change", f"{1000.0 * pose_graph.translation_change_m:.4f} mm"),
            (
                "before vs after",
                (
                    "identical to numerical precision when the graph holds only sequential constraints — with no loop "
                    "closing it, the measurements are exactly the input relative poses and the input trajectory is "
                    "already the optimum"
                ),
            ),
        ],
        console,
    )


def log_reconstruction(stage: WalkthroughStage, mapping: MappingResult, console: str) -> None:
    """Stage 7: the mapped cloud, plus every bundle-adjustment round's statistics.

    Args:
        stage: This stage's timeline slot and prose.
        mapping: The result `run_reconstruction_stage` returned, read before the
            extrinsic refinement may adjust the same reconstruction in place.
        console: What it printed — the per-pass triangulation counts live here.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    positions: Float64[ndarray, "n 3"] = np.asarray(
        [point.xyz for point in mapping.reconstruction.points3D.values()], dtype=np.float64
    ).reshape(-1, 3)
    rr.log(f"{stage.entity_root}/points", rr.Points3D(positions, colors=SELECTED_COLOR, radii=POINT_RADIUS))

    rounds: tuple[RoundStats, ...] = mapping.rounds
    errors: Float64[ndarray, "r"] = np.asarray([stats.mean_reprojection_error_px for stats in rounds], dtype=np.float64)
    observations: Int64[ndarray, "r"] = np.asarray([stats.num_observations for stats in rounds], dtype=np.int64)
    rr.log(f"{stage.entity_root}/round_chart/mean_reprojection_error_px", rr.BarChart(errors, color=STEREO_COLOR))
    rr.log(f"{stage.entity_root}/round_chart/num_observations", rr.BarChart(observations, color=SELECTED_COLOR))

    # The per-round series lives on its own timeline: `stage` holds one index for
    # the whole stage, so five rounds would collapse onto a single point there.
    rr.reset_time()
    for stats in rounds:
        rr.set_time(ROUND_TIMELINE, sequence=stats.round_index)
        rr.log(f"{stage.entity_root}/rounds/mean_reprojection_error_px", rr.Scalars(stats.mean_reprojection_error_px))
        rr.log(f"{stage.entity_root}/rounds/num_observations", rr.Scalars(float(stats.num_observations)))
        rr.log(f"{stage.entity_root}/rounds/max_pixel_error", rr.Scalars(stats.max_pixel_error))
        rr.log(f"{stage.entity_root}/rounds/seconds", rr.Scalars(stats.seconds))
    rr.reset_time()
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)

    table: list[str] = [
        "## Rounds",
        "",
        "| round | gate px | merged | completed | filtered | observations | reprojection px | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    table.extend(
        f"| {stats.round_index} | {stats.max_pixel_error:.2f} | {stats.num_merged} | {stats.num_completed} | "
        f"{stats.num_filtered} | {stats.num_observations} | {stats.mean_reprojection_error_px:.4f} | {stats.seconds:.2f} |"
        for stats in rounds
    )
    log_notes(
        stage,
        [
            ("registered images", str(mapping.num_registered_images)),
            ("images observing a point", str(mapping.num_images_with_observations)),
            ("points", f"{mapping.num_points3D:,}"),
            ("observations", f"{mapping.num_observations:,}"),
            ("mean reprojection error", f"{mapping.mean_reprojection_error_px:.4f} px"),
            ("mean track length", f"{mapping.mean_track_length:.3f}"),
            ("bundle-adjustment rounds", str(len(rounds))),
            ("triangulation / BA seconds", f"{mapping.triangulation_seconds:.2f} / {mapping.bundle_adjustment_seconds:.2f}"),
            (
                "per-round clouds",
                (
                    "not logged: `MappingResult` keeps one reconstruction and the per-round counters, so the cloud "
                    "can only be read after the last round. The `round` timeline carries the counters instead"
                ),
            ),
        ],
        console,
        "\n".join(table),
    )


def log_extrinsic_refinement(
    stage: WalkthroughStage,
    pose_graph_meta: FramesMeta,
    refinement: ExtrinsicRefinementStageResult,
    arrow_scale: float,
    console: str,
) -> None:
    """Stage 7b: how far the refinement moved each camera on the rig.

    The shifts are sub-millimetre, so the arrows are exaggerated by `arrow_scale`
    and the notes carry the real numbers.

    Args:
        stage: This stage's timeline slot and prose.
        pose_graph_meta: The collection the refinement started from.
        refinement: What `run_extrinsic_refinement_stage` returned.
        arrow_scale: Factor the drawn arrows are multiplied by.
        console: What the stage printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    rig_frames: tuple[RigFrame, ...] = pose_graph_meta.rig_frames()
    world_T_rig: pycolmap.Rigid3d = rig_frames[0].world_T_vehicle
    camera_ids: list[int] = sorted(pose_graph_meta.cameras)
    before: Float64[ndarray, "n 3"] = np.asarray(
        [(world_T_rig * pose_graph_meta.cameras[key].vehicle_T_cam).translation for key in camera_ids], dtype=np.float64
    ).reshape(-1, 3)
    after: Float64[ndarray, "n 3"] = np.asarray(
        [(world_T_rig * refinement.frames_meta.cameras[key].vehicle_T_cam).translation for key in camera_ids], dtype=np.float64
    ).reshape(-1, 3)
    labels: list[str] = [pose_graph_meta.cameras[key].sensor_name for key in camera_ids]

    # The rig, not the trajectory, sets the scale here: the cameras sit centimetres
    # apart on one vehicle, and a radius taken from the whole sweep would swallow them.
    extent: float = scene_extent(before)
    # The sensor names ride on the arrows, not on both: labelling the points too
    # prints every camera's name twice, on top of itself.
    rr.log(
        f"{stage.entity_root}/extrinsics/before",
        rr.Points3D(before, colors=INPUT_COLOR, radii=NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/extrinsics/after",
        rr.Points3D(after, colors=SOLVED_COLOR, radii=SOLVED_NODE_SHRINK * NODE_RADIUS_FRACTION * extent),
    )
    rr.log(
        f"{stage.entity_root}/extrinsics/shift",
        rr.Arrows3D(origins=before, vectors=arrow_scale * (after - before), colors=LOOP_COLOR, labels=labels),
    )

    table: list[str] = [
        "## Per-camera movement",
        "",
        "| camera | sensor | translation mm | rotation deg | rig origin |",
        "| --- | --- | --- | --- | --- |",
    ]
    table.extend(
        f"| {change.camera_params_id} | {change.sensor_name} | {change.translation_change_mm:.4f} | "
        f"{change.rotation_change_deg:.5f} | {'yes' if change.is_reference else ''} |"
        for change in refinement.changes
    )
    largest: ExtrinsicChange | None = max(refinement.changes, key=lambda change: change.translation_change_mm, default=None)
    log_notes(
        stage,
        [
            ("alternation rounds run", str(refinement.rounds_run)),
            ("cameras refined", str(len(refinement.changes))),
            ("largest translation change", "n/a" if largest is None else f"{largest.translation_change_mm:.4f} mm ({largest.sensor_name})"),
            ("arrow exaggeration", f"x{arrow_scale:g}"),
            ("points after refinement", f"{refinement.mapping.num_points3D:,}"),
            ("mean reprojection error", f"{refinement.mapping.mean_reprojection_error_px:.4f} px"),
        ],
        console,
        "\n".join(table),
    )


def log_run_as_rig(stage: WalkthroughStage, frames_meta: FramesMeta, index: int, *, label: str, colour: Rgb) -> int:
    """Draw one run as an exoego rig plus its trajectory, under this stage's subtree.

    `simplecv.log_rig_pose_stream` stamps only `frame_time`, so the rig's first
    pose is logged again on `stage`: without it the rig would have nowhere to
    stand while the walkthrough's own timeline is the active one.

    Args:
        stage: This stage's timeline slot.
        frames_meta: The run's collection, carrying its calibration and its poses.
        index: Rig entity index; 0 lands at `<stage>/rigs/rig_00`.
        label: Provenance string logged on the rig node.
        colour: Frustum and trajectory tint identifying the run.

    Returns:
        The number of rig frames drawn.
    """
    track: RigTrack = rig_track_from_frames_meta(frames_meta)
    if len(track) == 0:
        return 0
    rigs_path: str = f"{stage.entity_root.lstrip('/')}/rigs"
    rig: Rig = build_rig(frames_meta, index, track)
    log_rig_static(rig, world_path=rigs_path)
    log_rig_pose_stream(rig, timestamps_ns=1000 * track.timestamps_microseconds, world_path=rigs_path, timeline=TIMELINE)

    rig_path: str = f"/{rigs_path}/{entity_id('rig', index)}"
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    rr.log(rig_path, rr.Transform3D(translation=track.world_t_rig[0], mat3x3=track.world_R_rig[0]))
    rr.log(rig_path, rr.AnyValues(source=label, num_rig_frames=len(track)), static=True)
    for sensor in rig.calibration.cameras:
        rr.log(
            f"{rig_path}/{entity_id('cam', sensor.index)}/pinhole",
            rr.Pinhole.from_fields(image_plane_distance=IMAGE_PLANE_DISTANCE, color=(*colour, 255)),
            static=True,
        )
    rr.log(
        f"{stage.entity_root}/tracks/{entity_id('rig', index)}/trajectory",
        rr.LineStrips3D(
            polyline_segments(track.world_t_rig),
            colors=colour,
            radii=EDGE_RADIUS_FRACTION * scene_extent(track.world_t_rig),
        ),
    )
    return len(track)


def log_export(
    stage: WalkthroughStage,
    options: PipelineOptions,
    export: ExportStageResult,
    mapping: MappingResult,
    input_meta: FramesMeta,
    reference_run: Path | None,
    dataset: str,
    console: str,
) -> None:
    """Stage 8: the coloured cloud, the input and colsfm rigs, and the benchmark table.

    Args:
        stage: This stage's timeline slot and prose.
        options: The run's options, for the workspace and the input directory.
        export: What `run_export_stage` returned.
        mapping: The final mapping result, now coloured.
        input_meta: The dataset's own input collection, drawn as the reference rig.
        reference_run: A blob run in the cuSFM layout, or None to skip the comparison.
        dataset: Dataset label for the benchmark report heading.
        console: What the stage printed.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    positions: Float64[ndarray, "n 3"] = np.asarray(
        [point.xyz for point in mapping.reconstruction.points3D.values()], dtype=np.float64
    ).reshape(-1, 3)
    colours: Int64[ndarray, "n 3"] = np.asarray(
        [point.color for point in mapping.reconstruction.points3D.values()], dtype=np.int64
    ).reshape(-1, 3)
    rr.log(f"{stage.entity_root}/points", rr.Points3D(positions, colors=colours.astype(np.uint8), radii=POINT_RADIUS))

    log_run_as_rig(stage, input_meta, 0, label="input frames_meta.json", colour=INPUT_COLOR)
    log_run_as_rig(stage, export.optimised, 1, label="colsfm (this run)", colour=SELECTED_COLOR)

    rows: MarkdownRows = [
        ("points exported", f"{mapping.num_points3D:,}"),
        ("points that got a colour", f"{export.num_coloured:,}"),
        ("TUM trajectories written", str(len(export.pose_files))),
        ("workspace", f"`{options.output_dir}`"),
    ]
    rows.extend(_log_benchmark(stage, options, reference_run, dataset))
    log_notes(stage, rows, console)


def _log_benchmark(stage: WalkthroughStage, options: PipelineOptions, reference_run: Path | None, dataset: str) -> MarkdownRows:
    """Compare this run against a blob reference and log the report as the closing document.

    Args:
        stage: This stage's timeline slot.
        options: The run's options, for the workspace and the input directory.
        reference_run: A blob run in the cuSFM layout, or None.
        dataset: Dataset label for the report heading.

    Returns:
        Rows describing the comparison, for the notes table.
    """
    if reference_run is None or not (reference_run / "sparse").is_dir():
        rr.log(
            f"{stage.entity_root}/report",
            rr.TextDocument(
                f"# No reference run\n\nNothing to compare against: `{reference_run}` is not a cuSFM run directory.\n",
                media_type="text/markdown",
            ),
        )
        return [("benchmark", f"skipped; `{reference_run}` is not on disk")]

    blob: RunArtifacts = read_run(reference_run, "blob")
    colsfm: RunArtifacts = read_run(options.output_dir, "colsfm")
    comparison: Comparison = compare_runs(blob, colsfm, options.input_dir, dataset, AcceptanceBounds())
    rr.log(f"{stage.entity_root}/report", rr.TextDocument(render_markdown_report(comparison), media_type="text/markdown"))
    log_run_as_rig(stage, blob.frames_meta, 2, label=f"blob reference ({reference_run})", colour=REFERENCE_COLOR)
    return [
        ("benchmark reference", f"`{reference_run}`"),
        ("reference points", f"{comparison.run_a.reconstruction.num_points3D:,}"),
        ("this run's points", f"{comparison.run_b.reconstruction.num_points3D:,}"),
        ("acceptance bounds met", f"{sum(1 for check in comparison.acceptance if check.passed)}/{len(comparison.acceptance)}"),
    ]


