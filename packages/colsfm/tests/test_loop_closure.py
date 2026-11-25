"""Tests for `colsfm.loop_closure`, the Python replacement for cuSFM's loop-closure stage.

Seams under test (all public):

* `find_loop_edges` — retrieval, the candidate cap, verification, rig conversion, gating.
* `best_per_query_rig` — one loop per query rig frame, at `good_loop_threshold` inliers.
* `is_good_match` / `minimum_time_gap_seconds` / `session_duration_seconds` — the gates.
* `LoopClosureConfig.from_pose_graph` — the config profile's loop gates.
* `colsfm.rig_geometry.calibrated_cameras` — the cameras the estimators need.

The scene is a synthetic rig walking a square and coming back along its first side, so
there is one true revisit with a known ground-truth relative pose. Its images are 3-D
points projected into pinhole cameras with pixel noise and gross outliers, and its matcher
is a plain `match_fn`, so nothing here needs a GPU, a network or the matching worker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
import pytest
from google.protobuf.message import Message
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as hypothesis_arrays
from jaxtyping import Float32, Float64, Int
from loop_helpers import ProjectedPoints, matcher_from_point_indices, project_and_mask
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm import RESOURCES_DIR, loop_closure
from colsfm.config import CusfmConfig, PoseGraphConfig, read_config_directory
from colsfm.database import read_keypoints_batch
from colsfm.frames_meta import FramesMeta, parse_message, read_frames_meta
from colsfm.geometry import rigid3d_from_matrix
from colsfm.loop_closure import (
    FunnelCounters,
    LoopCandidate,
    LoopClosureConfig,
    LoopClosureDiagnostics,
    LoopClosureResult,
    LoopSearchPlan,
    MatchFunction,
    RetrievalHit,
    agrees_with_prior,
    best_per_query_rig,
    find_loop_edges,
    is_good_match,
    minimum_time_gap_seconds,
    plan_loop_search,
    prior_distance_bound,
    session_duration_seconds,
    verify_loop_plan,
)
from colsfm.loop_pose import DEFAULT_RIG_POSE_CONFIG, Keypoints, RigPoseConfig, RigPoseEstimate
from colsfm.matching import MatchingOptions
from colsfm.pose_graph import PoseGraphEdge, RigNode, gate_loop_edges, sequential_edges, solve_pose_graph
from colsfm.retrieval import (
    ALIKED_DESCRIPTOR_DIM,
    GOOD_SCORE_THRESHOLD,
    BruteForceConfig,
    RetrievalConfig,
    RetrievalIndex,
    build_retrieval_index,
)
from colsfm.rig_geometry import calibrated_cameras
from colsfm.run_config import PipelineOptions
from colsfm.schema import KEYFRAMES_METADATA_COLLECTION, load_schema
from colsfm.stages import LoopClosureStageResult, run_loop_closure_stage

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""Repo root, so data paths resolve regardless of the working directory."""

GALILEO_KEYFRAMES_META: Final[Path] = REPO_ROOT / "data/r2b_galileo/frames_meta.json"
"""The 226-keyframe blob run, used to check the temporal gate against real timestamps."""

LOOP_CLOSURE_CONFIG_DIR: Final[Path] = RESOURCES_DIR / "configs/loop-closure-fixed"
"""The one shipped profile whose loop gates are not proto3's zeros."""

RIG_PERIOD_US: Final[int] = 500_000
"""Half a second between rig frames, so a 48-frame sequence spans 23.5 s."""

IMAGE_WIDTH: Final[int] = 640
"""Synthetic image width in pixels."""

IMAGE_HEIGHT: Final[int] = 480
"""Synthetic image height in pixels."""

FOCAL_LENGTH_PX: Final[float] = 400.0
"""Synthetic focal length, giving a 77 deg horizontal field of view."""

KEYPOINTS_PER_IMAGE: Final[int] = 150
"""Observations kept per image, so every image carries the same descriptor count."""

VEHICLE_R_CAMERA: Final[Float64[ndarray, "3 3"]] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""Optical frame (x right, y down, z forward) expressed in the vehicle FLU frame."""


# --------------------------------------------------------------------------------------
# synthetic scene
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SquareRigScene:
    """A two-camera rig walking a square and retracing its first side."""

    frames_meta: FramesMeta
    """Metadata whose `camera_to_world` holds slightly noisy priors."""
    database_path: Path
    """COLMAP database holding one keypoint block per keyframe."""
    index: RetrievalIndex
    """Retrieval index over the same keyframes."""
    world_T_rig: dict[int, pycolmap.Rigid3d]
    """Exact ground-truth rig pose per `synced_sample_id`."""
    point_index_by_keypoint: dict[int, Int[ndarray, "n_keypoints"]]
    """Which 3-D point each keypoint of each keyframe observes."""
    revisit_offset: int
    """Rig frames between an outbound frame and the frame that retraces it."""
    n_rigs: int
    """How many rig frames the sequence has."""


def _square_trajectory(
    side_frames: int = 10, side_meters: float = 4.0, revisit_frames: int = 8, lateral_offset_m: float = 0.3
) -> list[pycolmap.Rigid3d]:
    """Walk a square, then retrace the first side 30 cm to the left.

    The lateral offset is what makes the revisit usable: two exactly coincident views give
    a degenerate essential matrix, so the pair needs a real baseline.

    Args:
        side_frames: Rig frames per side of the square.
        side_meters: Side length in metres.
        revisit_frames: Rig frames spent retracing the first side.
        lateral_offset_m: How far left of the outbound track the retrace runs.

    Returns:
        `world_T_vehicle` per rig frame, in time order.
    """
    step: float = side_meters / side_frames
    corners: list[tuple[float, float]] = [(0.0, 0.0), (side_meters, 0.0), (side_meters, side_meters), (0.0, side_meters)]
    directions: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    yaws_deg: list[float] = [0.0, 90.0, 180.0, 270.0]

    poses: list[pycolmap.Rigid3d] = []
    for side in range(4):
        for frame in range(side_frames):
            position: Float64[ndarray, "3"] = np.array(
                [corners[side][0] + directions[side][0] * step * frame, corners[side][1] + directions[side][1] * step * frame, 0.0]
            )
            poses.append(rigid3d_from_matrix(Rotation.from_euler("z", yaws_deg[side], degrees=True).as_matrix(), position))
    for frame in range(revisit_frames):
        position = np.array([step * frame, lateral_offset_m, 0.0])
        poses.append(rigid3d_from_matrix(Rotation.from_euler("z", 0.0, degrees=True).as_matrix(), position))
    return poses


def _camera_extrinsics(baseline_m: float = 0.15) -> dict[int, pycolmap.Rigid3d]:
    """Two forward-looking cameras straddling the vehicle centre.

    Args:
        baseline_m: Distance between the two optical centres, in metres.

    Returns:
        `vehicle_T_cam` per `camera_params_id`; 0 is the left camera, 1 the right.
    """
    return {
        0: rigid3d_from_matrix(VEHICLE_R_CAMERA, np.array([0.0, baseline_m / 2.0, 0.0])),
        1: rigid3d_from_matrix(VEHICLE_R_CAMERA, np.array([0.0, -baseline_m / 2.0, 0.0])),
    }


def _projection_matrix_message(message: Message) -> None:
    """Fill a `MatrixD` with the synthetic 3x4 rectified projection matrix.

    Args:
        message: The `calibration_parameters.projection_matrix` submessage to fill.
    """
    message.row_count = 3
    message.column_count = 4
    message.data.extend([FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH / 2.0, 0.0, 0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _write_rigid(message: Message, pose: pycolmap.Rigid3d) -> None:
    """Write a `Rigid3d` into a `RigidTransform3d` message as axis-angle in degrees.

    Args:
        message: The `RigidTransform3d` submessage to fill.
        pose: The transform to store.
    """
    rotation_vector: Float64[ndarray, "3"] = Rotation.from_quat(pose.rotation.quat).as_rotvec()
    angle_rad: float = float(np.linalg.norm(rotation_vector))
    axis: Float64[ndarray, "3"] = np.array([1.0, 0.0, 0.0]) if angle_rad == 0.0 else rotation_vector / angle_rad
    message.axis_angle.x, message.axis_angle.y, message.axis_angle.z = (float(value) for value in axis)
    message.axis_angle.angle_degrees = float(np.rad2deg(angle_rad))
    message.translation.x, message.translation.y, message.translation.z = (float(value) for value in pose.translation)


def _build_frames_meta(
    world_T_cam: dict[int, pycolmap.Rigid3d],
    camera_of_keyframe: dict[int, int],
    rig_of_keyframe: dict[int, int],
    timestamps_us: dict[int, int],
    extrinsics: dict[int, pycolmap.Rigid3d],
) -> FramesMeta:
    """Assemble a `KeyframesMetadataCollection` for the synthetic rig.

    Args:
        world_T_cam: Prior camera pose per keyframe id.
        camera_of_keyframe: `camera_params_id` per keyframe id.
        rig_of_keyframe: `synced_sample_id` per keyframe id.
        timestamps_us: Capture time in microseconds per keyframe id.
        extrinsics: `vehicle_T_cam` per `camera_params_id`.

    Returns:
        The parsed metadata.
    """
    collection_class = load_schema().message_class(KEYFRAMES_METADATA_COLLECTION)
    message: Message = collection_class()
    for keyframe_id in sorted(world_T_cam):
        entry = message.keyframes_metadata.add()
        entry.id = keyframe_id
        entry.camera_params_id = camera_of_keyframe[keyframe_id]
        entry.timestamp_microseconds = timestamps_us[keyframe_id]
        entry.image_name = f"cam{camera_of_keyframe[keyframe_id]}/{keyframe_id}.jpeg"
        entry.synced_sample_id = rig_of_keyframe[keyframe_id]
        _write_rigid(entry.camera_to_world, world_T_cam[keyframe_id])
    for camera_id, vehicle_T_cam in extrinsics.items():
        sensor = message.camera_params_id_to_camera_params[camera_id]
        sensor.sensor_meta_data.sensor_id = camera_id
        sensor.sensor_meta_data.sensor_name = f"cam{camera_id}"
        sensor.sensor_meta_data.frequency = 2.0
        _write_rigid(sensor.sensor_meta_data.sensor_to_vehicle_transform, vehicle_T_cam)
        sensor.calibration_parameters.image_width = IMAGE_WIDTH
        sensor.calibration_parameters.image_height = IMAGE_HEIGHT
        _projection_matrix_message(sensor.calibration_parameters.projection_matrix)
    stereo = message.stereo_pair.add()
    stereo.left_camera_param_id = 0
    stereo.right_camera_param_id = 1
    stereo.baseline_meters = 0.15
    return parse_message(message)


def make_square_rig_scene(tmp_path: Path, n_points: int = 12000, pixel_noise_px: float = 0.5, seed: int = 3) -> SquareRigScene:
    """Build the whole synthetic scene: poses, points, images, database and index.

    Priors written into `frames_meta.json` carry 2 mm / 0.1 deg of noise, so the local
    odometry the metric map is triangulated from is not the ground truth and the accuracy
    assertions are not tautological.

    Args:
        tmp_path: Directory the COLMAP database is written into.
        n_points: How many 3-D landmarks to scatter around the trajectory.
        pixel_noise_px: One-sigma keypoint noise in pixels.
        seed: Seed for the landmarks, the descriptors and the noise.

    Returns:
        The scene.

    Raises:
        ValueError: When a keyframe sees fewer landmarks than the scene needs.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    trajectory: list[pycolmap.Rigid3d] = _square_trajectory()
    extrinsics: dict[int, pycolmap.Rigid3d] = _camera_extrinsics()
    points_xyz: Float64[ndarray, "n_points 3"] = np.stack(
        [
            generator.uniform(-6.0, 12.0, n_points),
            generator.uniform(-6.0, 12.0, n_points),
            generator.uniform(-1.5, 2.5, n_points),
        ],
        axis=1,
    )
    point_descriptors: Float32[ndarray, "n_points 128"] = generator.standard_normal((n_points, ALIKED_DESCRIPTOR_DIM)).astype(np.float32)
    point_descriptors /= np.linalg.norm(point_descriptors, axis=1, keepdims=True)

    world_T_rig: dict[int, pycolmap.Rigid3d] = {}
    world_T_cam_prior: dict[int, pycolmap.Rigid3d] = {}
    camera_of_keyframe: dict[int, int] = {}
    rig_of_keyframe: dict[int, int] = {}
    timestamps_us: dict[int, int] = {}
    keypoints_by_keyframe: dict[int, Keypoints] = {}
    point_index_by_keypoint: dict[int, Int[ndarray, "n_keypoints"]] = {}
    descriptors_by_keyframe: dict[int, Float32[ndarray, "n_keypoints 128"]] = {}
    cameras: dict[int, pycolmap.Camera] = {
        camera_id: pycolmap.Camera(
            camera_id=camera_id,
            model="PINHOLE",
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            params=[FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0],
        )
        for camera_id in extrinsics
    }

    for rig_index, pose in enumerate(trajectory):
        rig_id: int = rig_index + 1
        world_T_rig[rig_id] = pose
        for camera_id, vehicle_T_cam in extrinsics.items():
            keyframe_id: int = rig_index * len(extrinsics) + camera_id + 1
            world_T_camera: pycolmap.Rigid3d = pose * vehicle_T_cam
            projected: ProjectedPoints = project_and_mask(
                world_T_camera.inverse(), cameras[camera_id], points_xyz, min_depth_m=0.5, max_depth_m=np.inf
            )
            # Nearest first: a resection constrains the translation component ALONG the
            # optical axis far more weakly than the two lateral ones, and near points are
            # what fix it.
            visible_indices: Int[ndarray, "n_visible"] = np.flatnonzero(projected.visible)
            chosen: Int[ndarray, "n_keypoints"] = visible_indices[np.argsort(projected.depth_m[visible_indices])][:KEYPOINTS_PER_IMAGE]
            if len(chosen) < KEYPOINTS_PER_IMAGE:
                raise ValueError(f"keyframe {keyframe_id} sees only {len(chosen)} points; scatter more landmarks")
            keypoints_by_keyframe[keyframe_id] = projected.pixels[chosen] + generator.standard_normal((len(chosen), 2)) * pixel_noise_px
            point_index_by_keypoint[keyframe_id] = chosen
            noisy_descriptors: Float32[ndarray, "n_keypoints 128"] = point_descriptors[chosen] + generator.standard_normal(
                (len(chosen), ALIKED_DESCRIPTOR_DIM)
            ).astype(np.float32) * np.float32(0.02)
            descriptors_by_keyframe[keyframe_id] = np.ascontiguousarray(noisy_descriptors / np.linalg.norm(noisy_descriptors, axis=1, keepdims=True))
            world_T_cam_prior[keyframe_id] = pycolmap.Rigid3d(
                pycolmap.Rotation3d(
                    (
                        Rotation.from_quat(world_T_camera.rotation.quat) * Rotation.from_rotvec(generator.standard_normal(3) * np.deg2rad(0.1))
                    ).as_quat()
                ),
                np.asarray(world_T_camera.translation, dtype=np.float64) + generator.standard_normal(3) * 0.002,
            )
            camera_of_keyframe[keyframe_id] = camera_id
            rig_of_keyframe[keyframe_id] = rig_id
            timestamps_us[keyframe_id] = rig_index * RIG_PERIOD_US

    database_path: Path = tmp_path / "database.db"
    frames_meta: FramesMeta = _build_frames_meta(world_T_cam_prior, camera_of_keyframe, rig_of_keyframe, timestamps_us, extrinsics)
    with pycolmap.Database.open(str(database_path)) as database:
        for camera in calibrated_cameras(frames_meta).values():
            database.write_camera(camera, use_camera_id=True)
        for keyframe_id, keypoints in keypoints_by_keyframe.items():
            database.write_image(
                pycolmap.Image(
                    image_id=keyframe_id, name=f"cam{camera_of_keyframe[keyframe_id]}/{keyframe_id}.png", camera_id=camera_of_keyframe[keyframe_id]
                ),
                use_image_id=True,
            )
            database.write_keypoints(keyframe_id, np.ascontiguousarray(keypoints, dtype=np.float32))

    index: RetrievalIndex = build_retrieval_index(
        descriptors_by_keyframe, RetrievalConfig(brute_force=BruteForceConfig(max_descriptors_per_image=KEYPOINTS_PER_IMAGE))
    )
    return SquareRigScene(
        frames_meta=frames_meta,
        database_path=database_path,
        index=index,
        world_T_rig=world_T_rig,
        point_index_by_keypoint=point_index_by_keypoint,
        revisit_offset=40,
        n_rigs=len(trajectory),
    )


def make_match_function(scene: SquareRigScene, outlier_ratio: float = 0.25, seed: int = 11) -> MatchFunction:
    """Build a `match_fn` from the scene's known correspondences, with gross outliers added.

    Args:
        scene: The synthetic scene.
        outlier_ratio: Fraction of extra, wrong matches relative to the true ones.
        seed: Seed for the outlier draw.

    Returns:
        A callable `match_fn(image_id_a, image_id_b) -> Int[ndarray, "n_matches 2"]`.
    """
    return matcher_from_point_indices(scene.point_index_by_keypoint, outlier_ratio, seed)


SEEDED_SERIAL_CONFIG: LoopClosureConfig = LoopClosureConfig(num_threads=1, rig_pose=replace(DEFAULT_RIG_POSE_CONFIG, ransac_random_seed=0))
"""One thread and a pinned RANSAC seed: two searches over the same input must return the same edges."""

SEEDED_CONFIG: LoopClosureConfig = LoopClosureConfig(rig_pose=replace(DEFAULT_RIG_POSE_CONFIG, ransac_random_seed=0))
"""The default settings with a pinned RANSAC seed, at the default thread count.

Unseeded (`ransac_random_seed=-1`), each generalized estimator draws a fresh seed per call, so a rig
pair near the inlier or direction gate falls either way between two searches of the same input and
their diagnostics (sometimes their edges) differ."""


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> SquareRigScene:
    """The synthetic square-walking rig, built once for the module."""
    return make_square_rig_scene(tmp_path_factory.mktemp("loop_closure"))


@pytest.fixture(scope="module")
def match_fn(scene: SquareRigScene) -> MatchFunction:
    """A matcher over the synthetic scene."""
    return make_match_function(scene)


@pytest.fixture(scope="module")
def loop_result(scene: SquareRigScene, match_fn: MatchFunction) -> LoopClosureResult:
    """The loop-closure run over the synthetic scene, with the stage switched on and the RANSAC seed pinned."""
    return find_loop_edges(scene.frames_meta, scene.database_path, scene.index, SEEDED_CONFIG, match_fn)


# --------------------------------------------------------------------------------------
# c. the search plan
#
# Written before the discovery run was deleted, and asserting against literals
# captured from it: the pipeline used to learn the pair list by *running* the whole
# search with a matcher that returned nothing, throwing the result away and running
# it again. `plan_loop_search` has to name exactly the same pairs, and
# `verify_loop_plan` has to produce exactly the same edges.
# --------------------------------------------------------------------------------------


def recording_matcher(recorded: set[tuple[int, int]]) -> MatchFunction:
    """A matcher that records the pairs it is asked for and returns nothing.

    Exactly the probe the pipeline used to run a whole search with. Kept here, in
    the tests, so the plan can be compared against what the search really asks for.

    Args:
        recorded: Set the requested pairs are added to, normalised low-id first.

    Returns:
        A `match_fn` that always answers with an empty match array.
    """

    def record(image_id_a: int, image_id_b: int) -> Int[ndarray, "0 2"]:
        """Record one pair and return no matches, so nothing verifies."""
        recorded.add((min(image_id_a, image_id_b), max(image_id_a, image_id_b)))
        return np.zeros((0, 2), dtype=np.int64)

    return record


def test_the_plan_names_exactly_the_pairs_the_search_asks_for(scene: SquareRigScene) -> None:
    """`plan_loop_search` replaces a whole discovery run and must name the same pairs.

    The old first pass ran retrieval, shortlisting, local-map triangulation and
    `_observations` against a matcher that returned nothing, purely to collect the
    pair list. The plan derives that list instead — local-map stereo and neighbour
    pairs, then the source-to-target observation pairs — and this compares the two.
    """
    config: LoopClosureConfig = LoopClosureConfig()
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, config)
    requested: set[tuple[int, int]] = set()
    find_loop_edges(scene.frames_meta, scene.database_path, scene.index, config, recording_matcher(requested))
    assert set(plan.image_pairs) == requested
    assert list(plan.image_pairs) == sorted(requested), "the plan is sorted, as the old `sorted(requested)` was"
    assert plan.image_pairs, "the synthetic scene has candidates, so it has pairs to match"


def test_the_plan_keeps_the_shortlisted_rig_pairs_and_the_retrieval_diagnostics(scene: SquareRigScene) -> None:
    """Retrieval runs once, so its funnel counters belong to the plan, not to a second pass."""
    config: LoopClosureConfig = LoopClosureConfig()
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, config)
    assert plan.rig_pairs, "the scene revisits its own track, so rig pairs are shortlisted"
    normalised: set[tuple[int, int]] = {(min(pair), max(pair)) for pair in plan.rig_pairs}
    assert len(normalised) == len(plan.rig_pairs), "the shortlist keeps one hit per unordered rig pair"
    assert [source for source, _ in plan.rig_pairs] == sorted(source for source, _ in plan.rig_pairs), (
        "the groups are walked in ascending source-rig order, as the serial pass walked them"
    )
    assert plan.queries > 0
    assert plan.candidates_retrieved >= len(plan.rig_pairs)


def test_planning_then_verifying_gives_the_same_edges_as_one_call(
    scene: SquareRigScene, match_fn: MatchFunction, loop_result: LoopClosureResult
) -> None:
    """The two-step form is the one-step form; `find_loop_edges` is now their composition.

    Both sides pin the RANSAC seed (`SEEDED_CONFIG`): unseeded, two searches of the same input differ
    in their counters 13 runs in 20, so this comparison would test the sampler, not the composition.
    """
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, SEEDED_CONFIG)
    stepwise: LoopClosureResult = verify_loop_plan(plan, match_fn)
    assert [(edge.source, edge.target) for edge in stepwise.edges] == [(edge.source, edge.target) for edge in loop_result.edges]
    assert stepwise.diagnostics == loop_result.diagnostics


# --------------------------------------------------------------------------------------
# c. the prior bound: a rig pair no accepted edge can come from is never matched
# --------------------------------------------------------------------------------------

UNBOUNDED_SCENE_RIG_PAIRS: Final[int] = 46
"""Rig pairs the square scene shortlists under `LoopClosureConfig()`, captured before the prior bound existed."""

UNBOUNDED_SCENE_IMAGE_PAIRS: Final[int] = 248
"""Image pairs that shortlist needs, captured at the same commit."""


def _prior_distance_m(plan: LoopSearchPlan, rig_pair: tuple[int, int]) -> float:
    """The distance between two rig frames in the prior the plan was made from.

    Args:
        plan: A plan.
        rig_pair: `(source rig id, target rig id)`.

    Returns:
        `||t||` of the prior `source_T_target`, metres.
    """
    source_xyz: Float64[ndarray, "3"] = np.asarray(plan.rig_index.world_T_rig[rig_pair[0]].translation, dtype=np.float64)
    target_xyz: Float64[ndarray, "3"] = np.asarray(plan.rig_index.world_T_rig[rig_pair[1]].translation, dtype=np.float64)
    return float(np.linalg.norm(source_xyz - target_xyz))


def _observation_pairs(plan: LoopSearchPlan, rig_pair: tuple[int, int]) -> set[tuple[int, int]]:
    """The source-to-target image pairs one rig pair's measurement asks for (same camera of the target rig).

    Args:
        plan: A plan.
        rig_pair: `(source rig id, target rig id)`.

    Returns:
        The image pairs, low id first.
    """
    pairs: set[tuple[int, int]] = set()
    for camera_id in plan.geometry.camera_ids:
        anchor: int | None = plan.rig_index.keyframe(rig_pair[0], camera_id)
        observed: int | None = plan.rig_index.keyframe(rig_pair[1], camera_id)
        if anchor is not None and observed is not None:
            pairs.add((min(anchor, observed), max(anchor, observed)))
    return pairs


def _prior_path_m(plan: LoopSearchPlan, rig_pair: tuple[int, int]) -> float:
    """The path the prior travels between two rig frames, summed over the rig frames in between.

    Args:
        plan: A plan.
        rig_pair: `(source rig id, target rig id)`.

    Returns:
        Metres along the prior trajectory.
    """
    sequence: tuple[int, ...] = plan.rig_index.sequence
    low, high = sorted((sequence.index(rig_pair[0]), sequence.index(rig_pair[1])))
    return sum(_prior_distance_m(plan, (sequence[step], sequence[step + 1])) for step in range(low, high))


def test_the_prior_bound_is_the_edge_gate_plus_the_tighter_of_the_prior_gate_and_the_drift_allowance() -> None:
    """Edge gate T plus min(P, 3.5 m + 2 % of the prior path); a closed edge gate (KITTI's T = 0) reaches 30 m instead of T."""
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=3.0, max_prior_translation_m=1.5), path_m=0.0) == 4.5
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=3.0, max_prior_translation_m=1.5), path_m=1000.0) == 4.5
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=3.0), path_m=0.0) == pytest.approx(6.5)
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=1.0), path_m=100.0) == pytest.approx(6.5)
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=3.0, max_prior_rotation_deg=10.0), path_m=50.0) == pytest.approx(7.5)
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=0.0), path_m=600.0) == pytest.approx(45.5)
    # A zero rotation gate rejects every edge as well (`gate_loop_edges` needs both), so the reach is the closed one.
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=1.0, max_rotation_deg=0.0), path_m=600.0) == pytest.approx(45.5)
    assert prior_distance_bound(LoopClosureConfig(max_translation_m=0.0, max_prior_translation_m=1.5), path_m=600.0) == pytest.approx(45.5), (
        "with closed edge gates no edge is accepted, so the prior gate bounds nothing: the drift allowance applies"
    )


@settings(max_examples=300, deadline=None)
@given(
    measured_xyz=hypothesis_arrays(np.float64, 3, elements=st.floats(-4.0, 4.0)),
    measured_rotvec=hypothesis_arrays(np.float64, 3, elements=st.floats(-3.0, 3.0)),
    error_xyz=hypothesis_arrays(np.float64, 3, elements=st.floats(-2.0, 2.0)),
    error_rotvec=hypothesis_arrays(np.float64, 3, elements=st.floats(-0.3, 0.3)),
)
def test_no_rig_pair_beyond_the_bound_can_pass_both_gates(
    measured_xyz: Float64[ndarray, "3"],
    measured_rotvec: Float64[ndarray, "3"],
    error_xyz: Float64[ndarray, "3"],
    error_rotvec: Float64[ndarray, "3"],
) -> None:
    """Triangle inequality: `|t_meas| <= T` and `|t(prior^-1 meas)| <= P` imply `|t_prior| <= T + P`.

    The prior is built as `measured * error^-1`, so `agrees_with_prior` sees exactly `error`; every case that passes
    both the prior gate and the edge gate must have a prior distance inside the bound.
    """
    config: LoopClosureConfig = LoopClosureConfig(max_translation_m=3.0, max_rotation_deg=180.0, max_prior_translation_m=1.5)
    measured: pycolmap.Rigid3d = rigid3d_from_matrix(Rotation.from_rotvec(measured_rotvec).as_matrix(), measured_xyz)
    error: pycolmap.Rigid3d = rigid3d_from_matrix(Rotation.from_rotvec(error_rotvec).as_matrix(), error_xyz)
    prior: pycolmap.Rigid3d = measured * error.inverse()
    template: LoopCandidate = _candidate(1, 2, 200)
    candidate: LoopCandidate = replace(template, estimate=replace(template.estimate, source_T_target=measured))
    edge: PoseGraphEdge = PoseGraphEdge(source=1, target=2, source_T_target=measured, information=np.eye(6), kind="loop")
    passes_both: bool = agrees_with_prior(candidate, {1: pycolmap.Rigid3d(), 2: prior}, config) and bool(
        gate_loop_edges([edge], config.max_translation_m, config.max_rotation_deg)
    )
    if passes_both:
        assert float(np.linalg.norm(prior.translation)) <= prior_distance_bound(config, path_m=0.0) + 1e-9


def test_a_rig_pair_the_prior_puts_beyond_the_bound_is_never_matched(scene: SquareRigScene) -> None:
    """With T = P = 0.5 m the bound is 1 m: the scene's shortlisted pairs 1.2-3.6 m apart in the prior leave the plan.

    Their observation pairs are not matched, and `rejected_by_prior_bound` counts them.
    """
    unbounded: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    bounded: LoopSearchPlan = plan_loop_search(
        scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(max_translation_m=0.5, max_prior_translation_m=0.5)
    )
    far: list[tuple[int, int]] = [pair for pair in unbounded.rig_pairs if _prior_distance_m(unbounded, pair) > 1.0]
    assert len(far) >= 10, "the scene must shortlist pairs beyond 1 m for the test to mean anything"
    assert not set(far) & set(bounded.rig_pairs)
    assert all(_prior_distance_m(bounded, pair) <= 1.0 for pair in bounded.rig_pairs)
    assert bounded.counters.rejected_by_prior_bound == len(far)
    for pair in far:
        assert not _observation_pairs(bounded, pair) & set(bounded.image_pairs), f"{pair} is beyond the bound but its pairs are planned"
    assert {(3, 43), (4, 44)} <= {(min(pair), max(pair)) for pair in bounded.rig_pairs}, "the true revisits 0.3 m apart stay"


def test_the_default_bound_keeps_every_rig_pair_of_the_square_scene(scene: SquareRigScene) -> None:
    """The scene's shortlisted pairs are at most 3.6 m apart in a prior that hardly drifts: no default bound drops one.

    The same holds for closed edge gates (KITTI's T = 0, whose loop pairs feed the mapper) and with only a rotation prior gate.
    """
    for config in (
        LoopClosureConfig(),
        LoopClosureConfig(max_prior_rotation_deg=10.0),
        LoopClosureConfig(max_translation_m=0.0),
        LoopClosureConfig(max_translation_m=0.0, max_prior_translation_m=1.5),
    ):
        plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, config)
        assert len(plan.rig_pairs) == UNBOUNDED_SCENE_RIG_PAIRS
        assert len(plan.image_pairs) == UNBOUNDED_SCENE_IMAGE_PAIRS
        assert plan.counters.rejected_by_prior_bound == 0


def test_without_a_prior_gate_a_rig_pair_beyond_the_drift_allowance_is_never_matched(scene: SquareRigScene) -> None:
    """No prior gate, T = 0.5 m and a flat 0.5 m allowance: the bound is 1 m, as with the prior gate of the B test."""
    default: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    bounded: LoopSearchPlan = plan_loop_search(
        scene.frames_meta,
        scene.database_path,
        scene.index,
        LoopClosureConfig(max_translation_m=0.5, prior_drift_floor_m=0.5, prior_drift_per_metre=0.0),
    )
    assert list(bounded.rig_pairs) == [pair for pair in default.rig_pairs if _prior_distance_m(default, pair) <= 1.0]
    assert bounded.counters.rejected_by_prior_bound == len(default.rig_pairs) - len(bounded.rig_pairs) >= 10


def test_the_drift_allowance_grows_with_the_prior_path_between_the_rig_frames(scene: SquareRigScene) -> None:
    """T = 0.5 m, no floor, 10 % of the path: a pair is kept when its prior distance is at most 0.5 m + 0.1 x its prior path."""
    default: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    bounded: LoopSearchPlan = plan_loop_search(
        scene.frames_meta,
        scene.database_path,
        scene.index,
        LoopClosureConfig(max_translation_m=0.5, prior_drift_floor_m=0.0, prior_drift_per_metre=0.1),
    )
    expected: list[tuple[int, int]] = [
        pair for pair in default.rig_pairs if _prior_distance_m(default, pair) <= 0.5 + 0.1 * _prior_path_m(default, pair)
    ]
    assert list(bounded.rig_pairs) == expected
    assert any(_prior_distance_m(default, pair) > 1.5 for pair in expected), "a long path must keep a pair a flat 1.5 m bound would drop"
    assert len(expected) < len(default.rig_pairs), "and some pair must still be beyond the allowance"


def test_the_bound_drops_after_the_cap(scene: SquareRigScene) -> None:
    """Pure subtraction: no rig pair takes a slot the bound freed, and the measurement order stays."""
    unbounded: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    bounded: LoopSearchPlan = plan_loop_search(
        scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(max_translation_m=0.5, max_prior_translation_m=0.5)
    )
    assert list(bounded.rig_pairs) == [pair for pair in unbounded.rig_pairs if _prior_distance_m(unbounded, pair) <= 1.0]
    kept: set[tuple[int, int]] = {(min(pair), max(pair)) for pair in bounded.rig_pairs}
    assert set(bounded.queried_by) == kept


# --------------------------------------------------------------------------------------
# c. loop closure end to end
# --------------------------------------------------------------------------------------


def test_the_stage_is_on_by_default_and_disabling_it_is_inert(scene: SquareRigScene, isaac_config: CusfmConfig) -> None:
    """`--loop-closure` is on, as in the blob, and the stage's `enabled=False` runs nothing.

    The default flipped on 2026-09-05 (NOTES.md decision 10, revised) so every run pays for
    the same stage the blob runs; the disabled path must stay a clean no-op for ablations.
    There is exactly one switch — the stage argument — so this is the only place that can
    assert the off path: `LoopClosureConfig` no longer carries a second copy of it, and
    nothing below `run_loop_closure_stage` has a disabled representation to keep consistent.
    """
    assert PipelineOptions(input_dir=Path(), output_dir=Path()).loop_closure is True
    result: LoopClosureStageResult = run_loop_closure_stage(
        scene.frames_meta, scene.database_path, isaac_config.pose_graph, MatchingOptions(), enabled=False
    )
    assert result.edges == []
    assert result.num_pairs_matched == 0
    assert result.diagnostics is None


def test_the_score_gate_is_one_constant_and_can_be_overridden(scene: SquareRigScene, match_fn: MatchFunction, loop_result: LoopClosureResult) -> None:
    """`good_score_threshold` is None by default and resolves to `GOOD_SCORE_THRESHOLD`.

    The two `colsfm.retrieval` backends score different quantities, but both calibrations
    landed on the same 0.1, so there is one constant rather than a per-backend table. The
    diagnostics report the value that was actually applied, exactly as they do for the
    temporal gap.
    """
    assert LoopClosureConfig().good_score_threshold is None
    assert scene.index.backend == "brute_force"
    assert loop_result.diagnostics.good_score_threshold == GOOD_SCORE_THRESHOLD

    strict: LoopClosureResult = find_loop_edges(
        scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(good_score_threshold=0.99), match_fn
    )
    assert strict.diagnostics.good_score_threshold == 0.99
    assert strict.diagnostics.counters.rejected_by_score > 0
    assert strict.edges == []


def test_the_funnel_counters_reconcile(loop_result: LoopClosureResult) -> None:
    """Every retrieval hit the index examined is accounted for by exactly one gate.

    The counters used to come from two different queries — one gated, one not — so
    `candidates_retrieved` and `rejected_by_time` described different candidate sets and the
    printed funnel did not compose. They now come from the same ranking walk.
    """
    diagnostics: LoopClosureDiagnostics = loop_result.diagnostics
    counters: FunnelCounters = diagnostics.counters
    assert counters.queries > 0
    assert counters.candidates_retrieved > 0
    assert counters.rejected_by_time > 0
    survivors: int = counters.candidates_retrieved - counters.rejected_by_time - counters.rejected_by_score - counters.rejected_by_same_rig
    assert survivors >= counters.after_candidates >= counters.after_deduplication
    measured: int = (
        counters.rejected_no_matches
        + counters.rejected_by_geometry
        + counters.rejected_by_is_good
        + counters.rejected_by_direction
        + counters.verified
    )
    assert measured == counters.after_deduplication
    assert counters.verified == len(loop_result.candidates)
    assert diagnostics.edges == len(loop_result.edges) <= counters.verified


def test_loop_edges_recover_the_ground_truth_relative_rig_pose(scene: SquareRigScene, loop_result: LoopClosureResult) -> None:
    """Every emitted edge is within 1 cm and 0.5 deg of the true rig-to-rig transform.

    Rotation, translation direction and translation *magnitude* all come from the images
    through the local metric map and the generalized resection; the priors only place the
    source rig's own neighbours, and they carry 2 mm of noise. So this checks the
    triangulation, the resection and the rig conventions together.
    """
    assert loop_result.edges, f"no loop edges: {loop_result.diagnostics}"
    for edge in loop_result.edges:
        assert edge.kind == "loop"
        expected: pycolmap.Rigid3d = scene.world_T_rig[edge.source].inverse() * scene.world_T_rig[edge.target]
        error: pycolmap.Rigid3d = expected.inverse() * edge.source_T_target
        translation_error_m: float = float(np.linalg.norm(error.translation))
        rotation_error_deg: float = float(np.rad2deg(np.linalg.norm(Rotation.from_quat(error.rotation.quat).as_rotvec())))
        assert translation_error_m < 0.01, f"edge {edge.source}->{edge.target} off by {translation_error_m * 1e3:.1f} mm"
        assert rotation_error_deg < 0.5, f"edge {edge.source}->{edge.target} off by {rotation_error_deg:.3f} deg"


def test_loop_edges_link_the_outbound_track_to_its_retrace(scene: SquareRigScene, loop_result: LoopClosureResult) -> None:
    """The edges connect the first side of the square to the frames that retrace it."""
    linked: set[tuple[int, int]] = {(min(edge.source, edge.target), max(edge.source, edge.target)) for edge in loop_result.edges}
    assert linked
    for low, high in linked:
        assert high - low >= scene.revisit_offset - 4
    print(f"loop closure diagnostics: {loop_result.diagnostics}")


@pytest.fixture(scope="module")
def clean_match_fn(scene: SquareRigScene) -> MatchFunction:
    """A matcher with no gross outliers, for the comparisons between two whole runs.

    Both generalized estimators are RANSAC and pycolmap's `set_random_seed` does not reach
    them, so a pair sitting on the inlier gate falls either way between two runs of the same
    code. Dropping the 25 % outliers narrows that band — what is left of it comes from the
    keypoint noise and the landmark depth error, which RANSAC still has to sample against —
    so the runs being compared differ over as few pairs as this scene allows.
    """
    return make_match_function(scene, outlier_ratio=0.0)


@pytest.fixture(scope="module")
def serial_result(scene: SquareRigScene, clean_match_fn: MatchFunction) -> LoopClosureResult:
    """The reference run: one worker thread, the outlier-free matcher."""
    return find_loop_edges(scene.frames_meta, scene.database_path, scene.index, SEEDED_SERIAL_CONFIG, clean_match_fn)


def _poses_by_rig_pair(result: LoopClosureResult) -> dict[tuple[int, int], pycolmap.Rigid3d]:
    """Index a run's edges by the unordered rig pair they connect.

    Args:
        result: A finished loop-closure run.

    Returns:
        The measured `source_T_target` per unordered rig pair.
    """
    return {(min(edge.source, edge.target), max(edge.source, edge.target)): edge.source_T_target for edge in result.edges}


@pytest.mark.parametrize("bearing_budget", [None, 8], ids=["default-cache", "one-image-per-worker"])
def test_the_thread_count_does_not_reach_the_result(
    scene: SquareRigScene,
    clean_match_fn: MatchFunction,
    serial_result: LoopClosureResult,
    bearing_budget: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight workers measure the same pairs in the same order and get the same poses.

    With a budget of 8 images over 8 workers every bearing lookup evicts, while the workers
    ask for overlapping images at once: one estimator shared between them failed exactly
    there (a hit's reorder after another thread's eviction), so each worker owns its own.

    A threading bug in this stage would be one of two things: a worker reading another
    group's local map, which puts metres of error into a pose, or the results coming back
    out of submission order, which reorders the edge list the caller solves. Both are
    checked here — the poses on every shared rig pair agree to a millimetre against edges a
    metre long, and both runs' candidates come back in `(source rig, rig pair)` order.

    Both runs pin the RANSAC seed (`RigPoseConfig.ransac_random_seed`): unseeded, COLMAP main's
    generalized samplers flip up to three of eight edges across the inlier gate between two runs
    of identical code, threaded or serial, which would hide a threading bug in noise.
    """
    assert LoopClosureConfig().num_threads >= 1
    if bearing_budget is not None:
        monkeypatch.setattr(loop_closure, "BEARING_CACHE_IMAGES", bearing_budget)
    threaded: LoopClosureResult = find_loop_edges(
        scene.frames_meta, scene.database_path, scene.index, replace(SEEDED_SERIAL_CONFIG, num_threads=8), clean_match_fn
    )
    assert threaded.diagnostics.counters.after_deduplication == serial_result.diagnostics.counters.after_deduplication
    for result in (serial_result, threaded):
        keys: list[tuple[int, tuple[int, int]]] = [
            (
                candidate.hit.source_rig_id,
                (min(candidate.hit.source_rig_id, candidate.hit.target_rig_id), max(candidate.hit.source_rig_id, candidate.hit.target_rig_id)),
            )
            for candidate in result.candidates
        ]
        assert keys == sorted(keys), "the measured pairs must come back in the order the shortlist was walked in"

    serial_poses: dict[tuple[int, int], pycolmap.Rigid3d] = _poses_by_rig_pair(serial_result)
    threaded_poses: dict[tuple[int, int], pycolmap.Rigid3d] = _poses_by_rig_pair(threaded)
    shared: set[tuple[int, int]] = set(serial_poses) & set(threaded_poses)
    assert len(shared) >= 0.8 * max(len(serial_poses), len(threaded_poses)), f"{len(shared)} shared of {len(serial_poses)}/{len(threaded_poses)}"
    assert [pair for pair in serial_poses if pair in shared] == [pair for pair in threaded_poses if pair in shared]
    for pair in shared:
        np.testing.assert_allclose(threaded_poses[pair].translation, serial_poses[pair].translation, atol=1e-3)
        np.testing.assert_allclose(threaded_poses[pair].rotation.matrix(), serial_poses[pair].rotation.matrix(), atol=1e-3)


def test_reading_the_keypoints_once_and_passing_them_in_replaces_the_database_read(
    scene: SquareRigScene, clean_match_fn: MatchFunction, serial_result: LoopClosureResult
) -> None:
    """`find_loop_edges(keypoints=...)` skips the database read and measures the same pairs.

    The pipeline runs the stage twice — once to learn the pair list, once for real — and the
    read is 148 MB on RoboCap, so it must be possible to do it once. That the pre-read map
    really is all the stage needs from the database is checked by pointing the database
    argument at a path that does not exist: without the keypoints, the same call raises.
    """
    keypoints: dict[int, Keypoints] = read_keypoints_batch(scene.database_path, scene.index.image_ids, dtype=np.float64)
    assert len(keypoints) == len(scene.index.image_ids)
    absent: Path = Path("no/such/database.db")
    reused: LoopClosureResult = find_loop_edges(scene.frames_meta, absent, scene.index, SEEDED_SERIAL_CONFIG, clean_match_fn, keypoints=keypoints)
    assert reused.diagnostics.counters.after_deduplication == serial_result.diagnostics.counters.after_deduplication
    shared: set[tuple[int, int]] = set(_poses_by_rig_pair(reused)) & set(_poses_by_rig_pair(serial_result))
    assert len(shared) >= 0.8 * len(serial_result.edges)
    with pytest.raises(FileNotFoundError):
        find_loop_edges(scene.frames_meta, absent, scene.index, LoopClosureConfig(), clean_match_fn)


def test_loop_edges_reduce_trajectory_error_when_the_odometry_drifts(scene: SquareRigScene, loop_result: LoopClosureResult) -> None:
    """A pose graph of drifting sequential edges plus these loop edges lands closer to truth.

    The sequential measurements carry a small per-step error that integrates into a
    trajectory drift, exactly as odometry does; the loop edges are the ones just verified
    against ground truth. Solving with both must beat solving with the sequential edges
    alone, which reproduce the drift exactly.
    """
    generator: np.random.Generator = np.random.default_rng(5)
    rig_ids: list[int] = sorted(scene.world_T_rig)
    drifted: dict[int, pycolmap.Rigid3d] = {rig_ids[0]: scene.world_T_rig[rig_ids[0]]}
    for previous, current in zip(rig_ids, rig_ids[1:], strict=False):
        true_step: pycolmap.Rigid3d = scene.world_T_rig[previous].inverse() * scene.world_T_rig[current]
        perturbation: pycolmap.Rigid3d = pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_rotvec(generator.standard_normal(3) * np.deg2rad(0.25)).as_quat()),
            generator.standard_normal(3) * 0.004,
        )
        drifted[current] = drifted[previous] * (true_step * perturbation)

    nodes: list[RigNode] = [RigNode(rig_id=rig_id, world_T_rig=drifted[rig_id], timestamp_us=(rig_id - 1) * RIG_PERIOD_US) for rig_id in rig_ids]
    odometry_edges: list[PoseGraphEdge] = sequential_edges(nodes, connected_keyframe_num=1)

    def trajectory_error(poses: dict[int, pycolmap.Rigid3d]) -> float:
        """Mean absolute position error against ground truth, in metres."""
        return float(np.mean([np.linalg.norm(poses[rig_id].translation - scene.world_T_rig[rig_id].translation) for rig_id in rig_ids]))

    without_loops: float = trajectory_error(solve_pose_graph(nodes, odometry_edges).world_T_rig)
    with_loops: float = trajectory_error(solve_pose_graph(nodes, [*odometry_edges, *loop_result.edges]).world_T_rig)
    print(f"pose graph mean position error: {without_loops * 1e3:.1f} mm without loops, {with_loops * 1e3:.1f} mm with them")
    assert without_loops > 0.02, "the injected drift is too small for the test to mean anything"
    assert with_loops < 0.6 * without_loops


def test_stored_weights_scale_the_information_matrix_by_the_inlier_count(scene: SquareRigScene, match_fn: MatchFunction) -> None:
    """`use_stored_weights` reproduces the blob's `inliers * loop_residual_weight * I6`.

    It is off by default: the blob's own solver discards the stored matrix, and measured
    against its archived Galileo result the identity lands 0.28 mm away against 2.17 mm
    (`docs/spec/pose_graph_main.md` §8 item 6).

    The check reads the inlier count off the candidate rather than comparing two runs
    edge for edge: both generalized estimators are RANSAC and pycolmap's own
    `set_random_seed` does not reach them, so a pair sitting on the inlier gate falls either
    way between runs. The poses themselves agree to 0.1 mm; only the gate decision moves.
    """
    config: LoopClosureConfig = LoopClosureConfig(use_stored_weights=True)
    identity_run: LoopClosureResult = find_loop_edges(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig(), match_fn)
    weighted_run: LoopClosureResult = find_loop_edges(scene.frames_meta, scene.database_path, scene.index, config, match_fn)
    assert identity_run.edges
    assert weighted_run.edges
    for edge in identity_run.edges:
        np.testing.assert_array_equal(edge.information, np.eye(6))

    inliers_by_pair: dict[tuple[int, int], int] = {
        (candidate.hit.source_rig_id, candidate.hit.target_rig_id): candidate.estimate.num_inliers for candidate in weighted_run.candidates
    }
    for edge in weighted_run.edges:
        expected: float = inliers_by_pair[(edge.source, edge.target)] * config.loop_residual_weight
        np.testing.assert_allclose(edge.information, np.eye(6) * expected)


def test_a_candidate_keeps_its_hit_and_its_measurement_whole(loop_result: LoopClosureResult) -> None:
    """`LoopCandidate` composes the retrieval hit with the metric estimate.

    Neither half is re-flattened into the other, so a reader can tell at a glance which
    numbers retrieval decided and which the estimator measured.
    """
    assert loop_result.candidates
    candidate: LoopCandidate = loop_result.candidates[0]
    assert isinstance(candidate.hit, RetrievalHit)
    assert candidate.hit.source_rig_id != candidate.hit.target_rig_id
    assert 0.0 <= candidate.hit.score <= 1.0
    assert candidate.estimate.num_inliers <= candidate.estimate.num_observations
    assert candidate.estimate.num_landmarks > 0


# --------------------------------------------------------------------------------------
# c. one loop per query rig frame
# --------------------------------------------------------------------------------------


def _candidate(source_rig_id: int, target_rig_id: int, num_inliers: int) -> LoopCandidate:
    """A verified candidate between two rig frames with a chosen inlier count.

    Args:
        source_rig_id: The rig frame whose query found it.
        target_rig_id: The rig frame it found.
        num_inliers: Its measured inliers.

    Returns:
        The candidate.
    """
    hit: RetrievalHit = RetrievalHit(
        query_keyframe_id=source_rig_id, candidate_keyframe_id=target_rig_id, source_rig_id=source_rig_id, target_rig_id=target_rig_id, score=0.5, delta_seconds=30.0
    )
    estimate: RigPoseEstimate = RigPoseEstimate(
        source_T_target=pycolmap.Rigid3d(),
        num_inliers=num_inliers,
        num_observations=2 * num_inliers,
        num_landmarks=num_inliers,
        num_cameras=2,
        direction_disagreement_deg=float("nan"),
    )
    return LoopCandidate(hit=hit, estimate=estimate)


def test_each_query_rig_frame_keeps_only_its_best_measured_pair() -> None:
    """Rig frame 1 kept 10 and 11 as candidates: only 11, the one with more inliers, becomes an edge."""
    weaker, stronger = _candidate(1, 10, 200), _candidate(1, 11, 300)
    counters: FunnelCounters = FunnelCounters()
    queried_by: dict[tuple[int, int], frozenset[int]] = {(1, 10): frozenset({1}), (1, 11): frozenset({1})}
    assert best_per_query_rig([weaker, stronger], queried_by, 150, counters) == [stronger]
    assert counters.not_best == 1 and counters.below_good_loop == 0


def test_a_best_pair_below_the_good_loop_threshold_is_dropped() -> None:
    """The best of rig frame 1 has 120 inliers, under cuSFM's 150: no edge, and the counter says why."""
    counters: FunnelCounters = FunnelCounters()
    assert best_per_query_rig([_candidate(1, 10, 120)], {(1, 10): frozenset({1})}, 150, counters) == []
    assert counters.below_good_loop == 1


def test_a_pair_both_rig_frames_kept_as_candidates_is_one_edge() -> None:
    """1 and 10 found each other; the pair is the best of both and is kept once."""
    counters: FunnelCounters = FunnelCounters()
    pair: LoopCandidate = _candidate(1, 10, 400)
    assert best_per_query_rig([pair], {(1, 10): frozenset({1, 10})}, 150, counters) == [pair]
    assert counters.not_best == 0


# --------------------------------------------------------------------------------------
# d. the gates
# --------------------------------------------------------------------------------------


def test_is_good_accepts_a_strong_pair_and_rejects_a_weak_one() -> None:
    """`inliers > 30 and inliers / matches > 0.25`, both strict (spec §6.6, §7.1 item 6)."""
    config: LoopClosureConfig = LoopClosureConfig()
    assert is_good_match(num_inliers=60, num_matches=200, config=config) is True
    assert is_good_match(num_inliers=30, num_matches=100, config=config) is False
    assert is_good_match(num_inliers=50, num_matches=200, config=config) is False
    assert is_good_match(num_inliers=0, num_matches=0, config=config) is False


def test_the_inlier_floor_is_declared_once() -> None:
    """`LoopClosureConfig.min_inliers` reads the estimator's own floor rather than copying it.

    They are the blob's one `min_matches_num` (spec §6.6): the estimator refuses to solve on
    fewer, and `is_good_match` refuses to accept fewer. Two fields would have needed a
    `dataclasses.replace` at every call site to stay equal.
    """
    assert LoopClosureConfig().min_inliers == RigPoseConfig().min_inliers
    raised: LoopClosureConfig = LoopClosureConfig(rig_pose=replace(RigPoseConfig(), min_inliers=99))
    assert raised.min_inliers == 99
    assert is_good_match(num_inliers=99, num_matches=100, config=raised) is False
    assert is_good_match(num_inliers=100, num_matches=100, config=raised) is True


def test_the_config_profile_supplies_the_loop_gates() -> None:
    """`from_pose_graph` maps `pose_graph_config.pb.txt` onto the stage's gate fields.

    Read off the shipped `loop-closure-fixed` profile rather than a hand-built config,
    because the point of the classmethod is that the pipeline no longer spells the five
    assignments at its call site — and the profile's own 1.0 m / 10 deg are what a real run
    gets.
    """
    if not LOOP_CLOSURE_CONFIG_DIR.is_dir():
        pytest.skip(f"the loop-closure config profile is not present at {LOOP_CLOSURE_CONFIG_DIR}")
    pose_graph: PoseGraphConfig = read_config_directory(LOOP_CLOSURE_CONFIG_DIR).pose_graph
    config: LoopClosureConfig = LoopClosureConfig.from_pose_graph(pose_graph)
    assert config.loop_closure_interval_ratio == pose_graph.loop_closure_interval_ratio
    assert config.max_translation_m == pose_graph.loop_edge_translation_threshold_meters == 1.0
    assert config.max_rotation_deg == pose_graph.loop_edge_rotation_threshold_degrees == 10.0
    assert config.loop_residual_weight == pose_graph.loop_residual_weight


def _loop_edge(translation_m: float, rotation_deg: float) -> PoseGraphEdge:
    """Build a loop edge with a chosen translation magnitude and rotation angle.

    Args:
        translation_m: Translation magnitude in metres, along `x`.
        rotation_deg: Rotation angle in degrees, about `z`.

    Returns:
        The edge.
    """
    return PoseGraphEdge(
        source=1,
        target=2,
        source_T_target=pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_euler("z", rotation_deg, degrees=True).as_quat()), np.array([translation_m, 0.0, 0.0])
        ),
        information=np.eye(6),
        kind="loop",
    )


def test_the_translation_gate_accepts_a_short_loop_and_rejects_a_long_one() -> None:
    """`loop_edge_translation_threshold_meters` (spec §6.4 step 7), at the 3.0 m default."""
    config: LoopClosureConfig = LoopClosureConfig()
    inside: PoseGraphEdge = _loop_edge(translation_m=2.0, rotation_deg=0.0)
    outside: PoseGraphEdge = _loop_edge(translation_m=5.0, rotation_deg=0.0)
    kept: list[PoseGraphEdge] = gate_loop_edges([inside, outside], config.max_translation_m, config.max_rotation_deg)
    assert kept == [inside]


def test_the_rotation_gate_accepts_a_small_turn_and_rejects_a_large_one() -> None:
    """`loop_edge_rotation_threshold_degrees` (spec §6.4 step 7), at the 30 deg default."""
    config: LoopClosureConfig = LoopClosureConfig()
    inside: PoseGraphEdge = _loop_edge(translation_m=1.0, rotation_deg=20.0)
    outside: PoseGraphEdge = _loop_edge(translation_m=1.0, rotation_deg=75.0)
    kept: list[PoseGraphEdge] = gate_loop_edges([inside, outside], config.max_translation_m, config.max_rotation_deg)
    assert kept == [inside]


def test_the_shipped_zero_thresholds_are_dead() -> None:
    """The blob's 0.0 defaults reject every candidate, which is why the loop path is dead."""
    edges: list[PoseGraphEdge] = [_loop_edge(translation_m=0.3, rotation_deg=1.0)]
    assert gate_loop_edges(edges, max_translation_m=0.0, max_rotation_deg=0.0) == []


# --------------------------------------------------------------------------------------
# d. the temporal gate
# --------------------------------------------------------------------------------------


def test_the_time_gate_is_the_larger_of_the_two_specs() -> None:
    """A candidate must pass both the fixed threshold and the session-ratio one."""
    long_session: list[int] = [0, 400_000_000]
    config: LoopClosureConfig = LoopClosureConfig()
    assert session_duration_seconds(long_session) == pytest.approx(400.0)
    assert minimum_time_gap_seconds(long_session, config) == pytest.approx(32.0)
    short_session: list[int] = [0, 20_000_000]
    assert minimum_time_gap_seconds(short_session, config) == pytest.approx(10.0)


def test_the_fixed_ten_second_gate_is_unusable_on_galileo() -> None:
    """Galileo lasts 0.933 s, so the blob's 10 s gate rejects every candidate in the dataset.

    Only the ratio gate (0.08 x 0.933 = 75 ms, about two rig frames) is live there, which is
    why `LoopClosureConfig.loop_interval_threshold_in_seconds` must be lowered per dataset
    and why the diagnostics report the gap that was applied.
    """
    if not GALILEO_KEYFRAMES_META.is_file():
        pytest.skip(f"Galileo metadata is not present at {GALILEO_KEYFRAMES_META}")
    frames_meta: FramesMeta = read_frames_meta(GALILEO_KEYFRAMES_META)
    timestamps_us: list[int] = [keyframe.timestamp_microseconds for keyframe in frames_meta.keyframes]
    duration: float = session_duration_seconds(timestamps_us)
    assert duration == pytest.approx(0.933, abs=0.01)
    assert minimum_time_gap_seconds(timestamps_us, LoopClosureConfig()) > duration
    lowered: LoopClosureConfig = LoopClosureConfig(loop_interval_threshold_in_seconds=0.0)
    assert minimum_time_gap_seconds(timestamps_us, lowered) == pytest.approx(0.08 * duration)


# --------------------------------------------------------------------------------------
# e. the cameras
# --------------------------------------------------------------------------------------


def test_every_camera_is_built_with_a_prior_focal_length(scene: SquareRigScene) -> None:
    """`calibrated_cameras` sets `has_prior_focal_length`, which nothing else does.

    `docs/spec/pycolmap-capabilities.md` §5: a freshly built camera has the flag false, and
    with it false every COLMAP estimator that can fall back to an uncalibrated model does —
    a fisheye pair comes back `DEGENERATE` with zero inliers and no warning. The loop stage
    no longer estimates a two-view geometry, but the same cameras go on to the mapper.
    """
    naive: pycolmap.Camera = pycolmap.Camera(
        camera_id=0,
        model="PINHOLE",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        params=[FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2],
    )
    assert naive.has_prior_focal_length is False
    cameras: dict[int, pycolmap.Camera] = calibrated_cameras(scene.frames_meta)
    assert cameras
    for camera in cameras.values():
        assert camera.has_prior_focal_length is True
