"""cuSFM's protobuf-text configs as typed frozen dataclasses.

`pycusfm/configs/<profile>/*.pb.txt` are `text_format` renderings of messages in
the vendored schema, one message per file:

| File | Message |
|---|---|
| `keypoint_creation_config.pb.txt` | `protos.common.image.KeypointCreationParams` |
| `match_pair_select_config.pb.txt` | `protos.visual.cusfm.MatchPairSelectConfig` |
| `matching_config.pb.txt` | `protos.common.image.KeypointMatchingParams` |
| `matching_task_worker_config.pb.txt` | `protos.visual.cusfm.MatchingTaskWorkerConfig` |
| `pose_graph_config.pb.txt` | `protos.visual.cusfm.PoseGraphConfig` |
| `vision_mapping_config.pb.txt` | `protos.visual.cusfm.VisionMappingConfig` |
| `vo_pose_optimize_ba_config.pb.txt` | `protos.visual.geometry.BundleAdjustmentConfig` |
| `image_retrieval_config.pb.txt` | `protos.visual.loop_closing.ImageRetrievalConfig` |
| `association_config.pb.txt` | `protos.visual.cusfm.AssociationWorkConfig` |

Parsing goes through `text_format` against those messages, so a field name the
schema does not know is a hard error rather than a silently ignored line, and an
unset field reads as its proto3 default exactly as the binaries see it.

Keyframe selection is the exception: its three knobs are gflags on
`feature_extractor_main`, not config fields, so `KeyframeSelectionConfig` carries
the gflag defaults and the isaac demo's overrides
(feature_extractor_main.md §2).

Field documentation follows `docs/spec/keypoints_mapper_main.md` §3 and
`docs/spec/pose_graph_main.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

from google.protobuf import text_format
from google.protobuf.message import Message

from colsfm.schema import CusfmSchema, load_schema

LossFunctionType: TypeAlias = Literal["TRIVIAL", "SOFT_L1", "CAUCHY", "HUBER"]
"""Ceres robust loss applied to the whitened residual."""

BundleAdjustmentFrameType: TypeAlias = Literal["CAMERA_FRAME", "VEHICLE_FRAME", "VEHICLE_RIG"]
"""Which body BA optimises: one pose per camera, per vehicle, or per rig."""

PoseConstrainsType: TypeAlias = Literal["AUTO", "POSITION_CONSTRAINS", "RELATIVE_CONSTRAINS", "CONSTANT_CONSTRAINS"]
"""Which pose prior the mapper adds to the BA problem."""

LocalizationModeType: TypeAlias = Literal["NONE", "FIX", "ADJUST", "PRE_BUILD"]
"""How a previously built map is treated."""

TriangulationMethodType: TypeAlias = Literal[
    "NORMALIZED_COORDINATES", "PIXEL_COORDINATES", "MIDPOINT", "NORMALIZED_COORDINATES_WITH_DEPTH"
]
"""Triangulation formulation; isaac leaves it unset, i.e. `NORMALIZED_COORDINATES`."""

KeyFrameSearchMethod: TypeAlias = Literal["RADIUS_SEARCH", "IMAGE_RETRIEVAL", "TRACK_2D_OVERLAP_IR", "POSE_GRAPH_ASSOCIATION"]
"""How the task builder picks image pairs to match."""

GlobalNonMaxMethod: TypeAlias = Literal["NONE", "SUPPRESSION_VIA_SQUARE_COVERING", "TOP_M"]
"""Keypoint spatial thinning strategy."""

MESSAGE_BY_CONFIG_STEM: Final[dict[str, str]] = {
    "keypoint_creation_config": "protos.common.image.KeypointCreationParams",
    "match_pair_select_config": "protos.visual.cusfm.MatchPairSelectConfig",
    "matching_config": "protos.common.image.KeypointMatchingParams",
    "matching_task_worker_config": "protos.visual.cusfm.MatchingTaskWorkerConfig",
    "pose_graph_config": "protos.visual.cusfm.PoseGraphConfig",
    "vision_mapping_config": "protos.visual.cusfm.VisionMappingConfig",
    "vo_pose_optimize_ba_config": "protos.visual.geometry.BundleAdjustmentConfig",
    "image_retrieval_config": "protos.visual.loop_closing.ImageRetrievalConfig",
    "association_config": "protos.visual.cusfm.AssociationWorkConfig",
    "localizer_config": "protos.visual.cuvgl.KeypointsLocalizerConfig",
    "query_map_config": "protos.visual.cuvgl.QueryMapConfig",
}
"""Which schema message each `<profile>/*.pb.txt` file renders."""


@dataclass(frozen=True, slots=True)
class KeyframeSelectionConfig:
    """`feature_extractor_main` gflags that drive keyframe selection.

    Not a `.pb.txt` file: these are command-line flags, defaulted here to the
    binary's own defaults. The isaac demo overrides only the distance.
    """

    min_inter_frame_distance_m: float = 0.5
    """Displacement gate in metres; the demo passes 0.0 to keep all 226 Galileo frames."""
    min_inter_frame_rotation_degrees: float = 5.0
    """Rotation gate in degrees, OR-combined with the distance gate."""
    sample_sync_threshold_microseconds: int = 50
    """Rig grouping window; the isaac demo passes 100."""
    stereo_pair_non_baseline_max_distance_m: float = 0.0
    """Auto-derive stereo pairs when the metadata declares none; the demo passes 0.001."""


@dataclass(frozen=True, slots=True)
class LearningBasedDetectorConfig:
    """ALIKED or SuperPoint detector settings from `KeypointCreationParams`."""

    max_key_points_for_tensorrt: int
    """Keypoint budget the TensorRT engine is built for."""
    detector_threshold: float
    """Score threshold below which a detection is discarded."""
    num_points_tolerance_fraction: float
    """Fraction by which the realised keypoint count may fall short of the budget."""
    global_non_max_method: GlobalNonMaxMethod
    """Spatial thinning applied after detection."""


@dataclass(frozen=True, slots=True)
class KeypointCreationConfig:
    """`keypoint_creation_config.pb.txt`, restricted to the detector the pipeline runs."""

    aliked_detector: LearningBasedDetectorConfig
    """ALIKED settings; the only detector the open pipeline uses."""
    super_point_detector: LearningBasedDetectorConfig
    """SuperPoint settings, kept so a profile switch stays typed."""


@dataclass(frozen=True, slots=True)
class MatchPairSelectConfig:
    """`match_pair_select_config.pb.txt`: which image pairs get matched."""

    search_translation_threshold_meters: float
    """Radius-search translation gate between two keyframes."""
    search_rotation_threshold_degrees: float
    """Radius-search rotation gate between two keyframes."""
    max_frame_pair_size_per_file: int
    """Pairs per emitted matching task file."""
    min_time_interval_between_keyframes_seconds: float
    """Minimum time gap before two keyframes may be paired."""
    min_distance_between_keyframes_meters: float
    """Minimum baseline before two keyframes may be paired."""
    collection_time_interval_seconds: float
    """Window used to group keyframes into downsampling collections."""
    max_keyframes_per_collection: int
    """Keyframes kept per collection when downsampling."""
    search_method: KeyFrameSearchMethod
    """Pair source: pose-graph association for isaac."""
    use_downsampling: bool
    """Whether to thin the candidate pairs at all."""
    query_result_number: int
    """Retrieval candidates per query, for `IMAGE_RETRIEVAL`."""
    min_shared_word_number: int
    """Minimum shared BoW words for a retrieval candidate."""


@dataclass(frozen=True, slots=True)
class LightGlueConfig:
    """`lightglue_params` inside a `KeypointMatchingParams`."""

    match_threshold: float
    """Minimum LightGlue score for a match to be kept."""
    match_top_k: int
    """Maximum matches returned per pair."""
    num_points_tolerance_fraction: float
    """Keypoint-count slack the engine profile allows."""
    global_non_max_method: GlobalNonMaxMethod
    """Spatial thinning applied to the keypoints before matching."""


@dataclass(frozen=True, slots=True)
class GeometryVerificationConfig:
    """`verification_config` inside `MatchingTaskWorkerConfig`: the RANSAC gate."""

    enable_geometry_verification: bool
    """Whether two-view geometry is estimated at all."""
    max_pixel_error: float
    """RANSAC inlier threshold in pixels."""
    min_ransac_confidence: float
    """RANSAC success probability driving the trial count."""
    point3d_max_error: float
    """Maximum 3-D point error accepted by the verifier."""
    depth_check_max_reproj_pixel_error: float
    """Reprojection gate for the depth consistency check."""


@dataclass(frozen=True, slots=True)
class MatchingTaskWorkerConfig:
    """`matching_task_worker_config.pb.txt`: matching plus verification."""

    lightglue: LightGlueConfig
    """LightGlue settings from `matching_params.lightglue_params`."""
    verification: GeometryVerificationConfig
    """Geometric verification settings."""
    max_pixel_error: float
    """Top-level RANSAC pixel gate; `verification_config` shadows it when set."""
    min_ransac_confidence: float
    """Top-level RANSAC confidence; `verification_config` shadows it when set."""


@dataclass(frozen=True, slots=True)
class BundleAdjustmentConfig:
    """`BundleAdjustmentConfig`, shared by the mapper and the pose graph.

    Field roles are from keypoints_mapper_main.md §3.2.
    """

    loss_type: LossFunctionType
    """Robust loss; isaac uses `CAUCHY`."""
    loss_function_scale: float
    """Loss scale, applied to the already-whitened residual."""
    max_num_iterations: int
    """Ceres iteration cap."""
    max_invalid_steps: int
    """Ceres `max_num_consecutive_invalid_steps`."""
    max_linear_solver_iterations: int
    """Ceres linear solver cap; irrelevant for a direct sparse solve."""
    num_threads: int
    """Ceres threads."""
    reprojection_error_standard_deviation: float
    """Pixel sigma whitening the reprojection residual."""
    absolute_position_constraint_error_meters: float
    """Sigma of the absolute position prior."""
    absolute_pose_translation_error_meters: float
    """Translation sigma of the absolute pose prior."""
    absolute_pose_rotation_error_degrees: float
    """Rotation sigma of the absolute pose prior."""
    relative_pose_translation_error_meters: float
    """Translation sigma of the relative pose prior."""
    relative_pose_rotation_error_degrees: float
    """Rotation sigma of the relative pose prior."""
    relative_constraint_use_pose_covariance: bool
    """Take sigmas from `pose_covariance` instead of the fixed values above."""
    extrinsic_error_meters: float
    """Translation sigma of the rig extrinsic prior."""
    extrinsic_error_degrees: float
    """Rotation sigma of the rig extrinsic prior."""
    do_rolling_shutter_correction: bool
    """Interpolate a per-row pose inside the residual."""
    ba_frame_type: BundleAdjustmentFrameType
    """Body whose pose is a free variable; isaac uses `VEHICLE_RIG`."""
    use_cudss_solver: bool
    """Use the cuDSS GPU sparse Cholesky."""
    minimizer_progress_to_stdout: bool
    """Print Ceres' per-iteration table."""
    verbose: bool
    """Print the extra `[BA COST]` group report."""


@dataclass(frozen=True, slots=True)
class PoseGraphConfig:
    """`pose_graph_config.pb.txt`: the rig-level pose graph."""

    connected_keyframe_num: int
    """How many previous rig frames each node gets a CONSECUTIVE edge to."""
    loop_closure_min_words: int
    """Minimum shared BoW words for a loop candidate."""
    loop_closure_query_count: int
    """Retrieval candidates queried per rig frame."""
    loop_closure_interval_ratio: float
    """Minimum loop separation as a fraction of the session duration."""
    use_incremental: bool
    """Grow the graph incrementally rather than in one batch."""
    loop_residual_weight: float
    """Weight applied to LOOP edge residuals relative to CONSECUTIVE ones."""
    good_loop_threshold: float
    """Score above which a loop candidate is accepted outright."""
    num_votes: int
    """Votes required before a loop candidate becomes an edge."""
    use_switchable_pgo: bool
    """Attach a switch variable to each loop edge (switchable-constraint PGO)."""
    confidence_to_pose_error_ratio: float
    """Converts a match confidence into a pose-error sigma."""
    loop_edge_prune_threshold: float
    """Residual above which a loop edge is dropped after a solve."""
    point3d_max_error: float
    """Maximum 3-D point error accepted when estimating a loop edge."""
    loop_edge_translation_threshold_meters: float
    """Translation gate a loop edge must pass; 0.0 disables the gate."""
    loop_edge_rotation_threshold_degrees: float
    """Rotation gate a loop edge must pass; 0.0 disables the gate."""
    use_sequential_edge_pose_estimation: bool
    """Re-estimate CONSECUTIVE edges from matches instead of trusting the input poses."""
    optimization: BundleAdjustmentConfig
    """Ceres settings for the pose-graph solve."""


@dataclass(frozen=True, slots=True)
class VisionMappingConfig:
    """`vision_mapping_config.pb.txt`: triangulation and bundle adjustment."""

    min_num_matches_per_pair: int
    """Drop an image pair with fewer matches than this."""
    depth_threshold: float
    """World-frame Z cap in metres, one axis and one-sided; not a camera depth."""
    search_depth: int
    """Maximum BFS hop count when growing one track over the match graph."""
    min_correspondences: int
    """Minimum track length; two-view tracks are dropped."""
    initial_max_pixel_error: float
    """First value of the linear per-round reprojection gate schedule."""
    final_max_pixel_error: float
    """Last value of that schedule; isaac runs 25, 20, 15, 10, 5."""
    min_triangulation_deg: float
    """Minimum triangulation angle in degrees."""
    max_ransac_iteration: int
    """RANSAC trial cap; the only `RansacConfig` field taken from config."""
    min_num_samples: int
    """RANSAC minimal sample size, in views."""
    iterative_triangulation_times: int
    """Maximum inner triangulation passes; exits early when a pass adds nothing."""
    maximum_cache_frame_size: int
    """Keyframe LRU cache size; performance only."""
    num_ba_iterations: int
    """Outer triangulate / BA / filter rounds."""
    max_observation_change: float
    """Early-exit threshold on the outer loop."""
    num_threads: int
    """Triangulation and Ceres threads; must be 1 for deterministic triangulation."""
    random_seed: int
    """Seeds the global libc `srand`; does not make threaded triangulation deterministic."""
    triangulation_method: TriangulationMethodType
    """Triangulation formulation; isaac leaves it at `NORMALIZED_COORDINATES`."""
    pose_constrains_type: PoseConstrainsType
    """Which pose prior enters the BA problem."""
    use_camera_extrinsic_constraint: bool
    """Add the rig extrinsic prior term."""
    use_relative_pose_constraint: bool
    """Add the relative pose prior term."""
    use_absolute_pose_constraints: bool
    """Add the absolute pose prior term."""
    use_absolute_position_constraints: bool
    """Add the absolute position prior term."""
    extrinsic_max_timestamp_diff_microseconds: int
    """Maximum timestamp skew for two images to count as one rig frame."""
    relative_pose_measurement_min_timestamp_gap_us: int
    """Minimum gap between frames joined by a relative constraint."""
    localization_mode: LocalizationModeType
    """How a previously built map is treated."""
    refine_ba: bool
    """Run the second, stringent BA stage."""
    image_pair_min_shared_points: int
    """Minimum shared points for a pair to enter the refine stage."""
    refine_ba_use_fixed_max_pixel_error: bool
    """Refine stage uses a fixed rather than decayed pixel gate."""
    refine_ba_avoid_loss_function: bool
    """Refine stage drops the robust loss."""
    bundle_adjustment: BundleAdjustmentConfig
    """Ceres settings for the mapping BA."""


@dataclass(frozen=True, slots=True)
class CusfmConfig:
    """One `pycusfm/configs/<profile>` directory, parsed."""

    profile: str
    """Directory name, `isaac` or `av`."""
    keypoint_creation: KeypointCreationConfig
    """`keypoint_creation_config.pb.txt`."""
    keyframe_selection: KeyframeSelectionConfig
    """gflag-sourced selection knobs, not read from the directory."""
    match_pair_select: MatchPairSelectConfig
    """`match_pair_select_config.pb.txt`."""
    matching: LightGlueConfig
    """`matching_config.pb.txt`, the standalone matcher profile."""
    matching_task_worker: MatchingTaskWorkerConfig
    """`matching_task_worker_config.pb.txt`."""
    pose_graph: PoseGraphConfig
    """`pose_graph_config.pb.txt`."""
    vision_mapping: VisionMappingConfig
    """`vision_mapping_config.pb.txt`."""


def read_config_message(path: Path, message_full_name: str, schema: CusfmSchema | None = None) -> Message:
    """Parse one `.pb.txt` config against its schema message.

    Args:
        path: The config file.
        message_full_name: Fully qualified message name it renders.
        schema: Loaded cuSFM schema; the vendored one by default.

    Returns:
        The parsed message.

    Raises:
        text_format.ParseError: When the file names a field the message lacks.
    """
    resolved_schema: CusfmSchema = load_schema() if schema is None else schema
    return text_format.Parse(path.read_text(), resolved_schema.message_class(message_full_name)())


def _enum_name(message: Message, field_name: str, schema: CusfmSchema) -> str:
    """Read one enum field of a message as its symbolic name.

    Args:
        message: The message holding the field.
        field_name: Enum field to read.
        schema: Loaded cuSFM schema.

    Returns:
        The enum value's name.
    """
    enum_type = message.DESCRIPTOR.fields_by_name[field_name].enum_type
    return schema.enum_value_name(enum_type.full_name, getattr(message, field_name))


def _detector(message: Message, schema: CusfmSchema) -> LearningBasedDetectorConfig:
    """Convert a `LearningBasedDetectorParams` message.

    Args:
        message: The detector parameters.
        schema: Loaded cuSFM schema.

    Returns:
        The typed detector settings.
    """
    return LearningBasedDetectorConfig(
        max_key_points_for_tensorrt=int(message.max_key_points_for_tensorrt),
        detector_threshold=float(message.detector_threshold),
        num_points_tolerance_fraction=float(message.num_points_tolerance_fraction),
        global_non_max_method=_enum_name(message, "global_non_max_method", schema),
    )


def _bundle_adjustment(message: Message, schema: CusfmSchema) -> BundleAdjustmentConfig:
    """Convert a `BundleAdjustmentConfig` message.

    Args:
        message: The Ceres settings.
        schema: Loaded cuSFM schema.

    Returns:
        The typed settings.
    """
    return BundleAdjustmentConfig(
        loss_type=_enum_name(message, "loss_type", schema),
        loss_function_scale=float(message.loss_function_scale),
        max_num_iterations=int(message.max_num_iterations),
        max_invalid_steps=int(message.max_invalid_steps),
        max_linear_solver_iterations=int(message.max_linear_solver_iterations),
        num_threads=int(message.num_threads),
        reprojection_error_standard_deviation=float(message.reprojection_error_standard_deviation),
        absolute_position_constraint_error_meters=float(message.absolute_position_constraint_error_meters),
        absolute_pose_translation_error_meters=float(message.absolute_pose_translation_error_meters),
        absolute_pose_rotation_error_degrees=float(message.absolute_pose_rotation_error_degrees),
        relative_pose_translation_error_meters=float(message.relative_pose_translation_error_meters),
        relative_pose_rotation_error_degrees=float(message.relative_pose_rotation_error_degrees),
        relative_constraint_use_pose_covariance=bool(message.relative_constraint_use_pose_covariance),
        extrinsic_error_meters=float(message.extrinsic_error_meters),
        extrinsic_error_degrees=float(message.extrinsic_error_degrees),
        do_rolling_shutter_correction=bool(message.do_rolling_shutter_correction),
        ba_frame_type=_enum_name(message, "ba_frame_type", schema),
        use_cudss_solver=bool(message.use_cudss_solver),
        minimizer_progress_to_stdout=bool(message.minimizer_progress_to_stdout),
        verbose=bool(message.verbose),
    )


def _lightglue(message: Message, schema: CusfmSchema) -> LightGlueConfig:
    """Convert a `LightGlueMatcherParams` message.

    Args:
        message: The LightGlue parameters.
        schema: Loaded cuSFM schema.

    Returns:
        The typed settings.
    """
    return LightGlueConfig(
        match_threshold=float(message.match_threshold),
        match_top_k=int(message.match_top_k),
        num_points_tolerance_fraction=float(message.num_points_tolerance_fraction),
        global_non_max_method=_enum_name(message, "global_non_max_method", schema),
    )


def read_config_directory(
    directory: Path,
    keyframe_selection: KeyframeSelectionConfig | None = None,
    schema: CusfmSchema | None = None,
) -> CusfmConfig:
    """Parse a whole `pycusfm/configs/<profile>` directory.

    Args:
        directory: The profile directory, e.g. `pycusfm/configs/isaac`.
        keyframe_selection: gflag-sourced selection knobs; defaults to the binary's defaults.
        schema: Loaded cuSFM schema; the vendored one by default.

    Returns:
        The typed configuration.

    Raises:
        FileNotFoundError: When a required config file is missing.
        text_format.ParseError: When a config names a field the schema does not know.
    """
    resolved_schema: CusfmSchema = load_schema() if schema is None else schema

    def read(stem: str) -> Message:
        path: Path = directory / f"{stem}.pb.txt"
        if not path.is_file():
            raise FileNotFoundError(f"No {stem}.pb.txt in {directory}")
        return read_config_message(path, MESSAGE_BY_CONFIG_STEM[stem], resolved_schema)

    keypoint_creation_message: Message = read("keypoint_creation_config")
    match_pair_message: Message = read("match_pair_select_config")
    matching_message: Message = read("matching_config")
    worker_message: Message = read("matching_task_worker_config")
    pose_graph_message: Message = read("pose_graph_config")
    mapping_message: Message = read("vision_mapping_config")

    return CusfmConfig(
        profile=directory.name,
        keypoint_creation=KeypointCreationConfig(
            aliked_detector=_detector(keypoint_creation_message.aliked_detector, resolved_schema),
            super_point_detector=_detector(keypoint_creation_message.super_point_detector, resolved_schema),
        ),
        keyframe_selection=KeyframeSelectionConfig() if keyframe_selection is None else keyframe_selection,
        match_pair_select=MatchPairSelectConfig(
            search_translation_threshold_meters=float(match_pair_message.search_translation_threshold_meters),
            search_rotation_threshold_degrees=float(match_pair_message.search_rotation_threshold_degrees),
            max_frame_pair_size_per_file=int(match_pair_message.max_frame_pair_size_per_file),
            min_time_interval_between_keyframes_seconds=float(match_pair_message.min_time_interval_between_keyframes_seconds),
            min_distance_between_keyframes_meters=float(match_pair_message.min_distance_between_keyframes_meters),
            collection_time_interval_seconds=float(match_pair_message.collection_time_interval_seconds),
            max_keyframes_per_collection=int(match_pair_message.max_keyframes_per_collection),
            search_method=_enum_name(match_pair_message, "search_method", resolved_schema),
            use_downsampling=bool(match_pair_message.use_downsampling),
            query_result_number=int(match_pair_message.query_result_number),
            min_shared_word_number=int(match_pair_message.min_shared_word_number),
        ),
        matching=_lightglue(matching_message.lightglue_params, resolved_schema),
        matching_task_worker=MatchingTaskWorkerConfig(
            lightglue=_lightglue(worker_message.matching_params.lightglue_params, resolved_schema),
            verification=GeometryVerificationConfig(
                enable_geometry_verification=bool(worker_message.verification_config.enable_geometry_verification),
                max_pixel_error=float(worker_message.verification_config.max_pixel_error),
                min_ransac_confidence=float(worker_message.verification_config.min_ransac_confidence),
                point3d_max_error=float(worker_message.verification_config.point3d_max_error),
                depth_check_max_reproj_pixel_error=float(worker_message.verification_config.depth_check_max_reproj_pixel_error),
            ),
            max_pixel_error=float(worker_message.max_pixel_error),
            min_ransac_confidence=float(worker_message.min_ransac_confidence),
        ),
        pose_graph=PoseGraphConfig(
            connected_keyframe_num=int(pose_graph_message.connected_keyframe_num),
            loop_closure_min_words=int(pose_graph_message.loop_closure_min_words),
            loop_closure_query_count=int(pose_graph_message.loop_closure_query_count),
            loop_closure_interval_ratio=float(pose_graph_message.loop_closure_interval_ratio),
            use_incremental=bool(pose_graph_message.use_incremental),
            loop_residual_weight=float(pose_graph_message.loop_residual_weight),
            good_loop_threshold=float(pose_graph_message.good_loop_threshold),
            num_votes=int(pose_graph_message.num_votes),
            use_switchable_pgo=bool(pose_graph_message.use_switchable_pgo),
            confidence_to_pose_error_ratio=float(pose_graph_message.confidence_to_pose_error_ratio),
            loop_edge_prune_threshold=float(pose_graph_message.loop_edge_prune_threshold),
            point3d_max_error=float(pose_graph_message.point3d_max_error),
            loop_edge_translation_threshold_meters=float(pose_graph_message.loop_edge_translation_threshold_meters),
            loop_edge_rotation_threshold_degrees=float(pose_graph_message.loop_edge_rotation_threshold_degrees),
            use_sequential_edge_pose_estimation=bool(pose_graph_message.use_sequential_edge_pose_estimation),
            optimization=_bundle_adjustment(pose_graph_message.optimization_config, resolved_schema),
        ),
        vision_mapping=VisionMappingConfig(
            min_num_matches_per_pair=int(mapping_message.min_num_matches_per_pair),
            depth_threshold=float(mapping_message.depth_threshold),
            search_depth=int(mapping_message.search_depth),
            min_correspondences=int(mapping_message.min_correspondences),
            initial_max_pixel_error=float(mapping_message.initial_max_pixel_error),
            final_max_pixel_error=float(mapping_message.final_max_pixel_error),
            min_triangulation_deg=float(mapping_message.min_triangulation_deg),
            max_ransac_iteration=int(mapping_message.max_ransac_iteration),
            min_num_samples=int(mapping_message.min_num_samples),
            iterative_triangulation_times=int(mapping_message.iterative_triangulation_times),
            maximum_cache_frame_size=int(mapping_message.maximum_cache_frame_size),
            num_ba_iterations=int(mapping_message.num_ba_iterations),
            max_observation_change=float(mapping_message.max_observation_change),
            num_threads=int(mapping_message.num_threads),
            random_seed=int(mapping_message.random_seed),
            triangulation_method=_enum_name(mapping_message, "triangulation_method", resolved_schema),
            pose_constrains_type=_enum_name(mapping_message, "pose_constrains_type", resolved_schema),
            use_camera_extrinsic_constraint=bool(mapping_message.use_camera_extrinsic_constraint),
            use_relative_pose_constraint=bool(mapping_message.use_relative_pose_constraint),
            use_absolute_pose_constraints=bool(mapping_message.use_absolute_pose_constraints),
            use_absolute_position_constraints=bool(mapping_message.use_absolute_position_constraints),
            extrinsic_max_timestamp_diff_microseconds=int(mapping_message.extrinsic_max_timestamp_diff_microseconds),
            relative_pose_measurement_min_timestamp_gap_us=int(mapping_message.relative_pose_measurement_min_timestamp_gap_us),
            localization_mode=_enum_name(mapping_message, "localization_mode", resolved_schema),
            refine_ba=bool(mapping_message.refine_ba),
            image_pair_min_shared_points=int(mapping_message.image_pair_min_shared_points),
            refine_ba_use_fixed_max_pixel_error=bool(mapping_message.refine_ba_use_fixed_max_pixel_error),
            refine_ba_avoid_loss_function=bool(mapping_message.refine_ba_avoid_loss_function),
            bundle_adjustment=_bundle_adjustment(mapping_message.bundle_adjustment_config, resolved_schema),
        ),
    )
