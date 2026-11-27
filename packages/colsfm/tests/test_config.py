"""The shipped `.pb.txt` profiles must load with the values the specs quote."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from google.protobuf import text_format

from colsfm import RESOURCES_DIR
from colsfm.config import (
    MESSAGE_BY_CONFIG_STEM,
    CusfmConfig,
    KeyframeSelectionConfig,
    PoseGraphConfig,
    read_config_directory,
    read_config_message,
)


@pytest.fixture(scope="module")
def isaac(repo_root: Path) -> CusfmConfig:
    """The isaac profile, the one both shipped demos run."""
    return read_config_directory(RESOURCES_DIR / "configs" / "isaac")


@pytest.fixture(scope="module")
def av(repo_root: Path) -> CusfmConfig:
    """The AV profile, kept as the contrast case in the spec tables."""
    return read_config_directory(RESOURCES_DIR / "configs" / "av")


def test_every_shipped_config_parses_against_its_schema_message(repo_root: Path) -> None:
    """All 11 isaac configs load with no unknown fields."""
    directory: Path = RESOURCES_DIR / "configs" / "isaac"
    for stem, message_full_name in MESSAGE_BY_CONFIG_STEM.items():
        path: Path = directory / f"{stem}.pb.txt"
        assert path.is_file(), path
        assert read_config_message(path, message_full_name) is not None


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    """A typo in a config must fail loudly, not be ignored."""
    path: Path = tmp_path / "pose_graph_config.pb.txt"
    path.write_text("connected_keyframe_num: 1\nno_such_field: 3\n")
    with pytest.raises(text_format.ParseError):
        read_config_message(path, "protos.visual.cusfm.PoseGraphConfig")


def test_isaac_vision_mapping_matches_the_spec_table(isaac: CusfmConfig) -> None:
    """keypoints_mapper_main.md §3.1, isaac column."""
    mapping = isaac.vision_mapping
    assert mapping.min_num_matches_per_pair == 15
    assert mapping.depth_threshold == 300
    assert mapping.search_depth == 15
    assert mapping.min_correspondences == 3
    assert mapping.initial_max_pixel_error == pytest.approx(25.0)
    assert mapping.final_max_pixel_error == pytest.approx(5.0)
    assert mapping.min_triangulation_deg == pytest.approx(2.0)
    assert mapping.max_ransac_iteration == 1000
    assert mapping.min_num_samples == 3
    assert mapping.iterative_triangulation_times == 10
    assert mapping.num_ba_iterations == 5
    assert mapping.max_observation_change == pytest.approx(0.0005)
    assert mapping.num_threads == 8
    assert mapping.random_seed == 1
    assert mapping.triangulation_method == "NORMALIZED_COORDINATES"
    assert mapping.pose_constrains_type == "AUTO"
    assert mapping.localization_mode == "FIX"
    assert mapping.use_camera_extrinsic_constraint is True
    assert mapping.use_relative_pose_constraint is True
    assert mapping.use_absolute_pose_constraints is False
    assert mapping.use_absolute_position_constraints is False
    assert mapping.extrinsic_max_timestamp_diff_microseconds == 100
    assert mapping.relative_pose_measurement_min_timestamp_gap_us == 100000
    assert mapping.refine_ba is False


def test_isaac_bundle_adjustment_matches_the_spec_table(isaac: CusfmConfig) -> None:
    """keypoints_mapper_main.md §3.2, isaac column, including `ba_frame_type`."""
    ba = isaac.vision_mapping.bundle_adjustment
    assert ba.loss_type == "CAUCHY"
    assert ba.loss_function_scale == pytest.approx(1.0)
    assert ba.max_num_iterations == 200
    assert ba.max_invalid_steps == 10
    assert ba.max_linear_solver_iterations == 100
    assert ba.num_threads == 8
    assert ba.reprojection_error_standard_deviation == pytest.approx(4.0)
    assert ba.relative_pose_translation_error_meters == pytest.approx(0.1)
    assert ba.relative_pose_rotation_error_degrees == pytest.approx(5.0)
    assert ba.extrinsic_error_meters == pytest.approx(0.01)
    assert ba.extrinsic_error_degrees == pytest.approx(2.0)
    assert ba.relative_constraint_use_pose_covariance is False
    assert ba.do_rolling_shutter_correction is False
    assert ba.ba_frame_type == "VEHICLE_RIG"
    assert ba.use_cudss_solver is True


def test_av_profile_differs_where_the_spec_says_it_does(av: CusfmConfig) -> None:
    """The AV column of the same tables: looser gates, rolling shutter, refine BA."""
    assert av.vision_mapping.initial_max_pixel_error == pytest.approx(40.0)
    assert av.vision_mapping.final_max_pixel_error == pytest.approx(8.0)
    assert av.vision_mapping.min_triangulation_deg == pytest.approx(1.0)
    assert av.vision_mapping.refine_ba is True
    assert av.vision_mapping.triangulation_method == "MIDPOINT"
    assert av.vision_mapping.pose_constrains_type == "POSITION_CONSTRAINS"
    assert av.vision_mapping.bundle_adjustment.ba_frame_type == "VEHICLE_FRAME"
    assert av.vision_mapping.bundle_adjustment.do_rolling_shutter_correction is True


def test_isaac_pose_graph_matches_the_shipped_file(isaac: CusfmConfig) -> None:
    """`pose_graph_config.pb.txt`, plus the gates it leaves at their zero defaults."""
    pose_graph = isaac.pose_graph
    assert pose_graph.connected_keyframe_num == 1
    assert pose_graph.loop_closure_query_count == 20
    assert pose_graph.loop_closure_interval_ratio == pytest.approx(0.08)
    assert pose_graph.loop_residual_weight == pytest.approx(0.25)
    assert pose_graph.good_loop_threshold == pytest.approx(150.0)
    assert pose_graph.loop_candidates_per_frame == 3, "no colsfm.toml in isaac: the default"
    assert pose_graph.loop_edge_translation_threshold_meters == pytest.approx(0.0)
    assert pose_graph.loop_edge_rotation_threshold_degrees == pytest.approx(0.0)


def test_isaac_matching_and_pair_selection(isaac: CusfmConfig) -> None:
    """The matcher's own profile is far stricter than the task worker's."""
    assert isaac.matching.match_threshold == pytest.approx(0.99)
    assert isaac.matching_task_worker.lightglue.match_threshold == pytest.approx(0.3)
    assert isaac.matching_task_worker.lightglue.match_top_k == 500
    assert isaac.matching_task_worker.lightglue.num_points_tolerance_fraction == pytest.approx(0.3)
    assert isaac.matching_task_worker.verification.enable_geometry_verification is True
    assert isaac.matching_task_worker.verification.max_pixel_error == pytest.approx(4.0)
    assert isaac.matching_task_worker.verification.min_ransac_confidence == pytest.approx(0.98)
    assert isaac.match_pair_select.search_method == "POSE_GRAPH_ASSOCIATION"
    assert isaac.match_pair_select.use_downsampling is True
    assert isaac.match_pair_select.max_keyframes_per_collection == 6
    assert isaac.match_pair_select.min_distance_between_keyframes_meters == pytest.approx(0.05)


def test_isaac_keypoint_creation(isaac: CusfmConfig) -> None:
    """`keypoint_creation_config.pb.txt`, ALIKED branch."""
    aliked = isaac.keypoint_creation.aliked_detector
    assert aliked.max_key_points_for_tensorrt == 4000
    assert aliked.detector_threshold == pytest.approx(0.005)
    assert aliked.num_points_tolerance_fraction == pytest.approx(0.3)
    assert aliked.global_non_max_method == "SUPPRESSION_VIA_SQUARE_COVERING"


def test_keyframe_selection_defaults_are_the_gflag_defaults(isaac: CusfmConfig) -> None:
    """Selection is not in the config directory; the defaults come from the binary."""
    assert isaac.keyframe_selection == KeyframeSelectionConfig(
        min_inter_frame_distance_m=0.5,
        min_inter_frame_rotation_degrees=5.0,
        sample_sync_threshold_microseconds=50,
        stereo_pair_non_baseline_max_distance_m=0.0,
    )


def test_demo_overrides_can_be_supplied(repo_root: Path) -> None:
    """The isaac demo runs 0.0 m, 5 deg, 100 us, 0.001 m (feature_extractor_main.md §2)."""
    demo: KeyframeSelectionConfig = KeyframeSelectionConfig(
        min_inter_frame_distance_m=0.0,
        min_inter_frame_rotation_degrees=5.0,
        sample_sync_threshold_microseconds=100,
        stereo_pair_non_baseline_max_distance_m=0.001,
    )
    config: CusfmConfig = read_config_directory(RESOURCES_DIR / "configs" / "isaac", demo)
    assert config.keyframe_selection.min_inter_frame_distance_m == pytest.approx(0.0)
    assert config.keyframe_selection.sample_sync_threshold_microseconds == 100


def test_the_kitti_profile_verifies_one_loop_candidate_per_rig_frame() -> None:
    """`loop_candidates_per_frame` comes from the profile's own `colsfm.toml`; cuSFM's `num_votes` is not read."""
    assert read_config_directory(RESOURCES_DIR / "configs" / "kitti").pose_graph.loop_candidates_per_frame == 1
    assert read_config_directory(RESOURCES_DIR / "configs" / "loop-closure-fixed").pose_graph.loop_candidates_per_frame == 3


def test_the_insta360_cube_profile_is_its_base_with_looser_loop_gates_and_the_prior_bound() -> None:
    """insta360-cube keeps only what differs from loop-closure-fixed; every other stem is read from that base."""
    configs: Path = RESOURCES_DIR / "configs"
    cube: CusfmConfig = read_config_directory(configs / "insta360-cube")
    base: CusfmConfig = read_config_directory(configs / "loop-closure-fixed")
    pose_graph: PoseGraphConfig = replace(
        base.pose_graph, connected_keyframe_num=2, good_loop_threshold=40.0, loop_edge_translation_threshold_meters=3.0,
        loop_edge_rotation_threshold_degrees=180.0, loop_max_prior_translation_m=1.5, loop_max_prior_rotation_deg=10.0,
    )
    assert cube == replace(base, profile="insta360-cube", pose_graph=pose_graph)
    assert sorted(path.name for path in (configs / "insta360-cube").iterdir()) == ["colsfm.toml", "pose_graph_config.pb.txt"]


def test_a_profile_without_a_base_still_needs_every_file(tmp_path: Path) -> None:
    (tmp_path / "colsfm.toml").write_text("")
    with pytest.raises(FileNotFoundError, match="keypoint_creation_config"):
        read_config_directory(tmp_path)


def test_a_colsfm_profile_file_with_an_unknown_field_is_refused_naming_the_file(tmp_path: Path) -> None:
    from colsfm.config import read_colsfm_profile

    (tmp_path / "colsfm.toml").write_text("num_votes = 3\n")
    with pytest.raises(ValueError, match="colsfm.toml"):
        read_colsfm_profile(tmp_path)
