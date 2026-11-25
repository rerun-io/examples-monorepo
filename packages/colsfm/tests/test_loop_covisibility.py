"""Tests for the covisibility radius: how far apart two rig frames of this scene can be and still share structure.

Seams under test (all public):

* `colsfm.loop_covisibility.triangulated_depths_m` — depth of matched bearings under a relative pose.
* `colsfm.loop_covisibility.covisibility_radius_m` — the scene-depth quantile over stage-4 pairs.
* `colsfm.database.read_inlier_sample` — the stage-4 inliers the radius is learned from.
* `colsfm.loop_closure.plan_loop_search` with a radius — rig pairs beyond the edge bound but inside the radius are
  matched for the mapper only, never measured.
* `colsfm.loop_mapper_check.pose_checked_inliers` — those mapper-only pairs are checked against the prior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64, Int64, UInt32
from numpy import ndarray
from test_loop_closure import SquareRigScene, make_square_rig_scene  # pyrefly: ignore[missing-import]

from colsfm.database import ImagePair, read_inlier_sample
from colsfm.loop_closure import LoopClosureConfig, LoopSearchPlan, plan_loop_search
from colsfm.loop_covisibility import covisibility_radius_m, spans_two_rig_frames, triangulated_depths_m
from colsfm.loop_mapper_check import pose_checked_inliers
from colsfm.loop_pairs import observation_pairs
from colsfm.pairs import normalise_pair

EDGE_BOUND_CONFIG: LoopClosureConfig = LoopClosureConfig(max_translation_m=0.5, max_prior_translation_m=0.5)
"""T = P = 0.5 m: the edge bound is 1 m, and the square scene shortlists pairs 1.2-3.6 m apart beyond it."""


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> SquareRigScene:
    """The synthetic square-walking rig, built once for the module."""
    return make_square_rig_scene(tmp_path_factory.mktemp("loop_covisibility"))


def _true_matches(scene: SquareRigScene, image_a: int, image_b: int) -> UInt32[ndarray, "n_matches 2"]:
    """Keypoint index pairs of the two images that observe the same scene point, lower image id first."""
    low, high = sorted((image_a, image_b))
    points_low: Int64[ndarray, "n_low"] = np.asarray(scene.point_index_by_keypoint[low], dtype=np.int64)
    points_high: Int64[ndarray, "n_high"] = np.asarray(scene.point_index_by_keypoint[high], dtype=np.int64)
    _, index_low, index_high = np.intersect1d(points_low, points_high, return_indices=True)
    return np.ascontiguousarray(np.stack([index_low, index_high], axis=1), dtype=np.uint32)


def _distance_m(plan: LoopSearchPlan, rig_pair: tuple[int, int]) -> float:
    source_xyz: Float64[ndarray, "3"] = np.asarray(plan.rig_index.world_T_rig[rig_pair[0]].translation, dtype=np.float64)
    target_xyz: Float64[ndarray, "3"] = np.asarray(plan.rig_index.world_T_rig[rig_pair[1]].translation, dtype=np.float64)
    return float(np.linalg.norm(source_xyz - target_xyz))


def test_matched_bearings_triangulate_to_the_depth_of_their_points() -> None:
    """Points 2-40 m in front of camera a, camera b 1.5 m to the side and turned 10 deg: depths come back exactly."""
    generator: np.random.Generator = np.random.default_rng(0)
    depths_m: Float64[ndarray, "n"] = generator.uniform(2.0, 40.0, 50)
    directions: Float64[ndarray, "n 3"] = np.c_[generator.uniform(-0.5, 0.5, (50, 2)), np.ones(50)]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points_a: Float64[ndarray, "n 3"] = directions * depths_m[:, None]
    b_from_a: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.0, np.sin(np.radians(5.0)), 0.0, np.cos(np.radians(5.0))])), np.array([-1.5, 0.0, 0.0])
    )
    points_b: Float64[ndarray, "n 3"] = points_a @ b_from_a.rotation.matrix().T + b_from_a.translation
    bearings_b: Float64[ndarray, "n 3"] = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    np.testing.assert_allclose(triangulated_depths_m(b_from_a, directions, bearings_b), depths_m, rtol=1e-6)


def test_a_point_behind_either_camera_has_no_depth() -> None:
    """Bearings that only meet behind the cameras give no depth."""
    b_from_a: pycolmap.Rigid3d = pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([-1.0, 0.0, 0.0]))
    point_a: Float64[ndarray, "1 3"] = np.array([[0.5, 0.0, 5.0]])
    point_b: Float64[ndarray, "1 3"] = point_a + b_from_a.translation
    bearings_a: Float64[ndarray, "2 3"] = np.vstack([point_a / np.linalg.norm(point_a), -point_a / np.linalg.norm(point_a)])
    bearings_b: Float64[ndarray, "2 3"] = np.vstack([point_b / np.linalg.norm(point_b), -point_b / np.linalg.norm(point_b)])
    depths_m: Float64[ndarray, "2"] = triangulated_depths_m(b_from_a, bearings_a, bearings_b)
    assert depths_m[0] == pytest.approx(np.linalg.norm(point_a), rel=1e-6)
    assert np.isnan(depths_m[1])


def test_the_radius_is_the_depth_quantile_of_the_scene_points_across_pairs(scene: SquareRigScene) -> None:
    """Consecutive pairs of the square scene see landmarks inside an 18 m box: q0.9 of the per-pair medians lies in
    1-12 m (2.8 m measured), and a higher quantile reaches further."""
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    sequence: tuple[int, ...] = plan.rig_index.sequence
    inliers: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = {}
    for step in range(len(sequence) - 1):
        for camera_id in plan.geometry.camera_ids:
            image_a: int | None = plan.rig_index.keyframe(sequence[step], camera_id)
            image_b: int | None = plan.rig_index.keyframe(sequence[step + 1], camera_id)
            if image_a is not None and image_b is not None:
                inliers[normalise_pair(image_a, image_b)] = _true_matches(scene, image_a, image_b)
    median_radius_m: float = covisibility_radius_m(inliers, plan.rig_index, plan.geometry, plan.keypoints, quantile=0.5)
    radius_m: float = covisibility_radius_m(inliers, plan.rig_index, plan.geometry, plan.keypoints, quantile=0.9)
    assert 1.0 < radius_m < 12.0
    assert median_radius_m < radius_m


def test_pairs_within_one_rig_frame_or_without_inliers_give_no_radius(scene: SquareRigScene) -> None:
    """A stereo pair of one rig frame says nothing about how far apart two rig frames can see the same points."""
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    rig_id: int = plan.rig_index.sequence[0]
    left: int | None = plan.rig_index.keyframe(rig_id, plan.geometry.camera_ids[0])
    right: int | None = plan.rig_index.keyframe(rig_id, plan.geometry.camera_ids[1])
    assert left is not None and right is not None
    stereo_only: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = {normalise_pair(left, right): _true_matches(scene, left, right)}
    assert covisibility_radius_m(stereo_only, plan.rig_index, plan.geometry, plan.keypoints, quantile=0.9) == 0.0
    assert covisibility_radius_m({}, plan.rig_index, plan.geometry, plan.keypoints, quantile=0.9) == 0.0


def test_the_inlier_sample_reads_every_kth_pair_with_inliers(tmp_path: Path) -> None:
    """Ten pairs with inliers and one without: a cap of 5 reads every second pair with inliers, in pair-id order."""
    database_path: Path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path) as database:
        for image_b in range(2, 13):
            geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
            geometry.inlier_matches = np.zeros((0, 2), dtype=np.uint32) if image_b == 7 else np.full((image_b, 2), image_b, dtype=np.uint32)
            database.write_two_view_geometry(1, image_b, geometry)
    assert sorted(read_inlier_sample(database_path, max_pairs=100)) == [(1, b) for b in range(2, 13) if b != 7]
    sample: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = read_inlier_sample(database_path, max_pairs=5)
    assert sorted(sample) == [(1, 2), (1, 4), (1, 6), (1, 9), (1, 11)]
    assert sample[(1, 4)].shape == (4, 2) and int(sample[(1, 4)][0, 0]) == 4


def test_the_sample_strides_over_cross_frame_pairs_only(scene: SquareRigScene, tmp_path: Path) -> None:
    """Stereo and consecutive rows interleave in pair-id order (stereo, left, right per rig frame); a stride of 3 over
    all of them lands on stereo pairs only and learns no radius, so the stride runs over the cross-frame pairs."""
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    sequence: tuple[int, ...] = plan.rig_index.sequence
    pairs: list[ImagePair] = []
    for rig_id, next_rig_id in zip(sequence[:-1], sequence[1:], strict=True):
        images: list[int | None] = [plan.rig_index.keyframe(rig_id, camera_id) for camera_id in plan.geometry.camera_ids]
        following: list[int | None] = [plan.rig_index.keyframe(next_rig_id, camera_id) for camera_id in plan.geometry.camera_ids]
        left, right = images
        assert left is not None and right is not None and following[0] is not None and following[1] is not None
        triple: list[ImagePair] = [normalise_pair(left, right), normalise_pair(left, following[0]), normalise_pair(right, following[1])]
        if all(len(_true_matches(scene, *pair)) > 0 for pair in triple):  # an empty row would shift the stride's phase
            pairs.extend(triple)
    database_path: Path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path) as database:
        for image_a, image_b in pairs:
            geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
            geometry.inlier_matches = _true_matches(scene, image_a, image_b)
            database.write_two_view_geometry(image_a, image_b, geometry)
    max_pairs: int = len(pairs) // 3  # stride 3: every pick is a stereo row
    unfiltered: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = read_inlier_sample(database_path, max_pairs)
    assert covisibility_radius_m(unfiltered, plan.rig_index, plan.geometry, plan.keypoints, quantile=0.9) == 0.0
    eligible = spans_two_rig_frames(plan.rig_index)
    sample: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = read_inlier_sample(database_path, max_pairs, eligible=eligible)
    assert 0 < len(sample) <= max_pairs
    assert all(eligible(pair) for pair in sample)
    assert covisibility_radius_m(sample, plan.rig_index, plan.geometry, plan.keypoints, quantile=0.9) > 1.0


def test_a_rig_pair_beyond_the_edge_bound_inside_the_radius_is_matched_for_the_mapper_only(scene: SquareRigScene) -> None:
    """Edge bound 1 m, radius 2.5 m: pairs 1-2.5 m apart are matched (observation pairs only) but never measured; pairs
    beyond 2.5 m are not matched at all."""
    unbounded: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, LoopClosureConfig())
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, EDGE_BOUND_CONFIG, covisibility_radius_m=2.5)
    edge_only: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, EDGE_BOUND_CONFIG)
    mapper_only: list[tuple[int, int]] = [pair for pair in unbounded.rig_pairs if 1.0 < _distance_m(unbounded, pair) <= 2.5]
    beyond: list[tuple[int, int]] = [pair for pair in unbounded.rig_pairs if _distance_m(unbounded, pair) > 2.5]
    assert len(mapper_only) >= 5 and len(beyond) >= 5, "the scene must have pairs in both bands for the test to mean anything"
    assert plan.rig_pairs == edge_only.rig_pairs, "the measured rig pairs, and so the edges, are those of the edge bound"
    assert list(plan.mapper_only_rig_pairs) == mapper_only
    assert plan.counters.rejected_by_prior_bound == len(beyond)
    assert plan.counters.matched_for_mapper_only == len(mapper_only)
    walk: set[ImagePair] = {
        normalise_pair(o.anchor, o.observed) for pair in mapper_only for o in observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, pair)
    }
    assert set(plan.image_pairs) == set(edge_only.image_pairs) | walk
    assert list(plan.image_pairs) == sorted(plan.image_pairs)


def test_a_radius_inside_the_edge_bound_changes_nothing(scene: SquareRigScene) -> None:
    """Where the radius is shorter than the edge bound (indoor scenes, every profile without far pairs) the plan is B's."""
    edge_only: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, EDGE_BOUND_CONFIG)
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, EDGE_BOUND_CONFIG, covisibility_radius_m=0.8)
    assert plan.rig_pairs == edge_only.rig_pairs
    assert plan.image_pairs == edge_only.image_pairs
    assert plan.mapper_only_rig_pairs == ()


def test_the_pairs_of_a_mapper_only_rig_pair_are_checked_against_the_prior(scene: SquareRigScene) -> None:
    """Aliased matches planted in a mapper-only pair leave the mapper; its true matches stay."""
    plan: LoopSearchPlan = plan_loop_search(scene.frames_meta, scene.database_path, scene.index, EDGE_BOUND_CONFIG, covisibility_radius_m=2.5)
    walk: list[ImagePair] = sorted(
        {
            normalise_pair(o.anchor, o.observed)
            for pair in plan.mapper_only_rig_pairs
            for o in observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, pair)
        }
    )
    pair: ImagePair = max(walk, key=lambda p: len(_true_matches(scene, *p)))
    true: UInt32[ndarray, "n_true 2"] = _true_matches(scene, *pair)
    assert len(true) >= 30
    generator: np.random.Generator = np.random.default_rng(5)
    wrong: UInt32[ndarray, "n_true 2"] = np.ascontiguousarray(
        np.stack([generator.permutation(true[:, 0]), generator.integers(0, len(scene.point_index_by_keypoint[pair[1]]), len(true))], axis=1),
        dtype=np.uint32,
    )
    planted: UInt32[ndarray, "n_inliers 2"] = np.vstack([true, wrong])

    def read(image_a: int, image_b: int) -> UInt32[ndarray, "n_inliers 2"]:
        return planted if normalise_pair(image_a, image_b) == pair else _true_matches(scene, image_a, image_b)

    overrides: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = pose_checked_inliers(plan, [], walk, read)
    kept: set[tuple[int, int]] = {tuple(row) for row in overrides[pair].tolist()}
    true_rows: set[tuple[int, int]] = {tuple(row) for row in true.tolist()}
    assert len(kept & true_rows) >= 0.95 * len(true)
    assert len(kept - true_rows) <= 0.3 * len(true)
