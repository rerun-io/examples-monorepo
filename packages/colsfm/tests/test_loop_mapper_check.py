"""Tests for `colsfm.loop_mapper_check`: which loop-stage inliers reach the mapper.

Seams under test (all public):

* `colsfm.loop_pairs.observation_pairs` — the observation walk, one enumeration for the plan and the check.
* `colsfm.loop_mapper_check.sampson_error_px` — the epipolar error of a match through the rig.
* `colsfm.loop_mapper_check.pose_checked_inliers` — the check itself, over the square scene of `test_loop_closure`.

The scene is `test_loop_closure`'s square-walking stereo rig with its one true revisit; its matcher knows every true
correspondence, so the tests can plant matches that a measured pose must reject.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64, Int64, UInt32
from numpy import ndarray
from test_loop_closure import SEEDED_SERIAL_CONFIG, SquareRigScene, make_match_function, make_square_rig_scene  # pyrefly: ignore[missing-import]

from colsfm.database import ImagePair
from colsfm.loop_closure import LoopClosureResult, LoopSearchPlan, MatchFunction, plan_loop_search, verify_loop_plan
from colsfm.loop_mapper_check import pose_checked_inliers, sampson_error_px
from colsfm.loop_pairs import ObservationPair, local_map_partners, observation_pairs, required_image_pairs
from colsfm.loop_pose import RigPoseEstimator
from colsfm.pairs import normalise_pair


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> SquareRigScene:
    """The synthetic square-walking rig, built once for the module."""
    return make_square_rig_scene(tmp_path_factory.mktemp("loop_mapper_check"))


@pytest.fixture(scope="module")
def match_fn(scene: SquareRigScene) -> MatchFunction:
    """True matches only: every planted wrong match below is the test's own."""
    return make_match_function(scene, outlier_ratio=0.0)


@pytest.fixture(scope="module")
def plan(scene: SquareRigScene) -> LoopSearchPlan:
    """The seeded plan over the scene."""
    return plan_loop_search(scene.frames_meta, scene.database_path, scene.index, SEEDED_SERIAL_CONFIG)


@pytest.fixture(scope="module")
def result(plan: LoopSearchPlan, match_fn: MatchFunction) -> LoopClosureResult:
    """The measured plan."""
    return verify_loop_plan(plan, match_fn)


def test_observation_pairs_and_required_image_pairs_agree(plan: LoopSearchPlan) -> None:
    """The plan's pair list is the local maps plus exactly the observation walk, one enumeration for both."""
    observed: set[ImagePair] = set()
    for rig_pair in plan.rig_pairs:
        walk: list[ObservationPair] = observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, rig_pair)
        observed.update(normalise_pair(o.anchor, o.observed) for o in walk)
    local: set[ImagePair] = set()
    for source_rig_id in {source for source, _ in plan.rig_pairs}:
        for camera_id in plan.geometry.camera_ids:
            anchor: int | None = plan.rig_index.keyframe(source_rig_id, camera_id)
            if anchor is not None:
                local.update(
                    normalise_pair(anchor, partner)
                    for partner in local_map_partners(plan.rig_index, plan.geometry, plan.config.rig_pose.neighbour_span, source_rig_id, camera_id)
                )
    required: tuple[ImagePair, ...] = required_image_pairs(
        plan.rig_index, plan.geometry, plan.keypoints, plan.config.rig_pose.neighbour_span, plan.rig_pairs
    )
    assert observed and set(required) == observed | local


def test_the_estimator_asks_for_exactly_the_observation_walk(plan: LoopSearchPlan, match_fn: MatchFunction) -> None:
    """`RigPoseEstimator` matches each rig pair's observation pairs in the walk's order, and the walk names their cameras."""
    asked: list[tuple[int, int]] = []

    def recording_match_fn(image_id_a: int, image_id_b: int) -> Int64[ndarray, "n_matches 2"]:
        asked.append((image_id_a, image_id_b))
        return np.asarray(match_fn(image_id_a, image_id_b), dtype=np.int64)

    estimator: RigPoseEstimator = RigPoseEstimator(plan.rig_index, plan.geometry, plan.keypoints, recording_match_fn, plan.config.rig_pose)
    camera_of: dict[int, int] = {image: camera for (_, camera), image in plan.rig_index.keyframe_by_rig_camera.items()}
    for rig_pair in plan.rig_pairs:
        walk: list[ObservationPair] = observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, rig_pair)
        asked.clear()
        # An empty local map: `measure` asks for the observation pairs only.
        estimator.measure(*rig_pair, landmarks={})
        assert walk and asked == [(o.anchor, o.observed) for o in walk]
        assert all((camera_of[o.anchor], camera_of[o.observed]) == (o.anchor_camera_id, o.observed_camera_id) for o in walk)


def _true_matches(scene: SquareRigScene, image_a: int, image_b: int) -> Int64[ndarray, "n_matches 2"]:
    """Keypoint index pairs of the two images that observe the same scene point."""
    points_a: Int64[ndarray, "n_a"] = np.asarray(scene.point_index_by_keypoint[image_a], dtype=np.int64)
    points_b: Int64[ndarray, "n_b"] = np.asarray(scene.point_index_by_keypoint[image_b], dtype=np.int64)
    common, index_a, index_b = np.intersect1d(points_a, points_b, return_indices=True)
    return np.stack([index_a, index_b], axis=1).astype(np.int64)


def _revisit_observation(plan: LoopSearchPlan, scene: SquareRigScene) -> tuple[tuple[int, int], ObservationPair]:
    """A shortlisted rig pair and one of its observation pairs that shares many true matches."""
    best: tuple[int, tuple[int, int], ObservationPair] | None = None
    for rig_pair in plan.rig_pairs:
        for observation in observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, rig_pair):
            count: int = len(_true_matches(scene, observation.anchor, observation.observed))
            if best is None or count > best[0]:
                best = (count, rig_pair, observation)
    assert best is not None and best[0] >= 30
    return best[1], best[2]


def test_sampson_error_through_the_rig_is_zero_for_exact_matches(scene: SquareRigScene, plan: LoopSearchPlan) -> None:
    """Exact projections of shared points have zero epipolar error under the true pose through the rig; a turned pose does not."""
    (source_rig_id, target_rig_id), observation = _revisit_observation(plan, scene)
    matches: Int64[ndarray, "n_matches 2"] = _true_matches(scene, observation.anchor, observation.observed)
    position_a: int = plan.geometry.position_of(observation.anchor_camera_id)
    position_b: int = plan.geometry.position_of(observation.observed_camera_id)
    true_b_from_a: pycolmap.Rigid3d = (
        plan.geometry.cams_from_rig[position_b]
        * scene.world_T_rig[target_rig_id].inverse()
        * scene.world_T_rig[source_rig_id]
        * plan.geometry.cams_from_rig[position_a].inverse()
    )
    # Exact pixels: re-project the true points instead of reading the scene's noisy keypoints.
    rays_a: Float64[ndarray, "n_matches 3"] = np.asarray(
        plan.geometry.cameras[position_a].cam_from_img(np.asarray(plan.keypoints[observation.anchor])[matches[:, 0]]), dtype=np.float64
    )
    bearings_a: Float64[ndarray, "n_matches 3"] = np.c_[rays_a, np.ones(len(rays_a))]
    depths: Float64[ndarray, "n_matches"] = np.linspace(2.0, 9.0, len(matches))
    points_b: Float64[ndarray, "n_matches 3"] = (true_b_from_a.rotation.matrix() @ (bearings_a * depths[:, None]).T).T + true_b_from_a.translation
    focal_px: float = float(plan.geometry.cameras[position_b].mean_focal_length())
    errors: Float64[ndarray, "n_matches"] = sampson_error_px(true_b_from_a, bearings_a, points_b, focal_px)
    assert np.max(errors) < 1e-6
    turned: pycolmap.Rigid3d = (
        pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.0, 0.0, np.sin(np.deg2rad(1.0)), np.cos(np.deg2rad(1.0))])), true_b_from_a.translation)
        * true_b_from_a
    )
    assert np.median(sampson_error_px(turned, bearings_a, points_b, focal_px)) > 2.0, "a 2 deg turn moves the epipolar lines by pixels"


def _planted(scene: SquareRigScene, observation: ObservationPair, seed: int = 5) -> tuple[UInt32[ndarray, "n_inliers 2"], int]:
    """The pair's true matches followed by as many random pairings, low image id first; and the number of true ones."""
    true: Int64[ndarray, "n_matches 2"] = _true_matches(scene, observation.anchor, observation.observed)
    generator: np.random.Generator = np.random.default_rng(seed)
    wrong: Int64[ndarray, "n_matches 2"] = np.stack(
        [generator.permutation(true[:, 0]), generator.integers(0, len(scene.point_index_by_keypoint[observation.observed]), len(true))], axis=1
    )
    rows: Int64[ndarray, "n_inliers 2"] = np.vstack([true, wrong])
    if observation.anchor > observation.observed:
        rows = rows[:, ::-1]
    return np.ascontiguousarray(rows, dtype=np.uint32), len(true)


def _reader(scene: SquareRigScene, planted: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]]):
    """Inliers per pair: the planted rows where given, else the true matches."""

    def read(image_a: int, image_b: int) -> UInt32[ndarray, "n_inliers 2"]:
        pair: ImagePair = normalise_pair(image_a, image_b)
        if pair in planted:
            return planted[pair]
        return np.ascontiguousarray(_true_matches(scene, *pair), dtype=np.uint32)

    return read


def _observation_image_pairs(plan: LoopSearchPlan) -> list[ImagePair]:
    return sorted(
        {
            normalise_pair(o.anchor, o.observed)
            for rig_pair in plan.rig_pairs
            for o in observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, rig_pair)
        }
    )


def _measured_observation(plan: LoopSearchPlan, scene: SquareRigScene, result: LoopClosureResult) -> ObservationPair:
    """The observation pair with the most true matches among the rig pairs that gave a verified candidate."""
    verified: set[tuple[int, int]] = {(c.hit.source_rig_id, c.hit.target_rig_id) for c in result.candidates}
    walk: list[ObservationPair] = [o for rig_pair in verified for o in observation_pairs(plan.rig_index, plan.geometry, plan.keypoints, rig_pair)]
    return max(walk, key=lambda o: len(_true_matches(scene, o.anchor, o.observed)))


def test_a_loop_pair_whose_inliers_contradict_its_measured_edge_loses_them(
    scene: SquareRigScene, plan: LoopSearchPlan, result: LoopClosureResult
) -> None:
    """Planted aliased matches in a verified rig pair's image pair leave the mapper; its true matches stay."""
    assert result.candidates
    observation: ObservationPair = _measured_observation(plan, scene, result)
    pair: ImagePair = normalise_pair(observation.anchor, observation.observed)
    rows, num_true = _planted(scene, observation)
    overrides: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = pose_checked_inliers(
        plan, result.candidates, _observation_image_pairs(plan), _reader(scene, {pair: rows})
    )
    kept: set[tuple[int, int]] = {tuple(row) for row in overrides[pair].tolist()}
    true_rows: set[tuple[int, int]] = {tuple(row) for row in rows[:num_true].tolist()}
    assert len(kept & true_rows) >= 0.95 * num_true
    assert len(kept - true_rows) <= 0.2 * num_true
    assert set(overrides) == {pair}, "the true matches of every other pair agree with their pose"


def test_the_pairs_of_a_rig_pair_without_a_candidate_are_checked_against_the_prior(
    scene: SquareRigScene, plan: LoopSearchPlan, result: LoopClosureResult
) -> None:
    """KITTI's revisits reach the map through unverified rig pairs: their true matches stay, aliased ones go."""
    observation: ObservationPair = _measured_observation(plan, scene, result)
    pair: ImagePair = normalise_pair(observation.anchor, observation.observed)
    rows, num_true = _planted(scene, observation)
    overrides: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = pose_checked_inliers(
        plan, [], _observation_image_pairs(plan), _reader(scene, {pair: rows})
    )
    kept: set[tuple[int, int]] = {tuple(row) for row in overrides[pair].tolist()}
    true_rows: set[tuple[int, int]] = {tuple(row) for row in rows[:num_true].tolist()}
    assert len(kept & true_rows) >= 0.95 * num_true
    assert len(kept - true_rows) <= 0.3 * num_true
    assert set(overrides) == {pair}


def test_a_verified_candidate_the_edge_gates_reject_keeps_its_pairs(scene: SquareRigScene, match_fn: MatchFunction) -> None:
    """The KITTI guard: edge gates at 0.0 accept no edge, yet every true loop match stays for the mapper."""
    closed: LoopSearchPlan = plan_loop_search(
        scene.frames_meta, scene.database_path, scene.index, replace(SEEDED_SERIAL_CONFIG, max_translation_m=0.0, max_rotation_deg=0.0)
    )
    measured: LoopClosureResult = verify_loop_plan(closed, match_fn)
    assert measured.edges == [] and measured.candidates
    assert pose_checked_inliers(closed, measured.candidates, _observation_image_pairs(closed), _reader(scene, {})) == {}


def test_local_map_pairs_are_never_touched(scene: SquareRigScene, plan: LoopSearchPlan, result: LoopClosureResult) -> None:
    """Only observation pairs are checked: a local-map pair keeps whatever inliers it has."""
    observed: set[ImagePair] = set(_observation_image_pairs(plan))
    local: list[ImagePair] = [pair for pair in plan.image_pairs if pair not in observed]
    assert local
    garbage: dict[ImagePair, UInt32[ndarray, "n_inliers 2"]] = {
        pair: np.ascontiguousarray(np.stack([np.arange(40), np.arange(40)[::-1]], axis=1), dtype=np.uint32) for pair in local
    }
    assert pose_checked_inliers(plan, result.candidates, list(plan.image_pairs), _reader(scene, garbage)) == {}
