"""Tests for `colsfm.loop_pose`, the metric rig-to-rig relative-pose estimator.

Seams under test (all public):

* `RigPoseEstimator.measure` — the whole measurement: local metric map, generalized PnP,
  the inlier and direction gates.
* `RigPoseEstimator.local_map` — the local map on its own.
* `RigPoseEstimator.measure_group` — one local map shared across a source rig's candidates.
* `build_rig_geometry` / `RigFrameIndex` — the rig plumbing the two estimators need.

The scene is a synthetic four-camera rig shaped like RoboCap's: a declared stereo pair
straddling the vehicle centre plus two cameras looking sideways, so the rig spans three
view directions and no single camera sees the whole scene. The rig drives along a straight
line and a later rig frame revisits an earlier one from 40 cm away.

The point of the scene is the **prior**: the source rig's own neighbours carry accurate
local odometry (2 mm, 0.05 deg — what a visual-inertial front end delivers over 100 ms),
while the revisit frame's prior is 1.5 m and 12 deg wrong, the drift a loop closure exists
to remove. An estimate that leaned on the prior for scale, or for anything else, could not
pass these tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64, Int
from loop_helpers import ProjectedPoints, matcher_from_point_indices, project_and_mask
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm import loop_pose
from colsfm.geometry import rigid3d_from_matrix
from colsfm.loop_pose import (
    AnchorLandmarks,
    Bearings,
    Keypoints,
    Matches,
    MatchFunction,
    RigFrameIndex,
    RigGeometry,
    RigLandmarks,
    RigPoseConfig,
    RigPoseEstimate,
    RigPoseEstimator,
    RigPoseOutcome,
    RigPoseRejection,
)

IMAGE_WIDTH: Final[int] = 960
"""Synthetic image width in pixels."""

IMAGE_HEIGHT: Final[int] = 600
"""Synthetic image height in pixels."""

FOCAL_LENGTH_PX: Final[float] = 400.0
"""Synthetic focal length, giving a 100 deg horizontal field of view."""

STEP_METERS: Final[float] = 0.12
"""Distance between consecutive rig frames, RoboCap's own median rig hop."""

STEREO_BASELINE_M: Final[float] = 0.086
"""Distance between the declared stereo pair, RoboCap's own front baseline."""

SOURCE_RIG_ID: Final[int] = 4
"""Rig frame the local map is built around; it has neighbours on both sides."""

TARGET_RIG_ID: Final[int] = 11
"""Rig frame that revisits the source, 40 cm to its left."""


@dataclass(frozen=True, slots=True)
class FourCameraScene:
    """A four-camera rig driving in a straight line, with one revisit."""

    geometry: RigGeometry
    """Camera ids, `cams_from_rig` and the calibrated cameras."""
    index: RigFrameIndex
    """Rig frames, their time order and their (drifting) prior poses."""
    keypoints: dict[int, Keypoints]
    """Keypoint pixels per image id."""
    point_index_by_image: dict[int, Int[ndarray, "n_keypoints"]]
    """Which 3-D point each keypoint of each image observes."""
    world_T_rig_true: dict[int, pycolmap.Rigid3d]
    """Exact rig pose per rig id, which the priors only approximate."""
    points_xyz: Float64[ndarray, "n_points 3"]
    """The true landmark positions in the world frame."""

    @property
    def source_T_target(self) -> pycolmap.Rigid3d:
        """The ground-truth relative pose the estimator has to recover."""
        return self.world_T_rig_true[SOURCE_RIG_ID].inverse() * self.world_T_rig_true[TARGET_RIG_ID]


def _looking_at(forward: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """Optical frame of a camera pointing along `forward` in the vehicle FLU frame.

    Args:
        forward: Float64 optical axis with shape `[3]`, in the vehicle frame.

    Returns:
        `vehicle_R_cam` with shape `[3, 3]`; the optical z axis is `forward` and y points
        down.
    """
    optical_z: Float64[ndarray, "3"] = np.asarray(forward, dtype=np.float64) / np.linalg.norm(forward)
    optical_x: Float64[ndarray, "3"] = np.cross(optical_z, np.array([0.0, 0.0, 1.0]))
    optical_x /= np.linalg.norm(optical_x)
    return np.stack([optical_x, np.cross(optical_z, optical_x), optical_z], axis=1)


def _extrinsics() -> dict[int, pycolmap.Rigid3d]:
    """RoboCap's rig shape: a sideways stereo pair plus a forward and a rear camera.

    Returns:
        `vehicle_T_cam` per `camera_params_id`; 0 and 1 are the declared stereo pair.
    """
    return {
        0: rigid3d_from_matrix(_looking_at(np.array([0.0, 1.0, 0.0])), np.array([STEREO_BASELINE_M / 2.0, 0.01, 0.0])),
        1: rigid3d_from_matrix(_looking_at(np.array([0.0, 1.0, 0.0])), np.array([-STEREO_BASELINE_M / 2.0, 0.01, 0.0])),
        2: rigid3d_from_matrix(_looking_at(np.array([1.0, 0.0, 0.0])), np.array([0.114, -0.086, 0.015])),
        3: rigid3d_from_matrix(_looking_at(np.array([-1.0, 0.0, 0.0])), np.array([-0.125, -0.098, 0.004])),
    }


def _trajectory() -> dict[int, pycolmap.Rigid3d]:
    """Eight frames driving along +x, then eight retracing them 40 cm to the left.

    Returns:
        The true `world_T_vehicle` per rig id, in time order.
    """
    poses: dict[int, pycolmap.Rigid3d] = {}
    for step in range(8):
        poses[step] = rigid3d_from_matrix(np.eye(3), np.array([step * STEP_METERS, 0.0, 0.0]))
    for step in range(8):
        poses[8 + step] = rigid3d_from_matrix(
            Rotation.from_euler("z", 3.0, degrees=True).as_matrix(), np.array([(3 + step) * STEP_METERS, 0.4, 0.02])
        )
    return poses


def make_four_camera_scene(pixel_noise_px: float = 0.5, seed: int = 5) -> FourCameraScene:
    """Build the synthetic rig, its images and its deliberately drifting priors.

    Args:
        pixel_noise_px: One-sigma keypoint noise in pixels.
        seed: Seed for the landmarks and the noise.

    Returns:
        The scene.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    extrinsics: dict[int, pycolmap.Rigid3d] = _extrinsics()
    trajectory: dict[int, pycolmap.Rigid3d] = _trajectory()
    cameras: dict[int, pycolmap.Camera] = {}
    for camera_id in extrinsics:
        camera: pycolmap.Camera = pycolmap.Camera(
            model="PINHOLE", width=IMAGE_WIDTH, height=IMAGE_HEIGHT, params=[FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0]
        )
        camera.camera_id = camera_id
        camera.has_prior_focal_length = True
        cameras[camera_id] = camera

    points_xyz: Float64[ndarray, "n_points 3"] = np.stack(
        [generator.uniform(-6.0, 8.0, 9000), generator.uniform(-8.0, 8.0, 9000), generator.uniform(-1.0, 2.5, 9000)], axis=1
    )

    keypoints: dict[int, Keypoints] = {}
    point_index_by_image: dict[int, Int[ndarray, "n_keypoints"]] = {}
    keyframe_by_rig_camera: dict[tuple[int, int], int] = {}
    for rig_id, world_T_rig in trajectory.items():
        for camera_id, vehicle_T_cam in extrinsics.items():
            image_id: int = rig_id * len(extrinsics) + camera_id
            keyframe_by_rig_camera[(rig_id, camera_id)] = image_id
            projected: ProjectedPoints = project_and_mask(
                (world_T_rig * vehicle_T_cam).inverse(), cameras[camera_id], points_xyz, min_depth_m=0.6, max_depth_m=25.0
            )
            chosen: Int[ndarray, "n_keypoints"] = np.flatnonzero(projected.visible)
            keypoints[image_id] = projected.pixels[chosen] + generator.standard_normal((len(chosen), 2)) * pixel_noise_px
            point_index_by_image[image_id] = chosen

    # Priors: the source rig's neighbourhood is locally accurate, the revisit has drifted.
    world_T_rig_prior: dict[int, pycolmap.Rigid3d] = {}
    drift: pycolmap.Rigid3d = rigid3d_from_matrix(Rotation.from_euler("z", 12.0, degrees=True).as_matrix(), np.array([1.1, -0.9, 0.3]))
    for rig_id, world_T_rig in trajectory.items():
        jitter: pycolmap.Rigid3d = pycolmap.Rigid3d(
            pycolmap.Rotation3d(
                (Rotation.from_quat(world_T_rig.rotation.quat) * Rotation.from_rotvec(generator.standard_normal(3) * np.deg2rad(0.05))).as_quat()
            ),
            np.asarray(world_T_rig.translation, dtype=np.float64) + generator.standard_normal(3) * 0.002,
        )
        world_T_rig_prior[rig_id] = drift * jitter if rig_id >= 8 else jitter

    geometry: RigGeometry = RigGeometry(
        camera_ids=tuple(sorted(extrinsics)),
        cams_from_rig=tuple(extrinsics[camera_id].inverse() for camera_id in sorted(extrinsics)),
        cameras=tuple(cameras[camera_id] for camera_id in sorted(extrinsics)),
        stereo_partner={0: 1, 1: 0},
    )
    index: RigFrameIndex = RigFrameIndex(
        sequence=tuple(sorted(trajectory)),
        world_T_rig=dict(world_T_rig_prior),
        keyframe_by_rig_camera=keyframe_by_rig_camera,
    )
    return FourCameraScene(
        geometry=geometry,
        index=index,
        keypoints=keypoints,
        point_index_by_image=point_index_by_image,
        world_T_rig_true=dict(trajectory),
        points_xyz=points_xyz,
    )


def make_match_function(scene: FourCameraScene, outlier_ratio: float = 0.2, seed: int = 17) -> MatchFunction:
    """Build a `match_fn` from the scene's known correspondences, with 20 % gross outliers.

    Args:
        scene: The synthetic scene.
        outlier_ratio: Wrong matches added, as a fraction of the true ones.
        seed: Seed for the outlier draw.

    Returns:
        A callable `match_fn(image_id_a, image_id_b) -> Int[ndarray, "n_matches 2"]`.
    """
    return matcher_from_point_indices(scene.point_index_by_image, outlier_ratio, seed)


def make_estimator(scene: FourCameraScene, config: RigPoseConfig | None = None, match_fn: MatchFunction | None = None) -> RigPoseEstimator:
    """Build an estimator over the scene, with the default matcher unless one is given.

    Args:
        scene: The synthetic scene.
        config: Gates and thresholds; the defaults when None.
        match_fn: The matcher; the scene's own when None.

    Returns:
        The estimator.
    """
    return RigPoseEstimator(
        index=scene.index,
        geometry=scene.geometry,
        keypoints=scene.keypoints,
        match_fn=make_match_function(scene) if match_fn is None else match_fn,
        config=RigPoseConfig() if config is None else config,
    )


@pytest.fixture(scope="module")
def scene() -> FourCameraScene:
    """The synthetic four-camera rig, built once for the module."""
    return make_four_camera_scene()


@pytest.fixture(scope="module")
def outcome(scene: FourCameraScene) -> RigPoseOutcome:
    """The estimator's answer for the one revisit in the scene."""
    return make_estimator(scene).measure(SOURCE_RIG_ID, TARGET_RIG_ID)


@pytest.fixture(scope="module")
def estimate(outcome: RigPoseOutcome) -> RigPoseEstimate:
    """The successful measurement, so the tests below need no isinstance dance."""
    assert isinstance(outcome, RigPoseEstimate), f"the revisit was rejected: {outcome}"
    return outcome


def test_the_local_map_is_metric_without_any_loop_prior(scene: FourCameraScene) -> None:
    """`local_map` puts real metres in the rig frame.

    The only metric inputs are the rig extrinsics and the source rig frame's own
    neighbourhood, so every landmark can be checked against the true point it came from,
    expressed in the true rig frame. Getting the scale wrong by a percent would show here
    long before it showed in a trajectory.
    """
    landmarks: RigLandmarks = make_estimator(scene).local_map(SOURCE_RIG_ID)
    assert set(landmarks) == {scene.index.keyframe(SOURCE_RIG_ID, camera_id) for camera_id in scene.geometry.camera_ids}
    assert sum(len(anchor) for anchor in landmarks.values()) > 200

    rig_T_world: pycolmap.Rigid3d = scene.world_T_rig_true[SOURCE_RIG_ID].inverse()
    rotation: Float64[ndarray, "3 3"] = np.asarray(rig_T_world.rotation.matrix(), dtype=np.float64)
    translation: Float64[ndarray, "3"] = np.asarray(rig_T_world.translation, dtype=np.float64)
    relative_errors: list[float] = []
    for anchor_image_id, anchor in landmarks.items():
        point_index: Int[ndarray, "n_keypoints"] = scene.point_index_by_image[anchor_image_id]
        assert np.all(np.diff(anchor.keypoint_indices) > 0), "landmark indices must be ascending for the searchsorted join"
        for keypoint_index, point_in_rig in zip(anchor.keypoint_indices, anchor.points_in_rig, strict=True):
            truth: Float64[ndarray, "3"] = rotation @ scene.points_xyz[point_index[keypoint_index]] + translation
            relative_errors.append(float(np.linalg.norm(point_in_rig - truth) / np.linalg.norm(truth)))
    assert float(np.median(relative_errors)) < 0.05, f"median landmark error {np.median(relative_errors) * 100:.1f} % of its range"


def test_the_metric_relative_rig_pose_is_recovered_without_a_prior(scene: FourCameraScene, estimate: RigPoseEstimate) -> None:
    """The revisit is measured to 1 cm and 0.2 deg, against a prior that is 1.5 m wrong.

    This is the whole point of the stage: the loop edge must be able to contradict the
    odometry about distance, which the shipped two-view-plus-prior-scale measurement cannot
    (`colsfm.loop_closure` module docstring).
    """
    truth: pycolmap.Rigid3d = scene.source_T_target
    prior: pycolmap.Rigid3d = scene.index.world_T_rig[SOURCE_RIG_ID].inverse() * scene.index.world_T_rig[TARGET_RIG_ID]
    assert float(np.linalg.norm(np.asarray(prior.translation) - np.asarray(truth.translation))) > 1.0

    translation_error_m: float = float(np.linalg.norm(np.asarray(estimate.source_T_target.translation) - np.asarray(truth.translation)))
    rotation_error_deg: float = float(
        np.rad2deg(np.linalg.norm(Rotation.from_quat((estimate.source_T_target.inverse() * truth).rotation.quat).as_rotvec()))
    )
    assert translation_error_m < 0.01, f"translation off by {translation_error_m * 1e3:.1f} mm"
    assert rotation_error_deg < 0.2, f"rotation off by {rotation_error_deg:.3f} deg"


def test_the_estimate_carries_its_own_inlier_evidence(estimate: RigPoseEstimate) -> None:
    """The result reports the inliers, observations and cameras a caller has to gate on."""
    assert estimate.num_inliers >= 30
    assert estimate.num_observations > estimate.num_inliers
    assert estimate.num_cameras == 4
    assert estimate.num_landmarks > 200


def test_a_pair_with_too_few_observations_names_that_gate(scene: FourCameraScene) -> None:
    """An inlier floor above what the pair can even offer stops before the resection."""
    rejected: RigPoseOutcome = make_estimator(scene, RigPoseConfig(min_inliers=10_000_000)).measure(SOURCE_RIG_ID, TARGET_RIG_ID)
    assert isinstance(rejected, RigPoseRejection)
    assert rejected.reason == "no_observations"


def test_a_pair_whose_resection_keeps_too_few_inliers_names_that_gate(scene: FourCameraScene, estimate: RigPoseEstimate) -> None:
    """`too_few_inliers` is a distinct outcome from `no_observations`, and it is reachable.

    The two floors are different quantities: one is how many 2-D-3-D correspondences the
    matcher and the local map produced, the other is how many of them the generalized
    resection explained. A threshold set between the pair's own two counts must therefore
    pass the first gate and fail the second — which is what pins that the second is not dead
    code (it used to be unreachable, because the observation gate always fired first).
    """
    floor: int = (estimate.num_inliers + estimate.num_observations) // 2
    assert estimate.num_inliers < floor <= estimate.num_observations
    rejected: RigPoseOutcome = make_estimator(scene, RigPoseConfig(min_inliers=floor)).measure(SOURCE_RIG_ID, TARGET_RIG_ID)
    assert isinstance(rejected, RigPoseRejection)
    assert rejected.reason == "too_few_inliers"


def test_two_rig_frames_that_share_no_view_produce_nothing(scene: FourCameraScene) -> None:
    """A matcher that finds no correspondence cannot yield an edge."""

    def no_matches(image_id_a: int, image_id_b: int) -> Matches:
        """A matcher that never matches anything."""
        del image_id_a, image_id_b
        return np.zeros((0, 2), dtype=np.int64)

    empty: RigPoseOutcome = make_estimator(scene, match_fn=no_matches).measure(SOURCE_RIG_ID, TARGET_RIG_ID)
    assert isinstance(empty, RigPoseRejection)
    assert empty.reason == "no_observations"


def test_the_direction_cross_check_can_be_switched_off(scene: FourCameraScene, estimate: RigPoseEstimate) -> None:
    """`max_direction_disagreement_deg=None` skips the epipolar solve and reports `nan`.

    With the check on, the reported disagreement is a real angle; with it off, no second
    estimate is solved at all, so there is nothing to report. Measured on RoboCap the check
    costs 66 of 921 pairs and buys 10 mm of trajectory RMSE, which is why it is on.
    """
    assert not np.isnan(estimate.direction_disagreement_deg)
    assert estimate.direction_disagreement_deg < RigPoseConfig().max_direction_disagreement_deg

    unchecked: RigPoseOutcome = make_estimator(scene, RigPoseConfig(max_direction_disagreement_deg=None)).measure(SOURCE_RIG_ID, TARGET_RIG_ID)
    assert isinstance(unchecked, RigPoseEstimate)
    assert np.isnan(unchecked.direction_disagreement_deg)


def test_a_group_measures_every_target_against_one_shared_local_map(scene: FourCameraScene, estimate: RigPoseEstimate) -> None:
    """`measure_group` answers in the caller's own order and matches the one-off measurement.

    The local map depends only on the source rig frame, so sharing it across a source's
    candidates must not change any of their answers — which is what lets the loop-closure
    stage hand one group to one worker thread.
    """
    targets: tuple[int, ...] = (TARGET_RIG_ID, TARGET_RIG_ID + 1, TARGET_RIG_ID - 1)
    outcomes: list[RigPoseOutcome] = make_estimator(scene).measure_group(SOURCE_RIG_ID, targets)
    assert len(outcomes) == len(targets)
    first: RigPoseOutcome = outcomes[0]
    assert isinstance(first, RigPoseEstimate)
    assert first.num_landmarks == estimate.num_landmarks
    np.testing.assert_allclose(first.source_T_target.translation, estimate.source_T_target.translation, atol=1e-3)


def test_every_needed_image_pair_is_requested_even_when_nothing_matches(scene: FourCameraScene) -> None:
    """The estimator asks the matcher for its pairs before it gates on anything.

    The pipeline discovers which pairs to match by running the stage once with a matcher
    that returns nothing (`colsfm.pipeline._find_loop_edges`), so a pair the estimator only
    asks for *after* a gate would never be matched and the stage would be empty in
    production while passing every test.
    """
    requested: set[tuple[int, int]] = set()

    def record(image_id_a: int, image_id_b: int) -> Matches:
        """Record the pair and return no matches."""
        requested.add((min(image_id_a, image_id_b), max(image_id_a, image_id_b)))
        return np.zeros((0, 2), dtype=np.int64)

    make_estimator(scene, match_fn=record).measure(SOURCE_RIG_ID, TARGET_RIG_ID)

    matcher: MatchFunction = make_match_function(scene)
    with_matches: set[tuple[int, int]] = set()

    def record_and_match(image_id_a: int, image_id_b: int) -> Matches:
        """Record the pair and return the real matches."""
        with_matches.add((min(image_id_a, image_id_b), max(image_id_a, image_id_b)))
        return matcher(image_id_a, image_id_b)

    make_estimator(scene, match_fn=record_and_match).measure(SOURCE_RIG_ID, TARGET_RIG_ID)
    assert requested == with_matches


def test_the_direction_cross_check_reuses_the_pairs_the_observations_already_matched(scene: FourCameraScene) -> None:
    """The epipolar cross-check adds no image pair and no second match call per pair.

    It used to re-run `match_fn` over the same-camera pairs the observation pass had just
    read. Counting the calls pins that it now reads them off the observation pass instead:
    with the check on and off, the matcher is asked for exactly the same pairs the same
    number of times.
    """
    calls: dict[tuple[int, int], int] = {}
    matcher: MatchFunction = make_match_function(scene)

    def counting(image_id_a: int, image_id_b: int) -> Matches:
        """Count how often each ordered pair is asked for."""
        calls[(image_id_a, image_id_b)] = calls.get((image_id_a, image_id_b), 0) + 1
        return matcher(image_id_a, image_id_b)

    make_estimator(scene, match_fn=counting).measure(SOURCE_RIG_ID, TARGET_RIG_ID)
    with_check: dict[tuple[int, int], int] = dict(calls)
    calls.clear()
    make_estimator(scene, RigPoseConfig(max_direction_disagreement_deg=None), match_fn=counting).measure(SOURCE_RIG_ID, TARGET_RIG_ID)
    assert with_check == calls
    assert max(with_check.values()) == 1


def test_the_neighbour_span_reaches_the_frames_it_says_it_does(scene: FourCameraScene) -> None:
    """`neighbour_span` decides how much of the local trajectory the map triangulates from."""
    assert scene.index.neighbours(SOURCE_RIG_ID, span=1) == (3, 5)
    assert scene.index.neighbours(SOURCE_RIG_ID, span=2) == (2, 3, 5, 6)
    assert scene.index.neighbours(0, span=2) == (1, 2)


def test_an_anchors_landmarks_are_packed_for_the_join(scene: FourCameraScene) -> None:
    """`AnchorLandmarks` carries parallel, sorted arrays rather than a dict per point.

    The join in the observation pass is a `searchsorted` against these, so they are packed
    once at triangulation instead of re-sorted for every candidate the source rig has.
    """
    landmarks: RigLandmarks = make_estimator(scene).local_map(SOURCE_RIG_ID)
    anchor: AnchorLandmarks = landmarks[scene.index.keyframe(SOURCE_RIG_ID, 0)]
    assert anchor.keypoint_indices.shape == (len(anchor),)
    assert anchor.points_in_rig.shape == (len(anchor), 3)
    assert np.all(np.diff(anchor.keypoint_indices) > 0)


def test_the_bearing_memo_is_bounded_and_still_answers(
    scene: FourCameraScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memo is an LRU, not a growing dict, and evicting from it changes no bearing.

    An unbounded memo held one float64 `[num_keypoints, 3]` array per image for the
    whole stage — 4528 of them on RoboCap — for a measurement that only ever reads a
    source rig frame, its temporal neighbours and one target. The cache is a speed
    device, so the only thing that must survive a smaller one is the answer.
    """
    wanted: dict[int, Bearings] = {
        image_id: make_estimator(scene).bearings(image_id) for image_id in sorted(scene.keypoints)
    }
    monkeypatch.setattr(loop_pose, "BEARING_CACHE_IMAGES", 2)
    estimator: RigPoseEstimator = make_estimator(scene)

    for image_id, expected in wanted.items():
        np.testing.assert_array_equal(estimator.bearings(image_id), expected)
    assert len(estimator._bearings) <= 2, "the memo holds the working set, not the sequence"
    # Re-asking after every eviction has to give the same arrays, in either direction.
    for image_id, expected in reversed(list(wanted.items())):
        np.testing.assert_array_equal(estimator.bearings(image_id), expected)
