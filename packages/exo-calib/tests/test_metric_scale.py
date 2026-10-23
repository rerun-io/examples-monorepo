import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float64, Int64
from numpy import ndarray

from exo_calib.ops.correspondences import ObservationSet
from exo_calib.ops.metric_scale import estimate_body_rescale, estimate_metric_rescale, fuse_metric_scales, grouped_medians


def _reference(group: Int64[ndarray, " n"], values: Float64[ndarray, " n"], min_count: int) -> tuple[Float64[ndarray, " g"], int]:
    """The mask-and-median loop the vectorized helper replaced."""
    medians: list[float] = []
    n_samples: int = 0
    for group_id in np.unique(group):
        members: Float64[ndarray, " m"] = values[group == group_id]
        if members.size >= min_count:
            medians.append(float(np.median(members)))
            n_samples += members.size
    return np.asarray(medians, dtype=np.float64), n_samples


@settings(max_examples=300, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    n_samples=st.integers(0, 400),
    n_groups=st.integers(1, 40),
    min_count=st.integers(1, 6),
)
def test_grouped_medians_matches_per_group_numpy_median(seed: int, n_samples: int, n_groups: int, min_count: int) -> None:
    rng = np.random.default_rng(seed)
    group: Int64[ndarray, " n"] = rng.integers(0, n_groups, n_samples).astype(np.int64)
    # Repeated values exercise the even-count midpoint and ties in the sort.
    values: Float64[ndarray, " n"] = rng.choice(rng.uniform(0.5, 1.5, 12), n_samples)

    medians, pooled = grouped_medians(group, values, min_count)
    expected_medians, expected_pooled = _reference(group, values, min_count)

    np.testing.assert_array_equal(medians, expected_medians)
    assert pooled == expected_pooled


def test_grouped_medians_drops_small_groups_and_reports_pooled_count() -> None:
    group: Int64[ndarray, " n"] = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    values: Float64[ndarray, " n"] = np.array([3.0, 1.0, 2.0, 9.0, 9.0, 4.0, 1.0, 3.0, 2.0])

    medians, pooled = grouped_medians(group, values, min_count=3)

    np.testing.assert_array_equal(medians, np.array([2.0, 2.5]))
    assert pooled == 7


def test_grouped_medians_on_no_samples_is_empty() -> None:
    medians, pooled = grouped_medians(np.zeros(0, dtype=np.int64), np.zeros(0), min_count=3)

    assert medians.shape == (0,)
    assert pooled == 0


def test_an_observation_outside_the_frame_is_skipped_not_clamped_to_its_edge() -> None:
    """Undistorted pixels can leave the frame; at the clamped edge they would read the wrong depth (here 10x the true one)."""
    K: Float64[ndarray, "3 3"] = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    n_views, n_frames, inside, outside = 2, 6, 3, 4
    joint_x: Float64[ndarray, " j"] = np.array([0.0] * inside + [1.5] * outside)  # x 1.5 m at z 2 m: pixel 125 of 100
    points: Float64[ndarray, "n 3"] = np.column_stack((np.tile(joint_x, n_frames), np.zeros(n_frames * joint_x.size), np.full(n_frames * joint_x.size, 2.0)))
    n: int = points.shape[0]
    xy: Float64[ndarray, "n 2"] = (points @ K.T)[:, :2] / 2.0
    obs = ObservationSet(
        point_frame_idx=np.repeat(np.arange(n_frames), joint_x.size).astype(np.int64),
        point_joint_idx=np.tile(np.arange(joint_x.size), n_frames).astype(np.int64),
        obs_point_idx=np.repeat(np.arange(n), n_views).astype(np.int64),
        obs_view_idx=np.tile(np.arange(n_views), n).astype(np.int64),
        obs_xy=np.repeat(xy, n_views, axis=0),
        obs_conf=np.ones(n * n_views),
    )
    depth: Float64[ndarray, "100 100"] = np.full((100, 100), 4.0)  # metric depth twice the rig's: the rescale is 2
    depth[:, [0, 99]] = 40.0

    video_frames: Int64[ndarray, "v t"] = np.stack([np.arange(n_frames) + 10 * v for v in range(n_views)])  # camera v: its own frames
    asked: dict[int, list[int]] = {}

    def depth_at(view: int, indices: list[int]) -> list[Float64[ndarray, "100 100"]]:
        asked[view] = indices
        return [depth] * len(indices)

    scale = estimate_metric_rescale(np.repeat(np.eye(4)[None], n_views, axis=0), obs, points, video_frames, depth_at)

    assert scale == 2.0
    assert asked == {v: (np.arange(n_frames) + 10 * v).tolist() for v in range(n_views)}  # each camera's frame, not the first camera's


def arm_skeleton(frames: int, scale: float) -> tuple[Float64[ndarray, "n 3"], Int64[ndarray, " n"], Int64[ndarray, " n"]]:
    """COCO-17 shoulders, elbows and wrists with 0.260 m upper arms and 0.237 m forearms in random directions, plus hips and
    shoulders at odd distances (a seated person's hips, a broad chest), all shrunk by ``scale`` (a rig that small)."""
    rng = np.random.default_rng(7)
    rows: list[tuple[int, int, Float64[ndarray, "3"]]] = []
    for frame in range(frames):
        for shoulder, elbow, wrist, hip in ((5, 7, 9, 11), (6, 8, 10, 12)):
            s = rng.normal(size=3)
            e = s + 0.260 * (u := rng.normal(size=3)) / np.linalg.norm(u)
            w = e + 0.237 * (u := rng.normal(size=3)) / np.linalg.norm(u)
            rows += [(frame, shoulder, s), (frame, elbow, e), (frame, wrist, w), (frame, hip, s + np.array([0.0, 0.0, -0.1]))]
    return (np.array([p for *_, p in rows]) / scale, np.array([j for _, j, _ in rows]), np.array([f for f, *_ in rows]))


def test_arm_bones_give_the_factor_that_makes_a_shrunk_rig_metric() -> None:
    points_xyz, joint_idx, frame_idx = arm_skeleton(frames=20, scale=1.3)

    assert estimate_body_rescale(points_xyz, joint_idx, frame_idx) == pytest.approx(1.3, rel=1e-9)


def test_collapsed_arm_bones_are_no_evidence() -> None:
    """Joints triangulated onto one point measure no bone: they give no factor (not a division by zero), and next to real
    bones they do not count."""
    points_xyz, joint_idx, frame_idx = arm_skeleton(frames=20, scale=1.3)
    collapsed = points_xyz.copy()
    collapsed[:] = collapsed[0]
    assert estimate_body_rescale(collapsed, joint_idx, frame_idx) is None
    mixed = points_xyz.copy()
    mixed[frame_idx == 0] = mixed[frame_idx == 0][0]  # one frame's joints collapse
    assert estimate_body_rescale(mixed, joint_idx, frame_idx) == pytest.approx(1.3, rel=1e-9)


def test_too_few_arm_bones_give_no_factor() -> None:
    points_xyz, joint_idx, frame_idx = arm_skeleton(frames=2, scale=1.3)

    assert estimate_body_rescale(points_xyz, joint_idx, frame_idx) is None


def test_the_depth_and_body_scales_fuse_in_log_space_either_stands_alone_and_neither_keeps_the_rig() -> None:
    assert fuse_metric_scales(1.21, 1.0, body_weight=0.5) == pytest.approx(1.1)  # sqrt(1.21 x 1.0)
    assert fuse_metric_scales(None, 1.3, body_weight=0.5) == 1.3
    assert fuse_metric_scales(0.9, None, body_weight=0.5) == 0.9
    assert fuse_metric_scales(None, None, body_weight=0.5) == 1.0
