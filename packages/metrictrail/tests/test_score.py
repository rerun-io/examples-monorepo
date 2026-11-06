"""Unit tests for frame-to-timestamp mapping and the two scoring protocols on synthetic sequences."""

from dataclasses import replace

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from metrictrail.eval.protocols import KITTI, SEVEN_SCENES, TUM, PaperProtocol
from metrictrail.eval.score import FrameTimes, PaperContractError, PaperScore, frame_times, paper_score, vslamlab_replica_ate, vslamlab_trajectory
from metrictrail.eval.trajectory import AteResult
from metrictrail.eval.vslamlab_files import PoseTable


def poses_from_positions(xyz: Float64[ndarray, "n 3"], rng: np.random.Generator) -> Float64[ndarray, "n 4 4"]:
    """Camera-to-world poses with random rotations at the given centres."""
    world_T_cam: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (xyz.shape[0], 1, 1))
    world_T_cam[:, :3, :3] = Rotation.random(xyz.shape[0], random_state=int(rng.integers(1 << 31))).as_matrix()
    world_T_cam[:, :3, 3] = xyz
    return world_T_cam


def similarity_of(world_T_cam: Float64[ndarray, "n 4 4"], scale: float, rng: np.random.Generator) -> Float64[ndarray, "n 4 4"]:
    """Poses expressed in another world frame related by a similarity (what an up-to-scale SLAM run returns)."""
    rotation: Float64[ndarray, "3 3"] = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
    offset: Float64[ndarray, "3"] = rng.normal(size=3)
    moved: Float64[ndarray, "n 4 4"] = world_T_cam.copy()
    moved[:, :3, :3] = rotation @ world_T_cam[:, :3, :3]
    moved[:, :3, 3] = scale * world_T_cam[:, :3, 3] @ rotation.T + offset
    return moved


def gt_table(timestamps_ns: Int64[ndarray, "n"], xyz: Float64[ndarray, "n 3"]) -> PoseTable:
    """A GT table with identity orientations."""
    return PoseTable(timestamps_ns=timestamps_ns, xyz_m=xyz, quat_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (xyz.shape[0], 1)))


def test_frame_times_uses_rgb_csv_then_vslamlab_filename_formula() -> None:
    rgb: dict[str, int] = {"1305031102.175304.png": 1305031102175303936, "1305031102.243211.png": 1305031102243211008}
    frames: FrameTimes = frame_times(["1305031102.175304.png", "1305031102.211214.png", "1305031102.243211.png"], rgb, "filename_seconds")
    # The middle frame is missing from rgb.csv; VSLAM-LAB's own rule int(float(s) * 1e9) gives its stamp.
    assert frames.timestamps_ns.tolist() == [1305031102175303936, int(float("1305031102.211214") * 1e9), 1305031102243211008]
    assert frames.num_from_filename == 1


def test_frame_times_refuses_unknown_frames_without_fallback() -> None:
    with pytest.raises(ValueError, match="not in rgb.csv"):
        frame_times(["000000.png", "000001.png"], {"000000.png": 1}, "none")


def test_frame_times_refuses_non_increasing_stamps() -> None:
    with pytest.raises(ValueError, match="increasing"):
        frame_times(["a.png", "b.png"], {"a.png": 5, "b.png": 5}, "none")


def test_vslamlab_trajectory_drops_lost_poses_and_round_trips_rotations() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    world_T_cam: Float64[ndarray, "n 4 4"] = poses_from_positions(rng.normal(size=(6, 3)), rng)
    world_T_cam[2] = np.nan
    table: PoseTable = vslamlab_trajectory(world_T_cam, np.arange(6, dtype=np.int64) * 10)
    assert table.timestamps_ns.tolist() == [0, 10, 30, 40, 50]
    kept: Float64[ndarray, "k 4 4"] = world_T_cam[[0, 1, 3, 4, 5]]
    np.testing.assert_allclose(Rotation.from_quat(table.quat_xyzw).as_matrix(), kept[:, :3, :3], atol=1e-12)
    np.testing.assert_allclose(table.xyz_m, kept[:, :3, 3])


def seven_scenes_like(rng: np.random.Generator, num_frames: int) -> tuple[FrameTimes, PoseTable, Float64[ndarray, "n 4 4"]]:
    """A 30 Hz sequence with one GT pose per frame and VSLAM-LAB's synthetic stamps."""
    stamps: Int64[ndarray, "n"] = np.asarray([int(1e12 + float(i / 30.0) * 1e9) for i in range(num_frames)], dtype=np.int64)
    xyz: Float64[ndarray, "n 3"] = np.cumsum(rng.normal(scale=0.02, size=(num_frames, 3)), axis=0)
    frames = FrameTimes(names=[f"{i:06d}.png" for i in range(num_frames)], timestamps_ns=stamps, num_from_filename=0)
    return frames, gt_table(stamps, xyz), poses_from_positions(xyz, rng)


def test_paper_score_frame_index_with_stride_is_zero_for_an_exact_similarity() -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    frames, gt, world_T_cam = seven_scenes_like(rng, 100)
    ids: Int64[ndarray, "m"] = np.arange(0, 100, 5, dtype=np.int64)
    estimate: Float64[ndarray, "m 4 4"] = similarity_of(world_T_cam[ids], 0.4, rng)
    score: PaperScore = paper_score(estimate, ids, frames, gt, SEVEN_SCENES)
    assert score.sim3.rmse_m == pytest.approx(0.0, abs=1e-6)  # float32 GT cast leaves ~1e-8 m
    assert score.sim3.num_pairs == 20
    assert score.num_without_gt == 0
    assert score.se3.rmse_m > score.sim3.rmse_m


def test_paper_score_uses_float32_gt_where_the_paper_did() -> None:
    rng: np.random.Generator = np.random.default_rng(2)
    frames, gt, world_T_cam = seven_scenes_like(rng, 60)
    far: PoseTable = replace(gt, xyz_m=gt.xyz_m + 1234.5678901)
    ids: Int64[ndarray, "m"] = np.arange(0, 60, 5, dtype=np.int64)
    estimate: Float64[ndarray, "m 4 4"] = world_T_cam[ids].copy()
    estimate[:, :3, 3] += 1234.5678901
    exact: AteResult = paper_score(estimate, ids, frames, far, replace(SEVEN_SCENES, gt_float32=False)).sim3
    cast: AteResult = paper_score(estimate, ids, frames, far, SEVEN_SCENES).sim3
    assert exact.rmse_m == pytest.approx(0.0, abs=1e-9)
    assert 1e-6 < cast.rmse_m < 1e-4  # float32 spacing at 1234 m is 1.2e-4 m


@pytest.mark.parametrize(
    ("ids", "match"),
    [
        (np.arange(0, 95, 5, dtype=np.int64), "the paper evaluates 20"),
        (np.arange(0, 100, 4, dtype=np.int64)[:20], "position 1"),
    ],
)
def test_paper_score_refuses_runs_that_skip_frames(ids: Int64[ndarray, "m"], match: str) -> None:
    rng: np.random.Generator = np.random.default_rng(3)
    frames, gt, world_T_cam = seven_scenes_like(rng, 100)
    with pytest.raises(PaperContractError, match=match):
        paper_score(world_T_cam[ids], ids, frames, gt, SEVEN_SCENES)


def test_paper_score_refuses_non_finite_poses() -> None:
    rng: np.random.Generator = np.random.default_rng(4)
    frames, gt, world_T_cam = seven_scenes_like(rng, 30)
    world_T_cam[7, 0, 3] = np.inf
    with pytest.raises(PaperContractError, match="non-finite"):
        paper_score(world_T_cam, np.arange(30, dtype=np.int64), frames, gt, KITTI)


def tum_like(rng: np.random.Generator) -> tuple[FrameTimes, PoseTable, Float64[ndarray, "f 4 4"]]:
    """100 Hz GT with a 0.2 s hole, 30 Hz frames whose true centres lie on the GT path."""
    gt_ns: Int64[ndarray, "g"] = np.arange(0, 4_000_000_000, 10_000_000, dtype=np.int64)
    gt_ns = gt_ns[(gt_ns < 1_500_000_000) | (gt_ns > 1_700_000_000)]
    gt_xyz: Float64[ndarray, "g 3"] = np.column_stack([np.sin(gt_ns * 1e-9), np.cos(gt_ns * 1.3e-9), 0.1 * gt_ns * 1e-9])
    frame_ns: Int64[ndarray, "f"] = np.arange(15_000_000, 3_900_000_000, 33_333_333, dtype=np.int64)
    frames = FrameTimes(names=[f"{t}.png" for t in frame_ns.tolist()], timestamps_ns=frame_ns, num_from_filename=0)
    true_xyz: Float64[ndarray, "f 3"] = np.column_stack([np.sin(frame_ns * 1e-9), np.cos(frame_ns * 1.3e-9), 0.1 * frame_ns * 1e-9])
    return frames, gt_table(gt_ns, gt_xyz), poses_from_positions(true_xyz, rng)


def test_paper_tum_rule_drops_frames_without_gt_within_20ms_and_vslamlab_keeps_up_to_one_frame_period() -> None:
    rng: np.random.Generator = np.random.default_rng(5)
    frames, gt, world_T_cam = tum_like(rng)
    ids: Int64[ndarray, "f"] = np.arange(len(frames.names), dtype=np.int64)
    estimate: Float64[ndarray, "f 4 4"] = similarity_of(world_T_cam, 2.0, rng)
    paper: PaperScore = paper_score(estimate, ids, frames, gt, replace(TUM, gt_float32=False))
    # Frames inside the hole that are > 20 ms from both edges have no GT for the paper.
    nearest_dt_ns: Int64[ndarray, "f"] = np.min(np.abs(frames.timestamps_ns[:, None] - gt.timestamps_ns[None, :]), axis=1)
    assert paper.num_without_gt == int(np.sum(nearest_dt_ns > 20_000_000)) > 0
    replica: AteResult = vslamlab_replica_ate(vslamlab_trajectory(estimate, frames.timestamps_ns), gt, TUM.rgb_hz)
    # evo's t_max_diff is one frame period (33.3 ms): it keeps frames up to 33 ms from the hole edges.
    kept_by_evo: int = int(np.sum(nearest_dt_ns <= 1e9 / 30.0))
    assert replica.num_pairs == kept_by_evo > paper.sim3.num_pairs
    # Frame and GT stamps differ by up to 5 ms of motion, so neither ATE is 0, and the two pairings differ.
    assert 0.0 < paper.sim3.rmse_m < 0.01
    assert replica.rmse_m != paper.sim3.rmse_m


def test_protocol_tables_cover_the_paper_sequences() -> None:
    protocols: list[PaperProtocol] = [TUM, SEVEN_SCENES, KITTI]
    assert [len(protocol.upstream_cases) for protocol in protocols] == [9, 18, 11]
    assert SEVEN_SCENES.upstream_cases["chess_seq-03"] == "chess__seq-03"
    assert TUM.upstream_cases["rgbd_dataset_freiburg1_360"] == "360"
