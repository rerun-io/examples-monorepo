"""Synthetic sequences for unit tests only (never logged as a figure or a result)."""

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from metrictrail.eval.protocols import TUM
from metrictrail.eval.score import FrameTimes
from metrictrail.eval.vslamlab_files import PoseTable
from metrictrail.ground_truth import GtContext


@pytest.fixture
def tum_like_context(tmp_path_factory: pytest.TempPathFactory) -> tuple[GtContext, Float64[ndarray, "n 4 4"]]:
    """A 30 Hz sequence with 100 Hz GT on the same path, and its true camera-to-world poses per frame."""
    rng: np.random.Generator = np.random.default_rng(0)
    frame_ns: Int64[ndarray, "f"] = (1_305_031_100_000_000_000 + np.arange(60) * 33_333_333).astype(np.int64)
    gt_ns: Int64[ndarray, "g"] = (1_305_031_100_000_000_000 + np.arange(0, 2_100_000_000, 10_000_000)).astype(np.int64)

    def path(stamps: Int64[ndarray, "k"]) -> Float64[ndarray, "k 3"]:
        seconds: Float64[ndarray, "k"] = (stamps - stamps[0] if stamps.shape[0] else stamps).astype(np.float64) * 1e-9
        return np.column_stack([np.sin(seconds), np.cos(1.3 * seconds), 0.2 * seconds])

    gt_xyz: Float64[ndarray, "g 3"] = path(gt_ns)
    frame_xyz: Float64[ndarray, "f 3"] = np.column_stack(
        [np.sin((frame_ns - gt_ns[0]) * 1e-9), np.cos(1.3 * (frame_ns - gt_ns[0]) * 1e-9), 0.2 * (frame_ns - gt_ns[0]) * 1e-9]
    )
    world_T_cam: Float64[ndarray, "f 4 4"] = np.tile(np.eye(4), (60, 1, 1))
    world_T_cam[:, :3, :3] = Rotation.random(60, random_state=int(rng.integers(1 << 31))).as_matrix()
    world_T_cam[:, :3, 3] = frame_xyz
    frames = FrameTimes(names=[f"{stamp}.png" for stamp in frame_ns.tolist()], timestamps_ns=frame_ns, num_from_filename=0)
    gt = PoseTable(timestamps_ns=gt_ns, xyz_m=gt_xyz, quat_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (gt_ns.shape[0], 1)))
    return GtContext(sequence_dir=tmp_path_factory.mktemp("seq"), protocol=TUM, frames=frames, gt=gt, listed=np.ones(60, dtype=bool)), world_T_cam
