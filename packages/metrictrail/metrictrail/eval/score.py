"""Score one MetricTrail trajectory under the paper's protocol and under VSLAM-LAB's, from in-memory data.

Everything here is pure: the caller reads the files (``vipe_output``, ``vslamlab_files``) and passes arrays.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from metrictrail.eval.protocols import PaperProtocol, TimestampFallback
from metrictrail.eval.trajectory import AteResult, MatchedIndices, aligned_ate, evo_matching_indices, finite_rows, nearest_within
from metrictrail.eval.vslamlab_files import PoseTable


class PaperContractError(ValueError):
    """The prediction does not satisfy the paper evaluator's input contract (all frames, finite poses)."""


@dataclass(frozen=True, slots=True)
class FrameTimes:
    """Timestamp of every frame of a MetricTrail input directory, in ViPE's frame order (sorted file names)."""

    names: list[str]
    """Frame file names, sorted (``vipe/streams/frame_dir_stream.py`` order)."""
    timestamps_ns: Int64[ndarray, "f"]
    """VSLAM-LAB timestamp of each frame, nanoseconds."""
    num_from_filename: int
    """Frames absent from rgb.csv whose timestamp came from the file name."""


def frame_times(names: list[str], rgb_timestamps_ns: dict[str, int], fallback: TimestampFallback) -> FrameTimes:
    """Map frame file names to VSLAM-LAB timestamps.

    A frame listed in rgb.csv takes its ``ts_rgb_0 (ns)``. A frame rgb.csv does not list (TUM RGB frames with no
    depth frame within 0.02 s) takes ``int(float(stem) * 1e9)`` under ``filename_seconds``, which is how
    VSLAM-LAB's ``dataset_rgbdtum.py`` makes the rgb.csv stamps from the same names.

    Args:
        names: Frame file names in ViPE order.
        rgb_timestamps_ns: rgb.csv ``basename(path_rgb_0) -> ts_rgb_0 (ns)``.
        fallback: Rule for names missing from rgb.csv.

    Returns:
        The names, their timestamps and how many came from the file name.

    Raises:
        ValueError: If a name is missing and ``fallback`` is ``none``, or the timestamps are not strictly increasing.
    """
    stamps: list[int] = []
    num_from_filename: int = 0
    for name in names:
        if name in rgb_timestamps_ns:
            stamps.append(rgb_timestamps_ns[name])
            continue
        if fallback == "none":
            raise ValueError(f"frame {name!r} is not in rgb.csv and the protocol has no timestamp fallback")
        stamps.append(int(float(PurePosixPath(name).stem) * 1e9))
        num_from_filename += 1
    timestamps_ns: Int64[ndarray, "f"] = np.asarray(stamps, dtype=np.int64)
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("frame timestamps are not strictly increasing in frame order")
    return FrameTimes(names=list(names), timestamps_ns=timestamps_ns, num_from_filename=num_from_filename)


def vslamlab_trajectory(world_T_cam: Float64[ndarray, "n 4 4"], timestamps_ns: Int64[ndarray, "n"]) -> PoseTable:
    """The VSLAM-LAB trajectory of the finite tracked poses (camera-to-world, xyzw quaternion).

    Non-finite poses are left out, as a method that lost track writes no row; VSLAM-LAB then evaluates the rows that
    remain and reports their count.
    """
    keep: Bool[ndarray, "n"] = finite_rows(world_T_cam)
    kept: Float64[ndarray, "k 4 4"] = world_T_cam[keep]
    quat_xyzw: Float64[ndarray, "k 4"] = Rotation.from_matrix(kept[:, :3, :3]).as_quat() if kept.shape[0] else np.zeros((0, 4))
    return PoseTable(timestamps_ns=timestamps_ns[keep], xyz_m=kept[:, :3, 3].copy(), quat_xyzw=np.asarray(quat_xyzw, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class PaperScore:
    """ATE of one run under the paper's protocol."""

    sim3: AteResult
    """The paper's number: Sim(3) Umeyama alignment, translation RMSE."""
    se3: AteResult
    """Metric-scale diagnostic (no scale), not reported in the paper."""
    num_frames: int
    """Frames in the input directory."""
    num_tracked: int
    """Poses in the prediction."""
    num_without_gt: int
    """Tracked frames the paper's rule gives no GT pose (dropped, as the paper's evaluator drops NaN GT rows)."""


def paper_score(
    world_T_cam: Float64[ndarray, "n 4 4"], source_frame_ids: Int64[ndarray, "n"], frames: FrameTimes, gt: PoseTable, protocol: PaperProtocol
) -> PaperScore:
    """Score a trajectory the way the paper's evaluator does.

    Contract (``metrictrail_eval`` tum.py / sevenscenes.py / kitti.py): a finite pose for every source frame
    ``0, k, 2k, ...`` of the input directory. GT per tracked frame: the GT row with the frame's index
    (``frame_index``: 7-Scenes, KITTI) or the nearest GT stamp within ``protocol.max_dt_ns`` (``nearest_time``:
    TUM). Frames without GT are dropped. GT positions pass through float32 where the paper's GT did.

    Raises:
        PaperContractError: If the tracked frames are not exactly ``0, k, 2k, ...`` or a pose is non-finite.
    """
    expected_ids: Int64[ndarray, "e"] = np.arange(len(frames.names), dtype=np.int64)[:: protocol.frame_stride]
    if not np.array_equal(source_frame_ids, expected_ids):
        raise PaperContractError(
            f"tracked {source_frame_ids.shape[0]} frames, the paper evaluates {expected_ids.shape[0]} "
            f"(every {protocol.frame_stride}th of {len(frames.names)}); first differing ids: "
            f"{_first_difference(source_frame_ids, expected_ids)}"
        )
    if not finite_rows(world_T_cam).all():
        raise PaperContractError(f"{int((~finite_rows(world_T_cam)).sum())} tracked poses are non-finite")
    pairs: GtPairs = paper_gt_pairs(source_frame_ids, frames, gt, protocol)
    est_xyz: Float64[ndarray, "m 3"] = world_T_cam[pairs.rows, :3, 3]
    return PaperScore(
        sim3=aligned_ate(est_xyz, pairs.gt_xyz, "sim3"),
        se3=aligned_ate(est_xyz, pairs.gt_xyz, "se3"),
        num_frames=len(frames.names),
        num_tracked=int(source_frame_ids.shape[0]),
        num_without_gt=int(source_frame_ids.shape[0] - pairs.rows.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class GtPairs:
    """Tracked frames that have a GT pose under the paper's rule, and those GT positions."""

    rows: Int64[ndarray, "m"]
    """Row of each pair in the tracked trajectory (index into ``source_frame_ids``)."""
    gt_xyz: Float64[ndarray, "m 3"]
    """GT camera centre of each pair, metres (through float32 where the paper's GT was)."""


def paper_gt_pairs(source_frame_ids: Int64[ndarray, "n"], frames: FrameTimes, gt: PoseTable, protocol: PaperProtocol) -> GtPairs:
    """Pair tracked frames with GT by the paper's rule, without the paper's all-frames contract.

    GT per tracked frame: the GT row with the frame's index (``frame_index``: 7-Scenes, KITTI) or the nearest GT stamp
    within ``protocol.max_dt_ns`` (``nearest_time``: TUM). Frames without GT are left out.
    """
    gt_rows: Int64[ndarray, "n"]
    if protocol.gt_match == "frame_index":
        gt_rows = np.where(source_frame_ids < gt.timestamps_ns.shape[0], source_frame_ids, -1).astype(np.int64)
    else:
        gt_rows = nearest_within(frames.timestamps_ns[source_frame_ids], gt.timestamps_ns, protocol.max_dt_ns)
    has_gt: Bool[ndarray, "n"] = gt_rows >= 0
    gt_xyz: Float64[ndarray, "m 3"] = gt.xyz_m[gt_rows[has_gt]]
    if protocol.gt_float32:
        gt_xyz = gt_xyz.astype(np.float32).astype(np.float64)
    return GtPairs(rows=np.flatnonzero(has_gt).astype(np.int64), gt_xyz=gt_xyz)


def vslamlab_replica_ate(trajectory: PoseTable, gt: PoseTable, rgb_hz: float) -> AteResult:
    """VSLAM-LAB's number computed in-process: evo association on float64 ns stamps with ``max_diff = 1e9 / rgb_hz``,
    then Sim(3) Umeyama and translation RMSE (``evo_ape tum gt traj -va -as --t_max_diff``).

    A cross-check of the subprocess run of VSLAM-LAB's own code, not a replacement for it.
    """
    matched: MatchedIndices = evo_matching_indices(gt.timestamps_ns.astype(np.float64), trajectory.timestamps_ns.astype(np.float64), 1e9 / rgb_hz)
    return aligned_ate(trajectory.xyz_m[matched.est], gt.xyz_m[matched.gt], "sim3")


def _first_difference(actual: Int64[ndarray, "a"], expected: Int64[ndarray, "e"]) -> str:
    """Text naming the first index where two id lists differ."""
    common: int = min(actual.shape[0], expected.shape[0])
    differs: Int64[ndarray, "d"] = np.flatnonzero(actual[:common] != expected[:common])
    index: int = int(differs[0]) if differs.size else common
    return f"position {index}: got {actual[index : index + 3].tolist()}, expected {expected[index : index + 3].tolist()}"
