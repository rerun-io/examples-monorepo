"""Ground truth of a run on a VSLAM-LAB sequence: timestamps of the tracked frames and the paper's GT pairing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from metrictrail.eval.protocols import BENCHMARK_DATASETS, PROTOCOLS, PaperProtocol
from metrictrail.eval.score import FrameTimes, GtPairs, frame_times, paper_gt_pairs
from metrictrail.eval.trajectory import AteResult, aligned_ate, finite_rows, pair_errors_m
from metrictrail.eval.vslamlab_files import PoseTable, RgbCsv, read_pose_csv, read_rgb_csv


@dataclass(frozen=True, slots=True)
class GtContext:
    """What GT logging and scoring need for one sequence."""

    sequence_dir: Path
    """VSLAM-LAB sequence directory (``rgb.csv``, ``groundtruth.csv``)."""
    protocol: PaperProtocol
    """The paper's scoring rule for the benchmark."""
    frames: FrameTimes
    """Every input frame's VSLAM-LAB timestamp, in ViPE order."""
    gt: PoseTable
    """``groundtruth.csv``."""
    listed: Bool[ndarray, "f"]
    """Input frames rgb.csv lists (the others took their stamp from the file name: not in the base video)."""


def protocol_for(benchmark: str) -> PaperProtocol:
    """The paper protocol of an upstream benchmark preset name.

    Raises:
        ValueError: If no VSLAM-LAB dataset with a paper table covers the benchmark.
    """
    if benchmark not in BENCHMARK_DATASETS:
        raise ValueError(f"benchmark {benchmark!r} has no VSLAM-LAB paper protocol; choose one of {sorted(BENCHMARK_DATASETS)}")
    return PROTOCOLS[BENCHMARK_DATASETS[benchmark]]


def load_gt_context(sequence_dir: Path, protocol: PaperProtocol, frame_names: list[str]) -> GtContext:
    """Read ``rgb.csv`` and ``groundtruth.csv`` of a VSLAM-LAB sequence and time the input frames.

    ``frame_names`` are the run's input frame file names in ViPE order; rgb.csv gives each its timestamp (TUM frames it
    leaves out take VSLAM-LAB's own file-name formula).
    """
    rgb: RgbCsv = read_rgb_csv(sequence_dir / "rgb.csv")
    stamps_by_name: dict[str, int] = {Path(row.path).name: row.timestamp_ns for row in rgb.rows}
    frames: FrameTimes = frame_times(frame_names, stamps_by_name, protocol.timestamp_fallback)
    listed: Bool[ndarray, "f"] = np.array([name in stamps_by_name for name in frame_names], dtype=bool)
    return GtContext(sequence_dir=sequence_dir, protocol=protocol, frames=frames, gt=read_pose_csv(sequence_dir / "groundtruth.csv"), listed=listed)


@dataclass(frozen=True, slots=True)
class AlignedRun:
    """A trajectory paired with GT and aligned both ways (the paper's Sim(3), and SE(3) for metric scale)."""

    rows: Int64[ndarray, "m"]
    """Tracked-frame rows with GT and a finite pose."""
    gt_xyz: Float64[ndarray, "m 3"]
    """Their GT camera centres, metres."""
    sim3: AteResult
    """Sim(3) Umeyama alignment (the paper's metric)."""
    se3: AteResult
    """SE(3) alignment, scale 1 (metric-scale diagnostic)."""
    sim3_err_m: Float64[ndarray, "m"]
    """Per-pair error after Sim(3)."""
    se3_err_m: Float64[ndarray, "m"]
    """Per-pair error after SE(3)."""


def align_to_gt(world_T_cam: Float64[ndarray, "n 4 4"], source_frame_ids: Int64[ndarray, "n"], context: GtContext) -> AlignedRun:
    """Pair tracked frames with GT by the paper's rule (finite poses only) and align.

    Unlike :func:`metrictrail.eval.score.paper_score` this does not enforce the paper's all-frames contract: it is for
    figures and layers. The evaluation of record is :mod:`metrictrail.apis.vslamlab_eval`.

    Raises:
        ValueError: If fewer than 3 finite tracked frames have GT.
    """
    pairs: GtPairs = paper_gt_pairs(source_frame_ids, context.frames, context.gt, context.protocol)
    finite: Bool[ndarray, "m"] = finite_rows(world_T_cam[pairs.rows])
    rows: Int64[ndarray, "m"] = pairs.rows[finite]
    gt_xyz: Float64[ndarray, "m 3"] = pairs.gt_xyz[finite]
    if rows.shape[0] < 3:
        raise ValueError(f"only {rows.shape[0]} tracked frames with a finite pose have GT in {context.sequence_dir}")
    est_xyz: Float64[ndarray, "m 3"] = world_T_cam[rows, :3, 3]
    sim3: AteResult = aligned_ate(est_xyz, gt_xyz, "sim3")
    se3: AteResult = aligned_ate(est_xyz, gt_xyz, "se3")
    return AlignedRun(
        rows=rows,
        gt_xyz=gt_xyz,
        sim3=sim3,
        se3=se3,
        sim3_err_m=pair_errors_m(sim3, est_xyz, gt_xyz),
        se3_err_m=pair_errors_m(se3, est_xyz, gt_xyz),
    )
