"""Absolute trajectory error of a TUM trajectory against KITTI odometry ground truth.

The KITTI odometry devkit stores one 3x4 row-major `[R|t]` per frame in
`poses_gt_<seq>.txt`. That matrix is `world_T_cam0`: it maps a point in the
left-camera frame of frame *i* into the left-camera frame of frame 0. Timestamps
come from `times.txt`, in seconds, and are truncated to integer microseconds
exactly the way the fork's `data/kitti/get_framemeta_file_for_KITTI.py` truncates them, so
the ground truth keys line up with every `frames_meta.json` the pipeline derives.

Both cuSFM producers write TUM trajectories in the **vehicle** frame. For KITTI
the generated `frames_meta.json` gives camera 0 an identity
`sensor_to_vehicle_transform`, so vehicle == camera 0 and the TUM poses are
directly comparable to the ground truth with no frame change. `--check-frames-meta`
re-verifies that on the actual metadata rather than trusting this paragraph.

Three alignments are reported, matching the EVO toolkit's `evo_ape` modes:

| Mode | EVO flag | Fit |
|---|---|---|
| `unaligned` | (none) | identity |
| `se3` | `-a` | Umeyama with the scale held at 1 |
| `sim3` | `-as` | Umeyama with a free scale |

The SE(3) fit is `colsfm.benchmark.align_rigid`, used read-only. The Sim(3) fit
needs no second solver: Umeyama's rotation does not depend on the scale, and the
scale a similarity fit would have chosen is what `align_rigid` already returns as
`would_be_scale`, so the Sim(3) transform is `(s, R, mu_target - s R mu_source)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from colsfm.benchmark import RigidAlignment, align_rigid, match_timestamps
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.geometry import TumPose, parse_tum_line

Positions: TypeAlias = Float64[ndarray, "n 3"]
Timestamps: TypeAlias = Int64[ndarray, "n"]
AlignmentMode: TypeAlias = Literal["unaligned", "se3", "sim3"]

ALIGNMENT_MODES: Final[tuple[AlignmentMode, ...]] = ("unaligned", "se3", "sim3")
"""Every mode reported, in the order the results table lists them."""

MICROSECONDS_PER_SECOND: Final[float] = 1e6
"""`times.txt` holds seconds; every timestamp downstream is integer microseconds."""

KITTI_POSE_COLUMNS: Final[int] = 12
"""A KITTI ground-truth row is a 3x4 matrix flattened row-major."""


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A time-stamped sequence of body positions in one world frame."""

    timestamps_microseconds: Timestamps
    """Integer microseconds, ascending."""
    world_t_body: Positions
    """Body position in the world frame, metres."""

    def __len__(self) -> int:
        """Number of poses."""
        return len(self.timestamps_microseconds)

    def take(self, indices: Int64[ndarray, "m"]) -> Trajectory:
        """Select a subset of poses.

        Args:
            indices: Int64 pose indices with shape `[m]`.

        Returns:
            A trajectory holding only those poses, in the given order.
        """
        return Trajectory(
            timestamps_microseconds=self.timestamps_microseconds[indices],
            world_t_body=self.world_t_body[indices],
        )


def read_kitti_ground_truth(*, poses_path: Path, times_path: Path) -> Trajectory:
    """Read a KITTI odometry ground-truth trajectory.

    Args:
        poses_path: `poses_gt_<seq>.txt`, one flattened 3x4 `world_T_cam0` per line.
        times_path: `times.txt`, one timestamp in seconds per line.

    Returns:
        The camera-0 trajectory in the frame-0 camera frame.

    Raises:
        ValueError: When the two files disagree in length or a row is malformed.
    """
    pose_rows: Float64[ndarray, "n 12"] = np.loadtxt(poses_path, dtype=np.float64).reshape(-1, KITTI_POSE_COLUMNS)
    seconds: Float64[ndarray, "n"] = np.loadtxt(times_path, dtype=np.float64).reshape(-1)
    if len(pose_rows) != len(seconds):
        raise ValueError(f"{poses_path} has {len(pose_rows)} rows but {times_path} has {len(seconds)}")
    world_T_cam0: Float64[ndarray, "n 3 4"] = pose_rows.reshape(-1, 3, 4)
    # Truncation, not rounding: get_framemeta_file_for_KITTI.py uses int(seconds * 1e6).
    timestamps_microseconds: Timestamps = (seconds * MICROSECONDS_PER_SECOND).astype(np.int64)
    return Trajectory(
        timestamps_microseconds=timestamps_microseconds,
        world_t_body=np.ascontiguousarray(world_T_cam0[:, :, 3], dtype=np.float64),
    )


def read_tum_trajectory(path: Path) -> Trajectory:
    """Read a TUM trajectory file written by cuSFM or cuVSLAM.

    Args:
        path: A `timestamp tx ty tz qx qy qz qw` file.

    Returns:
        The trajectory, sorted ascending by timestamp.

    Raises:
        ValueError: When the file holds no poses.
    """
    poses: list[TumPose] = [parse_tum_line(line) for line in path.read_text().splitlines() if line.strip()]
    if not poses:
        raise ValueError(f"{path} holds no poses")
    order: Int64[ndarray, "n"] = np.argsort([pose.timestamp_microseconds for pose in poses]).astype(np.int64)
    ordered: list[TumPose] = [poses[index] for index in order]
    return Trajectory(
        timestamps_microseconds=np.asarray([pose.timestamp_microseconds for pose in ordered], dtype=np.int64),
        world_t_body=np.asarray([pose.world_T_body.translation for pose in ordered], dtype=np.float64).reshape(-1, 3),
    )


def vehicle_is_camera_zero(frames_meta_path: Path) -> bool:
    """Check that the TUM vehicle frame coincides with KITTI's camera 0.

    Args:
        frames_meta_path: A `frames_meta.json` describing the KITTI stereo rig.

    Returns:
        True when camera 0's `sensor_to_vehicle_transform` is the identity, so a
        vehicle-frame TUM pose is `world_T_cam0` and needs no frame change.
    """
    frames_meta: FramesMeta = read_frames_meta(frames_meta_path)
    vehicle_T_cam = frames_meta.cameras[0].vehicle_T_cam
    translation_norm: float = float(np.linalg.norm(np.asarray(vehicle_T_cam.translation, dtype=np.float64)))
    rotation_matrix: Float64[ndarray, "3 3"] = np.asarray(vehicle_T_cam.rotation.matrix(), dtype=np.float64)
    rotation_gap: float = float(np.abs(rotation_matrix - np.eye(3)).max())
    return translation_norm < 1e-9 and rotation_gap < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


@serde
@dataclass(frozen=True, slots=True)
class AteStatistics:
    """EVO's `evo_ape --pose_relation trans_part` statistics, in metres."""

    rmse: float
    """Root mean square of the per-pose position error."""
    mean: float
    """Arithmetic mean."""
    median: float
    """Median."""
    std: float
    """Population standard deviation, as EVO reports it."""
    min: float
    """Smallest per-pose error."""
    max: float
    """Largest per-pose error."""


def ate_statistics(errors_meters: Float64[ndarray, "n"]) -> AteStatistics:
    """Summarise a vector of per-pose position errors.

    Args:
        errors_meters: Float64 per-pose errors with shape `[n]`, in metres.

    Returns:
        The six statistics EVO prints for an absolute pose error run.
    """
    return AteStatistics(
        rmse=float(np.sqrt(np.mean(errors_meters**2))),
        mean=float(np.mean(errors_meters)),
        median=float(np.median(errors_meters)),
        std=float(np.std(errors_meters)),
        min=float(np.min(errors_meters)),
        max=float(np.max(errors_meters)),
    )


def apply_alignment(*, mode: AlignmentMode, alignment: RigidAlignment, source_xyz: Positions, target_xyz: Positions) -> Positions:
    """Move estimated positions into the ground-truth frame under one alignment mode.

    Args:
        mode: Which EVO alignment to imitate.
        alignment: The Umeyama fit of `source_xyz` onto `target_xyz`.
        source_xyz: Float64 estimated positions with shape `[n, 3]`.
        target_xyz: Float64 ground-truth positions with shape `[n, 3]`.

    Returns:
        Float64 positions with shape `[n, 3]` in the ground-truth frame.
    """
    if mode == "unaligned":
        return source_xyz
    if mode == "se3":
        return alignment.apply(source_xyz)
    scale: float = alignment.would_be_scale
    scaled_R: Float64[ndarray, "3 3"] = scale * alignment.target_R_source
    translation: Float64[ndarray, "3"] = target_xyz.mean(axis=0) - scaled_R @ source_xyz.mean(axis=0)
    return source_xyz @ scaled_R.T + translation


@serde
@dataclass(frozen=True, slots=True)
class TrajectoryReport:
    """Everything measured for one estimated trajectory."""

    label: str
    """Row name in the results table."""
    path: str
    """The TUM file the poses came from."""
    pose_count: int
    """Poses in the file."""
    matched_count: int
    """Poses paired with a ground-truth frame inside the tolerance."""
    trajectory_length_meters: float
    """Ground-truth path length over the matched poses."""
    sim3_scale: float
    """Scale the Sim(3) fit chose; 1.0 means metric scale survived."""
    ate_by_alignment: dict[str, AteStatistics]
    """ATE statistics keyed by alignment mode."""


def evaluate_trajectory(
    *, label: str, path: Path, ground_truth: Trajectory, tolerance_microseconds: int
) -> TrajectoryReport:
    """Compute the ATE of one TUM trajectory under every alignment mode.

    Args:
        label: Row name for the results table.
        path: The TUM file to evaluate.
        ground_truth: The KITTI reference trajectory.
        tolerance_microseconds: Largest accepted timestamp gap when associating.

    Returns:
        The report for this trajectory.

    Raises:
        ValueError: When no pose associates with the ground truth.
    """
    estimate: Trajectory = read_tum_trajectory(path)
    estimate_indices: Int64[ndarray, "m"]
    reference_indices: Int64[ndarray, "m"]
    estimate_indices, reference_indices = match_timestamps(
        estimate.timestamps_microseconds, ground_truth.timestamps_microseconds, tolerance_microseconds
    )
    if len(estimate_indices) == 0:
        raise ValueError(f"{path}: no pose matched the ground truth within {tolerance_microseconds} us")
    source_xyz: Positions = estimate.take(estimate_indices).world_t_body
    target_xyz: Positions = ground_truth.take(reference_indices).world_t_body
    alignment: RigidAlignment = align_rigid(source_xyz, target_xyz)

    ate_by_alignment: dict[str, AteStatistics] = {}
    for mode in ALIGNMENT_MODES:
        aligned_xyz: Positions = apply_alignment(mode=mode, alignment=alignment, source_xyz=source_xyz, target_xyz=target_xyz)
        errors_meters: Float64[ndarray, "m"] = np.linalg.norm(aligned_xyz - target_xyz, axis=1)
        ate_by_alignment[mode] = ate_statistics(errors_meters)

    segment_lengths: Float64[ndarray, "m"] = np.linalg.norm(np.diff(target_xyz, axis=0), axis=1)
    return TrajectoryReport(
        label=label,
        path=str(path),
        pose_count=len(estimate),
        matched_count=len(estimate_indices),
        trajectory_length_meters=float(segment_lengths.sum()),
        sim3_scale=alignment.would_be_scale,
        ate_by_alignment=ate_by_alignment,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


@serde
@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The whole evaluation, one entry per trajectory."""

    sequence_dir: str
    """KITTI sequence directory the ground truth came from."""
    ground_truth_pose_count: int
    """Poses in `poses_gt_<seq>.txt`."""
    vehicle_is_camera_zero: bool | None
    """Whether the checked `frames_meta.json` makes the vehicle frame camera 0."""
    trajectories: list[TrajectoryReport]
    """One report per evaluated TUM file, in the order given on the command line."""


@dataclass(frozen=True)
class Config:
    """Evaluate TUM trajectories against a KITTI odometry sequence."""

    sequence_dir: Path
    """KITTI sequence directory holding `times.txt` and `poses_gt_<seq>.txt`."""
    trajectories: dict[str, Path]
    """Label to TUM file, e.g. `--trajectories blob-cusfm data/kitti/06_result/...tum`."""
    ground_truth_name: str | None = None
    """Ground-truth file name; defaults to `poses_gt_<sequence dir name>.txt`."""
    check_frames_meta: Path | None = None
    """A `frames_meta.json` to verify the vehicle frame against, if given."""
    tolerance_microseconds: int = 2000
    """Largest accepted timestamp gap when associating estimate to ground truth."""
    output_json: Path | None = None
    """Where to write the machine-readable report, if anywhere."""


def format_table(report: EvaluationReport, mode: AlignmentMode) -> str:
    """Render one alignment mode's results as a markdown table.

    Args:
        report: The finished evaluation.
        mode: The alignment mode to tabulate.

    Returns:
        A markdown table, ready to paste into a results document.
    """
    lines: list[str] = [
        f"### {mode}",
        "",
        "| Trajectory | poses | matched | RMSE | mean | median | std | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in report.trajectories:
        stats: AteStatistics = entry.ate_by_alignment[mode]
        lines.append(
            f"| {entry.label} | {entry.pose_count} | {entry.matched_count} | "
            f"{stats.rmse:.3f} | {stats.mean:.3f} | {stats.median:.3f} | "
            f"{stats.std:.3f} | {stats.min:.3f} | {stats.max:.3f} |"
        )
    return "\n".join(lines)


def main(config: Config) -> None:
    """Run the evaluation and print one markdown table per alignment mode.

    Args:
        config: Parsed command-line configuration.
    """
    ground_truth_name: str = config.ground_truth_name or f"poses_gt_{config.sequence_dir.name}.txt"
    ground_truth: Trajectory = read_kitti_ground_truth(
        poses_path=config.sequence_dir / ground_truth_name,
        times_path=config.sequence_dir / "times.txt",
    )
    frame_check: bool | None = None if config.check_frames_meta is None else vehicle_is_camera_zero(config.check_frames_meta)

    reports: list[TrajectoryReport] = [
        evaluate_trajectory(
            label=label,
            path=path,
            ground_truth=ground_truth,
            tolerance_microseconds=config.tolerance_microseconds,
        )
        for label, path in config.trajectories.items()
    ]
    report: EvaluationReport = EvaluationReport(
        sequence_dir=str(config.sequence_dir),
        ground_truth_pose_count=len(ground_truth),
        vehicle_is_camera_zero=frame_check,
        trajectories=reports,
    )

    print(f"ground truth: {len(ground_truth)} poses from {config.sequence_dir / ground_truth_name}")
    if frame_check is not None:
        print(f"vehicle frame == camera 0 in {config.check_frames_meta}: {frame_check}")
    for entry in reports:
        print(f"{entry.label}: sim3 scale {entry.sim3_scale:.6f}, gt path length {entry.trajectory_length_meters:.1f} m")
    for mode in ALIGNMENT_MODES:
        print()
        print(format_table(report, mode))

    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(to_json(report, indent=2))
        print(f"\nwrote {config.output_json}")

