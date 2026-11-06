"""VSLAM-LAB's on-disk formats: sequence ``rgb.csv`` / ``groundtruth.csv`` and an experiment folder.

Formats (VSLAM-LAB @ 1533b4b):

* ``<benchmark>/<DATASET>/<sequence>/rgb.csv``: header ``ts_rgb_0 (ns),path_rgb_0,...``; one row per frame the
  benchmark offers (TUM: only RGB frames with a depth frame within 0.02 s).
* ``groundtruth.csv`` and ``<exp_it:05d>_KeyFrameTrajectory.csv``: header
  ``ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw``, comma separated, camera-to-world, Hamilton xyzw quaternion.
* experiment folder ``<exp>/<DATASET>/<sequence>/`` with ``groundtruth.csv``, ``rgb_exp.csv`` (the frames given
  to the method) and the trajectories, plus ``<exp>/vslamlab_exp_log.csv`` (one row per run).
  ``Evaluate.evaluate_functions.evaluate_sequence`` writes ``vslamlab_evaluation/ate.csv`` there.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

GT_HEADER: tuple[str, ...] = ("ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw")
"""Header of VSLAM-LAB ``groundtruth.csv`` and of every trajectory file it evaluates."""
TRAJECTORY_FILE_NAME: str = "KeyFrameTrajectory"
"""VSLAM-LAB ``path_constants.TRAJECTORY_FILE_NAME``."""
EXP_LOG_HEADER: tuple[str, ...] = (
    "method_name",
    "dataset_name",
    "sequence_name",
    "exp_it",
    "STATUS",
    "SUCCESS",
    "TIME",
    "RAM",
    "SWAP",
    "GPU",
    "COMMENTS",
    "EVALUATION",
)
"""Columns of ``vslamlab_exp_log.csv`` (``vslamlab_utilities.create_experiment_csv_log``)."""


@dataclass(frozen=True, slots=True)
class PoseTable:
    """A timestamped camera-to-world trajectory in VSLAM-LAB's layout."""

    timestamps_ns: Int64[ndarray, "n"]
    """Timestamps, nanoseconds."""
    xyz_m: Float64[ndarray, "n 3"]
    """Camera centres in the world frame, metres."""
    quat_xyzw: Float64[ndarray, "n 4"]
    """Camera-to-world rotations, Hamilton quaternions (x, y, z, w)."""


@dataclass(frozen=True, slots=True)
class RgbRow:
    """One row of a VSLAM-LAB ``rgb.csv``: the columns MetricTrail needs, the rest kept as text."""

    timestamp_ns: int
    """``ts_rgb_0 (ns)``."""
    path: str
    """``path_rgb_0``, relative to the sequence folder (``rgb_0/<name>``)."""
    cells: tuple[str, ...]
    """The whole row as written, for copying into ``rgb_exp.csv``."""


@dataclass(frozen=True, slots=True)
class RgbCsv:
    """A VSLAM-LAB ``rgb.csv``."""

    header: tuple[str, ...]
    """Column names as written."""
    rows: list[RgbRow]
    """Rows in file order."""


def read_rgb_csv(path: Path) -> RgbCsv:
    """Read a VSLAM-LAB ``rgb.csv``.

    Raises:
        ValueError: If the header lacks ``ts_rgb_0 (ns)`` / ``path_rgb_0`` or a timestamp is not an integer.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header: tuple[str, ...] = tuple(next(reader))
        if header[:2] != ("ts_rgb_0 (ns)", "path_rgb_0"):
            raise ValueError(f"{path}: expected 'ts_rgb_0 (ns),path_rgb_0,...', got {header}")
        rows: list[RgbRow] = []
        for line, cells in enumerate(reader, start=2):
            try:
                timestamp_ns: int = int(cells[0])
            except ValueError as error:
                raise ValueError(f"{path}:{line}: timestamp {cells[0]!r} is not an integer") from error
            rows.append(RgbRow(timestamp_ns=timestamp_ns, path=cells[1], cells=tuple(cells)))
    return RgbCsv(header=header, rows=rows)


def read_pose_csv(path: Path) -> PoseTable:
    """Read ``groundtruth.csv`` or a ``*_KeyFrameTrajectory.csv`` (integer ns stamps kept exact).

    Raises:
        ValueError: If the header differs from :data:`GT_HEADER` or a row does not have 8 numbers.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header: tuple[str, ...] = tuple(next(reader))
        if header != GT_HEADER:
            raise ValueError(f"{path}: expected header {','.join(GT_HEADER)}, got {header}")
        stamps: list[int] = []
        values: list[list[float]] = []
        for line, cells in enumerate(reader, start=2):
            if len(cells) != 8:
                raise ValueError(f"{path}:{line}: expected 8 columns, got {len(cells)}")
            try:
                stamps.append(int(cells[0]))
            except ValueError as error:
                raise ValueError(f"{path}:{line}: timestamp {cells[0]!r} is not integer nanoseconds") from error
            values.append([float(cell) for cell in cells[1:]])
    table: Float64[ndarray, "n 7"] = np.asarray(values, dtype=np.float64).reshape(-1, 7)
    return PoseTable(timestamps_ns=np.asarray(stamps, dtype=np.int64), xyz_m=table[:, :3].copy(), quat_xyzw=table[:, 3:].copy())


def write_pose_csv(path: Path, poses: PoseTable) -> None:
    """Write a trajectory in VSLAM-LAB's format (sorted by time, as ``evo_metric`` expects)."""
    order: Int64[ndarray, "n"] = np.argsort(poses.timestamps_ns, kind="stable")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(GT_HEADER)
        for row in order.tolist():
            xyz: list[str] = [f"{value:.9f}" for value in poses.xyz_m[row]]
            quat: list[str] = [f"{value:.12f}" for value in poses.quat_xyzw[row]]
            writer.writerow([str(int(poses.timestamps_ns[row])), *xyz, *quat])


def write_rgb_exp_csv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Write ``rgb_exp.csv``: the frames the method was given, in rgb.csv's columns."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@dataclass(frozen=True, slots=True)
class ExpLogRow:
    """One run in ``vslamlab_exp_log.csv``, ready for evaluation (``SUCCESS`` True, ``EVALUATION`` none)."""

    method_name: str
    """Baseline name (``metrictrail``)."""
    dataset_name: str
    """VSLAM-LAB dataset name."""
    sequence_name: str
    """VSLAM-LAB sequence name."""
    exp_it: int
    """Run index; names ``<exp_it:05d>_KeyFrameTrajectory.csv``."""
    runtime_s: float
    """Wall time of the run, seconds (``TIME``)."""


def write_exp_log(path: Path, rows: list[ExpLogRow]) -> None:
    """Write ``vslamlab_exp_log.csv`` for runs made outside VSLAM-LAB's runner."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(EXP_LOG_HEADER)
        for row in rows:
            writer.writerow(
                [row.method_name, row.dataset_name, row.sequence_name, row.exp_it, "completed", True, row.runtime_s, 0.0, 0.0, 0.0, "", "none"]
            )


@dataclass(frozen=True, slots=True)
class AteCsvRow:
    """One run in VSLAM-LAB's ``vslamlab_evaluation/ate.csv`` (``evo_res --save_table`` plus frame counts)."""

    exp_it: int
    """Run index parsed from ``traj_name`` (``00000_KeyFrameTrajectory.txt``)."""
    rmse_m: float
    """evo APE translation RMSE after Sim(3) alignment, metres."""
    num_frames: int
    """Rows of ``rgb_exp.csv`` (frames given to the method)."""
    num_tracked_frames: int
    """Rows of the trajectory file."""
    num_evaluated_frames: int
    """Pairs evo associated and evaluated."""


def read_ate_csv(path: Path) -> list[AteCsvRow]:
    """Read ``vslamlab_evaluation/ate.csv``.

    Raises:
        ValueError: If a required column is missing or ``traj_name`` does not name a run.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required: set[str] = {"traj_name", "rmse", "num_frames", "num_tracked_frames", "num_evaluated_frames"}
        missing: set[str] = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows: list[AteCsvRow] = []
        for record in reader:
            name: str = record["traj_name"]
            stem: str = name.split("_", 1)[0]
            if not stem.isdigit() or TRAJECTORY_FILE_NAME not in name:
                raise ValueError(f"{path}: traj_name {name!r} does not name a run")
            rows.append(
                AteCsvRow(
                    exp_it=int(stem),
                    rmse_m=float(record["rmse"]),
                    num_frames=int(float(record["num_frames"])),
                    num_tracked_frames=int(float(record["num_tracked_frames"])),
                    num_evaluated_frames=int(float(record["num_evaluated_frames"])),
                )
            )
    return rows


def trajectory_csv_name(exp_it: int) -> str:
    """``<exp_it:05d>_KeyFrameTrajectory.csv``."""
    return f"{exp_it:05d}_{TRAJECTORY_FILE_NAME}.csv"
