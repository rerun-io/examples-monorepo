"""Read a MetricTrail (ViPE ``pipeline=metrictrail``) output directory.

Upstream writes, per input video ``<name>`` (``vipe/metrictrail/pipeline.py`` ``_save_pose_artifacts`` @ ead2e00):

* ``pose/<name>.npz``: ``data`` [N, 4, 4] float64 OpenCV camera-to-world, ``inds`` = 0..N-1, ``keyframe_ids`` (DPVO's
  retained keyframes as tracked-frame rows: ``slam_map.dense_disp_frame_inds`` = ``dpvo.pg.tstamps_``),
  ``runtime`` and ``source_frame_ids`` [N] = the input-stream index of every tracked frame (frame ``i`` of a frame
  directory is its ``i``-th file in sorted order, ``vipe/streams/frame_dir_stream.py``);
* ``vipe/<name>_metrictrail.json``: the run record (``status``, ``input_raster``, ``tracking``, ...).

The npz is upstream's format, read here once at the boundary; this package never writes npz.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json


@dataclass(frozen=True, slots=True)
class MetricTrailTrajectory:
    """The tracked trajectory of one MetricTrail run on one sequence."""

    name: str
    """Upstream output name (the input directory name)."""
    world_T_cam: Float64[ndarray, "n 4 4"]
    """OpenCV camera-to-world pose of every tracked frame."""
    source_frame_ids: Int64[ndarray, "n"]
    """Input-stream index of every tracked frame (index into the sorted frame directory)."""
    runtime_s: float
    """Tracking wall time stored in the pose artifact, seconds."""
    keyframe_rows: Int64[ndarray, "k"]
    """Rows of DPVO's retained keyframes (``keyframe_ids``: DPVO timestamps, i.e. tracked-frame rows, not source ids)."""


@serde
@dataclass(frozen=True, slots=True)
class InputRasterRecord:
    """The part of the run record's ``input_raster`` the evaluation checks (third-party schema, unknown fields allowed)."""

    policy: str
    """``unchanged``, ``resize`` or ``cover_crop``."""
    frame_stride: int = 1
    """One frame in k kept by the in-the-wild frame selection (k > 1 above ``max_frames``)."""
    prepared_hw: list[int] | None = None
    """Tracking raster ``[height, width]``."""
    frames: int | None = None
    """Frames the pipeline read."""
    source_hw: list[int] | None = None
    """Decoded source frame ``[height, width]`` (``resize`` and ``cover_crop`` plans)."""
    resized_hw: list[int] | None = None
    """``[height, width]`` the source was resized to before the crop."""
    crop_left_top: list[int] | None = None
    """``[left, top]`` of the crop in the resized frame."""


@serde
@dataclass(frozen=True, slots=True)
class RuntimeRecord:
    """The run record's ``runtime`` block (third-party schema, unknown fields allowed)."""

    tracking_fps: float | None = None
    """Tracked frames per second of the kept tracking pass."""
    tracking_wall_seconds: float | None = None
    """Tracking-stage wall clock, seconds (calibration and model load excluded)."""
    max_memory_allocated_gib: float | None = None
    """CUDA peak allocation after model load, GiB."""


@serde
@dataclass(frozen=True, slots=True)
class TrackingRecord:
    """The run record's ``tracking`` block (third-party schema, unknown fields allowed)."""

    preset: str | None = None
    """DPVO backend preset."""
    tracking_seed: int | None = None
    """Seed the tracker ran with (``pipeline.slam.seed`` plus the case name)."""
    end_to_end_fps: float | None = None
    """Tracked frames per second, end to end."""


@serde
@dataclass(frozen=True, slots=True)
class RunRecord:
    """The part of ``vipe/<name>_metrictrail.json`` the evaluation checks (third-party schema, unknown fields allowed)."""

    status: str
    """``success`` for a finished run."""
    name: str
    """Output name."""
    input_raster: InputRasterRecord
    """Input raster policy and frame selection."""
    tracking_status: str
    """``tracked`` when DPVO initialised."""
    runtime_seconds: float | None = None
    """Whole-run wall clock, seconds."""
    intrinsics_source: str | None = None
    """``sharedk`` (RGB-only calibration) or the known-K source."""
    calibration_status: str | None = None
    """SharedK outcome, e.g. ``stable_gt_free``."""
    bound_K_fx_fy_cx_cy: list[float] | None = None
    """The K the tracker ran with, pixels on the tracking raster."""
    runtime: RuntimeRecord | None = None
    """Tracking runtime block."""
    tracking: TrackingRecord | None = None
    """Tracking block."""


def read_run_record(path: Path) -> RunRecord:
    """Read the run record.

    Raises:
        ValueError: If the file is not JSON or misses a checked field.
    """
    try:
        return from_json(RunRecord, path.read_text(encoding="utf-8"))
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: not a MetricTrail run record: {error}") from error


def read_trajectory(run_dir: Path, name: str) -> MetricTrailTrajectory:
    """Read ``pose/<name>.npz`` of a MetricTrail output directory.

    Raises:
        ValueError: If the archive misses ``data`` / ``source_frame_ids`` or their shapes disagree, or the
            source frame ids are not strictly increasing.
    """
    path: Path = run_dir / "pose" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        missing: list[str] = [key for key in ("data", "source_frame_ids") if key not in archive.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}; is this a pipeline=metrictrail pose artifact?")
        poses: ndarray = np.asarray(archive["data"])
        source_ids: ndarray = np.asarray(archive["source_frame_ids"])
        runtime_s: float = float(archive["runtime"]) if "runtime" in archive.files else float("nan")
        keyframe_rows: ndarray = np.asarray(archive["keyframe_ids"]) if "keyframe_ids" in archive.files else np.zeros(0, dtype=np.int64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or source_ids.shape != (poses.shape[0],):
        raise ValueError(f"{path}: data {poses.shape} and source_frame_ids {source_ids.shape} do not describe the same N frames")
    if not np.issubdtype(source_ids.dtype, np.integer) or np.any(np.diff(source_ids) <= 0) or (source_ids.size and source_ids[0] < 0):
        raise ValueError(f"{path}: source_frame_ids must be non-negative, integer and strictly increasing")
    return MetricTrailTrajectory(
        name=name,
        world_T_cam=poses.astype(np.float64),
        source_frame_ids=source_ids.astype(np.int64),
        runtime_s=runtime_s,
        keyframe_rows=keyframe_rows.astype(np.int64).reshape(-1),
    )
