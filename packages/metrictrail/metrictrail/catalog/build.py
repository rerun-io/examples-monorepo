"""Assemble a catalog layer's inputs and properties from a finished MetricTrail run and its evaluation report.

The metric numbers come from the evaluation of record (``report.json`` of :mod:`metrictrail.apis.vslamlab_eval`),
not from a second computation here (SPEC section 4); the layer's own Sim(3) fit must reproduce ``paper_ate_m`` to
1e-9 m or the build stops.
"""

import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from serde import SerdeError
from serde.json import from_json

from metrictrail.apis.vslamlab_eval import Report, RunScore
from metrictrail.artifacts import ColoredCloud, iter_depth_zip, subsample_cloud
from metrictrail.catalog.layer import (
    DEPTH_SOURCE,
    MAX_POINTS,
    POINTS_VOXEL_M,
    TIME_SOURCE,
    LayerInputs,
    LayerProperties,
    backproject_keyframes,
    matrix_to_quat_xyzw,
)
from metrictrail.frames import DisplayRaster, FrameSource, fit_for_display
from metrictrail.ground_truth import AlignedRun, GtContext, align_to_gt
from metrictrail.outputs import RunOutputs

KEYFRAME_CLOUD_STRIDE_PX: int = 4
"""Pixel stride of the keyframe cloud (SPEC section 3)."""
VALID_DEPTH_M: tuple[float, float] = (0.2, 500.0)
"""Depth kept by the keyframe cloud, metres (the run record's MoGe-2 provider valid rule)."""
ATE_TOLERANCE_M: float = 1e-9
"""Largest allowed difference between the layer's Sim(3) ATE and the report's ``paper_ate_m``."""


def read_report(path: Path) -> Report:
    """Read an evaluation ``report.json``.

    Raises:
        ValueError: If the file is not a report.
    """
    try:
        return from_json(Report, path.read_text(encoding="utf-8"))
    except SerdeError as error:
        raise ValueError(f"{path}: not a metrictrail evaluation report: {error}") from error


def find_score(report: Report, sequence: str, name: str, report_path: Path) -> RunScore:
    """The report's run of ``sequence`` / ``name`` (the report must hold exactly one).

    Raises:
        ValueError: If there is none or more than one.
    """
    matches: list[RunScore] = [score for score in report.runs if score.sequence == sequence and score.name == name]
    if len(matches) != 1:
        raise ValueError(f"{report_path}: {len(matches)} runs of {sequence}/{name}; expected one")
    return matches[0]


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Who produced the run (SPEC section 4: producer, commits, env)."""

    producer: str
    """Tree and commit that ran."""
    vipe_commit: str
    """ViPE commit."""
    repo_commit: str | None
    """Monorepo commit (our port only)."""
    env: str
    """Pixi environment of the run."""
    seed: int
    """``pipeline.slam.seed`` of the run."""


@dataclass(frozen=True, slots=True)
class SeedScores:
    """``paper_ate_m`` of every seed's run of this sequence."""

    seeds: list[int]
    """Seeds, in report order."""
    paper_ate_m: list[float]
    """The paper metric of each."""


def build_layer(
    outputs: RunOutputs,
    frames: FrameSource,
    context: GtContext,
    *,
    score: RunScore,
    report: Report,
    provenance: RunProvenance,
    raster_policy: DisplayRaster,
    tracking_stride: int,
    seed_scores: SeedScores | None,
) -> tuple[LayerInputs, LayerProperties]:
    """Layer inputs and properties of ``outputs`` on the VSLAM-LAB sequence of ``context``.

    Raises:
        ValueError: If a pose is non-finite (the paper contract), there is no keyframe depth, or the layer's Sim(3)
            ATE differs from the report's ``paper_ate_m`` by more than :data:`ATE_TOLERANCE_M`.
    """
    trajectory = outputs.trajectory
    if not np.isfinite(trajectory.world_T_cam).all():
        raise ValueError(f"{outputs.run_dir}/{outputs.name}: non-finite poses; the layer holds paper-contract runs only")
    if score.paper_ate_m is None or score.paper_se3_ate_m is None or score.paper_num_pairs is None:
        raise ValueError(f"report has no paper ATE for {score.sequence}: {score.paper_error}")
    aligned: AlignedRun = align_to_gt(trajectory.world_T_cam, trajectory.source_frame_ids, context)
    if abs(aligned.sim3.rmse_m - score.paper_ate_m) > ATE_TOLERANCE_M or abs(aligned.se3.rmse_m - score.paper_se3_ate_m) > ATE_TOLERANCE_M:
        raise ValueError(
            f"layer fit ATE {aligned.sim3.rmse_m!r} / {aligned.se3.rmse_m!r} m differs from the report's {score.paper_ate_m!r} / "
            f"{score.paper_se3_ate_m!r} m: the report is not of this run"
        )
    world_T_cam: Float64[ndarray, "n 4 4"] = trajectory.world_T_cam
    times_ns: Int64[ndarray, "n"] = context.frames.timestamps_ns[trajectory.source_frame_ids]
    keyframes: list[int] = [int(row) for row in trajectory.keyframe_rows.tolist() if 0 <= int(row) < world_T_cam.shape[0]]
    depth_path: Path = outputs.run_dir / "depth" / f"{outputs.name}.zip"
    depth_by_row: dict[int, Float32[ndarray, "h w"]] = dict(iter_depth_zip(depth_path, keyframes))
    kept: list[int] = [row for row in keyframes if row in depth_by_row]
    if not kept:
        raise ValueError(f"{depth_path}: no depth for any of the {len(keyframes)} keyframes")
    keyframe_rows: Int64[ndarray, "k"] = np.asarray(kept, dtype=np.int64)
    depths: list[Float32[ndarray, "h w"]] = [depth_by_row[row] for row in kept]
    record = outputs.record
    k_px: Float64[ndarray, "4"] = (
        np.asarray(record.bound_K_fx_fy_cx_cy, dtype=np.float64) if record is not None and record.bound_K_fx_fy_cx_cy else outputs.intrinsics[0]
    )
    raster_hw: tuple[int, int] = (int(depths[0].shape[0]), int(depths[0].shape[1]))
    voxel_m: float = POINTS_VOXEL_M[context.protocol.dataset]
    read = frames.reader()
    colors: list[UInt8[ndarray, "h w 3"]] = [fit_for_display(read(int(trajectory.source_frame_ids[row])), raster_hw, raster_policy) for row in kept]
    cloud: ColoredCloud = backproject_keyframes(
        world_T_cam, keyframe_rows, depths, colors, k_px, stride_px=KEYFRAME_CLOUD_STRIDE_PX, depth_range_m=VALID_DEPTH_M, voxel_m=voxel_m
    )
    se3: Float64[ndarray, "4 4"] = aligned.se3.gt_T_est
    sim3: Float64[ndarray, "4 4"] = aligned.sim3.gt_T_est
    per_seed: list[float] | None = seed_scores.paper_ate_m if seed_scores is not None else None
    gt_rule: str = (
        f"nearest GT within {context.protocol.max_dt_ns / 1e6:g} ms" if context.protocol.gt_match == "nearest_time" else "frame index"
    ) + (", float32 GT" if context.protocol.gt_float32 else "")
    properties = LayerProperties(
        producer=provenance.producer,
        vipe_commit=provenance.vipe_commit,
        repo_commit=provenance.repo_commit,
        env=provenance.env,
        host=socket.gethostname(),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dataset_path=str(frames.path),
        run_output_dir=str(outputs.run_dir),
        run_record=str(outputs.run_dir / "vipe" / f"{outputs.name}_metrictrail.json"),
        upstream_case=outputs.name,
        seed=provenance.seed,
        seeds=seed_scores.seeds if seed_scores is not None else None,
        slam_preset=record.tracking.preset if record is not None and record.tracking is not None else None,
        raster_policy=record.input_raster.policy if record is not None else raster_policy,
        tracking_stride=tracking_stride,
        raster_hw=[raster_hw[0], raster_hw[1]],
        intrinsics_source=record.intrinsics_source if record is not None else None,
        calibration_status=record.calibration_status if record is not None else None,
        k_fx_fy_cx_cy=[float(value) for value in k_px],
        num_frames=len(context.frames.names),
        num_tracked=int(world_T_cam.shape[0]),
        num_keyframes=len(kept),
        num_frames_not_in_base=int((~context.listed[trajectory.source_frame_ids]).sum()),
        time_source=TIME_SOURCE,
        display_alignment="se3_umeyama (run root Transform3D)",
        se3_translation=[float(value) for value in se3[:3, 3]],
        se3_quaternion_xyzw=matrix_to_quat_xyzw(se3[:3, :3]),
        alignment_gt_match=f"{gt_rule} (paper {context.protocol.dataset} rule)",
        sim3_scale=aligned.sim3.scale,
        sim3_translation=[float(value) for value in sim3[:3, 3]],
        sim3_quaternion_xyzw=matrix_to_quat_xyzw(sim3[:3, :3] / aligned.sim3.scale),
        paper_ate_m=score.paper_ate_m,
        paper_num_pairs=score.paper_num_pairs,
        se3_ate_m=score.paper_se3_ate_m,
        vslamlab_ate_m=score.vslamlab_ate_m,
        vslamlab_num_evaluated=score.vslamlab_num_evaluated_frames,
        paper_ate_m_per_seed=per_seed,
        paper_ate_m_mean=float(np.mean(per_seed)) if per_seed else None,
        paper_ate_m_std=float(np.std(per_seed, ddof=1)) if per_seed and len(per_seed) > 1 else None,
        paper_table=report.summary.paper_table,
        paper_dataset_value=report.summary.paper_value,
        paper_units=report.summary.units,
        depth_source=DEPTH_SOURCE,
        points_voxel_m=voxel_m,
        runtime_s=record.runtime_seconds if record is not None else None,
        tracking_fps=record.tracking.end_to_end_fps if record is not None and record.tracking is not None else None,
        max_memory_allocated_gib=record.runtime.max_memory_allocated_gib if record is not None and record.runtime is not None else None,
    )
    inputs = LayerInputs(
        world_T_cam=world_T_cam,
        times_ns=times_ns,
        intrinsics_px=k_px,
        raster_hw=raster_hw,
        keyframe_rows=keyframe_rows,
        keyframe_depth_m=depths,
        points=subsample_cloud(cloud, MAX_POINTS),
        gt_T_est_se3=se3,
    )
    return inputs, properties
