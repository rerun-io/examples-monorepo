"""Assemble a catalog layer's inputs and properties from a finished MetricTrail run and its evaluation report.

The metric numbers come from the evaluation of record (``report.json`` of :mod:`metrictrail.apis.vslamlab_eval`),
not from a second computation here (SPEC section 4); the layer's own Sim(3) fit must reproduce ``paper_ate_m`` to
1e-9 m or the build stops.
"""

import socket
import time
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.chunk import RrdReader
from serde import SerdeError
from serde.json import from_json

from metrictrail import entities
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
CLOUD_TOLERANCE_M: float = 1e-4
"""Largest allowed distance between the capture's SE(3)- and Sim(3)-aligned fused clouds mapped back to the estimate."""
NORMALS_SOURCE: str = (
    "live capture <case>.rrd: cam/normals (MoGe-2 normals, RGB (n+1)/2; an H.264 video of every tracked row, decoded and kept at "
    "the run's keyframe rows as JPEG; an older capture's JPEGs, logged where DPVO's keyframe count grew, as they are) at that "
    "row's video_time; metric/cloud_normals (upstream fused cloud, normals in GT axes by the SE(3) rotation) mapped back to the "
    "estimate world"
)
"""What a layer's normals entities hold (``normals_source``)."""


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
    """Monorepo commit the run used (our port only)."""
    repo_changes: list[str] | None
    """Uncommitted changes of the run's worktree (our port only); None when clean."""
    env: str
    """Pixi environment of the run."""
    seed: int
    """``pipeline.slam.seed`` of the run."""
    run_commit: str | None = None
    """The run's own commit when ``repo_commit`` is the branch commit its code was rebased to."""


@dataclass(frozen=True, slots=True)
class CaptureNormals:
    """The surface normals a run's live capture holds."""

    images_jpeg: dict[int, bytes]
    """Tracked row -> JPEG of that frame's normals (RGB ``(n + 1) / 2``)."""
    cloud: ColoredCloud
    """The fused cloud in the estimate's world, coloured by its normals in GT axes."""


def decode_normals_video(samples: dict[int, bytes], keep: Set[int]) -> dict[int, bytes]:
    """JPEGs (quality 92, as the captures logged them before) of the ``keep`` rows of an H.264 normals video, from its
    samples by tracked row. The video has no B-frames, so the decoded frames come in row order."""
    rows: list[int] = sorted(samples)
    decoder: av.video.codeccontext.VideoCodecContext = av.CodecContext.create("h264", "r")
    frames: list[av.VideoFrame] = [frame for row in rows for frame in decoder.decode(av.Packet(samples[row]))]
    frames += decoder.decode(None)
    if len(frames) != len(rows):
        raise ValueError(f"the normals video decoded to {len(frames)} frames for {len(rows)} samples")
    images: dict[int, bytes] = {}
    for row, frame in zip(rows, frames, strict=True):
        if row in keep:
            ok, jpeg = cv2.imencode(".jpg", frame.to_ndarray(format="bgr24"), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise ValueError(f"JPEG encoding of the normals of row {row} failed")
            images[row] = jpeg.tobytes()
    return images


def read_capture_normals(
    rrd: Path, gt_T_est_se3: Float64[ndarray, "4 4"], gt_T_est_sim3: Float64[ndarray, "4 4"], video_rows: Set[int]
) -> CaptureNormals:
    """``cam/normals`` (on the ``frame`` timeline, = tracked row) and ``metric/cloud_normals`` of a live capture.

    A current capture holds the normals as an H.264 video of every tracked row: the ``video_rows`` (the run's keyframes)
    are decoded into JPEGs. An older capture holds JPEGs of the rows where DPVO's keyframe count grew: all are kept.

    The cloud is logged SE(3)-aligned to GT; it is mapped back to the estimate's world here, and must agree with the
    capture's Sim(3)-aligned ``world/cloud_normals`` mapped back the same way (so the capture is of this run).

    Raises:
        ValueError: If an entity is missing or the two clouds disagree by more than :data:`CLOUD_TOLERANCE_M`.
    """
    images: dict[int, bytes] = {}
    video_samples: dict[int, bytes] = {}
    clouds: dict[str, Float32[ndarray, "c 3"]] = {}
    colors: UInt8[ndarray, "c 3"] | None = None
    for chunk in RrdReader(rrd).store().stream():
        path: str = chunk.entity_path.lstrip("/")
        if path == entities.CAM_NORMALS:
            batch = chunk.to_record_batch()
            names: list[str] = batch.schema.names
            if "EncodedImage:blob" in names:
                for frame, blob in zip(batch.column(entities.FRAME_TIMELINE).to_pylist(), batch.column("EncodedImage:blob").to_pylist(), strict=True):
                    images[int(frame)] = bytes(blob[0])
            elif "VideoStream:sample" in names and entities.FRAME_TIMELINE in names:  # the static codec chunk has neither
                for frame, sample in zip(batch.column(entities.FRAME_TIMELINE).to_pylist(), batch.column("VideoStream:sample").to_pylist(), strict=True):
                    video_samples[int(frame)] = bytes(sample[0])
        elif path in (entities.METRIC_CLOUD_NORMALS, entities.WORLD_CLOUD_NORMALS):
            batch = chunk.to_record_batch()
            clouds[path] = np.asarray(batch.column("Points3D:positions").to_pylist()[-1], dtype=np.float32)
            if path == entities.METRIC_CLOUD_NORMALS:
                packed: Int64[ndarray, "c"] = np.asarray(batch.column("Points3D:colors").to_pylist()[-1], dtype=np.int64)
                colors = np.stack([(packed >> 24) & 255, (packed >> 16) & 255, (packed >> 8) & 255], axis=1).astype(np.uint8)
    if video_samples:
        images.update(decode_normals_video(video_samples, video_rows))
    if not images or colors is None or len(clouds) != 2:
        raise ValueError(f"{rrd}: needs cam/normals, metric/cloud_normals and world/cloud_normals; found {len(images)} images, {sorted(clouds)}")
    from_se3: Float64[ndarray, "c 3"] = (clouds[entities.METRIC_CLOUD_NORMALS].astype(np.float64) - gt_T_est_se3[:3, 3]) @ gt_T_est_se3[:3, :3]
    sim3_inverse: Float64[ndarray, "4 4"] = np.linalg.inv(gt_T_est_sim3)
    from_sim3: Float64[ndarray, "c 3"] = clouds[entities.WORLD_CLOUD_NORMALS].astype(np.float64) @ sim3_inverse[:3, :3].T + sim3_inverse[:3, 3]
    if from_se3.shape != from_sim3.shape or float(np.abs(from_se3 - from_sim3).max()) > CLOUD_TOLERANCE_M:
        raise ValueError(f"{rrd}: its SE(3) and Sim(3) clouds do not map back to one estimate cloud; the capture is not of this alignment")
    return CaptureNormals(images_jpeg=images, cloud=ColoredCloud(points_xyz=from_se3.astype(np.float32), colors_rgb=colors))


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
    depth_source: str = DEPTH_SOURCE,
    capture_rrd: Path | None = None,
) -> tuple[LayerInputs, LayerProperties]:
    """Layer inputs and properties of ``outputs`` on the VSLAM-LAB sequence of ``context``.

    ``capture_rrd`` (the run's live capture) adds its surface normals (:func:`read_capture_normals`).

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
    normals: CaptureNormals | None = read_capture_normals(capture_rrd, se3, sim3, set(kept)) if capture_rrd is not None else None
    per_seed: list[float] | None = seed_scores.paper_ate_m if seed_scores is not None else None
    gt_rule: str = (
        f"nearest GT within {context.protocol.max_dt_ns / 1e6:g} ms" if context.protocol.gt_match == "nearest_time" else "frame index"
    ) + (", float32 GT" if context.protocol.gt_float32 else "")
    properties = LayerProperties(
        producer=provenance.producer,
        vipe_commit=provenance.vipe_commit,
        repo_commit=provenance.repo_commit,
        repo_changes=provenance.repo_changes,
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
        depth_source=depth_source,
        points_voxel_m=voxel_m,
        runtime_s=record.runtime_seconds if record is not None else None,
        tracking_fps=record.tracking.end_to_end_fps if record is not None and record.tracking is not None else None,
        max_memory_allocated_gib=record.runtime.max_memory_allocated_gib if record is not None and record.runtime is not None else None,
        run_commit=provenance.run_commit,
        normals_source=f"{NORMALS_SOURCE} ({capture_rrd})" if normals is not None else None,
        num_normals_images=len(normals.images_jpeg) if normals is not None else None,
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
        normals_jpeg=normals.images_jpeg if normals is not None else {},
        points_normals=subsample_cloud(normals.cloud, MAX_POINTS) if normals is not None else None,
    )
    return inputs, properties
