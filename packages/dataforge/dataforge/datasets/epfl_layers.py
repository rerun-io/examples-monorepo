"""EPFL layer writers using shared camera, video, keypoint and mesh contracts."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters

from dataforge import hands, schema, writing
from dataforge.datasets.epfl_actions import Segment, action_rows
from dataforge.datasets.epfl_source import (
    CAMERA_NAMES,
    CLOCK_SOURCE,
    EGO_RIG,
    EXO_CAMERAS,
    SOURCE_REVISION,
    EgoCamera,
    ExoCamera,
    Fit,
    FitSpec,
    PoseRow,
    holo_batches,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    annotation_context,
    frame_index_column,
    log_camera_node,
    log_dense_pose_track,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.timing import SequenceTimer


def write_base(
    recording: rr.RecordingStream,
    root: Path,
    cameras: dict[str, ExoCamera],
    ego_camera: EgoCamera,
    times: Int64[ndarray, "n"],
    source_count: int,
    identity: SequenceIdentity,
    actions: dict[str, list[Segment]],
    timer: SequenceTimer,
    work_root: Path,
) -> None:
    """Remux AV1 and log calibration and dense HoloLens poses on the device clock."""
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
    rr.log("/", annotation_context(), static=True, recording=recording)
    frames = np.arange(len(times), dtype=np.int64)
    work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="epfl-", dir=work_root) as work:

        def write_camera(rig: int, name: str, calibration: ExoCamera | EgoCamera, transform: Float64[ndarray, "4 4"]) -> None:
            width, height = calibration.size
            camera = PinholeParameters(
                name=name,
                extrinsics=Extrinsics(world_R_cam=transform[:3, :3], world_t_cam=transform[:3, 3]),
                intrinsics=Intrinsics.from_focal_principal_point(
                    camera_conventions="RDF",
                    fl_x=float(calibration.K[0, 0]),
                    fl_y=float(calibration.K[1, 1]),
                    cx=float(calibration.K[0, 2]),
                    cy=float(calibration.K[1, 2]),
                    width=width,
                    height=height,
                ),
            )
            log_camera_node(
                recording,
                rig,
                0,
                camera,
                name=name,
                kind="rgb",
                image_plane_distance=0.05,
                camera_model=calibration.camera_model,
            )
            rr.log(
                schema.cam_path(rig, 0),
                rr.AnyValues(distortion=calibration.dist, source_width=width, source_height=height, video_codec="av1"),
                static=True,
                recording=recording,
            )
            clip = root / "videos" / f"{name}.mp4"
            if len(times) < source_count:
                prefix = Path(work) / f"{name}.mp4"
                with timer.stage("remux"):
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-i",
                            str(clip),
                            "-map",
                            "0:v:0",
                            "-c:v",
                            "copy",
                            "-frames:v",
                            str(len(times)),
                            "-an",
                            str(prefix),
                        ],
                        check=True,
                    )
                clip = prefix
            with timer.stage("write:video"):
                log_video_stream(recording, clip, schema.video_path(rig, 0), times_ns=times, frame_indices=frames)
            if clip.parent == Path(work):
                clip.unlink()

        for rig, name in enumerate(EXO_CAMERAS):
            log_rig_node(recording, rig, reference=None, num_cameras=1, name=name, kind="exo")
            write_camera(rig, name, cameras[name], np.linalg.inv(cameras[name].word2cam))
        log_rig_node(recording, EGO_RIG, reference="cam_00", num_cameras=1, name="hololens", kind="ego")
        write_camera(EGO_RIG, "hololens", ego_camera, np.eye(4))
    start = 0
    for transforms in holo_batches(root / "meta_data/holo_data_wpose.csv", len(times), total=source_count):
        stop = start + len(transforms)
        log_dense_pose_track(recording, schema.rig_path(EGO_RIG), times_ns=times[start:stop], frame_indices=frames[start:stop], transforms=transforms)
        start = stop
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(times),
        num_cameras=len(CAMERA_NAMES),
        source_num_frames=source_count,
        source_revision=SOURCE_REVISION,
        clock_source=CLOCK_SOURCE,
        source_resolution=pa.array([f"{name}:{camera.size[0]}x{camera.size[1]}" for name, camera in (*cameras.items(), ("hololens", ego_camera))]),
        video_source="AV1 NVENC re-encode (CQ 38) of the Zenodo 15535461 release; same frame count",
        hololens_pose_note="Shipped noisy/drifting cam_T_world; missing poses are NaN, overlay misalignment is present in the source",
    )
    recording.send_property(
        "episode", rr.AnyValues(subject=identity.parts[1], split=identity.parts[0], activity=pa.array(sorted({s.text for s in actions["coarse"]})))
    )


def start_parameters(recording: rr.RecordingStream, specs: tuple[FitSpec, ...]) -> None:
    """State the source conventions once at each parameter entity."""
    for spec in specs:
        metadata: dict[str, str] = dict(root="Rh", translation_pivot="origin", source=spec.file)
        if spec.name == "body":
            metadata["gender"] = "neutral (assumed)"
        rr.log(spec.parameters_path, rr.AnyValues(use_pca=False, drop_untyped_nones=True, **metadata), static=True, recording=recording)


def log_parameters(recording: rr.RecordingStream, path: str, fits: list[Fit], times: Int64[ndarray, "n"], frames: Int64[ndarray, "n"]) -> None:
    """Keep all raw parameters, including rejected fits and per-frame betas."""
    columns = {}
    for field in ("poses", "Rh", "Th", "shapes"):
        values = np.stack([getattr(fit.parameters, field) for fit in fits])
        columns[field] = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])
    rr.send_columns(path, indexes=[time_column(times), frame_index_column(frames)], columns=rr.AnyValues.columns(**columns), recording=recording)
    rr.send_columns(
        path + "/l2_dist",
        indexes=[time_column(times), frame_index_column(frames)],
        columns=rr.Scalars.columns(scalars=[fit.parameters.l2_dist for fit in fits]),
        recording=recording,
    )


def write_pose(
    recording: rr.RecordingStream, rows: list[PoseRow], times: Int64[ndarray, "n"], frames: Int64[ndarray, "n"], *, specs: tuple[FitSpec, ...]
) -> None:
    """Write raw parameters for the fits owned by this layer."""
    for spec in specs:
        log_parameters(recording, spec.parameters_path, [getattr(row, spec.name) for row in rows], times, frames)


def write_hand_pose(
    recording: rr.RecordingStream, rows: list[PoseRow], times: Int64[ndarray, "n"], frames: Int64[ndarray, "n"], *, specs: tuple[FitSpec, ...]
) -> None:
    """The hand_pose layer owns joined COCO keypoints and the source frame id."""
    write_pose(recording, rows, times, frames, specs=specs)
    hands.log_keypoints3d(
        recording,
        times_ns=times,
        frame_indices=frames,
        positions=np.stack([row.positions for row in rows]),
        confidence=np.stack([row.confidence for row in rows]),
    )
    rr.send_columns(
        "/world/gt/rgb_frameid",
        indexes=[time_column(times), frame_index_column(frames)],
        columns=rr.Scalars.columns(scalars=[row.rgb_frameid for row in rows]),
        recording=recording,
    )


def project_keypoints(camera: ExoCamera, points: Float32[ndarray, "n 133 3"]) -> Float32[ndarray, "n 133 2"]:
    """Project through all eight coefficients; reject missing, behind and offscreen points."""
    flat = points.reshape(-1, 3).astype(np.float64)
    transform = camera.word2cam
    local = flat @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(local).all(axis=1) & (local[:, 2] > 0)
    pixels = np.full((len(flat), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        projected = cv2.projectPoints(local[valid], np.zeros(3), np.zeros(3), camera.K, camera.dist)[0].reshape(-1, 2)
        pixels[valid] = projected.astype(np.float32)
    visible = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.size[0])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.size[1])
    )
    pixels[~visible] = np.nan
    return pixels.reshape(points.shape[0], 133, 2)


def write_projections(
    recording: rr.RecordingStream, cameras: dict[str, ExoCamera], rows: list[PoseRow], times: Int64[ndarray, "n"], frames: Int64[ndarray, "n"]
) -> None:
    """Use shared HOT3D projection paths and confidence masking without modifying them."""
    recording.send_property("projections", rr.AnyValues(derived_from="coco133_xyz", camera_model=ExoCamera.camera_model))
    positions = np.stack([row.positions for row in rows])
    confidence = np.stack([row.confidence for row in rows])
    for rig, name in enumerate(EXO_CAMERAS):
        hands.log_keypoints2d(
            recording,
            rig,
            0,
            path=schema.coco133_uv_projected_path(rig, 0),
            times_ns=times,
            frame_indices=frames,
            positions=project_keypoints(cameras[name], positions),
            confidence=confidence,
        )


def write_actions(recording: rr.RecordingStream, actions: dict[str, list[Segment]], times: Int64[ndarray, "n"]) -> None:
    """Log active fine/coarse sets and explicit empty documents at segment ends."""
    recording.send_property("actions", rr.AnyValues(clock_rule="seconds -> round(s*30) -> timestamps.txt[frame], clamped to source endpoints"))
    for level, segments in actions.items():
        frames, texts = action_rows(segments, times)
        rr.send_columns(
            f"/task/actions/{level}",
            indexes=[time_column(times[frames]), frame_index_column(frames)],
            columns=rr.TextDocument.columns(text=texts),
            recording=recording,
        )
