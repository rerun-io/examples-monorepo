"""SHOW3D BASE builder; later annotation builders are siblings of write_base_layer."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde.json import to_json
from simplecv.camera_parameters import Extrinsics, PinholeParameters
from simplecv.camera_parameters import Intrinsics as CameraIntrinsics

from dataforge import schema, writing
from dataforge.datasets.show3d_source import (
    CAMERAS,
    BlurInfo,
    FrameClock,
    FrameInfo,
    HeadsetCalibration,
    HeadsetPose,
    HeadsetRig,
    Intrinsics,
    RecordingInfo,
    RigCalibration,
    Show3dCamera,
    agrees_with_frame,
    headset_rig,
    read_frame_clock,
    read_json,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import log_camera_node, log_pose_track, log_rig_node, log_video_stream
from dataforge.video_encoding import transcode_mp4_gray

VIDEO_CQ: int = 36
"""Selected on RTX 5090: CQ36 is the smallest tested output above 40 dB median.

Keyboard / birdhouse (MB, dB, seconds): builtin 53.8/44.61/9.9,
117.5/44.77/15.0; CQ28 93.2/45.58/9.6, 188.7/45.43/12.1;
CQ32 60.1/44.19/9.5, 128.0/44.03/12.1; CQ36 37.2/43.05/9.5,
87.6/42.58/12.0. Source: 62.4 / 127.4 MB; 20 frames per camera.
"""
VIDEO_GOP: int = 60
"""One second at the release's nominal 60 fps, shared with the measurement tool."""
VIDEO_CODEC: str = "av1"
"""AV1 NVENC with no B-frames."""


@dataclass(frozen=True, slots=True)
class SceneCamera:
    """Validated sidecars for one present camera."""

    camera: Show3dCamera
    """Stable source and Rerun identity."""
    video: Path
    """Source clip."""
    calibration: Intrinsics
    """Typed intrinsics."""
    calibration_json: str
    """Full rig calibration or headset intrinsics only."""
    rig_T_cam: Float64[ndarray, "4 4"]
    """Static camera pose in metres."""
    blur: BlurInfo
    """Normalized source boxes."""
    box_indices: Int64[ndarray, "n"]
    """Selected source indices for box rows."""
    box_positions: Int64[ndarray, "n"]
    """Positions in the selected frame arrays."""


@dataclass(frozen=True, slots=True)
class Scene(FrameClock):
    """Frame clock plus calibration, blur, and stereo data."""

    cameras: tuple[SceneCamera, ...]
    """Present cameras with all sidecars loaded."""
    poses: list[HeadsetPose]
    """Selected headset0 provenance rows."""
    offsets: Int64[ndarray, "n"]
    """Pose row positions found in the sorted frame indices."""


def read_scene(scene_dir: Path, *, scene_key: str, frame_limit: int | None = None) -> Scene:
    """Read the metadata clock, then validate camera sidecars against it."""
    if frame_limit is not None and frame_limit <= 0:
        raise ValueError("frame_limit must be positive")
    clock: FrameClock = read_frame_clock(scene_dir, scene_key)
    info: RecordingInfo = clock.info
    frames: list[FrameInfo] = clock.frames
    indices: Int64[ndarray, "n"] = clock.frame_indices
    present: tuple[Show3dCamera, ...] = tuple(camera for camera in CAMERAS if camera.source_name in info.resolution)
    if sum(camera.rig == 1 for camera in present) != 2:
        raise ValueError(f"{scene_key}: both headset cameras are required")
    calibrations: dict[str, Intrinsics] = {}
    texts: dict[str, str] = {}
    headsets: list[HeadsetCalibration] = []
    for camera in present:
        path: Path = scene_dir / f"camera_calibration/{camera.source_name}.json"
        text: str = path.read_text()
        calibration: Intrinsics
        if camera.rig == 1:
            headset: HeadsetCalibration = read_json(path, HeadsetCalibration, text=text)
            headsets.append(headset)
            calibration = headset
            texts[camera.source_name] = to_json(read_json(path, Intrinsics, text=text))
            poses: list[HeadsetPose] = list(headset.T_WorldFromCamera_by_index.values())
            pose_indices: Int64[ndarray, "n"] = np.array([pose.index for pose in poses], dtype=np.int64)
            positions: Int64[ndarray, "n"] = np.searchsorted(indices, pose_indices)
            if np.any(positions >= len(indices)) or not np.array_equal(indices[positions], pose_indices):
                raise ValueError(f"{scene_key}: headset pose index absent from frame_info")
            for pose, position in zip(poses, positions, strict=True):
                frame: FrameInfo = frames[int(position)]
                if not agrees_with_frame(pose, frame):
                    raise ValueError(f"{scene_key}: headset pose {pose.index} disagrees with frame_info")
        else:
            calibration = read_json(path, RigCalibration, text=text)
            texts[camera.source_name] = text
        if info.resolution[camera.source_name] != [calibration.ImageSizeY, calibration.ImageSizeX]:
            raise ValueError(f"{scene_key}/{camera.source_name}: resolution disagrees with calibration")
        calibrations[camera.source_name] = calibration
    stereo: HeadsetRig = headset_rig(headsets[0], headsets[1], scene=scene_key)
    count: int = info.num_frames if frame_limit is None else min(frame_limit, info.num_frames)
    cameras: list[SceneCamera] = []
    for camera in present:
        calibration = calibrations[camera.source_name]
        transform: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        if isinstance(calibration, RigCalibration):
            transform = calibration.T_WorldFromCamera.copy()
            transform[:3, 3] *= 0.001
        elif camera.cam == 1:
            transform = stereo.cam0_T_cam1
        blur: BlurInfo = read_json(scene_dir / f"blur_info/{camera.source_name}.mp4.json", BlurInfo)
        box_indices: Int64[ndarray, "n"] = np.array(sorted(int(key) for key in blur.blur_boxes), dtype=np.int64)
        positions = np.searchsorted(indices, box_indices)
        if np.any(positions >= len(indices)) or not np.array_equal(indices[positions], box_indices):
            raise ValueError(f"{scene_key}/{camera.source_name}: blur index absent from frame_info")
        selected: Bool[ndarray, "n"] = positions < count
        cameras.append(
            SceneCamera(
                camera,
                scene_dir / f"{camera.source_name}.mp4",
                calibration,
                texts[camera.source_name],
                transform,
                blur,
                box_indices[selected],
                positions[selected],
            )
        )
    poses = sorted(
        (pose for pose in headsets[0].T_WorldFromCamera_by_index.values() if pose.index <= indices[count - 1]), key=lambda pose: pose.index
    )
    pose_indices = np.array([pose.index for pose in poses], dtype=np.int64)
    offsets: Int64[ndarray, "n"] = np.searchsorted(indices, pose_indices)
    times: Int64[ndarray, "n"] = clock.times_ns[:count]
    print(
        f"{scene_key}: headset translation std {stereo.translation_std_m * 1000:.9g} mm; rotation max {stereo.rotation_max_deg:.9g} deg; "
        f"normalized blur boxes {sum(camera.blur.num_normalized_boxes for camera in cameras)}"
    )
    return Scene(
        info=info, frames=frames[:count], times_ns=times, frame_indices=indices[:count], cameras=tuple(cameras), poses=poses, offsets=offsets
    )


def pinhole(name: str, calibration: Intrinsics, rig_T_cam: Float64[ndarray, "4 4"]) -> PinholeParameters:
    """Build a camera from typed source intrinsics and its metre-valued transform."""
    if calibration.DistortionModel != "PinholePlane":
        raise ValueError(f"{name}: unsupported distortion model")
    return PinholeParameters(
        name=name,
        extrinsics=Extrinsics(world_R_cam=rig_T_cam[:3, :3], world_t_cam=rig_T_cam[:3, 3]),
        intrinsics=CameraIntrinsics.from_focal_principal_point(
            camera_conventions="RDF",
            fl_x=calibration.fx,
            fl_y=calibration.fy,
            cx=calibration.cx,
            cy=calibration.cy,
            height=calibration.ImageSizeY,
            width=calibration.ImageSizeX,
        ),
    )


def log_cameras(recording: rr.RecordingStream, scene: Scene, work_dir: Path) -> None:
    """Encode at most three clips at once, then log and remove them in order."""
    work_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = [work_dir / f"{camera.camera.source_name}.mp4" for camera in scene.cameras]
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures: list[Future[int]] = [
                executor.submit(transcode_mp4_gray, camera.video, clip, gop=VIDEO_GOP, cq=VIDEO_CQ, fps=int(scene.info.fps), frames=len(scene.frames))
                for camera, clip in zip(scene.cameras, clips, strict=True)
            ]
            for source, clip, future in zip(scene.cameras, clips, futures, strict=True):
                future.result()
                camera: Show3dCamera = source.camera
                log_camera_node(
                    recording,
                    camera.rig,
                    camera.cam,
                    pinhole(camera.source_name, source.calibration, source.rig_T_cam),
                    name=camera.source_name,
                    kind="grayscale",
                    image_plane_distance=0.05,
                )
                rr.log(
                    schema.cam_path(camera.rig, camera.cam),
                    rr.AnyValues(
                        source_calibration_json=source.calibration_json,
                        video_codec=VIDEO_CODEC,
                        gop=pa.array([VIDEO_GOP], type=pa.int64()),
                        cq=pa.array([VIDEO_CQ], type=pa.int64()),
                    ),
                    static=True,
                    recording=recording,
                )
                log_video_stream(
                    recording, clip, schema.video_path(camera.rig, camera.cam), times_ns=scene.times_ns, frame_indices=scene.frame_indices
                )
                clip.unlink()
                if source.box_indices.size:
                    lengths: list[int] = [len(source.blur.blur_boxes[str(index)]) for index in source.box_indices]
                    boxes: Float32[ndarray, "n 4"] = np.asarray(
                        [box for index in source.box_indices for box in source.blur.blur_boxes[str(index)]], dtype=np.float32
                    ).reshape(-1, 4)
                    rr.send_columns(
                        f"{schema.pinhole_path(camera.rig, camera.cam)}/blur_boxes",
                        indexes=scene.indexes(source.box_positions),
                        columns=rr.Boxes2D.columns(
                            centers=(boxes[:, :2] + boxes[:, 2:]) / 2.0, half_sizes=(boxes[:, 2:] - boxes[:, :2]) / 2.0
                        ).partition(lengths),
                        recording=recording,
                    )
    finally:
        for clip in clips:
            clip.unlink(missing_ok=True)


def log_headset(recording: rr.RecordingStream, scene: Scene) -> None:
    """Write pose transforms and their independent provenance columns."""
    if scene.poses:
        rr.send_columns(
            schema.rig_path(1),
            indexes=scene.indexes(scene.offsets),
            columns=rr.AnyValues.columns(is_synthesized=pa.array([pose.is_synthesized for pose in scene.poses], type=pa.bool_())),
            recording=recording,
        )
    provenance: dict[str, list[str | bool | None]] = {
        "pose_source": [pose.pose_source for pose in scene.poses],
        "is_pose_valid": [pose.is_pose_valid for pose in scene.poses],
    }
    for key, dtype in (("pose_source", pa.string()), ("is_pose_valid", pa.bool_())):
        rows: list[tuple[int, str | bool]] = [
            (int(offset), value) for offset, value in zip(scene.offsets, provenance[key], strict=True) if value is not None
        ]
        if rows:
            rr.send_columns(
                schema.rig_path(1),
                indexes=scene.indexes([position for position, _ in rows]),
                columns=rr.AnyValues.columns(**{key: pa.array([value for _, value in rows], type=dtype)}),
                recording=recording,
            )
    posed: list[tuple[int, Float64[ndarray, "4 4"]]] = [
        (int(offset), pose.T_WorldFromCamera) for pose, offset in zip(scene.poses, scene.offsets, strict=True) if pose.T_WorldFromCamera is not None
    ]
    if posed:
        posed_positions: list[int] = [position for position, _ in posed]
        transforms: Float64[ndarray, "n 4 4"] = np.stack([transform for _, transform in posed])
        log_pose_track(
            recording,
            schema.rig_path(1),
            times_ns=scene.times_ns[posed_positions],
            frame_indices=scene.frame_indices[posed_positions],
            translations_xyz=transforms[:, :3, 3] * 0.001,
            quaternions_xyzw=Rotation.from_matrix(transforms[:, :3, :3]).as_quat(),
        )


def log_frames(recording: rr.RecordingStream, scene: Scene) -> None:
    """Preserve the source clock, frame id and missing-camera rows."""
    rr.send_columns(
        "/frames",
        indexes=scene.indexes(slice(None)),
        columns=rr.AnyValues.columns(
            source_frame_id=pa.array([frame.agt_frame_id for frame in scene.frames], type=pa.int64()),
            source_timestamp_s=pa.array([frame.timestamp for frame in scene.frames], type=pa.float64()),
            missing_cameras=pa.array([frame.missing_cameras for frame in scene.frames], type=pa.list_(pa.string())),
        ),
        recording=recording,
    )


def write_base_layer(
    identity: SequenceIdentity,
    scene_dir: Path,
    target: Path,
    *,
    work_dir: Path,
    hf_revision: str,
    default_blueprint: rrb.Blueprint | None = None,
    frame_limit: int | None = None,
) -> None:
    """Validate a scene, then publish its BASE recording atomically."""
    scene: Scene = read_scene(scene_dir, scene_key=identity.sequence_key, frame_limit=frame_limit)
    with writing.atomic_recording(target, recording_id=identity.recording_id, default_blueprint=default_blueprint) as recording:
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True, recording=recording)
        log_rig_node(recording, 0, reference=None, num_cameras=sum(camera.camera.rig == 0 for camera in scene.cameras), name="back_rig", kind="exo")
        log_rig_node(recording, 1, reference="cam_00", num_cameras=2, name="quest3", kind="ego")
        log_cameras(recording, scene, work_dir)
        log_headset(recording, scene)
        log_frames(recording, scene)
        writing.send_capture_properties(
            recording,
            identity,
            hf_revision=hf_revision,
            num_frames=len(scene.frames),
            num_cameras=len(scene.cameras),
            num_synthesized_headset_poses=pa.array([sum(pose.is_synthesized for pose in scene.poses)], type=pa.int64()),
            source_start_time_s=pa.array([scene.frames[0].timestamp], type=pa.float64()),
            source_start_frame_id=pa.array([scene.frames[0].agt_frame_id], type=pa.int64()),
        )
