"""Columnar Assembly101 layers; compressed AV1 packets are never re-encoded."""

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import hands, schema, writing
from dataforge.datasets.assembly101_actions import Actions, action_rows
from dataforge.datasets.assembly101_calibration import ResolvedLens, camera_parameters, resolve_calibration
from dataforge.datasets.assembly101_source import (
    CLOCK_SOURCE,
    EGO_RIG,
    EXO_SERIALS,
    SOURCE_REVISION,
    EgoTransforms,
    frame_times,
    pose_path,
    read_confidence,
    read_hand_rows,
    read_pixels,
    read_record,
    read_transforms,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    annotation_context,
    frame_index_column,
    log_camera_node,
    log_pose_track,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.records import read_json
from dataforge.timing import SequenceTimer
from dataforge.world_up import WORLD_UP_VIEW_COORDINATES


@dataclass(frozen=True, slots=True)
class CameraSource:
    """Source video, stable serial ordering and source coordinate system."""

    path: Path
    """Read-only AV1 file."""
    serial: str
    """Hardware serial."""
    rig: int
    """Fixed exo rig 0..7 or moving ego rig 8."""
    cam: int
    """Camera within the rig."""

    @property
    def key(self) -> str:
        return f"{self.serial}:{'rgb' if self.rig != EGO_RIG else 'mono10bit'}"

    @property
    def source_resolution(self) -> tuple[int, int]:
        return (1920, 1080) if self.rig != EGO_RIG else (636, 480)


def camera_sources(root: Path, sequence: str) -> list[CameraSource]:
    """Sort exo by fixed hardware serial, then ego numerically, never by glob order."""
    folder: Path = root / "videos/av1-720-new" / sequence
    sources: list[CameraSource] = []
    for rig, serial in enumerate(EXO_SERIALS):
        path: Path = folder / f"{serial}_rgb_low.mp4"
        if path.is_file():
            sources.append(CameraSource(path, serial, rig, 0))
    ego: list[Path] = sorted(folder.glob("HMC_*_mono10bit_low.mp4"), key=lambda path: int(path.name.split("_")[1]))
    sources.extend(CameraSource(path, path.name.split("_")[1], EGO_RIG, index) for index, path in enumerate(ego))
    return sources


@dataclass(frozen=True, slots=True)
class Poses:
    """Verified pose metadata and one moving-rig track, in metres."""

    fixed: dict[str, Float64[ndarray, "4 4"]]
    """World-from-camera fixed transforms in source mm."""
    calibration: dict[str, ResolvedLens]
    """Session exo and serial ego lenses."""
    frames: Int64[ndarray, "n"]
    """Real pose frame keys."""
    world_T_rig: Float64[ndarray, "n 4 4"]
    """Temporal reference camera pose in metres."""
    rig_T_cam: dict[str, Float64[ndarray, "4 4"]]
    """Static headset calibration in metres."""
    timestamp_t0: float
    """Device-origin timestamp, not wall clock."""


@dataclass(frozen=True, slots=True)
class CameraVideo(CameraSource):
    """Container metadata read once for conversion and timing."""

    stored_resolution: tuple[int, int]
    """Stored width and height."""
    source_num_frames: int
    """Full container frame count."""
    num_frames: int
    """Frame count after the requested prefix limit."""
    scale: float = field(init=False)
    """Stored/source pixel scale for shipped 2D rows."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", self.stored_resolution[0] / self.source_resolution[0])


@dataclass(frozen=True, slots=True)
class Scene:
    """Videos and an optional complete pose record."""

    cameras: list[CameraVideo]
    """Source and rig ordering with container metadata."""
    poses: Poses | None
    """Absent only for a verified video-only sequence."""


def read_scene(root: Path, sequence: str, frame_limit: int | None, *, has_poses: bool) -> Scene:
    """Read container metadata once and preserve independent video/pose lengths."""
    sources: list[CameraSource] = camera_sources(root, sequence)
    cameras: list[CameraVideo] = []
    for camera in sources:
        with av.open(str(camera.path)) as container:
            stream = container.streams.video[0]
            stored: tuple[int, int] = (stream.width, stream.height)
            source_count: int = stream.frames
            if stream.codec_context.codec.canonical_name != "av1" or stream.codec_context.has_b_frames:
                raise ValueError(f"{camera.path}: expected mirror AV1 without B-frames")
            if source_count == 0:
                source_count = sum(packet.pts is not None for packet in container.demux(stream))
        count: int = source_count if frame_limit is None else min(source_count, frame_limit)
        cameras.append(CameraVideo(camera.path, camera.serial, camera.rig, camera.cam, stored, source_count, count))
    return Scene(cameras, read_poses(root, sequence, sources, frame_limit) if has_poses else None)


def read_poses(root: Path, sequence: str, cameras: list[CameraSource], frame_limit: int | None) -> Poses:
    """Read pose members on their real frame keys, independent of video lengths."""
    fixed: dict[str, Float64[ndarray, "4 4"]] = read_transforms(pose_path(root, "camera_extrinsics_fixed", sequence))
    calibration: dict[str, ResolvedLens] = resolve_calibration(root, fixed)
    timestamps: dict[str, float] = read_json(pose_path(root, "timestamp", sequence), dict[str, float])
    ego: dict[str, dict[str, Float64[ndarray, "4 4"]]] = read_record(pose_path(root, "camera_extrinsics_ego", sequence), EgoTransforms).data
    keys: list[str] = sorted(ego, key=int)
    if set(keys) != set(timestamps):
        raise ValueError(f"{sequence}: ego pose and timestamp frame keys disagree")
    if frame_limit is not None:
        keys = [key for key in keys if int(key) < frame_limit]
    frames: Int64[ndarray, "n"] = np.array(keys, dtype=np.int64)
    reference: str = next(camera.key for camera in cameras if camera.rig == EGO_RIG)
    track: Float64[ndarray, "n 4 4"] = np.array([ego[key][reference] for key in keys], dtype=np.float64)
    relative: dict[str, Float64[ndarray, "4 4"]] = {}
    for camera in cameras:
        if camera.rig == EGO_RIG:
            transform: Float64[ndarray, "4 4"] = np.linalg.inv(track[0]) @ ego[keys[0]][camera.key]
            transform[:3, 3] *= 0.001
            relative[camera.key] = transform
    track[:, :3, 3] *= 0.001
    return Poses(fixed, calibration, frames, track, relative, timestamps["0"])


def remux_prefix(source: Path, target: Path, count: int) -> None:
    """Copy the first N AV1 packets into an MP4 without decoding or encoding."""
    with av.open(str(source)) as container, av.open(str(target), "w") as output:
        stream = container.streams.video[0]
        destination = output.add_stream_from_template(stream, opaque=True)
        seen: int = 0
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            if seen >= count:
                break
            packet.stream = destination
            output.mux(packet)
            seen += 1
        if seen != count:
            raise ValueError(f"{source}: only {seen} packets, expected {count}")


def write_base(
    recording: rr.RecordingStream,
    scene: Scene,
    identity: SequenceIdentity,
    actions: Actions,
    frame_limit: int | None,
    timer: SequenceTimer,
) -> None:
    """Static exo rigs, one moving headset and independent start-aligned videos."""
    rr.log("/", WORLD_UP_VIEW_COORDINATES["+y"], annotation_context(), static=True, recording=recording)
    for rig in sorted({camera.rig for camera in scene.cameras}):
        log_rig_node(
            recording,
            rig,
            reference="cam_00" if rig == EGO_RIG and scene.poses is not None else None,
            num_cameras=sum(camera.rig == rig for camera in scene.cameras),
            kind="ego" if rig == EGO_RIG else "exo",
        )
    calibration: dict[str, ResolvedLens] = {}
    timestamp_t0: float | None = None
    if scene.poses is not None:
        poses: Poses = scene.poses
        calibration = poses.calibration
        timestamp_t0 = poses.timestamp_t0
        log_pose_track(
            recording,
            schema.rig_path(EGO_RIG),
            times_ns=frame_times(poses.frames),
            frame_indices=poses.frames,
            translations_xyz=poses.world_T_rig[:, :3, 3],
            quaternions_xyzw=Rotation.from_matrix(poses.world_T_rig[:, :3, :3]).as_quat(),
        )
        for camera in scene.cameras:
            transform: Float64[ndarray, "4 4"] = poses.rig_T_cam[camera.key] if camera.rig == EGO_RIG else poses.fixed[camera.key].copy()
            if camera.rig != EGO_RIG:
                transform[:3, 3] *= 0.001
            resolved: ResolvedLens | None = calibration.get(camera.key)
            if resolved is not None:
                log_camera_node(
                    recording,
                    camera.rig,
                    camera.cam,
                    camera_parameters(resolved.lens, transform, camera.stored_resolution),
                    name=camera.serial,
                    kind="grayscale" if camera.rig == EGO_RIG else "rgb",
                    image_plane_distance=0.05,
                    camera_model=resolved.lens.DistortionModel,
                )
            else:
                rr.log(
                    schema.cam_path(camera.rig, camera.cam),
                    rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
                    static=True,
                    recording=recording,
                )
    for camera in scene.cameras:
        count: int = camera.num_frames
        resolved = calibration.get(camera.key)
        if resolved is None:
            # log_camera_node names calibrated cameras; a None here would clear its static name/kind.
            rr.log(
                schema.cam_path(camera.rig, camera.cam),
                rr.AnyValues(name=camera.serial, kind="grayscale" if camera.rig == EGO_RIG else "rgb"),
                static=True,
                recording=recording,
            )
        rr.log(
            schema.cam_path(camera.rig, camera.cam),
            rr.AnyValues(
                source_resolution=pa.array(camera.source_resolution),
                stored_resolution=pa.array(camera.stored_resolution),
                source_num_frames=camera.source_num_frames,
                num_frames=count,
                video_codec="av1",
                calibration_source=resolved.source if resolved is not None else "none",
            ),
            static=True,
            recording=recording,
        )
        frames: Int64[ndarray, "n"] = np.arange(count, dtype=np.int64)
        with timer.stage("remux"), TemporaryDirectory(prefix="assembly101-") as folder:
            video: Path = camera.path
            if frame_limit is not None:
                video = Path(folder) / "prefix.mp4"
                remux_prefix(camera.path, video, count)
            log_video_stream(recording, video, schema.video_path(camera.rig, camera.cam), times_ns=frame_times(frames), frame_indices=frames)
    exo_source: str = next(
        (calibration[camera.key].source for camera in scene.cameras if camera.rig != EGO_RIG and camera.key in calibration),
        "none",
    )
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=max(camera.num_frames for camera in scene.cameras),
        num_cameras=len(scene.cameras),
        source_revision=SOURCE_REVISION,
        clock_source=CLOCK_SOURCE,
        timestamp_t0=timestamp_t0,
        source_resolution=pa.array([f"{camera.serial}:{camera.source_resolution[0]}x{camera.source_resolution[1]}" for camera in scene.cameras]),
        calibration_source=exo_source,
        hand_pose_present=scene.poses is not None,
        actions_present=bool(actions.coarse or actions.fine),
        coarse_actions_present=bool(actions.coarse),
        fine_actions_present=bool(actions.fine),
    )


def write_hands(recording: rr.RecordingStream, root: Path, sequence: str, scene: Scene, frame_limit: int | None) -> None:
    """Read each hand member once and send dense column batches on its actual keys."""
    confidence = read_confidence(pose_path(root, "hand_confidences", sequence))
    for frames, positions, scores in read_hand_rows(
        pose_path(root, "landmarks3D", sequence), confidence, dimensions=3, scale=0.001, frame_limit=frame_limit
    ):
        hands.log_keypoints3d(recording, times_ns=frame_times(frames), frame_indices=frames, positions=positions, confidence=scores)
    cameras: dict[str, CameraVideo] = {camera.key: camera for camera in scene.cameras}
    for key, (frames, pixels, scores) in read_pixels(
        pose_path(root, "landmarks2D", sequence), confidence, {key: camera.scale for key, camera in cameras.items()}, frame_limit
    ):
        camera: CameraSource = cameras[key]
        hands.log_keypoints2d(
            recording, camera.rig, camera.cam, times_ns=frame_times(frames), frame_indices=frames, positions=pixels, confidence=scores
        )


def write_actions(recording: rr.RecordingStream, actions: Actions, frame_limit: int | None) -> None:
    """One columnar text track per granularity; empty documents clear ended segments."""
    for granularity, segments in (("coarse", actions.coarse), ("fine", actions.fine)):
        frames, texts = action_rows(segments, frame_limit)
        if len(frames):
            rr.send_columns(
                f"/task/actions/{granularity}",
                indexes=[time_column(frame_times(frames)), frame_index_column(frames)],
                columns=rr.TextDocument.columns(text=texts),
                recording=recording,
            )
