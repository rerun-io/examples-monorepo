"""Ego-Exo4D layer writers: the base layer from the raw take, its sidecars, and the lens-model projections.

Rig 0 is the wearer's Aria (moving, device frame, the closed-loop trajectory as ``world_T_rig``) with
RGB, SLAM left/right and the eye-tracking camera; rigs 1..N are the localized GoPros, static, in
``gopro_calibs.csv`` order. Every stream logs on the take clock (Aria RGB capture time) at 30 Hz.

Base also writes two sidecars, so the derived layers never read the raw take: ``frames.npz`` (frame times and
the device pose at each) and ``cameras.json`` (the GoPro rows and the Aria calibration document).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import serde
from serde.json import to_json
from simplecv.camera_parameters import Fisheye62Parameters, Fisheye624Parameters
from simplecv.sensors.camera import fisheye624
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge import aria, hands, schema, writing
from dataforge.datasets.egoexo4d_body import HmFit, coco133_from_openpose67
from dataforge.datasets.egoexo4d_source import FPS, KB4, GoproCalib, Take, stored_size
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import annotation_context, log_camera_node, log_camera_source, log_dense_pose_track, log_rig_node, log_video_stream
from dataforge.records import read_json
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, parallel_clips, transcode_mp4

EGO_RIG: int = 0
"""The Aria; fixed at 0 so the blueprint can name it whatever the number of GoPros."""
EXO_SLOTS: int = 5
"""GoPro panes the dataset blueprint lays out (rigs 1..5); Ego-Exo4D captures carry four or five."""
FISHEYE624: str = "fisheye624"
"""``camera_model`` of the Aria cameras."""
ARIA_STREAMS: tuple[tuple[str, str, str | None, bool], ...] = (
    ("rgb", "214-1", "camera-rgb", False),
    ("slam-left", "1201-1", "camera-slam-left", True),
    ("slam-right", "1201-2", "camera-slam-right", True),
    ("et", "211-1", None, True),
)
"""(readable stream id, VRS stream id, calibration label, gray) per Aria camera, in cam_MM order. The eye-tracking
stream packs both eye cameras side by side, so no single calibration describes it: it is logged as video only."""
IMAGE_ROTATION_CW_DEG: int = 90
"""Ego-Exo4D's Aria MP4s are a quarter turn clockwise from the sensor readout; the calibration is turned to match."""
FRAMES_SIDECAR: str = "frames.npz"
CAMERAS_SIDECAR: str = "cameras.json"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CamerasSidecar:
    """What the projections layer needs to know about the cameras, written by base."""

    gopros: list[GoproCalib]
    """Localized GoPros at native resolution, in rig order (rig 1 first)."""
    aria_calib_json: str
    """The Aria's factory ``calib_json`` document, verbatim from the take's VRS."""
    aria_sizes: dict[str, tuple[int, int]]
    """Stored (rotated) width and height of each calibrated Aria stream, by calibration label."""

    def exo_cameras(self) -> list[Fisheye62Parameters]:
        """Each GoPro's KB4 lens at its stored size, with ``world_T_cam``."""
        return [calib.camera(*stored_size(calib)) for calib in self.gopros]

    def aria_cameras(self) -> dict[str, Fisheye624Parameters]:
        """Each calibrated Aria camera in the stored (rotated) orientation; ``rig_T_cam`` is ``device_T_camera``."""
        device: aria.DeviceCalibration = aria.DeviceCalibration.from_json(self.aria_calib_json, "cameras.json")
        return {label: logged_calibration(device.camera(label), width, height) for label, (width, height) in self.aria_sizes.items()}


def logged_calibration(factory: Fisheye624Parameters, width: int, height: int) -> Fisheye624Parameters:
    """A factory camera fitted to a stream stored ``width`` x ``height`` after the quarter turn."""
    return fisheye624.rotate_cw90(aria.rescale_to_stream(factory, height, width))


def video_size(path: Path) -> tuple[int, int]:
    """Width and height of an mp4's first video stream."""
    with av.open(str(path)) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        return stream.codec_context.width, stream.codec_context.height


@dataclass(frozen=True, slots=True)
class BaseInputs:
    """One take's raw inputs, resolved by the dataset."""

    take: Take
    """The takes.json entry."""
    root: Path
    """Raw root (the egoexo CLI output directory)."""
    gopros: list[GoproCalib]
    """Localized GoPros, rig order."""
    calib_json: str
    """The Aria VRS ``calib_json`` tag."""
    times_ns: Int64[ndarray, "n"]
    """Take clock."""
    world_T_device: Float64[ndarray, "n 4 4"]
    """Aria pose at each frame, NaN where the trajectory has a gap."""


def write_base(recording: rr.RecordingStream, identity: SequenceIdentity, inputs: BaseInputs, timer: SequenceTimer, work: Path) -> CamerasSidecar:
    """Encode every stream, log cameras, videos and the Aria trajectory; return the cameras sidecar."""
    take: Take = inputs.take
    times: Int64[ndarray, "n"] = inputs.times_ns
    frames: Int64[ndarray, "n"] = np.arange(len(times), dtype=np.int64)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)  # MPS worlds are gravity-aligned, z up
    rr.log("/", annotation_context(), static=True, recording=recording)
    device: aria.DeviceCalibration = aria.DeviceCalibration.from_json(inputs.calib_json, f"{take.take_name} calib_json")

    slots: list[tuple[int, int, Path, bool, tuple[int, int] | None]] = []
    for rig, calib in enumerate(inputs.gopros, start=1):
        slots.append((rig, 0, inputs.root / take.video(calib.cam_uid, "0"), False, stored_size(calib)))
    for cam, (stream, _, _, gray) in enumerate(ARIA_STREAMS):
        slots.append((EGO_RIG, cam, inputs.root / take.video(take.aria, stream), gray, None))
    sources: dict[tuple[int, int], tuple[int, int]] = {(rig, cam): video_size(path) for rig, cam, path, _, _ in slots}

    def encode(source: Path, clip: Path, gray: bool, size: tuple[int, int] | None) -> None:
        transcode_mp4(source, clip, gop=AV1_GOP, cq=AV1_CQ, fps=FPS, frames=len(times), gray=gray, size=size, decode="cuda")

    clips: list[Path] = [work / f"rig_{rig:02d}_cam_{cam:02d}.mp4" for rig, cam, _, _, _ in slots]
    jobs = [(clip, partial(encode, path, clip, gray, size)) for clip, (_, _, path, gray, size) in zip(clips, slots, strict=True)]
    resolutions: list[str] = []
    aria_sizes: dict[str, tuple[int, int]] = {}
    log_rig_node(recording, EGO_RIG, reference=None, num_cameras=len(ARIA_STREAMS), name=take.aria, kind="ego")
    for rig, calib in enumerate(inputs.gopros, start=1):
        log_rig_node(recording, rig, reference=None, num_cameras=1, name=calib.cam_uid, kind="exo")
    with parallel_clips(jobs, timer, workers=6) as encoded:
        for (rig, cam, _, _, size), clip in zip(slots, encoded, strict=True):
            width, height = sources[(rig, cam)]
            if rig == EGO_RIG:
                stream, stream_id, label, gray = ARIA_STREAMS[cam]
                name: str = f"{take.aria}_{stream_id}"
                if label is not None:
                    calibration: Fisheye624Parameters = logged_calibration(device.camera(label), width, height)
                    aria_sizes[label] = (width, height)
                    log_camera_node(
                        recording,
                        EGO_RIG,
                        cam,
                        calibration.to_fisheye62(),
                        name=label,
                        kind="grayscale" if gray else "rgb",
                        image_plane_distance=0.05,
                        camera_model=FISHEYE624,
                        image_rotation_cw_deg=IMAGE_ROTATION_CW_DEG,
                    )
                else:  # both eye cameras in one frame: no single calibration, so no pinhole
                    rr.log(schema.cam_path(EGO_RIG, cam), rr.AnyValues(name="camera-et", kind="grayscale"), static=True, recording=recording)
                resolutions.append(
                    log_camera_source(
                        recording,
                        EGO_RIG,
                        cam,
                        name=name,
                        source_width=width,
                        source_height=height,
                        video_codec="av1",
                        cq=AV1_CQ,
                        gop=AV1_GOP,
                        stream_id=stream_id,
                    )
                )
            else:
                calib = inputs.gopros[rig - 1]
                assert size is not None
                log_camera_node(recording, rig, 0, calib.camera(*size), name=calib.cam_uid, kind="rgb", image_plane_distance=0.1, camera_model=KB4)
                resolutions.append(
                    log_camera_source(
                        recording,
                        rig,
                        0,
                        name=calib.cam_uid,
                        source_width=width,
                        source_height=height,
                        stored_width=size[0],
                        stored_height=size[1],
                        video_codec="av1",
                        cq=AV1_CQ,
                        gop=AV1_GOP,
                    )
                )
            with timer.stage("write:video"):
                log_video_stream(recording, clip, schema.video_path(rig, cam), times_ns=times, frame_indices=frames)
    log_dense_pose_track(recording, schema.rig_path(EGO_RIG), times_ns=times, frame_indices=frames, transforms=inputs.world_T_device)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(times),
        num_cameras=len(slots),
        clock_source=f"timesync.csv {take.aria}_214-1_capture_timestamp_ns (Aria device clock), rows timesync_start_idx..timesync_end_idx-1",
        source_resolution=pa.array(resolutions),
        image_rotation_cw_deg=IMAGE_ROTATION_CW_DEG,
        trajectory_coverage=float(np.isfinite(inputs.world_T_device).all(axis=(1, 2)).mean()),
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            drop_untyped_nones=True,
            take_uid=take.take_uid,
            activity=take.parent_task_name,
            task=take.task_name,
            university=take.university_name,
            capture=take.capture.capture_name,
        ),
    )
    return CamerasSidecar(gopros=inputs.gopros, aria_calib_json=inputs.calib_json, aria_sizes=aria_sizes)


def write_sidecars(directory: Path, cameras: CamerasSidecar, times_ns: Int64[ndarray, "n"], world_T_device: Float64[ndarray, "n 4 4"]) -> None:
    """Write both sidecars, each through a temp file so an existing one is never half-written."""
    directory.mkdir(parents=True, exist_ok=True)
    staged: Path = directory / f".{FRAMES_SIDECAR}.tmp.npz"
    np.savez(staged, times_ns=times_ns, world_T_device=world_T_device)
    staged.replace(directory / FRAMES_SIDECAR)
    staged = directory / f".{CAMERAS_SIDECAR}.tmp"
    staged.write_text(to_json(cameras))
    staged.replace(directory / CAMERAS_SIDECAR)


@dataclass(frozen=True, slots=True)
class FrameSidecar:
    """``frames.npz``: the take clock and the Aria pose on it."""

    times_ns: Int64[ndarray, "n"]
    world_T_device: Float64[ndarray, "n 4 4"]


def read_sidecars(directory: Path) -> tuple[FrameSidecar, CamerasSidecar]:
    """Both sidecars base wrote; a missing one names the base layer that has to run first."""
    for name in (FRAMES_SIDECAR, CAMERAS_SIDECAR):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"{directory / name} is missing: convert the base layer first (--force rebuilds it from the raw take)")
    with np.load(directory / FRAMES_SIDECAR, allow_pickle=False) as npz:
        frames: FrameSidecar = FrameSidecar(npz["times_ns"], npz["world_T_device"])
    return frames, read_json(directory / CAMERAS_SIDECAR, CamerasSidecar)


def write_projections(recording: rr.RecordingStream, fit: HmFit, frames: FrameSidecar, cameras: CamerasSidecar, rows: Int64[ndarray, "t"]) -> None:
    """The fit's COCO-133 keypoints through every calibrated camera's full lens model (KB4 GoPros, FISHEYE624 Aria)."""
    positions: Float32[ndarray, "t 133 3"] = coco133_from_openpose67(fit.joints3d[rows])
    positions[~fit.valid[rows]] = np.nan
    world_T_device: Float64[ndarray, "t 4 4"] = frames.world_T_device[rows]
    flat: Float64[ndarray, "p 3"] = positions.reshape(-1, 3).astype(np.float64)

    def pixels() -> Iterator[tuple[tuple[int, int], Float32[ndarray, "t 133 2"]]]:
        for rig, camera in enumerate(cameras.exo_cameras(), start=1):
            cam_T_world: Float64[ndarray, "4 4"] = camera.extrinsics.cam_T_world
            yield (rig, 0), project_fisheye62(flat @ cam_T_world[:3, :3].T + cam_T_world[:3, 3], camera).reshape(-1, 133, 2).astype(np.float32)
        labels: dict[str, int] = {label: cam for cam, (_, _, label, _) in enumerate(ARIA_STREAMS) if label is not None}
        for label, calibration in cameras.aria_cameras().items():
            # Per frame: cam_T_world = inv(world_T_device @ device_T_camera), applied to that frame's joints at once.
            cam_T_world: Float64[ndarray, "t 4 4"] = np.linalg.inv(world_T_device @ calibration.rig_T_cam.matrix())
            local: Float64[ndarray, "t 133 3"] = (
                np.einsum("tij,tkj->tki", cam_T_world[:, :3, :3], positions.astype(np.float64)) + cam_T_world[:, None, :3, 3]
            )
            projected: Float64[ndarray, "p 2"] = np.full((len(flat), 2), np.nan)
            points: Float64[ndarray, "p 3"] = local.reshape(-1, 3)
            usable = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
            projected[usable] = fisheye624.project(calibration, points[usable])
            yield (EGO_RIG, labels[label]), projected.reshape(-1, 133, 2).astype(np.float32)

    confidence: Float32[ndarray, "t 133"] = np.isfinite(positions).all(axis=-1).astype(np.float32)
    hands.log_projections(
        recording,
        pixels(),
        times_ns=frames.times_ns[rows],
        frame_indices=rows,
        confidence=confidence,
        camera_model=f"{KB4} (GoPros), {FISHEYE624} (Aria)",
        calibration_source="gopro_calibs.csv; Aria factory calib_json",
    )
