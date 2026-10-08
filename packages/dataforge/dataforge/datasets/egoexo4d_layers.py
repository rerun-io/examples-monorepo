"""Ego-Exo4D layer writers: the base layer from the raw take, its sidecars, and the lens-model projections.

Rig 0 is the wearer's Aria (moving, device frame, the closed-loop trajectory as ``world_T_rig``) with
RGB, SLAM left/right and the eye-tracking camera; rigs 1..N are the localized GoPros, static, in
``gopro_calibs.csv`` order. Every stream logs on the take clock (Aria RGB capture time) at 30 Hz.

Right after base is published the dataset writes one sidecar, ``take.npz``, so the derived layers never read the raw take:
the frame times, the device pose at each, and the camera record (GoPro rows, Aria calibration document) as JSON.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import SerdeError, from_dict, serde
from serde.json import to_json
from simplecv.camera_parameters import Fisheye62Parameters, Fisheye624Parameters
from simplecv.sensors.camera import fisheye624
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge import aria, hands, schema, writing
from dataforge.datasets.egoexo4d_body import HmFit, coco133_from_openpose67
from dataforge.datasets.egoexo4d_source import FPS, KB4, GoproCalib, Take, frame_peaks, sample_phase, stored_size
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import annotation_context, log_camera_node, log_camera_source, log_dense_pose_track, log_rig_node, log_video_stream
from dataforge.records import decode
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, mp4_frame_count, parallel_clips, transcode_mp4, work_dir

EGO_RIG: int = 0
"""The Aria; fixed at 0 so the blueprint can name it whatever the number of GoPros."""
EXO_SLOTS: int = 5
"""GoPro panes the dataset blueprint lays out (rigs 1..5); Ego-Exo4D captures carry four or five."""
FISHEYE624: str = "FISHEYE624"
"""``camera_model`` of the Aria projections; the camera nodes log its Fisheye62 reduction, as hot3d's do."""
IMAGE_ROTATION_CW_DEG: int = 90
"""Ego-Exo4D's Aria MP4s are a quarter turn clockwise from the sensor readout; the calibration is turned to match."""
SIDECAR: str = "take.npz"
"""One file, so the clock, poses and cameras of a take always come from the same base."""


class AriaStream(NamedTuple):
    """One Aria camera stream of a take."""

    readable: str
    """Its key in takes.json ``frame_aligned_videos``."""
    stream_id: str
    """VRS stream id, also the MP4 name suffix."""
    label: str | None
    """Factory calibration label; None for the eye cameras, which share one frame and so have no single calibration."""
    gray: bool
    """Monochrome sensor."""
    step: int
    """Frame-aligned frames per real sample: 3 for the 10 Hz eye cameras, whose 30 Hz video pads each image with black."""


ARIA_STREAMS: tuple[AriaStream, ...] = (
    AriaStream("rgb", aria.RGB_STREAM_ID, aria.STREAM_LABELS[aria.RGB_STREAM_ID], False, 1),
    AriaStream("slam-left", aria.SLAM_LEFT_STREAM_ID, aria.STREAM_LABELS[aria.SLAM_LEFT_STREAM_ID], True, 1),
    AriaStream("slam-right", aria.SLAM_RIGHT_STREAM_ID, aria.STREAM_LABELS[aria.SLAM_RIGHT_STREAM_ID], True, 1),
    AriaStream("et", "211-1", None, True, 3),
)
"""The Aria cameras in ``cam_MM`` order."""


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
        device: aria.DeviceCalibration = aria.DeviceCalibration.from_json(self.aria_calib_json, SIDECAR)
        return {label: logged_calibration(device.camera(label), width, height) for label, (width, height) in self.aria_sizes.items()}


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FrameSidecar:
    """The take clock and the Aria pose on it."""

    times_ns: Int64[ndarray, "n"]
    """Take clock, ns."""
    world_T_device: Float64[ndarray, "n 4 4"]
    """Aria pose per frame, NaN where the trajectory has a gap."""


def logged_calibration(factory: Fisheye624Parameters, width: int, height: int) -> Fisheye624Parameters:
    """A factory camera fitted to a stream stored ``width`` x ``height`` after the quarter turn."""
    return fisheye624.rotate_cw90(aria.rescale_to_stream(factory, height, width))


def restored_size(readout: Fisheye624Parameters, width: int, height: int) -> tuple[int, int] | None:
    """The upright size to store an Aria MP4 at when the release resized its upright image back to the readout's shape.

    The MP4s are a quarter turn from the readout, so a non-square camera ships portrait (SLAM 480x640). About one take in
    seven (seen at IIITH, NUS, Uniandes and UPenn) ships its SLAM videos at the readout's landscape 640x480 instead: the
    upright image, stretched. None when the MP4 needs no rescale.
    """
    if width != height and (width > height) == (readout.width > readout.height):
        return height, width
    return None


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
    frames: FrameSidecar
    """Take clock and the Aria pose at each frame (a preview's first frames)."""
    take_frames: int
    """Frames in the take; every source video must hold exactly this many."""
    clock_filled: int
    """Frames whose timesync stamp was missing and repeats the previous one."""


class Slot(NamedTuple):
    """One camera to encode and log: where it goes, what to encode, and how to log its node."""

    rig: int
    cam: int
    source: Path
    """Frame-aligned MP4."""
    gray: bool
    size: tuple[int, int] | None
    """Stored size when rescaled (the GoPros)."""
    step: int
    """Frame-aligned frames per real sample (``AriaStream.step``)."""
    log_node: Callable[[int, int], str]
    """Logs the camera node given the source width and height; returns its ``source_resolution`` entry."""


def write_base(recording: rr.RecordingStream, identity: SequenceIdentity, inputs: BaseInputs, timer: SequenceTimer) -> CamerasSidecar:
    """Encode every stream, log cameras, videos and the Aria trajectory; return the cameras sidecar."""
    take: Take = inputs.take
    times: Int64[ndarray, "n"] = inputs.frames.times_ns
    frames: Int64[ndarray, "n"] = np.arange(len(times), dtype=np.int64)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)  # MPS worlds are gravity-aligned, z up
    rr.log("/", annotation_context(), static=True, recording=recording)
    device: aria.DeviceCalibration = aria.DeviceCalibration.from_json(inputs.calib_json, f"{take.take_name} calib_json")
    aria_sizes: dict[str, tuple[int, int]] = {}
    # The viewer fits its eye to the layout (a kitchen, a soccer pitch), so a GoPro frustum keeps one on-screen size only if it
    # grows with the layout.
    centers_xy: Float64[ndarray, "c 2"] = np.array([calib.world_T_cam[:2, 3] for calib in inputs.gopros])
    frustum_m: float = max(0.1, 0.05 * float(np.linalg.norm(centers_xy - centers_xy.mean(axis=0), axis=1).max()))

    def log_aria(cam: int, stream: AriaStream, size: tuple[int, int] | None, width: int, height: int) -> str:
        if stream.label is None:  # both eye cameras in one frame: no single calibration, so no pinhole
            rr.log(schema.cam_path(EGO_RIG, cam), rr.AnyValues(name="camera-et", kind="grayscale"), static=True, recording=recording)
        else:
            aria_sizes[stream.label] = size or (width, height)
            log_camera_node(
                recording,
                EGO_RIG,
                cam,
                logged_calibration(device.camera(stream.label), *aria_sizes[stream.label]).to_fisheye62(),
                name=stream.label,
                kind="grayscale" if stream.gray else "rgb",
                image_plane_distance=0.05,
                camera_model="FISHEYE624 (thin-prism omitted)",
                image_rotation_cw_deg=IMAGE_ROTATION_CW_DEG,
            )
        name: str = f"{take.aria}_{stream.stream_id}"
        return log_camera_source(
            recording,
            EGO_RIG,
            cam,
            name=name,
            source_width=width,
            source_height=height,
            stored_width=None if size is None else size[0],
            stored_height=None if size is None else size[1],
            video_codec="av1",
            cq=AV1_CQ,
            gop=AV1_GOP,
            stream_id=stream.stream_id,
        )

    def log_gopro(rig: int, calib: GoproCalib, size: tuple[int, int], width: int, height: int) -> str:
        log_camera_node(recording, rig, 0, calib.camera(*size), name=calib.cam_uid, kind="rgb", image_plane_distance=frustum_m, camera_model=KB4)
        return log_camera_source(
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

    slots: list[Slot] = []
    for cam, stream in enumerate(ARIA_STREAMS):
        source: Path = inputs.root / take.video(take.aria, stream.readable)
        restored: tuple[int, int] | None = None if stream.label is None else restored_size(device.camera(stream.label), *video_size(source))
        slots.append(Slot(EGO_RIG, cam, source, stream.gray, restored, stream.step, partial(log_aria, cam, stream, restored)))
    for rig, calib in enumerate(inputs.gopros, start=1):
        size: tuple[int, int] = stored_size(calib)
        slots.append(Slot(rig, 0, inputs.root / take.video(calib.cam_uid, "0"), False, size, 1, partial(log_gopro, rig, calib, size)))
    log_rig_node(recording, EGO_RIG, reference=None, num_cameras=len(ARIA_STREAMS), name=take.aria, kind="ego")
    for rig, calib in enumerate(inputs.gopros, start=1):
        log_rig_node(recording, rig, reference=None, num_cameras=1, name=calib.cam_uid, kind="exo")

    samples: list[Int64[ndarray, "m"]] = []
    for slot in slots:  # -frames:v would silently cut a longer source, which is then deleted
        count: int = mp4_frame_count(slot.source)
        if count != inputs.take_frames:
            raise ValueError(f"{slot.source}: {count} frames, the take's timesync rows give {inputs.take_frames}")
        phase: int = 0 if slot.step == 1 else sample_phase(frame_peaks(slot.source, len(times)), slot.step, str(slot.source))
        samples.append(frames[phase :: slot.step])
    resolutions: list[str] = []
    with work_dir("egoexo4d-") as work:
        clips: list[Path] = [work / f"rig_{slot.rig:02d}_cam_{slot.cam:02d}.mp4" for slot in slots]

        def encode(slot: Slot, kept: Int64[ndarray, "m"], clip: Path) -> None:
            every: tuple[int, int] | None = None if slot.step == 1 else (slot.step, int(kept[0]))
            transcode_mp4(
                slot.source, clip, gop=AV1_GOP, cq=AV1_CQ, fps=FPS, frames=len(kept), gray=slot.gray, size=slot.size, every=every, decode="cuda"
            )

        jobs: list[tuple[Path, Callable[[], None]]] = [
            (clip, partial(encode, slot, kept, clip)) for slot, kept, clip in zip(slots, samples, clips, strict=True)
        ]
        with parallel_clips(jobs, timer, workers=6) as encoded:
            for slot, kept, clip in zip(slots, samples, encoded, strict=True):
                resolutions.append(slot.log_node(*video_size(slot.source)))
                with timer.stage("write:video"):  # a padded stream keeps its real samples' times on the shared frame timeline
                    log_video_stream(recording, clip, schema.video_path(slot.rig, slot.cam), times_ns=times[kept], frame_indices=kept)
    log_dense_pose_track(recording, schema.rig_path(EGO_RIG), times_ns=times, frame_indices=frames, transforms=inputs.frames.world_T_device)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(times),
        num_cameras=len(slots),
        source_num_frames=inputs.take_frames,
        clock_source=f"timesync.csv {take.aria}_{aria.RGB_STREAM_ID}_capture_timestamp_ns (Aria device clock), rows timesync_start_idx..timesync_end_idx-1",
        clock_filled_frames=inputs.clock_filled,
        source_resolution=pa.array(resolutions),
        image_rotation_cw_deg=IMAGE_ROTATION_CW_DEG,
        trajectory_coverage=float(np.isfinite(inputs.frames.world_T_device).all(axis=(1, 2)).mean()),
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


def write_sidecar(path: Path, cameras: CamerasSidecar, frames: FrameSidecar) -> None:
    """Publish the take's sidecar atomically; call it after base is published (``read_sidecar`` relies on the order)."""
    with writing.atomic_write(path) as staged, staged.open("wb") as stream:
        np.savez(stream, times_ns=frames.times_ns, world_T_device=frames.world_T_device, cameras_json=np.array(to_json(cameras)))


def read_sidecar(path: Path, base: Path) -> tuple[FrameSidecar, CamerasSidecar]:
    """The sidecar of the base at ``base``; one written before that base (a rebuild that failed in between) is refused."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: convert the base layer first (--force rebuilds it from the raw take)")
    if path.stat().st_mtime_ns < base.stat().st_mtime_ns:
        raise ValueError(f"{path} predates {base}, so it may describe an older base: reconvert base with --force")
    with np.load(path, allow_pickle=False) as npz:
        try:
            frames: FrameSidecar = from_dict(FrameSidecar, {"times_ns": npz["times_ns"], "world_T_device": npz["world_T_device"]})
        except SerdeError as error:
            raise ValueError(f"{path}: {error}") from error
        cameras: CamerasSidecar = decode(CamerasSidecar, str(npz["cameras_json"]), source=f"{path}:cameras_json")
    return frames, cameras


def write_projections(recording: rr.RecordingStream, fit: HmFit, frames: FrameSidecar, cameras: CamerasSidecar, rows: Int64[ndarray, "t"]) -> None:
    """The fit's COCO-133 keypoints through every calibrated camera's full lens model (KB4 GoPros, FISHEYE624 Aria)."""
    positions: Float32[ndarray, "t 133 3"] = coco133_from_openpose67(fit.joints3d[rows])
    positions[~fit.valid[rows]] = np.nan
    world_T_device: Float64[ndarray, "t 4 4"] = frames.world_T_device[rows]

    def pixels() -> Iterator[tuple[tuple[int, int], Float32[ndarray, "t 133 2"]]]:
        flat: Float64[ndarray, "p 3"] = positions.reshape(-1, 3).astype(np.float64)
        for rig, camera in enumerate(cameras.exo_cameras(), start=1):
            cam_T_world: Float64[ndarray, "4 4"] = camera.extrinsics.cam_T_world
            yield (rig, 0), project_fisheye62(flat @ cam_T_world[:3, :3].T + cam_T_world[:3, 3], camera).reshape(-1, 133, 2).astype(np.float32)
        calibrations: dict[str, Fisheye624Parameters] = cameras.aria_cameras()
        for cam, stream in enumerate(ARIA_STREAMS):
            if stream.label in calibrations:
                yield (EGO_RIG, cam), aria.project_frames(calibrations[stream.label], world_T_device, positions)

    hands.log_projections(
        recording,
        pixels(),
        times_ns=frames.times_ns[rows],
        frame_indices=rows,
        confidence=np.isfinite(positions).all(axis=-1).astype(np.float32),
        camera_model=f"{KB4} (GoPros), {FISHEYE624} (Aria)",
        calibration_source="gopro_calibs.csv; Aria factory calib_json",
    )
