"""Insta360 dual-fisheye ``.insv`` clips: one base recording per clip, and on request a derived panorama layer.

The clip is read by ``insta360_io`` (packages/insta360-io: kornia-staging's ``.insv`` reader and CPU stitch). Its two
HEVC lens tracks are remuxed, not re-encoded: each sample becomes an Annex B access unit with the parameter sets in
front of every IRAP. ``cam_00`` and ``cam_01`` are lens 0 and lens 1, whichever track holds them (``stream_id``).

Cameras. Each lens is a camera with the factory model itself, the unified (Mei) sphere with its distortion, written
like TartanAir writes ``equirectangular``: ``DistortionModel`` ``mei_v6`` (or ``mei_v3`` for a clip without a V6 lens
string) and ``DistortionCoefficients`` ``xi, half_field_rad, d0 .. d12`` (``mei_v3``: ``xi, half_field_rad, k1, k2,
k3, p1, p2``), with ``fx, fy, cx, cy`` in the pinhole matrix (track pixels, window crop applied). A ray further than
``half_field_rad`` from the optical axis does not project.

Rig frame = IMU frame (``reference = 'imu_00'``). Each lens is posed as ``T_imu_lens = T_imu_P * T_P_lens``, with
``T_P_lens`` the factory rotation and centre in the vendor rig frame ``P`` (``InsvLens.pano_from_lens_rotation``,
``pano_centre_m``: +Z the nominal lens-0 axis, +Y toward the handle, origin at the midpoint of the lens centres) and
``T_imu_P = (R_imu_P, R_imu_P * lever_arm)`` from the device profile.

Clock. ``video_time`` is the trailer clock, counted from the earliest frame or IMU sample (``capture:start_time_ns``).
IMU times are the trailer's, without the vendor's ``gyro_timestamp`` subtraction: the profile's fit measured on that
clock. Frame ``i`` is stamped at its effective time on the IMU clock, the exposure-record-4 stamp plus the profile's
``frame_time_shutter_slope * shutter_i + frame_time_constant_s``: the mid-exposure time of the lens image's MIDDLE row.

Rolling shutter. Both lenses read rows top to bottom; each camera node states its readout (first row to last row) as
``rolling_shutter_readout_s``, so row ``v`` of an ``H``-row image is read at ``t + readout_s * (v - (H - 1) / 2) / H``.
The value is the clip's own ``rolling_shutter_ms`` when it states one (the profile must agree within its tolerance),
else the profile's.

Exposure. Each camera's per-frame exposure time is a series at ``cam_NN/exposure``: record 4 (VIN0) for lens 0 and
record 12 (VIN1) for lens 1 (record 4 again on a clip without record 12, the X5), on the frame times.

Device profile. What the file does not carry (``R_P_imu``, the lever arm, the frame-time model, the readout, the IMU
noise) is a measured record in ``insta360_profiles.toml``, selected by the clip's camera type, lens type, frame rate
(from the exposure stamps) and readout; a clip outside every profile's measured mode is refused, naming the property
that differs. The IMU noise model is a placeholder, marked as such in its ``source``.

Panorama layer (``--panorama``; off by default). ``panorama`` holds the 1920x960 equirectangular panorama at
``/world/rig_00/pano``, stitched on the CPU from the two lens images of every frameset and encoded AV1 on NVENC, on
the frame times. It is a derived output: nothing reads it as input, slam-rs tracks the two lens cameras of the base,
and the lens brightness steps at the seams are not corrected. The stitch reads the two cameras from the base
recording as written (float32 statics), so a later ``--panorama`` run adds the layer beside an existing base without
writing the base again. On clip B (X4 Air, 1920) the base alone takes about 13 s, with the panorama about 131 s.

Cube layer (``--cube``; off by default). ``cube`` holds the six-face cubemap of ``insta360_cube`` for every
``cube_stride``-th frameset: ``N x N`` pinholes at ONE centre (the midpoint of the two lens centres), cube axes = camera
0's, side faces joined at their centre line from the two lens images (no blend). ``/world/rig_00/cam_02 .. cam_07`` =
``cube6_front0``, ``cube6_front1``, ``cube6_left``, ``cube6_right``, ``cube6_up``, ``cube6_down``, each with its name on
the node, a static face-from-IMU ``Transform3D`` (``ChildFromParent``), a plain ``Pinhole`` (RDF) and an RGB AV1
``VideoStream`` on the source frameset's stamp. The tables come from the base recording's two cameras through
``MeiCamera.project``, so the layer can be added beside an existing base. On ``cube_device`` ``cuda`` (default) the
stage stays on the GPU: NVDEC decode, one remap per face, NVENC from device memory; ``cpu`` runs the same code on the CPU.

Panorama frame ``E``. Axes = camera 0's (RDF), origin = the midpoint of the two camera centres. Pixel ``(u, v)``
(pixel centres at integers) looks along longitude ``lon = (u + 0.5) pi / 960 - pi / 2``, ``lon = atan2(x, z)`` in
``E``, and latitude ``lat = (v + 0.5) pi / 960 - pi / 2``, ``+y`` down: the left half is camera 0's hemisphere (centre
column 480 = camera 0's ``+Z``), the right half camera 1's (column 1440 = ``-Z``). Each half is stitched from its own
camera centre (only camera 1's rotation in camera 0 enters the tables), so the panorama ignores the lens-centre
baseline: 33.3 mm on the X4 Air, 31.9 mm on the X5.

From the native poses to the panorama. ``E`` is fixed in the rig: ``T_cam0_E = (I, c1 / 2)``, with ``c1`` camera 1's
centre in camera 0 (the translation of ``cam0_T_cam1``, from the ``cam_00`` and ``cam_01`` ``Transform3D``s,
``ChildFromParent``, as ``stitch_from_recording`` reads them). A COLMAP ``EQUIRECTANGULAR`` camera ``C`` (single
centre; ``u = W (lon_C + pi) / (2 pi)`` in pixel-corner coordinates, image centre = ``+Z_C``, ``+y`` down) sees this
panorama with its columns a quarter turn on, ``lon_E = lon_C + pi / 2``, so ``T_cam0_C = (Ry(+90 deg), c1 / 2)`` with
``Ry(+90 deg) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]`` (``+Z_C`` is camera 0's ``+X``). slam-rs reports ``world_from_rig``
in the IMU frame ``imu_00`` (the rig reference), so a tracked pose ``T_world_imu`` becomes
``T_world_C = T_world_imu * T_imu_cam0 * T_cam0_C`` (COLMAP's ``cam_from_world`` is its inverse), with ``T_imu_cam0``
the inverse of the ``cam_00`` ``Transform3D``. Every factor is a static of the base, so the transform needs no file of
its own; with the profiles' zero lever arm the midpoint, and so the panorama centre, is the IMU origin.
"""

from __future__ import annotations

import contextlib
import itertools
import tomllib
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import av
import insta360_io
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
import torch
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from serde import SerdeError, coerce, serde
from serde.toml import from_toml
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.imu_calibration import ImuCalibration
from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import Encoder, VideoStream

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets import insta360_cube as cube
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import ImuChannel, log_camera_source, log_imu, log_rig_node, log_video_stream, time_column
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, nvenc_slot, work_dir

RIG: int = 0
"""One rig per clip."""
PANORAMA_LAYER: str = "panorama"
"""The derived panorama, written only under ``--panorama``: an AV1 ``VideoStream`` at ``PANO_ENTITY`` on the frame times."""
PANO_ENTITY: str = f"{schema.rig_path(RIG)}/pano"
"""Where the panorama layer puts its video."""
PANORAMA_SIZE: tuple[int, int] = (1920, 960)
"""Width and height of ``insta360_io.PanoramaStitch.panorama``."""
CUBE_LAYER: str = "cube"
"""The cubemap, written only under ``--cube``: six faces per sampled frameset, ``cam_02 .. cam_07``."""
CUBE_BATCH: int = 16
"""Framesets the cube layer decodes, joins and encodes per step."""
FIRST_FACE_CAM: int = 2
"""Camera index of the first face; face ``k`` of ``insta360_cube.FACES`` is ``2 + k``."""
FACE_CAMS: range = range(FIRST_FACE_CAM, FIRST_FACE_CAM + len(cube.FACES))
"""``cam_02 .. cam_07``: the cube faces."""
RUN_SOURCE: str = "slam_rs"
"""The run layer the layouts show: slam-rs tracks these clips."""
PROFILES_TOML: Path = Path(__file__).with_name("insta360_profiles.toml")
"""The measured device profiles."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class ImuNoise:
    """An IMU noise model as ``simplecv.imu_calibration.ImuCalibration`` takes it."""

    gyro_noise_density: float
    """rad/s/sqrt(Hz)."""
    accel_noise_density: float
    """m/s^2/sqrt(Hz)."""
    gyro_bias_random_walk: float
    """rad/s^2/sqrt(Hz)."""
    accel_bias_random_walk: float
    """m/s^3/sqrt(Hz)."""
    rate_hz: float
    """Sample rate the densities are stated for."""
    source: str
    """Where the numbers come from."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """The calibration of one Insta360 body in one video mode that the file does not carry."""

    name: str
    """Profile id, written into every recording converted with it."""
    version: int
    """Bumped whenever a measured number changes."""
    provenance: str
    """Where the numbers were measured."""
    camera_type: str
    """``InsvClip.camera_type`` of the body."""
    lens_type: int
    """Lens type of the factory lens strings (131 X4 Air, 113 X5). The lens size is not a property of the profile:
    the reader states the factory lenses at the track resolution, so one profile serves every resolution."""
    frame_rate_hz: float
    """Frame rate of the measured mode."""
    frame_rate_tolerance_hz: float
    """Largest difference from ``frame_rate_hz`` a clip may have."""
    readout_s: float
    """Rolling-shutter readout of one lens image, seconds (first row to last)."""
    readout_tolerance_s: float
    """Largest difference between a clip's own readout and ``readout_s``."""
    pano_from_imu_measured: Float64[ndarray, "3 3"]
    """``R_P_imu`` as fitted (rows: P axes in IMU coordinates); orthonormalised before use."""
    lever_arm_m: Float64[ndarray, "3"]
    """Vector from the IMU to the lens midpoint, metres, in P."""
    frame_time_shutter_slope: float
    """``a`` of the per-frame offset ``a * shutter_i + b`` (shutter = exposure record 4 duration, s)."""
    frame_time_constant_s: float
    """``b`` of the same model, seconds."""
    imu_noise: ImuNoise
    """The IMU noise model."""

    def __post_init__(self) -> None:
        if self.frame_rate_tolerance_hz < 0.0 or self.readout_tolerance_s < 0.0:
            raise ValueError(f"profile {self.name}: a tolerance is negative")
        if self.frame_rate_hz <= 0.0 or self.readout_s <= 0.0:
            raise ValueError(f"profile {self.name}: frame rate and readout must be positive")


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class DeviceProfiles:
    """``insta360_profiles.toml``: every measured body and mode."""

    profile: list[DeviceProfile]
    """The profiles, as ``[[profile]]`` tables."""


def load_profiles(path: Path = PROFILES_TOML) -> list[DeviceProfile]:
    """Every profile of a profile file.

    Raises:
        ValueError: If the file is not TOML or does not hold the ``DeviceProfile`` schema; the message names the file.
    """
    try:
        return from_toml(DeviceProfiles, path.read_text()).profile
    except (SerdeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{path}: not an Insta360 device profile file: {error}") from error


@dataclass(frozen=True, slots=True)
class CaptureMode:
    """What a clip states about the mode it was recorded in, the properties a profile is selected by."""

    camera_type: str
    """``InsvClip.camera_type``."""
    lens_type: int
    """The lens type both lenses of the V3 factory string state."""
    frame_rate_hz: float
    """From the median exposure-record-4 stamp interval."""
    readout_s: float | None
    """The clip's own ``rolling_shutter_ms``, in seconds; None when the clip does not state one."""


def mode_differences(profile: DeviceProfile, mode: CaptureMode) -> list[str]:
    """Each property in which ``mode`` lies outside what ``profile`` was measured on, as a readable reason."""
    differences: list[str] = []
    if mode.camera_type != profile.camera_type:
        differences.append(f"camera type {mode.camera_type!r} is not {profile.camera_type!r}")
    if mode.lens_type != profile.lens_type:
        differences.append(f"lens type {mode.lens_type} is not {profile.lens_type}")
    if abs(mode.frame_rate_hz - profile.frame_rate_hz) > profile.frame_rate_tolerance_hz:
        differences.append(f"frame rate {mode.frame_rate_hz:.4f} Hz is not {profile.frame_rate_hz} +- {profile.frame_rate_tolerance_hz} Hz")
    if mode.readout_s is not None and abs(mode.readout_s - profile.readout_s) > profile.readout_tolerance_s:
        differences.append(f"rolling-shutter readout {mode.readout_s * 1e3:.4f} ms is not {profile.readout_s * 1e3:.4f} +- {profile.readout_tolerance_s * 1e3:.4f} ms")
    return differences


def select_profile(profiles: list[DeviceProfile], mode: CaptureMode) -> DeviceProfile:
    """The profile measured in ``mode``.

    Raises:
        ValueError: If no profile matches; the message names, per profile of the same camera type, what differs.
    """
    reasons: list[str] = []
    for profile in profiles:
        differences: list[str] = mode_differences(profile, mode)
        if not differences:
            return profile
        if profile.camera_type == mode.camera_type:
            reasons.append(f"{profile.name}: {'; '.join(differences)}")
    if not reasons:
        raise ValueError(f"no device profile for a {mode.camera_type!r} (profiles: {sorted(p.name for p in profiles)})")
    raise ValueError(f"the clip is outside every measured {mode.camera_type!r} mode: {' | '.join(reasons)}")


def orthonormal(matrix: Float64[ndarray, "3 3"]) -> Float64[ndarray, "3 3"]:
    """The rotation nearest to ``matrix`` (SVD polar factor, determinant +1)."""
    u, _, vt = np.linalg.svd(matrix)
    rotation: Float64[ndarray, "3 3"] = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return rotation


def frame_times_ns(record_t_us: Int64[ndarray, " n"], shutter_s: Float64[ndarray, " n"], profile: DeviceProfile) -> Int64[ndarray, " n"]:
    """Each frame's effective time on the IMU clock: the record-4 stamp plus the profile's per-frame offset.

    Raises:
        ValueError: If the times are not strictly increasing.
    """
    shift_s: Float64[ndarray, " n"] = profile.frame_time_shutter_slope * shutter_s + profile.frame_time_constant_s
    times: Int64[ndarray, " n"] = record_t_us * 1000 + np.round(shift_s * 1e9).astype(np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("the per-frame time model makes frame times non-increasing")
    return times


def imu_from_lens(lens: insta360_io.InsvLens, imu_from_pano: Float64[ndarray, "3 3"], lever_arm_m: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """``T_imu_lens = T_imu_P * T_P_lens``, with ``T_imu_P = (R_imu_P, R_imu_P * lever_arm)``."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = imu_from_pano @ np.asarray(lens.pano_from_lens_rotation, dtype=np.float64)
    pose[:3, 3] = imu_from_pano @ (np.asarray(lens.pano_centre_m, dtype=np.float64) + lever_arm_m)
    return pose


def gray_frames(clip: insta360_io.InsvClip, track: int, frames: int, *, batch_frames: int, threads: int) -> Iterator[UInt8[ndarray, "h w"]]:
    """The first ``frames`` pictures of one lens track as C-contiguous ``gray8``, read ``batch_frames`` samples at a time.

    ``reformat(format="gray8")`` as slam-rs's feed decodes, so the panorama is stitched from the pixels the tracker sees.
    """
    for frame in _hevc_pictures(clip, track, frames, batch_frames=batch_frames, threads=threads):
        converted: av.VideoFrame = frame.reformat(format="gray8")
        plane = converted.planes[0]
        padded: UInt8[ndarray, "h stride"] = np.frombuffer(plane, dtype=np.uint8).reshape(converted.height, plane.line_size)
        yield padded[:, : converted.width].copy()


def _hevc_pictures(clip: insta360_io.InsvClip, track: int, frames: int, *, batch_frames: int, threads: int) -> Iterator[av.VideoFrame]:
    """The pictures of the first ``frames`` samples of one lens track, read ``batch_frames`` samples at a time."""
    decoder = av.CodecContext.create("hevc", "r")
    decoder.thread_count = threads
    decoder.thread_type = "NONE" if threads == 1 else "FRAME"
    for first in range(0, frames, batch_frames):
        samples, _ = clip.read_samples(track, first, min(first + batch_frames, frames))
        for sample in samples:
            yield from decoder.decode(av.Packet(sample))
    yield from decoder.decode(None)


@dataclass(frozen=True, slots=True)
class LensCameras:
    """The two lens cameras of a base recording as written (float32 statics)."""

    cameras: list[insta360_io.MeiCamera]
    """Lens 0 and lens 1."""
    lens_from_rig: list[Float64[ndarray, "4 4"]]
    """``T_lens_imu`` of each lens, as the ``ChildFromParent`` ``Transform3D`` states it."""
    size: tuple[int, int]
    """Lens image width and height."""

    @property
    def lens1_from_lens0(self) -> Float64[ndarray, "3 3"]:
        """Lens 1's rotation from lens 0."""
        return self.lens_from_rig[1][:3, :3] @ self.lens_from_rig[0][:3, :3].T


def cameras_from_recording(base: Path) -> LensCameras:
    """A base recording's ``cam_00`` and ``cam_01``: each its ``DistortionModel``, ``DistortionCoefficients``, pinhole matrix and pose.

    Raises:
        ValueError: If a camera is not a Mei camera or its statics are missing.
    """
    entities: list[str] = [schema.cam_path(RIG, cam) for cam in range(2)] + [schema.pinhole_path(RIG, cam) for cam in range(2)]
    statics: dict[str, object] = {}
    for chunk in rrc.RrdReader(base).stream():
        if chunk.is_static and chunk.entity_path in entities:
            batch = chunk.to_record_batch()
            statics.update({f"{chunk.entity_path}:{name}": batch.column(name).to_pylist()[0] for name in batch.schema.names if not name.startswith("rerun")})
    cameras: list[insta360_io.MeiCamera] = []
    cam_T_rig: list[Float64[ndarray, "4 4"]] = []
    for cam in range(2):
        pinhole: str = schema.pinhole_path(RIG, cam)
        node: str = schema.cam_path(RIG, cam)
        missing: list[str] = [
            key
            for key in (f"{pinhole}:simplecv.components.DistortionModel", f"{pinhole}:simplecv.components.DistortionCoefficients", f"{pinhole}:Pinhole:image_from_camera", f"{node}:Transform3D:mat3x3", f"{node}:Transform3D:translation")
            if not statics.get(key)
        ]
        if missing:
            raise ValueError(f"{base}: camera {cam} lacks {missing}")
        model: str = str(statics[f"{pinhole}:simplecv.components.DistortionModel"][0])  # pyrefly: ignore[bad-index]
        coefficients: list[float] = list(statics[f"{pinhole}:simplecv.components.DistortionCoefficients"][0])  # pyrefly: ignore[bad-index]
        # Rerun stores the 3x3 matrices column-major.
        k: Float64[ndarray, "3 3"] = np.asarray(statics[f"{pinhole}:Pinhole:image_from_camera"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        cameras.append(insta360_io.MeiCamera(model, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), coefficients))
        pose: Float64[ndarray, "4 4"] = np.eye(4)
        pose[:3, :3] = np.asarray(statics[f"{node}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        pose[:3, 3] = np.asarray(statics[f"{node}:Transform3D:translation"][0], dtype=np.float64)  # pyrefly: ignore[bad-index]
        cam_T_rig.append(pose)
    width, height = (int(v) for v in statics[f"{schema.pinhole_path(RIG, 0)}:Pinhole:resolution"][0])  # pyrefly: ignore[bad-index]
    return LensCameras(cameras=cameras, lens_from_rig=cam_T_rig, size=(width, height))


def stitch_from_recording(base: Path) -> insta360_io.PanoramaStitch:
    """The panorama stitch of a base recording's two cameras; camera 1's pose in camera 0 is ``cam0_T_rig * cam1_T_rig^-1``."""
    lenses: LensCameras = cameras_from_recording(base)
    cam0_T_cam1: Float64[ndarray, "4 4"] = lenses.lens_from_rig[0] @ np.linalg.inv(lenses.lens_from_rig[1])
    return insta360_io.PanoramaStitch(lenses.cameras[0], lenses.cameras[1], cam0_T_cam1.tolist(), lenses.size)


def rgb_frames(clip: insta360_io.InsvClip, track: int, indices: Int64[ndarray, " k"], *, threads: int) -> Iterator[UInt8[ndarray, "h w 3"]]:
    """The pictures ``indices`` (increasing) of one lens track as ``rgb24`` (PyAV, CPU); every picture up to the last is decoded."""
    wanted: set[int] = {int(i) for i in indices}
    stop: int = int(indices[-1]) + 1
    decoded: int = 0
    for frame in _hevc_pictures(clip, track, stop, batch_frames=300, threads=threads):
        if decoded in wanted:
            yield frame.to_ndarray(format="rgb24")
        decoded += 1
    if decoded < stop:
        raise ValueError(f"track {track} decoded {decoded} pictures of the {stop} the cube layer needs")


def lens_batches(clip: insta360_io.InsvClip, lens: int, indices: Int64[ndarray, " k"], device: torch.device, *, threads: int) -> Iterator[UInt8[Tensor, "b 3 h w"]]:
    """The RGB pictures ``indices`` (increasing) of one lens on ``device``, ``CUBE_BATCH`` at a time.

    ``cuda``: each batch's samples, from a keyframe at or before its first index, muxed into an in-memory MP4 (simplecv's
    ``wrap_mp4``) and decoded on NVDEC by torchcodec. ``cpu``: PyAV with ``threads`` frame threads, as torchcodec's CPU
    decoder runs about 1.6x slower here (it seeks per picture; measured on clip B).
    """
    track: int = clip.lens_track[lens]
    if device.type == "cpu":
        frames: Iterator[UInt8[ndarray, "h w 3"]] = rgb_frames(clip, track, indices, threads=threads)
        for _ in range(0, indices.size, CUBE_BATCH):
            yield torch.from_numpy(np.stack(list(itertools.islice(frames, CUBE_BATCH)))).permute(0, 3, 1, 2)
        return
    start: int = 0
    for first in range(0, indices.size, CUBE_BATCH):
        wanted: Int64[ndarray, " b"] = indices[first : first + CUBE_BATCH]
        samples, keyframes = clip.read_samples(track, start, int(wanted[-1]) + 1)
        decoder = VideoDecoder(wrap_mp4(samples, keyframes.tolist(), 30, codec="hevc"), device=str(device), seek_mode="exact")
        yield decoder.get_frames_at((wanted - start).tolist()).data
        start += int(np.flatnonzero(keyframes)[-1])


def lens_pictures(clip: insta360_io.InsvClip, indices: Int64[ndarray, " k"], device: torch.device, *, threads: int) -> Iterator[tuple[UInt8[Tensor, "b 3 h w"], UInt8[Tensor, "b 3 h w"]]]:
    """``lens_batches`` of both lenses, decoded side by side on two threads, one batch ahead of the caller."""
    batches: list[Iterator[UInt8[Tensor, "b 3 h w"]]] = [lens_batches(clip, lens, indices, device, threads=threads) for lens in range(2)]
    count: int = -(-indices.size // CUBE_BATCH)
    with ThreadPoolExecutor(2) as pool:
        ahead: list[Future[UInt8[Tensor, "b 3 h w"]]] = [pool.submit(next, batch) for batch in batches]
        for step in range(count):
            lens0, lens1 = (future.result() for future in ahead)
            if step + 1 < count:
                ahead = [pool.submit(next, batch) for batch in batches]
            yield lens0, lens1


def cube_indices(frames: int, stride: int, limit: int | None) -> Int64[ndarray, " k"]:
    """Framesets the cube layer samples: ``0, stride, 2 stride, ...``, the first ``limit`` of them.

    Raises:
        ValueError: If ``stride`` or ``limit`` is not positive.
    """
    if stride < 1 or (limit is not None and limit < 1):
        raise ValueError(f"cube stride {stride} and limit {limit} must be positive")
    return np.arange(0, frames, stride, dtype=np.int64)[:limit]


def encode_face_videos(
    faces: Iterable[UInt8[Tensor, "f b 3 n n"]], videos: list[Path], *, size: int, gop: int, cq: int, device: torch.device
) -> list[int]:
    """Encode batches of ``size`` x ``size`` RGB faces into one AV1 mp4 per face, one NVENC session each, fed from ``device``.

    Every encoder slot is reserved before the first encoder opens: a busy host waits for all of them at once instead of
    opening some encoders and stalling the rest.

    Returns:
        Frames each encoder took.

    Raises:
        ValueError: If ``DATAFORGE_NVENC_SLOTS`` is below the number of faces.
    """
    options: dict[str, int] = {"rc": 1, "cq": cq, "bf": 0, "g": gop}  # rc 1 = vbr; no B-frames
    fed: list[int] = [0] * len(videos)
    with nvenc_slot(len(videos)), contextlib.ExitStack() as opened:
        streams: list[VideoStream] = []
        for video in videos:
            encoder: Encoder = Encoder()
            streams.append(encoder.add_video(height=size, width=size, frame_rate=30, device=device, codec="av1_nvenc", preset="p4", extra_options=options))
            opened.enter_context(encoder.open_file(video))
        for batch in faces:
            for k, (stream, face) in enumerate(zip(streams, batch, strict=True)):
                stream.add_frames(face)
                fed[k] += face.shape[0]
    return fed


@dataclass(frozen=True, slots=True)
class ClipSource:
    """One ``.insv`` clip on disk."""

    path: Path
    """The clip."""


@dataclass(frozen=True, slots=True)
class Capture:
    """One clip read and checked, with its profile applied: everything both layers are written from."""

    clip: insta360_io.InsvClip
    """The open clip."""
    profile: DeviceProfile
    """The profile its mode selected."""
    readout_s: float
    """Rolling-shutter readout every camera node states."""
    lenses: list[insta360_io.InsvLens]
    """Lens 0 and lens 1: V6 when the clip has a V6 string, else V3."""
    frame_ns: Int64[ndarray, " n_frames"]
    """Frame times on the trailer clock, ns."""
    exposure_s: Float64[ndarray, "2 n_frames"]
    """Per-frame exposure time of lens 0 (record 4) and lens 1 (record 12)."""
    imu_ns: Int64[ndarray, " n_imu"]
    """IMU times on the trailer clock, ns."""
    accel: Float64[ndarray, "n_imu 3"]
    """Specific force, m/s^2, IMU axes."""
    gyro: Float64[ndarray, "n_imu 3"]
    """Rate, rad/s, IMU axes."""
    imu_from_pano: Float64[ndarray, "3 3"]
    """``R_imu_P``, the profile's measured rotation orthonormalised."""

    @property
    def start_ns(self) -> int:
        """The earliest frame or IMU time: ``video_time`` zero."""
        return int(min(self.imu_ns[0], self.frame_ns[0]))


def lens_type(v3: str) -> int:
    """The lens type both lenses of a V3 factory lens string state: the last per-lens token.

    Raises:
        ValueError: If the two lenses state different types.
    """
    tokens: list[str] = v3.split("_")
    count: int = int(float(tokens[0]))
    per_lens: int = (len(tokens) - 2) // count
    types: set[int] = {int(float(tokens[per_lens * (k + 1)])) for k in range(count)}
    if len(types) != 1:
        raise ValueError(f"the V3 lens string states lens types {sorted(types)}; a profile needs one")
    return types.pop()


def read_capture(path: Path, profiles: list[DeviceProfile], frame_limit: int | None) -> Capture:
    """Open ``path``, select its profile and compute the frame times.

    Raises:
        ValueError: If the clip lacks two lenses or a sensor record, or no profile was measured in its mode.
    """
    clip: insta360_io.InsvClip = insta360_io.InsvClip(str(path))
    frame_t_us, exposure_vin0 = clip.frame_exposures(False)
    _, exposure_vin1 = clip.frame_exposures(True)
    if frame_t_us.size < 2:
        raise ValueError(f"{path}: {frame_t_us.size} exposure-record-4 entries; a clip needs two frames or more")
    mode: CaptureMode = CaptureMode(
        camera_type=clip.camera_type,
        lens_type=lens_type(dict(clip.factory_strings)["v3"]),
        frame_rate_hz=1e6 / float(np.median(np.diff(frame_t_us))),
        readout_s=None if clip.rolling_shutter_ms is None else clip.rolling_shutter_ms * 1e-3,
    )
    profile: DeviceProfile = select_profile(profiles, mode)
    lenses: list[insta360_io.InsvLens] = clip.lenses_v6 or clip.lenses
    frames: int = clip.num_frames if frame_limit is None else min(frame_limit, clip.num_frames)
    frame_ns: Int64[ndarray, " n"] = frame_times_ns(frame_t_us[:frames], exposure_vin0[:frames], profile)
    imu_t_us, accel, gyro = clip.imu()
    imu_ns: Int64[ndarray, " m"] = imu_t_us * 1000
    if frame_limit is not None:
        keep = imu_ns <= frame_ns[-1] + 1_000_000_000
        imu_ns, accel, gyro = imu_ns[keep], accel[keep], gyro[keep]
    return Capture(
        clip=clip,
        profile=profile,
        readout_s=profile.readout_s if mode.readout_s is None else mode.readout_s,
        lenses=lenses,
        frame_ns=frame_ns,
        exposure_s=np.stack([exposure_vin0[:frames], exposure_vin1[:frames]]),
        imu_ns=imu_ns,
        accel=accel,
        gyro=gyro,
        imu_from_pano=orthonormal(profile.pano_from_imu_measured).T,
    )


@dataclass
class Insta360Config(FrameLimitedConfig):
    """Insta360 dual-fisheye ``.insv`` clips (local device captures) as a two-camera + IMU rig."""

    command: ClassVar[str] = "insta360"
    """Registry key and CLI subcommand."""

    _target: type = field(default_factory=lambda: Insta360Dataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "insta360")
    """Directory holding the ``.insv`` clips; each clip is one sequence."""
    batch_frames: int = 300
    """Samples read and sent per call, so a clip is never held in memory whole."""
    panorama: bool = False
    """Also write the panorama layer (CPU stitch + AV1 NVENC, about 10x the base's time). Run again with it to add the
    layer beside bases already written."""
    decode_threads: int = 4
    """PyAV threads per lens decoder of the panorama layer, and of the cube layer on ``cpu``."""
    pano_cq: int = 34
    """NVENC constant-quality target of the panorama: a viewing layer, not an analysis input."""
    pano_gop: int = 30
    """Panorama keyframe interval in frames, about one second."""
    cube: bool = False
    """Also write the cube layer: the six-face cubemap per sampled frameset (RGB remap + AV1 NVENC)."""
    cube_stride: int = 10
    """The cube layer samples every this-many framesets, from frameset 0."""
    cube_max_framesets: int | None = None
    """Keep only the first this-many sampled framesets (a sample pack); None keeps all."""
    cube_face_size: int | None = None
    """Face side ``N`` (a multiple of 32); None = ``lens_size * 800 / 1920``."""
    cube_cq: int = 14
    """NVENC constant-quality target of the faces: they feed SfM and splat training, so near visually lossless."""
    cube_gop: int = 10
    """Face keyframe interval in sampled framesets."""
    cube_device: Literal["cuda", "cpu"] = "cuda"
    """Where the faces are decoded and joined: ``cuda`` (NVDEC, CUDA remap, NVENC from device memory) or ``cpu``."""


class Insta360Dataset(DataforgeDataset[Insta360Config, ClipSource]):
    """Converts ``.insv`` clips into a base layer (lens videos, IMU, calibration) and, on request, a panorama layer."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, PANORAMA_LAYER, CUBE_LAYER)

    def download(self) -> None:
        """Local device captures: check that the clip directory holds a clip."""
        if not any(self.config.root.glob("*.insv")):
            raise FileNotFoundError(f"no .insv clip in {self.config.root}: copy the camera's clips there")

    def discover(self) -> list[tuple[SequenceIdentity, ClipSource]]:
        """Every ``.insv`` clip directly in the root, named by its file stem."""
        return [(SequenceIdentity(self.config.name, (path.stem,)), ClipSource(path)) for path in sorted(self.config.root.glob("*.insv"))]

    def convert(self, identity: SequenceIdentity, source: ClipSource, *, force: bool) -> Path:
        """Write the pending layers of one clip: the base, and the panorama under ``--panorama``."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        if not self.config.panorama:
            pending = [layer for layer in pending if layer != PANORAMA_LAYER]
        if not self.config.cube:
            pending = [layer for layer in pending if layer != CUBE_LAYER]
        if pending:
            capture: Capture = read_capture(source.path, load_profiles(), self.config.frame_limit)
            self.write_layers(
                identity,
                targets,
                pending,
                {
                    paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, source, capture),
                    PANORAMA_LAYER: lambda recording: self.write_panorama(recording, capture, targets[paths.BASE_LAYER]),
                    CUBE_LAYER: lambda recording: self.write_cube(recording, capture, targets[paths.BASE_LAYER]),
                },
            )
        return targets[paths.BASE_LAYER]

    def write_base(self, recording: rr.RecordingStream, identity: SequenceIdentity, source: ClipSource, capture: Capture) -> None:
        """The two lens cameras with their remuxed videos and exposure, the IMU with its noise model, the properties."""
        clip: insta360_io.InsvClip = capture.clip
        width, height = clip.resolution
        frames: int = int(capture.frame_ns.size)
        video_ns: Int64[ndarray, " n"] = capture.frame_ns - capture.start_ns
        log_rig_node(recording, RIG, reference="imu_00", num_cameras=2, name=clip.camera_type)
        imu_ns: Int64[ndarray, " m"] = capture.imu_ns - capture.start_ns
        log_imu(recording, RIG, 0, gyro=ImuChannel(imu_ns, capture.gyro), accel=ImuChannel(imu_ns, capture.accel), name="imu")
        noise: ImuNoise = capture.profile.imu_noise
        imu_node: str = schema.imu_path(RIG, 0)
        recording.log(
            imu_node,
            ImuCalibration(
                gyro_noise_density=noise.gyro_noise_density,
                accel_noise_density=noise.accel_noise_density,
                gyro_bias_random_walk=noise.gyro_bias_random_walk,
                accel_bias_random_walk=noise.accel_bias_random_walk,
                rate_hz=noise.rate_hz,
                source=noise.source,
            ),
            static=True,
        )
        recording.log(imu_node, rr.AnyValues(applied_time_shift_ns=0, time_shift_source="none: frames carry the profile's time model"), static=True)
        lever_arm: Float64[ndarray, "3"] = capture.profile.lever_arm_m
        for cam, lens in enumerate(capture.lenses):
            camera: insta360_io.MeiCamera = lens.camera
            track: int = clip.lens_track[cam]
            name: str = f"lens{cam}"
            node: str = schema.cam_path(RIG, cam)
            rr.log(
                node,
                rr.AnyValues(name=name, kind="rgb", camera_model=camera.model, rolling_shutter_readout_s=capture.readout_s),
                static=True,
                recording=recording,
            )
            imu_T_lens: Float64[ndarray, "4 4"] = imu_from_lens(lens, capture.imu_from_pano, lever_arm)
            lens_R_imu: Float64[ndarray, "3 3"] = imu_T_lens[:3, :3].T
            rr.log(
                node,
                rr.Transform3D(translation=-lens_R_imu @ imu_T_lens[:3, 3], mat3x3=lens_R_imu, relation=rr.TransformRelation.ChildFromParent),
                static=True,
                recording=recording,
            )
            pinhole: rr.Pinhole = rr.Pinhole(
                image_from_camera=[[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
                width=width,
                height=height,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.05,
            )
            distortion: CameraDistortion = CameraDistortion(camera.model, np.asarray(camera.coefficients, dtype=np.float64))
            rr.log(schema.pinhole_path(RIG, cam), PinholeWithDistortion(pinhole, distortion), static=True, recording=recording)
            log_camera_source(recording, RIG, cam, name=name, source_width=width, source_height=height, video_codec="hevc", stream_id=f"track{track}")
            exposure: str = f"{node}/exposure"
            rr.send_columns(exposure, indexes=[time_column(video_ns)], columns=rr.Scalars.columns(scalars=capture.exposure_s[cam]), recording=recording)
            rr.log(exposure, rr.SeriesLines(names=["exposure_s"]), static=True, recording=recording)
            video: str = schema.video_path(RIG, cam)
            rr.log(video, rr.VideoStream(codec=rr.VideoCodec.H265), static=True, recording=recording)
            for first in range(0, frames, self.config.batch_frames):
                last: int = min(first + self.config.batch_frames, frames)
                samples, keyframes = clip.read_samples(track, first, last)
                rr.send_columns(
                    video,
                    indexes=[time_column(video_ns[first:last])],
                    columns=rr.VideoStream.columns(sample=samples, is_keyframe=keyframes.tolist()),
                    recording=recording,
                )
                # Bound memory to one batch: without a flush the sink holds most of the clip.
                recording.flush()
        self.timer.capture_s = float(video_ns[-1] - video_ns[0]) / 1e9
        writing.send_capture_properties(
            recording,
            identity,
            num_cameras=2,
            num_frames=frames,
            start_time_ns=capture.start_ns,
            duration_s=self.timer.capture_s,
            clock_source="Insta360 trailer clock; frames at the middle row's mid-exposure time",
        )
        recording.send_property(
            "insta360",
            rr.AnyValues(
                source_file=source.path.name,
                camera_type=clip.camera_type,
                firmware=clip.firmware,
                profile=capture.profile.name,
                profile_version=capture.profile.version,
                pano_from_imu=capture.imu_from_pano.T.ravel().tolist(),
            ),
        )

    def write_panorama(self, recording: rr.RecordingStream, capture: Capture, base: Path) -> None:
        """Stitch every frameset's two lens images into the panorama (frame ``E``, module docstring) and log it as AV1 on the frame times.

        The stitch is built from the two cameras as the base recording states them (``stitch_from_recording``), so it
        is the stitch of anything else that reads that recording.
        """
        clip: insta360_io.InsvClip = capture.clip
        frames: int = int(capture.frame_ns.size)
        stitch: insta360_io.PanoramaStitch = stitch_from_recording(base)
        lens0, lens1 = (
            gray_frames(clip, clip.lens_track[cam], frames, batch_frames=self.config.batch_frames, threads=self.config.decode_threads) for cam in range(2)
        )
        panoramas: Iterator[bytes] = (stitch.panorama(a, b).tobytes() for a, b in zip(lens0, lens1, strict=True))
        width, height = PANORAMA_SIZE
        with work_dir("insta360-pano-") as work:
            video: Path = work / "pano.mp4"
            with self.timer.stage("encode"):
                # fps is the container's nominal rate only; log_video_stream puts every sample on its frame time.
                encode_frames_to_mp4(
                    panoramas, video, source=FrameSource("gray8", width=width, height=height), fps=30, gop=self.config.pano_gop, cq=self.config.pano_cq
                )
            rr.log(PANO_ENTITY, rr.AnyValues(video_codec="av1", cq=self.config.pano_cq, gop=self.config.pano_gop), static=True, recording=recording)
            samples: int = log_video_stream(recording, video, PANO_ENTITY, times_ns=capture.frame_ns - capture.start_ns)
        if samples != frames:
            raise ValueError(f"the panorama holds {samples} frames for {frames} framesets")

    def write_cube(self, recording: rr.RecordingStream, capture: Capture, base: Path) -> None:
        """Join every sampled frameset's two lens images into the six cube faces and log each face as AV1 on the frameset's stamp.

        On ``cube_device``: both lenses decoded batch by batch, each face one remap of the stacked pair, six NVENC sessions fed
        from the device (``cuda``: no host copy of a picture).
        """
        clip: insta360_io.InsvClip = capture.clip
        lenses: LensCameras = cameras_from_recording(base)
        size: int = self.config.cube_face_size or cube.default_face_size(lenses.size[0])
        indices: Int64[ndarray, " k"] = cube_indices(int(capture.frame_ns.size), self.config.cube_stride, self.config.cube_max_framesets)
        times_ns: Int64[ndarray, " k"] = capture.frame_ns[indices] - capture.start_ns
        faces: list[cube.CubeFace] = cube.cube_faces(size)
        tables: list[cube.FaceTable] = [cube.face_table(face, lenses.cameras, lenses.lens1_from_lens0) for face in faces]
        recording.send_property(
            "cube",
            rr.AnyValues(face_size=size, stride=self.config.cube_stride, framesets=int(indices.size), first_cam=FIRST_FACE_CAM, cq=self.config.cube_cq, gop=self.config.cube_gop),
        )
        for cam, face in zip(FACE_CAMS, faces, strict=True):
            node: str = schema.cam_path(RIG, cam)
            rr.log(
                node,
                rr.AnyValues(name=face.name, kind="rgb", camera_model="pinhole", video_codec="av1", cq=self.config.cube_cq, gop=self.config.cube_gop),
                static=True,
                recording=recording,
            )
            face_T_rig: Float64[ndarray, "4 4"] = cube.face_from_rig(face, lenses.lens_from_rig)
            rr.log(
                node,
                rr.Transform3D(translation=face_T_rig[:3, 3], mat3x3=face_T_rig[:3, :3], relation=rr.TransformRelation.ChildFromParent),
                static=True,
                recording=recording,
            )
            rr.log(
                schema.pinhole_path(RIG, cam),
                rr.Pinhole(image_from_camera=face.intrinsics, width=size, height=size, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05),
                static=True,
                recording=recording,
            )
        device: torch.device = torch.device(self.config.cube_device)
        join = cube.CubeJoin(tables, lenses.size, device)
        with work_dir("insta360-cube-") as work:
            videos: list[Path] = [work / f"{face.name}.mp4" for face in faces]
            with self.timer.stage("cube:faces"):
                fed: list[int] = encode_face_videos(
                    (join.faces(lens0, lens1) for lens0, lens1 in lens_pictures(clip, indices, device, threads=self.config.decode_threads)),
                    videos,
                    size=size,
                    gop=self.config.cube_gop,
                    cq=self.config.cube_cq,
                    device=device,
                )
            if fed != [indices.size] * len(faces):
                raise ValueError(f"the face encoders took {fed} frames for {indices.size} framesets")
            for cam, video in zip(FACE_CAMS, videos, strict=True):
                samples: int = log_video_stream(recording, video, schema.video_path(RIG, cam), times_ns=times_ns)
                if samples != indices.size:
                    raise ValueError(f"cam_{cam:02d} holds {samples} frames for {indices.size} framesets")

    def default_blueprint(self) -> rrb.Blueprint:
        """Both lenses and the panorama beside the rig in 3D, over the IMU plots."""
        return blueprints.rig_blueprint(
            [
                blueprints.camera_view("lens0", RIG, 0, contents=[schema.video_path(RIG, 0)]),
                blueprints.camera_view("lens1", RIG, 1, contents=[schema.video_path(RIG, 1)]),
                rrb.Spatial2DView(name="panorama", origin=PANO_ENTITY),
            ],
            rig=RIG,
            run_source=RUN_SOURCE,
            # Lens 0 looks along z_P = -y_imu; image up is -y_P = -x_imu.
            eye_controls=blueprints.follow_eye_controls((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
            plots=blueprints.imu_plots(RIG, 0),
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Preview card: the rig beside lens 0."""
        return blueprints.table_blueprint(
            2,
            rig=RIG,
            run_source=RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
            front_pane=blueprints.camera_view("lens0", RIG, 0, contents=[schema.video_path(RIG, 0)]),
        )
