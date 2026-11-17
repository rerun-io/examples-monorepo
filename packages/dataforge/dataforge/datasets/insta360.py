"""Insta360 dual-fisheye ``.insv`` clips: one base recording per clip, and a derived panorama layer for viewing.

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
record 12 (VIN1) for lens 1, on the frame times.

Device profile. What the file does not carry (``R_P_imu``, the lever arm, the frame-time model, the readout, the IMU
noise) is a measured record in ``insta360_profiles.toml``, selected by the clip's camera type, lens size, frame rate
(from the exposure stamps) and readout; a clip outside every profile's measured mode is refused, naming the property
that differs. The IMU noise model is a placeholder, marked as such in its ``source``.

Panorama layer. ``panorama`` holds the 1920x960 viewing panorama at ``/world/rig_00/pano``, stitched on the CPU from
the two lens images of every frameset (lens 0's hemisphere on the left) and encoded AV1 on NVENC, on the frame times.
It is for viewing only: nothing reads it as input, and the lens brightness steps at the seams are not corrected.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import av
import insta360_io
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from serde import SerdeError, coerce, serde
from serde.toml import from_toml
from simplecv.imu_calibration import ImuCalibration
from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import ImuChannel, log_camera_source, log_imu, log_rig_node, log_video_stream, time_column
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, work_dir

RIG: int = 0
"""One rig per clip."""
PANORAMA_LAYER: str = "panorama"
"""The derived viewing panorama: an AV1 ``VideoStream`` at ``PANO_ENTITY`` on the frame times."""
PANO_ENTITY: str = f"{schema.rig_path(RIG)}/pano"
"""Where the panorama layer puts its video."""
PANORAMA_SIZE: tuple[int, int] = (1920, 960)
"""Width and height of ``PanoramaStitch.panorama``."""
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
    lens_width: int
    """Lens track width, pixels, of the measured mode."""
    lens_height: int
    """Lens track height, pixels."""
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
        if self.lens_width < 1 or self.lens_height < 1 or self.frame_rate_hz <= 0.0 or self.readout_s <= 0.0:
            raise ValueError(f"profile {self.name}: lens size, frame rate and readout must be positive")


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
    lens_width: int
    """Lens track width, pixels."""
    lens_height: int
    """Lens track height, pixels."""
    frame_rate_hz: float
    """From the median exposure-record-4 stamp interval."""
    readout_s: float | None
    """The clip's own ``rolling_shutter_ms``, in seconds; None when the clip does not state one."""


def mode_differences(profile: DeviceProfile, mode: CaptureMode) -> list[str]:
    """Each property in which ``mode`` lies outside what ``profile`` was measured on, as a readable reason."""
    differences: list[str] = []
    if mode.camera_type != profile.camera_type:
        differences.append(f"camera type {mode.camera_type!r} is not {profile.camera_type!r}")
    if (mode.lens_width, mode.lens_height) != (profile.lens_width, profile.lens_height):
        differences.append(f"lens size {mode.lens_width}x{mode.lens_height} is not {profile.lens_width}x{profile.lens_height}")
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


def nearest_permutation(rotation: Float64[ndarray, "3 3"]) -> Float64[ndarray, "3 3"]:
    """The signed axis permutation nearest to ``rotation``, row by row."""
    permutation: Float64[ndarray, "3 3"] = np.zeros((3, 3))
    for row in range(3):
        column: int = int(np.argmax(np.abs(rotation[row])))
        permutation[row, column] = np.sign(rotation[row, column])
    return permutation


def angle_to_nearest_permutation_deg(rotation: Float64[ndarray, "3 3"]) -> float:
    """Angle in degrees between ``rotation`` and the signed axis permutation nearest to it."""
    permutation: Float64[ndarray, "3 3"] = nearest_permutation(rotation)
    cosine: float = (float(np.trace(permutation.T @ rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


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
    decoder = av.CodecContext.create("hevc", "r")
    decoder.thread_count = threads
    decoder.thread_type = "NONE" if threads == 1 else "FRAME"

    def gray(frame: av.VideoFrame) -> UInt8[ndarray, "h w"]:
        converted: av.VideoFrame = frame.reformat(format="gray8")
        plane = converted.planes[0]
        padded: UInt8[ndarray, "h stride"] = np.frombuffer(plane, dtype=np.uint8).reshape(converted.height, plane.line_size)
        return padded[:, : converted.width].copy()

    for first in range(0, frames, batch_frames):
        samples, _ = clip.read_samples(track, first, min(first + batch_frames, frames))
        for sample in samples:
            yield from (gray(frame) for frame in decoder.decode(av.Packet(sample)))
    yield from (gray(frame) for frame in decoder.decode(None))


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


def read_capture(path: Path, profiles: list[DeviceProfile], frame_limit: int | None) -> Capture:
    """Open ``path``, select its profile and compute the frame times.

    Raises:
        ValueError: If the clip lacks two lenses or a sensor record, or no profile was measured in its mode.
    """
    clip: insta360_io.InsvClip = insta360_io.InsvClip(str(path))
    width, height = clip.resolution
    frame_t_us, exposure_vin0 = clip.frame_exposures(False)
    vin1_t_us, exposure_vin1 = clip.frame_exposures(True)
    if frame_t_us.size < 2:
        raise ValueError(f"{path}: {frame_t_us.size} exposure-record-4 entries; a clip needs two frames or more")
    if vin1_t_us.size != frame_t_us.size:
        raise ValueError(f"{path}: exposure record 12 (VIN1) holds {vin1_t_us.size} frame entries, record 4 {frame_t_us.size}")
    mode: CaptureMode = CaptureMode(
        camera_type=clip.camera_type,
        lens_width=width,
        lens_height=height,
        frame_rate_hz=1e6 / float(np.median(np.diff(frame_t_us))),
        readout_s=None if clip.rolling_shutter_ms is None else clip.rolling_shutter_ms * 1e-3,
    )
    profile: DeviceProfile = select_profile(profiles, mode)
    lenses: list[insta360_io.InsvLens] = clip.lenses_v6 or clip.lenses
    if len(lenses) != 2:
        raise ValueError(f"{path}: the factory lens string holds {len(lenses)} lenses; a dual fisheye has two")
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
    """Insta360 dual-fisheye ``.insv`` clips (local device captures) as a two-camera + IMU rig, with a panorama layer."""

    command: ClassVar[str] = "insta360"
    """Registry key and CLI subcommand."""

    _target: type = field(default_factory=lambda: Insta360Dataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "insta360")
    """Directory holding the ``.insv`` clips; each clip is one sequence."""
    batch_frames: int = 300
    """Samples read and sent per call, so a clip is never held in memory whole."""
    decode_threads: int = 4
    """PyAV threads per lens decoder of the panorama layer."""
    pano_cq: int = 34
    """NVENC constant-quality target of the panorama: a viewing layer, not an analysis input."""
    pano_gop: int = 30
    """Panorama keyframe interval in frames, about one second."""


class Insta360Dataset(DataforgeDataset[Insta360Config, ClipSource]):
    """Converts ``.insv`` clips into a base layer (lens videos, IMU, calibration) and a panorama layer."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, PANORAMA_LAYER)

    def download(self) -> None:
        """Local device captures: check that the clip directory holds a clip."""
        if not any(self.config.root.glob("*.insv")):
            raise FileNotFoundError(f"no .insv clip in {self.config.root}: copy the camera's clips there")

    def discover(self) -> list[tuple[SequenceIdentity, ClipSource]]:
        """Every ``.insv`` clip directly in the root, named by its file stem."""
        return [(SequenceIdentity(self.config.name, (path.stem,)), ClipSource(path)) for path in sorted(self.config.root.glob("*.insv"))]

    def convert(self, identity: SequenceIdentity, source: ClipSource, *, force: bool) -> Path:
        """Write the pending base and panorama layers of one clip."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        if pending:
            capture: Capture = read_capture(source.path, load_profiles(), self.config.frame_limit)
            self.write_layers(
                identity,
                targets,
                pending,
                {
                    paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, source, capture),
                    PANORAMA_LAYER: lambda recording: self.write_panorama(recording, capture),
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
            track: int = clip.lens_track[cam]
            name: str = f"lens{cam}"
            node: str = schema.cam_path(RIG, cam)
            rr.log(
                node,
                rr.AnyValues(name=name, kind="rgb", camera_model=lens.model, rolling_shutter_readout_s=capture.readout_s),
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
                image_from_camera=[[lens.fx, 0.0, lens.cx], [0.0, lens.fy, lens.cy], [0.0, 0.0, 1.0]],
                width=width,
                height=height,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.05,
            )
            distortion: CameraDistortion = CameraDistortion(lens.model, np.asarray(lens.coefficients, dtype=np.float64))
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

    def write_panorama(self, recording: rr.RecordingStream, capture: Capture) -> None:
        """Stitch every frameset's two lens images into the viewing panorama and log it as AV1 on the frame times."""
        clip: insta360_io.InsvClip = capture.clip
        frames: int = int(capture.frame_ns.size)
        # STAGE 2 SWAP: the stitch is built from the factory V6 string here; Phase 1's stitch takes the two cameras as
        # written on the camera nodes and the pose of camera 1 in camera 0.
        strings: dict[str, str] = dict(clip.factory_strings)
        if "v6" not in strings:
            raise ValueError("the panorama stitch is built from the V6 lens string, which this clip lacks")
        stitch: insta360_io.PanoramaStitch = insta360_io.PanoramaStitch(strings["v6"], clip.window_crop, clip.resolution)
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
