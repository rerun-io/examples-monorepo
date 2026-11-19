"""The compiled half of ``insta360_io`` (Rust: ``insta360_io_core`` over kornia-staging's ``kornia_staging_io::insta360``,
and ``kornia_staging_imgproc::panorama``).

``InsvClip`` reads a clip's container index and trailer as the file states them: metadata, the factory lens
calibration, IMU samples, per-frame exposure. ``InsvRig`` applies the clip's device profile (the built-in
``insta360_profiles.toml``): lens poses in the IMU frame, frame times on the IMU clock, and the lens tracks cut into
GOPs and read as packed Annex B samples; ``insta360_io.InsvReader`` turns it into Rerun chunks. ``PanoramaStitch`` turns the two lens images of a
frameset into the 1920x960 viewing panorama through the two hemisphere tiles, on the CPU, with the same tables and
integer remap as slam-rs's dual-fisheye lane, from two ``MeiCamera``s and the pose of camera 1 in camera 0. Every
refusal raises ``ValueError``.
"""

from typing import Literal

from jaxtyping import Bool, Float64, Int32, Int64, UInt8
from numpy import ndarray

class MeiCamera:
    """A unified (Mei) sphere camera with the Insta360 factory distortion, as a calibration states it."""

    def __init__(self, model: str, fx: float, fy: float, cx: float, cy: float, coefficients: list[float]) -> None:
        """``model`` ``mei_v3`` (``xi, half_field_rad, k1, k2, k3, p1, p2``) or ``mei_v6`` (``xi, half_field_rad, d0 .. d12``).

        Raises ``ValueError`` on an unknown model, a wrong coefficient count or invalid values.
        """
    @property
    def model(self) -> str: ...
    @property
    def fx(self) -> float:
        """Focal along x, pixels (Mei form)."""
    @property
    def fy(self) -> float: ...
    @property
    def cx(self) -> float:
        """Principal point x, pixels (pixel centres at integers)."""
    @property
    def cy(self) -> float: ...
    @property
    def coefficients(self) -> list[float]:
        """The coefficients in the order of ``model``."""
    def project(self, points: Float64[ndarray, "n 3"]) -> tuple[Float64[ndarray, "n 2"], Bool[ndarray, " n"]]:
        """Project camera-frame points (any norm) to pixels (centres at integers) and a validity mask.

        Invalid where the forward model rejects the point (more than ``half_field_rad`` off the axis, zero or
        non-finite); the pixel is NaN there. Raises ``ValueError`` unless ``points`` is ``float64[n, 3]``.
        """

class InsvLens:
    """One lens of an Insta360 factory calibration at the track resolution, window crop applied."""

    @property
    def camera(self) -> MeiCamera: ...
    @property
    def pano_from_lens_rotation(self) -> list[list[float]]:
        """``R_P_lens``, row-major: lens vectors into the vendor rig frame P."""
    @property
    def pano_centre_m(self) -> list[float]:
        """Lens centre in P, metres, origin at the midpoint of the lens centres."""

class InsvClip:
    """An Insta360 ``.insv`` clip: trailer metadata, factory lenses, IMU, frame times and the two HEVC lens tracks."""

    def __init__(self, path: str) -> None:
        """Read the container index and the trailer; raises ``ValueError`` on a malformed file."""
    @property
    def camera_type(self) -> str: ...
    @property
    def firmware(self) -> str: ...
    @property
    def first_frame_timestamp_us(self) -> int: ...
    @property
    def rolling_shutter_ms(self) -> float | None: ...
    @property
    def gyro_timestamp_ms(self) -> float | None: ...
    @property
    def factory_strings(self) -> list[tuple[str, str]]:
        """Raw lens strings present in the metadata, as ``(version name, string)``: ``v1``, ``v2``, ``v3``, ``v6``."""
    @property
    def lens_track(self) -> tuple[int, int]:
        """Video track index of lens 0 and lens 1."""
    @property
    def lenses(self) -> list[InsvLens]: ...
    @property
    def lenses_v6(self) -> list[InsvLens]:
        """V6 factory calibration per lens (lens order); empty without a V6 string."""
    @property
    def window_crop(self) -> tuple[int, int, int, int, int, int] | None:
        """``(src_w, src_h, dst_w, dst_h, offset_x, offset_y)`` of the lens strings."""
    @property
    def resolution(self) -> tuple[int, int]: ...
    @property
    def num_frames(self) -> int: ...
    def imu(self) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 3"], Float64[ndarray, "n 3"]]:
        """Times on the trailer clock (us), accel (m/s^2) and gyro (rad/s) in the IMU's axes."""
    def frame_exposures(self, vin1: bool) -> tuple[Int64[ndarray, " n_frames"], Float64[ndarray, " n_frames"]]:
        """Per-frame exposure entries of record 4 (``vin1=False``) or 12: times (us) and exposure times (s)."""

class ImuNoise:
    """The IMU noise model of a device profile."""

    @property
    def gyro_noise_density(self) -> float:
        """rad/s/sqrt(Hz)."""
    @property
    def accel_noise_density(self) -> float:
        """m/s^2/sqrt(Hz)."""
    @property
    def gyro_bias_random_walk(self) -> float:
        """rad/s^2/sqrt(Hz)."""
    @property
    def accel_bias_random_walk(self) -> float:
        """m/s^3/sqrt(Hz)."""
    @property
    def rate_hz(self) -> float: ...
    @property
    def source(self) -> str: ...

class InsvRig:
    """A clip with its device profile applied when one fits: two lens cameras, times on one clock, the GOP demux.

    Every time is on the trailer clock in ns. With a profile, frame ``i`` is the exposure-record-4 stamp plus the
    profile's ``slope * shutter_i + constant`` (the mid-exposure time of the middle row on the IMU clock); without one,
    the raw stamp.
    """

    def __init__(self, path: str, frame_limit: int | None = None) -> None:
        """Read the clip and select its profile; ``frame_limit`` keeps the first frames and the IMU up to 1 s past them.

        Raises ``ValueError`` on a malformed clip, fewer than two frames, a ``frame_limit`` of 0, or a lens track whose
        pictures are presented out of decode order (B-frames).
        """
    @property
    def camera_type(self) -> str: ...
    @property
    def firmware(self) -> str: ...
    @property
    def profile_name(self) -> str | None:
        """The measured profile the clip's body and mode selected; ``None`` when none fits (``profile_note`` says why)."""
    @property
    def profile_version(self) -> int | None: ...
    @property
    def profile_note(self) -> str:
        """Empty with a profile; without one, why there is none."""
    @property
    def imu_noise(self) -> ImuNoise | None: ...
    @property
    def readout_s(self) -> float | None:
        """Rolling-shutter readout of each lens image (first row to last): the clip's own, else the profile's."""
    @property
    def pano_from_imu(self) -> list[list[float]] | None:
        """``R_P_imu``, row-major, orthonormalised; ``None`` without a profile."""
    @property
    def codec(self) -> Literal["h264", "hevc"]:
        """Codec of both lens tracks, as FFmpeg names it: ``h264`` or ``hevc``."""
    @property
    def lenses(self) -> list[InsvLens]:
        """Lens 0 and lens 1: V6 when the clip has a V6 string, else V3."""
    def rig_from_lens(self, lens: int) -> list[list[float]]:
        """``T_rig_lens``, row-major 4x4. With a profile rig = IMU frame, ``(R_imu_P R_P_lens, R_imu_P (c_P + lever_arm))``;
        without one rig = the vendor frame P, ``(R_P_lens, c_P)``."""
    @property
    def lens_track(self) -> tuple[int, int]: ...
    @property
    def resolution(self) -> tuple[int, int]: ...
    @property
    def num_frames(self) -> int: ...
    @property
    def start_ns(self) -> int:
        """The earliest frame or IMU time, ns."""
    def frame_ns(self) -> Int64[ndarray, " n_frames"]: ...
    def exposure_s(self) -> Float64[ndarray, "2 n_frames"]:
        """Exposure time per frame of lens 0 (record 4) and lens 1 (record 12; record 4 without it)."""
    def imu(self) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 3"], Float64[ndarray, "n 3"]]:
        """Times (ns), accel (m/s^2) and gyro (rad/s) in the IMU's axes."""
    def gops(self, lens: int) -> Int64[ndarray, "n_gops 2"]:
        """``[start, stop)`` frame ranges: a keyframe and the samples up to the next one (the first opens on frame 0).

        Raises ``IndexError`` for a lens other than 0 and 1.
        """
    def read_packed(self, lens: int, start: int, stop: int) -> tuple[UInt8[ndarray, " n_bytes"], Int32[ndarray, " n_plus_1"], Bool[ndarray, " n"]]:
        """Samples ``start..stop`` as Annex B access units back to back, their int32 offsets and IRAP flags (no GIL).

        Raises ``IndexError`` for a lens other than 0 and 1 or a range outside ``0..num_frames``.
        """

class PanoramaStitch:
    """Two lens images in, the 1920x960 viewing panorama out, through the two hemisphere tiles.

    The panorama frame is camera 0's axes at the midpoint of the two camera centres: camera 0's hemisphere on the left
    half (looking along its +Z), camera 1's on the right (-Z).
    """

    def __init__(self, camera0: MeiCamera, camera1: MeiCamera, cam0_from_cam1: list[list[float]], lens_size: tuple[int, int]) -> None:
        """``cam0_from_cam1``: the pose of camera 1 in camera 0 (only its rotation enters the tables); ``lens_size``: ``(width, height)``.

        Raises ``ValueError`` on a matrix that is not a rigid transform or a lens below 4x4.
        """
    def panorama(self, lens0: UInt8[ndarray, "h w"], lens1: UInt8[ndarray, "h w"]) -> UInt8[ndarray, "960 1920"]:
        """The panorama of lens 0's and lens 1's C-contiguous images; ``ValueError`` on a wrong size."""
