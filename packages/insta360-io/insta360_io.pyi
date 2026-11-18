"""Insta360 ``.insv`` clips (Rust: kornia-staging's ``kornia_staging_io::insta360`` and ``kornia_staging_imgproc::panorama``).

``InsvClip`` reads a clip's container index and trailer: metadata, the factory lens calibration, IMU samples, per-frame
exposure and the two HEVC lens tracks as Annex B access units. ``PanoramaStitch`` turns the two lens images of a
frameset into the 1920x960 viewing panorama through the two hemisphere tiles, on the CPU, with the same tables and
integer remap as slam-rs's dual-fisheye lane, from two ``MeiCamera``s and the pose of camera 1 in camera 0. Every
refusal raises ``ValueError``.
"""

from jaxtyping import Bool, Float64, Int64, UInt8
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
    def read_samples(self, track: int, start: int, stop: int) -> tuple[list[bytes], Bool[ndarray, " n"]]:
        """Samples ``start..stop`` of one track as Annex B access units with parameter sets on IRAPs, and IRAP flags."""

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
