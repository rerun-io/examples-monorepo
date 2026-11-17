"""Types for the compiled ``slam_rs._core`` extension (built by the ``slam-rs-build`` task).

The signatures here are the only static check on the FFI boundary, so they stay
exact: no ``Any``, and every array carries its dtype.
"""

from collections.abc import Sequence
from typing import ClassVar, Literal

from jaxtyping import Bool, Float32, Float64, Int32, Int64, UInt8
from numpy import ndarray
from simplecv.imu_calibration import ImuCalibration

from slam_rs import BackendRequest, Lane
from slam_rs.catalog_calibration import CameraStatics
from slam_rs.catalog_feed import RigProfile
from slam_rs.catalog_timing import ImuStream
from slam_rs.rig import CameraCalib, ImuCalib

__version__: str

gpu_backend: Literal["wgpu"] | None
"""Which GPU runtime this build's frontend carries, or None for a CPU-only build.

Every pixi build task builds the core with the ``gpu-wgpu`` cargo feature, which
carries ``"wgpu"``. ``None`` is a build without it (the cargo default of the core
crate, kept for wasm, iOS and embedded consumers): there ``backend="auto"`` runs
the CPU and ``backend="gpu"`` is refused.
"""

def device_probes_run() -> int:
    """How many device capability probes this process has run.

    Validation of the default device runs once per process, so this is 0
    before the first ``auto`` or ``gpu`` request and constant after it. The
    unit test tier asserts it stays at 0.
    """

def select_backend(request: BackendRequest) -> Lane:
    """Resolve a backend request to the lane :class:`Vio` would run (``slam_rs::select_backend``).

    ``auto`` returns ``"gpu"`` when this build carries the GPU frontend and the
    host has an adapter that passes the storage and subgroup probes; otherwise
    it returns ``"cpu"`` and writes one warning, naming the reason, to stderr
    (once per process). ``"gpu"`` raises ``ValueError`` with that reason instead.
    The probe runs once per process.
    """

class VioStatus:
    """How far the estimator has got.

    ``NeedMoreImu`` refuses a frame for retry. ``Tracking`` carries an estimate.
    ``Buffered`` accepts the first frame in lag mode; submit the next frame or
    call ``flush()``, without retrying it.

    A PyO3 enum, not a ``enum.Enum``: it carries no ``name`` or ``value``, it is
    unhashable, and ``VioStatus(1)`` raises ``TypeError``. It does convert to
    ``int`` and compares equal both to its own variants and to their ordinals.
    """

    NeedMoreImu: ClassVar[VioStatus]
    Tracking: ClassVar[VioStatus]
    Buffered: ClassVar[VioStatus]
    NoVisualFeatures: ClassVar[VioStatus]
    __hash__: ClassVar[None]

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class BootstrapAttempt:
    """One solve attempt of the inertial bootstrap (``config.vio_bootstrap``)."""

    @property
    def first_t_ns(self) -> int:
        """First frameset of the window, ns."""

    @property
    def last_t_ns(self) -> int:
        """Last frameset of the window, ns: when the attempt ran."""

    @property
    def frames(self) -> int:
        """Framesets in the window."""

    @property
    def tracks(self) -> int:
        """Tracks handed to the solver."""

    @property
    def tracks_used(self) -> int:
        """Inlier tracks of the solve."""

    @property
    def verdict(self) -> str:
        """The verdict's name: ``Accepted`` or the rejection reason, e.g. ``LowExcitation``."""

    @property
    def accepted(self) -> bool:
        """Whether the solver accepted the window."""

    @property
    def excitation_mps2(self) -> float:
        """Excitation, m/s^2."""

    @property
    def residual_rms_rad(self) -> float | None:
        """RMS bearing error of the solve, rad; None when the attempt reports none."""

    @property
    def condition(self) -> float | None:
        """Condition number of the reduced linear system; None when the attempt reports none."""

    @property
    def gravity_norm_unconstrained(self) -> float | None:
        """Unconstrained ``|g|``, m/s^2; None when the attempt reports none."""

    @property
    def scale_sigma(self) -> float | None:
        """Relative 1-sigma of the speed; None when the attempt reports none."""

    def __repr__(self) -> str: ...

class BootstrapReport:
    """Every attempt of the inertial bootstrap and where it started the estimator."""

    @property
    def attempts(self) -> list[BootstrapAttempt]:
        """The attempts in order."""

    @property
    def start_t_ns(self) -> int | None:
        """The accepted window's first frameset, ns (the estimator's first state); None before a window is accepted."""

    @property
    def accepted_at_t_ns(self) -> int | None:
        """The frameset that completed the accepted window, ns (the first pose published); None before a window is accepted."""

    @property
    def replayed(self) -> int:
        """Framesets replayed into the fresh estimator."""

    def __repr__(self) -> str: ...

class VioResult:
    """What one :meth:`Vio.track` call produced."""

    @property
    def status(self) -> VioStatus: ...
    @property
    def t_ns(self) -> int: ...
    @property
    def world_from_rig(self) -> Float64[ndarray, " 7"] | None:
        """``[tx, ty, tz, qx, qy, qz, qw]``, metres and a unit quaternion (xyzw).

        Only ``Tracking`` supplies a pose; every other status returns ``None``.
        """

    @property
    def velocity(self) -> Float64[ndarray, " 3"] | None:
        """Rig velocity in the world frame, m/s."""

    @property
    def gyro_bias(self) -> Float64[ndarray, " 3"] | None:
        """Gyroscope bias estimate, rad/s."""

    @property
    def accel_bias(self) -> Float64[ndarray, " 3"] | None:
        """Accelerometer bias estimate, m/s^2."""

    def __repr__(self) -> str: ...

class VioSnapshot:
    """The estimator's window, its landmarks and the last measured frame's statistics.

    The window is the 15-dof states followed by the pose-only blocks, each oldest
    first; :attr:`window_keyframe` and :attr:`window_long_term` say what each
    frame is, :attr:`kf_ids` is the keyframes as an id list, and
    :attr:`marginalized` is what the last marginalization removed. Everything is
    a copy, so a snapshot stays valid across the next :meth:`Vio.track`.
    """

    @property
    def t_ns(self) -> int:
        """Frameset timestamp of the newest state in the window."""

    @property
    def window_t_ns(self) -> Int64[ndarray, " n_frames"]: ...
    @property
    def window_poses(self) -> Float64[ndarray, "n_frames 7"]:
        """``[tx, ty, tz, qx, qy, qz, qw]`` per window frame, metres and a unit quaternion (xyzw)."""

    @property
    def window_keyframe(self) -> Bool[ndarray, " n_frames"]:
        """Whether each window frame is a keyframe, as the estimator itself answers it."""

    @property
    def window_long_term(self) -> Bool[ndarray, " n_frames"]:
        """Whether each window frame is a long-term keyframe."""

    @property
    def kf_ids(self) -> Int64[ndarray, " n_keyframes"]:
        """The keyframes' timestamps, oldest first."""

    @property
    def marginalized(self) -> Int64[ndarray, " n_marginalized"]:
        """Frames the last marginalization removed from the window."""

    @property
    def landmark_ids(self) -> Int64[ndarray, " n_landmarks"]:
        """Landmark ids, which are the ids of the keypoints that spawned them."""

    @property
    def landmark_hosts(self) -> Int64[ndarray, " n_landmarks"]:
        """Timestamp of the keyframe hosting each landmark."""

    @property
    def landmark_positions(self) -> Float64[ndarray, "n_landmarks 3"]:
        """Landmark positions in the world frame, metres."""

    @property
    def lm_iterations(self) -> int:
        """Levenberg-Marquardt steps the last frame took; the rejected ones are the rest."""

    @property
    def lm_lambda(self) -> float:
        """Damping the last step solved with; ``0.0`` when no step ran."""

    @property
    def lm_error_before(self) -> float:
        """Total cost before the first step; ``0.0`` when no step ran."""

    @property
    def lm_error_after(self) -> float:
        """Total cost after the last step; ``0.0`` when no step ran."""

    @property
    def num_observations(self) -> int:
        """Landmark observations the window holds."""

    @property
    def frame_update(self) -> str:
        """What the non-keyframe frame update did: ``not_attempted``, ``taken``, or ``declined_<precondition>``."""

    @property
    def gpu_one_wait(self) -> bool:
        """The GPU lane collected this frameset's temporal and stereo results in one device wait; always False on the CPU lane."""

    @property
    def gpu_lookahead(self) -> bool:
        """The GPU lane reused this frameset's lookahead image upload; always False on the CPU lane."""

    @property
    def timings_ms(self) -> dict[str, float]:
        """Wall time each stage took on the last frame, in milliseconds.

        Estimator keys: ``predict``, ``keyframe``, ``optimize``, ``linearize``,
        ``solver``, ``back_substitution``, ``error``, ``marginalize``, ``measure``.
        Frontend keys: ``frontend_pyramid``, ``frontend_detect``,
        ``frontend_track`` (temporal only; formerly included stereo),
        ``frontend_stereo`` (cross-camera matching plus epipolar filter),
        and ``frontend_imu`` (the frontend's motion preintegration).

        ``keyframe`` includes decision and landmark initialization on keyframes,
        and is zero otherwise. ``optimize`` contains ``linearize``, ``solver``,
        ``back_substitution`` and ``error``. ``measure`` contains ``keyframe``,
        ``optimize``, ``marginalize``, state prediction and untimed glue.
        ``predict`` includes that state prediction plus IMU integration before
        ``measure``; its integration portion is outside ``measure``. Timers
        overlap and must not be summed as independent frame costs.
        With frontend lag, frontend timers describe the newer input images;
        estimator timers describe the returned pose.
        """

    def __repr__(self) -> str: ...

class Vio:
    """The VIO pipeline, driven one frameset at a time.

    The default runs frontend and estimator in sequence. ``port.frontend_lag``
    overlaps this frame's frontend with the previous frame's estimator. Both
    schedules repeat bit-identically for the same input and configuration.

    The refusals are the ones :class:`OpticalFlow` makes — a value the core
    refuses is a ``ValueError``, an object of the wrong type a ``TypeError`` and
    an integer outside the parameter's own type an ``OverflowError`` — never a
    Rust panic.
    """

    def __init__(
        self,
        calibration: Calibration,
        config: VioConfig,
        *,
        threads: int = 1,
        max_keypoints: int | None = None,
        backend: BackendRequest = "auto",
        dual_fisheye: tuple[str, tuple[int, int, int, int, int, int] | None, float] | None = None,
    ) -> None:
        """Build the pipeline for one rig; JSON files arrive through ``from_json``.

        ``backend`` chooses the frontend through :func:`select_backend`. The
        default ``"auto"`` runs the pyramid, patch build and KLT tracker through
        CubeCL on this host's GPU when it can, and otherwise the CPU port with
        one warning that names the reason. ``"gpu"`` raises ``ValueError``
        instead of falling back: in a core built without the ``gpu-wgpu`` cargo
        feature, and on a host with no adapter, a driver that will not
        initialise, or a device that fails the storage or subgroup probe. None
        of them is a ``PanicException``: a failure the probe catches is a
        ``ValueError`` too, with the runtime's own panic message left on
        stderr. ``"cpu"`` always runs the CPU port.

        ``threads`` is **inert on the GPU lane**: only the CPU patch tracker
        reads it and the GPU tracker holds no work pool. It is accepted rather
        than refused on the GPU lane so one call site can select either lane.

        Raises ``ValueError`` on everything :class:`OpticalFlow` refuses, and on
        a config asking for a path this port does not have:
        ``vio_linearization_type`` other than ``ABS_QR``, ``vio_sqrt_marg``
        false, or ``vio_enforce_realtime``, which Offline mode cannot honour.
        """

    @property
    def camera_count(self) -> int: ...
    def bearings(self, camera: int, pixels: Float32[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]:
        """Unit bearings of frontend camera ``camera``'s pixels in frontend camera 0's frame (P on a panorama rig); NaN where a pixel does not unproject."""
    @property
    def panorama_tile_origins(self) -> tuple[int, int] | None:
        """Panorama column of each hemisphere tile's column 0 when the rig is one full panorama, else None.

        Tile ``k``'s pixel ``(x, y)`` is the panorama's ``((x + origins[k]) mod W, y)``.
        """
    @property
    def panorama_ingest_ns(self) -> int | None:
        """Wall time of the last dual-fisheye host ingest in ns: the CPU remap of both lenses, or on the GPU lane the
        lens byte copy (the stitch runs on the device); None for any other rig."""
    def panorama_image(self) -> UInt8[ndarray, "h w"] | None:
        """The stitched 1920x960 panorama of the last dual-fisheye frameset (``dual_fisheye`` given), else None.

        On the GPU lane it is remapped on the host from the lens bytes, only when asked: no device read-back.
        """
    @property
    def gpu(self) -> bool:
        """Whether the frontend runs on the GPU."""
    @property
    def backend(self) -> Lane:
        """The lane this pipeline runs: ``"cpu"`` or ``"gpu"``."""
    def push_imu(self, t_ns: int, gyro: Sequence[float], accel: Sequence[float]) -> None:
        """Add one uncalibrated IMU sample.

        Raises ``ValueError`` unless ``t_ns`` strictly follows the last sample
        and every component is finite.
        """

    def push_imu_batch(
        self,
        t_ns: Int64[ndarray, " n_samples"],
        gyro: Float64[ndarray, "n_samples 3"],
        accel: Float64[ndarray, "n_samples 3"],
    ) -> None:
        """Add a batch of samples, all or nothing.

        Raises what :meth:`push_imu` raises, for any sample of the batch, and
        keeps none of it when it does: the estimator is left where it was, so the
        batch can be corrected and pushed again.
        """

    def track(self, t_ns: int, images: Sequence[UInt8[ndarray, "h w"]]) -> VioResult:
        """Process one frameset of ``camera_count`` C-contiguous ``(h, w)`` uint8 images.

        With ``port.frontend_lag`` enabled, the first call returns ``Buffered``.
        Later calls return the previous frame's estimate with its own ``t_ns``.
        Call ``flush()`` to receive the final pose at the end of the stream.

        Raises ``ValueError`` on a bad dtype, rank or layout, on the wrong number
        of images, unless every image is the size the calibration gives its
        camera, and unless ``t_ns`` is strictly after the last accepted frameset.
        On a GPU lane a device that dies mid-run raises ``ValueError`` here as
        well, never a ``PanicException``.
        """

    def flush(self) -> VioResult | None:
        """Drain the final lagged pose exactly once and finish any deferred solve."""

    def snapshot(self) -> VioSnapshot | None:
        """The last measured frame, or None before the first one.

        With frontend lag its timestamp precedes ``flow_frame()``; frontend
        timers still describe the most recent input images.
        """

    def flow_frame(self) -> FlowFrame | None:
        """The keypoints the frontend tracked on the last accepted frameset, or None before the first."""

    def bootstrap_report(self) -> BootstrapReport | None:
        """The inertial bootstrap's attempts so far (``config.vio_bootstrap``), or None on the stereo start."""

    @property
    def frontend_timings_ms(self) -> dict[str, float]:
        """What the frontend lane cost on the last frameset that ran it, in ms: ``imu``, ``pyramid``, ``detect``,
        ``track`` and ``stereo``. Exists before the estimator has a state, unlike a snapshot's timings."""

    def __repr__(self) -> str: ...

class VioConfig:
    """VIO configuration loaded from the package's JSON inputs."""

    def __init__(self) -> None:
        """Default VIO configuration."""

    @staticmethod
    def from_json(text: str) -> VioConfig:
        """Read a VIO config JSON file; keys it omits keep their default."""

    def to_json(self) -> str:
        """Write the config back in the JSON format slam-rs reads, ``value0`` wrapper and all."""

    optical_flow_image_safe_radius: float
    """Circular mask that hides a fisheye's black corners, in pixels; 0 disables it."""
    vio_gravity_ms2: float
    """Magnitude of gravity in m/s²; the feed sets the recording's own value when it carries one."""
    vio_bootstrap: str
    """``config.vio_bootstrap``: ``"stereo"`` (default) or ``"inertial"``; anything else raises ``ValueError``."""
    @property
    def panorama_dual_fisheye(self) -> bool:
        """``config.panorama_dual_fisheye``: the caller hands ``Vio`` the factory lenses (``dual_fisheye``)."""

    def __repr__(self) -> str: ...

class Calibration:
    """Camera-IMU calibration: extrinsics, intrinsics and the noise model."""

    @staticmethod
    def from_json(text: str) -> Calibration:
        """Read a calibration JSON file."""

    @staticmethod
    def from_catalog(cameras: Sequence[CameraCalib], imu: ImuCalib) -> Calibration:
        """Build the calibration from the feed's dataclasses; ``imu.imu_T_body`` is unused."""

    def to_json(self) -> str:
        """Write the calibration back in the JSON format slam-rs reads, ``value0`` wrapper and all."""

    @property
    def camera_count(self) -> int: ...
    @property
    def resolution(self) -> list[tuple[int, int]]:
        """Each camera's ``(width, height)`` in pixels, in rig order."""

    @property
    def camera_models(self) -> list[Literal["pinhole", "kb4", "pinhole-radtan8", "ds", "eucm", "ucm"]]:
        """Camera model names in rig order."""

    @property
    def intrinsics(self) -> list[list[float]]:
        """fx, fy, cx, cy, then model-specific parameters per camera."""

    @property
    def imu_T_cam(self) -> list[Float64[ndarray, "4 4"]]:
        """Copied camera-to-IMU matrices in rig order."""

    @property
    def imu_update_rate(self) -> float:
        """IMU rate in Hz."""

    @property
    def gyro_noise_std(self) -> list[float]:
        """Gyroscope noise density per axis."""

    @property
    def accel_noise_std(self) -> list[float]:
        """Accelerometer noise density per axis."""

    @property
    def gyro_bias_std(self) -> list[float]:
        """Gyroscope bias random walk per axis."""

    @property
    def accel_bias_std(self) -> list[float]:
        """Accelerometer bias random walk per axis."""

    @property
    def cam_time_offset_ns(self) -> int:
        """Camera clock offset in nanoseconds."""

    def __repr__(self) -> str: ...

class FlowFrame:
    """What one :meth:`OpticalFlow.process` call produced, copied out of the frontend."""

    @property
    def t_ns(self) -> int: ...
    @property
    def camera_count(self) -> int: ...
    @property
    def cell_size(self) -> int:
        """``optical_flow_detection_grid_size``: the occupancy cell's side in pixels."""

    @property
    def cell_origin(self) -> tuple[int, int]:
        """``(x_start, y_start)``: the top-left corner of cell ``(0, 0)`` in pixels."""

    def ids(self, camera: int) -> Int64[ndarray, " n_tracks"]:
        """One camera's keypoint ids, ascending."""

    def positions(self, camera: int) -> Float32[ndarray, "n_tracks 2"]:
        """One camera's keypoint positions in pixels."""

    def transforms(self, camera: int) -> Float32[ndarray, "n_tracks 2 3"]:
        """One camera's 2x3 warps ``[[m00, m01, tx], [m10, m11, ty]]``."""

    def occupancy(self, camera: int) -> Int32[ndarray, "rows columns"]:
        """One camera's occupancy counts over camera 0's detection grid."""

    def num_new(self, camera: int) -> int:
        """Ids one camera gained on this frameset: detections plus stereo matches."""

    def num_tracks(self, camera: int) -> int:
        """Keypoints one camera carries."""

    def __repr__(self) -> str: ...

class OpticalFlow:
    """Frame-to-frame optical flow, driven one frameset at a time.

    Pattern 51 only, which is what every shipped config asks for; another
    ``optical_flow_pattern`` raises ``ValueError`` rather than tracking with the
    wrong pattern.

    A value the core refuses is a ``ValueError``, an object of the wrong type a
    ``TypeError``, an integer outside the parameter's own type an
    ``OverflowError`` and a camera past the end of the rig an ``IndexError``.
    None of them is a Rust panic, which would arrive as a
    ``pyo3_runtime.PanicException`` that ``except Exception`` does not catch.
    """

    def __init__(
        self,
        calibration: Calibration,
        config: VioConfig,
        *,
        threads: int = 1,
        max_keypoints: int | None = None,
    ) -> None:
        """Build a frontend for one rig; JSON files arrive through ``from_json``.

        Raises ``ValueError`` on a config the frontend cannot run — another
        pattern or flow type, a detector threshold ladder that never ends or
        never runs, more pyramid levels than the patch buffers allow — on a
        ``max_keypoints`` or ``threads`` past the core's ceiling, and on a
        calibration whose resolution over ``optical_flow_detection_grid_size``
        asks for more occupancy cells than one buffer may hold. Each of those is
        a memory or thread request rather than a plain number.
        """

    @property
    def camera_count(self) -> int: ...
    @property
    def last_keypoint_id(self) -> int: ...
    @property
    def t_ns(self) -> int | None:
        """Timestamp of the last accepted frameset, or None before the first."""

    def process(self, t_ns: int, images: Sequence[UInt8[ndarray, "h w"]]) -> FlowFrame:
        """Track and detect on one frameset of ``camera_count`` C-contiguous images.

        Raises ``ValueError`` on a bad dtype, rank or layout, on the wrong number
        of images, unless every image is the size the calibration gives its
        camera, and unless ``t_ns`` is strictly after the last accepted frameset.
        Any ``int64`` is a timestamp, negative ones included; a refused frameset
        leaves the frontend exactly as the last accepted one did.
        """

    def __repr__(self) -> str: ...


def catalog_camera_calib(index: int, statics: CameraStatics, downscale: int) -> CameraCalib: ...
def catalog_imu_calib(calibration: ImuCalibration, imu_t_body: Float64[ndarray, "4 4"], applied_time_shift_ns: int) -> ImuCalib: ...
def catalog_frame_nearest_anchor(times: Int64[ndarray, " n_frames"], cursor: int, anchor_t_ns: int, tolerance_ns: int) -> tuple[int | None, int]: ...
def catalog_match_framesets(camera_t_ns: Sequence[Int64[ndarray, " n_frames"]], tolerance_ns: int) -> tuple[Int64[ndarray, " n_framesets"], Int64[ndarray, "n_framesets n_cameras"]]: ...
def catalog_pair_imu(gyro_t_ns: Int64[ndarray, " n_gyro"], gyro_rad_s: Float64[ndarray, "n_gyro 3"], accel_t_ns: Int64[ndarray, " n_accel"], accel_m_s2: Float64[ndarray, "n_accel 3"], interpolate: bool) -> ImuStream: ...


def catalog_rig_profile(dataset: str) -> "RigProfile":
    """Return the camera selection and clock rules owned by Rust."""
    ...


class InsvLens:
    """One lens of an Insta360 factory calibration at the track resolution, window crop applied (Mei sphere model)."""

    @property
    def cx(self) -> float: ...
    @property
    def cy(self) -> float: ...
    @property
    def pano_from_lens_rotation(self) -> list[list[float]]:
        """``R_P_lens``, row-major: lens vectors into the panorama frame P."""
    @property
    def pano_centre_m(self) -> list[float]:
        """Lens centre in P, metres, origin at the midpoint of the lens centres."""
    def radius_px(self, theta: Sequence[float]) -> Float64[ndarray, " n"]:
        """Image radius in track pixels of rays ``theta`` radians off the axis (radial part of the model only)."""


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
    """The dual-fisheye lane's stitch without ``Vio``: two lens images in, the 1920x960 viewing panorama out.

    Built from the factory inputs ``Vio(dual_fisheye=...)`` takes; the same tables and host remap as the lane.
    """

    def __init__(self, lens_string: str, window_crop_px: tuple[int, int, int, int, int, int] | None, lens_size: tuple[int, int]) -> None:
        """``lens_string``: raw V3/V6 factory string; ``window_crop_px``: its window crop or None; ``lens_size``: ``(width, height)``.

        Raises ``ValueError`` on a bad string.
        """
    def panorama(self, lens0: UInt8[ndarray, "h w"], lens1: UInt8[ndarray, "h w"]) -> UInt8[ndarray, "960 1920"]:
        """The panorama of lens 0's and lens 1's C-contiguous images (lens 0's hemisphere on the left); ``ValueError`` on a wrong size."""
