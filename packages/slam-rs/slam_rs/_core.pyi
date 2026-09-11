"""Types for the compiled ``slam_rs._core`` extension (built by the ``slam-rs-build`` task).

The signatures here are the only static check on the FFI boundary, so they stay
exact: no ``Any``, and every array carries its dtype.
"""

from collections.abc import Sequence
from typing import ClassVar, Literal

from jaxtyping import Bool, Float32, Float64, Int32, Int64, UInt8
from numpy import ndarray

from slam_rs.catalog_feed import CameraCalib, ImuCalib

__version__: str

gpu_backend: Literal["wgpu"] | None
"""Which GPU runtime this build's frontend carries, or None for the CPU-only default build.

A core built with the ``gpu-wgpu`` cargo feature carries ``"wgpu"``; nothing a
caller passes to ``gpu=True`` names the runtime, this does, and it is what a
fleet row's lane is named from (:func:`slam_rs.apis.fleet_check.this_lane`).
``None`` is the default build, whose ``gpu=True`` is refused.
"""

class VioStatus:
    """How far the estimator has got.

    Offline mode has exactly these two states: a measured frameset always has a
    state and an uncovered one never does, so there is no third, "initialising"
    status to branch on.

    A PyO3 enum, not a ``enum.Enum``: it carries no ``name`` or ``value``, it is
    unhashable, and ``VioStatus(1)`` raises ``TypeError``. It does convert to
    ``int`` and compares equal both to its own variants and to their ordinals.
    """

    NeedMoreImu: ClassVar[VioStatus]
    Tracking: ClassVar[VioStatus]
    __hash__: ClassVar[None]

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class VioResult:
    """What one :meth:`Vio.track` call produced."""

    @property
    def status(self) -> VioStatus: ...
    @property
    def t_ns(self) -> int: ...
    @property
    def world_from_rig(self) -> Float64[ndarray, " 7"]:
        """``[tx, ty, tz, qx, qy, qz, qw]``, metres and a unit quaternion (xyzw)."""

    @property
    def velocity(self) -> Float64[ndarray, " 3"]:
        """Rig velocity in the world frame, m/s."""

    @property
    def gyro_bias(self) -> Float64[ndarray, " 3"]:
        """Gyroscope bias estimate, rad/s."""

    @property
    def accel_bias(self) -> Float64[ndarray, " 3"]:
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
        """

    def __repr__(self) -> str: ...

class Vio:
    """The VIO pipeline, driven one frameset at a time.

    Offline mode (D17): the frontend and the backend run to completion in the
    calling thread, so every result is final and a repeat run over the same
    input is bit-identical.

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
        gpu: bool = False,
    ) -> None:
        """Build the pipeline for one rig; JSON files arrive through ``from_json``.

        ``gpu`` runs the frontend's pyramid, patch build and KLT tracker through
        CubeCL on this host's GPU instead of the CPU port. The default is the
        CPU, which is what every accuracy reference was produced on. A core
        built without the ``gpu-wgpu`` cargo feature raises ``ValueError`` for
        ``gpu=True`` rather than quietly running on the CPU, and so does a host
        that has the feature and no GPU to run it on: a missing driver library,
        a driver that will not initialise, no visible device and no graphics
        adapter each raise ``ValueError`` naming what is absent. None of them is
        a ``PanicException``, which is what CubeCL's own unwrapped bring-up
        would otherwise produce. A failure no probe anticipates is caught rather
        than raised, so it is a ``ValueError`` too — with the runtime's own panic
        message left on stderr, which is the only account of a case the probe did
        not know to ask about.

        ``threads`` is **inert on the GPU lane**: only the CPU patch tracker
        reads it and the GPU tracker holds no work pool. It is accepted rather
        than refused alongside ``gpu=True`` so one call site can select either
        lane.

        Raises ``ValueError`` on everything :class:`OpticalFlow` refuses, and on
        a config asking for a path this port does not have:
        ``vio_linearization_type`` other than ``ABS_QR``, ``vio_sqrt_marg``
        false, or ``vio_enforce_realtime``, which Offline mode cannot honour.
        """

    @property
    def camera_count(self) -> int: ...
    @property
    def gpu(self) -> bool:
        """Whether the frontend runs on the GPU."""
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

        Raises ``ValueError`` on a bad dtype, rank or layout, on the wrong number
        of images, unless every image is the size the calibration gives its
        camera, and unless ``t_ns`` is strictly after the last accepted frameset.
        On a GPU lane a device that dies mid-run raises ``ValueError`` here as
        well, never a ``PanicException``.
        """

    def snapshot(self) -> VioSnapshot | None:
        """The window and the last measured frame, or None before the first one."""

    def flow_frame(self) -> FlowFrame | None:
        """The keypoints the frontend tracked on the last accepted frameset, or None before the first."""

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
