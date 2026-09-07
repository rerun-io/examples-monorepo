"""Types for the compiled ``slam_rs._core`` extension (built by the ``slam-rs-build`` task).

The signatures here are the only static check on the FFI boundary, so they stay
exact: no ``Any``, and every array carries its dtype.
"""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from jaxtyping import Float32, Int32, Int64, UInt8
from numpy import ndarray
from numpy.typing import NDArray

from slam_rs.catalog_feed import CameraCalib, ImuCalib

__version__: str

class VioStatus:
    """How far the estimator has got.

    A PyO3 enum, not a ``enum.Enum``: it carries no ``name`` or ``value``, it is
    unhashable, and ``VioStatus(1)`` raises ``TypeError``. It does convert to
    ``int`` and compares equal both to its own variants and to their ordinals.
    """

    NotInitialised: ClassVar[VioStatus]
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
    def world_from_rig(self) -> NDArray[np.float64]:
        """``[tx, ty, tz, qx, qy, qz, qw]``, metres and a unit quaternion (xyzw)."""

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Rig velocity in the world frame, m/s, shape ``(3,)``."""

    @property
    def gyro_bias(self) -> NDArray[np.float64]:
        """Gyroscope bias estimate, rad/s, shape ``(3,)``."""

    @property
    def accel_bias(self) -> NDArray[np.float64]:
        """Accelerometer bias estimate, m/s^2, shape ``(3,)``."""

    def __repr__(self) -> str: ...

class Vio:
    """The estimator, driven one frameset at a time."""

    def __init__(self, camera_count: int = 2, min_imu_samples: int = 1) -> None: ...
    @property
    def camera_count(self) -> int: ...
    def push_imu(self, t_ns: int, gyro: Sequence[float], accel: Sequence[float]) -> None:
        """Add one IMU sample; raises ``ValueError`` unless ``t_ns`` follows the last one."""

    def push_imu_batch(
        self,
        t_ns: NDArray[np.int64],
        gyro: NDArray[np.float64],
        accel: NDArray[np.float64],
    ) -> None:
        """Add ``n`` samples: ``t_ns`` is ``(n,)``, ``gyro`` and ``accel`` are ``(n, 3)``."""

    def track(self, t_ns: int, images: Sequence[NDArray[np.uint8]]) -> VioResult:
        """Process one frameset of ``camera_count`` C-contiguous ``(h, w)`` uint8 images."""

class VioConfig:
    """basalt's ``VioConfig``, as ``data/**/*_config.json`` carries it."""

    def __init__(self) -> None:
        """basalt's own defaults, the ones its constructor sets."""

    @staticmethod
    def from_json(text: str) -> VioConfig:
        """Read one of basalt's config files; keys it omits keep their default."""

    def to_json(self) -> str:
        """Write the config back in basalt's shape, ``value0`` wrapper and all."""

    optical_flow_image_safe_radius: float
    """Circular mask that hides a fisheye's black corners, in pixels; 0 disables it."""

    def __repr__(self) -> str: ...

class Calibration:
    """basalt's camera-IMU calibration: extrinsics, intrinsics and the noise model."""

    @staticmethod
    def from_json(text: str) -> Calibration:
        """Read one of basalt's calibration files."""

    @staticmethod
    def from_catalog(cameras: Sequence[CameraCalib], imu: ImuCalib) -> Calibration:
        """Build the calibration from the feed's dataclasses; ``imu.imu_T_body`` is unused."""

    def to_json(self) -> str:
        """Write the calibration back in basalt's shape, ``value0`` wrapper and all."""

    @property
    def camera_count(self) -> int: ...
    @property
    def resolution(self) -> list[tuple[int, int]]:
        """Each camera's ``(width, height)`` in pixels, in rig order."""

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

    def ids(self, camera: int) -> Int64[ndarray, " n"]:
        """One camera's keypoint ids, ascending."""

    def positions(self, camera: int) -> Float32[ndarray, "n 2"]:
        """One camera's keypoint positions in pixels."""

    def transforms(self, camera: int) -> Float32[ndarray, "n 2 3"]:
        """One camera's 2x3 warps ``[[m00, m01, tx], [m10, m11, ty]]``."""

    def responses(self, camera: int) -> Float32[ndarray, " n"]:
        """One camera's detector responses; ``-1`` where basalt records none."""

    def occupancy(self, camera: int) -> Int32[ndarray, "rows columns"]:
        """One camera's occupancy counts over camera 0's detection grid."""

    def num_new(self, camera: int) -> int:
        """Ids one camera gained on this frameset: detections plus stereo matches."""

    def num_tracks(self, camera: int) -> int:
        """Keypoints one camera carries."""

    def __repr__(self) -> str: ...

class OpticalFlow:
    """basalt's ``FrameToFrameOpticalFlow``, driven one frameset at a time.

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
        """Build a frontend for one rig; basalt's own files arrive through ``from_json``.

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
    def frame_counter(self) -> int: ...
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
