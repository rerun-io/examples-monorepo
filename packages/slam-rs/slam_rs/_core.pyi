"""Types for the compiled ``slam_rs._core`` extension (built by the ``slam-rs-build`` task).

The signatures here are the only static check on the FFI boundary, so they stay
exact: no ``Any``, and every array carries its dtype.
"""

import enum
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__version__: str

class VioStatus(enum.Enum):
    """How far the estimator has got."""

    NotInitialised = 0
    NeedMoreImu = 1
    Tracking = 2

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
