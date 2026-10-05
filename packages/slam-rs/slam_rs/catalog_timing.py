"""Match camera timestamps and pair inertial samples without catalog I/O."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs import _core


@dataclass(slots=True, frozen=True)
class ImuStream:
    """A segment's paired inertial measurements on one clock."""

    t_ns: Int64[ndarray, " n_samples"]
    """Sample timestamps, strictly increasing."""
    gyro_rad_s: Float64[ndarray, "n_samples 3"]
    """Angular velocity, rad/s."""
    accel_m_s2: Float64[ndarray, "n_samples 3"]
    """Linear acceleration, m/s^2."""

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])

    def between(self, first_ns: int, last_ns: int) -> "ImuStream":
        """The samples with ``first_ns < t <= last_ns``, half-open at the start."""
        start: int = int(np.searchsorted(self.t_ns, first_ns, side="right"))
        stop: int = int(np.searchsorted(self.t_ns, last_ns, side="right"))
        # Own the small result so queued frames do not retain a whole segment.
        return ImuStream(
            t_ns=self.t_ns[start:stop].copy(),
            gyro_rad_s=self.gyro_rad_s[start:stop].copy(),
            accel_m_s2=self.accel_m_s2[start:stop].copy(),
        )


def _frame_nearest_anchor(times: Int64[ndarray, " n_frames"], cursor: int, anchor_t_ns: int, tolerance_ns: int) -> tuple[int | None, int]:
    """Select a camera frame nearest the anchor, breaking ties towards the later frame.

    Accept the inclusive tolerance. On an incomplete frameset, discard a selected
    frame only if it is earlier than the anchor: it cannot serve a later anchor.
    A future frame stays available. On completion, the caller advances past each
    selected frame so no image is reused.

    Args:
        times: The camera's frame timestamps, in time order.
        cursor: The first frame no earlier frameset has consumed.
        anchor_t_ns: Camera 0's frame timestamp.
        tolerance_ns: How far a frame may sit from the anchor's and still join it.

    Returns:
        The frame this camera contributes, or ``None`` if it has none within the
        tolerance, and the cursor this camera stands on if the frameset falls.
    """
    return _core.catalog_frame_nearest_anchor(times, cursor, anchor_t_ns, tolerance_ns)


def match_framesets(camera_t_ns: Sequence[Int64[ndarray, " n_frames"]], tolerance_ns: int) -> tuple[Int64[ndarray, " n_framesets"], Int64[ndarray, "n_framesets n_cameras"]]:
    """Group camera frames around camera 0 anchors.

    Every camera must have a nearest frame within the inclusive tolerance.
    Use the median timestamp; for even counts, use the lower middle plus half
    the integer gap. Complete framesets consume all selected images once.
    Incomplete ones discard only frames earlier than the anchor.
    Allow max(1, ceil(interior_anchors * 0.001)) incomplete interior anchors;
    exterior anchors do not count against the rig.

    Args:
        camera_t_ns: Each fed camera's frame timestamps, in time order.
        tolerance_ns: How far a frame may sit from the anchor's and still join it.

    Returns:
        The frameset timestamps, and the frame each camera contributes to each.

    Raises:
        ValueError: If no camera was given, a camera has no frames, the frameset
            timestamps do not strictly increase, too many interior framesets are
            incomplete, or no frameset is complete.
    """
    return _core.catalog_match_framesets(camera_t_ns, tolerance_ns)


def pair_accel_onto_gyro(
    gyro_t_ns: Int64[ndarray, " n_gyro"],
    gyro_rad_s: Float64[ndarray, "n_gyro 3"],
    accel_t_ns: Int64[ndarray, " n_accel"],
    accel_m_s2: Float64[ndarray, "n_accel 3"],
) -> ImuStream:
    """Linearly interpolate the accelerometer onto the gyroscope's timestamps.

    A gyroscope sample outside the accelerometer's own span is dropped rather
    than held at an endpoint: ``numpy.interp`` clamps, which would feed the
    estimator a constant acceleration over a stretch it has no measurement for.
    Require accelerometer coverage on both sides of each retained gyroscope sample.

    Args:
        gyro_t_ns: Gyroscope timestamps, strictly increasing.
        gyro_rad_s: Angular velocity, rad/s.
        accel_t_ns: Accelerometer timestamps, non-decreasing; a repeated one keeps its first sample.
        accel_m_s2: Linear acceleration, m/s^2.

    Returns:
        One stream on the gyroscope's clock, covering only the overlap.

    Raises:
        ValueError: If a channel is too short to interpolate with, or the two
            spans do not overlap, so the paired stream would be empty.
    """
    return _core.catalog_pair_imu(gyro_t_ns, gyro_rad_s, accel_t_ns, accel_m_s2, True)
