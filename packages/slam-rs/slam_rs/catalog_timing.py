"""Match camera timestamps and pair inertial samples without catalog I/O."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray


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
        keep: Bool[ndarray, " n_samples"] = (self.t_ns > first_ns) & (self.t_ns <= last_ns)
        return ImuStream(t_ns=self.t_ns[keep], gyro_rad_s=self.gyro_rad_s[keep], accel_m_s2=self.accel_m_s2[keep])


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
    index: int = cursor
    if index >= len(times):
        return None, cursor
    while index + 1 < len(times) and abs(int(times[index + 1]) - anchor_t_ns) <= abs(int(times[index]) - anchor_t_ns):
        index += 1
    if abs(int(times[index]) - anchor_t_ns) > tolerance_ns:
        return None, (index + 1 if int(times[index]) < anchor_t_ns else cursor)
    return index, cursor


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
    if not camera_t_ns:
        raise ValueError("a frameset needs at least one camera")
    for position, times in enumerate(camera_t_ns):
        if times.size == 0:
            raise ValueError(f"camera {position} has no frames, so it is not part of this recording")
    # The span every camera covers: only an anchor inside it can be expected to
    # have partners, so only a drop inside it counts against the run.
    overlap_start: int = max(int(times[0]) for times in camera_t_ns)
    overlap_end: int = min(int(times[-1]) for times in camera_t_ns)
    cursors: list[int] = [0] * len(camera_t_ns)
    t_ns: list[int] = []
    rows: list[list[int]] = []
    interior_anchors: int = 0
    interior_drops: int = 0
    for anchor_index, anchor_t_ns in enumerate(camera_t_ns[0].tolist()):
        interior: bool = overlap_start <= anchor_t_ns <= overlap_end
        interior_anchors += interior
        row: list[int] = [anchor_index]
        # Where each camera would land: commit these to the cursors only
        # once the whole frameset stands.
        selected: list[int] = list(cursors)
        for position in range(1, len(camera_t_ns)):
            # The cursor this camera stands on if the frameset falls: a camera that
            # fell behind can never catch this anchor again, one running ahead keeps
            # its frame. On the frameset standing, that value is where it already was.
            index, cursors[position] = _frame_nearest_anchor(camera_t_ns[position], cursors[position], anchor_t_ns, tolerance_ns)
            if index is None:
                break
            selected[position] = index
            row.append(index)
        if len(row) != len(camera_t_ns):
            interior_drops += interior
            continue
        for position in range(1, len(camera_t_ns)):
            cursors[position] = selected[position] + 1
        members: list[int] = sorted(int(camera_t_ns[position][frame]) for position, frame in enumerate(row))
        middle: int = len(members) // 2
        frameset_t_ns: int = members[middle] if len(members) % 2 else members[middle - 1] + (members[middle] - members[middle - 1]) // 2
        if t_ns and frameset_t_ns <= t_ns[-1]:
            raise ValueError(f"frameset timestamps are not strictly increasing: {frameset_t_ns} follows {t_ns[-1]}")
        t_ns.append(frameset_t_ns)
        rows.append(row)
    # Allow one interior drop per thousand anchors, with a minimum of one.
    allowed_drops: int = max(1, math.ceil(interior_anchors * 0.001))
    if interior_drops > allowed_drops:
        raise ValueError(
            f"{interior_drops} of {interior_anchors} interior framesets are incomplete, more than the {allowed_drops} "
            f"basalt allows: the cameras are not one recording within {tolerance_ns} ns"
        )
    if not rows:
        raise ValueError(f"no frameset has all {len(camera_t_ns)} cameras within {tolerance_ns} ns of camera 0")
    return np.array(t_ns, dtype=np.int64), np.array(rows, dtype=np.int64)


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
    # Deduplicate both channels before pairing, keeping the first
    # sample of each equal-timestamp run. Keeping the second duplicate would
    # change interpolation across the following gap.
    first_of_run: Bool[ndarray, " n_accel"] = np.ones(accel_t_ns.size, dtype=bool)
    first_of_run[1:] = np.diff(accel_t_ns) != 0
    accel_t_ns = accel_t_ns[first_of_run]
    accel_m_s2 = accel_m_s2[first_of_run]
    if gyro_t_ns.size == 0 or accel_t_ns.size < 2:
        raise ValueError(
            f"pairing needs a gyroscope sample and two accelerometer samples to interpolate between; "
            f"got {gyro_t_ns.size} gyro and {accel_t_ns.size} accel samples"
        )
    inside: Bool[ndarray, " n_gyro"] = (gyro_t_ns >= accel_t_ns[0]) & (gyro_t_ns <= accel_t_ns[-1])
    if not bool(inside.any()):
        # Refuse an empty inertial stream:
        # this rig requires paired IMU measurements.
        raise ValueError(
            f"the two inertial channels do not overlap, so nothing pairs: the gyroscope spans "
            f"{int(gyro_t_ns[0])}..{int(gyro_t_ns[-1])} ns and the accelerometer {int(accel_t_ns[0])}..{int(accel_t_ns[-1])} ns"
        )
    paired_t_ns: Int64[ndarray, " n_paired"] = gyro_t_ns[inside]
    interpolated: Float64[ndarray, "n_paired 3"] = np.column_stack(
        [np.interp(paired_t_ns, accel_t_ns, accel_m_s2[:, axis]) for axis in range(accel_m_s2.shape[1])]
    )
    return ImuStream(t_ns=paired_t_ns, gyro_rad_s=gyro_rad_s[inside], accel_m_s2=interpolated)


