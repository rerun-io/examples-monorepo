"""Trajectory CSV input/output and rigid-aligned absolute trajectory error.

The estimate drives nearest-neighbor association within 5 ms. Rigid Umeyama
alignment fixes scale at one, then positional RMSE measures the residual.
The gate, replay tool and tests share this implementation.

CSV columns are t_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z, with a comment header.
Timestamps remain integer nanoseconds. Quaternions stay w-first in memory
and convert to XYZW only at the logging boundary.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform

ASSOCIATION_TOLERANCE_NS: int = 5_000_000
"""Largest timestamp gap a reference pose and a candidate pose may be associated across."""
MIN_ASSOCIATED_POSES: int = 10
"""Fewest associations the gate accepts before it calls the comparison meaningless."""
CSV_HEADER: str = "#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z"
"""CSV header emitted by :func:`write_trajectory`."""


@dataclass(slots=True, frozen=True)
class Trajectory:
    """A time-stamped rigid-body trajectory in one world frame."""

    t_ns: Int64[ndarray, " n"]
    """Pose timestamps, integer nanoseconds on the estimator's clock."""
    position_m: Float64[ndarray, "n 3"]
    """World-frame positions, metres."""
    quaternion_wxyz: Float64[ndarray, "n 4"]
    """World-from-body rotations, **w-first** unit quaternions."""

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])


@dataclass(slots=True, frozen=True)
class Association:
    """Which reference poses found a candidate pose inside the tolerance."""

    matched: Bool[ndarray, " n_reference"]
    """True where the reference pose has a candidate within the tolerance."""
    candidate_index: Int64[ndarray, " n_reference"]
    """Nearest candidate index per reference pose; meaningless where ``matched`` is False."""

    @property
    def count(self) -> int:
        """How many reference poses were associated."""
        return int(self.matched.sum())


@dataclass(slots=True, frozen=True)
class AteResult:
    """Rigid-aligned absolute trajectory error of a candidate against a reference."""

    rmse_m: float
    """Root-mean-square position residual after alignment."""
    max_m: float
    """Largest position residual after alignment."""
    median_m: float
    """Median position residual after alignment."""
    n_associated: int
    """Estimate poses the association matched."""
    n_estimate: int
    """Poses in the estimate, which is the trajectory under test."""
    n_reference: int
    """Poses in the reference trajectory."""
    count_delta: float
    """Relative pose-count difference, ``|n_reference - n_estimate| / n_estimate``."""
    alignment: SimilarityTransform
    """The rigid transform that took the reference positions onto the estimate's."""

    def summary(self) -> str:
        """Pose counts and trajectory errors, in centimetres."""
        return (
            f"poses: estimate={self.n_estimate} reference={self.n_reference} "
            f"associated={self.n_associated} (count delta {self.count_delta:.1%})\n"
            f"ATE  : rmse={self.rmse_m * 100:.2f} cm  max={self.max_m * 100:.2f} cm  median={self.median_m * 100:.2f} cm"
        )


def empty_trajectory() -> Trajectory:
    """A trajectory with no poses.

    What a reference a run does not have looks like: a segment with no ``gt``
    layer. Callers then test :func:`len` rather than ``None``, and the one
    array shape is right for :func:`numpy.concatenate`.
    """
    return Trajectory(t_ns=np.zeros(0, dtype=np.int64), position_m=np.zeros((0, 3)), quaternion_wxyz=np.zeros((0, 4)))


def read_trajectory(path: Path) -> Trajectory:
    """Read a EuRoC-style trajectory CSV.

    Comment lines starting with ``#`` are skipped, which covers both the trajectory
    header (``#timestamp [ns], p_x, ...``) and the sidecar header
    (``#timestamp [ns],p_RS_R_x [m], ...``). The timestamp column is parsed as an
    integer directly from the text: routing it through float64 would be exact for
    today's clocks but silently lossy past 2^53 ns.

    Args:
        path: Trajectory CSV.

    Returns:
        The trajectory, with w-first quaternions exactly as stored.

    Raises:
        ValueError: If a data row does not have eight comma-separated fields.
    """
    rows: list[list[str]] = [line.split(",") for line in path.read_text().splitlines() if line and not line.lstrip().startswith("#")]
    for line_number, fields in enumerate(rows):
        if len(fields) != 8:
            raise ValueError(f"{path}: data row {line_number} has {len(fields)} fields, expected 8")
    t_ns: Int64[ndarray, " n"] = np.array([int(fields[0]) for fields in rows], dtype=np.int64)
    values: Float64[ndarray, "n 7"] = np.array([[float(field) for field in fields[1:]] for fields in rows], dtype=np.float64).reshape(len(rows), 7)
    return Trajectory(t_ns=t_ns, position_m=values[:, 0:3], quaternion_wxyz=values[:, 3:7])


def write_trajectory(path: Path, trajectory: Trajectory) -> None:
    """Write a trajectory in the trajectory CSV format, timestamps as exact integers.

    Args:
        path: Destination CSV; parent directories are created.
        trajectory: Poses to write, w-first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [CSV_HEADER]
    # ``tolist()`` gives Python floats, whose repr round-trips exactly; formatting a
    # numpy scalar with ``!r`` would write "np.float64(0.0)" under numpy 2.
    for t_ns, pose in zip(trajectory.t_ns.tolist(), np.hstack([trajectory.position_m, trajectory.quaternion_wxyz]).tolist(), strict=True):
        lines.append(f"{t_ns}," + ",".join(repr(value) for value in pose))
    path.write_text("\n".join(lines) + "\n")


def shift_clock(trajectory: Trajectory, offset_ns: int) -> Trajectory:
    """Move a trajectory onto another clock by adding a constant nanosecond offset.

    The catalog indexes a segment on ``video_time``, which is relative to
    ``property:capture:start_time_ns``; trajectory CSV exports, including the ``gt.csv``
    sidecars, are on the absolute device clock. On the Index smoke segment the two
    differ by 10,433,867,587,166 ns, so exporting relative timestamps produces a
    file that associates with **nothing**. This is the one place the conversion
    happens, and it happens at the CSV boundary.

    Args:
        trajectory: Poses on the source clock.
        offset_ns: Nanoseconds to add; ``capture.start_time_ns`` to go from ``video_time`` to absolute.

    Returns:
        The same poses on the shifted clock. Positions and rotations are shared, not copied.

    Raises:
        ValueError: If the shift would take a timestamp out of int64. Timestamps
            are exact integers because a nanosecond clock past 2**53 is not one
            in a float, and the same choice leaves the last representable
            timestamp one addition from ``-2**63``: int64 addition wraps it to
            the other end of the clock in silence, which reads as a run that
            tracked a different window rather than as a failed conversion
            (S25 review).
    """
    # Which end of the clock the shift can leave is the offset's sign, and an
    # empty trajectory has no timestamp to move at all.
    if len(trajectory) != 0:
        extreme: int = int(trajectory.t_ns.max() if offset_ns > 0 else trajectory.t_ns.min())
        if not -(2**63) <= extreme + offset_ns < 2**63:
            raise ValueError(
                f"a timestamp of {extreme} ns cannot be shifted by {offset_ns} ns: the result leaves the int64 clock, "
                f"and int64 addition would wrap it to the other end instead of failing"
            )
    return Trajectory(
        t_ns=trajectory.t_ns + np.int64(offset_ns),
        position_m=trajectory.position_m,
        quaternion_wxyz=trajectory.quaternion_wxyz,
    )


def associate(reference: Trajectory, candidate: Trajectory, tolerance_ns: int = ASSOCIATION_TOLERANCE_NS) -> Association:
    """Match every reference pose to its nearest candidate pose in time.

    Both trajectories must be sorted by timestamp.

    Args:
        reference: Trajectory whose timestamps drive the association.
        candidate: Trajectory searched for the nearest timestamp.
        tolerance_ns: Largest gap that still counts as a match, inclusive.

    Returns:
        The per-reference-pose match mask and nearest candidate index.

    Raises:
        ValueError: If either trajectory is empty.
    """
    if len(reference) == 0 or len(candidate) == 0:
        raise ValueError(f"association needs poses on both sides; got reference={len(reference)} candidate={len(candidate)}")
    index: Int64[ndarray, " n_reference"] = np.searchsorted(candidate.t_ns, reference.t_ns).astype(np.int64)
    index = np.clip(index, 1, len(candidate) - 1)
    # Step back one when the earlier candidate is the closer of the two neighbours.
    index -= np.abs(candidate.t_ns[index - 1] - reference.t_ns) < np.abs(candidate.t_ns[index] - reference.t_ns)
    matched: Bool[ndarray, " n_reference"] = np.abs(candidate.t_ns[index] - reference.t_ns) <= tolerance_ns
    return Association(matched=matched, candidate_index=index)


def rigid_alignment(source: Float64[ndarray, "n 3"], target: Float64[ndarray, "n 3"]) -> SimilarityTransform:
    """Least-squares rigid transform from source to target, with scale fixed at one.

    No variance floor is imposed: stationary and micrometre-scale trajectories
    can align without raising. Tests cross-check the shared Umeyama helper on
    inputs where both implementations are defined.

    Args:
        source: Points to move, one XYZ per row.
        target: Points to move onto, row-aligned with ``source``.

    Returns:
        The rigid transform, with ``scale`` always ``1.0``.

    Raises:
        ValueError: If the shapes differ or there are no points.
    """
    if source.shape != target.shape:
        raise ValueError(f"source and target shapes must match; got {source.shape} vs {target.shape}")
    if source.shape[0] == 0:
        raise ValueError("rigid alignment needs at least one point pair")
    source_mean: Float64[ndarray, " 3"] = source.mean(axis=0)
    target_mean: Float64[ndarray, " 3"] = target.mean(axis=0)
    svd: tuple[Float64[ndarray, "3 3"], Float64[ndarray, " 3"], Float64[ndarray, "3 3"]] = np.linalg.svd(
        (target - target_mean).T @ (source - source_mean)
    )
    u: Float64[ndarray, "3 3"] = svd[0]
    vt: Float64[ndarray, "3 3"] = svd[2]
    # u and vt are orthogonal, so the determinant is exactly +/-1 and the sign is
    # never zero; this only ever flips a reflection back into a rotation.
    sign: float = float(np.sign(np.linalg.det(u @ vt)))
    dst_R_src: Float64[ndarray, "3 3"] = u @ np.diag([1.0, 1.0, sign]) @ vt
    return SimilarityTransform(dst_R_src=dst_R_src, dst_t_src=target_mean - dst_R_src @ source_mean, scale=1.0)


def ate(estimate: Trajectory, reference: Trajectory, tolerance_ns: int = ASSOCIATION_TOLERANCE_NS) -> AteResult:
    """Rigid-aligned absolute trajectory error against the reference trajectory.

    Each estimate pose selects its nearest reference pose within tolerance.
    Estimate-driven association prevents dense ground truth from changing metric
    weights. Scale stays one because fitting it would hide a metric VIO error.
    The metric needs at least one association; the ground-truth gate separately
    checks coverage and run validity.

    Args:
        estimate: Trajectory under test, whose poses drive the association.
        reference: Trajectory taken as truth, searched for the nearest pose.
        tolerance_ns: Association tolerance passed to :func:`associate`.

    Returns:
        The error statistics and the alignment that produced them.

    Raises:
        ValueError: If no pose associates, which leaves nothing to compare.
    """
    association: Association = associate(estimate, reference, tolerance_ns)
    source: Float64[ndarray, "n_associated 3"] = reference.position_m[association.candidate_index[association.matched]]
    target: Float64[ndarray, "n_associated 3"] = estimate.position_m[association.matched]
    if source.shape[0] == 0:
        raise ValueError(f"no pose associated within {tolerance_ns} ns; the two trajectories may be on different clocks")
    alignment: SimilarityTransform = rigid_alignment(source, target)
    errors: Float64[ndarray, " n_associated"] = np.linalg.norm(alignment.apply(source) - target, axis=1)
    return AteResult(
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        max_m=float(errors.max()),
        median_m=float(np.median(errors)),
        n_associated=association.count,
        n_estimate=len(estimate),
        n_reference=len(reference),
        count_delta=abs(len(reference) - len(estimate)) / len(estimate),
        alignment=alignment,
    )


def nonfinite_position_text(trajectory: Trajectory) -> str | None:
    """Why an estimate cannot be aligned, when one of its positions is not finite; None when every one is.

    :func:`ate` cannot refuse this itself in the shape its callers read:
    :func:`rigid_alignment` hands the cross-covariance to ``np.linalg.svd``,
    which raises ``LinAlgError: SVD did not converge`` on a NaN — not the
    :class:`ValueError` a caller catches, and not a row. Both fleet tools
    therefore ask this **before** the alignment, so a machine whose estimator
    diverged is reported rather than lost to a traceback, which is the one
    machine the fleet lane exists to find. The clause names the first offending
    pose, because the timestamp is where a diverged run is read from.

    Args:
        trajectory: The estimate about to be scored.

    Returns:
        The clause, ready for a row's ``unscored`` field, or None when the
        estimate is finite throughout (including when it has no pose at all).
    """
    finite: Bool[ndarray, " n"] = np.isfinite(trajectory.position_m).all(axis=1)
    if bool(finite.all()):
        return None
    first: int = int(np.flatnonzero(~finite)[0])
    return (
        f"{int((~finite).sum())} of {len(trajectory)} estimated positions is not finite, "
        f"the first at {int(trajectory.t_ns[first])} ns; there is nothing to align"
    )


def coverage(reference: Trajectory, candidate: Trajectory) -> float:
    """Fraction of the reference time span that the candidate spans.

    A candidate that tracks perfectly for the first second of a minute scores
    well on ATE and badly here, which is the point: ATE alone cannot see a run
    that stopped early.

    Args:
        reference: Trajectory taken as truth.
        candidate: Trajectory under test.

    Returns:
        Overlap of the two time spans divided by the reference span, clamped to
        ``[0, 1]``; ``0.0`` when the reference has no extent, and when either
        trajectory has no pose at all.
    """
    # Either side empty is that documented zero and not an `IndexError` out of
    # `t_ns[-1]`: a segment with no `gt` layer spans nothing to cover, and a
    # machine that tracked nothing covers nothing of it (S25 review).
    if len(reference) == 0 or len(candidate) == 0:
        return 0.0
    reference_span_ns: int = int(reference.t_ns[-1] - reference.t_ns[0])
    if reference_span_ns <= 0:
        return 0.0
    overlap_ns: int = int(min(reference.t_ns[-1], candidate.t_ns[-1]) - max(reference.t_ns[0], candidate.t_ns[0]))
    return float(np.clip(overlap_ns / reference_span_ns, 0.0, 1.0))


def extent_m(trajectory: Trajectory) -> float:
    """Bounding-box diagonal in metres, or zero for an empty trajectory.

    This diagnostic measures spatial extent; ground-truth acceptance is defined
    by the manifest baseline gate.

    Args:
        trajectory: The trajectory to measure.

    Returns:
        The bounding box's diagonal in metres.
    """
    if len(trajectory) == 0:
        return 0.0
    return float(np.linalg.norm(trajectory.position_m.max(axis=0) - trajectory.position_m.min(axis=0)))
