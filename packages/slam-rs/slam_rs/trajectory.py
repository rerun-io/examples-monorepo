"""EuRoC/basalt trajectory CSVs and the rigid-aligned ATE that gates a run.

The arithmetic is the basalt fork's ``golden_compare.py``: nearest-neighbour
association with a 5 ms tolerance, rigid Umeyama alignment of the candidate onto
the reference, and RMSE of the residual positions. It lives here so the gate,
the replay tool and the tests all use one implementation.

The file format is basalt's own: a ``#``-prefixed header line, then
``t_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z``. Timestamps are integer nanoseconds
and never pass through a float on the way in or out; the quaternion is
**w-first** on disk and stays w-first in memory, because that is what both the
``gt.csv`` sidecars and basalt's writer use. Rerun's XYZW ordering is converted
exactly once, at the logging boundary.
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
COUNT_SLACK: float = 0.02
"""Largest relative pose-count difference the gate tolerates."""
CSV_HEADER: str = "#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z"
"""Header basalt's own writer emits, and the one :func:`write_trajectory` reproduces."""


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
    """Poses the association matched."""
    n_reference: int
    """Poses in the reference trajectory."""
    n_candidate: int
    """Poses in the candidate trajectory."""
    count_delta: float
    """Relative pose-count difference, ``|n_candidate - n_reference| / n_reference``."""
    alignment: SimilarityTransform
    """The rigid transform that took the candidate positions onto the reference."""

    def summary(self) -> str:
        """The two lines ``golden_compare.py`` prints, in centimetres."""
        return (
            f"poses: reference={self.n_reference} candidate={self.n_candidate} "
            f"associated={self.n_associated} (count delta {self.count_delta:.1%})\n"
            f"ATE  : rmse={self.rmse_m * 100:.2f} cm  max={self.max_m * 100:.2f} cm  median={self.median_m * 100:.2f} cm"
        )


def read_trajectory(path: Path) -> Trajectory:
    """Read a EuRoC/basalt trajectory CSV.

    Comment lines starting with ``#`` are skipped, which covers both the basalt
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
    """Write a trajectory in basalt's CSV form, timestamps as exact integers.

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
    ``property:capture:start_time_ns``; every basalt CSV, including the ``gt.csv``
    sidecars, is on the absolute device clock. On the Index smoke segment the two
    differ by 10,433,867,587,166 ns, so exporting relative timestamps produces a
    file that associates with **nothing**. This is the one place the conversion
    happens, and it happens at the CSV boundary.

    Args:
        trajectory: Poses on the source clock.
        offset_ns: Nanoseconds to add; ``capture.start_time_ns`` to go from ``video_time`` to absolute.

    Returns:
        The same poses on the shifted clock. Positions and rotations are shared, not copied.
    """
    return Trajectory(
        t_ns=trajectory.t_ns + np.int64(offset_ns),
        position_m=trajectory.position_m,
        quaternion_wxyz=trajectory.quaternion_wxyz,
    )


def associate(reference: Trajectory, candidate: Trajectory, tolerance_ns: int = ASSOCIATION_TOLERANCE_NS) -> Association:
    """Match every reference pose to its nearest candidate pose in time.

    Both trajectories must be sorted by timestamp, as basalt's writer emits them.

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
    """Least-squares rigid transform taking ``source`` onto ``target``, scale fixed at 1.

    This is ``golden_compare.py``'s inline arithmetic, ported unchanged, because
    D14 gates on parity with the fork's numbers. In particular there is **no
    variance floor**: a stationary rig, or a trajectory spanning micrometres,
    aligns to itself with zero error rather than raising. The shared
    ``simplecv.ops.umeyama_alignment`` rejects a source variance of 1e-9 or less
    even when it is not estimating a scale, which would turn those runs into
    errors instead of the passes the fork reports. That helper cross-checks this
    one in the tests, on inputs where both are defined.

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


def ate(reference: Trajectory, candidate: Trajectory, tolerance_ns: int = ASSOCIATION_TOLERANCE_NS) -> AteResult:
    """Rigid-aligned absolute trajectory error of ``candidate`` against ``reference``.

    The alignment fixes the scale at 1: a visual-inertial estimator is metric, so
    a fitted scale would hide a real error.

    Whether the result is meaningful is :func:`passes_gate`'s question, not this
    one's — a two-pose comparison returns a number here and fails the gate there,
    exactly as the fork orders it.

    Args:
        reference: Trajectory taken as truth.
        candidate: Trajectory under test.
        tolerance_ns: Association tolerance passed to :func:`associate`.

    Returns:
        The error statistics and the alignment that produced them.

    Raises:
        ValueError: If no pose associates, which leaves nothing to compare.
    """
    association: Association = associate(reference, candidate, tolerance_ns)
    source: Float64[ndarray, "n_associated 3"] = candidate.position_m[association.candidate_index[association.matched]]
    target: Float64[ndarray, "n_associated 3"] = reference.position_m[association.matched]
    if source.shape[0] == 0:
        raise ValueError(f"no pose associated within {tolerance_ns} ns; the two trajectories may be on different clocks")
    alignment: SimilarityTransform = rigid_alignment(source, target)
    errors: Float64[ndarray, " n_associated"] = np.linalg.norm(alignment.apply(source) - target, axis=1)
    return AteResult(
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        max_m=float(errors.max()),
        median_m=float(np.median(errors)),
        n_associated=association.count,
        n_reference=len(reference),
        n_candidate=len(candidate),
        count_delta=abs(len(candidate) - len(reference)) / len(reference),
        alignment=alignment,
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
        ``[0, 1]``; ``0.0`` when the reference has no extent.
    """
    reference_span_ns: int = int(reference.t_ns[-1] - reference.t_ns[0])
    if reference_span_ns <= 0:
        return 0.0
    overlap_ns: int = int(min(reference.t_ns[-1], candidate.t_ns[-1]) - max(reference.t_ns[0], candidate.t_ns[0]))
    return float(np.clip(overlap_ns / reference_span_ns, 0.0, 1.0))


def passes_gate(
    result: AteResult,
    tolerance_m: float,
    count_slack: float = COUNT_SLACK,
    min_associated: int = MIN_ASSOCIATED_POSES,
) -> bool:
    """The basalt fork's PASS criteria, ported verbatim.

    Args:
        result: Statistics from :func:`ate`.
        tolerance_m: Largest ATE RMSE the gate accepts, in metres.
        count_slack: Largest relative pose-count difference the gate accepts.
        min_associated: Fewest associated poses that make the comparison meaningful.

    Returns:
        True when the run passes all three criteria.
    """
    return result.n_associated >= min_associated and result.rmse_m < tolerance_m and result.count_delta <= count_slack
