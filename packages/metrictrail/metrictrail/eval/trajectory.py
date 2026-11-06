"""Pure trajectory math for MetricTrail evaluation: timestamp association and aligned ATE.

Two association rules live here because the two protocols differ:

* :func:`nearest_within` is the paper's rule (LingBot-Map's TUM adapter, which made the
  prepared GT the MetricTrail evaluator reads): every frame takes the GT pose nearest in time,
  and frames with no GT pose within the tolerance (inclusive) have no GT.
* :func:`evo_matching_indices` reproduces ``evo.core.sync.associate_trajectories``, which
  VSLAM-LAB's ``evo_ape tum ... --t_max_diff`` uses: each stamp of the shorter list takes the
  nearest stamp of the longer one within ``max_diff`` (many-to-one allowed), on float64 stamps.

The alignment is ``simplecv.ops.umeyama`` (the same least-squares formula as evo's
``geometry.umeyama_alignment``); the ATE is the RMSE of the translation residuals after it.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment

AlignmentMode: TypeAlias = Literal["sim3", "se3"]
"""``sim3``: rotation, translation and scale (the paper and VSLAM-LAB); ``se3``: no scale (metric-scale diagnostic)."""


@dataclass(frozen=True, slots=True)
class AteResult:
    """Absolute trajectory error after one alignment of the estimate onto the ground truth."""

    rmse_m: float
    """Root mean square of the translation residuals, metres."""
    scale: float
    """Scale of the alignment (1.0 for ``se3``)."""
    num_pairs: int
    """Number of estimate/GT position pairs the alignment and the RMSE used."""
    gt_T_est: Float64[ndarray, "4 4"]
    """The alignment as a homogeneous matrix (scale folded into the rotation block)."""


def nearest_within(query_ns: Int64[ndarray, "q"], reference_ns: Int64[ndarray, "r"], max_dt_ns: int) -> Int64[ndarray, "q"]:
    """Index of the nearest reference stamp for every query stamp, or -1 when it is farther than ``max_dt_ns``.

    Ties go to the earlier reference stamp (``np.argmin`` semantics). Several queries may take the same
    reference stamp. Integer nanoseconds keep the difference exact.

    Args:
        query_ns: Query timestamps, nanoseconds.
        reference_ns: Reference timestamps, nanoseconds, strictly increasing.
        max_dt_ns: Largest accepted absolute difference, nanoseconds (inclusive).

    Returns:
        Reference index per query, -1 where no reference stamp is within ``max_dt_ns``.

    Raises:
        ValueError: If ``reference_ns`` is empty or not strictly increasing.
    """
    if reference_ns.shape[0] == 0:
        raise ValueError("reference timestamps are empty")
    if np.any(np.diff(reference_ns) <= 0):
        raise ValueError("reference timestamps must be strictly increasing")
    right: Int64[ndarray, "q"] = np.clip(np.searchsorted(reference_ns, query_ns, side="left"), 0, reference_ns.shape[0] - 1).astype(np.int64)
    left: Int64[ndarray, "q"] = np.clip(right - 1, 0, reference_ns.shape[0] - 1)
    dt_left: Int64[ndarray, "q"] = np.abs(query_ns - reference_ns[left])
    dt_right: Int64[ndarray, "q"] = np.abs(query_ns - reference_ns[right])
    nearest: Int64[ndarray, "q"] = np.where(dt_left <= dt_right, left, right)
    nearest_dt: Int64[ndarray, "q"] = np.minimum(dt_left, dt_right)
    return np.where(nearest_dt <= max_dt_ns, nearest, -1).astype(np.int64)


@dataclass(frozen=True, slots=True)
class MatchedIndices:
    """Row indices of matched estimate/GT pairs."""

    est: Int64[ndarray, "m"]
    """Estimate row of each pair."""
    gt: Int64[ndarray, "m"]
    """GT row of each pair."""


def evo_matching_indices(gt_stamps: Float64[ndarray, "g"], est_stamps: Float64[ndarray, "e"], max_diff: float) -> MatchedIndices:
    """``evo.core.sync.associate_trajectories(gt, est, max_diff)`` as indices (evo 1.34-1.38).

    The shorter trajectory is walked in order; each of its stamps takes ``argmin |long - stamp|`` and is kept
    when that difference is at most ``max_diff``. With equal lengths the GT is the long one (evo's
    ``snd_longer`` is strict). VSLAM-LAB passes nanosecond stamps and ``max_diff = 1e9 / rgb_hz``.

    Args:
        gt_stamps: GT timestamps in the units of the file evo reads (VSLAM-LAB: nanoseconds as float64).
        est_stamps: Estimate timestamps, same units.
        max_diff: Largest accepted absolute difference, same units (inclusive).

    Returns:
        The matched rows, in the walk order of the shorter list.
    """
    est_longer: bool = est_stamps.shape[0] > gt_stamps.shape[0]
    short: Float64[ndarray, "s"] = gt_stamps if est_longer else est_stamps
    long: Float64[ndarray, "l"] = est_stamps if est_longer else gt_stamps
    short_rows: list[int] = []
    long_rows: list[int] = []
    for short_row, stamp in enumerate(short):
        diffs: Float64[ndarray, "l"] = np.abs(long - stamp)
        long_row: int = int(np.argmin(diffs))
        if diffs[long_row] <= max_diff:
            short_rows.append(short_row)
            long_rows.append(long_row)
    short_index: Int64[ndarray, "m"] = np.asarray(short_rows, dtype=np.int64)
    long_index: Int64[ndarray, "m"] = np.asarray(long_rows, dtype=np.int64)
    if est_longer:
        return MatchedIndices(est=long_index, gt=short_index)
    return MatchedIndices(est=short_index, gt=long_index)


def aligned_ate(est_xyz: Float64[ndarray, "n 3"], gt_xyz: Float64[ndarray, "n 3"], mode: AlignmentMode) -> AteResult:
    """ATE RMSE after one Umeyama alignment of the estimate positions onto the GT positions.

    Args:
        est_xyz: Estimated camera centres, row-aligned with ``gt_xyz``.
        gt_xyz: Ground-truth camera centres, metres.
        mode: ``sim3`` estimates a scale, ``se3`` keeps scale 1.

    Returns:
        The RMSE, the scale, the pair count and the alignment.

    Raises:
        ValueError: If an input is non-finite or the alignment is degenerate (fewer than 3 pairs, no spread).
    """
    if not (np.isfinite(est_xyz).all() and np.isfinite(gt_xyz).all()):
        raise ValueError("aligned_ate needs finite positions; drop invalid pairs first")
    gt_from_est: SimilarityTransform = umeyama_alignment(est_xyz, gt_xyz, allow_scaling=mode == "sim3")
    residual: Float64[ndarray, "n 3"] = gt_xyz - gt_from_est.apply(est_xyz)
    rmse_m: float = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return AteResult(rmse_m=rmse_m, scale=gt_from_est.scale, num_pairs=int(est_xyz.shape[0]), gt_T_est=gt_from_est.as_matrix())


def finite_rows(poses: Float64[ndarray, "n 4 4"]) -> Bool[ndarray, "n"]:
    """True for every pose whose 16 entries are finite."""
    return np.isfinite(poses.reshape(poses.shape[0], 16)).all(axis=1)


def apply_alignment(gt_T_est: Float64[ndarray, "4 4"], est_xyz: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    """Estimate positions moved into the GT frame by an alignment matrix (scale folded into the rotation block)."""
    return est_xyz @ gt_T_est[:3, :3].T + gt_T_est[:3, 3]


def pair_errors_m(result: AteResult, est_xyz: Float64[ndarray, "n 3"], gt_xyz: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n"]:
    """Per-pair translation error after ``result``'s alignment, metres; its RMS is ``result.rmse_m``."""
    return np.linalg.norm(gt_xyz - apply_alignment(result.gt_T_est, est_xyz), axis=1)


def running_rmse(errors: Float64[ndarray, "n"]) -> Float64[ndarray, "n"]:
    """RMSE of the first ``k`` errors, for every ``k`` (the running ATE a viewer plots over time)."""
    return np.sqrt(np.cumsum(errors**2) / np.arange(1, errors.shape[0] + 1))
