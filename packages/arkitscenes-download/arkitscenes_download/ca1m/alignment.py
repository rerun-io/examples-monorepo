"""Clock diagnostics and FARO-to-ARKit rigid alignment for CA-1M."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float64, Int64


@dataclass(frozen=True, slots=True)
class RigidAlignment:
    """Proper rigid transform mapping source points into target coordinates."""

    rotation_33: Float64[np.ndarray, "3 3"]
    """Target-from-source rotation matrix."""
    translation_xyz: Float64[np.ndarray, "3"]
    """Target-frame translation in metres."""
    rms_m: float
    """Root-mean-square correspondence residual in metres."""
    source_extent2_m: float
    """Second principal extent of the source trajectory in metres.

    Rank-1 (collinear) trajectories leave rotation about the motion axis
    unobservable while still reporting a perfect RMS, so a small second
    extent means the fitted orientation cannot be trusted.
    """


@dataclass(frozen=True, slots=True)
class ClockDiagnostics:
    """Nearest-calibration timestamp matching statistics."""

    nearest_indices: Int64[np.ndarray, "n"]
    """Nearest calibration sample index for each GT timestamp."""
    deltas_s: Float64[np.ndarray, "n"]
    """Absolute timestamp differences in seconds."""
    median_delta_s: float
    """Median absolute timestamp difference in seconds."""
    max_delta_s: float
    """Maximum absolute timestamp difference in seconds."""
    fraction_over_10ms: float
    """Fraction of matches farther than ten milliseconds."""
    should_warn: bool
    """Whether the measured clock agreement violates the warning policy."""


def rigid_umeyama(
    source_xyz: Float64[np.ndarray, "n 3"], target_xyz: Float64[np.ndarray, "n 3"]
) -> RigidAlignment:
    """Fit a no-scale proper rigid transform from source to target.

    Args:
        source_xyz: Float64 source points with shape ``(n, 3)``.
        target_xyz: Float64 corresponding target points with shape ``(n, 3)``.

    Returns:
        The proper target-from-source rotation, translation, and RMS residual.
    """
    num_points: int = source_xyz.shape[0]
    if num_points < 3:
        raise ValueError(f"at least three correspondences are required, got {num_points}")

    source_mean_xyz: Float64[np.ndarray, "3"] = np.mean(source_xyz, axis=0)
    target_mean_xyz: Float64[np.ndarray, "3"] = np.mean(target_xyz, axis=0)
    source_centered_xyz: Float64[np.ndarray, "n 3"] = source_xyz - source_mean_xyz
    target_centered_xyz: Float64[np.ndarray, "n 3"] = target_xyz - target_mean_xyz
    covariance_33: Float64[np.ndarray, "3 3"] = source_centered_xyz.T @ target_centered_xyz
    svd_result: tuple[np.ndarray, np.ndarray, np.ndarray] = np.linalg.svd(covariance_33)
    left_33: Float64[np.ndarray, "3 3"] = svd_result[0]
    right_transposed_33: Float64[np.ndarray, "3 3"] = svd_result[2]
    reflection_fix_33: Float64[np.ndarray, "3 3"] = np.eye(3, dtype=np.float64)
    if np.linalg.det(right_transposed_33.T @ left_33.T) < 0.0:
        reflection_fix_33[2, 2] = -1.0
    rotation_33: Float64[np.ndarray, "3 3"] = right_transposed_33.T @ reflection_fix_33 @ left_33.T
    translation_xyz: Float64[np.ndarray, "3"] = target_mean_xyz - rotation_33 @ source_mean_xyz
    fitted_xyz: Float64[np.ndarray, "n 3"] = (rotation_33 @ source_xyz.T).T + translation_xyz
    residual_xyz: Float64[np.ndarray, "n 3"] = target_xyz - fitted_xyz
    rms_m: float = float(np.sqrt(np.mean(np.sum(residual_xyz * residual_xyz, axis=1))))
    source_singular_values: Float64[np.ndarray, "3"] = np.linalg.svd(source_centered_xyz, compute_uv=False)
    source_extent2_m: float = float(source_singular_values[1] / np.sqrt(num_points))
    return RigidAlignment(rotation_33, translation_xyz, rms_m, source_extent2_m)


def diagnose_clock(
    gt_times_s: Float64[np.ndarray, "n"], calibration_times_s: Float64[np.ndarray, "m"]
) -> ClockDiagnostics:
    """Match GT times to the nearest calibration samples and warn on drift.

    Args:
        gt_times_s: Float64 capture-relative GT times with shape ``(n,)``.
        calibration_times_s: Float64 sorted calibration times with shape ``(m,)``.

    Returns:
        Nearest indices and clock-agreement statistics.
    """
    insertion_indices: Int64[np.ndarray, "n"] = np.searchsorted(calibration_times_s, gt_times_s).astype(np.int64)
    left_indices: Int64[np.ndarray, "n"] = np.clip(insertion_indices - 1, 0, len(calibration_times_s) - 1)
    right_indices: Int64[np.ndarray, "n"] = np.clip(insertion_indices, 0, len(calibration_times_s) - 1)
    left_deltas_s: Float64[np.ndarray, "n"] = np.abs(gt_times_s - calibration_times_s[left_indices])
    right_deltas_s: Float64[np.ndarray, "n"] = np.abs(gt_times_s - calibration_times_s[right_indices])
    choose_right: Bool[np.ndarray, "n"] = right_deltas_s < left_deltas_s
    nearest_indices: Int64[np.ndarray, "n"] = np.where(choose_right, right_indices, left_indices).astype(np.int64)
    deltas_s: Float64[np.ndarray, "n"] = np.abs(gt_times_s - calibration_times_s[nearest_indices])
    median_delta_s: float = float(np.median(deltas_s))
    max_delta_s: float = float(np.max(deltas_s))
    fraction_over_10ms: float = float(np.mean(deltas_s > 0.010))
    # Diagnostics only — the capture-aware warning is printed at the ingest
    # boundary (ingest_capture), which knows the video id.
    should_warn: bool = median_delta_s > 0.002 or fraction_over_10ms > 0.05
    return ClockDiagnostics(
        nearest_indices,
        deltas_s,
        median_delta_s,
        max_delta_s,
        fraction_over_10ms,
        should_warn,
    )
