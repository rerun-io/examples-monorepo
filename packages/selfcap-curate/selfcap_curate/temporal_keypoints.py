"""Centred body filtering and bounded temporal interpolation."""

import numpy as np
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d, median_filter

from selfcap_curate.keypoints import Keypoints3D

BODY: slice = slice(3, 17)
BODY_SIGMA_FRAMES: float = 2.0
MAX_SECOND_DIFFERENCE_M: float = 0.03


def responsive_body(xyz: Float32[ndarray, "n j 3"], rate: float) -> Float32[ndarray, "n j 3"]:
    """Median-5 jump replacement, centred Gaussian, then follow cleaned fast motion.

    Every operation is time symmetric. Speeds below 0.3 m/s use the Gaussian;
    speeds above 0.5 m/s use the jump-cleaned measurement, never the raw jump.
    Long missing runs are kept missing rather than filtering across them.
    """
    filled, _, _ = bridge_short_gaps(xyz, np.ones(xyz.shape[:2], np.float32), rate)
    output: Float32[ndarray, "n j 3"] = np.full_like(xyz, np.nan)
    for joint in range(xyz.shape[1]):
        valid: Bool[ndarray, "n"] = np.isfinite(filled[:, joint]).all(-1)
        boundaries: Int64[ndarray, "b"] = np.r_[0, np.flatnonzero(np.diff(valid.astype(int))) + 1, len(valid)]
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if not valid[begin]:
                continue
            source: Float32[ndarray, "m 1 3"] = filled[begin:end, joint:joint + 1].copy()
            median: Float32[ndarray, "m 1 3"] = median_filter(source, size=(5, 1, 1), mode="nearest")
            acceleration: Float32[ndarray, "m 1"] = np.zeros(source.shape[:2], np.float32)
            acceleration[1:-1] = np.linalg.norm(np.diff(source, n=2, axis=0), axis=-1)
            jumps: Bool[ndarray, "m 1"] = (acceleration > MAX_SECOND_DIFFERENCE_M * (24.0 / rate) ** 2) & (np.linalg.norm(source - median, axis=-1) > 0.01)
            clean: Float32[ndarray, "m 1 3"] = source.copy()
            clean[jumps] = np.nan
            clean, _, _ = bridge_short_gaps(clean, np.ones(clean.shape[:2], np.float32), rate)
            clean_valid: Bool[ndarray, "m"] = np.isfinite(clean[:, 0]).all(-1)
            # A sustained run of rejected measurements still has a measured trajectory
            # at both ends. Use that trajectory only for the centred filtering pass.
            for axis in range(3):
                good: Int64[ndarray, "g"] = np.flatnonzero(clean_valid)
                if len(good):
                    clean[:, 0, axis] = np.interp(np.arange(len(clean)), good, clean[good, 0, axis])
            smooth: Float32[ndarray, "m 1 3"] = gaussian_filter1d(clean, BODY_SIGMA_FRAMES, axis=0, mode="reflect", truncate=3.0)
            speed: Float32[ndarray, "m 1"] = np.linalg.norm(np.gradient(median, axis=0) * rate, axis=-1) if len(source) > 1 else np.zeros(source.shape[:2], np.float32)
            weight: Float32[ndarray, "m 1 1"] = np.clip((speed - 0.3) / 0.2, 0.0, 1.0)[..., None]
            output[begin:end, joint:joint + 1] = (1.0 - weight) * smooth + weight * clean
    return output


def stable_body(result: Keypoints3D, baseline: Float32[ndarray, "n 133 3"]) -> Float32[ndarray, "n 133 3"]:
    """Filter per-frame triangulated body positions; preserve the caller's hand treatment."""
    output: Float32[ndarray, "n 133 3"] = baseline.copy()
    output[:, BODY] = responsive_body(result.xyz[:, BODY], result.rate)
    return output


def bridge_short_gaps(
    positions: Float32[ndarray, "n j 3"],
    confidence: Float32[ndarray, "n j"],
    rate: float,
) -> tuple[Float32[ndarray, "n j 3"], Float32[ndarray, "n j"], Bool[ndarray, "n j"]]:
    """Interpolate bounded gaps up to 0.5 seconds; inferred samples carry 75% endpoint confidence."""
    output: Float32[ndarray, "n j 3"] = positions.copy()
    scores: Float32[ndarray, "n j"] = confidence.copy()
    bridged: Bool[ndarray, "n j"] = np.zeros(positions.shape[:2], dtype=bool)
    for joint in range(positions.shape[1]):
        valid: Int64[ndarray, "v"] = np.flatnonzero(np.isfinite(positions[:, joint]).all(-1))
        for left, right in zip(valid[:-1], valid[1:], strict=True):
            if right - left <= 1 or (right - left - 1) / rate > 0.5:
                continue
            fraction: Float32[ndarray, "g"] = np.arange(1, right - left, dtype=np.float32) / (right - left)
            output[left + 1:right, joint] = positions[left, joint] + fraction[:, None] * (positions[right, joint] - positions[left, joint])
            scores[left + 1:right, joint] = 0.75 * min(confidence[left, joint], confidence[right, joint])
            bridged[left + 1:right, joint] = True
    return output, scores, bridged
