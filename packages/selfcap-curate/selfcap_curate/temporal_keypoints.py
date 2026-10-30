"""Stable body triangulation while retaining per-frame geometric support."""

from itertools import combinations

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.ndimage import median_filter

from selfcap_curate.keypoints import Camera, Keypoints3D, TriangulationBatch, one_euro, support_for_points, triangulate_frames

BODY: slice = slice(3, 17)
BODY_BETA: float = 0.3
MAX_SECOND_DIFFERENCE_M: float = 0.03  # At 24 Hz: reject body acceleration above 17.28 m/s².


def stable_body(result: Keypoints3D, baseline: Float32[ndarray, "n 133 3"]) -> Float32[ndarray, "n 133 3"]:
    """Choose a continuous view-pair path, median-filter it, then respect the 8 px support gate.

    Temporal outliers without a supported smooth estimate are missing, never replaced by a raw jump.
    Hands and face retain the caller's existing treatment.
    """
    cameras: list[Camera] = list(result.cameras.values())
    if len(cameras) < 2:
        return baseline
    uv: Float64[ndarray, "n v 133 2"] = np.stack([view.xy for view in result.selected.values()], axis=1).astype(np.float64)
    scores: Float64[ndarray, "n v 133"] = np.stack([view.scores for view in result.selected.values()], axis=1)
    pairs: list[tuple[int, int]] = list(combinations(range(len(cameras)), 2))
    candidates: Float32[ndarray, "n k j 3"] = np.stack([
        triangulate_frames([cameras[a], cameras[b]], uv[:, [a, b], BODY], scores[:, [a, b], BODY]).xyz
        for a, b in pairs
    ], axis=1)
    finite: Bool[ndarray, "n k j"] = np.isfinite(candidates).all(-1)
    cost: Float64[ndarray, "n k j"] = np.empty(candidates.shape[:-1])
    for pair in range(len(pairs)):
        positions: Float32[ndarray, "n 133 3"] = baseline.copy()
        positions[:, BODY] = candidates[:, pair]
        support: TriangulationBatch = support_for_points(result, positions)
        cost[:, pair] = 4.0 * (len(cameras) - support.n_views[:, BODY]) + 0.1 * np.nan_to_num(support.max_reprojection_px[:, BODY], nan=100.0)
    cost[~finite] = 1e6
    selected: Float32[ndarray, "n 133 3"] = baseline.copy()
    switch_cost: Float64[ndarray, "k k"] = (1.0 - np.eye(len(pairs))) * 2.0
    for joint in range(candidates.shape[2]):
        valid: Bool[ndarray, "n"] = finite[:, :, joint].any(1)
        boundaries: Int64[ndarray, "b"] = np.r_[0, np.flatnonzero(np.diff(valid.astype(int))) + 1, len(valid)]
        selected[:, joint + BODY.start] = np.nan
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if not valid[begin]:
                continue
            accumulated: Float64[ndarray, "k"] = cost[begin, :, joint].copy()
            history: list[Int64[ndarray, "k"]] = []
            for frame in range(begin + 1, end):
                distance: Float32[ndarray, "k k"] = np.linalg.norm(candidates[frame, :, joint, None] - candidates[frame - 1, None, :, joint], axis=-1)
                transition: Float64[ndarray, "k k"] = accumulated[None, :] + np.nan_to_num(distance, nan=100.0) * 300.0 + switch_cost
                previous: Int64[ndarray, "k"] = transition.argmin(1)
                history.append(previous)
                accumulated: Float64[ndarray, "k"] = cost[frame, :, joint] + transition[np.arange(len(pairs)), previous]
            state: int = int(accumulated.argmin())
            for frame in range(end - 1, begin - 1, -1):
                selected[frame, joint + BODY.start] = candidates[frame, state, joint]
                if frame > begin:
                    state: int = int(history[frame - begin - 1][state])
    median: Float32[ndarray, "n j 3"] = selected[:, BODY].copy()
    for joint in range(median.shape[1]):
        valid: Bool[ndarray, "n"] = np.isfinite(median[:, joint]).all(-1)
        boundaries: Int64[ndarray, "b"] = np.r_[0, np.flatnonzero(np.diff(valid.astype(int))) + 1, len(valid)]
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if valid[begin]:
                median[begin:end, joint] = median_filter(median[begin:end, joint], size=(5, 1), mode="nearest")
    desired: Float32[ndarray, "n 133 3"] = selected.copy()
    desired[:, BODY] = (one_euro(median, result.rate, beta=BODY_BETA) + one_euro(median[::-1], result.rate, beta=BODY_BETA)[::-1]) / 2.0
    lower: Float32[ndarray, "n 133"] = np.zeros(selected.shape[:2], dtype=np.float32)
    upper: Float32[ndarray, "n 133"] = np.ones_like(lower)
    for _ in range(12):
        fraction: Float32[ndarray, "n 133"] = (lower + upper) / 2.0
        candidate: Float32[ndarray, "n 133 3"] = selected + (desired - selected) * fraction[..., None]
        supported: Bool[ndarray, "n 133"] = support_for_points(result, candidate).n_views >= 2
        lower: Float32[ndarray, "n 133"] = np.where(supported, fraction, lower)
        upper: Float32[ndarray, "n 133"] = np.where(supported, upper, fraction)
    output: Float32[ndarray, "n 133 3"] = selected + (desired - selected) * lower[..., None]
    # A rare jump can have valid per-frame rays but no temporally plausible supported location.
    acceleration: Float32[ndarray, "n j"] = np.linalg.norm(np.diff(output[:, BODY], n=2, axis=0), axis=-1)
    reject: Bool[ndarray, "n j"] = acceleration > MAX_SECOND_DIFFERENCE_M * (24.0 / result.rate) ** 2
    body: Float32[ndarray, "n j 3"] = output[:, BODY]
    body[1:-1][reject] = np.nan
    output[:, BODY] = body
    output[:, :BODY.start] = baseline[:, :BODY.start]
    output[:, BODY.stop:] = baseline[:, BODY.stop:]
    return output.astype(np.float32)
