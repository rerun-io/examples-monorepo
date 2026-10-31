"""Rotation-speed correlations, ported from frozen SelfCap v2 common.py.

Lag L means source(t) matches target(t + L). Video clocks are refined against
high-rate gyro samples, never against another video's frame-grid peaks.
"""

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from simplecv.catalog_video import CatalogVideo
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor

from selfcap_curate.records import Lag, RepeatPattern


@dataclass(frozen=True, slots=True)
class SpeedSeries:
    """Rotation speed and its clock in seconds."""

    times: Float64[ndarray, "n"]
    """Strictly increasing sample times in seconds."""
    speed: Float64[ndarray, "n"]
    """Angular speed in radians per second."""

    def __post_init__(self) -> None:
        if len(self.times) != len(self.speed) or not np.all(np.diff(self.times) > 0):
            raise ValueError("speed samples need matching, strictly increasing times")
        if not np.isfinite(self.times).all() or not np.isfinite(self.speed).all():
            raise ValueError("speed samples must be finite")


def correlate(source: SpeedSeries, target: SpeedSeries, *, half_width_s: float = 8.0, center_s: float = 0.0, fs: float = 200.0) -> Lag:
    """Find the normalized correlation peak with parabolic sub-sample refinement.

    Peaks within 5 ms of either search edge are missing measurements. Nulls
    replace the scratch code's NaN so machine-written JSON is portable.
    """
    if half_width_s <= 0.0 or fs <= 0.0:
        raise ValueError("search width and sample rate must be positive")
    lags: Float64[ndarray, "l"] = np.arange(center_s - half_width_s, center_s + half_width_s + 1e-9, 1.0 / fs)
    if len(source.times) < 2 or len(target.times) < 2:
        return Lag(None, None, None, 0.0)
    start: float = float(max(source.times[0], target.times[0] - lags.min()))
    stop: float = float(min(source.times[-1], target.times[-1] - lags.max()))
    if stop - start < 10.0:
        return Lag(None, None, None, round(stop - start, 1))
    grid: Float64[ndarray, "n"] = np.arange(start, stop, 1.0 / fs)
    sampled: Float64[ndarray, "n"] = np.interp(grid, source.times, source.speed)
    normalized: Float64[ndarray, "n"] = (sampled - sampled.mean()) / (sampled.std() + 1e-9)
    # grid[i] + lags[j] = grid[0] + lags[0] + (i + j) / fs: one interpolation, every shift a window of it.
    values: Float64[ndarray, "m"] = np.interp(grid[0] + lags[0] + np.arange(len(grid) + len(lags) - 1) / fs, target.times, target.speed)
    n: int = len(grid)
    sums: Float64[ndarray, "m1"] = np.concatenate([[0.0], np.cumsum(values)])
    squares: Float64[ndarray, "m1"] = np.concatenate([[0.0], np.cumsum(values * values)])
    mean: Float64[ndarray, "l"] = (sums[n:] - sums[:-n]) / n
    std: Float64[ndarray, "l"] = np.sqrt(np.maximum((squares[n:] - squares[:-n]) / n - mean * mean, 0.0))
    cc: Float64[ndarray, "l"] = (np.correlate(values, normalized, mode="valid") - mean * normalized.sum()) / ((std + 1e-9) * n)
    j: int = int(np.argmax(cc))
    fraction: float = 0.0
    if 0 < j < len(cc) - 1:
        denominator: float = float(cc[j - 1] - 2 * cc[j] + cc[j + 1])
        fraction = float(0.5 * (cc[j - 1] - cc[j + 1]) / denominator) if denominator < 0.0 else 0.0
    best: float = float(lags[j] + fraction / fs)
    clipped: bool = abs(best - center_s) >= half_width_s - 0.005 - 1e-9
    second: float = float(np.max(np.where(np.abs(lags - lags[j]) > 0.1, cc, -1.0)))
    return Lag(None if clipped else round(best, 5), round(float(cc[j]), 3), round(second, 3), round(stop - start, 1), clipped)


def frame_rotation(previous: UInt8[ndarray, "h w"], image: UInt8[ndarray, "h w"], inverse: Float64[ndarray, "3 3"]) -> float | None:
    """Rotation angle between two frames: tracked corners, then a twice-trimmed best rotation."""
    corners = cv2.goodFeaturesToTrack(previous, 120, 0.01, 8)
    if corners is None or len(corners) < 12:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, image, corners, corners.copy())
    keep = status[:, 0] == 1
    a, b = corners[keep, 0], tracked[keep, 0]
    if len(a) < 12:
        return None
    source: Float64[ndarray, "k 3"] = (inverse @ np.c_[a, np.ones(len(a))].T).T
    target: Float64[ndarray, "k 3"] = (inverse @ np.c_[b, np.ones(len(b))].T).T
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    rotation: Float64[ndarray, "3 3"] = np.eye(3)
    for _ in range(2):
        u, _, vt = np.linalg.svd(target.T @ source)
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        residual = np.linalg.norm(target - source @ rotation.T, axis=1)
        keep = residual <= np.quantile(residual, 0.66)
        source, target = source[keep], target[keep]
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


@cache
def tracking_pool() -> ThreadPoolExecutor:
    """Frame pairs are independent; OpenCV releases the GIL. BLAS must run single-threaded (OMP_NUM_THREADS=1)."""
    return ThreadPoolExecutor(max(4, (os.cpu_count() or 8) - 4))


GRAY_BATCH: int = 16
"""Frames per float-RGB luma batch (~0.4 GB per 1080p video); links decodes three videos at once in every session sharing the GPU."""


def gray_frames(video: CatalogVideo, first: int, last: int, scale: float) -> Iterator[UInt8[Tensor, "b h w"]]:
    """CUDA-decode samples first..last from the keyframe at or before first; BT.601 luma, resized by scale, one batch at a time."""
    from torchcodec.decoders import VideoDecoder

    key: int = max(i for i in range(first + 1) if video.keyframes[i])
    decoder = VideoDecoder(
        wrap_mp4(video.samples[key : last + 1], video.keyframes[key : last + 1], 30, codec=video.codec), device="cuda", seek_mode="exact"
    )
    luma: Float32[Tensor, "1 3 1 1"] = torch.tensor([0.299, 0.587, 0.114], device="cuda").view(1, 3, 1, 1)
    for begin in range(first - key, last - key + 1, GRAY_BATCH):
        rgb: UInt8[Tensor, "b 3 h w"] = decoder.get_frames_in_range(begin, min(begin + GRAY_BATCH, last - key + 1)).data
        size: tuple[int, int] = (int(round(rgb.shape[2] * scale)), int(round(rgb.shape[3] * scale)))
        # Antialiased bilinear is the closest GPU match to swscale's default; the tracker reacts to the filter by a few ms.
        gray = F.interpolate((rgb.float() * luma).sum(1, keepdim=True), size=size, mode="bilinear", align_corners=False, antialias=True)
        yield gray.round_().clamp_(0, 255).to(torch.uint8)[:, 0]


def rotation_speed(
    video: CatalogVideo, focal_px: float, width_px: int, start_s: float, stop_s: float, *, changes: list[tuple[float, float]] | None = None
) -> SpeedSeries:
    """Measure rotation between distinct frames in a capture-time window.

    Decode, gray and frame differences run on the GPU; corner tracking runs per distinct frame pair on a thread pool.
    """

    cv2.setNumThreads(1)
    t: Float64[ndarray, "f"] = video.t_ns * 1e-9
    inside: Int64[ndarray, "w"] = np.flatnonzero((t >= start_s) & (t <= stop_s))
    if not len(inside):
        return SpeedSeries(np.empty(0), np.empty(0))
    first, last = int(inside[0]), int(inside[-1])
    scale: float = 512.0 / width_px
    # Differences and the CPU copy go batch by batch (carrying the previous frame), so the window never sits whole on the GPU.
    batches: list[UInt8[ndarray, "b h w"]] = []
    steps: list[Tensor] = []
    previous: UInt8[Tensor, "p h w"] | None = None
    for gray in gray_frames(video, first, last, scale):
        stack: UInt8[Tensor, "s h w"] = gray if previous is None else torch.cat([previous, gray])
        steps.append((stack[1:].to(torch.int16) - stack[:-1].to(torch.int16)).abs().float().mean((1, 2)))
        previous = gray[-1:]
        batches.append(gray.cpu().numpy())
    differences: Float64[ndarray, "m"] = torch.cat(steps).double().cpu().numpy()
    frames: UInt8[ndarray, "n h w"] = np.concatenate(batches)
    height: int = frames.shape[1]
    times: Float64[ndarray, "n"] = t[first : last + 1]
    if changes is not None:
        changes.extend(zip(times[1:].tolist(), differences.tolist(), strict=True))
    kept: list[int] = [0, *(int(i) + 1 for i in np.flatnonzero(differences >= 0.3))]
    inverse: Float64[ndarray, "3 3"] = np.linalg.inv(
        np.array([[focal_px * scale, 0.0, 256.0], [0.0, focal_px * scale, height / 2.0], [0.0, 0.0, 1.0]])
    )
    pairs: list[tuple[int, int]] = list(zip(kept[:-1], kept[1:], strict=True))
    angles = list(tracking_pool().map(lambda pair: frame_rotation(frames[pair[0]], frames[pair[1]], inverse), pairs))
    samples = [
        (0.5 * (times[j] + times[i]), angle / max(times[j] - times[i], 1e-3))
        for (i, j), angle in zip(pairs, angles, strict=True)
        if angle is not None
    ]
    return SpeedSeries(np.asarray([s[0] for s in samples], dtype=np.float64), np.asarray([s[1] for s in samples], dtype=np.float64))


def fit_pattern(times: Float64[ndarray, "n"], differences: Float64[ndarray, "n"]) -> RepeatPattern | None:
    """Search native rate ±1500 ppm and phase; still pairs carry no clock evidence."""
    if len(times) < 201:
        return None
    pad = np.pad(np.nan_to_num(differences, nan=0.0), 8, mode="edge")
    local = np.percentile(sliding_window_view(pad, 17)[: len(differences)], 80, axis=1)
    moving = local > 1.0
    repeats = (differences < 0.2 * local) & moving
    use = repeats | ((differences > 0.5 * local) & moving)
    if not use.any():
        return None
    ppms: Int64[ndarray, "r"] = np.arange(-1500, 1501, 10)
    phases: Float64[ndarray, "p"] = np.arange(0.0, 1.0, 0.02)
    scores: Float64[ndarray, "r p"] = np.empty((len(ppms), len(phases)))
    for row, ppm in enumerate(ppms):
        native = np.floor(times[None, :] * 24.0 * (1.0 + int(ppm) * 1e-6) + phases[:, None])
        predicted = np.concatenate([np.zeros((len(phases), 1), dtype=bool), np.diff(native, axis=1) == 0.0], axis=1)
        scores[row] = (predicted[:, use] == repeats[use]).mean(axis=1)
    row, column = divmod(int(np.argmax(scores)), len(phases))  # first best in (ppm, phase) order, as the scan kept it
    agreement, ppm, phase = float(scores[row, column]), int(ppms[row]), float(phases[column])
    native = np.floor(times * 24.0 * (1.0 + ppm * 1e-6) + phase)
    return RepeatPattern(
        ppm,
        round(phase, 2),
        round(agreement, 4),
        int(use.sum()),
        round(float(moving.mean()), 3),
        round(float((np.diff(native) == 0.0).mean()), 3),
        int(native[-1] - native[0] + 1),
    )


def pose_speed(times: Float64[ndarray, "p"], quaternions: Float64[ndarray, "p 4"]) -> SpeedSeries:
    """Preserve v2 angular speeds from quantized, non-normalized source quaternions."""
    # Preserve v2's quaternion-to-matrix formula. Normalizing here changes
    # the angular-speed signal of quantized, slightly non-unit source poses.
    x, y, z, w = quaternions.T
    rotations: Float64[ndarray, "p 3 3"] = np.stack(
        [
            np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], -1),
            np.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], -1),
            np.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], -1),
        ],
        1,
    )
    relative: Float64[ndarray, "r 3 3"] = np.einsum("nji,njk->nik", rotations[:-1], rotations[1:])
    speed: Float64[ndarray, "r"] = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)) / np.maximum(
        np.diff(times), 1e-3
    )
    return SpeedSeries(0.5 * (times[1:] + times[:-1]), speed)
