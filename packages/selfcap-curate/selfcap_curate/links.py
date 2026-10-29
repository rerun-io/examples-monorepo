"""Rotation-speed correlations, ported from frozen SelfCap v2 common.py.

Lag L means source(t) matches target(t + L). Video clocks are refined against
high-rate gyro samples, never against another video's frame-grid peaks.
"""

import io
from dataclasses import dataclass

import av
import cv2
import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.catalog_video import CatalogVideo
from simplecv.catalog_video_codec import wrap_mp4

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
    scores: list[float] = []
    for lag in lags:
        shifted: Float64[ndarray, "n"] = np.interp(grid + lag, target.times, target.speed)
        scores.append(float(((shifted - shifted.mean()) / (shifted.std() + 1e-9) * normalized).mean()))
    cc: Float64[ndarray, "l"] = np.asarray(scores, dtype=np.float64)
    j: int = int(np.argmax(cc))
    fraction: float = 0.0
    if 0 < j < len(cc) - 1:
        denominator: float = float(cc[j - 1] - 2 * cc[j] + cc[j + 1])
        fraction = float(0.5 * (cc[j - 1] - cc[j + 1]) / denominator) if denominator < 0.0 else 0.0
    best: float = float(lags[j] + fraction / fs)
    clipped: bool = abs(best - center_s) >= half_width_s - 0.005 - 1e-9
    second: float = float(np.max(np.where(np.abs(lags - lags[j]) > 0.1, cc, -1.0)))
    return Lag(None if clipped else round(best, 5), round(float(cc[j]), 3), round(second, 3), round(stop - start, 1), clipped)


def rotation_speed(
    video: CatalogVideo, focal_px: float, width_px: int, start_s: float, stop_s: float, *, changes: list[tuple[float, float]] | None = None
) -> SpeedSeries:
    """Measure rotation between distinct frames in a capture-time window."""
    cv2.setNumThreads(1)
    scale: float = 512.0 / width_px
    times: list[float] = []
    speeds: list[float] = []
    previous: UInt8[ndarray, "h w"] | None = None
    last: UInt8[ndarray, "h w"] | None = None
    previous_time: float = 0.0
    inverse: Float64[ndarray, "3 3"] | None = None
    data: bytes = wrap_mp4(video.samples, video.keyframes, 30, codec=video.codec)
    with av.open(io.BytesIO(data), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            time_s: float = float(video.t_ns[index]) * 1e-9
            if time_s < start_s:
                continue
            if time_s > stop_s:
                break
            image: UInt8[ndarray, "h w"] = frame.to_ndarray(format="gray", width=512, height=int(round(frame.height * scale)))
            if inverse is None:
                matrix: Float64[ndarray, "3 3"] = np.array(
                    [[focal_px * scale, 0.0, 256.0], [0.0, focal_px * scale, image.shape[0] / 2.0], [0.0, 0.0, 1.0]]
                )
                inverse = np.linalg.inv(matrix)
            if last is not None:
                difference: float = float(np.abs(image.astype(np.int16) - last.astype(np.int16)).mean())
                if changes is not None:
                    changes.append((time_s, difference))
                if difference < 0.3:
                    last = image
                    continue
            last = image
            if previous is not None:
                corners = cv2.goodFeaturesToTrack(previous, 120, 0.01, 8)
                if corners is not None and len(corners) >= 12:
                    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, image, corners, corners.copy())
                    keep = status[:, 0] == 1
                    a, b = corners[keep, 0], tracked[keep, 0]
                    if len(a) >= 12:
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
                        times.append(0.5 * (time_s + previous_time))
                        speeds.append(float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)) / max(time_s - previous_time, 1e-3)))
            previous, previous_time = image, time_s
    return SpeedSeries(np.asarray(times, dtype=np.float64), np.asarray(speeds, dtype=np.float64))


def fit_pattern(times: Float64[ndarray, "n"], differences: Float64[ndarray, "n"]) -> RepeatPattern | None:
    """Search native rate ±1500 ppm and phase; still pairs carry no clock evidence."""
    if len(times) < 201:
        return None
    pad = np.pad(np.nan_to_num(differences, nan=0.0), 8, mode="edge")
    local = np.array([np.percentile(pad[i : i + 17], 80) for i in range(len(differences))])
    moving = local > 1.0
    repeats = (differences < 0.2 * local) & moving
    use = repeats | ((differences > 0.5 * local) & moving)
    if not use.any():
        return None
    best: tuple[float, int, float] = (-1.0, 0, 0.0)
    for ppm in range(-1500, 1501, 10):
        base = times * 24.0 * (1.0 + ppm * 1e-6)
        for phase in np.arange(0.0, 1.0, 0.02):
            native = np.floor(base + phase)
            predicted = np.r_[False, np.diff(native) == 0.0]
            agreement = float((predicted == repeats)[use].mean())
            if agreement > best[0]:
                best = agreement, ppm, float(phase)
    agreement, ppm, phase = best
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
