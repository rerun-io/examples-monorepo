"""Coarse links followed by gyro-referenced video clock refinement."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from selfcap_curate.catalog_io import EpisodeInputs
from selfcap_curate.links import SpeedSeries, correlate, fit_pattern, rotation_speed
from selfcap_curate.records import Lag, Links

SEARCH_HALF_WIDTH_S: dict[str, float] = {
    "left_to_pose": 8.0,
    "rgb_to_left": 0.4,
    "quest_video_to_pose": 1.0,
    "left_to_quest_video": 8.0,
    "left_to_gyro": 8.0,
    "gyro_to_pose": 8.0,
}


def measure_links(
    inputs: EpisodeInputs,
    *,
    window_s: tuple[float, float],
    search_half_width_s: Mapping[str, float] = SEARCH_HALF_WIDTH_S,
) -> Links:
    """Measure a central 50 s window and refine all three videos at 1 kHz.

    Explicit windows and per-link search widths permit comparison with frozen
    scratch measurements. Neither option adjusts the catalog timestamps.
    Coarse video-to-video results seed refinement only. Search widths retain
    v2's narrow RGB and Quest-to-pose seeds; device-to-device searches use ±8 s.
    """
    changes: list[list[tuple[float, float]]] = [[], [], []]
    # One video decodes on the GPU while another's frame pairs are tracked on the CPU.
    with ThreadPoolExecutor(len(inputs.videos)) as videos:
        series: list[SpeedSeries] = list(
            videos.map(
                lambda video, focal, width, change: rotation_speed(video, focal, width, *window_s, changes=change),
                inputs.videos,
                inputs.focal_px,
                inputs.width_px,
                changes,
            )
        )
    left, rgb, quest = series
    lp: Lag = correlate(left, inputs.pose, half_width_s=search_half_width_s["left_to_pose"])
    rl: Lag = correlate(rgb, left, half_width_s=search_half_width_s["rgb_to_left"])
    qp: Lag = correlate(quest, inputs.pose, half_width_s=search_half_width_s["quest_video_to_pose"])
    lq: Lag = correlate(left, quest, half_width_s=search_half_width_s["left_to_quest_video"])
    lg: Lag = correlate(left, inputs.gyro, half_width_s=search_half_width_s["left_to_gyro"])
    gp: Lag = correlate(inputs.gyro, inputs.pose, half_width_s=search_half_width_s["gyro_to_pose"])
    fine_left: Lag = Lag(None, None, None, 0.0)
    fine_rgb: Lag = fine_left
    fine_quest: Lag = fine_left
    if lg.lag_s is not None:
        fine_left = correlate(left, inputs.gyro, half_width_s=0.25, center_s=lg.lag_s, fs=1000.0)
        fine_rgb = correlate(rgb, inputs.gyro, half_width_s=0.25, center_s=lg.lag_s + (rl.lag_s or 0.0), fs=1000.0)
    # Pose chain supplies only the coarse search center, never the reported fine lag.
    if qp.lag_s is not None and gp.lag_s is not None:
        fine_quest = correlate(quest, inputs.gyro, half_width_s=0.25, center_s=qp.lag_s - gp.lag_s, fs=1000.0)
    elif lq.lag_s is not None and lg.lag_s is not None:
        fine_quest = correlate(quest, inputs.gyro, half_width_s=0.25, center_s=lg.lag_s - lq.lag_s, fs=1000.0)
    patterns = [
        fit_pattern(np.asarray(change, dtype=np.float64)[:, 0], np.asarray(change, dtype=np.float64)[:, 1]) if change else None
        for change in changes[:2]
    ]
    return Links(lp, rl, qp, lg, gp, patterns[0], patterns[1], lq, fine_left, fine_rgb, fine_quest)
