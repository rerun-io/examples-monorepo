"""Compare five catalog rotation-speed signals on the cut and aligned clocks."""

from dataclasses import dataclass, field

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import CatalogClient
from simplecv.rerun_log_utils import RerunTyroConfig

from selfcap_curate.catalog_io import CATALOG_URL, DATASET_NAME, EpisodeInputs, open_segment, read_episode
from selfcap_curate.links import SpeedSeries, correlate, rotation_speed
from selfcap_curate.records import Lag

NAMES: tuple[str, ...] = ("OAK left video", "OAK rgb video", "Quest right video", "OAK gyro", "Quest head pose")
COLORS: tuple[tuple[int, int, int], ...] = ((230, 230, 230), (240, 80, 80), (80, 170, 255), (250, 200, 60), (90, 220, 120))


@dataclass
class Config:
    """One named catalog segment and its evidence window."""

    segment_id: str
    """Full SelfCap segment id; no corpus enumeration."""
    catalog_url: str = CATALOG_URL
    """Read-only catalog endpoint."""
    window_seconds: float = 12.0
    """Duration of the highest-variation head-motion window (at least 11 s)."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Shared viewer, headless, connection and save options."""


@dataclass(frozen=True, slots=True)
class Curve:
    """Windowed z-score signal and residual correction toward the gyro clock."""

    series: SpeedSeries
    """Windowed rotation speed."""
    normalized: Float64[ndarray, "n"]
    """Per-window zero-mean, unit-variance speed (zero for constant signals)."""
    lag_ms: float | None
    """Milliseconds to add to source time; None for unmeasurable or clipped peaks."""


def measure_curves(series: tuple[SpeedSeries, ...], start_s: float, stop_s: float, *, gyro_index: int = 3) -> tuple[Curve, ...]:
    """Measure each window's lag at 1 kHz within ±0.25 s against its gyro."""
    windowed: list[SpeedSeries] = []
    for signal in series:
        keep = (signal.times >= start_s) & (signal.times <= stop_s)
        windowed.append(SpeedSeries(signal.times[keep], signal.speed[keep]))
    curves: list[Curve] = []
    for signal in windowed:
        if len(signal.times) < 2:
            raise ValueError("selected window has fewer than two speed samples")
        lag: Lag = correlate(signal, windowed[gyro_index], half_width_s=0.25, fs=1000.0)
        deviation: float = float(signal.speed.std())
        normalized: Float64[ndarray, "n"] = (signal.speed - signal.speed.mean()) / (deviation if deviation > 0.0 else 1.0)
        curves.append(Curve(signal, normalized, None if lag.lag_s is None else lag.lag_s * 1000.0))
    return tuple(curves)


def gyro_zoom_center(gyro: SpeedSeries, start_s: float, stop_s: float) -> float:
    """Pick the local gyro maximum with the greatest contrast over ±50 ms.

    Use only gyro samples, leaving 0.5 s on each side for the complete zoom.
    This favors sharp peaks over broad humps and is independent of video timing.
    """
    peaks = np.flatnonzero((gyro.speed[1:-1] > gyro.speed[:-2]) & (gyro.speed[1:-1] >= gyro.speed[2:])) + 1
    peaks = peaks[(gyro.times[peaks] >= start_s + 0.5) & (gyro.times[peaks] <= stop_s - 0.5)]
    if not len(peaks):
        raise ValueError("selected window contains no interior gyro peak for the 1 s zoom")
    times: Float64[ndarray, "n"] = gyro.times[peaks]
    contrast: Float64[ndarray, "n"] = gyro.speed[peaks] - 0.5 * (
        np.interp(times - 0.05, gyro.times, gyro.speed) + np.interp(times + 0.05, gyro.times, gyro.speed)
    )
    return float(times[int(np.argmax(contrast))])


def build_blueprint(segment_id: str, zoom_center_s: float) -> rrb.Blueprint:
    """Lay out two overview plots and a fixed one-second aligned zoom (40% height)."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(name=f"AS CUT · capture_time · {segment_id}", origin="as_cut"),
            rrb.TimeSeriesView(name="ALIGNED · video_time · coinciding peaks show alignment", origin="aligned"),
            rrb.TimeSeriesView(
                name="ALIGNED · 1 s zoom",
                origin="aligned",
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "t",
                        start=rrb.TimeRangeBoundary.absolute(seconds=zoom_center_s - 0.5),
                        end=rrb.TimeRangeBoundary.absolute(seconds=zoom_center_s + 0.5),
                    )
                ],
            ),
            row_shares=[0.3, 0.3, 0.4],
        ),
        collapse_panels=True,
    )


def main(config: Config) -> dict[str, tuple[Curve, ...]]:
    """Write overview and zoom views and return their measured residuals for reports."""
    if not np.isfinite(config.window_seconds) or config.window_seconds < 11.0:
        raise ValueError("window_seconds must be finite and at least 11 s for correlation")
    dataset = CatalogClient(config.catalog_url).get_dataset(DATASET_NAME)
    segment = open_segment(dataset, config.segment_id)
    aligned: EpisodeInputs = read_episode(segment, timeline="video_time")
    cut: EpisodeInputs = read_episode(segment)
    signals: tuple[SpeedSeries, ...] = (aligned.gyro, aligned.pose, cut.gyro, cut.pose)
    start: float = max(float(signal.times[0]) for signal in signals)
    stop: float = min(float(signal.times[-1]) for signal in signals)
    for inputs in (aligned, cut):
        start = max(start, *(float(video.t_ns[0]) * 1e-9 + 0.1 for video in inputs.videos))
        stop = min(stop, *(float(video.t_ns[-1]) * 1e-9 - 0.1 for video in inputs.videos))
    candidates: Float64[ndarray, "n"] = np.arange(start, stop - config.window_seconds, 0.5)
    if not len(candidates):
        raise ValueError("segment has insufficient common duration for the requested window")
    # The same numerical interval is used on both clocks. Pick vigorous head motion
    # from the high-rate gyro without decoding entire videos just to choose a window.
    scores: list[float] = [
        float(np.interp(np.arange(t, t + config.window_seconds, 0.02), aligned.gyro.times, aligned.gyro.speed).std()) for t in candidates
    ]
    start = float(candidates[int(np.argmax(scores))])
    stop = start + config.window_seconds
    result: dict[str, tuple[Curve, ...]] = {}
    for clock, inputs in (("as_cut", cut), ("aligned", aligned)):
        videos = tuple(
            rotation_speed(video, focal, width, start - 0.5, stop + 0.5)
            for video, focal, width in zip(inputs.videos, inputs.focal_px, inputs.width_px, strict=True)
        )
        curves: tuple[Curve, ...] = measure_curves((*videos, inputs.gyro, inputs.pose), start, stop)
        result[clock] = curves
        for name, color, curve in zip(NAMES, COLORS, curves, strict=True):
            label: str = "unmeasurable vs gyro" if curve.lag_ms is None else f"{curve.lag_ms:+.1f} ms vs gyro"
            entity: str = f"{clock}/{name.lower().replace(' ', '_')}"
            rr.log(entity, rr.SeriesLines(colors=color, names=f"{name} ({label})", widths=2.0), static=True)
            rr.send_columns(entity, indexes=[rr.TimeColumn("t", duration=curve.series.times)], columns=rr.Scalars.columns(scalars=curve.normalized))
            print(f"{clock}: {name}: {label}", flush=True)
    zoom_center: float = gyro_zoom_center(aligned.gyro, start, stop)
    rr.send_blueprint(build_blueprint(config.segment_id, zoom_center))
    print(f"zoom centre: {zoom_center:.9f} s (video_time)", flush=True)
    config.rr_config.rec_stream.flush()
    print(f"window: {start:.3f}–{stop:.3f} s", flush=True)
    return result
