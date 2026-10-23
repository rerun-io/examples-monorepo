import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb

from dagkit.app import run_app, with_pipeline
from dagkit.graph import AsOf, Block, DropOldest, Graph
from dagkit.node import Emit, Event, Mode, RunContext, Tick
from dagkit.runners import RunReport
from dagkit.sinks import RerunSink
from dagkit.sources import IterSource
from dagkit.time import TimeNs
from dagkit.types import Log, row_times


@dataclass(frozen=True, slots=True)
class Sample:
    """One point of the helix and its signal value."""

    xyz: tuple[float, float, float]
    value: float

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(f"{root}/point", rr.Points3D([self.xyz], radii=0.03)), Log(f"/signal{root}", rr.Scalars(self.value))]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/point", f"/signal{root}"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Sample"]]:
        points: list = table[f"{root}/point:Points3D:positions"].to_pylist()
        scalars: list = table[f"/signal{root}:Scalars:scalars"].to_pylist()
        return ((t_ns, cls(tuple(xyz[0]), value[0])) for t_ns, xyz, value in zip(row_times(table, timeline), points, scalars, strict=True))


class Smooth:
    """Smooth the signal with a gain.

    Moves its state a ``gain`` fraction towards each raw sample; ``delay_s`` simulates a slow step.
    """

    inputs, outputs = {"raw": Sample, "gain": float}, {"smooth": Sample}

    def __init__(self, delay_s: float) -> None:
        self.delay_s: float = delay_s

    def reset(self, ctx: RunContext) -> None:
        self.state: Sample | None = None

    def step(self, tick: Tick, out: Emit) -> None:
        time.sleep(self.delay_s)
        raw, gain = tick.value, tick.asof["gain"]
        assert isinstance(raw, Sample) and isinstance(gain, float)
        prev: Sample = self.state or raw
        x, y = (p + gain * (r - p) for p, r in zip(prev.xyz[:2], raw.xyz[:2], strict=True))
        self.state = Sample((x, y, raw.xyz[2] + 0.3), prev.value + gain * (raw.value - prev.value))
        out.put("smooth", tick.t_ns, self.state)

    def flush(self, out: Emit) -> None:
        pass


@dataclass
class DemoRerunConfig:
    connect: str | None = None
    save: Path | None = None
    headless: bool = False  # no native viewer (the generated UI renders into its own pane)

    def __post_init__(self) -> None:
        self.rec_stream = rr.RecordingStream("dagkit_synthetic_demo")
        if self.connect is not None:
            self.rec_stream.connect_grpc(self.connect)
        elif self.save is not None:
            self.rec_stream.save(self.save)
        elif not self.headless:
            self.rec_stream.spawn()


@dataclass
class Config:
    rr_config: DemoRerunConfig = field(default_factory=DemoRerunConfig)
    seconds: float = 10.0
    hz: float = 30.0
    smooth_ms: float = 60.0  # smoother step time; above 1000 / hz its DropOldest input drops in realtime mode (see the health view)
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "realtime"


def build(config: Config, recording: rr.RecordingStream) -> Graph:
    def events() -> Iterator[Event]:
        start: float = time.perf_counter()
        for i in range(int(config.seconds * config.hz)):
            t_s: float = i / config.hz
            time.sleep(max(0.0, start + t_s - time.perf_counter()))  # paced like a live device
            if i % int(config.hz) == 0:  # gain first at a shared time, so the AsOf join always has one
                yield Event(int(t_s * 1e9), "gain", 0.1 + 0.3 * (i // int(config.hz) % 3))
            wobble: float = 0.15 * math.sin(37.0 * t_s)
            yield Event(int(t_s * 1e9), "raw", Sample((math.cos(t_s) + wobble, math.sin(t_s), 0.1 * t_s), math.sin(2.0 * t_s) + wobble))

    graph = Graph("synthetic", about="dagkit's synthetic demo: a paced helix, a smoother that keeps only the newest sample (it drops when slow), drawn live.")
    src = graph.source("helix", IterSource({"raw": Sample, "gain": float}, events))
    smooth = graph.node("smooth", Smooth(config.smooth_ms / 1e3))
    sink = graph.sink("rerun", RerunSink(recording, {"raw": Sample, "smooth": Sample}, {"raw": "/world/raw", "smooth": "/world/smooth"}))
    graph.edge(src["raw"], smooth["raw"], DropOldest(1))
    graph.edge(src["gain"], smooth["gain"], AsOf())
    graph.edge(src["raw"], sink["raw"], Block(8))
    graph.edge(smooth["smooth"], sink["smooth"], Block(8))
    return graph


def main(config: Config) -> RunReport:
    signal: rrb.TimeSeriesView = rrb.TimeSeriesView(origin="/signal", name="signal (video_time)")
    health: rrb.TimeSeriesView = rrb.TimeSeriesView(origin="/dagkit/health", name="runner health (wall_time)")
    trail = rrb.VisibleTimeRange("video_time", start=rrb.TimeRangeBoundary.cursor_relative(seconds=-4.0), end=rrb.TimeRangeBoundary.cursor_relative())
    helix = rrb.Spatial3DView(origin="/world", name="helix (last 4 s)", time_ranges=trail, eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital))
    outputs = [rrb.Spatial3DView(origin="/world", name="2 smooth · helix", time_ranges=trail), rrb.TimeSeriesView(origin="/dagkit/health", name="run health")]
    config.rr_config.rec_stream.send_blueprint(with_pipeline(rrb.Blueprint(rrb.Horizontal(helix, rrb.Vertical(signal, health))), outputs))
    return run_app(config, build)
