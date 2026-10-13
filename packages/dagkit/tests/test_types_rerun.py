"""Integration lane: Rerun round-trips through an in-process server (a catalog), sources and registration."""

from pathlib import Path

import pytest
import rerun as rr
from dagkit_fakes import Collect, Point

from dagkit.graph import Graph
from dagkit.node import Event
from dagkit.runners import run_sequential
from dagkit.sinks import LayerSink, open_layer, segment_layers
from dagkit.sources import CatalogSegmentSource, Column, IterSource, RecordingSource, SegmentRef
from dagkit.time import STATIC, GraphError
from dagkit.types import Producer, rerun_roundtrip

pytestmark = pytest.mark.integration
PRODUCER = Producer("dagkit", "0.1.0+test", "points", "x86_64-cuda")


def test_rerun_roundtrip_timed_and_static() -> None:
    point = Point((1.0, 2.0, 3.0), "a")
    assert rerun_roundtrip(point, "/world", "video_time", 1234) == [(1234, point)]
    assert rerun_roundtrip(point, "/world", "video_time", STATIC) == [(STATIC, point)]


def write_layer(tmp_path: Path, segment: SegmentRef, register: str = "off") -> Path:
    recording, partial = open_layer(segment, "points", PRODUCER, output_dir=tmp_path)
    sink = LayerSink(recording, partial, segment, "points", {"point": Point}, {"point": "/points"}, register=register)  # type: ignore[arg-type]
    events = [Event(t, "point", Point((float(t), 0.0, 0.0), f"p{t}")) for t in (10, 20)]
    graph = Graph("points")
    graph.checkpoint(graph.source("src", IterSource({"point": Point}, lambda: iter(events)))["point"], sink)
    run_sequential(graph)
    assert sink.published is not None
    return sink.published


def test_recording_source_reads_a_published_layer_back(tmp_path: Path) -> None:
    path: Path = write_layer(tmp_path, SegmentRef("ds", "seg0"))
    graph = Graph("read")
    src = graph.source("src", RecordingSource(path, {"point": Column(Point, "/points")}))
    sink = Collect(("point",))
    sink.inputs = {"point": Point}
    seen: list[tuple[int, Point]] = []
    sink.on_step = lambda tick: seen.append((tick.t_ns, tick.value))  # type: ignore[arg-type]
    sink.step = lambda tick, out: sink.on_step(tick)  # type: ignore[method-assign, misc]
    graph.edge(src["point"], graph.sink("sink", sink)["point"])
    run_sequential(graph)
    assert [t for t, _ in seen] == [10, 20] and seen[1][1] == Point((20.0, 0.0, 0.0), "p20")


def test_register_new_adds_the_layer_once_and_refuses_a_second(tmp_path: Path) -> None:
    base = rr.RecordingStream("dagkit-test", recording_id="seg0", send_properties=False)
    base.save(tmp_path / "base.rrd")
    base.log("/base", rr.Points3D([[0, 0, 0]]), static=True)
    base.flush()
    base.disconnect()
    with rr.server.Server(host="127.0.0.1", datasets={"ds": [str(tmp_path / "base.rrd")]}) as server:
        segment = SegmentRef("ds", "seg0", catalog_url=server.url())
        write_layer(tmp_path / "layers", segment, register="new")
        assert segment_layers(segment) == ["base", "points"]
        with pytest.raises(GraphError, match="already has layer 'points'"):
            write_layer(tmp_path / "layers", segment, register="new")
        source = CatalogSegmentSource(segment, {"point": Column(Point, "/points")})
        assert [e.t_ns for e in source.events()] == [10, 20]
