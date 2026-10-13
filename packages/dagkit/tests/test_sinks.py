"""LayerSink / LayerFileSink publish policy without a catalog (register='off'), through a real run."""

import stat
from pathlib import Path

import pytest
from dagkit_fakes import Point

from dagkit.graph import Block, Graph
from dagkit.node import Control, Event, RunContext, Tick
from dagkit.runners import run_sequential, run_threaded
from dagkit.sinks import LayerFileSink, LayerSink, open_layer
from dagkit.sources import IterSource, SegmentRef
from dagkit.testing import Recorder
from dagkit.time import GraphError
from dagkit.types import LayerFile, Producer

PRODUCER = Producer("dagkit", "0.1.0+test", "points", "x86_64-cuda")


def point_graph(sink: LayerSink, n: int, control: Control | None = None) -> Graph:
    def events():  # noqa: ANN202
        for t in range(n):
            if control is not None and t == 2:
                control.stop()
            yield Event(t * 1000, "point", Point((float(t), 0.0, 0.0), f"p{t}"))

    graph = Graph("points")
    src = graph.source("src", IterSource({"point": Point}, events))
    graph.checkpoint(src["point"], sink)
    return graph


def layer_sink(tmp_path: Path, segment: SegmentRef) -> LayerSink:
    recording, partial = open_layer(segment, "points", PRODUCER, output_dir=tmp_path)
    assert partial.name.startswith(".points.") and partial.name.endswith(".partial")
    return LayerSink(recording, partial, segment, "points", {"point": Point}, {"point": "/points"}, register="off")


def test_complete_run_publishes_an_immutable_0644_file(tmp_path: Path) -> None:
    sink = layer_sink(tmp_path, SegmentRef("ds", "seg0"))
    run_sequential(point_graph(sink, 5))
    assert sink.published is not None and sink.published.parent == tmp_path / "seg0"
    assert sink.published.name.startswith("points.") and sink.published.suffix == ".rrd"
    assert stat.S_IMODE(sink.published.stat().st_mode) == 0o644
    assert sorted(p.name for p in (tmp_path / "seg0").iterdir()) == [sink.published.name]
    sink.flush(Recorder())  # a second flush publishes nothing
    assert len(list((tmp_path / "seg0").iterdir())) == 1


def test_stopped_and_windowed_runs_get_an_infix(tmp_path: Path) -> None:
    control = Control()
    stopped = layer_sink(tmp_path, SegmentRef("ds", "seg0"))
    run_threaded(point_graph(stopped, 50, control), mode="lossless", control=control)
    windowed = layer_sink(tmp_path, SegmentRef("ds", "seg1", window_ns=(0, 3000)))
    run_sequential(point_graph(windowed, 3))
    assert stopped.published is not None and stopped.published.name.startswith("points.stopped.")
    assert windowed.published is not None and windowed.published.name.startswith("points.window.")


def test_failed_run_deletes_the_partial(tmp_path: Path) -> None:
    sink = layer_sink(tmp_path, SegmentRef("ds", "seg0"))

    def events():  # noqa: ANN202
        yield Event(0, "point", Point((0.0, 0.0, 0.0), "a"))
        raise OSError("device gone")

    graph = Graph("fails")
    graph.checkpoint(graph.source("src", IterSource({"point": Point}, events))["point"], sink)
    with pytest.raises(OSError, match="device gone"):
        run_sequential(graph)
    assert sink.published is None and list((tmp_path / "seg0").iterdir()) == []


def test_layer_sink_refuses_realtime(tmp_path: Path) -> None:
    sink = layer_sink(tmp_path, SegmentRef("ds", "seg0"))
    with pytest.raises(GraphError, match="lossless only"):
        sink.reset(RunContext("g", "layer", "realtime"))


def test_layer_file_sink_publishes_what_another_writer_produced(tmp_path: Path) -> None:
    written: Path = tmp_path / "rust-out.rrd"
    written.write_bytes(b"rrd")
    graph = Graph("files")
    src = graph.source("src", IterSource({"layer": LayerFile}, lambda: iter([Event(0, "layer", LayerFile(written))])))
    sink = LayerFileSink(SegmentRef("ds", "seg0"), "hands", output_dir=tmp_path / "out", register="off")
    graph.edge(src["layer"], graph.sink("publish", sink)["layer"], Block())
    run_sequential(graph)
    assert sink.published is not None and sink.published.read_bytes() == b"rrd" and not written.exists()
    sink.step(Tick(0, "layer", LayerFile(written), {}), Recorder())
    with pytest.raises(FileNotFoundError):
        sink.flush(Recorder())
