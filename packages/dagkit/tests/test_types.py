"""Types tier, unit lane: serde round-trips and the `to_rerun` structure (no catalog)."""

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import rerun as rr
from dagkit_fakes import Point
from hypothesis import given
from hypothesis import strategies as st
from serde import from_dict, to_dict
from serde.json import from_json, to_json

from dagkit.node import Fenced, Tick
from dagkit.sources import SegmentRef
from dagkit.testing import layer_table
from dagkit.time import STATIC
from dagkit.types import LayerFile, Log, Producer, Type, row_times

text = st.text(st.characters(codec="utf-8", exclude_categories=("Cs",)), max_size=12)


@given(st.builds(Producer, text, text, text, text, text, text))
def test_producer_json_roundtrip(producer: Producer) -> None:
    assert from_json(Producer, to_json(producer)) == producer


@given(st.builds(SegmentRef, text, text, text, text, st.none() | st.tuples(st.integers(-(2**62), 0), st.integers(1, 2**62))))
def test_segment_ref_roundtrip(segment: SegmentRef) -> None:
    assert from_dict(SegmentRef, to_dict(segment)) == segment


def test_layer_file_roundtrip(tmp_path: Path) -> None:
    layer = LayerFile(tmp_path / "hands.abc.rrd")
    assert from_json(LayerFile, to_json(layer)) == layer


coord = st.floats(-1e6, 1e6, width=32)


@given(st.tuples(coord, coord, coord), text)
def test_point_type_serde_and_rerun_structure(xyz: tuple[float, float, float], label: str) -> None:
    point = Point(xyz, label)
    assert isinstance(point, Type)
    assert from_json(Point, to_json(point)) == point
    logs: list[Log] = list(point.to_rerun("/world/a"))
    assert [log.path for log in logs] == Point.contents("/world/a") == ["/world/a/point"]
    assert all(isinstance(log.data, rr.Points3D) and not log.static for log in logs)


def test_row_times_reads_the_index_or_gives_static() -> None:
    timed = pa.table({"video_time": pa.array([5, 7], pa.duration("ns")), "v": [1, 2]})
    assert row_times(timed, "video_time") == [5, 7]
    assert row_times(pa.table({"v": [1]}), "video_time") == [STATIC]


def test_fenced_and_tick_are_frozen_values() -> None:
    fenced = Fenced(3, None)
    tick = Tick(1, "x", fenced, {"a": None})
    assert tick.value == Fenced(3, None) and tick.asof == {"a": None}


@given(st.lists(st.tuples(st.integers(0, 10**12), st.tuples(coord, coord, coord), text), min_size=1, max_size=4, unique_by=lambda v: v[0]))
def test_layer_table_reads_back_through_from_layer(points: list[tuple[int, tuple[float, float, float], str]]) -> None:
    values: list[tuple[int, Point]] = sorted((t, Point(xyz, label)) for t, xyz, label in points)
    table: pa.Table = layer_table(values, "/world")
    assert list(Point.from_layer(table, "/world", "video_time")) == values


def test_layer_table_repeats_statics_in_every_row() -> None:
    class Static(Point):
        def to_rerun(self, root: str) -> Iterable[Log]:
            yield Log(f"{root}/static", rr.Points3D([self.xyz], labels=[self.label]), static=True)

    table: pa.Table = layer_table([(5, Point((1.0, 2.0, 3.0), "a")), (STATIC, Static((0.0, 0.0, 0.0), "s")), (9, Point((4.0, 5.0, 6.0), "b"))], "/w")
    assert row_times(table, "video_time") == [5, 9] and table["/w/static:Points3D:labels"].to_pylist() == [["s"], ["s"]]
    assert layer_table([(STATIC, Static((0.0, 0.0, 0.0), "s"))], "/w").num_rows == 1
