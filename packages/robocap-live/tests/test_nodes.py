"""The hands node and its graph on synthetic framesets, through the real Rust core without networks (unit lane: no GPU, no catalog)."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dagkit.graph import Graph
from dagkit.node import Event, RunContext, Tick
from dagkit.runners import run_sequential, run_threaded
from dagkit.sinks import LayerFileSink
from dagkit.sources import IterSource, SegmentRef
from dagkit.testing import Recorder, collect_edges
from dagkit.time import STATIC
from dagkit.types import LayerFile
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.types.frames import LumaFrameset
from simplecv.types.rig import RigCameras

from robocap_live import _core
from robocap_live.graphs.hands_layer import hands_layer_graph
from robocap_live.nodes.hands import HandsNode, HandsParams
from robocap_live.types import CapRig, CatalogPoses, LayerRun

SEGMENT = SegmentRef("robocap", "robocap__test__s00000001")
VIEWS: tuple[str, ...] = tuple(f"rig_00/cam_{i:02d}" for i in range(6))
T0, STEP = 1_000_000_000, 33_000_000


def no_nets(*args: object, **kwargs: object) -> _core.HandsLayer:
    """The real core with ``nets="none"``: the pipeline, the pose store and the layer file, but no hand networks."""
    return _core.HandsLayer(*args, **(kwargs | {"nets": "none", "models_dir": None}))  # type: ignore[arg-type]


def cap_rig() -> CapRig:
    cam_from_rig: Float64[ndarray, "6 4 4"] = np.tile(np.eye(4), (6, 1, 1))
    cam_from_rig[:, 0, 3] = np.arange(6) * 0.05
    intrinsics: Float64[ndarray, "6 3 3"] = np.tile(np.array([[600.0, 0.0, 959.5], [0.0, 600.0, 539.5], [0.0, 0.0, 1.0]]), (6, 1, 1))
    rig = RigCameras(VIEWS, intrinsics, cam_from_rig, np.tile(np.array([_core.FULL_WIDTH, _core.FULL_HEIGHT], dtype=np.int64), (6, 1)))
    return CapRig(rig, tuple(_core.CAMERA_NAMES), np.zeros((6, 8)))


def poses(frames: int) -> CatalogPoses:
    """A pose 150 us after every frameset but the second (the core's 2 ms rule matches them; the second is held)."""
    t_ns = np.asarray([T0 + i * STEP + 150_000 for i in range(frames) if i != 1], dtype=np.int64)
    world_from_rig: Float64[ndarray, "p 4 4"] = np.tile(np.eye(4), (len(t_ns), 1, 1))
    world_from_rig[:, 0, 3] = np.arange(len(t_ns)) * 0.01
    return CatalogPoses(t_ns, world_from_rig)


def framesets(count: int) -> list[LumaFrameset]:
    plane: UInt8[ndarray, "h w"] = np.full((_core.FULL_HEIGHT, _core.FULL_WIDTH), 80, dtype=np.uint8)
    return [
        LumaFrameset(
            T0 + i * STEP,
            VIEWS,
            tuple(None if c == 5 and i % 2 else plane for c in range(6)),
            tuple(None if c == 5 and i % 2 else T0 + i * STEP + c for c in range(6)),
        )
        for i in range(count)
    ]


def node(tmp_path: Path) -> HandsNode:
    return HandsNode(SEGMENT, HandsParams(models_dir=tmp_path), tmp_path, core=no_nets)


def context(count: int) -> RunContext:
    return RunContext("hands-layer", "hands", "lossless", statics={"cap_rig": cap_rig(), "poses": poses(count)})


def test_flush_finishes_the_layer_once_and_emits_its_file_and_counts_at_the_last_frameset(tmp_path: Path) -> None:
    hands, out = node(tmp_path), Recorder()
    hands.reset(context(4))
    for frames in framesets(4):
        hands.step(Tick(frames.t_ns, "frames", frames, {}), out)
    assert out.items == []
    hands.flush(out)
    hands.flush(out)  # a second flush emits nothing (and does not finish the finished core again)
    (layer_port, t_layer, layer), (run_port, t_run, run) = out.items
    assert (layer_port, run_port) == ("layer", "run") and t_layer == t_run == T0 + 3 * STEP
    assert isinstance(layer, LayerFile) and layer.path.is_file() and layer.path.parent == tmp_path / SEGMENT.segment
    assert isinstance(run, LayerRun) and (run.framesets, run.with_pose, run.held_pose) == (4, 3, 1)


@settings(max_examples=8, deadline=None)
@given(offsets_us=st.lists(st.one_of(st.none(), st.integers(-1_900, 1_900), st.integers(2_100, 4_000)), min_size=1, max_size=5))
def test_each_frameset_takes_the_pose_within_2_ms_else_the_newest_earlier_one(
    tmp_path_factory: pytest.TempPathFactory, offsets_us: list[int | None]
) -> None:
    """``offsets_us[i]``: frameset i's pose sits that far from it (None: no pose). Within 2 ms it is the frameset's own, else held."""
    tmp_path: Path = tmp_path_factory.mktemp("hands")
    t_ns = np.asarray([T0 + i * STEP + 1_000 * o for i, o in enumerate(offsets_us) if o is not None], dtype=np.int64)
    table = CatalogPoses(t_ns, np.tile(np.eye(4), (len(t_ns), 1, 1)))
    hands, out = node(tmp_path), Recorder()
    hands.reset(RunContext("hands-layer", "hands", "lossless", statics={"cap_rig": cap_rig(), "poses": table}))
    for frames in framesets(len(offsets_us)):
        hands.step(Tick(frames.t_ns, "frames", frames, {}), out)
    hands.flush(out)
    own: list[bool] = [o is not None and abs(o) <= 2_000 for o in offsets_us]
    run = out.items[1][2]
    assert isinstance(run, LayerRun) and run.framesets == len(offsets_us) and run.with_pose == sum(own)
    assert run.held_pose == sum(not mine and any(own[:i]) for i, mine in enumerate(own))
    assert [t for _, t, _ in out.items] == [T0 + (len(offsets_us) - 1) * STEP] * 2


def test_reset_aborts_the_running_core_and_builds_a_fresh_one(tmp_path: Path) -> None:
    hands, out = node(tmp_path), Recorder()
    hands.reset(context(3))
    first: Path = next((tmp_path / SEGMENT.segment).glob(".hands.*.partial"))
    hands.step(Tick(T0, "frames", framesets(1)[0], {}), out)
    hands.reset(context(3))  # idempotent: the half-written layer is dropped, the new core starts from frameset 0
    assert not first.exists() and len(list((tmp_path / SEGMENT.segment).glob(".hands.*.partial"))) == 1
    for frames in framesets(3):
        hands.step(Tick(frames.t_ns, "frames", frames, {}), out)
    hands.flush(out)
    run = out.items[1][2]
    assert isinstance(run, LayerRun) and run.framesets == 3


def test_abort_deletes_the_partial_and_reset_refuses_a_segment_without_poses(tmp_path: Path) -> None:
    hands = node(tmp_path)
    hands.reset(context(2))
    hands.step(Tick(T0, "frames", framesets(1)[0], {}), Recorder())
    hands.abort()
    assert list((tmp_path / SEGMENT.segment).iterdir()) == []
    with pytest.raises(ValueError, match="slam_rs"):
        hands.reset(RunContext("hands-layer", "hands", "lossless", statics={"cap_rig": cap_rig()}))


def graph(tmp_path: Path, count: int) -> Graph:
    def events() -> Iterator[Event]:
        yield Event(STATIC, "cap_rig", cap_rig())
        yield Event(STATIC, "poses", poses(count))
        yield from (Event(f.t_ns, "frames", f) for f in framesets(count))

    source = IterSource({"frames": LumaFrameset, "cap_rig": CapRig, "poses": CatalogPoses}, events)
    publish = LayerFileSink(SEGMENT, "hands", output_dir=tmp_path, register="off")
    return hands_layer_graph(source, node(tmp_path), {"publish": publish})


def ticks(seen: list[object]) -> list[tuple[str, int]]:
    """(port, time) of each tick a node or sink received (``collect_edges`` rows)."""
    return [cast(tuple[str, int], cast(tuple, row)[:2]) for row in seen]


def test_the_graph_validates_and_runs_the_same_ticks_sequential_and_threaded(tmp_path: Path) -> None:
    built: Graph = graph(tmp_path / "sequential", 5)
    built.validate()
    sequential = collect_edges(built, lambda g: run_sequential(g))
    threaded = collect_edges(graph(tmp_path / "threaded", 5), lambda g: run_threaded(g, mode="lossless"))
    assert ticks(sequential["hands"]) == ticks(threaded["hands"]) == [("frames", f.t_ns) for f in framesets(5)]
    assert ticks(sequential["publish"]) == ticks(threaded["publish"]) == [("layer", framesets(5)[-1].t_ns)]
    for run in ("sequential", "threaded"):
        (published,) = (tmp_path / run / SEGMENT.segment).glob("hands.*.rrd")
        assert published.stat().st_size > 0 and not list((tmp_path / run / SEGMENT.segment).glob(".*.partial"))
