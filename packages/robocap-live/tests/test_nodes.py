"""The hands node and its graph on synthetic framesets, through the real Rust core without networks (unit lane: no GPU, no catalog)."""

import threading
import time
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
from serde import to_dict
from simplecv.types.frames import LumaFrameset
from simplecv.types.rig import RigCameras
from slam_rs.types import CatalogPoses

from robocap_live import _core
from robocap_live.graphs import hands_layer
from robocap_live.graphs.hands_layer import hands_layer_graph, segment_source
from robocap_live.nodes.hands import HandsNode, HandsParams
from robocap_live.types import CapRig, LayerRun

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


def test_the_run_carries_the_cores_stage_totals_and_the_stream_wall_as_the_old_tool_did(tmp_path: Path) -> None:
    """Main's LayerRun held the core's downsample, hands and output stage totals and the frameset loop's wall time; the node
    fills them again from the summary its core finishes with."""
    summaries: list[_core.LayerSummary] = []

    class Spy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.core, self.push, self.abort = (core := no_nets(*args, **kwargs)), core.push, core.abort
            self.nets: str = core.nets

        def finish(self) -> _core.LayerSummary:
            summaries.append(self.core.finish())
            return summaries[-1]

    hands, out = HandsNode(SEGMENT, HandsParams(models_dir=tmp_path), tmp_path, core=Spy), Recorder()  # type: ignore[arg-type]
    hands.reset(context(4))
    started: float = time.perf_counter()
    for frames in framesets(4):
        hands.step(Tick(frames.t_ns, "frames", frames, {}), out)
    hands.flush(out)
    run, (summary,) = out.items[1][2], summaries
    assert isinstance(run, LayerRun)
    assert (run.downsample_s, run.hands_s, run.output_s) == tuple(summary.stage_total_ms(s) / 1e3 for s in ("downsample", "hands", "output"))
    assert 0.0 < run.loop_s <= time.perf_counter() - started


@settings(max_examples=8, deadline=None)
@given(offsets_us=st.lists(st.one_of(st.none(), st.integers(-1_900, 1_900), st.integers(2_100, 4_000)), min_size=1, max_size=5))
def test_a_frameset_counts_its_own_pose_within_2_ms_else_holds_an_earlier_one(
    tmp_path_factory: pytest.TempPathFactory, offsets_us: list[int | None]
) -> None:
    """``offsets_us[i]``: frameset i's pose sits that far from it (None: no pose). Within 2 ms it is the frameset's own, else held.

    Only this own / held split shows without networks (the layer holds no pose). Which earlier pose a held frameset takes is the
    Rust pose store's test (``sched::queue``); the pose values reach the world landmarks, which the graph tier bands.
    """
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


def test_reset_hands_the_core_the_whole_pose_table_unchanged(tmp_path: Path) -> None:
    seen: list[tuple[object, ...]] = []

    def spy(*args: object, **kwargs: object) -> _core.HandsLayer:
        seen.append(args)
        return no_nets(*args, **kwargs)

    hands, table = HandsNode(SEGMENT, HandsParams(models_dir=tmp_path), tmp_path, core=spy), poses(6)
    hands.reset(RunContext("hands-layer", "hands", "lossless", statics={"cap_rig": cap_rig(), "poses": table}))
    hands.abort()
    ((_, _, segment, t_ns, world_from_rig),) = seen
    assert segment == SEGMENT.segment
    np.testing.assert_array_equal(t_ns, table.t_ns)
    np.testing.assert_array_equal(world_from_rig, table.world_from_rig)


class BorrowedCore:
    """A core that behaves like the PyO3 ``HandsLayer``: ``push`` holds the object while it waits, ``abort`` refuses to run beside it."""

    def __init__(self, partial: Path) -> None:
        self.nets, self.borrowed, self.aborted = "none", False, False
        self.pushing, self.release = threading.Event(), threading.Event()
        partial.touch()

    def push(self, *args: object) -> int:
        self.borrowed = True
        self.pushing.set()
        self.release.wait(5.0)
        self.borrowed = False
        return 0

    def abort(self) -> None:
        if self.borrowed:
            raise RuntimeError("Already borrowed")
        self.aborted = True


def test_abort_from_another_thread_waits_for_a_running_push_then_stops_the_core(tmp_path: Path) -> None:
    """A failing runner thread calls ``abort()`` at once, possibly while ``step`` is inside the core's ``push``."""
    cores: list[BorrowedCore] = []

    def borrowed(rig: object, partial: Path, *args: object, **kwargs: object) -> _core.HandsLayer:
        cores.append(BorrowedCore(partial))
        return cast(_core.HandsLayer, cores[-1])

    hands = HandsNode(SEGMENT, HandsParams(models_dir=tmp_path), tmp_path, core=borrowed)
    hands.reset(context(2))
    stepping = threading.Thread(target=hands.step, args=(Tick(T0, "frames", framesets(1)[0], {}), Recorder()))
    stepping.start()
    assert cores[0].pushing.wait(5.0)
    aborting = threading.Thread(target=hands.abort)
    aborting.start()
    aborting.join(0.2)
    assert aborting.is_alive() and not cores[0].aborted  # it waits for the push
    cores[0].release.set()
    stepping.join(5.0)
    aborting.join(5.0)
    assert cores[0].aborted and hands.core is None and list((tmp_path / SEGMENT.segment).iterdir()) == []


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


def ticks(seen: list[object]) -> list[tuple[object, ...]]:
    """Each tick a node or sink received (``collect_edges`` rows): port, time, the frameset's camera times and pixel sums (a layer
    file's path differs per run) and the AsOf snapshot as plain data."""
    rows: list[tuple[object, ...]] = []
    for port, t_ns, value, asof in cast(list[tuple[str, int, object, dict[str, object]]], seen):
        pixels = (value.cam_t_ns, [None if p is None else int(p.sum()) for p in value.luma]) if isinstance(value, LumaFrameset) else None
        rows.append((port, t_ns, pixels, {k: None if v is None else to_dict(v) for k, v in asof.items()}))
    return rows


def test_the_graph_validates_and_runs_the_same_ticks_sequential_and_threaded(tmp_path: Path) -> None:
    built: Graph = graph(tmp_path / "sequential", 5)
    built.validate()
    sequential = collect_edges(built, lambda g: run_sequential(g))
    threaded = collect_edges(graph(tmp_path / "threaded", 5), lambda g: run_threaded(g, mode="lossless"))
    statics: dict[str, object] = {"cap_rig": to_dict(cap_rig()), "poses": to_dict(poses(5))}
    expected = [("frames", f.t_ns, (f.cam_t_ns, [None if p is None else int(p.sum()) for p in f.luma]), statics) for f in framesets(5)]
    assert ticks(sequential["hands"]) == ticks(threaded["hands"]) == expected
    assert ticks(sequential["publish"]) == ticks(threaded["publish"]) == [("layer", framesets(5)[-1].t_ns, None, {})]
    for run in ("sequential", "threaded"):
        (published,) = (tmp_path / run / SEGMENT.segment).glob("hands.*.rrd")
        assert published.stat().st_size > 0 and not list((tmp_path / run / SEGMENT.segment).glob(".*.partial"))


def test_the_segment_source_refuses_a_segment_without_slam_rs_a_path_like_id_or_another_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hands layer reads the rig pose entity, which an older ``slam`` layer also writes, so ``slam_rs`` must be there."""
    layers: dict[str, list[str]] = {SEGMENT.segment: ["base", "slam_rs"], "robocap__other__s00000002": ["base", "slam"]}
    monkeypatch.setattr(hands_layer, "segment_layers", lambda segment: layers.get(segment.segment, ["base", "slam_rs"]))
    assert segment_source(SEGMENT).outputs["frames"] is LumaFrameset
    no_slam_rs, path_like, nested = (SegmentRef("robocap", s) for s in ("robocap__other__s00000002", "../escape", "robocap__a/b__s00000001"))
    for refused in (no_slam_rs, path_like, nested, SegmentRef("robocap", SEGMENT.segment, timeline="capture_time")):
        with pytest.raises(ValueError, match="slam_rs"):
            segment_source(refused)
