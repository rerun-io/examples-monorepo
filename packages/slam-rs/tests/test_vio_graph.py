"""The VIO node and graph on a synthetic two-camera rig through the real CPU core (unit lane: no GPU, no catalog).

The node hands each frameset whole (its images and the IMU samples since the previous one) to ``Lockstep.push``, whatever runner
delivers the one stream, and however the samples are split across framesets (D17); its outputs follow the dagkit node contract.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from dagkit.graph import Graph
from dagkit.node import Control, Event, RunContext, Tick
from dagkit.runners import RunReport, run_sequential, run_threaded
from dagkit.sinks import LayerFileSink, RerunSink
from dagkit.sources import IterSource, SegmentRef
from dagkit.testing import Recorder, check_graph_docs, check_picture, collect_edges, layer_table
from dagkit.time import STATIC
from dagkit.verify import Case
from fixture_types import FRAME_PERIOD_NS, IMU_PERIOD_NS, CameraFactory, TextureFactory, gravity_batch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import from_dict, to_dict

from slam_rs import _core, verify
from slam_rs.apis.catalog_layer import vio_publish
from slam_rs.catalog_feed import MSD_RIG, RIG_ENTITY, ROBOCAP_RIG, Frameset, ImuCalib, ImuStream, SegmentFeed
from slam_rs.config import load_slam_config
from slam_rs.graphs import vio as graphs_vio
from slam_rs.graphs.vio import VioSource, vio_graph
from slam_rs.nodes.vio import VioNode, VioParams
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import Trajectory
from slam_rs.types import CatalogPoses, RigPose, VioRig

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
SEGMENT = SegmentRef("msd-index", "msd-index__test__synthetic")
STEPS: int = 5


CONFIG: str = _core.VioConfig().to_json()
"""The synthetic rig takes the core's default config (a catalog dataset's comes from ``VioParams.for_dataset``)."""


def framesets(texture: TextureFactory, splits: list[int] | None = None) -> list[Frameset]:
    """``STEPS`` framesets of a still rig, each with its 1 kHz samples; ``splits[i]`` samples of frameset ``i`` move to ``i + 1``
    (a short batch leaves the frameset uncovered, so it is held until the next batch)."""
    batches: list[Int64[ndarray, " n"]] = [
        np.arange(s * FRAME_PERIOD_NS, (s + 1) * FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64) for s in range(STEPS)
    ]
    for i, moved in enumerate(splits or []):
        if moved and i + 1 < STEPS:
            batches[i], batches[i + 1] = (
                batches[i][: len(batches[i]) - moved],
                np.concatenate([batches[i][len(batches[i]) - moved :], batches[i + 1]]),
            )
    out: list[Frameset] = []
    for step, t_ns in enumerate(batches):
        gyro, accel = gravity_batch(t_ns)
        out.append(Frameset(step * FRAME_PERIOD_NS, [texture(step, 0), texture(step - 2, 0)], ImuStream(t_ns, gyro, accel), None))
    return out


def rig(camera: CameraFactory, imu: ImuCalib) -> VioRig:
    return VioRig((camera(0, 0.0), camera(1, 0.1)), imu, 0)


def stream(rig: VioRig, sets: list[Frameset]) -> Iterator[Event]:
    """What ``VioSource`` emits: STATIC rig, then each frameset (images and IMU batch) at its time."""
    yield Event(STATIC, "rig", rig)
    for f in sets:
        yield Event(f.t_ns, "frames", f)


def ticks(rig: VioRig, sets: list[Frameset]) -> list[Tick]:
    return [Tick(e.t_ns, e.port, e.value, {"rig": rig}) for e in stream(rig, sets) if e.t_ns != STATIC]


class Strict(Recorder):
    """A recorder that holds the node to its watermarks: no output at or before a time the node declared final."""

    def __init__(self) -> None:
        super().__init__()
        self.wm: int = STATIC
        self.retries: int = 0  # framesets the estimator refused for want of IMU before the flush (Lockstep.retries)

    def put(self, port: str, t_ns: int, value: object) -> None:
        assert t_ns > self.wm, f"{port} at {t_ns} after watermark {self.wm}"
        super().put(port, t_ns, value)

    def watermark(self, t_ns: int) -> None:
        assert t_ns >= self.wm, f"watermark {t_ns} after {self.wm}"
        self.wm = t_ns


def node_outputs(node: VioNode, rig: VioRig, sets: list[Frameset]) -> Strict:
    out = Strict()
    node.reset(RunContext("test", "vio", "lossless", statics={"rig": rig}))
    for tick in ticks(rig, sets):
        node.step(tick, out)
    assert node.lockstep is not None
    out.retries = node.lockstep.retries
    node.flush(out)
    return out


def lockstep_poses(rig: VioRig, sets: list[Frameset]) -> list[Float64[ndarray, " 7"]]:
    lockstep = Lockstep(_core.Vio(_core.Calibration.from_catalog(list(rig.cameras), rig.imu), _core.VioConfig()))
    results = [r for f in sets for _, r in lockstep.push(f)] + [r for _, r in lockstep.flush()]
    return [np.asarray(r.world_from_rig) for r in results]


HOLD: int = FRAME_PERIOD_NS // IMU_PERIOD_NS - 1
"""Samples to move so that a batch ends AT its frame's time: the estimator then refuses that frameset until the next batch."""


@settings(max_examples=6, deadline=None)
@given(splits=st.lists(st.integers(min_value=0, max_value=HOLD), min_size=STEPS, max_size=STEPS))
@example(splits=[HOLD] * STEPS)
def test_the_node_makes_lockstep_calls_whatever_the_imu_split(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, tmp_path_factory: pytest.TempPathFactory, splits: list[int]
) -> None:
    """D17 through the node: moving a frameset's last samples to the next batch changes no pose, also when the batch then ends at
    its frame and the frameset is held and retried; the node's poses are the ones ``Lockstep.push`` gets from the same framesets."""
    vrig: VioRig = rig(camera, imu)
    reference: list[Float64[ndarray, " 7"]] = lockstep_poses(vrig, framesets(texture))
    out: Strict = node_outputs(VioNode(SEGMENT, VioParams(backend="cpu"), tmp_path_factory.mktemp("vio"), "cpu", CONFIG), vrig, framesets(texture, splits))
    assert out.retries == sum(moved == HOLD for moved in splits[: STEPS - 1]), "each frameset whose batch ends at its frame is held once"
    poses: list[RigPose] = [v for p, _, v in out.items if p == "pose"]  # type: ignore[misc]
    assert len(poses) == len(reference) >= 2
    for pose, expected in zip(poses, reference, strict=True):
        np.testing.assert_array_equal(pose.position_m, expected[:3])
        np.testing.assert_array_equal(np.roll(pose.quaternion_wxyz, -1), expected[3:])


def test_reset_is_idempotent_flush_drains_once_and_outputs_keep_time_order(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, tmp_path: Path
) -> None:
    node = VioNode(SEGMENT, VioParams(backend="cpu"), tmp_path, "cpu", CONFIG)
    vrig: VioRig = rig(camera, imu)
    runs: list[Recorder] = []
    for _ in range(2):
        node.reset(RunContext("test", "vio", "lossless", statics={"rig": vrig}))  # a reset over a used node starts afresh
        runs.append(node_outputs(node, vrig, framesets(texture)))
        drained: int = len(runs[-1].items)
        node.flush(runs[-1])
        assert len(runs[-1].items) == drained, "a second flush emitted outputs"
    for out in runs:
        times: dict[str, list[int]] = {}
        for port, t_ns, _ in out.items:
            times.setdefault(port, []).append(t_ns)
        assert all(a <= b for t in times.values() for a, b in zip(t, t[1:], strict=False))
        assert [p for p, _, _ in out.items].count("layer") == 1 and out.items[-1][0] == "layer"
        assert out.items[-1][1] == (STEPS - 1) * FRAME_PERIOD_NS, "the layer leaves at the last frameset's time"
    first, second = ([(p, t, to_dict(v)) for p, t, v in out.items if p == "pose"] for out in runs)
    np.testing.assert_equal(first, second)
    assert node.run is not None and node.run.framesets == STEPS and node.run.lost == 0


def graph_of(camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, tmp_path: Path) -> Graph:
    """The App's graph over the synthetic stream: the layer publisher and the live pose sink."""
    vrig: VioRig = rig(camera, imu)
    source = IterSource({"rig": VioRig, "frames": Frameset}, lambda: stream(vrig, framesets(texture)))
    publish = LayerFileSink(SEGMENT, "slam_rs", output_dir=tmp_path, register="off")
    viewer = RerunSink(rr.RecordingStream("slam-rs-test"), {"pose": RigPose}, {"pose": "/world/rig"})
    return vio_graph(source, VioNode(SEGMENT, VioParams(backend="cpu"), tmp_path, "cpu", CONFIG), {"publish": publish, "viewer": viewer})


def test_the_app_graph_validates_draws_its_declaration_and_documents_itself(tmp_path: Path) -> None:
    """The graph the App runs (its source reads nothing until the run starts)."""
    graph: Graph = vio_publish(SEGMENT, VioParams(decode_device="cpu"), tmp_path, "off", rr.RecordingStream("slam-rs-test"))[0]
    graph.validate()
    check_picture(graph, tmp_path)
    check_graph_docs(graph, snapshot=PACKAGE_DIR / "GRAPH.md")


def test_sequential_and_threaded_deliver_the_same_ticks_and_publish_the_same_poses(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, tmp_path: Path
) -> None:
    runs: list[dict[str, list[object]]] = []
    written: list[Trajectory] = []
    for name, runner in (("sequential", run_sequential), ("threaded", run_threaded)):
        graph: Graph = graph_of(camera, imu, texture, tmp_path / name)
        runs.append(collect_edges(graph, lambda g, runner=runner: runner(g, mode="lossless")))
        (published,) = (tmp_path / name / SEGMENT.segment).glob("slam_rs.*.rrd")
        assert published.stat().st_size > 0
        run = cast(VioNode, graph.nodes["vio"].obj).run
        assert run is not None
        written.append(run.estimate)  # what write_layer wrote into the published file
    ticks_in = [[(p, t, type(v).__name__) for p, t, v, _ in seen["vio"]] for seen in runs]  # type: ignore[misc]
    poses = [[(p, t, to_dict(v)) for p, t, v, _ in seen["viewer"]] for seen in runs]  # type: ignore[misc]
    assert ticks_in[0] == ticks_in[1] and len(ticks_in[0]) == STEPS
    np.testing.assert_equal(poses[0], poses[1])
    assert len(poses[0]) == STEPS
    for field in ("t_ns", "position_m", "quaternion_wxyz"):
        np.testing.assert_array_equal(getattr(written[0], field), getattr(written[1], field))


@pytest.mark.parametrize("runner", [run_sequential, run_threaded])
@pytest.mark.parametrize("stop_after", [0, 1])
def test_a_stop_before_two_poses_ends_stopped_without_a_layer(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, tmp_path: Path, runner: Callable[..., RunReport], stop_after: int
) -> None:
    """Ctrl-C while the segment loads (or after one frameset) is a stop, not a failed layer write; nothing is published."""
    control = Control()
    graph: Graph = graph_of(camera, imu, texture, tmp_path)
    source = cast(IterSource, graph.nodes["segment"].obj)
    events = source.events

    def stopping() -> Iterator[Event]:
        for event in events():
            if event.port == "frames" and event.t_ns == stop_after * FRAME_PERIOD_NS:
                control.stop()
            yield event

    source.events = stopping
    assert runner(graph, mode="lossless", control=control).status == "stopped"
    assert not list(tmp_path.rglob("*.rrd")) and not list(tmp_path.rglob("*.partial"))
    assert cast(VioNode, graph.nodes["vio"].obj).lockstep is None


def test_a_failed_run_releases_the_estimator_and_leaves_no_partial_layer(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """abort() after a tracking failure drops the native estimator; after a publish failure, the unmoved layer file goes too."""
    graph: Graph = graph_of(camera, imu, texture, tmp_path / "tracking")
    with monkeypatch.context() as patched:
        patched.setattr(Lockstep, "push", lambda self, frameset: (_ for _ in ()).throw(RuntimeError("tracking failed")))
        with pytest.raises(RuntimeError, match="tracking failed"):
            run_sequential(graph)
    assert cast(VioNode, graph.nodes["vio"].obj).lockstep is None
    graph = graph_of(camera, imu, texture, tmp_path / "publish")
    monkeypatch.setattr(LayerFileSink, "step", lambda self, tick, out: (_ for _ in ()).throw(OSError("publish failed")))
    with pytest.raises(OSError, match="publish failed"):
        run_sequential(graph)
    assert not list((tmp_path / "publish").rglob("*.partial")) and not list((tmp_path / "publish").rglob("*.rrd"))


def test_the_source_releases_the_segment_when_a_run_fails_although_the_error_is_kept(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open segment (decoders, cached video) closes with the run, not when the caller drops the exception."""
    vrig: VioRig = rig(camera, imu)
    lifecycle: list[str] = []
    feed = Mock(spec=SegmentFeed, cameras=vrig.cameras, imu=vrig.imu, export_offset_ns=0)
    feed.framesets.side_effect = lambda stop_ns: iter(framesets(texture))

    @contextmanager
    def segment(*_args: object, **_kwargs: object) -> Iterator[SegmentFeed]:
        lifecycle.append("open")
        try:
            yield feed
        finally:
            lifecycle.append("closed")

    monkeypatch.setattr(graphs_vio, "open_segment", segment)
    monkeypatch.setattr(VioNode, "reset", lambda self, ctx: (_ for _ in ()).throw(ValueError("reset failed")))
    graph: Graph = vio_graph(VioSource(SEGMENT, MSD_RIG, "cpu"), VioNode(SEGMENT, VioParams(backend="cpu"), tmp_path, "cpu", CONFIG), {})
    with pytest.raises(ValueError, match="reset failed") as kept:
        run_threaded(graph, mode="lossless")
    assert kept.value is not None and lifecycle == ["open", "closed"]


def test_the_source_sends_each_frameset_whole_at_its_estimator_time(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One event per frameset, its images and its IMU batch together (what ``Lockstep.push`` takes), after the STATIC rig."""
    vrig: VioRig = rig(camera, imu)
    sets: list[Frameset] = framesets(texture)
    feed = Mock(spec=SegmentFeed, cameras=vrig.cameras, imu=vrig.imu, export_offset_ns=0)
    feed.framesets.side_effect = lambda stop_ns: iter(sets)
    monkeypatch.setattr(graphs_vio, "open_segment", lambda *_args, **_kwargs: nullcontext(feed))
    events: list[Event] = list(cast(IterSource, VioSource(SEGMENT, MSD_RIG, "cpu")).events())
    assert [(e.port, e.t_ns) for e in events] == [("rig", STATIC)] + [("frames", f.t_ns - imu.cam_time_offset_ns) for f in sets]
    assert all(e.value is f for e, f in zip(events[1:], sets, strict=True))


@settings(max_examples=25, deadline=None)
@given(
    position=st.lists(st.floats(-100.0, 100.0), min_size=3, max_size=3),
    quaternion=st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4).filter(lambda q: sum(x * x for x in q) > 1e-3),
)
def test_a_rig_pose_round_trips_serde_and_maps_to_one_transform_at_its_root(position: list[float], quaternion: list[float]) -> None:
    pose = RigPose(np.asarray(position), np.asarray(quaternion) / np.linalg.norm(quaternion))
    back: RigPose = from_dict(RigPose, to_dict(pose))
    np.testing.assert_array_equal(back.position_m, pose.position_m)
    np.testing.assert_array_equal(back.quaternion_wxyz, pose.quaternion_wxyz)
    (log,) = pose.to_rerun("/world/rig_00")
    assert log.path == "/world/rig_00" and not log.static and isinstance(log.data, rr.Transform3D)
    assert RigPose.contents("/world/rig_00") == ["/world/rig_00"]



@settings(max_examples=40, deadline=None)
@given(
    times=st.lists(st.integers(0, 10**12), min_size=1, max_size=30, unique=True),
    seed=st.integers(0, 2**16),
    broken=st.lists(st.integers(0, 29), max_size=5),
)
def test_the_pose_table_reads_the_layers_rows_whole_sorted_and_finite(times: list[int], seed: int, broken: list[int]) -> None:
    rng = np.random.default_rng(seed)
    quaternion: Float64[ndarray, "n 4"] = Rotation.random(len(times), random_state=seed).as_quat().astype(np.float32).astype(np.float64)
    translation: Float64[ndarray, "n 3"] = rng.uniform(-5.0, 5.0, (len(times), 3)).astype(np.float32).astype(np.float64)
    bad: set[int] = {b for b in broken if b < len(times)}
    nan: Float64[ndarray, " 3"] = np.array([np.nan, 0.0, 0.0])
    values: list[tuple[int, RigPose]] = [(t, RigPose(nan if i in bad else translation[i], np.roll(quaternion[i], 1))) for i, t in enumerate(times)]
    keep: list[int] = sorted((i for i in range(len(times)) if i not in bad), key=lambda i: times[i])
    rows: pa.Table = layer_table(values, RIG_ENTITY)  # sorted by time: shuffle, so that the reader has to sort
    read: list[tuple[int, CatalogPoses]] = list(CatalogPoses.from_layer(rows.take(rng.permutation(rows.num_rows)), RIG_ENTITY, "video_time"))
    if not keep:
        assert read == []
        return
    ((t_ns, poses),) = read
    assert t_ns == STATIC and poses.t_ns.tolist() == [times[i] for i in keep]
    np.testing.assert_allclose(poses.world_from_rig[:, :3, 3], translation[keep])
    np.testing.assert_allclose(poses.world_from_rig[:, :3, :3], Rotation.from_quat(quaternion[keep]).as_matrix(), atol=1e-12)
    back: CatalogPoses = from_dict(CatalogPoses, to_dict(poses))
    np.testing.assert_array_equal(back.world_from_rig, poses.world_from_rig)
    assert back.t_ns.dtype == np.int64


def test_the_pose_table_refuses_a_translation_without_its_rotation_as_rig_pose_does_and_is_never_written() -> None:
    rows: pa.Table = layer_table([(t, RigPose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))) for t in (10, 20)], RIG_ENTITY)
    quaternion: str = f"{RIG_ENTITY}:Transform3D:quaternion"
    rows = rows.set_column(rows.column_names.index(quaternion), quaternion, pa.array([None, rows[quaternion][1].as_py()], rows[quaternion].type))
    for reader in (CatalogPoses, RigPose):
        with pytest.raises(ValueError, match="a translation without a quaternion"):
            list(reader.from_layer(rows, RIG_ENTITY, "video_time"))
    with pytest.raises(NotImplementedError, match="slam_rs"):
        CatalogPoses(np.zeros(0, np.int64), np.zeros((0, 4, 4))).to_rerun(RIG_ENTITY)

def test_a_dataset_takes_its_rig_profile_and_its_config_from_one_place() -> None:
    settings = load_slam_config()
    assert VioParams("fast").for_dataset("robocap") == (ROBOCAP_RIG, settings.robocap_config_text("fast"))
    assert VioParams("reference").for_dataset("msd-index") == (MSD_RIG, settings.vio_config_text("msd-index", "reference"))


def test_the_decoder_follows_the_device_choice() -> None:
    assert VioParams(decode_device="cpu").decoder() == "cpu"
    torch = pytest.importorskip("torch")
    assert VioParams(decode_device="auto").decoder() == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("suffix", ["partial", "rrd"])
def test_pre_scores_itself_with_one_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str) -> None:
    pre: Path = tmp_path / f"short.pre.{suffix}"
    pre.touch()
    poses: Trajectory = Trajectory(np.array([1, 2, 3], dtype=np.int64), np.eye(3), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))
    reads: list[Path] = []

    def read(path: Path) -> tuple[Trajectory, float]:
        reads.append(path)
        return poses, 0.0

    monkeypatch.setattr(verify, "_read", read)
    scores: dict[str, float] = verify.score(Case("short", "short", "pre-port"), pre)
    assert reads == [pre]
    assert scores == pytest.approx({"ate_vs_pre_cm": 0.0, "max_pos_vs_pre_cm": 0.0, "poses_delta_pct": 0.0, "pose_count": 3.0})
