"""Node contracts with fake models on a synthetic three-camera ring (no GPU, no catalog, no weights)."""

from collections.abc import Mapping

import numpy as np
from dagkit.node import Emit, Fenced, PortType, RunContext, Tick
from dagkit.testing import check_node_contract
from dagkit.time import TimeNs
from mamma_fakes import BODY, FakeDetector, FakeEncoder, FakeFitter, FakeLandmarker, FakeMasks, FakeScene, host

from mamma.nodes.fit import FitNode
from mamma.nodes.landmarks import LandmarksNode
from mamma.nodes.track import TrackNode, TrackParams
from mamma.ops.window_fit import FitterConfig
from mamma.types import BodyTopology, Landmarks3d, MultiViewLandmarks, MultiViewTracks, SmplxFits

FACES: np.ndarray = np.zeros((2, 3), dtype=np.int32)


class HostEmit:
    def __init__(self, out: Emit) -> None:
        self.out = out

    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        self.out.put(port, t_ns, host(value))

    def watermark(self, t_ns: TimeNs) -> None:
        self.out.watermark(t_ns)


class Hosted:
    """A node whose outputs hold tensors, putting host copies: dagkit's contract check compares outputs by value."""

    def __init__(self, node: TrackNode | LandmarksNode | FitNode) -> None:
        self.node = node
        self.inputs: Mapping[str, PortType] = node.inputs
        self.outputs: Mapping[str, PortType] = node.outputs

    def reset(self, ctx: RunContext) -> None:
        self.node.reset(ctx)

    def step(self, tick: Tick, out: Emit) -> None:
        self.node.step(tick, HostEmit(out))

    def flush(self, out: Emit) -> None:
        self.node.flush(HostEmit(out))


def track_outputs(scene: FakeScene) -> list[tuple[str, int, object]]:
    ticks = [Tick(e.t_ns, "frames", e.value, {"rig": scene.rig}) for e in list(scene.events(12))[1:]]
    node = TrackNode(TrackParams(expected_subjects=1), FakeDetector(), FakeEncoder(), FakeMasks())  # type: ignore[arg-type]
    raw: list[tuple[str, int, object]] = []

    class Keep:
        def put(self, port: str, t_ns: int, value: object) -> None:
            raw.append((port, t_ns, value))

        def watermark(self, t_ns: int) -> None:
            pass

    node.reset(RunContext("mamma-capture", "track", "lossless"))
    for tick in ticks:
        node.step(tick, Keep())
    assert check_node_contract(Hosted(node), ticks)  # reset idempotent, outputs monotonic, flush drains
    return raw


def test_track_bootstraps_one_person_and_tracks_it_in_every_view() -> None:
    scene = FakeScene()
    outputs = track_outputs(scene)
    assert [t for _, t, _ in outputs] == [scene.frames(i).t_ns for i in range(12)]
    for _, _, value in outputs:
        assert isinstance(value, Fenced) and isinstance(value.value, MultiViewTracks)
        tracks: MultiViewTracks = value.value
        assert list(zip(tracks.view.tolist(), tracks.person.tolist(), strict=True)) == [(0, 0), (1, 0), (2, 0)]  # one row per (view, person)
        np.testing.assert_array_equal(tracks.boxes_xyxy, [[20.0, 8.0, 43.0, 43.0]] * 3)
        np.testing.assert_array_equal(tracks.scale_xy, 2.0)


def test_landmarks_follow_the_tracks_and_keep_engine_pixels() -> None:
    scene = FakeScene()
    frames = {f.frame_index: f for f in map(scene.frames, range(12))}
    ticks = [Tick(t, "tracks", value, {"frames": Fenced(frames[value.value.frame_index], None)}) for _, t, value in track_outputs(scene)]  # type: ignore[attr-defined]
    outputs = check_node_contract(Hosted(LandmarksNode(FakeLandmarker(scene))), ticks)
    assert [t for _, t, _ in outputs] == [tick.t_ns for tick in ticks]
    for _, _, value in outputs:
        assert isinstance(value, dict) and value["person"].tolist() == [0, 0, 0]
        np.testing.assert_allclose(value["xy"][1], scene.pixels(1), rtol=1e-6)


def landmark_ticks(scene: FakeScene, count: int, persons: tuple[int, ...] = (0,)) -> list[Tick]:
    ticks: list[Tick] = []
    for index in range(count):
        view = np.repeat(np.arange(3), len(persons))
        dense = FakeLandmarker(scene)(scene.frames(index).rgb, view, None, None)  # type: ignore[arg-type]
        value = MultiViewLandmarks(
            index, scene.rig.names, view, np.tile(np.asarray(persons), 3), dense.xy, dense.log_variance, dense.visibility, dense.floor_contact,
            np.full((3, 2), 2.0, dtype=np.float32),
        )  # fmt: skip
        ticks.append(Tick(scene.frames(index).t_ns, "landmarks", Fenced(value, None), {"rig": scene.rig}))
    return ticks


def test_fit_emits_every_frame_once_in_time_order_and_flush_drains_the_window() -> None:
    scene = FakeScene()
    ticks = landmark_ticks(scene, 20)
    node = FitNode(FitterConfig(), FACES, lambda K, T: FakeFitter())  # type: ignore[arg-type,return-value]
    outputs = check_node_contract(node, ticks)
    fits = [(t, v) for p, t, v in outputs if p == "fit"]
    assert [v.frame_index for _, v in fits] == list(range(20)) and [t for t, _ in fits] == [tick.t_ns for tick in ticks]  # type: ignore[attr-defined]
    for _, fit in fits:
        assert isinstance(fit, SmplxFits) and fit.persons == (0,)
        np.testing.assert_allclose(fit.vertices[0, 0], BODY.mean(axis=0), atol=1e-3)  # the gated triangulation found the body
    topologies = [v for p, _, v in outputs if p == "topology"]
    assert len(topologies) == 1 and isinstance(topologies[0], BodyTopology) and topologies[0].persons == (0,)
    clouds = [v for p, _, v in outputs if p == "kp3d"]
    assert len(clouds) == 20 and all(isinstance(c, Landmarks3d) and c.valid.all() for c in clouds)


def test_fit_heads_of_a_person_who_starts_late_still_come_out_in_order() -> None:
    scene = FakeScene()
    ticks = landmark_ticks(scene, 6) + landmark_ticks(scene, 20, (0, 1))[6:]  # person 1 joins at frame 6
    node = FitNode(FitterConfig(), FACES, lambda K, T: FakeFitter())  # type: ignore[arg-type,return-value]
    outputs = check_node_contract(node, ticks)
    persons: dict[int, list[int]] = {0: [], 1: []}
    for port, _, value in outputs:
        if port == "fit":
            for person in value.persons:  # type: ignore[attr-defined]
                persons[person].append(value.frame_index)  # type: ignore[attr-defined]
    assert persons == {0: list(range(20)), 1: list(range(6, 20))}
    assert [v.persons for p, _, v in outputs if p == "topology"] == [(0,), (1,)]  # type: ignore[attr-defined]
