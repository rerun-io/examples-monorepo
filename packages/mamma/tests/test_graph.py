"""The capture graph with fake models: it validates, and both runners deliver the same outputs (lossless; no GPU, no catalog)."""

from collections.abc import Mapping

import numpy as np
from dagkit.graph import Graph
from dagkit.node import Emit, Fenced, PortType, RunContext, Tick
from dagkit.runners import run_sequential, run_threaded
from dagkit.sources import IterSource
from mamma_fakes import FakeDetector, FakeEncoder, FakeFitter, FakeLandmarker, FakeMasks, FakeScene, host
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras

from mamma.apis.capture import PORTS
from mamma.graphs.capture import capture_graph
from mamma.nodes.fit import FitNode
from mamma.nodes.landmarks import LandmarksNode
from mamma.nodes.track import TrackNode, TrackParams
from mamma.ops.window_fit import FitterConfig


class Keep:
    """Sink keeping a host copy of every value it receives."""

    outputs: Mapping[str, PortType] = {}

    def __init__(self) -> None:
        self.inputs: dict[str, PortType] = dict(PORTS)
        self.items: list[tuple[str, int, object]] = []

    def reset(self, ctx: RunContext) -> None:
        self.items = []

    def step(self, tick: Tick, out: Emit) -> None:
        self.items.append((tick.port, tick.t_ns, host(tick.value)))

    def flush(self, out: Emit) -> None:
        pass


def graph(scene: FakeScene) -> tuple[Graph, Keep]:
    video = IterSource({"frames": Fenced[MultiViewFrames], "rig": RigCameras}, lambda: scene.events(16))
    keep = Keep()
    nodes = (
        TrackNode(TrackParams(expected_subjects=1), FakeDetector(), FakeEncoder(), FakeMasks()),  # type: ignore[arg-type]
        LandmarksNode(FakeLandmarker(scene)),
        FitNode(FitterConfig(), np.zeros((2, 3), dtype=np.int32), lambda K, T: FakeFitter()),  # type: ignore[arg-type,return-value]
    )
    return capture_graph(video, *nodes, {"keep": keep}), keep


def test_capture_graph_validates_and_the_runners_agree() -> None:
    scene = FakeScene()
    built, keep = graph(scene)
    built.validate()
    assert run_sequential(built).status == "complete"
    sequential = sorted(keep.items, key=lambda item: (item[0], item[1]))
    built, keep = graph(scene)
    assert run_threaded(built, mode="lossless").status == "complete"
    threaded = sorted(keep.items, key=lambda item: (item[0], item[1]))
    np.testing.assert_equal(threaded, sequential)
    counts: dict[str, int] = {port: sum(p == port for p, _, _ in sequential) for port in PORTS}
    assert counts == {"tracks": 16, "landmarks": 16, "fit": 16, "kp3d": 16, "topology": 1}
