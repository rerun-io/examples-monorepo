"""The capture graph with fake models: it validates, and both runners deliver the same outputs (lossless; no GPU, no catalog)."""

import time

import numpy as np
import pytest
from dagkit.graph import Graph
from dagkit.node import Fenced
from dagkit.runners import run_sequential, run_threaded
from dagkit.sources import IterSource
from dagkit.testing import Collect
from mamma_fakes import FakeDetector, FakeEncoder, FakeFitter, FakeLandmarker, FakeMasks, FakeScene
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras

from mamma.apis.capture import PORTS
from mamma.graphs.capture import TRACKS_QUEUE, capture_graph
from mamma.models.landmarks.estimator import DenseLandmarks
from mamma.nodes.fit import FitNode
from mamma.nodes.landmarks import LandmarksNode
from mamma.nodes.track import TrackNode, TrackParams
from mamma.ops.window_fit import FitterConfig


class SlowLandmarker(FakeLandmarker):
    """Landmarks slower than decode and tracking: in realtime the source runs as far ahead as the queues let it."""

    def __call__(self, *args, **kwargs) -> DenseLandmarks:  # type: ignore[no-untyped-def]
        time.sleep(0.01)
        return super().__call__(*args, **kwargs)


def graph(scene: FakeScene, landmarker: FakeLandmarker | None = None) -> tuple[Graph, Collect]:
    video = IterSource({"frames": Fenced[MultiViewFrames], "rig": RigCameras}, lambda: scene.events(16))
    keep = Collect(PORTS)
    nodes = (
        TrackNode(TrackParams(expected_subjects=1), FakeDetector(), FakeEncoder(), FakeMasks()),  # type: ignore[arg-type]
        LandmarksNode(landmarker or FakeLandmarker(scene)),
        FitNode(FitterConfig(emit_stride=1), np.zeros((2, 3), dtype=np.int32), lambda K, T: FakeFitter()),  # type: ignore[arg-type,return-value]
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
    assert counts == {"rig": 1, "tracks": 16, "landmarks": 16, "fit": 16, "kp3d": 16, "topology": 1, "view": 16, "params": 16, "health": 16}


@pytest.mark.parametrize("tracks_queue", [TRACKS_QUEUE, 8])
def test_realtime_crops_every_tick_from_its_own_frameset_whatever_the_queue_size(monkeypatch, tracks_queue: int) -> None:  # type: ignore[no-untyped-def]
    """Realtime keeps Block edges (no drops here) and the crop frames travel with the tracks, so a slow landmarks node crops
    every tick from its own frameset however far ahead a longer tracks queue lets the source run."""
    scene = FakeScene()
    built, keep = graph(scene)
    assert run_sequential(built).status == "complete"
    sequential = sorted(keep.items, key=lambda item: (item[0], item[1]))
    monkeypatch.setattr("mamma.graphs.capture.TRACKS_QUEUE", tracks_queue)
    built, keep = graph(scene, SlowLandmarker(scene))
    assert run_threaded(built, mode="realtime").status == "complete"
    np.testing.assert_equal(sorted(keep.items, key=lambda item: (item[0], item[1])), sequential)
