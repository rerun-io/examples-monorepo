"""Node tier (decision 18): frontend and backend contracts with fake models and kernels (CPU only, no weights)."""

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from dagkit.node import RunContext, Tick
from dagkit.testing import Recorder, check_node_contract
from mast3r.model import AsymmetricMASt3R
from simplecv.types.frames import MultiViewFrames

from mast3r_slam.nodes import backend as backend_module
from mast3r_slam.nodes import frontend as frontend_module
from mast3r_slam.nodes.backend import Backend
from mast3r_slam.nodes.frontend import Frontend
from mast3r_slam.nodes.shared import SharedSlam
from mast3r_slam.nodes.tracking import FrameTracker
from mast3r_slam.types import Frame, KeyframeAdded, KeyframeCloud, MapUpdate, Mode, RelocRequest, SlamConfig, TrackedFrame

SIZE: int = 224
PATCHES: int = SIZE * SIZE // 256
CFG = SlamConfig(img_size=224, keyframe_buffer=8)
MODEL: AsymmetricMASt3R = AsymmetricMASt3R.__new__(AsymmetricMASt3R)
"""Stands in for the weights: the fakes below replace every model call."""


@pytest.fixture
def shared() -> Iterator[SharedSlam]:
    resource = SharedSlam(SIZE, SIZE, CFG.keyframe_buffer, use_calib=False, device="cpu")
    yield resource
    resource.close()


def observe(frame: Frame) -> None:
    """What MASt3R would leave on a frame: features and a point map (a fronto-parallel plane)."""
    frame.feat, frame.pos = torch.zeros(1, PATCHES, 1024), torch.zeros(1, PATCHES, 2, dtype=torch.long)
    frame.update_pointmap(torch.ones(SIZE * SIZE, 3), torch.full((SIZE * SIZE, 1), 10.0), CFG.tracking)  # confident


def fake_mono(model: object, frame: Frame) -> tuple[torch.Tensor, torch.Tensor]:
    frame.feat, frame.pos = torch.zeros(1, PATCHES, 1024), torch.zeros(1, PATCHES, 2, dtype=torch.long)
    return torch.ones(SIZE * SIZE, 3), torch.ones(SIZE * SIZE, 1)


def fake_track(self: FrameTracker, frame: Frame) -> tuple[bool, bool]:
    """Every 3rd frame becomes a keyframe; frame 7 loses tracking."""
    observe(frame)
    return frame.frame_id % 3 == 2, frame.frame_id == 7


def frame_ticks(n: int) -> list[Tick]:
    rng = np.random.default_rng(0)
    rgb = [torch.from_numpy(rng.integers(0, 255, (1, 3, 30, 40), dtype=np.uint8)) for _ in range(n)]
    return [Tick(i * 1000, "frames", MultiViewFrames(i * 1000, i, ("cam",), rgb[i], torch.tensor([i * 1000])), {}) for i in range(n)]


def test_frontend_contract_and_mode_flow(shared: SharedSlam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontend_module, "mast3r_inference_mono", fake_mono)
    monkeypatch.setattr(FrameTracker, "track", fake_track)
    ctx = RunContext("slam", "frontend", "realtime", resources={"mast3r": MODEL, "shared": shared})
    outputs = check_node_contract(Frontend(CFG, device="cpu"), frame_ticks(10), ctx)
    tasks = [(t, v) for port, t, v in outputs if port == "task"]
    tracked = [v.frame_index for port, _, v in outputs if port == "tracked" and isinstance(v, TrackedFrame)]
    assert tasks[:3] == [(0, KeyframeAdded(0)), (2000, KeyframeAdded(1)), (5000, KeyframeAdded(2))]
    # Frame 7 is lost; frame 8 asks for relocalisation; nobody answers, so frame 9 leaves the request's shared frame alone.
    assert tasks[3:] == [(8000, RelocRequest(8))]
    assert tracked == [0, 1, 2, 3, 4, 5, 6]
    assert shared.states.get_mode() == Mode.TERMINATED


def test_realtime_frontend_keeps_the_requested_frame_until_the_backend_answers(shared: SharedSlam, monkeypatch: pytest.MonkeyPatch) -> None:
    """Realtime: lost frames that arrive while a relocalisation request is queued neither overwrite the one shared current
    frame nor queue another request, so the backend relocalises exactly the frame it was asked for."""
    monkeypatch.setattr(frontend_module, "mast3r_inference_mono", fake_mono)
    monkeypatch.setattr(FrameTracker, "track", fake_track)
    monkeypatch.setattr(backend_module, "FactorGraph", FakeGraph)
    monkeypatch.setattr(backend_module, "RetrievalDatabase", FakeRetrieval)
    resources = {"mast3r": MODEL, "shared": shared}
    frontend, backend, out = Frontend(CFG, device="cpu"), Backend(CFG), Recorder()
    frontend.reset(RunContext("slam", "frontend", "realtime", resources=resources))
    backend.reset(RunContext("slam", "backend", "realtime", resources=resources))
    for tick in frame_ticks(11):  # frame 7 is lost, frame 8 asks; frames 9 and 10 arrive before the backend answers
        frontend.step(tick, out)
    assert shared.states.get_frame().frame_id == 8
    tasks = [v for port, _, v in out.items if port == "task"]
    assert [v for v in tasks if isinstance(v, RelocRequest)] == [RelocRequest(8)]
    for t, task in [(t, v) for port, t, v in out.items if port == "task"]:
        backend.step(Tick(t, "task", task, {}), Recorder())
    assert shared.backend_idle() and shared.states.get_mode() == Mode.TRACKING


def test_frontend_refuses_a_mid_run_reset(shared: SharedSlam) -> None:
    """The backend's final poses and the Nerfstudio export read the one shared map at flush: a ``Control.reset`` (a new
    world) would clear it under them, so it fails the run instead."""
    with pytest.raises(NotImplementedError, match="reset"):
        Frontend(CFG, device="cpu").reset(RunContext("slam", "frontend", "lossless", resources={"mast3r": MODEL, "shared": shared}, world=1))


def test_frontend_reset_returns_to_init(shared: SharedSlam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontend_module, "mast3r_inference_mono", fake_mono)
    frontend = Frontend(CFG, device="cpu")
    ctx = RunContext("slam", "frontend", "realtime", resources={"mast3r": MODEL, "shared": shared})
    frontend.reset(ctx)
    frontend.step(frame_ticks(1)[0], Recorder())
    assert shared.states.get_mode() == Mode.TRACKING and len(shared.keyframes) == 1
    frontend.reset(ctx)
    assert shared.states.get_mode() == Mode.INIT and len(shared.keyframes) == 0 and shared.backend_idle()


def test_lossless_frontend_waits_for_the_backend_and_abort_ends_the_wait(shared: SharedSlam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontend_module, "mast3r_inference_mono", fake_mono)
    frontend = Frontend(CFG, device="cpu")
    frontend.reset(RunContext("slam", "frontend", "lossless", resources={"mast3r": MODEL, "shared": shared}))
    out = Recorder()
    frontend.step(frame_ticks(1)[0], out)  # INIT queues keyframe 0 for the backend

    def acknowledge() -> None:
        time.sleep(0.3)
        with shared.states.lock:
            shared.states.global_optimizer_tasks.pop(0)

    start = time.perf_counter()
    threading.Thread(target=acknowledge).start()
    frontend.flush(out)
    assert time.perf_counter() - start >= 0.3  # flushed only after the acknowledgement
    assert [port for port, _, _ in out.items] == ["task", "tracked"]

    frontend.reset(RunContext("slam", "frontend", "lossless", resources={"mast3r": MODEL, "shared": shared}))
    frontend.step(frame_ticks(1)[0], Recorder())
    threading.Timer(0.2, frontend.abort).start()  # the runner aborts a failed run: the wait must end, not hang
    with pytest.raises(RuntimeError, match="waited for the backend"):
        frontend.flush(Recorder())


class FakeGraph:
    def __init__(self, *_: object, **__: object) -> None:
        self.ii, self.jj = torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

    def add_factors(self, ii: list[int], jj: list[int], min_match_frac: float, is_reloc: bool = False) -> bool:
        self.ii, self.jj = torch.cat([self.ii, torch.tensor(ii)]), torch.cat([self.jj, torch.tensor(jj)])
        return True

    def solve_GN_rays(self) -> None:
        pass


class FakeRetrieval:
    def __init__(self, *_: object, **__: object) -> None:
        self.size = 0

    def update(self, frame: Frame, add_after_query: bool, k: int, min_thresh: float = 0.0) -> list[int]:
        found = [0] if self.size > 1 else []
        self.size += add_after_query
        return found


def test_backend_contract(shared: SharedSlam, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module, "FactorGraph", FakeGraph)
    monkeypatch.setattr(backend_module, "RetrievalDatabase", FakeRetrieval)

    def ticks() -> list[Tick]:  # three keyframes, then a relocalisation request; the frontend's queue mirrors the edge
        shared.reset()
        for i in range(3):
            frame = Frame(10 * i, torch.zeros(1, 3, SIZE, SIZE), torch.tensor([[SIZE, SIZE]]), torch.tensor([[SIZE, SIZE]]), torch.zeros(SIZE, SIZE, 3))
            observe(frame)
            shared.keyframes.append(frame)
            shared.states.queue_global_optimization(i)
        shared.states.set_frame(frame)
        shared.states.set_mode(Mode.RELOC)
        shared.states.queue_reloc()
        return [Tick(1000 * i, "task", KeyframeAdded(i), {}) for i in range(3)] + [Tick(3000, "task", RelocRequest(20), {})]

    backend = Backend(CFG)
    ctx = RunContext("slam", "backend", "lossless", resources={"mast3r": MODEL, "shared": shared})
    backend.reset(ctx)
    out = Recorder()
    for tick in ticks():
        backend.step(tick, out)
    assert shared.backend_idle() and shared.states.get_mode() == Mode.TRACKING  # every task acknowledged, relocalised
    backend.flush(out)
    maps = [v for port, _, v in out.items if port == "map" and isinstance(v, MapUpdate)]
    assert [len(m.world_sim3_cam) for m in maps] == [3, 3, 3, 4]  # the buffer already holds keyframes 0-2; reloc adds 3
    assert np.stack([maps[-1].ii, maps[-1].jj], axis=1).tolist() == [[0, 1], [0, 2], [1, 2], [3, 0]]
    finals = [(t, v.frame_index) for port, t, v in out.items if port == "keyframes" and isinstance(v, TrackedFrame)]
    assert finals == [(0, 0), (1000, 10), (2000, 20), (3000, 20)]  # the relocalised frame is keyframe 3, at its own time
    clouds = [v for port, _, v in out.items if port == "cloud" and isinstance(v, KeyframeCloud)]
    assert [c.index for c in clouds] == [0, 1, 2] and all(len(c.points) == SIZE * SIZE for c in clouds)  # new keyframes only
    backend.flush(out)
    assert len(out.items) == len(maps) + len(finals) + len(clouds)  # a second flush emits nothing

    backend.reset(ctx)
    for tick in ticks():
        backend.step(tick, Recorder())
        shared.states.set_mode(Mode.TERMINATED)  # realtime stop: what is still queued is dropped and counted
    assert backend.dropped == 3 and shared.backend_idle()


def test_backend_refuses_to_relocalise_a_frame_it_was_not_asked_for(shared: SharedSlam, monkeypatch: pytest.MonkeyPatch) -> None:
    """The current frame crosses processes through shared memory; a stale one (the writes of the request's frame not yet
    visible) must fail loudly instead of relocalising the previous frame at the request's time."""
    monkeypatch.setattr(backend_module, "FactorGraph", FakeGraph)
    monkeypatch.setattr(backend_module, "RetrievalDatabase", FakeRetrieval)
    shared.reset()
    frame = Frame(20, torch.zeros(1, 3, SIZE, SIZE), torch.tensor([[SIZE, SIZE]]), torch.tensor([[SIZE, SIZE]]), torch.zeros(SIZE, SIZE, 3))
    observe(frame)
    shared.states.set_frame(frame)
    shared.states.set_mode(Mode.RELOC)
    shared.states.queue_reloc()
    backend = Backend(CFG)
    backend.reset(RunContext("slam", "backend", "lossless", resources={"mast3r": MODEL, "shared": shared}))
    with pytest.raises(RuntimeError, match="frame 20, .* frame 22"):
        backend.step(Tick(3000, "task", RelocRequest(22), {}), Recorder())


def test_a_realtime_stop_during_relocalisation_stays_terminated(shared: SharedSlam, monkeypatch: pytest.MonkeyPatch) -> None:
    """The frontend's flush sets TERMINATED while the backend relocalises; the backend's success must not undo the stop."""
    monkeypatch.setattr(backend_module, "FactorGraph", FakeGraph)
    monkeypatch.setattr(backend_module, "RetrievalDatabase", FakeRetrieval)
    shared.reset()
    frame = Frame(20, torch.zeros(1, 3, SIZE, SIZE), torch.tensor([[SIZE, SIZE]]), torch.tensor([[SIZE, SIZE]]), torch.zeros(SIZE, SIZE, 3))
    observe(frame)
    shared.states.set_frame(frame)
    shared.states.set_mode(Mode.RELOC)
    shared.states.queue_reloc()
    backend = Backend(CFG)
    backend.reset(RunContext("slam", "backend", "realtime", resources={"mast3r": MODEL, "shared": shared}))

    def stopped_while_relocalising(frame: Frame) -> bool:
        shared.states.set_mode(Mode.TERMINATED)
        return True

    monkeypatch.setattr(backend, "_relocalise", stopped_while_relocalising)
    backend.step(Tick(3000, "task", RelocRequest(20), {}), Recorder())
    assert shared.states.get_mode() == Mode.TERMINATED and shared.backend_idle()


def test_calibrated_tracking_measures_every_keyframe_pixel_at_its_depth(shared: SharedSlam) -> None:
    # The calibrated path under beartype (dev env): both point maps go onto their pixel rays, and each keyframe pixel is
    # measured as (u, v, log z) in row-major order.
    tracker = FrameTracker(MODEL, shared.keyframes, CFG.tracking, CFG.matching, use_calib=True)
    frame, keyframe = (Frame(i, torch.zeros(1, 3, SIZE, SIZE), torch.tensor([[SIZE, SIZE]]), torch.tensor([[SIZE, SIZE]]), torch.zeros(SIZE, SIZE, 3)) for i in (1, 0))
    observe(frame)
    observe(keyframe)  # every point at depth 1
    K = torch.tensor([[200.0, 0.0, 112.0], [0.0, 200.0, 112.0], [0.0, 0.0, 1.0]])

    *_, meas_k, valid_meas_k = tracker.get_points_poses(frame, keyframe, torch.arange(SIZE * SIZE), (SIZE, SIZE), True, K)

    pixel = torch.arange(SIZE * SIZE)
    assert meas_k is not None and valid_meas_k is not None and bool(valid_meas_k.all())
    torch.testing.assert_close(meas_k, torch.stack(((pixel % SIZE).float(), (pixel // SIZE).float(), torch.zeros(SIZE * SIZE)), dim=-1))
