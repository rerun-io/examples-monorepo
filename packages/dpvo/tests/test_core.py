"""DPVO core on CUDA with the default hooks: plain DPVO's RNG stream and the observer callbacks."""

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")

from lietorch_mojo import SE3  # noqa: E402
from torch import Tensor  # noqa: E402

from dpvo.config import DPVOConfig  # noqa: E402
from dpvo.dpvo import DPVO  # noqa: E402
from dpvo.hooks import DPVOObserver, FrameEvent, add_global_observer  # noqa: E402
from dpvo.net import VONet  # noqa: E402

pytestmark = pytest.mark.integration

HT, WD = 64, 96
M = 8


@pytest.fixture(scope="module")
def network() -> VONet:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for the DPVO kernels")
    torch.manual_seed(0)
    return VONet()


def frame(seed: int) -> Tensor:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, HT, WD), generator=generator, dtype=torch.uint8).cuda()


INTRINSICS: Tensor = torch.tensor([80.0, 80.0, 48.0, 32.0])


@pytest.mark.parametrize("gradient_bias", [False, True])
def test_first_frame_keeps_plain_rng_stream(network: VONet, gradient_bias: bool) -> None:
    tracker: DPVO = DPVO(DPVOConfig(buffer_size=32, patches_per_frame=M, gradient_bias=gradient_bias), network, ht=HT, wd=WD)
    with torch.no_grad():
        torch.manual_seed(3)
        tracker(0, frame(0), INTRINSICS.cuda())
        after: Tensor = torch.rand(1, device="cuda")

        # Pre-change order (main 347c62081): patchify draws x then y, then rand_like for the M depths.
        torch.manual_seed(3)
        width: int = 3 * M if gradient_bias else M
        x: Tensor = torch.randint(1, WD // 4 - 1, size=[1, width], device="cuda")
        y: Tensor = torch.randint(1, HT // 4 - 1, size=[1, width], device="cuda")
        depth: Tensor = torch.rand(1, M, 1, 1, device="cuda")
        expected_after: Tensor = torch.rand(1, device="cuda")

    patches: Tensor = tracker.pg.patches_[0]
    assert torch.equal(after, expected_after)
    assert torch.equal(patches[:, 2], depth[0].expand(M, 3, 3))
    if not gradient_bias:
        assert torch.equal(patches[:, :2, 1, 1], torch.stack([x[0], y[0]], dim=-1).float())


class EventLog(DPVOObserver):
    """Counts every callback."""

    def __init__(self) -> None:
        self.frames: list[FrameEvent] = []
        self.global_bas: int = 0
        self.culls: list[int] = []
        self.terminated: int = 0
        self.refined: int = 0

    def on_frame(self, dpvo: DPVO, event: FrameEvent) -> None:
        self.frames.append(event)

    def on_cull(self, dpvo: DPVO, k: int) -> None:
        self.culls.append(k)

    def on_global_ba(self, dpvo: DPVO) -> None:
        self.global_bas += 1

    def on_terminate(self, dpvo: DPVO) -> None:
        self.terminated += 1

    def refine_trajectory(self, dpvo: DPVO, poses: SE3) -> SE3:
        self.refined += 1
        return poses


def test_observers_see_every_frame_and_terminate(network: VONet, monkeypatch: pytest.MonkeyPatch) -> None:
    own: EventLog = EventLog()
    global_log: EventLog = EventLog()
    remove = add_global_observer(lambda dpvo: global_log)
    try:
        tracker: DPVO = DPVO(DPVOConfig(buffer_size=64, patches_per_frame=M, gradient_bias=False), network, ht=HT, wd=WD, observers=[own])
    finally:
        remove()
    probes: list[float] = [0.0, 5.0]  # second frame is rejected by the motion probe, the rest pass
    monkeypatch.setattr(tracker, "motion_probe", lambda: torch.tensor(probes.pop(0) if probes else 5.0))

    torch.manual_seed(0)
    with torch.no_grad():
        for t in range(12):
            tracker(t, frame(t), INTRINSICS.cuda())
        tracker.run_global_ba(t0=1)
        poses, tstamps = tracker.terminate()

    for log in (own, global_log):
        assert [event.tstamp for event in log.frames] == list(range(12))
        assert [event.frame_index for event in log.frames] == list(range(12))
        assert [event.keyframe_added for event in log.frames] == [True, False] + [True] * 10
        assert log.frames[2].keyframes_before == 1
        assert log.global_bas >= 1
        assert (log.terminated, log.refined) == (1, 1)
    assert tracker.is_initialized
    assert poses.shape == (12, 7) and tstamps.shape == (12,)
    assert bool(torch.isfinite(torch.from_numpy(poses)).all())
