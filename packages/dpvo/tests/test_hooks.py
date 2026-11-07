"""Hook wiring of the DPVO core on a CPU tracker (random network weights, no kernels, no GPU)."""

from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")

from jaxtyping import Float32, Int64  # noqa: E402
from torch import Tensor  # noqa: E402

from dpvo.config import DPVOConfig  # noqa: E402
from dpvo.dpvo import DPVO  # noqa: E402
from dpvo.hooks import DefaultGraphPolicy, DPVOObserver, FrameDepth, RandomDepthInit, add_global_observer  # noqa: E402
from dpvo.net import VONet  # noqa: E402

CPU = torch.device("cpu")
M = 4


@pytest.fixture(scope="module")
def network() -> VONet:
    torch.manual_seed(0)
    return VONet()


def make_tracker(network: VONet, **kwargs: object) -> DPVO:
    cfg: DPVOConfig = DPVOConfig(buffer_size=24, patches_per_frame=M, gradient_bias=False)
    return DPVO(cfg, network, ht=32, wd=32, device=CPU, **kwargs)  # pyrefly: ignore[bad-argument-type]


class Recorder(DPVOObserver):
    """Records the calls it sees."""

    def __init__(self, tag: str = "") -> None:
        self.tag = tag
        self.culls: list[tuple[int, float]] = []

    def on_cull(self, dpvo: DPVO, k: int) -> None:
        # value of the first frame attribute at the culled slot, before compaction
        self.culls.append((k, float(dpvo.pg.frame_attrs["prior"][k, 0])))


class RecordingPolicy(DefaultGraphPolicy):
    """Graph policy with a switchable veto that records its calls."""

    def __init__(self, keep: bool) -> None:
        self.keep = keep
        self.calls: list[tuple[str, int, bool]] = []

    def force_keep(self, dpvo: DPVO) -> bool:
        self.calls.append(("force_keep", dpvo.n, self.keep))
        return self.keep

    def on_cull(self, dpvo: DPVO, k: int) -> None:
        self.calls.append(("on_cull", k, False))

    def on_keep(self, dpvo: DPVO, k: int, forced: bool) -> None:
        self.calls.append(("on_keep", k, forced))


# ── global observers ────────────────────────────────────────────────────


@pytest.fixture
def cleanup() -> Iterator[list]:
    removers: list = []
    yield removers
    for remove in removers:
        remove()


def test_global_observer_factories_attach_after_own_observers(network: VONet, cleanup: list) -> None:
    built_for: list[DPVO] = []
    global_recorder: Recorder = Recorder("global")

    def factory(dpvo: DPVO) -> DPVOObserver:
        built_for.append(dpvo)
        return global_recorder

    cleanup.append(add_global_observer(factory))
    cleanup.append(add_global_observer(lambda dpvo: None))
    own: Recorder = Recorder("own")
    tracker: DPVO = make_tracker(network, observers=[own])
    assert built_for == [tracker]
    assert tracker.observers == [own, global_recorder]  # the None factory adds nothing


def test_global_observer_remover(network: VONet) -> None:
    remove = add_global_observer(lambda dpvo: Recorder())
    assert len(make_tracker(network).observers) == 1
    remove()
    remove()  # a second call is a no-op
    assert make_tracker(network).observers == []


# ── default depth prior keeps plain DPVO's RNG draws ────────────────────


def old_depth_init(tracker: DPVO, patches: Float32[Tensor, "1 M 3 3 3"]) -> None:
    """The pre-hook lines of DPVO.__call__ (P dpvo.py:525-529 at main 347c62081), verbatim."""
    patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
    if tracker.is_initialized:
        s = torch.median(tracker.pg.patches_[tracker.n - 3 : tracker.n, :, 2])
        patches[:, :, 2] = s


@pytest.mark.parametrize("initialized", [False, True])
def test_random_depth_init_matches_old_lines(network: VONet, initialized: bool) -> None:
    tracker: DPVO = make_tracker(network)
    tracker.is_initialized = initialized
    tracker.n = 5
    tracker.pg.patches_[:5, :, 2] = torch.rand(5, M, 3, 3)
    base: Float32[Tensor, "1 M 3 3 3"] = torch.rand(1, M, 3, 3, 3)

    expected: Float32[Tensor, "1 M 3 3 3"] = base.clone()
    torch.manual_seed(7)
    old_depth_init(tracker, expected)
    after_old: Float32[Tensor, "1"] = torch.rand(1)

    actual: Float32[Tensor, "1 M 3 3 3"] = base.clone()
    torch.manual_seed(7)
    RandomDepthInit().init_patch_depth(tracker, actual)
    after_new: Float32[Tensor, "1"] = torch.rand(1)

    assert torch.equal(actual, expected)
    assert torch.equal(after_new, after_old)  # the same number of draws
    assert RandomDepthInit().ba_prior(tracker, "local") is None
    assert RandomDepthInit().prepare(tracker, (32, 32), FrameDepth(depth=None, mask=None)) is None


# ── keyframe compaction of frame attributes ─────────────────────────────


def fill_graph(tracker: DPVO, n: int) -> None:
    """``n`` synthetic keyframes with distinct values and no edges."""
    tracker.n = n
    tracker.m = n * M
    tracker.is_initialized = True
    for idx in range(n):
        tracker.pg.tstamps_[idx] = 10 * idx
        tracker.pg.patches_[idx] = float(idx)
        tracker.pg.index_[idx] = idx


def test_keyframe_cull_shifts_and_zeroes_frame_attrs(network: VONet, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Recorder = Recorder()
    policy: RecordingPolicy = RecordingPolicy(keep=False)
    tracker: DPVO = make_tracker(network, observers=[recorder], graph_policy=policy)
    prior: Tensor = tracker.pg.add_frame_attr("prior", (M,), torch.float32)
    scale: Tensor = tracker.pg.add_frame_attr("scale", (), torch.float32)
    with pytest.raises(ValueError, match="already registered"):
        tracker.pg.add_frame_attr("prior", (M,), torch.float32)
    assert prior.shape == (24, M) and scale.shape == (24,)

    n: int = 10
    fill_graph(tracker, n)
    prior[:n] = torch.arange(1, n + 1, dtype=torch.float32)[:, None]
    scale[:n] = torch.arange(1, n + 1, dtype=torch.float32)
    monkeypatch.setattr(tracker, "motionmag", lambda i, j: 0.0)

    tracker.keyframe()

    k: int = n - tracker.cfg.keyframe_index
    assert tracker.n == n - 1
    assert policy.calls == [("force_keep", n, False), ("on_cull", k, False)]
    assert recorder.culls == [(k, float(k + 1))]
    expected: Float32[Tensor, "n"] = torch.tensor([1, 2, 3, 4, 5, 6, 8, 9, 10, 0], dtype=torch.float32)
    assert torch.equal(scale[:n], expected)
    assert torch.equal(prior[:n], expected[:, None].expand(n, M))
    assert torch.equal(tracker.pg.patches_[: n - 1, 0, 0, 0, 0], expected[: n - 1] - 1)
    assert int(tracker.pg.tstamps_[k]) == 10 * (k + 1)


@pytest.mark.parametrize("flow", [0.0, 100.0])
def test_keyframe_veto_keeps_frame(network: VONet, monkeypatch: pytest.MonkeyPatch, flow: float) -> None:
    policy: RecordingPolicy = RecordingPolicy(keep=True)
    tracker: DPVO = make_tracker(network, graph_policy=policy)
    fill_graph(tracker, 10)
    monkeypatch.setattr(tracker, "motionmag", lambda i, j: flow)

    tracker.keyframe()

    assert tracker.n == 10
    assert policy.calls == [("force_keep", 10, True), ("on_keep", 6, flow == 0.0)]


def test_default_graph_has_no_frame_attrs(network: VONet) -> None:
    tracker: DPVO = make_tracker(network)
    assert tracker.pg.frame_attrs == {}
    assert isinstance(tracker.graph_policy, DefaultGraphPolicy)
    assert isinstance(tracker.depth_prior, RandomDepthInit)


# ── global BA schedule and factor release ──────────────────────────────


def test_scheduled_global_ba_semantics(network: VONet) -> None:
    tracker: DPVO = DPVO(DPVOConfig(buffer_size=24, patches_per_frame=M, gba_schedule_k=8), network, ht=32, wd=32, device=CPU)
    tracker.n = 16
    assert not tracker._gba_scheduled_now()  # no inactive factors yet
    tracker.pg.ii_inac = torch.zeros(3, dtype=torch.long)
    assert tracker._gba_scheduled_now()
    assert not tracker._gba_scheduled_now()  # once per value of n
    tracker.n = 17
    assert not tracker._gba_scheduled_now()
    tracker.n = 24
    assert tracker._gba_scheduled_now()
    assert tracker._gba_sched_last_n == 24
    assert not make_tracker(network)._gba_scheduled_now()  # k = 0: never


def test_release_inactive_factors(network: VONet) -> None:
    tracker: DPVO = make_tracker(network)
    edges: Int64[Tensor, "5"] = torch.arange(5)
    tracker.pg.ii_inac, tracker.pg.jj_inac, tracker.pg.kk_inac = edges, edges, edges
    tracker.pg.weight_inac = torch.ones(1, 5, 2)
    tracker.pg.target_inac = torch.ones(1, 5, 2)

    tracker.release_inactive_factors()

    assert tracker.pg.ii_inac.numel() == tracker.pg.jj_inac.numel() == tracker.pg.kk_inac.numel() == 0
    assert tracker.pg.ii_inac.dtype == torch.long
    assert tracker.pg.weight_inac.shape == (1, 0, 2) and tracker.pg.target_inac.shape == (1, 0, 2)


def test_ring_length_follows_fmap_memory(network: VONet) -> None:
    lc: DPVO = DPVO(DPVOConfig(buffer_size=24, patches_per_frame=M, loop_closure=True, fmap_memory=512), network, ht=32, wd=32, device=CPU)
    assert (lc.pmem, lc.mem) == (1000, 512)
    assert lc.fmap1_.shape[1] == 512
    vo: DPVO = DPVO(DPVOConfig(buffer_size=24, patches_per_frame=M, fmap_memory=512), network, ht=32, wd=32, device=CPU)
    assert (vo.pmem, vo.mem) == (36, 36)
