"""MetricDPVO wiring on a CPU tracker (random network weights; no BA kernels run)."""

import inspect

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")

from lietorch_mojo import SE3  # noqa: E402
from torch import Tensor  # noqa: E402

from dpvo.fastba import BADepthPrior  # noqa: E402
from dpvo.hooks import FrameDepth, FrameEvent  # noqa: E402
from dpvo.metric import MetricDPVO, MetricDPVOConfig  # noqa: E402
from dpvo.metric.backend import CulledInterp, FinalRemeasure  # noqa: E402
from dpvo.metric.depth_prior import METRIC_DISPS, METRIC_WEIGHTS, MetricDepthPrior  # noqa: E402
from dpvo.metric.graph import MetricGraphPolicy  # noqa: E402
from dpvo.metric.scale_latent import keyframe_scale_latent  # noqa: E402
from dpvo.net import VONet  # noqa: E402

CPU = torch.device("cpu")
M = 8
HT, WD = 32, 48
INTRINSICS: Tensor = torch.tensor([40.0, 40.0, 24.0, 16.0])

LOCAL: MetricDPVOConfig = MetricDPVOConfig(buffer_size=24, patches_per_frame=M, prior_warmup=3)
REFINE: MetricDPVOConfig = MetricDPVOConfig(
    buffer_size=24,
    patches_per_frame=M,
    prior_warmup=3,
    loop_closure=True,
    metric_global_ba=True,
    store_inactive=True,
    normalize_global_ba=False,
    global_ba_t0="first",
    gba_schedule_k=8,
    culled_interp=True,
    remeasure_final="two",
    fmap_memory=512,
    scale_latent=True,
    scale_min_patches=2,
    loop_reverse_edges=True,
    keyframe_max_cull=10,
)


@pytest.fixture(scope="module")
def network() -> VONet:
    torch.manual_seed(0)
    return VONet()


def image(seed: int) -> Tensor:
    return torch.randint(0, 256, (3, HT, WD), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8)


def depth_map(seed: int) -> Tensor:
    return 1.0 + 4.0 * torch.rand(HT, WD, generator=torch.Generator().manual_seed(seed))


def event(added: bool) -> FrameEvent:
    return FrameEvent(
        tstamp=0, frame_index=0, keyframes_before=0, keyframe_added=added, image=image(0), intrinsics=INTRINSICS, depth=FrameDepth(None, None)
    )


def test_call_keeps_vipe_parameter_names() -> None:
    assert {"metric_depth", "metric_mask"} <= set(inspect.signature(MetricDPVO.__call__).parameters)


def test_construction_wires_the_hooks(network: VONet) -> None:
    local: MetricDPVO = MetricDPVO(LOCAL, network, HT, WD, device=CPU)
    assert isinstance(local.depth_prior, MetricDepthPrior) and isinstance(local.graph_policy, MetricGraphPolicy)
    assert [type(o) for o in local.observers] == [MetricDepthPrior]
    assert set(local.pg.frame_attrs) == {METRIC_DISPS, METRIC_WEIGHTS}
    assert local.pg.frame_attrs[METRIC_DISPS].shape == (24, M)
    assert local.mem == 36 and not local.cfg.store_inactive

    refine: MetricDPVO = MetricDPVO(REFINE, network, HT, WD, device=CPU)
    assert [type(o) for o in refine.observers] == [MetricDepthPrior, FinalRemeasure, CulledInterp]
    assert (refine.pmem, refine.mem) == (1000, 512)


def test_frames_without_depth_or_valid_pixels_are_rejected_before_patchify(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(LOCAL, network, HT, WD, device=CPU)
    with pytest.raises(ValueError, match="requires metric_depth"):
        tracker(0, image(2), INTRINSICS)
    with pytest.raises(ValueError, match="no valid pixels"):
        tracker(0, image(2), INTRINSICS, metric_depth=depth_map(2), metric_mask=torch.zeros(HT, WD, dtype=torch.bool))
    assert (tracker.n, tracker.counter, tracker.tlist) == (0, 0, [])


def test_warmup_latch_and_ba_prior_counters(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(LOCAL, network, HT, WD, device=CPU)
    prior: MetricDepthPrior = tracker.metric_prior
    tracker.pg.frame_attrs[METRIC_DISPS][:] = 0.5
    tracker.pg.frame_attrs[METRIC_WEIGHTS][:] = 1.0

    for _ in range(2):
        prior.on_frame(tracker, event(True))
    prior.on_frame(tracker, event(False))  # a frame that failed the motion probe does not count
    off: BADepthPrior | None = prior.ba_prior(tracker, "local")
    assert off is not None and off.lam == 0.0 and off.huber == 0.3 and off.disps.shape == (24 * M,)
    assert not prior.regularization_enabled

    prior.on_frame(tracker, event(True))  # the third keyframe latches the prior on after its own updates
    on: BADepthPrior | None = prior.ba_prior(tracker, "local")
    assert on is not None and on.lam == 1.0
    assert torch.equal(on.disps, tracker.pg.frame_attrs[METRIC_DISPS].reshape(-1))
    assert prior.ba_calls_with_prior == 1

    assert tracker.metric_prior_diagnostics() == {
        "keyframes_seen": 3,
        "regularization_enabled": True,
        "successful_ba_calls_with_metric_prior": 1,
        "inactive_factors": 0,
    }


def fill_keyframes(tracker: MetricDPVO, n: int) -> None:
    """``n`` keyframes 1 m apart, random patch centres and a prior that drifts in scale from the patch depths."""
    generator: torch.Generator = torch.Generator().manual_seed(8)
    tracker.n, tracker.m = n, n * M
    for idx in range(n):
        tracker.pg.poses_[idx] = SE3.exp(torch.tensor([[1.0 * idx, 0.0, 0.0, 0.0, 0.01 * idx, 0.0]])).data[0]
        tracker.pg.intrinsics_[idx] = INTRINSICS / 4
        tracker.pg.index_[idx] = idx
        tracker.pg.patches_[idx, :, 0] = torch.randint(1, WD // 4 - 1, (M, 1, 1), generator=generator).float()
        tracker.pg.patches_[idx, :, 1] = torch.randint(1, HT // 4 - 1, (M, 1, 1), generator=generator).float()
        tracker.pg.patches_[idx, :, 2] = 0.5 + 0.1 * torch.rand(M, 1, 1, generator=generator)
    per_frame_scale: Tensor = 1.0 + 0.05 * torch.arange(n, dtype=torch.float32)  # a scale drift for the latent to absorb
    tracker.pg.frame_attrs[METRIC_DISPS][:n] = tracker.pg.patches_[:n, :, 2, 1, 1] * per_frame_scale[:, None]
    tracker.pg.frame_attrs[METRIC_WEIGHTS][:n] = 1.0


def test_global_prior_applies_the_scale_latent_to_rows_below_n(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(REFINE, network, HT, WD, device=CPU)
    prior: MetricDepthPrior = tracker.metric_prior
    fill_keyframes(tracker, 6)
    tracker.pg.frame_attrs[METRIC_DISPS][6] = 0.7  # a slot >= n (e.g. a frame that failed the motion probe)
    disps: Tensor = tracker.pg.frame_attrs[METRIC_DISPS]

    before_latch: BADepthPrior | None = prior.ba_prior(tracker, "global")
    assert before_latch is not None and before_latch.lam == 0.0
    assert torch.equal(before_latch.disps, disps.reshape(-1))  # no latent without the prior

    prior.regularization_enabled = True
    scaled: BADepthPrior | None = prior.ba_prior(tracker, "global")
    assert scaled is not None and scaled.lam == 1.0
    s: Tensor = keyframe_scale_latent(
        tracker, disps, tracker.pg.frame_attrs[METRIC_WEIGHTS], clamp=0.35, min_patches=2, min_parallax_px=2.0, lambda_smooth=20.0
    )
    assert s.shape == (6,) and s.abs().max() > 0.0
    expected: Tensor = disps.clone()
    expected[:6] = disps[:6] * torch.exp(s)[:, None]
    assert torch.equal(scaled.disps.view(24, M), expected)
    assert torch.equal(scaled.weights, tracker.pg.frame_attrs[METRIC_WEIGHTS].reshape(-1))

    local: BADepthPrior | None = prior.ba_prior(tracker, "local")
    assert local is not None and torch.equal(local.disps, disps.reshape(-1))  # the local BA keeps the raw prior
    assert (prior.global_ba_calls, prior.global_ba_calls_with_prior, prior.ba_calls_with_prior) == (2, 1, 2)
    diagnostics: dict[str, int | bool] = tracker.metric_prior_diagnostics()
    assert diagnostics["metric_global_ba"] is True and diagnostics["normalize_called"] is False
    assert diagnostics["global_ba_calls"] == 2 and diagnostics["global_ba_calls_with_metric_prior"] == 1


def test_graph_policy_counts_loop_edges_and_builds_reverse_edges(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(REFINE, network, HT, WD, device=CPU)
    fill_keyframes(tracker, 20)
    lii: Tensor = torch.cat([2 * M + torch.arange(M), 3 * M + torch.arange(M)])  # patches of frames 2 and 3
    ljj: Tensor = torch.tensor([18] * M + [19] * M)
    extra: tuple[Tensor, Tensor] | None = tracker.metric_graph.extra_loop_edges(tracker, lii, ljj)
    assert extra is not None
    assert torch.equal(extra[0], torch.cat([18 * M + torch.arange(M), 19 * M + torch.arange(M)]))
    assert torch.equal(extra[1], torch.tensor([2] * M + [3] * M))
    assert tracker.metric_graph.loop_edges_added == 2 * M

    gba_only: MetricDPVOConfig = MetricDPVOConfig(
        buffer_size=24, patches_per_frame=M, loop_closure=True, metric_global_ba=True, store_inactive=True, normalize_global_ba=False, global_ba_t0="first"
    )
    plain: MetricDPVO = MetricDPVO(gba_only, network, HT, WD, device=CPU)
    assert plain.metric_graph.extra_loop_edges(plain, lii, ljj) is None
    assert plain.metric_graph.loop_edges_added == 2 * M


def test_cull_records_the_relative_pose_to_the_next_frame(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(REFINE, network, HT, WD, device=CPU)
    fill_keyframes(tracker, 10)
    for idx in range(10):
        tracker.pg.tstamps_[idx] = 3 * idx
    interp: CulledInterp = next(o for o in tracker.observers if isinstance(o, CulledInterp))
    interp.on_cull(tracker, 6)
    t_next, relative = interp.delta_next[18]
    expected = SE3(tracker.pg.poses_[7]) * SE3(tracker.pg.poses_[6]).inv()
    assert isinstance(expected, SE3)
    assert t_next == 21 and torch.equal(relative, expected.data)
