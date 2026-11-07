"""MetricDPVO on CUDA: the first frame's prior and RNG stream, rejection after patchify, and full runs of the presets.

The network has random weights, so the runs check the wiring (every hook, terminate, the diagnostics contract),
not accuracy; the lockstep against ViPE lives in the evidence harness.
"""

from collections.abc import Iterator

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")

from torch import Tensor  # noqa: E402

from dpvo.metric import MetricDPVO, MetricDPVOConfig  # noqa: E402
from dpvo.metric.backend import REMEASURE_PASSES, FinalRemeasure  # noqa: E402
from dpvo.metric.depth_prior import METRIC_DISPS, METRIC_WEIGHTS  # noqa: E402
from dpvo.net import VONet  # noqa: E402

pytestmark = pytest.mark.integration

M = 8
HT, WD = 64, 96
INTRINSICS: Tensor = torch.tensor([80.0, 80.0, 48.0, 32.0])
LOCAL: MetricDPVOConfig = MetricDPVOConfig(buffer_size=64, patches_per_frame=M, prior_warmup=3)


@pytest.fixture(scope="module")
def network() -> VONet:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for the DPVO kernels")
    torch.manual_seed(0)
    return VONet()


def image(seed: int) -> Tensor:
    return torch.randint(0, 256, (3, HT, WD), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8).cuda()


def depth_map(seed: int) -> Tensor:
    return (1.0 + 4.0 * torch.rand(HT, WD, generator=torch.Generator().manual_seed(seed))).cuda()


def test_first_frame_takes_the_prior_without_a_random_depth_draw(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(LOCAL, network, HT, WD)
    depth: Tensor = depth_map(1)
    torch.manual_seed(4)
    with torch.no_grad():
        tracker(0, image(1), INTRINSICS.cuda(), metric_depth=depth)
    after: Tensor = torch.rand(1, device="cuda")

    torch.manual_seed(4)
    x: Tensor = torch.randint(1, WD // 4 - 1, size=[1, M], device="cuda")
    y: Tensor = torch.randint(1, HT // 4 - 1, size=[1, M], device="cuda")
    assert torch.equal(after, torch.rand(1, device="cuda"))  # only patchify's two randint draws

    prior: Tensor = depth[1::4, 1::4].reciprocal()[y[0], x[0]]
    patches: Tensor = tracker.pg.patches_[0]
    assert torch.equal(patches[:, :2, 1, 1], torch.stack([x[0], y[0]], dim=-1).float())
    assert torch.equal(patches[:, 2], prior[:, None, None].expand(M, 3, 3))
    assert torch.equal(tracker.pg.frame_attrs[METRIC_DISPS][0], prior)
    assert torch.equal(tracker.pg.frame_attrs[METRIC_WEIGHTS][0], torch.ones(M, device="cuda"))
    assert (tracker.n, tracker.counter) == (1, 1)
    assert tracker.metric_prior.keyframes_seen == 1


def test_rejected_frames_leave_the_state_untouched(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(LOCAL, network, HT, WD)
    with pytest.raises(ValueError, match="requires metric_depth"):
        tracker(0, image(2), INTRINSICS)
    only_unsampled_cell: Tensor = torch.zeros(HT, WD, dtype=torch.bool, device="cuda")
    only_unsampled_cell[1, 1] = True  # grid cell (0, 0): RANDOM centres are drawn from 1..w4-2 only
    with torch.no_grad(), pytest.raises(ValueError, match="no valid sampled DPVO patches"):
        tracker(0, image(2), INTRINSICS.cuda(), metric_depth=depth_map(2), metric_mask=only_unsampled_cell)
    with pytest.raises(ValueError, match="no valid pixels"):
        tracker(0, image(2), INTRINSICS.cuda(), metric_depth=depth_map(2), metric_mask=torch.zeros(HT, WD, dtype=torch.bool, device="cuda"))
    assert (tracker.n, tracker.counter, tracker.tlist) == (0, 0, [])
    assert not tracker.pg.frame_attrs[METRIC_DISPS].any() and not tracker.pg.patches_.any()
    assert tracker.metric_prior.keyframes_seen == 0


def test_frame_without_valid_sampled_patches_is_rejected(network: VONet) -> None:
    tracker: MetricDPVO = MetricDPVO(LOCAL, network, HT, WD)
    only_unsampled_cell: Tensor = torch.zeros(HT, WD, dtype=torch.bool, device="cuda")
    only_unsampled_cell[1, 1] = True  # grid cell (0, 0): RANDOM centres are drawn from 1..w4-2 only
    with torch.no_grad(), pytest.raises(ValueError, match="no valid sampled DPVO patches"):
        tracker(0, image(2), INTRINSICS.cuda(), metric_depth=depth_map(2), metric_mask=only_unsampled_cell)
    assert (tracker.n, tracker.counter, tracker.tlist) == (0, 0, [])
    assert not tracker.pg.frame_attrs[METRIC_DISPS].any() and not tracker.pg.patches_.any()
    assert tracker.metric_prior.keyframes_seen == 0


def drifting_frames(count: int) -> Iterator[tuple[Tensor, Tensor]]:
    """A smooth random texture panned by 1 px per frame, with a slanted depth plane."""
    generator: torch.Generator = torch.Generator().manual_seed(21)
    texture: Tensor = torch.nn.functional.interpolate(torch.rand(1, 3, 24, 2 * WD // 4, generator=generator), size=(HT, 2 * WD), mode="bilinear")[0]
    depth: Tensor = (1.5 + torch.linspace(0.0, 2.0, WD))[None, :].expand(HT, WD).contiguous()
    for t in range(count):
        frame: Tensor = (255.0 * texture[:, :, t : t + WD]).to(torch.uint8)
        yield frame.cuda(), depth.cuda()


GBA_REFINE: MetricDPVOConfig = MetricDPVOConfig(
    buffer_size=128,
    patches_per_frame=M,
    prior_warmup=3,
    removal_window=10,
    optimization_window=6,
    patch_lifetime=8,
    keyframe_thresh=1.0,
    loop_closure=True,
    global_opt_freq=5,
    max_edge_age=100,
    metric_global_ba=True,
    store_inactive=True,
    normalize_global_ba=False,
    global_ba_t0="first",
    gba_schedule_k=4,
    culled_interp=True,
    remeasure_final="two",
    fmap_memory=64,
    remeasure_burnin=1,
    remeasure_warm_active=True,
    remeasure_carry_state=True,
    scale_latent=True,
    scale_min_patches=2,
    scale_min_parallax_px=0.0,
    loop_reverse_edges=True,
    keyframe_max_cull=2,
    keyframe_max_cull_rot_deg=0.0,
)


@pytest.mark.parametrize("preset", ["local", "gba_refine"])
def test_run_and_terminate(network: VONet, preset: str, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg: MetricDPVOConfig = LOCAL if preset == "local" else GBA_REFINE
    tracker: MetricDPVO = MetricDPVO(cfg, network, HT, WD)
    # Random weights predict almost no flow; let every frame pass the initialization motion check
    monkeypatch.setattr(tracker, "motion_probe", lambda: torch.tensor(10.0))
    count: int = 48
    torch.manual_seed(5)
    with torch.inference_mode():
        for t, (frame, depth) in enumerate(drifting_frames(count)):
            tracker(t, frame, INTRINSICS.cuda(), metric_depth=depth)
    with torch.no_grad():
        poses, tstamps = tracker.terminate()

    assert poses.shape == (count, 7) and poses.dtype == np.float32 and np.isfinite(poses).all()
    assert np.array_equal(tstamps, np.arange(count, dtype=np.float64))
    diagnostics: dict[str, int | bool] = tracker.metric_prior_diagnostics()
    assert diagnostics["regularization_enabled"] is True
    assert diagnostics["successful_ba_calls_with_metric_prior"] > 0
    assert diagnostics["inactive_factors"] == 0
    assert diagnostics["keyframes_seen"] >= tracker.n
    if preset == "gba_refine":
        assert diagnostics["global_ba_calls"] >= sum(REMEASURE_PASSES)  # every re-measurement iteration solves a global BA
        assert diagnostics["global_ba_calls_with_metric_prior"] > 0
        assert isinstance(diagnostics["loop_edges_added"], int)
        assert any(isinstance(o, FinalRemeasure) for o in tracker.observers)
    else:
        assert set(diagnostics) == {"keyframes_seen", "regularization_enabled", "successful_ba_calls_with_metric_prior", "inactive_factors"}
    assert tracker.pg.frame_attrs[METRIC_WEIGHTS][: tracker.n].sum() > 0
