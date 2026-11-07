"""dpvo.metric's numerics against ViPE's MetricTrail DPVO on synthetic tensors (bit-identical on the CPU).

ViPE is installed only in the metrictrail environments; elsewhere the module skips.
"""

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")
pytest.importorskip("vipe", reason="ViPE is installed only in the metrictrail env")

from lietorch_mojo import SE3  # noqa: E402
from torch import Tensor  # noqa: E402
from vipe.metrictrail.dpvo import burst_fixes as v_burst  # noqa: E402
from vipe.metrictrail.dpvo import graph_guards as v_graph  # noqa: E402
from vipe.metrictrail.dpvo import metric_depth as v_metric  # noqa: E402
from vipe.metrictrail.dpvo import scale_latent as v_scale  # noqa: E402

from dpvo.metric import backend, depth_prior, graph, scale_latent  # noqa: E402

HT, WD = 32, 48


def synthetic_depth(seed: int) -> tuple[Tensor, Tensor]:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    depth: Tensor = torch.rand(HT, WD, generator=generator) * 8.0
    depth[torch.rand(HT, WD, generator=generator) < 0.05] = float("nan")
    depth[torch.rand(HT, WD, generator=generator) < 0.05] = 900.0  # beyond depth_max
    mask: Tensor = torch.rand(HT, WD, generator=generator) > 0.2
    return depth, mask


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prepare_metric_disparity_matches_vipe(seed: int, use_mask: bool) -> None:
    depth, mask = synthetic_depth(seed)
    caller_mask: Tensor = mask.clone()
    ours: depth_prior.MetricDisparityGrid = depth_prior.prepare_metric_disparity(
        depth, caller_mask if use_mask else None, image_hw=(HT, WD), min_depth=0.2, max_depth=500.0
    )
    v_disps, v_valid, v_diag = v_metric.prepare_metric_disparity(
        depth, mask.clone() if use_mask else None, image_hw=(HT, WD), min_depth=0.2, max_depth=500.0
    )
    assert v_diag["has_valid_depth"]
    assert torch.equal(ours.disps, v_disps) and ours.disps.dtype == torch.float32
    assert torch.equal(ours.valid, v_valid)
    assert torch.equal(caller_mask, mask)  # unlike ViPE, the caller's mask is not ANDed in place


def test_prepare_rejects_a_frame_without_valid_grid_cells() -> None:
    depth: Tensor = torch.full((HT, WD), 2.0)
    mask: Tensor = torch.zeros(HT, WD, dtype=torch.bool)
    mask[0::4, 0::4] = True  # valid pixels exist, but none on the 1::4 grid
    _, _, v_diag = v_metric.prepare_metric_disparity(depth, mask.clone(), image_hw=(HT, WD), min_depth=0.2, max_depth=500.0)
    assert not v_diag["has_valid_depth"]
    with pytest.raises(ValueError, match="no valid pixels"):
        depth_prior.prepare_metric_disparity(depth, mask, image_hw=(HT, WD), min_depth=0.2, max_depth=500.0)


def test_patch_metric_priors_match_vipe() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(3)
    M: int = 40
    h4, w4 = HT // 4, WD // 4
    patches: Tensor = torch.rand(1, M, 3, 3, 3, generator=generator)
    patches[0, :, 0] = torch.randint(-1, w4 + 1, (M, 1, 1), generator=generator).float()  # a few out of bounds
    patches[0, :, 1] = torch.randint(-1, h4 + 1, (M, 1, 1), generator=generator).float()
    patches[0, :3, 2, 1, 1] = torch.tensor([float("nan"), -1.0, 0.0])
    valid_grid: Tensor = torch.rand(1, 1, h4, w4, generator=generator) > 0.3

    ours: depth_prior.PatchPriors = depth_prior.patch_metric_priors(patches, valid_grid)
    v_disps, v_weights = v_metric.patch_metric_priors(patches, valid_grid)
    torch.testing.assert_close(ours.disps, v_disps, rtol=0.0, atol=0.0, equal_nan=True)
    assert torch.equal(ours.weights, v_weights)


def offsets_inputs(seed: int) -> tuple[Tensor, Tensor]:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    log_ratio: Tensor = torch.randn(17, 48, generator=generator) * 0.4
    log_ratio[torch.rand(17, 48, generator=generator) < 0.05] = float("inf")
    valid: Tensor = torch.rand(17, 48, generator=generator) > torch.linspace(0.1, 0.95, 17)[:, None]
    valid[3] = False  # a row without data
    return log_ratio, valid


@pytest.mark.parametrize("seed", range(4))
def test_robust_log_scale_offsets_bit_identical(seed: int) -> None:
    log_ratio, valid = offsets_inputs(seed)
    ours: tuple[Tensor, Tensor, Tensor] = scale_latent.robust_log_scale_offsets(
        log_ratio, valid, huber=0.2, min_patches=16, patches_per_frame=48
    )
    vipe: tuple[Tensor, Tensor, Tensor] = v_burst.robust_log_scale_offsets(log_ratio, valid, huber=0.2, min_patches=16, patches_per_frame=48)
    for a, b in zip(ours, vipe, strict=True):
        assert torch.equal(a, b) and a.dtype == b.dtype


@pytest.mark.parametrize("seed", range(4))
def test_gba_scale_latent_bit_identical(seed: int) -> None:
    log_ratio, valid = offsets_inputs(seed)
    m, _, w = v_burst.robust_log_scale_offsets(log_ratio, valid, huber=0.2, min_patches=16, patches_per_frame=48)
    generator: torch.Generator = torch.Generator().manual_seed(100 + seed)
    parallax: Tensor = torch.rand(17, generator=generator) * 6.0
    parallax[5] = float("nan")
    kwargs: dict[str, float] = {"clamp": 0.35, "min_parallax_px": 2.0, "lambda_smooth": 20.0, "lambda_anchor": 1e-3}
    ours: Tensor = scale_latent.gba_scale_latent(m, w, parallax, **kwargs)
    vipe, _ = v_scale.gba_scale_latent(m, w, parallax, **kwargs)
    assert torch.equal(ours, vipe) and ours.dtype == torch.float32
    assert torch.equal(
        scale_latent.solve_smooth_scale(m, w, lambda_smooth=20.0, lambda_anchor=1e-3),
        v_scale.solve_smooth_scale(m, w, lambda_smooth=20.0, lambda_anchor=1e-3),
    )


def test_gba_scale_latent_without_usable_data_matches_vipe() -> None:
    m: Tensor = torch.tensor([0.1, -0.2, 0.05])
    for w, parallax in ((torch.zeros(3), torch.full((3,), 5.0)), (torch.ones(3), torch.zeros(3)), (torch.zeros(0), torch.zeros(0))):
        mm: Tensor = m[: w.numel()]
        kwargs: dict[str, float] = {"clamp": 0.35, "min_parallax_px": 2.0, "lambda_smooth": 20.0}
        assert torch.equal(scale_latent.gba_scale_latent(mm, w, parallax, **kwargs), v_scale.gba_scale_latent(mm, w, parallax, **kwargs)[0])


def random_pose(generator: torch.Generator, scale: float) -> Tensor:
    xi: Tensor = torch.randn(1, 6, generator=generator) * scale
    return SE3.exp(xi).data[0]


def test_hole_guard_matches_vipe_on_a_scripted_run() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(5)
    ours: graph.HoleGuard = graph.HoleGuard(3, 10.0)
    vipe = v_graph.HoleGuard(3, 10.0)
    # 1 = cull (with the cumulative relative pose), 0 = keep
    script: list[int] = [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    keeps: list[bool] = []
    for step, action in enumerate(script):
        assert ours.should_keep() == vipe.should_keep()
        keeps.append(ours.should_keep())
        if action:
            dP: Tensor = random_pose(generator, 0.15)
            ours.on_cull(dP)
            vipe.on_cull(dP)
        else:
            ours.on_keep()
            vipe.on_keep(step, forced=False)
        assert (ours.run, ours.rot_path_deg) == (vipe.run, vipe.rot_path_deg)
        assert (ours.last_q is None) == (vipe.last_q is None)
        if ours.last_q is not None:
            assert torch.equal(ours.last_q, vipe.last_q)
    assert any(keeps)  # the script reaches the guard


def test_disabled_hole_guard_never_keeps() -> None:
    guard: graph.HoleGuard = graph.HoleGuard(0, 10.0)
    for _ in range(20):
        guard.on_cull(SE3.exp(torch.full((1, 6), 0.3)).data[0])
    assert not guard.should_keep() and guard.run == 0


def test_reverse_loop_edges_match_vipe() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(9)
    M: int = 6
    ii: Tensor = torch.randint(0, 40, (30,), generator=generator)
    jj: Tensor = torch.randint(80, 120, (30,), generator=generator)
    ii = torch.cat([ii, ii[:5]])  # duplicates collapse
    jj = torch.cat([jj, jj[:5]])
    pairs: Tensor = graph.loop_pairs(ii, jj)
    assert torch.equal(pairs, v_graph.loop_pairs(ii, jj))
    for mem in (36, 100, 512):
        kk_r, jj_r = graph.reverse_loop_edges(pairs, 120, mem, M, 23)
        v_kk, v_jj, _, _ = v_graph.reverse_loop_edges(pairs, 120, mem, M, 23)
        assert torch.equal(kk_r, v_kk) and torch.equal(jj_r, v_jj)
    empty: Tensor = torch.empty(0, dtype=torch.long)
    assert graph.loop_pairs(empty, empty).shape == (0, 2)
    assert graph.reverse_loop_edges(graph.loop_pairs(empty, empty), 10, 36, M, 23)[0].numel() == 0


def test_culled_interp_matches_vipe() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(11)
    n_frames: int = 24
    poses: Tensor = torch.stack([random_pose(generator, 0.3) for _ in range(n_frames)])
    kept: Tensor = torch.zeros(n_frames, dtype=torch.bool)
    kept[[0, 1, 5, 6, 12, 13, 20, 23]] = True
    next_rel: dict[int, tuple[int, Tensor]] = {
        4: (5, random_pose(generator, 0.1)),  # run 1..4 -> corrected
        11: (12, random_pose(generator, 0.1)),  # run 7..11 -> corrected
        19: (21, random_pose(generator, 0.1)),  # points past the kept frame 20 -> skipped
        22: (23, random_pose(generator, 0.1)),
    }
    ours: Tensor = backend.culled_interp(poses, kept, next_rel)
    vipe, stats = v_burst.culled_interp(poses, kept, next_rel)
    assert stats["runs_corrected"] == 3 and stats["runs_skipped"] == 1
    assert torch.equal(ours, vipe)
    assert torch.equal(ours[kept], poses[kept])


@pytest.mark.integration
def test_keyframe_scale_latent_matches_vipe_update_scale_latent() -> None:
    """Our latent and parallax against ViPE's DPVO methods, called on a stub that holds the same CUDA buffers."""
    if not torch.cuda.is_available():
        pytest.skip("ViPE's dpvo module allocates on CUDA at import")
    from types import SimpleNamespace

    from vipe.metrictrail.dpvo.dpvo import DPVO as VipeDPVO

    from dpvo.metric import MetricDPVO, MetricDPVOConfig
    from dpvo.metric.depth_prior import METRIC_DISPS, METRIC_WEIGHTS
    from dpvo.net import VONet

    M, n = 16, 12
    cfg: MetricDPVOConfig = MetricDPVOConfig(
        buffer_size=32,
        patches_per_frame=M,
        loop_closure=True,
        metric_global_ba=True,
        store_inactive=True,
        normalize_global_ba=False,
        global_ba_t0="first",
        scale_latent=True,
        scale_min_patches=4,
        remeasure_final="two",
    )
    torch.manual_seed(0)
    tracker: MetricDPVO = MetricDPVO(cfg, VONet(), 64, 96)
    generator: torch.Generator = torch.Generator().manual_seed(13)
    tracker.n, tracker.m = n, n * M
    for idx in range(n):
        xi: Tensor = torch.tensor([[0.3 * idx, 0.05 * idx, 0.0, 0.0, 0.02 * idx, 0.0]]) + 0.01 * torch.randn(1, 6, generator=generator)
        tracker.pg.poses_[idx] = SE3.exp(xi).data[0].cuda()
        tracker.pg.intrinsics_[idx] = torch.tensor([20.0, 20.0, 12.0, 8.0], device="cuda")
        tracker.pg.index_[idx] = idx
        tracker.pg.patches_[idx, :, 0] = torch.randint(1, 23, (M, 1, 1), generator=generator).float().cuda()
        tracker.pg.patches_[idx, :, 1] = torch.randint(1, 15, (M, 1, 1), generator=generator).float().cuda()
        tracker.pg.patches_[idx, :, 2] = (0.3 + 0.4 * torch.rand(M, 1, 1, generator=generator)).cuda()
    disps: Tensor = tracker.pg.frame_attrs[METRIC_DISPS]
    weights: Tensor = tracker.pg.frame_attrs[METRIC_WEIGHTS]
    drift: Tensor = (1.0 + 0.04 * torch.randn(n, 1, generator=generator)).cuda()
    disps[:n] = tracker.pg.patches_[:n, :, 2, 1, 1] * drift * (1.0 + 0.05 * torch.randn(n, M, generator=generator).cuda())
    weights[:n] = (torch.rand(n, M, generator=generator) > 0.2).float().cuda()
    tracker.metric_prior.regularization_enabled = True

    stub = SimpleNamespace(
        n=n, M=M, P=3, ix=tracker.ix, poses=tracker.poses, patches=tracker.patches, intrinsics=tracker.intrinsics,
        pg=SimpleNamespace(patches_=tracker.pg.patches_, metric_disps_=disps, metric_weights_=weights,
                           metric_logscale_=torch.full((32,), 0.25, device="cuda"), tstamps_=tracker.pg.tstamps_),
        cfg=SimpleNamespace(GBA_SCALE_MIN_PATCHES=4, GBA_SCALE_CLAMP=0.35, GBA_SCALE_MIN_PARALLAX_PX=2.0, GBA_SCALE_SMOOTH=20.0),
        scale_latent_calls=0, scale_latent_std_hist=[], scale_latent_maxabs_hist=[], scale_latent_gated_hist=[],
        scale_latent_wmean_abs_max=0.0, scale_latent_clamped_total=0, scale_latent_last=None, global_ba_calls=0,
        scale_latent_solve_seconds=0.0,
    )
    stub._parallax_rows = lambda: VipeDPVO._parallax_rows(stub)
    VipeDPVO._update_scale_latent(stub)
    vipe_prior: Tensor = (disps * torch.exp(stub.pg.metric_logscale_)[:, None]).reshape(-1)

    from dpvo.metric.scale_latent import parallax_rows

    assert torch.equal(parallax_rows(tracker), VipeDPVO._parallax_rows(stub))
    assert stub.pg.metric_logscale_[:n].abs().max() > 0.0  # the latent is active
    ours = tracker.metric_prior.ba_prior(tracker, "global")
    assert ours is not None
    assert torch.equal(ours.disps[: n * M], vipe_prior[: n * M])  # rows >= n are never read by the BA
    assert torch.equal(ours.weights, weights.reshape(-1))
