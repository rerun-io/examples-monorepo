"""``open4dgs.fused_regularisers.core_regularisers`` equals TRiGS's per-step regulariser code (the training loop's
static-excluded terms): values and every parameter's gradient, also with no dynamic row."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

ZETA = ("zeta_c0_t", "zeta_c0_r", "zeta_c1_t", "zeta_c1_r", "zeta_c2_t", "zeta_c2_r", "zeta_base_t", "zeta_base_r")
"""Spelled out, not imported: the test checks that ``ZETA_KEYS`` (the trainer's packing) is this order."""


def _params(n: int, static_fraction: float) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    p = {k: torch.randn(n, 3, generator=g).requires_grad_(True) for k in ZETA}
    p["sh0"] = torch.randn(n, 3, generator=g).requires_grad_(True)
    p["opacities"] = torch.randn(n, generator=g).requires_grad_(True)
    p["log_durations"] = (torch.randn(n, 1, generator=g) * 0.5 - 1.0).requires_grad_(True)
    p["scales"] = (torch.randn(n, 3, generator=g) * 0.3 - 3.0).requires_grad_(True)
    p["is_static"] = (torch.rand(n, 1, generator=g) < static_fraction).float()
    p["means"] = torch.randn(n, 3, generator=g).requires_grad_(True)
    return p


def _reference(p: dict[str, torch.Tensor], n: int, gamma: torch.Tensor, rigid_idx: torch.Tensor, nn_indices: torch.Tensor, scale_rows: str):
    """The loop's code for these terms (trainer.py, static_exclude_from_* on, lambda_motion_residual 0)."""
    from trigs.core.losses import bezier_motion_loss, rigid_coherence_loss, scale_size_loss

    c0 = torch.cat([p["zeta_c0_t"], p["zeta_c0_r"]], dim=-1)
    c1 = torch.cat([p["zeta_c1_t"], p["zeta_c1_r"]], dim=-1)
    c2 = torch.cat([p["zeta_c2_t"], p["zeta_c2_r"]], dim=-1)
    dyn_mask = ~(p["is_static"].reshape(-1) > 0.5)
    dyn = torch.nonzero(dyn_mask).reshape(-1)
    zero = p["zeta_c0_t"].new_zeros(())
    any_dyn = dyn.numel() > 0
    motion = bezier_motion_loss(c0.index_select(0, dyn), c1.index_select(0, dyn), c2.index_select(0, dyn)) if any_dyn else zero
    zb = torch.cat([p["zeta_base_t"], p["zeta_base_r"]], dim=-1)
    rigid = rigid_coherence_loss(
        zb.index_select(0, rigid_idx), p["sh0"].index_select(0, rigid_idx), nn_indices, lambda_c=50.0, detach_color_weight=False
    )
    base_opa = torch.sigmoid(p["opacities"]).unsqueeze(-1)
    reg = (base_opa.index_select(0, dyn) * gamma.index_select(0, dyn).detach()).mean() if any_dyn else zero
    sigma_t = torch.exp(p["log_durations"])
    sigma = torch.clamp(0.2 - sigma_t.index_select(0, dyn), min=0.0).pow(2).mean() if any_dyn else zero
    scale = zero if scale_rows == "off" else scale_size_loss(p["scales"], n, slice(None) if scale_rows == "all" else dyn_mask)
    return motion, rigid, reg, sigma, scale


@pytest.mark.parametrize("nonfinite_static", [False, True])
@pytest.mark.parametrize("static_fraction", [0.3, 1.0])
@pytest.mark.parametrize("scale_rows", ["all", "dynamic", "off"])
def test_core_regularisers_equal_the_trigs_loop(static_fraction: float, scale_rows: str, nonfinite_static: bool) -> None:
    """``nonfinite_static``: excluded (static) rows with an overflowing log-scale and NaN motion coefficients, which the
    loop gathers away before its nonlinear terms (its opacity and duration terms gather after them, so the loop itself
    has NaN gradients there)."""
    from open4dgs.fused_regularisers import ZETA_KEYS, core_regularisers

    assert ZETA_KEYS == ZETA
    if nonfinite_static and scale_rows == "all":
        pytest.skip("the all-rows scale term reads static rows")
    n = 3000
    p = _params(n, static_fraction)
    if nonfinite_static:
        static = p["is_static"].reshape(-1) > 0.5
        with torch.no_grad():
            p["scales"][static] = 100.0
            p["zeta_c1_r"][static] = float("nan")
    gamma = torch.rand(n, 1)
    dyn = torch.nonzero(p["is_static"].reshape(-1) <= 0.5).reshape(-1)
    rigid_idx = dyn if dyn.numel() > 3 else torch.arange(10)
    nn_indices = torch.randint(0, int(rigid_idx.numel()), (int(rigid_idx.numel()), 3), generator=torch.Generator().manual_seed(2))
    leaves = [v for v in p.values() if v.requires_grad]
    ref = _reference(p, n, gamma, rigid_idx, nn_indices, scale_rows)
    got = core_regularisers(
        tuple(p[k] for k in ZETA),
        p["sh0"],
        p["opacities"],
        p["log_durations"],
        p["scales"],
        p["is_static"],
        gamma,
        rigid_idx,
        nn_indices,
        50.0,
        False,
        0.2,
        scale_rows,
    )
    weights = (1.0, 0.3, 0.7, 1.3, 0.9)
    for a, b in zip(got, ref, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    ga = torch.autograd.grad(torch.stack([w * t for w, t in zip(weights, got, strict=True)]).sum(), leaves, allow_unused=True)
    gb = torch.autograd.grad(torch.stack([w * t for w, t in zip(weights, ref, strict=True)]).sum(), leaves, allow_unused=True)
    for a, b in zip(ga, gb, strict=True):
        if a is None or b is None:
            assert (a is None or not a.any()) and (b is None or not b.any())
        else:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)


def test_seam_falls_back_outside_its_configuration() -> None:
    from trigs.core.trainer import TrigsTrainingConfig

    from open4dgs.trigs_trainer import Open4dgsTrainer

    p = _params(50, 0.3)
    gamma = torch.rand(50, 1)
    idx = torch.arange(10)
    nn = torch.zeros(10, 3, dtype=torch.long)
    on = Open4dgsTrainer(device="cpu", fused_regularisers=True)
    cfg = TrigsTrainingConfig()
    assert on._core_regularisers(p, cfg, gamma, nn, idx, 0.2) is not None
    assert Open4dgsTrainer(device="cpu")._core_regularisers(p, cfg, gamma, nn, idx, 0.2) is None
    assert on._core_regularisers(p, cfg, gamma, None, idx, 0.2) is None  # delayed pruning: no rigidity graph
    assert on._core_regularisers(p, TrigsTrainingConfig(lambda_motion_residual=0.1), gamma, nn, idx, 0.2) is None


def check_compiled_core_regularisers_on_cuda() -> None:
    """The compiled graph (CUDA) against the eager function, over rigidity graphs of different lengths, with no recompile
    after the first. Runs in a child process: under pytest ``PIXI_DEV_MODE`` selects eager."""
    from open4dgs.fused_regularisers import ZETA_KEYS, compiled_core_regularisers, core_regularisers

    compiled = compiled_core_regularisers(torch.device("cuda"))
    assert compiled is not core_regularisers
    n = 3000
    p = {k: v.detach().cuda().requires_grad_(v.requires_grad) for k, v in _params(n, 0.3).items()}
    gamma = torch.rand(n, 1, device="cuda")
    dyn = torch.nonzero(p["is_static"].reshape(-1) <= 0.5).reshape(-1)
    leaves = [v for v in p.values() if v.requires_grad]
    graphs: list[int] = []
    for i, m in enumerate((int(dyn.numel()), 1500, 900)):
        rigid_idx = dyn[:m]
        nn_indices = torch.randint(0, m, (m, 3), generator=torch.Generator().manual_seed(i)).cuda()
        torch._dynamo.maybe_mark_dynamic(rigid_idx, 0)
        torch._dynamo.maybe_mark_dynamic(nn_indices, 0)
        args = (
            tuple(p[k] for k in ZETA_KEYS),
            p["sh0"],
            p["opacities"],
            p["log_durations"],
            p["scales"],
            p["is_static"],
            gamma,
            rigid_idx,
            nn_indices,
            50.0,
            False,
            0.2,
            "dynamic",
        )
        ref, got = core_regularisers(*args), compiled(*args)
        graphs.append(int(torch._dynamo.utils.counters["stats"]["unique_graphs"]))
        for a, b in zip(got, ref, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        gb = torch.autograd.grad(torch.stack(ref).sum(), leaves, allow_unused=True)
        ga = torch.autograd.grad(torch.stack(got).sum(), leaves, allow_unused=True)
        for a, b in zip(ga, gb, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-8)
    assert graphs[0] >= 1 and graphs[1:] == [graphs[0]] * 2, f"compiled graphs after each length: {graphs}"


@pytest.mark.integration
def test_compiled_core_regularisers_equal_eager_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (torch.compile / Triton)")
    env: dict[str, str] = dict(os.environ)
    env["PIXI_DEV_MODE"] = "0"
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", "import test_fused_regularisers as m; m.check_compiled_core_regularisers_on_cuda()"], check=True, env=env)
