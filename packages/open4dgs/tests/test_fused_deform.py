"""``Open4dgsTrainer(fused_deform=True)._render_inputs`` (``open4dgs.fused_deform.gather_deform``) equals TRiGS's own
gather + deform + rasterizer-input prep: values and every parameter's gradient."""

import pytest
import torch

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")


def _params(dev: str, n: int = 4000) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def r(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * scale).to(dev).requires_grad_(True)

    params = {
        "times": (torch.rand(n, 1, generator=g) * 4.0).to(dev).requires_grad_(True),
        "log_durations": r(n, 1, scale=0.3),
        "means": r(n, 3),
        "quats": r(n, 4),
        "opacities": r(n),
        "scales": r(n, 3, scale=0.2),
        "anchors": r(n, 3),
        "sh0": r(n, 3),
        "shN": r(n, 15, 3, scale=0.1),
        "zeta_base_t": r(n, 3, scale=0.3),
        "zeta_base_r": r(n, 3, scale=0.3),
    }
    for c in ("c0", "c1", "c2"):
        params[f"zeta_{c}_t"], params[f"zeta_{c}_r"] = r(n, 3, scale=0.1), r(n, 3, scale=0.1)
    params["zeta_base_r"].data[:50] = 0.0  # rows in the translation-only gauge branch
    params["is_static"] = (torch.rand(n, 1, generator=g) < 0.2).float().to(dev)
    return params


def _render(trainer, params, idx, t, sh_degree, fused: bool):  # noqa: ANN001, ANN202
    from trigs.core.trainer import TrigsTrainingConfig

    trainer.fused_deform = fused
    trainer._active_idx_for_time = lambda *a, **k: idx
    out = trainer._render_inputs(params, None, 0, t, sh_degree, TrigsTrainingConfig())
    keys = ("means", "quats", "scales", "opacities", "colors", "gamma")
    loss = sum((out[k] * torch.linspace(0.5, 1.5, out[k].numel(), device=out[k].device).reshape(out[k].shape)).sum() for k in keys)
    grads = torch.autograd.grad(loss, [v for v in params.values() if v.requires_grad], allow_unused=True)
    return {k: out[k].detach() for k in keys} | {"sh_degree": out["sh_degree"]}, grads


@pytest.mark.parametrize("sh_degree", [0, 3])
@pytest.mark.parametrize("t", [1.7, 0.0])
def test_fused_render_inputs_equal_trigs(sh_degree: int, t: float) -> None:
    from open4dgs.trigs_trainer import Open4dgsTrainer

    params = _params("cpu")
    idx = torch.randperm(4000, generator=torch.Generator().manual_seed(1))[:2500].sort().values
    trainer = Open4dgsTrainer(device="cpu")
    ref, ref_g = _render(trainer, params, idx, t, sh_degree, fused=False)
    got, got_g = _render(trainer, params, idx, t, sh_degree, fused=True)
    assert got["sh_degree"] == ref["sh_degree"]
    for k in ("means", "quats", "scales", "opacities", "colors", "gamma"):
        torch.testing.assert_close(got[k], ref[k], rtol=1e-6, atol=1e-6)
    for a, b in zip(got_g, ref_g, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.integration
def test_compiled_fused_render_inputs_equal_trigs_on_cuda() -> None:
    """The compiled graph (CUDA) against TRiGS eager, over active sets of different lengths (one dynamic graph)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (torch.compile / Triton)")
    from open4dgs.trigs_trainer import Open4dgsTrainer

    params = _params("cuda")
    trainer = Open4dgsTrainer(device="cuda")
    for i, m in enumerate((2500, 3100, 1900)):
        idx = torch.randperm(4000, generator=torch.Generator().manual_seed(i))[:m].sort().values.cuda()
        t = torch.tensor(0.5 + i, device="cuda")
        ref, ref_g = _render(trainer, params, idx, t, 3, fused=False)
        got, got_g = _render(trainer, params, idx, t, 3, fused=True)
        for k in ("means", "quats", "scales", "opacities", "colors", "gamma"):
            torch.testing.assert_close(got[k], ref[k], rtol=1e-5, atol=1e-5)
        for a, b in zip(got_g, ref_g, strict=True):
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
