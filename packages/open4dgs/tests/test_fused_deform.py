"""``Open4dgsTrainer(fused_deform=True)._render_inputs`` (``open4dgs.fused_deform.gather_deform``) equals TRiGS's own
gather + deform + rasterizer-input prep: values and every parameter's gradient."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")


def _params(dev: str, n: int = 4000) -> torch.nn.ParameterDict:
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
    params["is_static"] = torch.nn.Parameter((torch.rand(n, 1, generator=g) < 0.2).float().to(dev), requires_grad=False)
    return torch.nn.ParameterDict(params)


def _cameras(dev: str):  # noqa: ANN202
    from open4dgs.trigs_adapter import camera_config

    K = torch.tensor([[110.0, 0.0, 63.2], [0.0, 105.0, 47.7], [0.0, 0.0, 1.0]])[None]
    return camera_config(torch.eye(4)[None], K, [(128, 96)], torch.device(dev))


def _render(trainer, params, idx, t, sh_degree, fused: bool):  # noqa: ANN001, ANN202
    from trigs.core.trainer import TrigsTrainingConfig

    trainer.fused_deform = fused
    trainer._active_idx_for_time = lambda *a, **k: idx
    out = trainer._render_inputs(params, _cameras(str(params["means"].device)), 0, t, sh_degree, TrigsTrainingConfig())
    keys = ("means", "quats", "scales", "opacities", "colors", "gamma")
    loss = torch.stack([(out[k] * torch.linspace(0.5, 1.5, out[k].numel(), device=out[k].device).reshape(out[k].shape)).sum() for k in keys]).sum()
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


def check_compiled_render_inputs_on_cuda() -> None:
    """The compiled graph (CUDA) against TRiGS eager, over active sets of different lengths, with no recompile after the
    first length (the active-row dim is dynamic). Runs in a child process: under pytest ``PIXI_DEV_MODE`` selects eager."""
    from open4dgs.fused_deform import compiled_gather_deform, gather_deform
    from open4dgs.trigs_trainer import Open4dgsTrainer

    assert compiled_gather_deform(torch.device("cuda")) is not gather_deform
    params = _params("cuda")
    trainer = Open4dgsTrainer(device="cuda")
    graphs: list[int] = []
    for i, m in enumerate((2500, 3100, 1900)):
        idx = torch.randperm(4000, generator=torch.Generator().manual_seed(i))[:m].sort().values.cuda()
        t = torch.tensor(0.5 + i, device="cuda")
        ref, ref_g = _render(trainer, params, idx, t, 3, fused=False)
        got, got_g = _render(trainer, params, idx, t, 3, fused=True)
        graphs.append(int(torch._dynamo.utils.counters["stats"]["unique_graphs"]))
        for k in ("means", "quats", "scales", "opacities", "colors", "gamma"):
            torch.testing.assert_close(got[k], ref[k], rtol=1e-5, atol=1e-5)
        for a, b in zip(got_g, ref_g, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    assert graphs[0] >= 1 and graphs[1:] == [graphs[0]] * 2, f"compiled graphs after each length: {graphs}"


@pytest.mark.integration
def test_compiled_fused_render_inputs_equal_trigs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (torch.compile / Triton)")
    env: dict[str, str] = dict(os.environ)
    env["PIXI_DEV_MODE"] = "0"
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", "import test_fused_deform as m; m.check_compiled_render_inputs_on_cuda()"], check=True, env=env)
