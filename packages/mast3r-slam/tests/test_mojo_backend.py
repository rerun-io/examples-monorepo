"""The Mojo kernels (mast3r_slam.mojo_backend) against the former CUDA extension's outputs on real MASt3R-SLAM inputs.

The references were frozen from `mast3r_slam._backends` on the 30-frame test run (provenance in
`kernel_references.py`). The kernels are CUDA-only, so every test is an integration test.

Tolerances:
- refine_matches: identical pixels (the Mojo kernel rounds its scores to half as the CUDA kernel does).
- iter_proj: identical convergence flags; pixels within 1e-3 for at least 99.5 % of the points. Fused
  multiply-adds round differently from nvcc's, and Levenberg-Marquardt's accept / reject step amplifies that
  for a few points (9 of 8,363 here, up to 0.55 px).
- Gauss-Newton: poses and the last increment within 1e-6 (measured 1.2e-7). Both implementations sum with
  float atomics, so neither is deterministic to the last bit.
"""

import pytest

torch = pytest.importorskip("torch", reason="MASt3R-SLAM requires PyTorch")
pytest.importorskip("rerun", reason="the references are an .rrd")
pytest.importorskip("mast3r_slam.mojo_backend._mast3r_mojo", reason="build the Mojo kernels first (pixi run -e mast3r-slam-dev _mast3r-slam-build-mojo-kernels)")

from kernel_references import REFERENCE_FILE, Record, load  # noqa: E402  # pyrefly: ignore[missing-import]
from torch import Tensor  # noqa: E402

from mast3r_slam import mojo_backend  # noqa: E402

if not REFERENCE_FILE.exists():
    pytest.skip(f"missing {REFERENCE_FILE.name}", allow_module_level=True)

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]

GN_TOLERANCE: float = 1e-6


def _on_cuda(rec: Record, *names: str) -> list[Tensor]:
    """Copies on the GPU: Gauss-Newton updates Twc in place, and the records are cached across tests."""
    return [rec.tensor(n).to("cuda", copy=True) for n in names]


def test_iter_proj_matches_cuda() -> None:
    rec: Record = load()["/iter_proj/late"]
    rays, pts, p_init = _on_cuda(rec, "rays", "pts", "p_init")
    max_iter, lambda_init, cost_thresh = rec.tensor("scalars").tolist()
    p_out, valid_out = mojo_backend.iter_proj(rays, pts, p_init, int(max_iter), lambda_init, cost_thresh)
    assert torch.equal(valid_out.cpu(), rec.tensor("valid_out"))
    close: Tensor = (p_out.cpu() - rec.tensor("p_out")).abs().amax(-1) <= 1e-3
    assert close.float().mean().item() >= 0.995, f"{int((~close).sum())} of {close.numel()} pixels differ by more than 1e-3"


def test_refine_matches_matches_cuda() -> None:
    rec: Record = load()["/refine_matches/late"]
    D11, D21, p1 = _on_cuda(rec, "D11", "D21", "p1")
    radius, dilation_max = (int(x) for x in rec.tensor("scalars").tolist())
    (p_out,) = mojo_backend.refine_matches(D11, D21, p1, radius, dilation_max)
    assert torch.equal(p_out.cpu(), rec.tensor("p_out"))


def test_gauss_newton_rays_matches_cuda() -> None:
    rec: Record = load()["/gn_rays/late"]
    Twc, Xs, Cs, ii, jj, idx, valid, Q = _on_cuda(rec, "Twc", "Xs", "Cs", "ii", "jj", "idx_ii2jj", "valid_match", "Q")
    sigma_ray, sigma_dist, C_thresh, Q_thresh, max_iter, delta_thresh = rec.tensor("scalars").tolist()
    dx: Tensor = mojo_backend.gauss_newton_rays(Twc, Xs, Cs, ii, jj, idx, valid, Q, sigma_ray, sigma_dist, C_thresh, Q_thresh, int(max_iter), delta_thresh)
    assert (Twc.cpu() - rec.tensor("Twc_out")).abs().max().item() <= GN_TOLERANCE
    assert (dx.cpu() - rec.tensor("dx_out")).abs().max().item() <= GN_TOLERANCE


def test_gauss_newton_calib_matches_cuda() -> None:
    rec: Record = load()["/gn_calib/late"]
    problem: Record = load()[rec.texts["problem"]]
    Twc, Cs, ii, jj, idx, valid, Q = _on_cuda(problem, "Twc", "Cs", "ii", "jj", "idx_ii2jj", "valid_match", "Q")
    Xs, K = _on_cuda(rec, "Xs", "K")
    height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh, max_iter, delta_thresh = rec.tensor("scalars").tolist()
    dx: Tensor = mojo_backend.gauss_newton_calib(
        Twc, Xs, Cs, K, ii, jj, idx, valid, Q, int(height), int(width), int(pixel_border), z_eps, sigma_pixel, sigma_depth,
        C_thresh, Q_thresh, int(max_iter), delta_thresh,
    )  # fmt: skip
    assert (Twc.cpu() - rec.tensor("Twc_out")).abs().max().item() <= GN_TOLERANCE
    assert (dx.cpu() - rec.tensor("dx_out")).abs().max().item() <= GN_TOLERANCE
