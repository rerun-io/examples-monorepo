from __future__ import annotations

import pytest
import torch
from torch import Tensor
import dpvo

cuda_corr = pytest.importorskip("dpvo._cuda_corr", reason="DPVO CUDA altcorr extension is not built")
mojo_corr = pytest.importorskip("dpvo_altcorr_mojo.backend", reason="dpvo_altcorr_mojo is not importable")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _assert_close(name: str, got: Tensor, expected: Tensor, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    diff = (got - expected).abs()
    assert torch.allclose(got, expected, atol=atol, rtol=rtol), (
        f"{name} mismatch: max_abs={float(diff.max()):.6g}, mean_abs={float(diff.mean()):.6g}, "
        f"shape={tuple(got.shape)}"
    )


def _sync() -> None:
    torch.cuda.synchronize()


def _disable_cuda_corr_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoCudaCorr:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"Mojo altcorr unexpectedly called CUDA fallback {name}")

    monkeypatch.setattr(dpvo, "_cuda_corr", NoCudaCorr())


def test_patchify_forward_matches_cuda() -> None:
    gen = torch.Generator(device="cuda").manual_seed(42)
    net = torch.randn(2, 4, 8, 9, device="cuda", generator=gen)
    coords = torch.tensor(
        [
            [[2.25, 3.75], [0.2, 0.8], [7.5, 6.1]],
            [[1.0, 1.0], [5.5, 5.25], [8.2, 7.7]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    for radius in (0, 1):
        (expected,) = cuda_corr.patchify_forward(net, coords, radius)
        _sync()
        (got,) = mojo_corr.patchify_forward(net, coords, radius)
        _sync()
        _assert_close(f"patchify_forward_r{radius}", got, expected)


def test_patchify_forward_half_matches_cuda_without_cuda_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = torch.Generator(device="cuda").manual_seed(46)
    net = torch.randn(2, 4, 8, 9, device="cuda", generator=gen, dtype=torch.float16)
    coords = torch.tensor(
        [
            [[2.25, 3.75], [0.2, 0.8], [7.5, 6.1]],
            [[1.0, 1.0], [5.5, 5.25], [8.2, 7.7]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    (expected,) = cuda_corr.patchify_forward(net, coords, 1)
    _sync()
    _disable_cuda_corr_fallback(monkeypatch)
    (got,) = mojo_corr.patchify_forward(net, coords, 1)
    _sync()
    assert got.dtype == torch.float16
    _assert_close("patchify_forward_half", got, expected)


def test_patchify_backward_matches_cuda() -> None:
    gen = torch.Generator(device="cuda").manual_seed(43)
    net = torch.randn(2, 3, 7, 8, device="cuda", generator=gen)
    coords = torch.tensor(
        [
            [[2.25, 3.75], [0.2, 0.8], [6.5, 5.1]],
            [[1.0, 1.0], [4.5, 4.25], [7.2, 6.7]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    radius = 1
    diameter = 2 * radius + 2
    grad = torch.randn(2, 3, 3, diameter, diameter, device="cuda", generator=gen)
    (expected,) = cuda_corr.patchify_backward(net, coords, grad, radius)
    _sync()
    (got,) = mojo_corr.patchify_backward(net, coords, grad, radius)
    _sync()
    _assert_close("patchify_backward", got, expected, atol=2e-4, rtol=2e-4)


def _corr_inputs(seed: int = 44) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    fmap1 = torch.randn(1, 4, 8, 3, 3, device="cuda", generator=gen)
    fmap2 = torch.randn(1, 3, 8, 7, 8, device="cuda", generator=gen)
    coords = torch.rand(1, 5, 2, 3, 3, device="cuda", generator=gen) * 4.0 + 1.0
    ii = torch.tensor([0, 1, 2, 3, 0], device="cuda", dtype=torch.int64)
    jj = torch.tensor([0, 1, 2, 0, 2], device="cuda", dtype=torch.int64)
    return fmap1, fmap2, coords, ii, jj


def test_corr_forward_matches_cuda() -> None:
    fmap1, fmap2, coords, ii, jj = _corr_inputs()
    radius = 1
    (expected,) = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
    _sync()
    (got,) = mojo_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
    _sync()
    _assert_close("corr_forward", got, expected, atol=2e-5, rtol=2e-5)


def test_corr_forward_half_matches_cuda_without_cuda_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fmap1, fmap2, coords, ii, jj = _corr_inputs(seed=47)
    fmap1 = fmap1.half()
    fmap2 = fmap2.half()
    radius = 1
    (expected,) = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
    _sync()
    _disable_cuda_corr_fallback(monkeypatch)
    (got,) = mojo_corr.forward(fmap1, fmap2, coords, ii, jj, radius)
    _sync()
    assert got.dtype == torch.float16
    _assert_close("corr_forward_half", got, expected, atol=2e-2, rtol=2e-2)


def test_corr_backward_matches_cuda() -> None:
    gen = torch.Generator(device="cuda").manual_seed(45)
    fmap1, fmap2, coords, ii, jj = _corr_inputs()
    radius = 1
    grad = torch.randn(1, 5, 2 * radius + 1, 2 * radius + 1, 3, 3, device="cuda", generator=gen)
    expected_fmap1, expected_fmap2 = cuda_corr.backward(fmap1, fmap2, coords, ii, jj, grad, radius)
    _sync()
    got_fmap1, got_fmap2 = mojo_corr.backward(fmap1, fmap2, coords, ii, jj, grad, radius)
    _sync()
    _assert_close("corr_backward_fmap1", got_fmap1, expected_fmap1, atol=2e-4, rtol=2e-4)
    _assert_close("corr_backward_fmap2", got_fmap2, expected_fmap2, atol=2e-4, rtol=2e-4)
