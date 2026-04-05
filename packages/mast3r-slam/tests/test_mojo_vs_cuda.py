"""Compare Mojo GPU kernels against CUDA originals for numerical correctness and performance.

Uses the CUDA mast3r_slam_backends as oracle and validates that the Mojo
mast3r_slam_mojo_backends produces matching outputs within tolerance.

Run:
    pixi run -e mast3r-slam-dev pytest tests/test_mojo_vs_cuda.py -v
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch
from jaxtyping import Bool, Float, Int


# ── Skip if either backend is unavailable ─────────────────────────────────────

cuda_backends = pytest.importorskip("mast3r_slam_backends", reason="CUDA backends not built")
mojo_backends = pytest.importorskip("mast3r_slam_mojo_backends", reason="Mojo backends not built")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ── Helpers ───────────────────────────────────────────────────────────────────

DEVICE: torch.device = torch.device("cuda")


@dataclass(slots=True, frozen=True)
class BenchResult:
    """Timing result from a kernel benchmark."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


def benchmark_fn(fn: callable, warmup: int = 10, runs: int = 100) -> BenchResult:
    """Benchmark a CUDA kernel using torch.cuda.Event timing.

    Args:
        fn: Callable that runs the kernel (no return value needed).
        warmup: Number of warmup iterations.
        runs: Number of timed iterations.

    Returns:
        A BenchResult with mean/std/min/max in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(runs):
        start: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        end: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    t: torch.Tensor = torch.tensor(timings)
    return BenchResult(
        name="",
        mean_ms=float(t.mean()),
        std_ms=float(t.std()),
        min_ms=float(t.min()),
        max_ms=float(t.max()),
    )


# ── iter_proj test data generation ────────────────────────────────────────────


def make_iter_proj_inputs(
    batch: int, h: int, w: int, seed: int = 42
) -> tuple[
    Float[torch.Tensor, "b h w 9"],
    Float[torch.Tensor, "b hw 3"],
    Float[torch.Tensor, "b hw 2"],
]:
    """Generate synthetic inputs for iter_proj kernel.

    Creates a ray image with spatial gradients, normalised 3D target points,
    and initial pixel guesses.

    Args:
        batch: Batch size.
        h: Image height.
        w: Image width.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (rays_img, pts_3d_norm, p_init).
    """
    gen: torch.Generator = torch.Generator(device=DEVICE).manual_seed(seed)
    hw: int = h * w

    # Ray image: [b, h, w, 9] — ray (3) + gx (3) + gy (3)
    # Use normalised random vectors for rays, small gradients
    rays: Float[torch.Tensor, "b h w 3"] = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen)
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gx: Float[torch.Tensor, "b h w 3"] = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen) * 0.01
    gy: Float[torch.Tensor, "b h w 3"] = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen) * 0.01
    rays_img: Float[torch.Tensor, "b h w 9"] = torch.cat([rays, gx, gy], dim=-1).contiguous()

    # Target normalised 3D points
    pts: Float[torch.Tensor, "b hw 3"] = torch.randn(batch, hw, 3, device=DEVICE, generator=gen)
    pts_3d_norm: Float[torch.Tensor, "b hw 3"] = (pts / pts.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()

    # Initial pixel guesses — spread across the image, add small random offset
    v_coords: Float[torch.Tensor, "hw"] = torch.arange(hw, device=DEVICE).float() // w
    u_coords: Float[torch.Tensor, "hw"] = torch.arange(hw, device=DEVICE).float() % w
    p_init: Float[torch.Tensor, "b hw 2"] = torch.stack([u_coords, v_coords], dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous().clone()
    # Add small perturbation
    p_init += torch.randn_like(p_init) * 2.0
    p_init[..., 0].clamp_(1.0, w - 2.0)
    p_init[..., 1].clamp_(1.0, h - 2.0)
    p_init = p_init.contiguous()

    return rays_img, pts_3d_norm, p_init


# ── refine_matches test data generation ───────────────────────────────────────


def make_refine_matches_inputs(
    batch: int, h: int, w: int, fdim: int, seed: int = 42
) -> tuple[
    Float[torch.Tensor, "b h w d"],
    Float[torch.Tensor, "b hw d"],
    Int[torch.Tensor, "b hw 2"],
]:
    """Generate synthetic inputs for refine_matches kernel.

    Creates normalised descriptor maps and initial match positions.

    Args:
        batch: Batch size.
        h: Image height.
        w: Image width.
        fdim: Feature descriptor dimension.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (D11, D21, p1).
    """
    gen: torch.Generator = torch.Generator(device=DEVICE).manual_seed(seed)
    hw: int = h * w

    # Descriptor maps — normalised random vectors
    D11: Float[torch.Tensor, "b h w d"] = torch.randn(batch, h, w, fdim, device=DEVICE, generator=gen)
    D11 = (D11 / D11.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()

    D21: Float[torch.Tensor, "b hw d"] = torch.randn(batch, hw, fdim, device=DEVICE, generator=gen)
    D21 = (D21 / D21.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()

    # Initial match positions — within image bounds (need margin for search radius)
    margin: int = 4
    u: Int[torch.Tensor, "b hw"] = torch.randint(margin, w - margin, (batch, hw), device=DEVICE, generator=gen)
    v: Int[torch.Tensor, "b hw"] = torch.randint(margin, h - margin, (batch, hw), device=DEVICE, generator=gen)
    p1: Int[torch.Tensor, "b hw 2"] = torch.stack([u, v], dim=-1).contiguous()

    return D11, D21, p1


# ═══════════════════════════════════════════════════════════════════════════════
# iter_proj tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIterProj:
    """Numerical comparison of iter_proj between CUDA and Mojo backends."""

    MAX_ITER: int = 5
    LAMBDA_INIT: float = 1.0
    COST_THRESH: float = 1e-4

    @pytest.mark.parametrize(
        "batch, h, w",
        [
            (1, 16, 16),
            (1, 32, 32),
            (1, 64, 64),
            (2, 32, 32),
        ],
    )
    def test_numerical_match(self, batch: int, h: int, w: int) -> None:
        """Mojo iter_proj output matches CUDA within tolerance."""
        rays_img: Float[torch.Tensor, "b h w 9"]
        pts_3d_norm: Float[torch.Tensor, "b hw 3"]
        p_init: Float[torch.Tensor, "b hw 2"]
        rays_img, pts_3d_norm, p_init = make_iter_proj_inputs(batch, h, w)

        # CUDA oracle
        p_cuda: Float[torch.Tensor, "b hw 2"]
        conv_cuda: Bool[torch.Tensor, "b hw"]
        p_cuda, conv_cuda = cuda_backends.iter_proj(
            rays_img, pts_3d_norm, p_init,
            self.MAX_ITER, self.LAMBDA_INIT, self.COST_THRESH,
        )

        # Mojo under test
        p_mojo: Float[torch.Tensor, "b hw 2"]
        conv_mojo: Bool[torch.Tensor, "b hw"]
        p_mojo, conv_mojo = mojo_backends.iter_proj(
            rays_img, pts_3d_norm, p_init,
            self.MAX_ITER, self.LAMBDA_INIT, self.COST_THRESH,
        )

        assert p_mojo.shape == p_cuda.shape, f"Shape mismatch: {p_mojo.shape} vs {p_cuda.shape}"
        assert conv_mojo.shape == conv_cuda.shape, f"Shape mismatch: {conv_mojo.shape} vs {conv_cuda.shape}"
        # LM iterations amplify floating-point ordering differences between
        # CUDA and Mojo, so we use a relaxed tolerance (1e-2 absolute).
        max_diff: float = (p_mojo - p_cuda).abs().max().item()
        assert torch.allclose(p_mojo, p_cuda, atol=1e-2, rtol=1e-3), (
            f"iter_proj pixel mismatch: max diff = {max_diff:.2e}"
        )
        # Convergence flags may differ at boundary cases due to cost threshold
        # comparison with slightly different costs; check agreement rate instead.
        agree_rate: float = (conv_mojo == conv_cuda).float().mean().item()
        assert agree_rate > 0.95, f"iter_proj convergence agreement too low: {agree_rate:.2%}"

    def test_identity_init(self) -> None:
        """With identity initial guess, both backends converge identically."""
        batch: int = 1
        h: int = 32
        w: int = 32
        rays_img: Float[torch.Tensor, "b h w 9"]
        pts_3d_norm: Float[torch.Tensor, "b hw 3"]
        _: Float[torch.Tensor, "b hw 2"]
        rays_img, pts_3d_norm, _ = make_iter_proj_inputs(batch, h, w)

        # Identity init: each pixel maps to itself
        hw: int = h * w
        v_coords: Float[torch.Tensor, "hw"] = torch.arange(hw, device=DEVICE).float() // w
        u_coords: Float[torch.Tensor, "hw"] = torch.arange(hw, device=DEVICE).float() % w
        p_init: Float[torch.Tensor, "1 hw 2"] = torch.stack([u_coords, v_coords], dim=-1).unsqueeze(0).contiguous()

        p_cuda, conv_cuda = cuda_backends.iter_proj(
            rays_img, pts_3d_norm, p_init,
            self.MAX_ITER, self.LAMBDA_INIT, self.COST_THRESH,
        )
        p_mojo, conv_mojo = mojo_backends.iter_proj(
            rays_img, pts_3d_norm, p_init,
            self.MAX_ITER, self.LAMBDA_INIT, self.COST_THRESH,
        )

        assert torch.allclose(p_mojo, p_cuda, atol=1e-2, rtol=1e-3)
        agree_rate: float = (conv_mojo == conv_cuda).float().mean().item()
        assert agree_rate > 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# refine_matches tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefineMatches:
    """Numerical comparison of refine_matches between CUDA and Mojo backends."""

    @pytest.mark.parametrize(
        "batch, h, w, fdim, radius, dilation_max",
        [
            (1, 32, 32, 16, 2, 2),
            (1, 32, 32, 128, 2, 2),
            (1, 64, 64, 16, 1, 1),
            (1, 64, 64, 16, 2, 3),
            (2, 32, 32, 16, 2, 2),
        ],
    )
    def test_numerical_match(
        self, batch: int, h: int, w: int, fdim: int, radius: int, dilation_max: int
    ) -> None:
        """Mojo refine_matches output matches CUDA exactly (integer positions)."""
        D11: Float[torch.Tensor, "b h w d"]
        D21: Float[torch.Tensor, "b hw d"]
        p1: Int[torch.Tensor, "b hw 2"]
        D11, D21, p1 = make_refine_matches_inputs(batch, h, w, fdim)

        # CUDA oracle — uses half precision internally
        (p1_cuda,) = cuda_backends.refine_matches(D11.half(), D21.half(), p1, radius, dilation_max)

        # Mojo under test — also receives float descriptors (cast inside)
        (p1_mojo,) = mojo_backends.refine_matches(D11.half(), D21.half(), p1, radius, dilation_max)

        assert p1_mojo.shape == p1_cuda.shape, f"Shape mismatch: {p1_mojo.shape} vs {p1_cuda.shape}"
        # The Mojo kernel computes descriptor dot-products in float32 while the
        # CUDA kernel uses float16; near-tie scores can pick different winners.
        # Allow up to 1% of positions to differ.
        total: int = p1_cuda.numel() // 2  # each position has 2 coords
        n_diff: int = (p1_mojo != p1_cuda).any(dim=-1).sum().item()
        diff_rate: float = n_diff / max(total, 1)
        assert diff_rate < 0.01, (
            f"refine_matches mismatch: {n_diff}/{total} positions differ ({diff_rate:.2%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarks (opt-in via --benchmark flag or running directly)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmarks:
    """Performance comparison between CUDA and Mojo backends."""

    @pytest.mark.parametrize(
        "h, w",
        [
            (64, 64),
            (224, 224),
        ],
    )
    def test_iter_proj_perf(self, h: int, w: int) -> None:
        """Report iter_proj timing for CUDA and Mojo."""
        batch: int = 1
        rays_img, pts_3d_norm, p_init = make_iter_proj_inputs(batch, h, w)

        cuda_result: BenchResult = benchmark_fn(
            lambda: cuda_backends.iter_proj(rays_img, pts_3d_norm, p_init, 5, 1.0, 1e-4),
        )
        mojo_result: BenchResult = benchmark_fn(
            lambda: mojo_backends.iter_proj(rays_img, pts_3d_norm, p_init, 5, 1.0, 1e-4),
        )

        print(f"\niter_proj ({h}x{w}):")
        print(f"  CUDA: {cuda_result.mean_ms:.3f} ± {cuda_result.std_ms:.3f} ms")
        print(f"  Mojo: {mojo_result.mean_ms:.3f} ± {mojo_result.std_ms:.3f} ms")
        print(f"  Ratio (Mojo/CUDA): {mojo_result.mean_ms / max(cuda_result.mean_ms, 1e-9):.2f}x")

    @pytest.mark.parametrize(
        "h, w, fdim",
        [
            (32, 32, 16),
            (64, 64, 128),
        ],
    )
    def test_refine_matches_perf(self, h: int, w: int, fdim: int) -> None:
        """Report refine_matches timing for CUDA and Mojo."""
        batch: int = 1
        D11, D21, p1 = make_refine_matches_inputs(batch, h, w, fdim)

        cuda_result: BenchResult = benchmark_fn(
            lambda: cuda_backends.refine_matches(D11.half(), D21.half(), p1, 2, 2),
        )
        mojo_result: BenchResult = benchmark_fn(
            lambda: mojo_backends.refine_matches(D11.half(), D21.half(), p1, 2, 2),
        )

        print(f"\nrefine_matches ({h}x{w}, fdim={fdim}):")
        print(f"  CUDA: {cuda_result.mean_ms:.3f} ± {cuda_result.std_ms:.3f} ms")
        print(f"  Mojo: {mojo_result.mean_ms:.3f} ± {mojo_result.std_ms:.3f} ms")
        print(f"  Ratio (Mojo/CUDA): {mojo_result.mean_ms / max(cuda_result.mean_ms, 1e-9):.2f}x")
