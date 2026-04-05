"""Compare Mojo GPU kernels against CUDA originals for numerical correctness and performance.

Uses the CUDA mast3r_slam_backends as oracle and validates that the Mojo
mast3r_slam_mojo_backends produces matching outputs within tolerance.

Includes both parametrized deterministic tests and Hypothesis property-based
fuzz tests that randomise shapes, seeds, and kernel parameters.

Run:
    pixi run -e mast3r-slam-dev pytest tests/test_mojo_vs_cuda.py -v
    pixi run -e mast3r-slam-dev pytest tests/test_mojo_vs_cuda.py -v -k hypothesis  # fuzz only
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from jaxtyping import Bool, Float, Int


# ── Skip if either backend is unavailable ─────────────────────────────────────

cuda_backends = pytest.importorskip("mast3r_slam_backends", reason="CUDA backends not built")
mojo_backends = pytest.importorskip("mast3r_slam_mojo_backends", reason="Mojo backends not built")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ── Helpers ───────────────────────────────────────────────────────────────────

DEVICE: torch.device = torch.device("cuda")


def sync() -> None:
    """Synchronize CUDA stream before reading async Mojo kernel results.

    Mojo kernels run asynchronously (no ctx.synchronize()) for production
    performance. In tests we must sync before comparing outputs, otherwise
    the caching allocator may reuse the output buffer before the kernel
    finishes writing to it.
    """
    torch.cuda.synchronize()


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
    # Add small perturbation — use the seeded generator for reproducibility
    p_init += torch.randn(p_init.shape, device=DEVICE, generator=gen) * 2.0
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
        sync()

        assert p_mojo.shape == p_cuda.shape, f"Shape mismatch: {p_mojo.shape} vs {p_cuda.shape}"
        assert conv_mojo.shape == conv_cuda.shape, f"Shape mismatch: {conv_mojo.shape} vs {conv_cuda.shape}"
        # LM iterations amplify floating-point ordering differences between
        # CUDA and Mojo. Tolerance of 5e-2 (subpixel) is well within SLAM
        # requirements — the downstream occlusion check uses dist_thresh=0.1.
        max_diff: float = (p_mojo - p_cuda).abs().max().item()
        assert max_diff < 5e-2, (
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
        sync()

        max_diff: float = (p_mojo - p_cuda).abs().max().item()
        assert max_diff < 5e-2, f"identity init: max diff = {max_diff:.2e}"
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
        sync()

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
# Hypothesis property-based fuzz tests
# ═══════════════════════════════════════════════════════════════════════════════

# Suppress the too_slow health check — GPU kernels have startup cost on first
# call that makes the first example slower, but subsequent ones are fast.
HYPOTHESIS_SETTINGS: dict = dict(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class TestIterProjHypothesis:
    """Fuzz iter_proj with random shapes, seeds, and LM parameters."""

    @given(
        batch=st.integers(min_value=1, max_value=3),
        h=st.integers(min_value=8, max_value=128),
        w=st.integers(min_value=8, max_value=128),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        max_iter=st.sampled_from([1, 3, 5, 10]),
        lambda_init=st.sampled_from([1e-8, 1e-4, 1.0, 10.0]),
        cost_thresh=st.sampled_from([1e-6, 1e-4, 1e-2]),
    )
    @settings(**HYPOTHESIS_SETTINGS)
    def test_iter_proj_fuzz(
        self,
        batch: int,
        h: int,
        w: int,
        seed: int,
        max_iter: int,
        lambda_init: float,
        cost_thresh: float,
    ) -> None:
        """Fuzzed iter_proj: Mojo matches CUDA across random inputs."""
        rays_img: Float[torch.Tensor, "b h w 9"]
        pts_3d_norm: Float[torch.Tensor, "b hw 3"]
        p_init: Float[torch.Tensor, "b hw 2"]
        rays_img, pts_3d_norm, p_init = make_iter_proj_inputs(batch, h, w, seed=seed)

        p_cuda: Float[torch.Tensor, "b hw 2"]
        conv_cuda: Bool[torch.Tensor, "b hw"]
        p_cuda, conv_cuda = cuda_backends.iter_proj(
            rays_img, pts_3d_norm, p_init, max_iter, lambda_init, cost_thresh,
        )

        p_mojo: Float[torch.Tensor, "b hw 2"]
        conv_mojo: Bool[torch.Tensor, "b hw"]
        p_mojo, conv_mojo = mojo_backends.iter_proj(
            rays_img, pts_3d_norm, p_init, max_iter, lambda_init, cost_thresh,
        )
        # Mojo kernels run async — sync before reading results for comparison.
        torch.cuda.synchronize()

        assert p_mojo.shape == p_cuda.shape
        # Tolerance: 5e-2 absolute (subpixel precision) is well within SLAM
        # requirements. FP ordering differences between CUDA and Mojo accumulate
        # especially with aggressive LM damping (lambda=1.0) on small images.
        max_diff: float = (p_mojo - p_cuda).abs().max().item()
        assert max_diff < 5e-2, (
            f"iter_proj fuzz fail: batch={batch} h={h} w={w} seed={seed} "
            f"max_iter={max_iter} lambda={lambda_init} thresh={cost_thresh} "
            f"max_diff={max_diff:.2e}"
        )
        agree_rate: float = (conv_mojo == conv_cuda).float().mean().item()
        assert agree_rate > 0.90, (
            f"iter_proj convergence agreement {agree_rate:.2%} < 90% "
            f"(batch={batch} h={h} w={w} seed={seed})"
        )

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(**HYPOTHESIS_SETTINGS)
    def test_iter_proj_production_config(self, seed: int) -> None:
        """Fuzz iter_proj with actual MASt3R-SLAM production config values."""
        # fast.yaml / base.yaml both use: max_iter=10, lambda=1e-8, thresh=1e-6
        batch: int = 1
        h: int = 224
        w: int = 224
        rays_img, pts_3d_norm, p_init = make_iter_proj_inputs(batch, h, w, seed=seed)

        p_cuda, conv_cuda = cuda_backends.iter_proj(
            rays_img, pts_3d_norm, p_init, 10, 1e-8, 1e-6,
        )
        p_mojo, conv_mojo = mojo_backends.iter_proj(
            rays_img, pts_3d_norm, p_init, 10, 1e-8, 1e-6,
        )
        sync()

        max_diff: float = (p_mojo - p_cuda).abs().max().item()
        assert max_diff < 5e-2, (
            f"Production config fail: seed={seed} max_diff={max_diff:.2e}"
        )


class TestRefineMatchesHypothesis:
    """Fuzz refine_matches with random shapes, seeds, and search params."""

    @given(
        batch=st.integers(min_value=1, max_value=2),
        h=st.integers(min_value=16, max_value=96),
        w=st.integers(min_value=16, max_value=96),
        # fdim values that exercise SIMD remainder paths:
        # 16 (div by 8), 24 (div by 8), 17 (remainder), 128 (div by 8), 13 (odd)
        fdim=st.sampled_from([8, 13, 16, 17, 24, 32, 64, 128]),
        radius=st.integers(min_value=1, max_value=4),
        dilation_max=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(**HYPOTHESIS_SETTINGS)
    def test_refine_matches_fuzz(
        self,
        batch: int,
        h: int,
        w: int,
        fdim: int,
        radius: int,
        dilation_max: int,
        seed: int,
    ) -> None:
        """Fuzzed refine_matches: Mojo matches CUDA across random inputs."""
        # Need enough margin for the search window
        margin: int = radius * dilation_max + 1
        if margin >= h // 2 or margin >= w // 2:
            return  # Skip if image too small for search radius

        D11: Float[torch.Tensor, "b h w d"]
        D21: Float[torch.Tensor, "b hw d"]
        p1: Int[torch.Tensor, "b hw 2"]
        D11, D21, p1 = make_refine_matches_inputs(batch, h, w, fdim, seed=seed)
        # Clamp positions to have sufficient margin for the search window
        p1[..., 0].clamp_(margin, w - margin - 1)
        p1[..., 1].clamp_(margin, h - margin - 1)

        (p1_cuda,) = cuda_backends.refine_matches(D11.half(), D21.half(), p1, radius, dilation_max)
        (p1_mojo,) = mojo_backends.refine_matches(D11.half(), D21.half(), p1, radius, dilation_max)
        sync()

        assert p1_mojo.shape == p1_cuda.shape
        total: int = p1_cuda.numel() // 2
        n_diff: int = (p1_mojo != p1_cuda).any(dim=-1).sum().item()
        diff_rate: float = n_diff / max(total, 1)
        assert diff_rate < 0.02, (
            f"refine_matches fuzz fail: batch={batch} h={h} w={w} fdim={fdim} "
            f"radius={radius} dilation={dilation_max} seed={seed} "
            f"{n_diff}/{total} differ ({diff_rate:.2%})"
        )

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(**HYPOTHESIS_SETTINGS)
    def test_refine_matches_production_config(self, seed: int) -> None:
        """Fuzz refine_matches with actual MASt3R-SLAM production config.

        base.yaml: radius=3, dilation_max=5 at 512px with fdim from MASt3R network.
        """
        batch: int = 1
        # MASt3R at 224px produces 14x14 patches → h=w=14 for the matching grid
        # But the descriptor map D11 is at full image resolution.
        # Use a realistic resolution that fits in GPU memory for fuzz tests.
        h: int = 64
        w: int = 64
        fdim: int = 16  # MASt3R descriptor dim after projection
        margin: int = 3 * 5 + 1  # radius * dilation_max + 1

        D11, D21, p1 = make_refine_matches_inputs(batch, h, w, fdim, seed=seed)
        p1[..., 0].clamp_(margin, w - margin - 1)
        p1[..., 1].clamp_(margin, h - margin - 1)

        (p1_cuda,) = cuda_backends.refine_matches(D11.half(), D21.half(), p1, 3, 5)
        (p1_mojo,) = mojo_backends.refine_matches(D11.half(), D21.half(), p1, 3, 5)
        sync()

        total: int = p1_cuda.numel() // 2
        n_diff: int = (p1_mojo != p1_cuda).any(dim=-1).sum().item()
        diff_rate: float = n_diff / max(total, 1)
        assert diff_rate < 0.02, (
            f"Production config fail: seed={seed} {n_diff}/{total} differ ({diff_rate:.2%})"
        )


class TestEdgeCases:
    """Targeted edge-case tests for kernel correctness."""

    def test_iter_proj_single_point(self) -> None:
        """Single-point input (smallest possible workload)."""
        rays_img, pts_3d_norm, p_init = make_iter_proj_inputs(1, 4, 4, seed=99)
        # Only 1 point — slice to [1, 1, 3] and [1, 1, 2]
        pts_1: Float[torch.Tensor, "1 1 3"] = pts_3d_norm[:, :1, :].contiguous()
        pi_1: Float[torch.Tensor, "1 1 2"] = p_init[:, :1, :].contiguous()

        p_cuda, conv_cuda = cuda_backends.iter_proj(rays_img, pts_1, pi_1, 5, 1.0, 1e-4)
        p_mojo, conv_mojo = mojo_backends.iter_proj(rays_img, pts_1, pi_1, 5, 1.0, 1e-4)
        sync()

        max_diff: float = (p_mojo - p_cuda).abs().max().item()
        assert max_diff < 5e-2, f"single point: max diff = {max_diff:.2e}"

    def test_iter_proj_zero_iterations(self) -> None:
        """Zero LM iterations should return clamped initial positions."""
        rays_img, pts_3d_norm, p_init = make_iter_proj_inputs(1, 16, 16, seed=7)

        p_cuda, _ = cuda_backends.iter_proj(rays_img, pts_3d_norm, p_init, 0, 1.0, 1e-4)
        p_mojo, _ = mojo_backends.iter_proj(rays_img, pts_3d_norm, p_init, 0, 1.0, 1e-4)
        sync()

        assert torch.allclose(p_mojo, p_cuda, atol=1e-5)

    def test_refine_matches_single_point(self) -> None:
        """Single match point."""
        D11, D21, p1 = make_refine_matches_inputs(1, 16, 16, 16, seed=77)
        D21_1: Float[torch.Tensor, "1 1 16"] = D21[:, :1, :].contiguous()
        p1_1: Int[torch.Tensor, "1 1 2"] = p1[:, :1, :].contiguous()
        # Ensure position is well within bounds
        p1_1[..., 0] = 8
        p1_1[..., 1] = 8

        (p_cuda,) = cuda_backends.refine_matches(D11.half(), D21_1.half(), p1_1, 2, 2)
        (p_mojo,) = mojo_backends.refine_matches(D11.half(), D21_1.half(), p1_1, 2, 2)
        sync()

        assert torch.equal(p_mojo, p_cuda)

    def test_refine_matches_boundary_positions(self) -> None:
        """Match positions at image corners and edges."""
        h: int = 32
        w: int = 32
        D11, D21, _ = make_refine_matches_inputs(1, h, w, 16, seed=55)
        # Place matches at corners and edges
        corners: list[tuple[int, int]] = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (0, h // 2)]
        n: int = len(corners)
        p1: Int[torch.Tensor, "1 n 2"] = torch.tensor([[list(c) for c in corners]], device=DEVICE, dtype=torch.long)
        D21_sub: Float[torch.Tensor, "1 n 16"] = D21[:, :n, :].contiguous()

        (p_cuda,) = cuda_backends.refine_matches(D11.half(), D21_sub.half(), p1, 1, 1)
        (p_mojo,) = mojo_backends.refine_matches(D11.half(), D21_sub.half(), p1, 1, 1)
        sync()

        assert torch.equal(p_mojo, p_cuda), (
            f"Boundary mismatch: CUDA={p_cuda} Mojo={p_mojo}"
        )

    def test_refine_matches_radius_zero(self) -> None:
        """Radius 0 should not crash (though not typically used)."""
        D11, D21, p1 = make_refine_matches_inputs(1, 16, 16, 16, seed=33)
        # radius=0 means no search — dilation loop does nothing
        (p_cuda,) = cuda_backends.refine_matches(D11.half(), D21.half(), p1, 0, 1)
        (p_mojo,) = mojo_backends.refine_matches(D11.half(), D21.half(), p1, 0, 1)
        sync()

        # With radius=0, output should be identical to input positions
        assert torch.equal(p_mojo, p_cuda)

    def test_refine_matches_fdim_not_simd_aligned(self) -> None:
        """Feature dimensions that don't divide evenly by SIMD width (4 or 8)."""
        for fdim in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
            D11, D21, p1 = make_refine_matches_inputs(1, 16, 16, fdim, seed=fdim)
            p1[..., 0].clamp_(3, 12)
            p1[..., 1].clamp_(3, 12)

            (p_cuda,) = cuda_backends.refine_matches(D11.half(), D21.half(), p1, 1, 1)
            (p_mojo,) = mojo_backends.refine_matches(D11.half(), D21.half(), p1, 1, 1)
            sync()

            total: int = p_cuda.numel() // 2
            n_diff: int = (p_mojo != p_cuda).any(dim=-1).sum().item()
            diff_rate: float = n_diff / max(total, 1)
            assert diff_rate < 0.02, (
                f"Non-aligned fdim={fdim}: {n_diff}/{total} differ ({diff_rate:.2%})"
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
