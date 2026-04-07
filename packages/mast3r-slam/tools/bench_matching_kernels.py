#!/usr/bin/env python
"""Benchmark CUDA vs Mojo matching kernels side-by-side.

Measures both backends under identical conditions using torch.cuda.Event
timing (median of 500 runs after 50 warmup iterations).

Usage:
    pixi run -e mast3r-slam-dev python tools/bench_matching_kernels.py

Requires both mast3r_slam._backends (.so from CUDA build) and
mast3r_slam_mojo_backends (.so from Mojo build) to be importable.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

# ── Backend imports ───────────────────────────────────────────────────────────

try:
    from mast3r_slam import _backends as cuda_be
except ImportError:
    print("ERROR: mast3r_slam._backends not found. Run: pixi run -e mast3r-slam-dev _build-cuda-kernels")
    sys.exit(1)

try:
    import mast3r_slam_mojo_backends as mojo_be
except ImportError:
    print("ERROR: mast3r_slam_mojo_backends not found. Run: pixi run -e mast3r-slam-dev _build-mojo-kernels")
    sys.exit(1)

assert torch.cuda.is_available(), "CUDA not available"
DEVICE: torch.device = torch.device("cuda")


# ── Benchmark harness ─────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class BenchResult:
    """Timing statistics from a kernel benchmark."""

    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


def bench(fn: Callable[..., object], warmup: int = 50, runs: int = 500) -> BenchResult:
    """Benchmark a GPU kernel using torch.cuda.Event timing.

    Args:
        fn: Callable that runs the kernel.
        warmup: Number of warmup iterations.
        runs: Number of timed iterations.

    Returns:
        BenchResult with timing statistics in milliseconds.
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

    t: Float[Tensor, "n_runs"] = torch.tensor(timings)
    return BenchResult(
        median_ms=float(t.median()),
        mean_ms=float(t.mean()),
        std_ms=float(t.std()),
        min_ms=float(t.min()),
        max_ms=float(t.max()),
    )


# ── Data generation ───────────────────────────────────────────────────────────


def make_iter_proj_data(
    batch: int, h: int, w: int, seed: int = 42,
) -> tuple[Float[Tensor, "b h w 9"], Float[Tensor, "b hw 3"], Float[Tensor, "b hw 2"]]:
    """Generate synthetic inputs for iter_proj."""
    gen: torch.Generator = torch.Generator(device=DEVICE).manual_seed(seed)
    hw: int = h * w
    rays: Float[Tensor, "b h w 3"] = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen)
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gx: Float[Tensor, "b h w 3"] = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen) * 0.01
    gy: Float[Tensor, "b h w 3"] = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen) * 0.01
    rays_img: Float[Tensor, "b h w 9"] = torch.cat([rays, gx, gy], dim=-1).contiguous()
    pts: Float[Tensor, "b hw 3"] = torch.randn(batch, hw, 3, device=DEVICE, generator=gen)
    pts_norm: Float[Tensor, "b hw 3"] = (pts / pts.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()
    p_init: Float[Tensor, "b hw 2"] = (torch.rand(batch, hw, 2, device=DEVICE, generator=gen) * (w - 4) + 2.0).contiguous()
    return rays_img, pts_norm, p_init


def make_refine_data(
    batch: int, h: int, w: int, fdim: int, seed: int = 42,
) -> tuple[Float[Tensor, "b h w d"], Float[Tensor, "b hw d"], Int[Tensor, "b hw 2"]]:
    """Generate synthetic inputs for refine_matches."""
    gen: torch.Generator = torch.Generator(device=DEVICE).manual_seed(seed)
    hw: int = h * w
    D11: Float[Tensor, "b h w d"] = torch.randn(batch, h, w, fdim, device=DEVICE, generator=gen)
    D11 = (D11 / D11.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()
    D21: Float[Tensor, "b hw d"] = torch.randn(batch, hw, fdim, device=DEVICE, generator=gen)
    D21 = (D21 / D21.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()
    p1: Int[Tensor, "b hw 2"] = torch.randint(4, min(h, w) - 4, (batch, hw, 2), device=DEVICE)
    return D11, D21, p1


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run all benchmarks and print results."""
    print("=" * 72)
    print("CUDA vs Mojo Matching Kernel Benchmarks")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Warmup: 50 | Runs: 500 | Metric: median (ms)")
    print("=" * 72)

    configs: list[dict] = [
        # iter_proj configs
        {"kernel": "iter_proj", "h": 64, "w": 64, "label": "64x64 (4K pts)"},
        {"kernel": "iter_proj", "h": 224, "w": 224, "label": "224x224 (50K pts)"},
        {"kernel": "iter_proj", "h": 512, "w": 512, "label": "512x512 (262K pts)"},
        # refine_matches configs
        {"kernel": "refine", "h": 32, "w": 32, "fdim": 16, "label": "32x32, d=16"},
        {"kernel": "refine", "h": 64, "w": 64, "fdim": 128, "label": "64x64, d=128"},
        {"kernel": "refine", "h": 128, "w": 128, "fdim": 16, "label": "128x128, d=16"},
    ]

    print(f"\n{'Kernel':<20} {'Size':<20} {'CUDA (ms)':<14} {'Mojo (ms)':<14} {'Ratio':<8}")
    print("-" * 72)

    for cfg in configs:
        if cfg["kernel"] == "iter_proj":
            h: int = cfg["h"]
            w: int = cfg["w"]
            rays_img, pts_norm, p_init = make_iter_proj_data(1, h, w)
            cuda_r: BenchResult = bench(lambda: cuda_be.iter_proj(rays_img, pts_norm, p_init, 5, 1.0, 1e-4))
            mojo_r: BenchResult = bench(lambda: mojo_be.iter_proj(rays_img, pts_norm, p_init, 5, 1.0, 1e-4))
            name: str = "iter_proj"
        else:
            h = cfg["h"]
            w = cfg["w"]
            fdim: int = cfg["fdim"]
            D11, D21, p1 = make_refine_data(1, h, w, fdim)
            cuda_r = bench(lambda: cuda_be.refine_matches(D11.half(), D21.half(), p1, 2, 2))
            mojo_r = bench(lambda: mojo_be.refine_matches(D11.half(), D21.half(), p1, 2, 2))
            name = "refine_matches"

        ratio: float = mojo_r.median_ms / max(cuda_r.median_ms, 1e-9)
        winner: str = "Mojo" if ratio < 1.0 else "CUDA" if ratio > 1.0 else "tie"
        print(
            f"{name:<20} {cfg['label']:<20} "
            f"{cuda_r.median_ms:>8.3f}      {mojo_r.median_ms:>8.3f}      "
            f"{ratio:>5.2f}x ({winner})"
        )

    print("-" * 72)
    print("Ratio < 1.0 = Mojo faster | Ratio > 1.0 = CUDA faster")


if __name__ == "__main__":
    main()
