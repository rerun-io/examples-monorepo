#!/usr/bin/env python
"""Benchmark CUDA vs Mojo kernels side-by-side.

Measures both backends under identical conditions using torch.cuda.Event
timing (median of 500 runs after 50 warmup iterations).

Coverage
--------
Covers all kernel families that have a Mojo implementation:
  - iter_proj (frontend: iterative projection)
  - refine_matches (frontend: descriptor refinement)
  - gauss_newton_rays (backend: GN linearisation + solve + retraction)

NOT benchmarked (no Mojo implementation — always routed to CUDA):
  - gauss_newton_points: linearisation step delegates to CUDA extension
  - gauss_newton_calib: linearisation step delegates to CUDA extension
  The Mojo wrappers for these call into the CUDA extension for the step kernel
  (see gn.mojo `gauss_newton_impl` and global_opt.py `_call_gn_backend`).

Input shapes are taken from real pipeline runs (instrumented with BENCH_LOG_SHAPES):
  - fast config: 224px, fdim=24, radius=3, dilation=5, batch=1..8
  - base config: 512px, fdim=24, same matching params
  - GN rays: 10-30 keyframes, 50K-262K points, 30-80 edges

Usage:
    PYTHONPATH=. pixi run -e mast3r-slam-dev python tools/bench_kernels.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import lietorch
import torch
from torch import Tensor

# ── Backend imports ───────────────────────────────────────────────────────────

def _import_required_module(module_name: str, install_hint: str) -> Any:
    try:
        return import_module(module_name)
    except ImportError:
        print(f"ERROR: {module_name} not found. Run: {install_hint}")
        sys.exit(1)


cuda_be: Any = _import_required_module("mast3r_slam._backends", "pixi run -e mast3r-slam _build-cuda-kernels")
mojo_be: Any = _import_required_module("mast3r_slam_mojo_backends", "pixi run -e mast3r-slam _build-mojo-kernels")

assert torch.cuda.is_available(), "CUDA not available"
DEVICE: torch.device = torch.device("cuda")


# ── Benchmark harness ─────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class BenchResult:
    """Timing statistics from a kernel benchmark."""

    median_ms: float


def bench(fn: Callable[..., object], warmup: int = 50, runs: int = 500) -> BenchResult:
    """Benchmark a GPU kernel using torch.cuda.Event timing."""
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

    t: Tensor = torch.tensor(timings)
    return BenchResult(
        median_ms=float(t.median()),
    )


# ── Data generation ───────────────────────────────────────────────────────────


def make_iter_proj_data(
    batch: int, h: int, w: int, seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate synthetic inputs for iter_proj matching real pipeline shapes."""
    gen: torch.Generator = torch.Generator(device=DEVICE).manual_seed(seed)
    hw: int = h * w
    rays: Tensor = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen)
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gx: Tensor = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen) * 0.01
    gy: Tensor = torch.randn(batch, h, w, 3, device=DEVICE, generator=gen) * 0.01
    rays_img: Tensor = torch.cat([rays, gx, gy], dim=-1).contiguous()
    pts: Tensor = torch.randn(batch, hw, 3, device=DEVICE, generator=gen)
    pts_norm: Tensor = (pts / pts.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()
    p_init: Tensor = (torch.rand(batch, hw, 2, device=DEVICE, generator=gen) * (w - 4) + 2.0).contiguous()
    return rays_img, pts_norm, p_init


def make_refine_data(
    batch: int, h: int, w: int, fdim: int, seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate synthetic inputs for refine_matches matching real pipeline shapes."""
    gen: torch.Generator = torch.Generator(device=DEVICE).manual_seed(seed)
    hw: int = h * w
    D11: Tensor = torch.randn(batch, h, w, fdim, device=DEVICE, generator=gen)
    D11 = (D11 / D11.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()
    D21: Tensor = torch.randn(batch, hw, fdim, device=DEVICE, generator=gen)
    D21 = (D21 / D21.norm(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()
    p1: Tensor = torch.randint(4, min(h, w) - 4, (batch, hw, 2), device=DEVICE)
    return D11, D21, p1


def make_gn_rays_data(
    n_poses: int, hw: int, n_edges: int, seed: int = 42,
) -> tuple[Tensor, ...]:
    """Generate synthetic inputs for gauss_newton_rays matching real pipeline shapes.

    Builds a plausible edge graph with sequential + loop-closure edges.
    """
    torch.manual_seed(seed)
    twc: Tensor = lietorch.Sim3.exp(0.02 * torch.randn(n_poses, 7, device=DEVICE)).data.contiguous().float()
    xs: Tensor = torch.randn(n_poses, hw, 3, device=DEVICE, dtype=torch.float32)
    xs[..., 2].abs_().add_(1.0)
    cs: Tensor = (0.5 + torch.rand(n_poses, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()

    # Build edge graph: sequential edges + loop closures
    undirected: list[tuple[int, int]] = [(i, i + 1) for i in range(n_poses - 1)]
    cursor: int = 0
    loop_span: int = 2
    while len(undirected) < n_edges // 2:
        i: int = cursor % max(n_poses - loop_span, 1)
        j: int = i + loop_span
        if j < n_poses and (i, j) not in undirected:
            undirected.append((i, j))
        else:
            loop_span = 3 if loop_span == 2 else 2
        cursor += 1
        if cursor > n_edges * 10:
            break

    ii_fwd: Tensor = torch.tensor([e[0] for e in undirected], device=DEVICE, dtype=torch.long)
    jj_fwd: Tensor = torch.tensor([e[1] for e in undirected], device=DEVICE, dtype=torch.long)
    ii: Tensor = torch.cat([ii_fwd, jj_fwd], dim=0).contiguous()
    jj: Tensor = torch.cat([jj_fwd, ii_fwd], dim=0).contiguous()
    actual_edges: int = int(ii.shape[0])

    idx: Tensor = torch.arange(hw, device=DEVICE, dtype=torch.long).unsqueeze(0).repeat(actual_edges, 1).contiguous()
    valid: Tensor = (torch.rand(actual_edges, hw, 1, device=DEVICE) < 0.14).contiguous()
    q: Tensor = (1.55 + 0.35 * torch.rand(actual_edges, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()

    return twc, xs, cs, ii, jj, idx, valid, q


def make_gn_runner(
    backend: Any,
    twc: Tensor,
    xs: Tensor,
    cs: Tensor,
    ii: Tensor,
    jj: Tensor,
    idx: Tensor,
    valid: Tensor,
    q: Tensor,
    sigma_ray: float,
    sigma_dist: float,
    c_thresh: float,
    q_thresh: float,
    max_iter: int,
    delta_thresh: float,
) -> Callable[[], object]:
    """Capture one GN rays benchmark configuration into a zero-argument runner."""

    def run() -> object:
        t_copy: Tensor = twc.clone()
        args = (t_copy, xs, cs, ii, jj, idx, valid, q, sigma_ray, sigma_dist, c_thresh, q_thresh, max_iter, delta_thresh)
        if backend is mojo_be:
            return backend.gauss_newton_rays(args)
        return backend.gauss_newton_rays(*args)

    return run


# ── Main ──────────────────────────────────────────────────────────────────────


def print_row(name: str, label: str, cuda_r: BenchResult, mojo_r: BenchResult) -> None:
    """Print one row of the benchmark table."""
    ratio: float = mojo_r.median_ms / max(cuda_r.median_ms, 1e-9)
    winner: str = "Mojo" if ratio < 1.0 else "CUDA" if ratio > 1.0 else "tie"
    print(
        f"{name:<22} {label:<28} "
        f"{cuda_r.median_ms:>8.3f}      {mojo_r.median_ms:>8.3f}      "
        f"{ratio:>5.2f}x ({winner})"
    )


def main() -> None:
    """Run all benchmarks and print results."""
    print("=" * 80)
    print("CUDA vs Mojo Kernel Benchmarks (all kernel families)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Warmup: 50 | Runs: 500 | Metric: median (ms)")
    print("=" * 80)

    print(f"\n{'Kernel':<22} {'Config':<28} {'CUDA (ms)':<14} {'Mojo (ms)':<14} {'Ratio':<8}")
    print("-" * 80)

    # ── iter_proj: real pipeline uses 224×224 and 512×512 ─────────────────────
    print("─── Frontend: iter_proj ───")
    for h, w, b, label in [
        (224, 224, 1, "fast b=1 (50K pts)"),
        (224, 224, 4, "fast b=4 (50K pts)"),
        (512, 512, 1, "base b=1 (262K pts)"),
    ]:
        rays_img, pts_norm, p_init = make_iter_proj_data(b, h, w)
        cuda_r: BenchResult = bench(lambda _r=rays_img, _p=pts_norm, _i=p_init: cuda_be.iter_proj(_r, _p, _i, 10, 1e-8, 1e-6))
        mojo_r: BenchResult = bench(lambda _r=rays_img, _p=pts_norm, _i=p_init: mojo_be.iter_proj(_r, _p, _i, 10, 1e-8, 1e-6))
        print_row("iter_proj", label, cuda_r, mojo_r)

    # ── refine_matches: fdim=24, half precision, radius=3, dilation=5 ─────────
    print("─── Frontend: refine_matches ───")
    for h, w, b, fdim, label in [
        (224, 224, 1, 24, "fast b=1, d=24"),
        (224, 224, 4, 24, "fast b=4, d=24"),
        (512, 512, 1, 24, "base b=1, d=24"),
    ]:
        D11, D21, p1 = make_refine_data(b, h, w, fdim)
        cuda_r = bench(lambda _d1=D11, _d2=D21, _p=p1: cuda_be.refine_matches(_d1.half(), _d2.half(), _p, 3, 5))
        mojo_r = bench(lambda _d1=D11, _d2=D21, _p=p1: mojo_be.refine_matches(_d1.half(), _d2.half(), _p, 3, 5))
        print_row("refine_matches", label, cuda_r, mojo_r)

    # ── gauss_newton_rays: the GN backend ─────────────────────────────────────
    # Real pipeline shapes: n_poses grows from 2 to ~88, hw=50176 (224px) or 262144 (512px)
    # Benchmark a small, medium, and large case.
    print("─── Backend: gauss_newton_rays ───")
    gn_configs: list[tuple[int, int, int, str]] = [
        (10, 50176, 30, "fast 10kf, 50K pts"),
        (30, 50176, 80, "fast 30kf, 50K pts"),
        (10, 262144, 30, "base 10kf, 262K pts"),
    ]
    for n_poses, hw, target_edges, label in gn_configs:
        twc, xs, cs, ii, jj, idx, valid, q = make_gn_rays_data(n_poses, hw, target_edges)
        sigma_ray: float = 0.5
        sigma_dist: float = 0.25
        c_thresh: float = 0.1
        q_thresh: float = 0.1
        max_iter: int = 10
        delta_thresh: float = 1e-6

        run_cuda = make_gn_runner(
            cuda_be, twc, xs, cs, ii, jj, idx, valid, q, sigma_ray, sigma_dist, c_thresh, q_thresh, max_iter, delta_thresh
        )
        run_mojo = make_gn_runner(
            mojo_be, twc, xs, cs, ii, jj, idx, valid, q, sigma_ray, sigma_dist, c_thresh, q_thresh, max_iter, delta_thresh
        )

        cuda_r = bench(run_cuda, warmup=10, runs=50)
        mojo_r = bench(run_mojo, warmup=10, runs=50)
        print_row("gauss_newton_rays", label, cuda_r, mojo_r)

    print("-" * 80)
    print("Ratio < 1.0 = Mojo faster | Ratio > 1.0 = CUDA faster")


if __name__ == "__main__":
    main()
