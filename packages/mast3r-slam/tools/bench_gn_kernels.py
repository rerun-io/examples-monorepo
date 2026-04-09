#!/usr/bin/env python
"""Benchmark GN public APIs and one-step accumulators."""
# ruff: noqa: I001

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

import mast3r_slam._backends as cuda_be  # pyrefly: ignore[missing-import]
import mast3r_slam.gn_backends as gn_be

assert torch.cuda.is_available(), "CUDA not available"
DEVICE: torch.device = torch.device("cuda")


@dataclass(slots=True, frozen=True)
class BenchResult:
    median_ms: float
    mean_ms: float
    std_ms: float


def bench(fn: Callable[[], object], warmup: int = 20, runs: int = 100) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    t = torch.tensor(timings)
    return BenchResult(float(t.median()), float(t.mean()), float(t.std()))


def make_graph_fixture(
    num_poses: int = 4,
    num_points: int = 256,
    num_edges: int = 6,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    Twc = torch.randn(num_poses, 8, device=DEVICE, generator=gen, dtype=torch.float32)
    Twc[:, 6] = 1.0
    Twc[:, 7] = 1.0
    Xs = torch.randn(num_poses, num_points, 3, device=DEVICE, generator=gen, dtype=torch.float32).contiguous()
    Cs = torch.rand(num_poses, num_points, 1, device=DEVICE, generator=gen, dtype=torch.float32).contiguous() + 0.5
    ii = torch.randint(0, num_poses - 1, (num_edges,), device=DEVICE, generator=gen, dtype=torch.long)
    jj = torch.clamp(ii + 1, max=num_poses - 1)
    idx_ii2jj = torch.randint(0, num_points, (num_edges, num_points), device=DEVICE, generator=gen, dtype=torch.long).contiguous()
    valid_match = (torch.rand(num_edges, num_points, 1, device=DEVICE, generator=gen) > 0.1).contiguous()
    Q = (torch.rand(num_edges, num_points, 1, device=DEVICE, generator=gen, dtype=torch.float32) * 2.0 + 1.0).contiguous()
    return Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q


def main() -> None:
    if not all(hasattr(cuda_be, name) for name in ("gauss_newton_rays_step", "gauss_newton_calib_step")):
        print("ERROR: CUDA GN step helpers are not built")
        sys.exit(1)

    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q = make_graph_fixture()
    K = torch.tensor([[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]], device=DEVICE, dtype=torch.float32)

    rows: list[tuple[str, str, BenchResult, BenchResult]] = []
    rows.append((
        "rays_step",
        "synthetic",
        bench(lambda: cuda_be.gauss_newton_rays_step(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5)),
        bench(lambda: gn_be.gauss_newton_rays_step(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5)),
    ))
    rows.append((
        "calib_step",
        "synthetic",
        bench(lambda: cuda_be.gauss_newton_calib_step(Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q, 256, 256, -10, 1e-6, 1.0, 10.0, 0.0, 1.5)),
        bench(lambda: gn_be.gauss_newton_calib_step(Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q, 256, 256, -10, 1e-6, 1.0, 10.0, 0.0, 1.5)),
    ))
    rows.append((
        "rays_public",
        "synthetic",
        bench(lambda: cuda_be.gauss_newton_rays(Twc.clone(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5, 5, 1e-8)),
        bench(lambda: gn_be.gauss_newton_rays(Twc.clone(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, 0.003, 10.0, 0.0, 1.5, 5, 1e-8)),
    ))

    print(f"{'name':<14} {'fixture':<10} {'cuda ms':>10} {'mojo ms':>10} {'ratio':>8}")
    for name, fixture, cuda_r, mojo_r in rows:
        ratio = mojo_r.median_ms / max(cuda_r.median_ms, 1e-9)
        print(f"{name:<14} {fixture:<10} {cuda_r.median_ms:>10.3f} {mojo_r.median_ms:>10.3f} {ratio:>8.3f}")


if __name__ == "__main__":
    main()
