#!/usr/bin/env python
# ruff: noqa: I001
from __future__ import annotations

import torch

import mast3r_slam._backends as cuda_be  # pyrefly: ignore[missing-import]
import mast3r_slam.max_ops as max_ops


def bench(fn, warmup: int = 20, runs: int = 100) -> float:
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
    return float(torch.tensor(timings).median())


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    gen = torch.Generator(device="cuda").manual_seed(0)
    poses = torch.randn(64, 8, device="cuda", generator=gen, dtype=torch.float32)
    poses[:, 6] = 1.0
    poses[:, 7] = 1.0
    dx = torch.randn(63, 7, device="cuda", generator=gen, dtype=torch.float32) * 1e-3
    num_fix = 1

    cuda_ms = bench(lambda: cuda_be.pose_retr(poses.clone(), dx, num_fix))
    max_ms = bench(lambda: max_ops.pose_retr(poses.clone(), dx, num_fix))
    ratio = max_ms / max(cuda_ms, 1e-9)
    print(f"{'name':<16} {'cuda ms':>10} {'max ms':>10} {'ratio':>8}")
    print(f"{'pose_retr':<16} {cuda_ms:>10.3f} {max_ms:>10.3f} {ratio:>8.3f}")


if __name__ == "__main__":
    main()
