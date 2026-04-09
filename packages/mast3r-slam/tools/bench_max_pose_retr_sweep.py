#!/usr/bin/env python
# ruff: noqa: I001
from __future__ import annotations

import torch

import mast3r_slam._backends as cuda_be  # pyrefly: ignore[missing-import]
import mast3r_slam.max_ops as max_ops


def bench(fn, warmup: int = 10, runs: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(start.elapsed_time(end))
    return float(torch.tensor(vals).median())


def main() -> None:
    print(f"{'poses':>8} {'cuda ms':>10} {'max ms':>10} {'ratio':>8}")
    for n in [64, 256, 1024, 4096, 16384]:
        gen = torch.Generator(device="cuda").manual_seed(0)
        poses = torch.randn(n, 8, device="cuda", generator=gen, dtype=torch.float32)
        poses[:, 6] = 1.0
        poses[:, 7] = 1.0
        dx = (
            torch.randn(n - 1, 7, device="cuda", generator=gen, dtype=torch.float32)
            * 1e-3
        )
        cuda_ms = bench(lambda poses=poses, dx=dx: cuda_be.pose_retr(poses.clone(), dx, 1))
        max_ms = bench(lambda poses=poses, dx=dx: max_ops.pose_retr(poses.clone(), dx, 1))
        print(f"{n:>8} {cuda_ms:>10.3f} {max_ms:>10.3f} {max_ms / max(cuda_ms, 1e-9):>8.3f}")


if __name__ == "__main__":
    main()
