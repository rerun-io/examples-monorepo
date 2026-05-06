#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import dpvo._cuda_ba as cuda_ba
import dpvo_fastba_mojo.backend as mojo_ba


def _bench(fn: Callable[[], object], warmup: int, runs: int) -> float:
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


def _case(name: str, cuda_fn: Callable[[], object], mojo_fn: Callable[[], object], max_ratio: float, warmup: int, runs: int) -> dict[str, Any]:
    cuda_ms = _bench(cuda_fn, warmup, runs)
    mojo_ms = _bench(mojo_fn, warmup, runs)
    ratio = mojo_ms / max(cuda_ms, 1e-9)
    return {
        "name": name,
        "cuda_ms": cuda_ms,
        "mojo_ms": mojo_ms,
        "ratio": ratio,
        "passed": ratio <= max_ratio,
    }


def _inputs(edges: int, patches_n: int, frames: int, seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    poses = torch.zeros(1, frames, 7, device="cuda", dtype=torch.float32)
    poses[..., 6] = 1.0
    poses[..., :3] = torch.randn(1, frames, 3, device="cuda", generator=gen) * 0.01

    patches = torch.empty(1, patches_n, 3, 3, 3, device="cuda", dtype=torch.float32)
    xy = torch.tensor(
        [
            [[316.0, 320.0, 324.0], [316.0, 320.0, 324.0], [316.0, 320.0, 324.0]],
            [[236.0, 236.0, 236.0], [240.0, 240.0, 240.0], [244.0, 244.0, 244.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    patches[:, :, :2] = xy
    patches[:, :, 2] = 1.0 + torch.rand(1, patches_n, 3, 3, device="cuda", generator=gen) * 0.05

    intrinsics = torch.tensor([[[320.0, 320.0, 320.0, 240.0]]], device="cuda", dtype=torch.float32)
    ii = torch.randint(0, frames, (edges,), device="cuda", dtype=torch.int64, generator=gen)
    jj = torch.randint(0, frames, (edges,), device="cuda", dtype=torch.int64, generator=gen)
    same = ii == jj
    jj[same] = (jj[same] + 1) % frames
    kk = torch.randint(0, patches_n, (edges,), device="cuda", dtype=torch.int64, generator=gen)
    target = cuda_ba.reproject(poses, patches, intrinsics, ii, jj, kk)[:, :, :, 1, 1].contiguous()
    target = target + torch.randn(1, edges, 2, device="cuda", generator=gen) * 0.01
    weight = torch.ones_like(target)
    lmbda = torch.as_tensor([1e-4], device="cuda")
    return poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--max-ratio", type=float, default=1.05)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for fastba benchmarks")
    if mojo_ba._native_ops() is None:
        raise SystemExit("Native Mojo backend is required for the speed gate. Run `pixi run -e dpvo-dev _build-dpvo-fastba-mojo-native` first.")

    cases: list[dict[str, Any]] = []

    poses, patches, intrinsics, _target, _weight, _lmbda, ii, jj, kk = _inputs(edges=8192, patches_n=4096, frames=64, seed=220)
    cases.append(_case(
        "reproject_p3_edges8192",
        lambda: cuda_ba.reproject(poses, patches, intrinsics, ii, jj, kk),
        lambda: mojo_ba.reproject(poses, patches, intrinsics, ii, jj, kk),
        args.max_ratio,
        args.warmup,
        args.runs,
    ))

    poses_ba, patches_ba, intrinsics_ba, target, weight, lmbda, ii_ba, jj_ba, kk_ba = _inputs(edges=8192, patches_n=4096, frames=8, seed=221)

    def cuda_dense_ba() -> None:
        poses_work = poses_ba.clone()
        patches_work = patches_ba.clone()
        cuda_ba.forward(poses_work.data, patches_work, intrinsics_ba, target, weight, lmbda, ii_ba, jj_ba, kk_ba, -1, 1, 8, 2, False)

    def mojo_dense_ba() -> None:
        poses_work = poses_ba.clone()
        patches_work = patches_ba.clone()
        mojo_ba.forward(poses_work.data, patches_work, intrinsics_ba, target, weight, lmbda, ii_ba, jj_ba, kk_ba, -1, 1, 8, 2, False)

    cases.append(_case(
        "ba_dense_iter2_edges8192_p3",
        cuda_dense_ba,
        mojo_dense_ba,
        args.max_ratio,
        args.warmup,
        args.runs,
    ))

    result = {
        "gpu_name": torch.cuda.get_device_name(0),
        "max_ratio": args.max_ratio,
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    for case in cases:
        status = "PASS" if case["passed"] else "FAIL"
        print(f"{status} {case['name']}: cuda={case['cuda_ms']:.4f}ms mojo={case['mojo_ms']:.4f}ms ratio={case['ratio']:.3f}")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
