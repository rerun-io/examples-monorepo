#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import dpvo._cuda_corr as cuda_corr
import dpvo_altcorr_mojo.backend as mojo_corr


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


def _patchify_inputs(radius: int, channels: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    net = torch.randn(1, channels, 64, 64, device="cuda", generator=gen)
    coords = torch.rand(1, 512, 2, device="cuda", generator=gen) * 58.0 + 2.0
    diameter = 2 * radius + 2
    grad = torch.randn(1, 512, channels, diameter, diameter, device="cuda", generator=gen)
    return net, coords, grad


def _corr_inputs(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    fmap1 = torch.randn(1, 256, 128, 3, 3, device="cuda", generator=gen)
    fmap2 = torch.randn(1, 8, 128, 64, 64, device="cuda", generator=gen)
    coords = torch.rand(1, 384, 2, 3, 3, device="cuda", generator=gen) * 58.0 + 2.0
    ii = torch.randint(0, 256, (384,), device="cuda", dtype=torch.int64, generator=gen)
    jj = torch.randint(0, 8, (384,), device="cuda", dtype=torch.int64, generator=gen)
    grad = torch.randn(1, 384, 7, 7, 3, 3, device="cuda", generator=gen)
    return fmap1, fmap2, coords, ii, jj, grad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--max-ratio", type=float, default=1.05)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for altcorr benchmarks")
    if mojo_corr._native_ops() is None:
        raise SystemExit("Native Mojo backend is required for the speed gate. Run `pixi run -e dpvo-dev _build-dpvo-altcorr-mojo-native` first.")

    cases: list[dict[str, Any]] = []

    net0, coords0, _ = _patchify_inputs(radius=0, channels=128, seed=50)
    cases.append(_case(
        "patchify_forward_r0_c128",
        lambda: cuda_corr.patchify_forward(net0, coords0, 0),
        lambda: mojo_corr.patchify_forward(net0, coords0, 0),
        args.max_ratio,
        args.warmup,
        args.runs,
    ))

    net1, coords1, grad1 = _patchify_inputs(radius=1, channels=128, seed=51)
    cases.append(_case(
        "patchify_forward_r1_c128",
        lambda: cuda_corr.patchify_forward(net1, coords1, 1),
        lambda: mojo_corr.patchify_forward(net1, coords1, 1),
        args.max_ratio,
        args.warmup,
        args.runs,
    ))
    cases.append(_case(
        "patchify_backward_r1_c128",
        lambda: cuda_corr.patchify_backward(net1, coords1, grad1, 1),
        lambda: mojo_corr.patchify_backward(net1, coords1, grad1, 1),
        args.max_ratio,
        args.warmup,
        args.runs,
    ))

    fmap1, fmap2, coords, ii, jj, grad = _corr_inputs(seed=52)
    cases.append(_case(
        "corr_forward_hot_r3_c128_p3",
        lambda: cuda_corr.forward(fmap1, fmap2, coords, ii, jj, 3),
        lambda: mojo_corr.forward(fmap1, fmap2, coords, ii, jj, 3),
        args.max_ratio,
        args.warmup,
        args.runs,
    ))
    cases.append(_case(
        "corr_backward_hot_r3_c128_p3",
        lambda: cuda_corr.backward(fmap1, fmap2, coords, ii, jj, grad, 3),
        lambda: mojo_corr.backward(fmap1, fmap2, coords, ii, jj, grad, 3),
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
