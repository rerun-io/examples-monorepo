#!/usr/bin/env python
"""Benchmark GN public APIs on captured real fixtures."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from mast3r_slam import _backends as cuda_be
from mast3r_slam import gn_backends as selected_be
from mast3r_slam.gn_fixture_utils import iter_gn_fixture_paths, load_gn_fixture


assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device("cuda")


@dataclass(slots=True, frozen=True)
class BenchResult:
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


@dataclass(slots=True, frozen=True)
class BackendRun:
    timing: BenchResult
    output: tuple[Any, ...]
    final_twc: Tensor


def _to_device(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=DEVICE, non_blocking=False)
    return value


def _clone_arg(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.clone()
    return value


def _bench(fn: Callable[[], object], warmup: int, runs: int) -> BenchResult:
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

    values = torch.tensor(timings, dtype=torch.float64)
    return BenchResult(
        median_ms=float(values.median()),
        mean_ms=float(values.mean()),
        std_ms=float(values.std(unbiased=False)),
        min_ms=float(values.min()),
        max_ms=float(values.max()),
    )


def _load_fixture_inputs(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    payload = load_gn_fixture(path)
    kind = str(payload["kind"])
    inputs = {key: _to_device(value) for key, value in dict(payload["inputs"]).items()}
    metadata = dict(payload.get("metadata", {}))
    return kind, inputs, metadata


def _call_public(backend: Any, kind: str, inputs: dict[str, Any]) -> tuple[Any, ...]:
    if kind == "rays":
        return tuple(backend.gauss_newton_rays(
            inputs["Twc"],
            inputs["Xs"],
            inputs["Cs"],
            inputs["ii"],
            inputs["jj"],
            inputs["idx_ii2jj"],
            inputs["valid_match"],
            inputs["Q"],
            float(inputs["sigma_ray"]),
            float(inputs["sigma_dist"]),
            float(inputs["C_thresh"]),
            float(inputs["Q_thresh"]),
            int(inputs["max_iter"]),
            float(inputs["delta_thresh"]),
        ))
    if kind == "calib":
        return tuple(backend.gauss_newton_calib(
            inputs["Twc"],
            inputs["Xs"],
            inputs["Cs"],
            inputs["K"],
            inputs["ii"],
            inputs["jj"],
            inputs["idx_ii2jj"],
            inputs["valid_match"],
            inputs["Q"],
            int(inputs["height"]),
            int(inputs["width"]),
            int(inputs["pixel_border"]),
            float(inputs["z_eps"]),
            float(inputs["sigma_pixel"]),
            float(inputs["sigma_depth"]),
            float(inputs["C_thresh"]),
            float(inputs["Q_thresh"]),
            int(inputs["max_iter"]),
            float(inputs["delta_thresh"]),
        ))
    raise ValueError(f"Unsupported fixture kind: {kind}")


def _call_step(backend: Any, kind: str, inputs: dict[str, Any]) -> tuple[Any, ...]:
    if kind == "rays":
        return tuple(backend.gauss_newton_rays_step(
            inputs["Twc"],
            inputs["Xs"],
            inputs["Cs"],
            inputs["ii"],
            inputs["jj"],
            inputs["idx_ii2jj"],
            inputs["valid_match"],
            inputs["Q"],
            float(inputs["sigma_ray"]),
            float(inputs["sigma_dist"]),
            float(inputs["C_thresh"]),
            float(inputs["Q_thresh"]),
        ))
    if kind == "calib":
        return tuple(backend.gauss_newton_calib_step(
            inputs["Twc"],
            inputs["Xs"],
            inputs["Cs"],
            inputs["K"],
            inputs["ii"],
            inputs["jj"],
            inputs["idx_ii2jj"],
            inputs["valid_match"],
            inputs["Q"],
            int(inputs["height"]),
            int(inputs["width"]),
            int(inputs["pixel_border"]),
            float(inputs["z_eps"]),
            float(inputs["sigma_pixel"]),
            float(inputs["sigma_depth"]),
            float(inputs["C_thresh"]),
            float(inputs["Q_thresh"]),
        ))
    raise ValueError(f"Unsupported fixture kind: {kind}")


def _run_backend(backend: Any, kind: str, base_inputs: dict[str, Any], warmup: int, runs: int) -> BackendRun:
    def invoke() -> tuple[Any, ...]:
        call_inputs = {key: _clone_arg(value) for key, value in base_inputs.items()}
        return _call_public(backend, kind, call_inputs)

    timing = _bench(invoke, warmup=warmup, runs=runs)
    final_inputs = {key: _clone_arg(value) for key, value in base_inputs.items()}
    output = _call_public(backend, kind, final_inputs)
    final_twc = final_inputs["Twc"].detach().clone()
    return BackendRun(timing=timing, output=output, final_twc=final_twc)


def _run_step_backend(backend: Any, kind: str, base_inputs: dict[str, Any], warmup: int, runs: int) -> tuple[BenchResult, tuple[Any, ...]]:
    def invoke() -> tuple[Any, ...]:
        call_inputs = {key: _clone_arg(value) for key, value in base_inputs.items()}
        return _call_step(backend, kind, call_inputs)

    timing = _bench(invoke, warmup=warmup, runs=runs)
    final_inputs = {key: _clone_arg(value) for key, value in base_inputs.items()}
    output = _call_step(backend, kind, final_inputs)
    return timing, output


def _max_abs_diff(lhs: Tensor, rhs: Tensor) -> float:
    return float((lhs - rhs).abs().max().item())


def _pose_error_report(cuda_twc: Tensor, selected_twc: Tensor) -> dict[str, float]:
    return {
        "translation_max_abs": _max_abs_diff(selected_twc[:, :3], cuda_twc[:, :3]),
        "quaternion_max_abs": _max_abs_diff(selected_twc[:, 3:7], cuda_twc[:, 3:7]),
        "scale_max_abs": _max_abs_diff(selected_twc[:, 7:], cuda_twc[:, 7:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_path", type=Path, help="Fixture .pt file or directory of fixture files")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    print(
        f"{'fixture':<24} {'scope':<8} {'kind':<8} {'cuda ms':>10} {'selected ms':>12} {'ratio':>8} "
        f"{'dtrans':>10} {'dquat':>10} {'dscale':>10} {'ddx':>10}"
    )
    for path in iter_gn_fixture_paths(args.fixture_path):
        kind, inputs, metadata = _load_fixture_inputs(path)
        step_cuda, step_cuda_out = _run_step_backend(cuda_be, kind, inputs, warmup=args.warmup, runs=args.runs)
        step_selected, step_selected_out = _run_step_backend(selected_be, kind, inputs, warmup=args.warmup, runs=args.runs)
        cuda_run = _run_backend(cuda_be, kind, inputs, warmup=args.warmup, runs=args.runs)
        selected_run = _run_backend(selected_be, kind, inputs, warmup=args.warmup, runs=args.runs)

        public_ratio = selected_run.timing.median_ms / max(cuda_run.timing.median_ms, 1e-9)
        step_ratio = step_selected.median_ms / max(step_cuda.median_ms, 1e-9)
        pose_err = _pose_error_report(cuda_run.final_twc, selected_run.final_twc)
        dx_err = float("nan")
        if cuda_run.output and selected_run.output:
            cuda_dx = cuda_run.output[0]
            selected_dx = selected_run.output[0]
            if isinstance(cuda_dx, Tensor) and isinstance(selected_dx, Tensor):
                dx_err = _max_abs_diff(selected_dx, cuda_dx)

        step_dx_err = float("nan")
        if step_cuda_out and step_selected_out:
            step_hs_cuda = step_cuda_out[0]
            step_hs_selected = step_selected_out[0]
            if isinstance(step_hs_cuda, Tensor) and isinstance(step_hs_selected, Tensor):
                step_dx_err = _max_abs_diff(step_hs_selected, step_hs_cuda)

        fixture_name = metadata.get("name", path.stem)
        print(
            f"{fixture_name:<24} {'step':<8} {kind:<8} {step_cuda.median_ms:>10.3f} "
            f"{step_selected.median_ms:>12.3f} {step_ratio:>8.3f} "
            f"{float('nan'):>10} {float('nan'):>10} {float('nan'):>10} {step_dx_err:>10.2e}"
        )
        print(
            f"{fixture_name:<24} {'public':<8} {kind:<8} {cuda_run.timing.median_ms:>10.3f} "
            f"{selected_run.timing.median_ms:>12.3f} {public_ratio:>8.3f} "
            f"{pose_err['translation_max_abs']:>10.2e} {pose_err['quaternion_max_abs']:>10.2e} "
            f"{pose_err['scale_max_abs']:>10.2e} {dx_err:>10.2e}"
        )


if __name__ == "__main__":
    main()
