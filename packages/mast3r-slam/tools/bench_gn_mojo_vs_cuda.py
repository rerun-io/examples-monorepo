from __future__ import annotations

from dataclasses import dataclass

import lietorch
import torch

from mast3r_slam.gn_backends import load_gn_backend
from mast3r_slam.max_gn_rays import available as max_available
from mast3r_slam.max_gn_rays import gauss_newton_rays as max_gauss_newton_rays

DEVICE = "cuda:0"
SYNTHETIC_ATOL = 1e-4
POSE_RTOL = 1e-4


@dataclass(frozen=True)
class BenchRow:
    name: str
    cuda_ms: float
    mojo_ms: float
    max_ms: float
    mojo_ratio: float
    max_ratio: float
    mojo_max_abs: float
    max_max_abs: float
    mojo_same: bool
    max_same: bool


@dataclass(frozen=True)
class RaysPreset:
    name: str
    n_poses: int
    h: int
    w: int
    undirected_edges: int
    valid_fraction: float
    q_base: float
    seed: int


def _sync() -> None:
    torch.cuda.synchronize()


def _benchmark(fn, warmup: int = 5, runs: int = 20) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    samples: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        samples.append(start.elapsed_time(end))
    return float(torch.tensor(samples).median().item())


def _make_representative_rays_inputs(preset: RaysPreset) -> tuple[torch.Tensor, ...]:
    """Build a synthetic GN rays workload shaped like the real SLAM path.

    Presets are driven by the actual image-size regimes we use in production:
    `224x224` for fast runs and `512x512` for base runs. Edge count and valid
    fraction are chosen to mimic a small sliding graph with loop-closure edges,
    instead of a trivial chain or a dense all-to-all toy graph.
    """
    n_poses = preset.n_poses
    h = preset.h
    w = preset.w
    seed = preset.seed
    torch.manual_seed(seed)
    hw = h * w
    twc = lietorch.Sim3.exp(0.02 * torch.randn(n_poses, 7, device=DEVICE)).data.contiguous().float()
    xs = torch.randn(n_poses, hw, 3, device=DEVICE, dtype=torch.float32)
    xs[..., 2].abs_().add_(1.0)
    cs = (0.5 + torch.rand(n_poses, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()

    undirected: list[tuple[int, int]] = [(i, i + 1) for i in range(n_poses - 1)]
    loop_span = 2
    cursor = 0
    while len(undirected) < preset.undirected_edges:
        i = cursor % max(n_poses - loop_span, 1)
        j = i + loop_span
        edge = (i, j)
        if j < n_poses and edge not in undirected:
            undirected.append(edge)
        else:
            loop_span = 3 if loop_span == 2 else 2
            cursor += 1
            continue
        cursor += 1

    ii_fwd = torch.tensor([e[0] for e in undirected], device=DEVICE, dtype=torch.long)
    jj_fwd = torch.tensor([e[1] for e in undirected], device=DEVICE, dtype=torch.long)
    ii = torch.cat([ii_fwd, jj_fwd], dim=0).contiguous()
    jj = torch.cat([jj_fwd, ii_fwd], dim=0).contiguous()
    n_edges = int(ii.shape[0])
    idx = torch.arange(hw, device=DEVICE, dtype=torch.long).unsqueeze(0).repeat(n_edges, 1).contiguous()
    valid = (torch.rand(n_edges, hw, 1, device=DEVICE) < preset.valid_fraction).contiguous()
    q = (preset.q_base + 0.35 * torch.rand(n_edges, hw, 1, device=DEVICE, dtype=torch.float32)).contiguous()
    return (
        twc,
        xs,
        cs,
        ii,
        jj,
        idx,
        valid,
        q,
        0.5,
        0.25,
        0.1,
        0.1,
        5,
        1e-6,
    )


def _run_synthetic_backend(backend_name: str, args: tuple[torch.Tensor | float | int, ...]) -> torch.Tensor:
    backend = load_gn_backend(backend_name)
    twc = args[0].clone()
    backend.gauss_newton_rays(
        twc,
        *args[1:],
    )
    _sync()
    return twc


def _run_max_backend(args: tuple[torch.Tensor | float | int, ...]) -> torch.Tensor:
    twc = args[0].clone()
    max_gauss_newton_rays(
        twc,
        *args[1:],
    )
    _sync()
    return twc


def bench_preset(preset: RaysPreset) -> BenchRow:
    args = _make_representative_rays_inputs(preset)
    if not max_available():
        raise RuntimeError("MAX GN rays custom-op package is not available in this environment")
    cuda_pose = _run_synthetic_backend("cuda", args)
    mojo_pose = _run_synthetic_backend("mojo", args)
    max_pose = _run_max_backend(args)
    cuda_ms = _benchmark(lambda: _run_synthetic_backend("cuda", args))
    mojo_ms = _benchmark(lambda: _run_synthetic_backend("mojo", args))
    max_ms = _benchmark(lambda: _run_max_backend(args))
    mojo_max_abs = float((cuda_pose - mojo_pose).abs().max().item())
    max_max_abs = float((cuda_pose - max_pose).abs().max().item())
    mojo_same = bool(torch.allclose(cuda_pose, mojo_pose, atol=SYNTHETIC_ATOL, rtol=POSE_RTOL))
    max_same = bool(torch.allclose(cuda_pose, max_pose, atol=SYNTHETIC_ATOL, rtol=POSE_RTOL))
    return BenchRow(
        preset.name,
        cuda_ms,
        mojo_ms,
        max_ms,
        mojo_ms / cuda_ms,
        max_ms / cuda_ms,
        mojo_max_abs,
        max_max_abs,
        mojo_same,
        max_same,
    )


def _print_table(rows: list[BenchRow]) -> None:
    print("| benchmark | cuda ms | mojo ms | max ms | mojo/cuda | max/cuda | mojo max abs diff | max max abs diff | mojo same | max same |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        print(
            f"| {row.name} | {row.cuda_ms:.3f} | {row.mojo_ms:.3f} | {row.max_ms:.3f} | {row.mojo_ratio:.3f} | {row.max_ratio:.3f} | {row.mojo_max_abs:.2e} | {row.max_max_abs:.2e} | {'yes' if row.mojo_same else 'no'} | {'yes' if row.max_same else 'no'} |"
        )


def main() -> None:
    rows = [
        bench_preset(RaysPreset(
            name="fast-like-rays",
            n_poses=8,
            h=224,
            w=224,
            undirected_edges=12,
            valid_fraction=0.18,
            q_base=1.55,
            seed=0,
        )),
        bench_preset(RaysPreset(
            name="base-like-rays",
            n_poses=10,
            h=512,
            w=512,
            undirected_edges=16,
            valid_fraction=0.14,
            q_base=1.55,
            seed=1,
        )),
    ]
    _print_table(rows)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    main()
