"""PyTorch-facing wrapper for the DPVO fastba Mojo backend."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    from max.experimental.torch import CustomOpLibrary
except ImportError:  # pragma: no cover - depends on installed MAX version
    from max.torch import CustomOpLibrary  # type: ignore[no-redef]


_OPS_DIR = Path(__file__).resolve().parent / "operations"
_NATIVE_BACKEND = Path(__file__).resolve().parent / "dpvo_fastba_mojo_backends.so"


@lru_cache(maxsize=1)
def _ops() -> Any:
    return CustomOpLibrary(_OPS_DIR)


@lru_cache(maxsize=1)
def _native_ops() -> Any | None:
    if not _NATIVE_BACKEND.exists():
        return None
    module_name = "dpvo_fastba_mojo_backends"
    spec = importlib.util.spec_from_file_location(module_name, _NATIVE_BACKEND)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _check_cuda_float(name: str, tensor: Tensor) -> Tensor:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {tensor.dtype}")
    return tensor.contiguous()


def _check_cuda_index(name: str, tensor: Tensor) -> Tensor:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.int64:
        raise TypeError(f"{name} must be torch.int64, got {tensor.dtype}")
    return tensor.contiguous()


def smoke_scale(x: Tensor) -> Tensor:
    """Small CustomOpLibrary smoke op used by tests."""
    x = _check_cuda_float("x", x)
    out = torch.empty_like(x)
    _ops().fastba_smoke_scale(out, x)
    torch.cuda.synchronize()
    return out


def _require_native_ops() -> Any:
    native_ops = _native_ops()
    if native_ops is None:
        raise RuntimeError("Native Mojo fastba backend is required. Run `pixi run -e dpvo-dev _build-dpvo-fastba-mojo-native`.")
    return native_ops


def neighbors(ii: Tensor, jj: Tensor) -> list[Tensor]:
    """Match ``_cuda_ba.neighbors`` using a small CPU grouping routine."""
    if ii.dtype != torch.int64 or jj.dtype != torch.int64:
        raise TypeError("ii and jj must be torch.int64 tensors")
    ii_cpu = ii.detach().cpu()
    jj_cpu = jj.detach().cpu()
    groups: dict[int, list[int]] = {}
    for edge, key in enumerate(ii_cpu.tolist()):
        groups.setdefault(int(key), []).append(edge)

    ix = torch.empty((ii_cpu.numel(),), dtype=torch.int64)
    jx = torch.empty((ii_cpu.numel(),), dtype=torch.int64)
    for edges in groups.values():
        edges.sort(key=lambda edge: int(jj_cpu[edge]))
        for pos, edge in enumerate(edges):
            ix[edge] = edges[pos - 1] if pos > 0 else -1
            jx[edge] = edges[pos + 1] if pos < len(edges) - 1 else -1
    return [ix.to(device=ii.device), jx.to(device=ii.device)]


def reproject(poses: Tensor, patches: Tensor, intrinsics: Tensor, ii: Tensor, jj: Tensor, kk: Tensor) -> Tensor:
    """Reproject patches into target frames."""
    for name, tensor in (("poses", poses), ("patches", patches), ("intrinsics", intrinsics)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be torch.float32, got {tensor.dtype}")
    for name, tensor in (("ii", ii), ("jj", jj), ("kk", kk)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.int64:
            raise TypeError(f"{name} must be torch.int64, got {tensor.dtype}")
    return _require_native_ops().reproject(poses, patches, intrinsics, ii, jj, kk)


def solve_system(
    J_Ginv_i: Tensor,
    J_Ginv_j: Tensor,
    ii: Tensor,
    jj: Tensor,
    res: Tensor,
    ep: float,
    lm: float,
    freen: int,
) -> list[Tensor]:
    """Sparse loop-closure solve is not implemented in the Mojo backend yet."""
    raise NotImplementedError(
        "dpvo_fastba_mojo.solve_system is not implemented yet. "
        "Use DPVO_FASTBA_BACKEND=cuda explicitly if this CUDA-only path is required."
    )


def forward(
    poses: Tensor,
    patches: Tensor,
    intrinsics: Tensor,
    target: Tensor,
    weight: Tensor,
    lmbda: Tensor,
    ii: Tensor,
    jj: Tensor,
    kk: Tensor,
    M: int,
    t0: int,
    t1: int,
    iterations: int,
    eff_impl: bool,
) -> list[Tensor]:
    """Bundle-adjust poses and patches in place."""
    if eff_impl:
        raise NotImplementedError(
            "dpvo_fastba_mojo.forward does not implement eff_impl=True yet. "
            "Use DPVO_FASTBA_BACKEND=cuda explicitly if the global BA E-block path is required."
        )
    native_ops = _require_native_ops()

    poses = _check_cuda_float("poses", poses)
    patches = _check_cuda_float("patches", patches)
    intrinsics = _check_cuda_float("intrinsics", intrinsics)
    target = _check_cuda_float("target", target)
    weight = _check_cuda_float("weight", weight)
    lmbda = _check_cuda_float("lmbda", lmbda)
    ii = _check_cuda_index("ii", ii)
    jj = _check_cuda_index("jj", jj)
    kk = _check_cuda_index("kk", kk)

    kx, ku = torch.unique(kk, sorted=True, return_inverse=True)
    active_poses = int(t1) - int(t0)
    unique_patches = int(kx.shape[0])
    for _ in range(int(iterations)):
        B, E, C, v, u = native_ops.ba_dense_accumulate((poses, patches, intrinsics, target, weight, ii, jj, kk, ku, int(t0), int(t1)))
        v = v.view(6 * active_poses, 1)
        u = u.view(unique_patches, 1)
        q = 1.0 / (C + lmbda).view(1, unique_patches)
        qt = q.transpose(0, 1)
        if active_poses == 0:
            dz = (qt * u).view(unique_patches)
            native_ops.patch_retract(patches, kx, dz)
        else:
            eq = E * q
            et = E.transpose(0, 1)
            s = B - torch.matmul(eq, et)
            y = v - torch.matmul(eq, u)
            ident = torch.eye(6 * active_poses, device=poses.device, dtype=poses.dtype)
            s = s + ident * (1e-4 * s + 1.0)
            chol = torch.linalg.cholesky_ex(s).L
            dx = torch.cholesky_solve(y, chol)
            dz = qt * (u - torch.matmul(et, dx))
            native_ops.pose_retract(poses, dx.view(active_poses, 6), int(t0), int(t1))
            native_ops.patch_retract(patches, kx, dz.view(unique_patches))
    return []


def BA(
    poses: Tensor,
    patches: Tensor,
    intrinsics: Tensor,
    target: Tensor,
    weight: Tensor,
    lmbda: Tensor,
    ii: Tensor,
    jj: Tensor,
    kk: Tensor,
    t0: int,
    t1: int,
    M: int = -1,
    iterations: int = 2,
    eff_impl: bool = False,
) -> None:
    """Run bundle adjustment through the backend-compatible ``forward`` API."""
    forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl)
