"""PyTorch-facing wrapper for the DPVO altcorr Mojo custom ops.

This module mirrors the existing ``dpvo._cuda_corr`` extension API so DPVO can
switch between the CUDA oracle and this standalone Mojo package without
changing call sites.
"""

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
_NATIVE_BACKEND = Path(__file__).resolve().parent / "dpvo_altcorr_mojo_backends.so"
_FEATURE_DTYPES = (torch.float16, torch.float32)


@lru_cache(maxsize=1)
def _ops() -> Any:
    return CustomOpLibrary(_OPS_DIR)


@lru_cache(maxsize=1)
def _native_ops() -> Any | None:
    if not _NATIVE_BACKEND.exists():
        return None
    module_name = "dpvo_altcorr_mojo_backends"
    spec = importlib.util.spec_from_file_location(module_name, _NATIVE_BACKEND)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _specialized(name: str, radius: int) -> Any:
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    return getattr(_ops(), name)[{"radius": int(radius)}]


def _check_cuda_tensor(name: str, tensor: Tensor, dtypes: torch.dtype | tuple[torch.dtype, ...] = torch.float32) -> Tensor:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if isinstance(dtypes, tuple):
        if tensor.dtype in dtypes:
            return tensor.contiguous()
        valid = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype {valid}, got {tensor.dtype}")
    if tensor.dtype != dtypes:
        raise TypeError(f"{name} must have dtype {dtypes}, got {tensor.dtype}")
    return tensor.contiguous()


def _check_feature_tensor(name: str, tensor: Tensor) -> Tensor:
    return _check_cuda_tensor(name, tensor, _FEATURE_DTYPES)


def _to_dtype(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _check_index_tensor(name: str, tensor: Tensor) -> Tensor:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.int64:
        raise TypeError(f"{name} must be torch.int64, got {tensor.dtype}")
    return tensor.contiguous()


def _corr_fractional_offsets(coords: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    x = coords[:, :, 0, None, None]
    y = coords[:, :, 1, None, None]
    return (x - x.floor()).to(dtype), (y - y.floor()).to(dtype)


def _interpolate_corr(corr: Tensor, coords: Tensor, radius: int) -> Tensor:
    diameter = 2 * int(radius) + 2
    dx, dy = _corr_fractional_offsets(coords, corr.dtype)
    out = (1 - dx) * (1 - dy) * corr[:, :, : diameter - 1, : diameter - 1]
    out = out + dx * (1 - dy) * corr[:, :, : diameter - 1, 1:diameter]
    out = out + (1 - dx) * dy * corr[:, :, 1:diameter, : diameter - 1]
    out = out + dx * dy * corr[:, :, 1:diameter, 1:diameter]
    return out.permute(0, 1, 3, 2, 4, 5)


def _expand_corr_grad(grad: Tensor, coords: Tensor, radius: int) -> Tensor:
    diameter = 2 * int(radius) + 2
    grad = grad.permute(0, 1, 3, 2, 4, 5).contiguous()
    dx, dy = _corr_fractional_offsets(coords, grad.dtype)
    corr_grad = torch.zeros(
        (coords.shape[0], coords.shape[1], diameter, diameter, coords.shape[3], coords.shape[4]),
        device=grad.device,
        dtype=grad.dtype,
    )
    corr_grad[:, :, : diameter - 1, : diameter - 1].add_((1 - dx) * (1 - dy) * grad)
    corr_grad[:, :, : diameter - 1, 1:diameter].add_(dx * (1 - dy) * grad)
    corr_grad[:, :, 1:diameter, : diameter - 1].add_((1 - dx) * dy * grad)
    corr_grad[:, :, 1:diameter, 1:diameter].add_(dx * dy * grad)
    return corr_grad


def smoke_scale(x: Tensor) -> Tensor:
    """Small CustomOpLibrary smoke op used by tests."""
    x = _check_cuda_tensor("x", x)
    out = torch.empty_like(x)
    _ops().altcorr_smoke_scale(out, x)
    torch.cuda.synchronize()
    return out


def patchify_forward(net: Tensor, coords: Tensor, radius: int) -> tuple[Tensor]:
    """Extract raw ``(2R + 2) x (2R + 2)`` patches from ``net``."""
    net = _check_feature_tensor("net", net)
    coords = _check_cuda_tensor("coords", coords)
    if net.ndim != 4:
        raise ValueError(f"net must have shape [B, C, H, W], got {tuple(net.shape)}")
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape [B, M, 2], got {tuple(coords.shape)}")

    batch, channels, _, _ = net.shape
    coord_batch, patches_n, _ = coords.shape
    if batch != coord_batch:
        raise ValueError(f"net batch ({batch}) and coords batch ({coord_batch}) differ")

    out_dtype = net.dtype
    net_native = net if net.dtype == torch.float32 else net.float()
    native_ops = _native_ops()
    if native_ops is not None:
        (patches,) = native_ops.patchify_forward(net_native, coords, int(radius))
        return (_to_dtype(patches, out_dtype),)

    diameter = 2 * int(radius) + 2
    patches = torch.empty((batch, patches_n, channels, diameter, diameter), device=net.device, dtype=torch.float32)
    _specialized("patchify_forward", radius)(patches, net_native, coords)
    torch.cuda.synchronize()
    return (_to_dtype(patches, out_dtype),)


def patchify_backward(net: Tensor, coords: Tensor, grad: Tensor, radius: int) -> tuple[Tensor]:
    """Gradient of :func:`patchify_forward` with respect to ``net``."""
    net = _check_feature_tensor("net", net)
    coords = _check_cuda_tensor("coords", coords)
    grad = _check_cuda_tensor("grad", grad, net.dtype)
    out_dtype = net.dtype
    net_native = net if net.dtype == torch.float32 else net.float()
    grad_native = grad if grad.dtype == torch.float32 else grad.float()
    native_ops = _native_ops()
    if native_ops is not None:
        (net_grad,) = native_ops.patchify_backward(net_native, coords, grad_native, int(radius))
        return (_to_dtype(net_grad, out_dtype),)

    net_grad = torch.zeros_like(net_native)
    torch.cuda.synchronize()
    _specialized("patchify_backward", radius)(net_grad, coords, grad_native)
    torch.cuda.synchronize()
    return (_to_dtype(net_grad, out_dtype),)


def forward(fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, radius: int) -> tuple[Tensor]:
    """Compute DPVO local correlation volumes."""
    fmap1 = _check_feature_tensor("fmap1", fmap1)
    fmap2 = _check_cuda_tensor("fmap2", fmap2, fmap1.dtype)
    coords = _check_cuda_tensor("coords", coords)
    ii = _check_index_tensor("ii", ii)
    jj = _check_index_tensor("jj", jj)
    if fmap1.ndim != 5 or fmap2.ndim != 5:
        raise ValueError("fmap1 and fmap2 must have shape [B, N, C, H, W]")
    if coords.ndim != 5 or coords.shape[2] != 2:
        raise ValueError(f"coords must have shape [B, M, 2, H_patch, W_patch], got {tuple(coords.shape)}")

    out_dtype = fmap1.dtype
    fmap1_native = fmap1 if fmap1.dtype == torch.float32 else fmap1.float()
    fmap2_native = fmap2 if fmap2.dtype == torch.float32 else fmap2.float()
    native_ops = _native_ops()
    if native_ops is not None:
        (corr,) = native_ops.corr_forward(fmap1_native, fmap2_native, coords, ii, jj, int(radius))
        return (_to_dtype(corr, out_dtype),)

    batch, edges, _, patch_h, patch_w = coords.shape
    neighborhood = 2 * int(radius) + 1
    corr = torch.empty((batch, edges, neighborhood, neighborhood, patch_h, patch_w), device=fmap1.device, dtype=torch.float32)
    _specialized("corr_forward", radius)(corr, fmap1_native, fmap2_native, coords, ii, jj)
    torch.cuda.synchronize()
    return (_to_dtype(corr, out_dtype),)


def backward(fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, grad: Tensor, radius: int) -> tuple[Tensor, Tensor]:
    """Gradient of :func:`forward` with respect to ``fmap1`` and ``fmap2``."""
    if fmap1.dtype == torch.float32 and fmap2.dtype == torch.float32 and coords.dtype == torch.float32 and grad.dtype == torch.float32:
        fmap1 = _check_cuda_tensor("fmap1", fmap1)
        fmap2 = _check_cuda_tensor("fmap2", fmap2)
        coords = _check_cuda_tensor("coords", coords)
        grad = _check_cuda_tensor("grad", grad)
        ii = _check_index_tensor("ii", ii)
        jj = _check_index_tensor("jj", jj)
        native_ops = _native_ops()
        if native_ops is not None:
            return native_ops.corr_backward((fmap1, fmap2, coords, ii, jj, grad, int(radius)))

        fmap1_grad = torch.zeros_like(fmap1)
        fmap2_grad = torch.zeros_like(fmap2)
        torch.cuda.synchronize()
        _specialized("corr_backward", radius)(fmap1_grad, fmap2_grad, fmap1, fmap2, coords, ii, jj, grad)
        torch.cuda.synchronize()
        return fmap1_grad, fmap2_grad

    fmap1 = _check_feature_tensor("fmap1", fmap1)
    fmap2 = _check_cuda_tensor("fmap2", fmap2, fmap1.dtype)
    coords = _check_cuda_tensor("coords", coords)
    grad = _check_cuda_tensor("grad", grad, fmap1.dtype)
    ii = _check_index_tensor("ii", ii)
    jj = _check_index_tensor("jj", jj)
    fmap1_dtype = fmap1.dtype
    fmap2_dtype = fmap2.dtype
    fmap1_native = fmap1 if fmap1.dtype == torch.float32 else fmap1.float()
    fmap2_native = fmap2 if fmap2.dtype == torch.float32 else fmap2.float()
    grad_native = grad if grad.dtype == torch.float32 else grad.float()
    native_ops = _native_ops()
    if native_ops is not None:
        fmap1_grad, fmap2_grad = native_ops.corr_backward((fmap1_native, fmap2_native, coords, ii, jj, grad_native, int(radius)))
        return _to_dtype(fmap1_grad, fmap1_dtype), _to_dtype(fmap2_grad, fmap2_dtype)

    fmap1_grad = torch.zeros_like(fmap1_native)
    fmap2_grad = torch.zeros_like(fmap2_native)
    torch.cuda.synchronize()
    _specialized("corr_backward", radius)(fmap1_grad, fmap2_grad, fmap1_native, fmap2_native, coords, ii, jj, grad_native)
    torch.cuda.synchronize()
    return _to_dtype(fmap1_grad, fmap1_dtype), _to_dtype(fmap2_grad, fmap2_dtype)
