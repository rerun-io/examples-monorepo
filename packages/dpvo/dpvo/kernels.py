"""Kernel-op dispatch by device type.

DPVO's native ops (correlation, patch extraction, bundle adjustment) live in
compiled backends. This module maps a ``torch.device`` type to the backend
that serves it, so callers pick the backend from their input tensor's
``.device`` and pass the tensors through unchanged: the lookup is a dict
access, with no copies and no device syncs.

The CUDA backend (the ``dpvo._cuda_corr`` and ``dpvo._cuda_ba`` extensions)
is imported lazily on the first CUDA lookup, and the Mojo backend
(``dpvo.mojo_backend``) on the first CPU or MPS lookup. Any other backend package
registers itself at import time::

    import dpvo.kernels
    dpvo.kernels.register("mps", corr=my_corr_module, ba=my_ba_module)

The extension signatures are the contract: a backend is any object (a
module of functions is enough) that satisfies :class:`CorrKernels` and
:class:`BAKernels` structurally.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class CorrKernels(Protocol):
    """Correlation and patch-extraction ops (the ``_cuda_corr`` contract)."""

    def forward(self, fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, radius: int) -> Sequence[Tensor]:
        """Correlation volume of ``fmap1[ii]`` against ``fmap2[jj]`` around ``coords``; returns ``(corr,)``."""
        ...

    def backward(self, fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, grad: Tensor, radius: int) -> Sequence[Tensor]:
        """Gradients of the correlation volume; returns ``(fmap1_grad, fmap2_grad)``."""
        ...

    def patchify_forward(self, net: Tensor, coords: Tensor, radius: int) -> Sequence[Tensor]:
        """Extract ``(2r + 2)²`` patches of ``net`` at ``coords``; returns ``(patches,)``."""
        ...

    def patchify_backward(self, net: Tensor, coords: Tensor, grad: Tensor, radius: int) -> Sequence[Tensor]:
        """Gradient of the patch extraction w.r.t. ``net``; returns ``(net_grad,)``."""
        ...


@runtime_checkable
class BAKernels(Protocol):
    """Bundle-adjustment ops (the ``_cuda_ba`` contract)."""

    def forward(
        self,
        poses: Tensor,
        patches: Tensor,
        intrinsics: Tensor,
        target: Tensor,
        weight: Tensor,
        lmbda: Tensor,
        depth_prior: Tensor,
        depth_prior_weight: Tensor,
        depth_prior_lambda: float,
        depth_prior_huber: float,
        ii: Tensor,
        jj: Tensor,
        kk: Tensor,
        M: int,
        t0: int,
        t1: int,
        iterations: int,
        eff_impl: bool,
    ) -> Sequence[Tensor]:
        """Gauss-Newton BA that updates ``poses`` and ``patches`` in place; returns an empty sequence."""
        ...

    def neighbors(self, ii: Tensor, jj: Tensor) -> Sequence[Tensor]:
        """Previous/next observation of the same patch per edge; returns ``(ix, jx)``."""
        ...

    def reproject(self, poses: Tensor, patches: Tensor, intrinsics: Tensor, ii: Tensor, jj: Tensor, kk: Tensor) -> Tensor:
        """Reproject every patch pixel from frame ``ii`` into frame ``jj``."""
        ...

    def solve_system(
        self, J_Ginv_i: Tensor, J_Ginv_j: Tensor, ii: Tensor, jj: Tensor, res: Tensor, ep: float, lm: float, freen: int
    ) -> Sequence[Tensor]:
        """Sparse Sim(3) pose-graph solve; returns ``(delta,)``."""
        ...


@dataclass(frozen=True, slots=True)
class _Backend:
    """The two op groups that serve one device type."""

    corr: CorrKernels
    """Correlation and patch-extraction ops."""
    ba: BAKernels
    """Bundle-adjustment ops."""


_BACKENDS: dict[str, _Backend] = {}


def register(device_type: str, *, corr: CorrKernels, ba: BAKernels) -> None:
    """Register the kernel backend for one device type (replaces an earlier registration).

    Args:
        device_type: ``torch.device.type`` the backend serves, e.g. ``"cuda"``.
        corr: Correlation and patch-extraction ops.
        ba: Bundle-adjustment ops.
    """
    _BACKENDS[device_type] = _Backend(corr=corr, ba=ba)


def _register_cuda() -> None:
    """Import the CUDA extensions and register them for ``"cuda"``."""
    from dpvo import _cuda_ba, _cuda_corr

    register("cuda", corr=_cuda_corr, ba=_cuda_ba)


def _backend(device: torch.device) -> _Backend:
    """Return the backend for ``device.type``, importing the CUDA or Mojo one on first use.

    Raises:
        LookupError: No backend is registered for ``device.type``.
    """
    backend: _Backend | None = _BACKENDS.get(device.type)
    if backend is None and device.type == "cuda":
        _register_cuda()
        backend = _BACKENDS["cuda"]
    if backend is None and device.type in ("cpu", "mps"):
        from dpvo import mojo_backend

        mojo_backend.register()
        backend = _BACKENDS[device.type]
    if backend is None:
        raise LookupError(
            f"no DPVO kernel backend for device type {device.type!r}; "
            f"a backend package must call dpvo.kernels.register({device.type!r}, corr=..., ba=...) "
            f"(registered: {sorted(_BACKENDS)})"
        )
    return backend


def corr_kernels(device: torch.device) -> CorrKernels:
    """Return the correlation/patchify ops for ``device``.

    Raises:
        LookupError: No backend is registered for ``device.type``.
    """
    return _backend(device).corr


def ba_kernels(device: torch.device) -> BAKernels:
    """Return the bundle-adjustment ops for ``device``.

    Raises:
        LookupError: No backend is registered for ``device.type``.
    """
    return _backend(device).ba
