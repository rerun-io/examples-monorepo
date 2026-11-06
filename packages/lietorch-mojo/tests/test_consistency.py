"""Bitwise consistency between kernels: the same rotation of the same point rounds the same way in every op.

Regression (fix-r2): with FP contraction left to the compiler, `act` and `mul`
fused different multiply-adds of R t, so float32 X.inv() * X kept ulp
residuals in ~29% of rows (CPU) where lietorch's CPU build cancels exactly.
`SO3.act` now spells out its FMAs.
"""

import pytest
import torch
from conftest import DEVICES, DTYPES, require_mojo_extension
from torch import Tensor

require_mojo_extension()

from lietorch_mojo import backends  # noqa: E402


def _poses(device: str, dtype: torch.dtype) -> Tensor:
    gen: torch.Generator = torch.Generator().manual_seed(3)
    xi: Tensor = torch.randn(4096, 6, generator=gen, dtype=torch.float64)
    xi[:, :3] *= 1000
    return backends.expm(3, xi.to(dtype).to(device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rotation_rounds_the_same_in_act_and_mul(device: str, dtype: torch.dtype) -> None:
    X: Tensor = _poses(device, dtype)
    Y: Tensor = backends.inv(3, X)
    Y0: Tensor = Y.clone()
    Y0[:, :3] = 0  # pure rotation: act gives R t, mul gives 0 + R t
    via_act: Tensor = backends.act(3, Y0, X[:, :3].contiguous())
    via_mul: Tensor = backends.mul(3, Y0, X)[:, :3]
    assert torch.equal(via_act, via_mul)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_inverse_times_self_cancels_in_almost_every_row(device: str, dtype: torch.dtype) -> None:
    """X.inv() * X has an exactly zero translation in all but ~1% of rows (|t| ~ 1e3).

    The rows left are those whose X.inv() quaternion changes bits when it is normalised again on
    load. In the fix-r2 diagnosis (xinvx_diag.py, float32, CPU) lietorch's CPU build leaves 0.9% of
    rows and this port 0.6%; before the explicit FMAs this port left 29%.
    """
    X: Tensor = _poses(device, dtype)
    residual: Tensor = backends.mul(3, backends.inv(3, X), X)[:, :3]
    assert (residual != 0).any(dim=1).float().mean().item() <= 0.01
