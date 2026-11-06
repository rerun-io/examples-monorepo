"""Raw backend-op comparison between lietorch_backends and lietorch_mojo.backends.

Both modules expose the same functions with the same signatures, so one table
of (op name, argument builder) drives forward and backward comparisons.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import torch
from cases import Se3Case
from torch import Tensor

SE3_ID: int = 3

# Upstream gradient fed to each backward op: (B, width) where width is the size
# of the op's output row (group-valued outputs get all 7 slots filled, as
# autograd does; lietorch reads only the first 6).
FORWARD_OPS: tuple[str, ...] = ("expm", "logm", "inv", "mul", "adj", "adjT", "act", "act4", "as_matrix", "projector", "Jinv")
BACKWARD_OPS: tuple[str, ...] = (
    "expm_backward",
    "logm_backward",
    "inv_backward",
    "mul_backward",
    "adj_backward",
    "adjT_backward",
    "act_backward",
    "act4_backward",
)
_GRAD_WIDTH: dict[str, int] = {
    "expm_backward": 7,
    "logm_backward": 6,
    "inv_backward": 7,
    "mul_backward": 7,
    "adj_backward": 6,
    "adjT_backward": 6,
    "act_backward": 3,
    "act4_backward": 4,
}


@dataclass(frozen=True, slots=True)
class OpInputs:
    """Tensors for one op call, already on the target device and dtype."""

    X: Tensor
    """Group elements (B, 7)."""
    Y: Tensor
    """Second group elements for mul (B, 7)."""
    a: Tensor
    """Tangent vectors (B, 6)."""
    p3: Tensor
    """Points (B, 3)."""
    p4: Tensor
    """Homogeneous points (B, 4)."""
    grads: dict[str, Tensor]
    """Upstream gradient per backward op name."""


def make_inputs(case: Se3Case, dtype: torch.dtype, device: str, seed: int = 0) -> OpInputs:
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    n: int = case.X.shape[0]
    grads: dict[str, Tensor] = {
        name: torch.randn(n, width, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype) for name, width in _GRAD_WIDTH.items()
    }
    return OpInputs(
        X=case.X.to(device=device, dtype=dtype).contiguous(),
        Y=case.X.roll(1, dims=0).to(device=device, dtype=dtype).contiguous(),
        a=case.a.to(device=device, dtype=dtype).contiguous(),
        p3=case.p3.to(device=device, dtype=dtype).contiguous(),
        p4=case.p4.to(device=device, dtype=dtype).contiguous(),
        grads=grads,
    )


def call_op(backend: ModuleType, op: str, inp: OpInputs) -> list[Tensor]:
    """Call `backend.<op>` on SE3 with the op's natural arguments; always return a list of outputs."""
    fn: Callable[..., Tensor | list[Tensor]] = getattr(backend, op)
    args: dict[str, tuple[Tensor, ...]] = {
        "expm": (inp.a,),
        "logm": (inp.X,),
        "inv": (inp.X,),
        "mul": (inp.X, inp.Y),
        "adj": (inp.X, inp.a),
        "adjT": (inp.X, inp.a),
        "act": (inp.X, inp.p3),
        "act4": (inp.X, inp.p4),
        "as_matrix": (inp.X,),
        "projector": (inp.X,),
        "Jinv": (inp.X, inp.a),
        "expm_backward": (inp.grads["expm_backward"], inp.a),
        "logm_backward": (inp.grads["logm_backward"], inp.X),
        "inv_backward": (inp.grads["inv_backward"], inp.X),
        "mul_backward": (inp.grads["mul_backward"], inp.X, inp.Y),
        "adj_backward": (inp.grads["adj_backward"], inp.X, inp.a),
        "adjT_backward": (inp.grads["adjT_backward"], inp.X, inp.a),
        "act_backward": (inp.grads["act_backward"], inp.X, inp.p3),
        "act4_backward": (inp.grads["act4_backward"], inp.X, inp.p4),
    }
    out: Tensor | list[Tensor] = fn(SE3_ID, *args[op])
    return out if isinstance(out, list) else [out]
