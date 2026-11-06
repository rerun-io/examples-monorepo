"""Autograd functions over the backend ops: a port of lietorch `group_ops.py`.

Same classes and the same autograd structure as upstream; the only change is
that the ops come from `lietorch_mojo.backends` instead of the compiled
`lietorch_backends` extension.
"""

from collections.abc import Callable
from typing import ClassVar

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from lietorch_mojo import backends


class GroupOp(torch.autograd.Function):
    """group operation base class"""

    forward_op: ClassVar[Callable[..., Tensor]]
    backward_op: ClassVar[Callable[..., list[Tensor]] | None]

    @classmethod
    def forward(cls, ctx: FunctionCtx, group_id: int, *inputs: Tensor) -> Tensor:
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        out: Tensor = cls.forward_op(ctx.group_id, *inputs)
        return out

    @classmethod
    def backward(cls, ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor | None, ...]:
        error_str: str = f"Backward operation not implemented for {cls}"
        assert cls.backward_op is not None, error_str

        inputs: tuple[Tensor, ...] = ctx.saved_tensors
        grad = grad.contiguous()
        grad_inputs: list[Tensor] = cls.backward_op(ctx.group_id, grad, *inputs)
        return (None,) + tuple(grad_inputs)


class Exp(GroupOp):
    """exponential map"""

    forward_op, backward_op = backends.expm, backends.expm_backward


class Log(GroupOp):
    """logarithm map"""

    forward_op, backward_op = backends.logm, backends.logm_backward


class Inv(GroupOp):
    """group inverse"""

    forward_op, backward_op = backends.inv, backends.inv_backward


class Mul(GroupOp):
    """group multiplication"""

    forward_op, backward_op = backends.mul, backends.mul_backward


class Adj(GroupOp):
    """adjoint operator"""

    forward_op, backward_op = backends.adj, backends.adj_backward


class AdjT(GroupOp):
    """adjoint operator"""

    forward_op, backward_op = backends.adjT, backends.adjT_backward


class Act3(GroupOp):
    """action on point"""

    forward_op, backward_op = backends.act, backends.act_backward


class Act4(GroupOp):
    """action on point"""

    forward_op, backward_op = backends.act4, backends.act4_backward


class Jinv(GroupOp):
    """adjoint operator"""

    forward_op, backward_op = backends.Jinv, None


class ToMatrix(GroupOp):
    """convert to matrix representation"""

    forward_op, backward_op = backends.as_matrix, None


### conversion operations to/from Euclidean embeddings ###


class FromVec(torch.autograd.Function):
    """convert vector into group object"""

    @classmethod
    def forward(cls, ctx: FunctionCtx, group_id: int, *inputs: Tensor) -> Tensor:
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx: FunctionCtx, grad: Tensor) -> tuple[None, Tensor]:
        inputs: tuple[Tensor, ...] = ctx.saved_tensors
        J: Tensor = backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), torch.linalg.pinv(J)).squeeze(-2)


class ToVec(torch.autograd.Function):
    """convert group object to vector"""

    @classmethod
    def forward(cls, ctx: FunctionCtx, group_id: int, *inputs: Tensor) -> Tensor:
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx: FunctionCtx, grad: Tensor) -> tuple[None, Tensor]:
        inputs: tuple[Tensor, ...] = ctx.saved_tensors
        J: Tensor = backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), J).squeeze(-2)
