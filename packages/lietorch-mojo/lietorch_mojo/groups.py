"""Lie group classes: a port of lietorch `groups.py` (VSLAM-LAB/lietorch @ 3e2695dd).

Same class names, method names, signatures and data layouts as upstream, so
`import lietorch_mojo as lietorch` is a drop-in. Only SE3 has a Mojo backend so
far; SO3, RxSO3 and Sim3 construct and reshape, but their ops raise
`NotImplementedError` from `lietorch_mojo.backends`.

Deviations from upstream (both are upstream crashes):
- `quaternion()` returns the quaternion slice of the data (ViPE's fix);
  upstream references an undefined `Quat` op and raises `NameError`.
- `Sim3(SO3)` builds the Sim3 data from the SO3 instance; upstream reads the
  class attribute `SO3.data` and raises `AttributeError`.
"""

from collections.abc import Sequence
from typing import ClassVar, Self, TypeAlias

import numpy as np
import torch
from torch import Tensor

from lietorch_mojo.broadcasting import broadcast_inputs
from lietorch_mojo.group_ops import Act3, Act4, Adj, AdjT, Exp, FromVec, GroupOp, Inv, Jinv, Log, Mul, ToVec
from lietorch_mojo.group_ops import ToMatrix as ToMatrix  # upstream groups.py imports it, so lietorch.groups.ToMatrix exists

# What upstream accepts, which is whatever numpy and torch accept there.
Dim: TypeAlias = int | np.integer
BatchShape: TypeAlias = Dim | tuple[Dim, ...] | list[Dim]
# torch factory keyword values (`out=` takes a Tensor).
FactoryKwarg: TypeAlias = torch.device | str | int | torch.dtype | bool | torch.Generator | torch.layout | Tensor | None
Scalar: TypeAlias = float | int | np.number | Tensor


class LieGroupParameter(torch.Tensor):
    """Wrapper class for LieGroup"""

    from torch._C import _disabled_torch_function_impl

    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, group: "LieGroup", requires_grad: bool = True) -> "LieGroupParameter":
        data: Tensor = torch.zeros(group.tangent_shape, device=group.data.device, dtype=group.data.dtype, requires_grad=True)

        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, group: "LieGroup") -> None:
        self.group = group

    def retr(self) -> "LieGroup":
        return self.group.retr(self)

    def log(self) -> Tensor:
        return self.retr().log()

    def inv(self) -> "LieGroup":
        return self.retr().inv()

    def adj(self, a: Tensor) -> Tensor:
        return self.retr().adj(a)

    def __mul__(self, other: object) -> "LieGroup | Tensor | None":
        if isinstance(other, LieGroupParameter):
            return self.retr() * other.retr()
        else:
            return self.retr() * other

    def add_(self, update: Tensor, alpha: Scalar) -> None:
        self.group = self.group.exp(alpha * update) * self.group

    def __getitem__(self, index: object) -> "LieGroup":
        return self.retr().__getitem__(index)


class LieGroup:
    """Base class for Lie Group"""

    group_name: ClassVar[str]
    group_id: ClassVar[int]
    manifold_dim: ClassVar[int]
    embedded_dim: ClassVar[int]
    id_elem: ClassVar[Tensor]

    def __init__(self, data: Tensor) -> None:
        self.data = data

    def __repr__(self) -> str:
        return f"{self.group_name}: size={self.shape}, device={self.device}, dtype={self.dtype}"

    @property
    def shape(self) -> torch.Size:
        return self.data.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def vec(self) -> Tensor:
        return self.apply_op(ToVec, self.data)

    @property
    def tangent_shape(self) -> torch.Size:
        return self.data.shape[:-1] + (self.manifold_dim,)

    @classmethod
    def Identity(cls, *batch_shape: BatchShape, **kwargs: FactoryKwarg) -> Self:
        """Construct identity element with batch shape"""

        if isinstance(batch_shape[0], tuple):
            batch_shape = batch_shape[0]

        elif isinstance(batch_shape[0], list):
            batch_shape = tuple(batch_shape[0])

        numel = np.prod(batch_shape)
        data: Tensor = cls.id_elem.reshape(1, -1)

        if "device" in kwargs:
            data = data.to(kwargs["device"])

        if "dtype" in kwargs:
            data = data.type(kwargs["dtype"])

        data = data.repeat(numel, 1)
        return cls(data).view(batch_shape)

    @classmethod
    def IdentityLike(cls, G: "LieGroup") -> Self:
        return cls.Identity(G.shape, device=G.data.device, dtype=G.data.dtype)

    @classmethod
    def InitFromVec(cls, data: Tensor) -> Self:
        return cls(cls.apply_op(FromVec, data))

    @classmethod
    def Random(cls, *batch_shape: BatchShape, sigma: Scalar = 1.0, **kwargs: FactoryKwarg) -> Self:
        """Construct random element with batch_shape by random sampling in tangent space"""

        if isinstance(batch_shape[0], tuple):
            batch_shape = batch_shape[0]

        elif isinstance(batch_shape[0], list):
            batch_shape = tuple(batch_shape[0])

        tangent_shape = batch_shape + (cls.manifold_dim,)
        xi: Tensor = torch.randn(tangent_shape, **kwargs)
        return cls.exp(sigma * xi)

    @classmethod
    def apply_op(cls, op: type[GroupOp] | type[FromVec] | type[ToVec], x: Tensor, y: Tensor | None = None) -> Tensor:
        """Apply group operator"""
        inputs, out_shape = broadcast_inputs(x, y)

        data: Tensor = op.apply(cls.group_id, *inputs)
        return data.view(out_shape + (-1,))

    @classmethod
    def exp(cls, x: Tensor) -> Self:
        """exponential map: x -> X"""
        return cls(cls.apply_op(Exp, x))

    def quaternion(self) -> Tensor:
        """extract quaternion"""
        if self.group_id in (1, 2):
            return self.data[..., :4]
        return self.data[..., 3:7]

    def log(self) -> Tensor:
        """logarithm map"""
        return self.apply_op(Log, self.data)

    def inv(self) -> Self:
        """group inverse"""
        return self.__class__(self.apply_op(Inv, self.data))

    def mul(self, other: "LieGroup") -> Self:
        """group multiplication"""
        return self.__class__(self.apply_op(Mul, self.data, other.data))

    def retr(self, a: Tensor) -> Self:
        """retraction: Exp(a) * X"""
        dX: Tensor = self.__class__.apply_op(Exp, a)
        return self.__class__(self.apply_op(Mul, dX, self.data))

    def adj(self, a: Tensor) -> Tensor:
        """adjoint operator: b = A(X) * a"""
        return self.apply_op(Adj, self.data, a)

    def adjT(self, a: Tensor) -> Tensor:
        """transposed adjoint operator: b = a * A(X)"""
        return self.apply_op(AdjT, self.data, a)

    def Jinv(self, a: Tensor) -> Tensor:
        return self.apply_op(Jinv, self.data, a)

    def act(self, p: Tensor) -> Tensor | None:
        """action on a point cloud"""

        # action on point
        if p.shape[-1] == 3:
            return self.apply_op(Act3, self.data, p)

        # action on homogeneous point
        elif p.shape[-1] == 4:
            return self.apply_op(Act4, self.data, p)

    def matrix(self) -> Tensor:
        """convert element to 4x4 matrix"""
        eye: Tensor = torch.eye(4, dtype=self.dtype, device=self.device)
        eye = eye.view([1] * (len(self.data.shape) - 1) + [4, 4])
        return self.__class__(self.data[..., None, :]).act(eye).transpose(-1, -2)

    def translation(self) -> Tensor:
        """extract translation component"""
        p: Tensor = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        p = p.view([1] * (len(self.data.shape) - 1) + [4])
        return self.apply_op(Act4, self.data, p)

    def detach(self) -> Self:
        return self.__class__(self.data.detach())

    def view(self, dims: tuple[Dim, ...]) -> Self:
        data_reshaped: Tensor = self.data.view(dims + (self.embedded_dim,))
        return self.__class__(data_reshaped)

    def __mul__(self, other: object) -> "LieGroup | Tensor | None":
        # group multiplication
        if isinstance(other, LieGroup):
            return self.mul(other)

        # action on point
        elif isinstance(other, torch.Tensor):
            return self.act(other)

    def __getitem__(self, index: object) -> Self:
        return self.__class__(self.data[index])

    def __setitem__(self, index: object, item: "LieGroup | Tensor") -> None:
        self.data[index] = item.data

    def to(self, *args: object, **kwargs: object) -> Self:
        return self.__class__(self.data.to(*args, **kwargs))

    def cpu(self) -> Self:
        return self.__class__(self.data.cpu())

    def cuda(self) -> Self:
        return self.__class__(self.data.cuda())

    def float(self, device: object) -> Self:
        return self.__class__(self.data.float())

    def double(self, device: object) -> Self:
        return self.__class__(self.data.double())

    def unbind(self, dim: Dim = 0) -> list[Self]:
        return [self.__class__(x) for x in self.data.unbind(dim=dim)]


class SO3(LieGroup):
    group_name = "SO3"
    group_id = 1
    manifold_dim = 3
    embedded_dim = 4

    # unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 1.0])

    def __init__(self, data: "Tensor | SE3") -> None:
        if isinstance(data, SE3):
            data = data.data[..., 3:7]

        super().__init__(data)


class RxSO3(LieGroup):
    group_name = "RxSO3"
    group_id = 2
    manifold_dim = 4
    embedded_dim = 5

    # unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 1.0, 1.0])

    def __init__(self, data: "Tensor | Sim3") -> None:
        if isinstance(data, Sim3):
            data = data.data[..., 3:8]

        super().__init__(data)


class SE3(LieGroup):
    group_name = "SE3"
    group_id = 3
    manifold_dim = 6
    embedded_dim = 7

    # translation, unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def __init__(self, data: "Tensor | SO3") -> None:
        if isinstance(data, SO3):
            translation: Tensor = torch.zeros_like(data.data[..., :3])
            data = torch.cat([translation, data.data], -1)

        super().__init__(data)

    def scale(self, s: Tensor) -> "SE3":
        t, q = self.data.split([3, 4], -1)
        t = t * s.unsqueeze(-1)
        return SE3(torch.cat([t, q], dim=-1))


class Sim3(LieGroup):
    group_name = "Sim3"
    group_id = 4
    manifold_dim = 7
    embedded_dim = 8

    # translation, unit quaternion, scale
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    def __init__(self, data: "Tensor | SO3 | SE3 | Sim3") -> None:
        if isinstance(data, SO3):
            scale: Tensor = torch.ones_like(data.data[..., :1])
            translation: Tensor = torch.zeros_like(data.data[..., :3])
            data = torch.cat([translation, data.data, scale], -1)

        elif isinstance(data, SE3):
            scale = torch.ones_like(data.data[..., :1])
            data = torch.cat([data.data, scale], -1)

        elif isinstance(data, Sim3):
            data = data.data

        super().__init__(data)


def cat(group_list: Sequence[LieGroup], dim: Dim) -> LieGroup:
    """Concatenate groups along dimension"""
    data: Tensor = torch.cat([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)


def stack(group_list: Sequence[LieGroup], dim: Dim) -> LieGroup:
    """Concatenate groups along dimension"""
    data: Tensor = torch.stack([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)
