import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .scatter_utils import scatter_softmax, scatter_sum


class LayerNorm1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x: Float[Tensor, "batch dim length"]) -> Float[Tensor, "batch dim length"]:
        return self.norm(x.transpose(1,2)).transpose(1,2)

class GatedResidual(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.gate: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.res: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch dim"]:
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):
    def __init__(self, dim: int = 512, expand: bool = True) -> None:
        super().__init__()
        self.dim: int = dim
        self.expand: bool = expand
        self.f: nn.Linear = nn.Linear(self.dim, self.dim)
        self.g: nn.Linear = nn.Linear(self.dim, self.dim)
        self.h: nn.Linear = nn.Linear(self.dim, self.dim)

    def forward(self, x: Float[Tensor, "batch edges dim"], ix: Int[Tensor, "batch edges"]) -> Float[Tensor, "..."]:
        _: Tensor
        jx: Tensor
        _, jx = torch.unique(ix, return_inverse=True)
        w: Float[Tensor, "..."] = scatter_softmax(self.g(x), jx, dim=1)
        y: Float[Tensor, "..."] = scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]

        return self.h(y)

class SoftAggBasic(nn.Module):
    def __init__(self, dim: int = 512, expand: bool = True) -> None:
        super().__init__()
        self.dim: int = dim
        self.expand: bool = expand
        self.f: nn.Linear = nn.Linear(self.dim, self.dim)
        self.g: nn.Linear = nn.Linear(self.dim,        1)
        self.h: nn.Linear = nn.Linear(self.dim, self.dim)

    def forward(self, x: Float[Tensor, "batch edges dim"], ix: Int[Tensor, "batch edges"]) -> Float[Tensor, "..."]:
        _: Tensor
        jx: Tensor
        _, jx = torch.unique(ix, return_inverse=True)
        w: Float[Tensor, "..."] = scatter_softmax(self.g(x), jx, dim=1)
        y: Float[Tensor, "..."] = scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]

        return self.h(y)


### Gradient Clipping and Zeroing Operations ###

GRAD_CLIP: float = 0.1

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return GradClip.apply(x)

class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x

class GradientZero(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        print(grad_x.abs().mean())
        return grad_x
