import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from mini_dpvo import _cuda_corr


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        fmap1: Float[Tensor, "1 mem channels h4 w4"],
        fmap2: Float[Tensor, "1 mem channels h8 w8"],
        coords: Float[Tensor, "1 n_edges ps ps 2"],
        ii: Int[Tensor, "n_edges"],
        jj: Int[Tensor, "n_edges"],
        radius: int,
        dropout: float,
    ) -> Float[Tensor, "1 n_edges corr_dim"]:
        """ forward correlation """
        ctx.save_for_backward(fmap1, fmap2, coords, ii, jj)
        ctx.radius = radius
        ctx.dropout = dropout
        corr: Float[Tensor, "1 n_edges corr_dim"]
        corr, = _cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)

        return corr

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: Float[Tensor, "..."],
    ) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."], None, None, None, None, None]:
        """ backward correlation """
        fmap1: Float[Tensor, "..."]
        fmap2: Float[Tensor, "..."]
        coords: Float[Tensor, "..."]
        ii: Int[Tensor, "..."]
        jj: Int[Tensor, "..."]
        fmap1, fmap2, coords, ii, jj = ctx.saved_tensors

        if ctx.dropout < 1:
            perm: Bool[Tensor, "n_edges"] = torch.rand(len(ii), device="cuda") < ctx.dropout
            coords = coords[:,perm]
            grad = grad[:,perm]
            ii = ii[perm]
            jj = jj[perm]

        fmap1_grad: Float[Tensor, "..."]
        fmap2_grad: Float[Tensor, "..."]
        fmap1_grad, fmap2_grad = \
            _cuda_corr.backward(fmap1, fmap2, coords, ii, jj, grad, ctx.radius)

        return fmap1_grad, fmap2_grad, None, None, None, None, None


class PatchLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        net: Float[Tensor, "..."],
        coords: Float[Tensor, "..."],
        radius: int,
    ) -> Float[Tensor, "..."]:
        """ forward patchify """
        ctx.radius = radius
        ctx.save_for_backward(net, coords)

        patches: Float[Tensor, "..."]
        patches, = _cuda_corr.patchify_forward(net, coords, radius)
        return patches

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: Float[Tensor, "..."],
    ) -> tuple[Float[Tensor, "..."], None, None]:
        """ backward patchify """
        net: Float[Tensor, "..."]
        coords: Float[Tensor, "..."]
        net, coords = ctx.saved_tensors
        grad, = _cuda_corr.patchify_backward(net, coords, grad, ctx.radius)

        return grad, None, None

def patchify(net: Float[Tensor, "..."], coords: Float[Tensor, "..."], radius: int, mode: str = 'bilinear') -> Float[Tensor, "..."]:
    """ extract patches """

    patches: Float[Tensor, "..."] = PatchLayer.apply(net, coords, radius)

    if mode == 'bilinear':
        offset: Float[Tensor, "..."] = (coords - coords.floor()).to(net.device)
        dx: Float[Tensor, "..."]
        dy: Float[Tensor, "..."]
        dx, dy = offset[:,:,None,None,None].unbind(dim=-1)

        d: int = 2 * radius + 1
        x00: Float[Tensor, "..."] = (1-dy) * (1-dx) * patches[...,:d,:d]
        x01: Float[Tensor, "..."] = (1-dy) * (  dx) * patches[...,:d,1:]
        x10: Float[Tensor, "..."] = (  dy) * (1-dx) * patches[...,1:,:d]
        x11: Float[Tensor, "..."] = (  dy) * (  dx) * patches[...,1:,1:]

        return x00 + x01 + x10 + x11

    return patches


def corr(
    fmap1: Float[Tensor, "..."],
    fmap2: Float[Tensor, "..."],
    coords: Float[Tensor, "..."],
    ii: Int[Tensor, "..."],
    jj: Int[Tensor, "..."],
    radius: int = 1,
    dropout: float = 1.0,
) -> Float[Tensor, "..."]:
    return CorrLayer.apply(fmap1, fmap2, coords, ii, jj, radius, dropout)
