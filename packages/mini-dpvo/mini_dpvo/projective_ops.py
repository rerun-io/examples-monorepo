from typing import Any

import torch
from jaxtyping import Float, Int
from lietorch import SE3
from torch import Tensor

MIN_DEPTH: float = 0.2

def extract_intrinsics(intrinsics: Float[Tensor, "... 4"]) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]:
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht: int, wd: int, **kwargs: Any) -> Float[Tensor, "h w 2"]:
    y: Float[Tensor, "h"]
    x: Float[Tensor, "w"]
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)


def iproj(patches: Float[Tensor, "... 3 ps ps"], intrinsics: Float[Tensor, "... 4"]) -> Float[Tensor, "... ps ps 4"]:
    """ inverse projection """
    x: Float[Tensor, "... ps ps"]
    y: Float[Tensor, "... ps ps"]
    d: Float[Tensor, "... ps ps"]
    x, y, d = patches.unbind(dim=2)

    fx: Float[Tensor, "..."]
    fy: Float[Tensor, "..."]
    cx: Float[Tensor, "..."]
    cy: Float[Tensor, "..."]
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    i: Float[Tensor, "... ps ps"] = torch.ones_like(d)
    xn: Float[Tensor, "... ps ps"] = (x - cx) / fx
    yn: Float[Tensor, "... ps ps"] = (y - cy) / fy

    X: Float[Tensor, "... ps ps 4"] = torch.stack([xn, yn, i, d], dim=-1)
    return X


def proj(X: Float[Tensor, "... 4"], intrinsics: Float[Tensor, "... 4"], depth: bool = False) -> Float[Tensor, "..."]:
    """ projection """

    Y: Float[Tensor, "..."]
    Z: Float[Tensor, "..."]
    W: Float[Tensor, "..."]
    X, Y, Z, W = X.unbind(dim=-1)

    fx: Float[Tensor, "..."]
    fy: Float[Tensor, "..."]
    cx: Float[Tensor, "..."]
    cy: Float[Tensor, "..."]
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    # d = 0.01 * torch.ones_like(Z)
    # d[Z > 0.01] = 1.0 / Z[Z > 0.01]
    # d = torch.ones_like(Z)
    # d[Z.abs() > 0.1] = 1.0 / Z[Z.abs() > 0.1]

    d: Float[Tensor, "..."] = 1.0 / Z.clamp(min=0.1)
    x: Float[Tensor, "..."] = fx * (d * X) + cx
    y: Float[Tensor, "..."] = fy * (d * Y) + cy

    if depth:
        return torch.stack([x, y, d], dim=-1)

    return torch.stack([x, y], dim=-1)


def transform(
    poses: SE3,
    patches: Float[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float[Tensor, "1 n_frames 4"],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    depth: bool = False,
    valid: bool = False,
    jacobian: bool = False,
    tonly: bool = False,
) -> Float[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."], tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]]:
    """ projective transform """

    # backproject
    X0: Float[Tensor, "1 n_edges ps ps 4"] = iproj(patches[:,kk], intrinsics[:,ii])

    # transform
    Gij: SE3 = poses[:, jj] * poses[:, ii].inv()

    if tonly:
        Gij[...,3:] = torch.as_tensor([0,0,0,1], device=Gij.device)

    X1: Float[Tensor, "1 n_edges ps ps 4"] = Gij[:,:,None,None] * X0

    # project
    x1: Float[Tensor, "..."] = proj(X1, intrinsics[:,jj], depth)


    if jacobian:
        p: int = X1.shape[2]
        X: Float[Tensor, "1 n_edges"]
        Y: Float[Tensor, "1 n_edges"]
        Z: Float[Tensor, "1 n_edges"]
        H: Float[Tensor, "1 n_edges"]
        X, Y, Z, H = X1[...,p//2,p//2,:].unbind(dim=-1)
        o: Float[Tensor, "1 n_edges"] = torch.zeros_like(H)
        i: Float[Tensor, "1 n_edges"] = torch.zeros_like(H)

        fx: Float[Tensor, "1 n_edges"]
        fy: Float[Tensor, "1 n_edges"]
        cx: Float[Tensor, "1 n_edges"]
        cy: Float[Tensor, "1 n_edges"]
        fx, fy, cx, cy = intrinsics[:,jj].unbind(dim=-1)

        d: Float[Tensor, "1 n_edges"] = torch.zeros_like(Z)
        d[Z.abs() > 0.2] = 1.0 / Z[Z.abs() > 0.2]

        Ja: Float[Tensor, "1 n_edges 4 6"] = torch.stack([
            H,  o,  o,  o,  Z, -Y,
            o,  H,  o, -Z,  o,  X,
            o,  o,  H,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(1, len(ii), 4, 6)

        Jp: Float[Tensor, "1 n_edges 2 4"] = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
        ], dim=-1).view(1, len(ii), 2, 4)

        Jj: Float[Tensor, "1 n_edges 2 6"] = torch.matmul(Jp, Ja)
        Ji: Float[Tensor, "1 n_edges 2 6"] = -Gij[:,:,None].adjT(Jj)

        Jz: Float[Tensor, "1 n_edges 2 1"] = torch.matmul(Jp, Gij.matrix()[...,:,3:])

        return x1, (Z > 0.2).float(), (Ji, Jj, Jz)

    if valid:
        return x1, (X1[...,2] > 0.2).float()

    return x1

def point_cloud(poses: SE3, patches: Float[Tensor, "..."], intrinsics: Float[Tensor, "..."], ix: Int[Tensor, "..."]) -> Float[Tensor, "..."]:
    """ generate point cloud from patches """
    return poses[:,ix,None,None].inv() * iproj(patches, intrinsics[:,ix])


def flow_mag(poses: SE3, patches: Float[Tensor, "..."], intrinsics: Float[Tensor, "..."], ii: Int[Tensor, "n_edges"], jj: Int[Tensor, "n_edges"], kk: Int[Tensor, "n_edges"], beta: float = 0.3) -> Float[Tensor, "..."]:
    """ projective transform """

    coords0: Float[Tensor, "..."] = transform(poses, patches, intrinsics, ii, ii, kk)
    coords1: Float[Tensor, "..."] = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False)
    coords2: Float[Tensor, "..."] = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    flow1: Float[Tensor, "..."] = (coords1 - coords0).norm(dim=-1)
    flow2: Float[Tensor, "..."] = (coords2 - coords0).norm(dim=-1)

    return beta * flow1 + (1-beta) * flow2
