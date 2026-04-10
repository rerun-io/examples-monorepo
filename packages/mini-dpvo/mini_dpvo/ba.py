import torch
from jaxtyping import Bool, Float, Int
from lietorch import SE3
from torch import Tensor

from . import projective_ops as pops
from .scatter_utils import scatter_sum


class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, H: Float[Tensor, "batch n n"], b: Float[Tensor, "batch n m"]) -> Float[Tensor, "batch n m"]:
        # don't crash training if cholesky decomp fails
        U: Float[Tensor, "batch n n"]
        info: Int[Tensor, "batch"]
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs: Float[Tensor, "batch n m"] = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "batch n m"]) -> tuple[Float[Tensor, "batch n n"] | None, Float[Tensor, "batch n m"] | None]:
        if ctx.failed:
            return None, None

        U: Float[Tensor, "batch n n"]
        xs: Float[Tensor, "batch n m"]
        U, xs = ctx.saved_tensors
        dz: Float[Tensor, "batch n m"] = torch.cholesky_solve(grad_x, U)
        dH: Float[Tensor, "batch n n"] = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

# utility functions for scattering ops
def safe_scatter_add_mat(
    A: Float[Tensor, "batch edges *rest"],
    ii: Int[Tensor, "edges"],
    jj: Int[Tensor, "edges"],
    n: int,
    m: int,
) -> Float[Tensor, "batch n_times_m *rest"]:
    v: Bool[Tensor, "edges"] = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(
    b: Float[Tensor, "batch edges *rest"],
    ii: Int[Tensor, "edges"],
    n: int,
) -> Float[Tensor, "batch n *rest"]:
    v: Bool[Tensor, "edges"] = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(
    disps: Float[Tensor, "batch n_patches ps ps"],
    dz: Float[Tensor, "batch edges 1 1"],
    ii: Int[Tensor, "edges"],
) -> Float[Tensor, "batch n_patches ps ps"]:
    ii: Int[Tensor, "edges"] = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(
    poses: SE3,
    dx: Float[Tensor, "batch edges 6"],
    ii: Int[Tensor, "edges"],
) -> SE3:
    ii: Int[Tensor, "edges"] = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def block_matmul(
    A: Float[Tensor, "b n1 m1 p1 q1"],
    B: Float[Tensor, "b n2 m2 p2 q2"],
) -> Float[Tensor, "b n1 m2 p1 q2"]:
    """ block matrix multiply """
    b: int
    n1: int
    m1: int
    p1: int
    q1: int
    b, n1, m1, p1, q1 = A.shape
    _b: int
    n2: int
    m2: int
    p2: int
    q2: int
    _b, n2, m2, p2, q2 = B.shape
    A: Float[Tensor, "b n1_p1 m1_q1"] = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B: Float[Tensor, "b n2_p2 m2_q2"] = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

def block_solve(
    A: Float[Tensor, "b n1 m1 p1 q1"],
    B: Float[Tensor, "b n2 m2 p2 q2"],
    ep: float = 1.0,
    lm: float = 1e-4,
) -> Float[Tensor, "b n1 m2 p1 q2"]:
    """ block matrix solve """
    b: int
    n1: int
    m1: int
    p1: int
    q1: int
    b, n1, m1, p1, q1 = A.shape
    _b: int
    n2: int
    m2: int
    p2: int
    q2: int
    _b, n2, m2, p2, q2 = B.shape
    A: Float[Tensor, "b n1_p1 m1_q1"] = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B: Float[Tensor, "b n2_p2 m2_q2"] = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)

    A: Float[Tensor, "b n1_p1 m1_q1"] = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)

    X: Float[Tensor, "b n1_p1 m2_q2"] = CholeskySolver.apply(A, B)
    return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A: Float[Tensor, "b n1 m1 p1 q1"]) -> None:
    import matplotlib.pyplot as plt
    b: int
    n1: int
    m1: int
    p1: int
    q1: int
    b, n1, m1, p1, q1 = A.shape
    A: Float[Tensor, "b n1_p1 m1_q1"] = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()

def BA(
    poses: SE3,
    patches: Float[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float[Tensor, "1 n_frames 4"],
    targets: Float[Tensor, "1 n_edges 2"],
    weights: Float[Tensor, "1 n_edges 2"],
    lmbda: float | Float[Tensor, "..."],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    bounds: tuple[float, float, float, float],
    ep: float = 100.0,
    PRINT: bool = False,
    fixedp: int = 1,
    structure_only: bool = False,
) -> tuple[SE3, Float[Tensor, "1 n_patches 3 ps ps"]]:
    """ bundle adjustment """

    b: int = 1
    n: int = max(ii.max().item(), jj.max().item()) + 1

    coords: Float[Tensor, "1 n_edges ps ps 2"]
    v: Float[Tensor, "1 n_edges ps ps"]
    Ji: Float[Tensor, "1 n_edges 2 6"]
    Jj: Float[Tensor, "1 n_edges 2 6"]
    Jz: Float[Tensor, "1 n_edges 2 1"]
    coords, v, (Ji, Jj, Jz) = \
        pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    p: int = coords.shape[3]
    r: Float[Tensor, "1 n_edges 2"] = targets - coords[...,p//2,p//2,:]

    v: Float[Tensor, "1 n_edges"] = v * (r.norm(dim=-1) < 250).float()

    in_bounds: Bool[Tensor, "1 n_edges"] = \
        (coords[...,p//2,p//2,0] > bounds[0]) & \
        (coords[...,p//2,p//2,1] > bounds[1]) & \
        (coords[...,p//2,p//2,0] < bounds[2]) & \
        (coords[...,p//2,p//2,1] < bounds[3])

    v: Float[Tensor, "1 n_edges"] = v * in_bounds.float()

    if PRINT:
        print((r * v[...,None]).norm(dim=-1).mean().item())

    r: Float[Tensor, "1 n_edges 2 1"] = (v[...,None] * r).unsqueeze(dim=-1)
    weights: Float[Tensor, "1 n_edges 2 1"] = (v[...,None] * weights).unsqueeze(dim=-1)

    wJiT: Float[Tensor, "1 n_edges 6 2"] = (weights * Ji).transpose(2,3)
    wJjT: Float[Tensor, "1 n_edges 6 2"] = (weights * Jj).transpose(2,3)
    wJzT: Float[Tensor, "1 n_edges 1 2"] = (weights * Jz).transpose(2,3)

    Bii: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJiT, Ji)
    Bij: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJiT, Jj)
    Bji: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJjT, Ji)
    Bjj: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJjT, Jj)

    Eik: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJiT, Jz)
    Ejk: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJjT, Jz)

    vi: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJiT, r)
    vj: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJjT, r)

    # fix first pose
    ii: Int[Tensor, "n_edges"] = ii.clone()
    jj: Int[Tensor, "n_edges"] = jj.clone()

    n: int = n - fixedp
    ii: Int[Tensor, "n_edges"] = ii - fixedp
    jj: Int[Tensor, "n_edges"] = jj - fixedp

    kx: Int[Tensor, "m"]
    kk: Int[Tensor, "n_edges"]
    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m: int = len(kx)

    B: Float[Tensor, "1 n n 6 6"] = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)

    E: Float[Tensor, "1 n m 6 1"] = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1)

    C: Float[Tensor, "1 m 1 1"] = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)

    v: Float[Tensor, "1 n 1 6 1"] = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)

    w: Float[Tensor, "1 m 1 1"] = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk, m)

    if isinstance(lmbda, torch.Tensor):
        lmbda: Float[Tensor, "..."] = lmbda.reshape(*C.shape)

    Q: Float[Tensor, "1 m 1 1"] = 1.0 / (C + lmbda)

    ### solve w/ schur complement ###
    EQ: Float[Tensor, "1 n m 6 1"] = E * Q[:,None]

    if structure_only or n == 0:
        dZ: Float[Tensor, "1 m 1 1"] = (Q * w).view(b, -1, 1, 1)

    else:
        S: Float[Tensor, "1 n n 6 6"] = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y: Float[Tensor, "1 n 1 6 1"] = v - block_matmul(EQ, w.unsqueeze(dim=2))
        dX: Float[Tensor, "1 n 1 6 1"] = block_solve(S, y, ep=ep, lm=1e-4)

        dZ: Float[Tensor, "1 m 1 1"] = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        dX: Float[Tensor, "1 n_poses 6"] = dX.view(b, -1, 6)
        dZ: Float[Tensor, "1 m 1 1"] = dZ.view(b, -1, 1, 1)

    x: Float[Tensor, "1 n_patches ps ps"]
    y_coord: Float[Tensor, "1 n_patches ps ps"]
    disps: Float[Tensor, "1 n_patches ps ps"]
    x, y_coord, disps = patches.unbind(dim=2)
    disps: Float[Tensor, "1 n_patches ps ps"] = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches: Float[Tensor, "1 n_patches 3 ps ps"] = torch.stack([x, y_coord, disps], dim=2)

    if not structure_only and n > 0:
        poses: SE3 = pose_retr(poses, dX, fixedp + torch.arange(n))

    return poses, patches
