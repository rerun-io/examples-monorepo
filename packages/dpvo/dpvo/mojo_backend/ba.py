"""Bundle adjustment on the Mojo kernels: the `BAKernels` protocol of `dpvo.kernels`.

Same signatures as the CUDA extension `dpvo._cuda_ba` after dpvo/metric-core (four depth-prior
arguments after `lmbda`; an empty prior is plain DPVO). `forward` is the extension's Gauss-Newton
driver (`cuda_ba` in ba_cuda.cu) line for line: the residual/Hessian, depth-prior and retraction
kernels are Mojo; the Schur complement and the Cholesky solve are PyTorch on the input device
(on MPS the 6n x 6n solve runs on the CPU: PyTorch has no MPS Cholesky solve).
The block-E path (`eff_impl=True`, global BA) builds the extension's `EfficentE` index tables with
PyTorch ops on the input device; its three Schur products are Mojo kernels.
`neighbors` is a PyTorch port of the extension's CPU function. Not ported: `solve_system`
(classical loop closure).
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from dpvo.mojo_backend._launch import launch


@dataclass(frozen=True, slots=True)
class _BlockE:
    """The extension's `EfficentE`: E stored per (source frame, target frame) block and patch slot."""

    ij_xself: Int64[Tensor, "2 e"]
    """Per edge: the block of (ii, jj) and of (ii, ii)."""
    e_lookup: Float32[Tensor, "blocks ppf 6"]
    patch_to_ku: Int64[Tensor, "frames ppf"]
    """Unique-patch position of slot k of frame i, -1 for an empty slot."""
    index: Int64[Tensor, "pairs 5"]
    """(i, j1, j2, block(i, j1), block(i, j2)) for every pair of blocks with the same source frame i."""
    block_index: Int64[Tensor, "blocks 2"]
    """(i, j) of every block."""
    t0: int
    ppf: int

    @staticmethod
    def build(ii: Tensor, jj: Tensor, kx: Tensor, ppf: int, t0: int) -> "_BlockE":
        n_frames: int = int(torch.maximum(ii.max(), jj.max())) + 1
        ij_uniq, inverse = torch.unique(torch.cat([ii * n_frames + jj, ii * n_frames + ii]), sorted=True, return_inverse=True)
        patch_to_ku: Tensor = torch.full((n_frames, ppf), -1, dtype=torch.int64, device=ii.device)
        patch_to_ku[kx // ppf, kx % ppf] = torch.arange(kx.shape[0], device=ii.device)
        # Blocks sorted by (i, j): each source frame's blocks are contiguous; pair every two blocks of one frame.
        block_i: Tensor = ij_uniq // n_frames
        counts: Tensor = torch.unique_consecutive(block_i, return_counts=True)[1]
        starts: Tensor = torch.cumsum(counts, 0) - counts
        per_block: Tensor = counts.repeat_interleave(counts)  # blocks sharing each block's source frame
        first: Tensor = torch.arange(ij_uniq.shape[0], device=ii.device).repeat_interleave(per_block)
        offsets: Tensor = torch.cumsum(per_block, 0) - per_block
        second: Tensor = starts.repeat_interleave(counts).repeat_interleave(per_block) + (torch.arange(first.shape[0], device=ii.device) - offsets.repeat_interleave(per_block))
        block_j: Tensor = ij_uniq % n_frames
        return _BlockE(
            ij_xself=inverse.view(2, ii.shape[0]),
            e_lookup=torch.zeros(ij_uniq.shape[0], ppf, 6, dtype=torch.float32, device=ii.device),
            patch_to_ku=patch_to_ku,
            index=torch.stack([block_i[first], block_j[first], block_j[second], first, second], dim=1).contiguous(),
            block_index=torch.stack([block_i, block_j], dim=1).contiguous(),
            t0=t0,
            ppf=ppf,
        )

    def eqet(self, n: int, q: Tensor) -> Tensor:
        out: Tensor = torch.zeros(6 * n, 6 * n, dtype=torch.float32, device=q.device)
        launch("eet", (self.t0, self.ppf), (self.e_lookup, q.reshape(-1).contiguous(), self.index, self.patch_to_ku, out))
        return out

    def ev(self, n: int, vec: Tensor) -> Tensor:
        out: Tensor = torch.zeros(6 * n, dtype=torch.float32, device=vec.device)
        launch("ev", (self.t0, self.ppf), (self.e_lookup, vec.reshape(-1).contiguous(), self.block_index, self.patch_to_ku, out))
        return out.view(-1, 1)

    def etv(self, m: int, vec: Tensor) -> Tensor:
        out: Tensor = torch.zeros(m, dtype=torch.float32, device=vec.device)
        launch("etv", (self.t0, self.ppf), (self.e_lookup, vec.reshape(-1).contiguous(), self.block_index, self.patch_to_ku, out))
        return out.view(-1, 1)


def _solve(S: Tensor, y: Tensor) -> Tensor:
    """`cholesky_solve(y, cholesky_ex(S).L)`, as the extension; on MPS through the CPU."""
    if S.device.type == "mps":
        return _solve(S.cpu(), y.cpu()).to(S.device)
    L: Tensor = torch.linalg.cholesky_ex(S).L
    return torch.cholesky_solve(y, L)


def forward(
    poses: Float32[Tensor, "..."],
    patches: Float32[Tensor, "..."],
    intrinsics: Float32[Tensor, "..."],
    target: Float32[Tensor, "..."],
    weight: Float32[Tensor, "..."],
    lmbda: Float32[Tensor, "1"],
    depth_prior: Float32[Tensor, "..."],
    depth_prior_weight: Float32[Tensor, "..."],
    depth_prior_lambda: float,
    depth_prior_huber: float,
    ii: Int64[Tensor, " e"],
    jj: Int64[Tensor, " e"],
    kk: Int64[Tensor, " e"],
    M: int,
    t0: int,
    t1: int,
    iterations: int,
    eff_impl: bool,
) -> list[Tensor]:
    """`iterations` Gauss-Newton steps on poses [t0, t1) and the patches of kk; updates `poses` and `patches` in place."""
    kx, ku = torch.unique(kk, sorted=True, return_inverse=True)
    n: int = t1 - t0
    m: int = kx.shape[0]
    p: int = patches.shape[3]
    poses_v: Tensor = poses.view(-1, 7)
    patches_v: Tensor = patches.view(-1, 3, p, p)
    intrinsics_v: Tensor = intrinsics.view(-1, 4)
    target_v: Tensor = target.view(-1, 2)
    weight_v: Tensor = weight.view(-1, 2)
    prior: Tensor = depth_prior.view(-1)
    prior_weight: Tensor = depth_prior_weight.view(-1)
    if prior.shape[0] != prior_weight.shape[0]:
        raise RuntimeError("depth_prior/depth_prior_weight size mismatch")
    if prior.shape[0] not in (0, patches_v.shape[0]):
        raise RuntimeError("depth_prior must be empty or one per patch")
    if not depth_prior_lambda >= 0.0:
        raise RuntimeError("depth_prior_lambda must be >= 0")
    if not depth_prior_huber > 0.0:
        raise RuntimeError("depth_prior_huber must be > 0")

    f32: dict = {"dtype": torch.float32, "device": poses.device}
    B: Tensor = torch.empty(6 * n, 6 * n, **f32)
    E: Tensor = torch.empty(6 * n, 0 if eff_impl else m, **f32)
    C: Tensor = torch.empty(m, **f32)
    v: Tensor = torch.empty(6 * n, **f32)
    u: Tensor = torch.empty(m, **f32)
    block_e: _BlockE | None = _BlockE.build(ii, jj, kx, M, t0) if eff_impl else None
    ij_xself: Tensor = block_e.ij_xself if block_e is not None else torch.empty(2, 0, dtype=torch.int64, device=poses.device)
    e_lookup: Tensor = block_e.e_lookup if block_e is not None else torch.empty(0, 0, 6, **f32)
    for _ in range(iterations):
        for t in (B, E, C, v, u, e_lookup):
            t.zero_()
        launch(
            "hessian",
            (t0, M if eff_impl else 0),
            (poses_v, patches_v, intrinsics_v, target_v, weight_v, ij_xself, ii, jj, kk, ku, B, E, e_lookup, C, v, u),
        )
        if depth_prior_lambda > 0.0 and prior.shape[0] > 0 and m > 0:
            launch("depth_prior", (float(depth_prior_lambda), float(depth_prior_huber)), (patches_v, prior, prior_weight, kx, C, u))

        Q: Tensor = 1.0 / (C + lmbda).view(1, m)
        Qt: Tensor = Q.t()
        uc: Tensor = u.view(m, 1)
        if n == 0:
            dZ: Tensor = (Qt * uc).view(m)
            launch("patch_retr", (), (kx, patches_v, dZ.contiguous()))
            continue
        if block_e is not None:
            S: Tensor = B - block_e.eqet(n, Q)
            y: Tensor = v.view(6 * n, 1) - block_e.ev(n, Qt * uc)
            S += torch.eye(6 * n, **f32) * (1e-4 * S + 1.0)
            dX: Tensor = _solve(S, y)
            dZ = Qt * (uc - block_e.etv(m, dX))
        else:
            EQ: Tensor = E * Q
            Et: Tensor = E.t()
            S = B - torch.matmul(EQ, Et)
            y = v.view(6 * n, 1) - torch.matmul(EQ, uc)
            S += torch.eye(6 * n, **f32) * (1e-4 * S + 1.0)
            dX = _solve(S, y)
            dZ = Qt * (uc - torch.matmul(Et, dX))
        launch("pose_retr", (t0,), (poses_v, dX.view(n, 6).contiguous()))
        launch("patch_retr", (), (kx, patches_v, dZ.view(m).contiguous()))
    return []


def reproject(
    poses: Float32[Tensor, "..."],
    patches: Float32[Tensor, "..."],
    intrinsics: Float32[Tensor, "..."],
    ii: Int64[Tensor, " e"],
    jj: Int64[Tensor, " e"],
    kk: Int64[Tensor, " e"],
) -> Float32[Tensor, "1 e 2 p p"]:
    """Pixel coordinates of patch kk[e] (from frame ii[e]) in frame jj[e], (1, E, 2, P, P)."""
    p: int = patches.shape[-1]
    coords: Tensor = torch.empty(ii.shape[0], 2, p, p, dtype=torch.float32, device=poses.device)
    if ii.shape[0] > 0:
        launch("reproject", (), (poses.view(-1, 7), patches.view(-1, 3, p, p), intrinsics.view(-1, 4), ii, jj, kk, coords))
    return coords.view(1, ii.shape[0], 2, p, p)


def neighbors(ii: Int64[Tensor, " e"], jj: Int64[Tensor, " e"]) -> list[Int64[Tensor, " e"]]:
    """For each edge, the previous and next edge of the same patch ii ordered by target frame jj (-1 at the ends).

    The extension's CPU function stable-sorts each patch's edges by jj; a stable sort by jj and then a stable
    sort by ii gives the same order (ties keep the edge order).
    """
    by_j: Tensor = torch.sort(jj, stable=True).indices
    order: Tensor = by_j[torch.sort(ii[by_j], stable=True).indices]
    same: Tensor = ii[order[1:]] == ii[order[:-1]]
    ix: Tensor = torch.full_like(ii, -1)
    jx: Tensor = torch.full_like(ii, -1)
    ix[order[1:]] = torch.where(same, order[:-1], -1)
    jx[order[:-1]] = torch.where(same, order[1:], -1)
    return [ix, jx]


def solve_system(J_Ginv_i: Tensor, J_Ginv_j: Tensor, ii: Tensor, jj: Tensor, res: Tensor, ep: float, lm: float, freen: int) -> list[Tensor]:
    """Classical loop closure's sparse pose-graph solve (Eigen in the extension): not ported."""
    raise NotImplementedError("dpvo.mojo_backend: solve_system (classical loop closure) is CUDA-only")
