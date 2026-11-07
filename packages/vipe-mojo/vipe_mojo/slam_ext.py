"""`vipe_ext.slam_ext` in Mojo: the dense bundle adjustment and the frame distance MetricTrail's DROID SLAM calls.

`ba_extended` is `ba_cuda_impl` (`csrc/slam_ext/geom_kernels.cu`) line for line. The residual / Jacobian
kernel, the per-frame sums (`accum_kernel`), the Schur products (`EEt6x6`, `Ev6x1`, `EvT6x1`) and the pose
retraction are Mojo kernels with CUDA's summation order (no float atomics: the result does not depend
on scheduling). The pose system is assembled and solved in float64 like the extension's Eigen
`SimplicialLLT`, densely and with PyTorch's Cholesky (on the CPU for MPS tensors, which have no float64);
a factorisation failure gives a zero step, as in the extension. The remaining tensor algebra is PyTorch
on the input device, with the extension's expressions.

Every other slam_ext function raises NotImplementedError when called.
"""

import functools

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from vipe_mojo._launch import block_reduce, launch
from vipe_mojo._unported import module_getattr

__getattr__ = module_getattr("slam_ext")

HIJ: int = 78
"""The lower triangle of the 12 x 12 block of an edge (poses i and j), row-major: (n, m) for m <= n."""
BA_SUMS: int = HIJ + 4 * 6 + 3
"""Per-edge sums of the BA kernel: hij, vi, vj, fi, fj, Hff, vf, flow energy (mojo/slam.mojo)."""


def frame_distance(
    poses: Float32[Tensor, "n 7"],
    disps: Float32[Tensor, "nd h w"],
    intrinsics: Float32[Tensor, "q 4"],
    pi: Int64[Tensor, " m"],
    pj: Int64[Tensor, " m"],
    qi: Int64[Tensor, " m"],
    qj: Int64[Tensor, " m"],
    di: Int64[Tensor, " m"],
    beta: float,
) -> Float32[Tensor, " m"]:
    """Mean optical flow from frame pi to pj over disparity map di, blended with the translation-only flow by beta.

    1000 when less than 75 % of the (beta-weighted) points land in front of camera j (`frame_distance_cuda`).
    """
    m: int = pi.shape[0]
    partial: Tensor = torch.empty(m, 3, 256, dtype=torch.float32, device=poses.device)
    if m == 0:
        return torch.zeros(0, dtype=torch.float32, device=poses.device)
    launch("frame_distance", (float(beta),), (poses, disps, intrinsics, pi, pj, qi, qj, di, partial))
    sums: Tensor = block_reduce(partial)
    accum, total, valid = sums.unbind(1)
    return torch.where(valid / (total + 1e-8) < 0.75, torch.full_like(accum, 1000.0), accum / valid)


class _Accum:
    """`accum_cuda(data, ix, jx)`: out[j] = sum of data[e] over the edges e with ix[e] == jx[j], in edge order.

    The extension builds the edge lists from `argsort(ix)` on the CPU; CPU sorting is stable, so each list
    is in increasing edge order.
    """

    def __init__(self, ix: Tensor, jx: Tensor) -> None:
        ix_cpu: Tensor = ix.cpu()
        order: Tensor = torch.argsort(ix_cpu, stable=True)
        ordered: Tensor = ix_cpu[order]
        jx_cpu: Tensor = jx.cpu()
        device: torch.device = ix.device
        self.order: Tensor = order.to(device)
        self.starts: Tensor = torch.searchsorted(ordered, jx_cpu, right=False).to(device)
        self.ends: Tensor = torch.searchsorted(ordered, jx_cpu, right=True).to(device)

    def __call__(self, data: Tensor) -> Tensor:
        out: Tensor = torch.empty(self.starts.shape[0], data.shape[1], dtype=torch.float32, device=data.device)
        if out.numel():
            launch("accum", (), (data.contiguous(), self.order, self.starts, self.ends, out))
        return out


@functools.cache
def _triangle() -> tuple[Tensor, Tensor]:
    """Row and column of each hij entry."""
    rows, cols = zip(*[(n, m) for n in range(12) for m in range(n + 1)], strict=True)
    return torch.tensor(rows), torch.tensor(cols)


def _blocks(sums: Tensor) -> Tensor:
    """The four 6 x 6 blocks (ii-ii, ii-jj, jj-ii, jj-jj) of every edge, (4 * num, 6, 6) as `Hs.reshape(-1, 6, 6)`."""
    rows, cols = _triangle()
    h: Tensor = torch.zeros(sums.shape[0], 12, 12, dtype=sums.dtype, device=sums.device)
    h[:, rows.to(sums.device), cols.to(sums.device)] = sums[:, :HIJ]
    h[:, cols.to(sums.device), rows.to(sums.device)] = sums[:, :HIJ]
    return torch.cat([h[:, :6, :6], h[:, :6, 6:], h[:, 6:, :6], h[:, 6:, 6:]])


class _System:
    """The extension's `SparseBlock` / `SparseSystem` as a dense float64 matrix: 6 per pose, plus one focal scalar."""

    def __init__(self, num_poses: int, with_scalar: bool, device: torch.device) -> None:
        self.p: int = num_poses
        self.dim: int = 6 * num_poses + int(with_scalar)
        self.a: Tensor = torch.zeros(self.dim, self.dim, dtype=torch.float64, device=device)
        self.b: Tensor = torch.zeros(self.dim, dtype=torch.float64, device=device)

    def _valid(self, *index: Tensor) -> Tensor:
        ok: Tensor = torch.ones_like(index[0], dtype=torch.bool)
        for i in index:
            ok &= (i >= 0) & (i < self.p)
        return ok

    def add_pose_blocks(self, blocks: Tensor, i: Tensor, j: Tensor) -> None:
        dev: torch.device = self.a.device
        i, j, blocks = i.to(dev), j.to(dev), blocks.to(dev).double()
        ok: Tensor = self._valid(i, j)
        grid: Tensor = torch.zeros(self.p, self.p, 6, 6, dtype=torch.float64, device=dev)
        grid.index_put_((i[ok], j[ok]), blocks[ok], accumulate=True)
        self.a[: 6 * self.p, : 6 * self.p] += grid.permute(0, 2, 1, 3).reshape(6 * self.p, 6 * self.p)

    def add_pose_rhs(self, values: Tensor, i: Tensor) -> None:
        dev: torch.device = self.b.device
        i, values = i.to(dev), values.to(dev).double()
        ok: Tensor = self._valid(i)
        rhs: Tensor = torch.zeros(self.p, 6, dtype=torch.float64, device=dev)
        rhs.index_put_((i[ok],), values[ok], accumulate=True)
        self.b[: 6 * self.p] += rhs.reshape(-1)

    def add_pose_scalar(self, values: Tensor, i: Tensor) -> None:
        dev: torch.device = self.a.device
        i, values = i.to(dev), values.to(dev).double()
        ok: Tensor = self._valid(i)
        col: Tensor = torch.zeros(self.p, 6, dtype=torch.float64, device=dev)
        col.index_put_((i[ok],), values[ok], accumulate=True)
        self.a[: 6 * self.p, 6 * self.p] += col.reshape(-1)
        self.a[6 * self.p, : 6 * self.p] += col.reshape(-1)

    def add_scalar(self, lhs: float, rhs: float) -> None:
        self.a[6 * self.p, 6 * self.p] += lhs
        self.b[6 * self.p] += rhs

    def solve(self, minus: "_System | None", lm: float, ep: float, scalar_lm: float, scalar_ep: float) -> tuple[Tensor, float]:
        """(dx (P, 6) float32, df) of (A - S + damping) x = b - s; zeros when the Cholesky factorisation fails (Eigen's LLT)."""
        a: Tensor = self.a - minus.a if minus is not None else self.a.clone()
        b: Tensor = self.b - minus.b if minus is not None else self.b
        is_scalar: Tensor = torch.arange(self.dim, device=a.device) >= 6 * self.p
        lms: Tensor = torch.where(is_scalar, scalar_lm, lm).to(torch.float64)
        eps: Tensor = torch.where(is_scalar, scalar_ep, ep).to(torch.float64)
        diag: Tensor = torch.diagonal(a)
        diag += eps + lms * diag
        factor, info = torch.linalg.cholesky_ex(a)
        if int(info) != 0:
            return torch.zeros(self.p, 6, dtype=torch.float32), 0.0
        x: Tensor = torch.cholesky_solve(b.view(-1, 1), factor).view(-1)
        df: float = float(x[6 * self.p]) if self.dim > 6 * self.p else 0.0
        return x[: 6 * self.p].view(self.p, 6).float(), df


def _schur_pairs(jj_exp: Tensor, kk_exp: Tensor, t0: int, t1: int) -> tuple[Tensor, Tensor, Tensor]:
    """`schur_block`'s index lists: rows whose target frame is in [t0, t1), and every pair of them that sees one depth map.

    Returns (idx (pairs, 3) = (row 1, row 2, depth), pose of row 1, pose of row 2), on the CPU.
    """
    rows: Tensor = torch.nonzero((jj_exp >= t0) & (jj_exp < t1)).view(-1)
    depth: Tensor = kk_exp[rows]
    r1, r2 = torch.nonzero(depth.view(-1, 1) == depth.view(1, -1), as_tuple=True)
    idx: Tensor = torch.stack([rows[r1], rows[r2], depth[r1]], dim=1).contiguous()
    return idx, jj_exp[rows[r1]] - t0, jj_exp[rows[r2]] - t0


def ba_extended(
    poses: Float32[Tensor, "n 7"],
    disps: Float32[Tensor, "n h w"],
    intrinsics: Float32[Tensor, "4"],
    disps_sens: Float32[Tensor, "n h w"],
    targets: Float32[Tensor, "e 2 h w"],
    weights: Float32[Tensor, "e 2 h w"],
    eta: Float32[Tensor, "k h w"],
    ii: Int64[Tensor, " e"],
    jj: Int64[Tensor, " e"],
    depth_active: Float32[Tensor, " n"],
    t0: int,
    t1: int,
    iterations: int,
    lm: float,
    ep: float,
    motion_only: bool,
    alpha: float,
    optimize_intrinsics: bool,
    intrinsics_lm: float,
    intrinsics_ep: float,
    intrinsics_scale: float,
    compute_energy: bool,
    flow_weight: float,
) -> list[Tensor | None]:
    """`iterations` Gauss-Newton steps on poses [t0, t1), the disparities of the edges' source frames and optionally the focal.

    Updates `poses`, `disps` and (with optimize_intrinsics) `intrinsics` in place. Returns [dx, dz, energy] of the
    last iteration as the extension does (dz is None for motion_only, energy is empty unless compute_energy).
    """
    for name, t in (("poses", poses), ("disps", disps), ("intrinsics", intrinsics), ("targets", targets), ("weights", weights), ("eta", eta)):
        if not t.is_contiguous() or t.dtype != torch.float32:
            raise ValueError(f"ba_extended: {name} must be a contiguous float32 tensor")
    device: torch.device = poses.device
    solve_device: torch.device = torch.device("cpu") if device.type == "mps" else device
    f32: dict = {"dtype": torch.float32, "device": device}
    num: int = ii.shape[0]
    ht, wd = disps.shape[1], disps.shape[2]
    hw: int = ht * wd
    p: int = t1 - t0
    disps_sens = disps_sens.contiguous()
    depth_active = depth_active.contiguous()
    ii = ii.contiguous()
    jj = jj.contiguous()

    ts: Tensor = torch.arange(t0, t1, device=device)
    ii_exp: Tensor = torch.cat([ts, ii])
    jj_exp: Tensor = torch.cat([ts, jj])
    kx, kk_exp = torch.unique(ii_exp, sorted=True, return_inverse=True)
    accum_ii_kx = _Accum(ii, kx)
    accum_ii_ts = _Accum(ii, ts)
    accum_exp_kx = _Accum(ii_exp, kx)
    pair_idx, pair_i, pair_j = _schur_pairs(jj_exp.cpu(), kk_exp.cpu(), t0, t1)
    pair_idx = pair_idx.to(device)
    rhs_rows: Tensor = jj_exp - t0

    cii: Tensor = torch.zeros(num, hw, **f32)
    wi: Tensor = torch.zeros(num, hw, **f32)
    eii: Tensor = torch.zeros(num, 6, hw, **f32)
    eij: Tensor = torch.zeros(num, 6, hw, **f32)
    efi: Tensor = torch.zeros(num, hw, **f32)
    partial: Tensor = torch.empty(num, BA_SUMS, 256, **f32)
    energy: Tensor = torch.empty(iterations if compute_energy else 0, **f32)
    dx: Tensor | None = None
    dz: Tensor | None = None

    for itr in range(iterations):
        if num:
            launch(
                "ba_partial",
                (int(optimize_intrinsics), float(intrinsics_scale), float(flow_weight)),
                (targets, weights, poses, disps, intrinsics, depth_active, ii, jj, cii, wi, eii, eij, efi, partial),
            )
        sums: Tensor = block_reduce(partial)
        system = _System(p, optimize_intrinsics, solve_device)
        system.add_pose_blocks(_blocks(sums), torch.cat([ii, ii, jj, jj]) - t0, torch.cat([ii, jj, ii, jj]) - t0)
        system.add_pose_rhs(torch.cat([sums[:, HIJ : HIJ + 6], sums[:, HIJ + 6 : HIJ + 12]]), torch.cat([ii, jj]) - t0)
        if optimize_intrinsics:
            system.add_pose_scalar(torch.cat([sums[:, HIJ + 12 : HIJ + 18], sums[:, HIJ + 18 : HIJ + 24]]), torch.cat([ii, jj]) - t0)
            system.add_scalar(sums[:, HIJ + 24].sum().item(), sums[:, HIJ + 25].sum().item())

        active: Tensor = torch.empty(0, **f32)
        mask: Tensor = torch.empty(0, **f32)
        if not motion_only:
            active = depth_active[kx].to(torch.float32).view(-1, 1)
            mask = (disps_sens[kx] > 0).to(torch.float32).view(-1, hw) * active
        if compute_energy:
            iter_energy: Tensor = sums[:, HIJ + 26].sum()
            if not motion_only:
                iter_energy = iter_energy + (mask * alpha * torch.pow((disps[kx] - disps_sens[kx]).view(-1, hw), 2)).sum()
            energy[itr] = iter_energy

        if motion_only:
            dx_cpu, df = system.solve(None, lm, ep, intrinsics_lm, intrinsics_ep)
            dx = dx_cpu.to(device)
            launch("pose_retr", (t0,), (poses, dx))
            if optimize_intrinsics:
                intrinsics[:2] += torch.tensor(df * intrinsics_scale, dtype=torch.float32, device=device)
            continue

        c: Tensor = active * (accum_ii_kx(cii) + eta.view(-1, hw) + mask * alpha) + (1 - active)
        w: Tensor = active * (accum_ii_kx(wi) - mask * alpha * (disps[kx] - disps_sens[kx]).view(-1, hw))
        q: Tensor = (1.0 / c).contiguous()
        ei: Tensor = accum_ii_ts(eii.view(num, 6 * hw)).view(p, 6, hw)
        e: Tensor = (torch.cat([ei, eij], 0) * depth_active[ii_exp].to(torch.float32).view(-1, 1, 1)).contiguous()
        w = w.contiguous()

        schur = _System(p, optimize_intrinsics, solve_device)
        if pair_idx.shape[0]:
            spp: Tensor = torch.empty(pair_idx.shape[0], 36, 256, **f32)
            launch("eet", (), (e, q, pair_idx, spp))
            schur.add_pose_blocks(block_reduce(spp).view(-1, 6, 6), pair_i, pair_j)
        if e.shape[0]:
            ev: Tensor = torch.empty(e.shape[0], 6, 256, **f32)
            launch("ev", (), (e, q, w, kk_exp, ev))
            schur.add_pose_rhs(block_reduce(ev), rhs_rows)
        ef: Tensor = torch.empty(0, **f32)
        if optimize_intrinsics:
            ef = accum_ii_kx(efi) * active
            rows: Tensor = torch.nonzero((jj_exp >= t0) & (jj_exp < t1)).view(-1)
            spf: Tensor = (e[rows] * (q[kk_exp[rows]] * ef[kk_exp[rows]]).unsqueeze(1)).sum(2)
            schur.add_pose_scalar(spf, jj_exp[rows] - t0)
            schur.add_scalar((ef * q * ef).sum().item(), (ef * q * w).sum().item())

        dx_cpu, df = system.solve(schur, lm, ep, intrinsics_lm, intrinsics_ep)
        dx = dx_cpu.to(device)
        dw: Tensor = torch.zeros(e.shape[0], hw, **f32)
        launch("evt", (), (e, dx, rhs_rows.contiguous(), dw))
        dz = q * (w - accum_exp_kx(dw))
        if optimize_intrinsics:
            dz = dz - q * ef * df
        launch("pose_retr", (t0,), (poses, dx))
        if optimize_intrinsics:
            intrinsics[:2] += torch.tensor(df * intrinsics_scale, dtype=torch.float32, device=device)
        disps.index_put_((kx,), disps[kx] + dz.view(-1, ht, wd))
    return [dx, dz, energy]
