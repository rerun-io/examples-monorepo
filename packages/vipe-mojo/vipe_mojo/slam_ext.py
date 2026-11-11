"""`vipe_ext.slam_ext` in Mojo: the dense bundle adjustment and the frame distance MetricTrail's DROID SLAM calls.

`ba_extended` is `ba_cuda_impl` (`csrc/slam_ext/geom_kernels.cu`) line for line. The residual / Jacobian
kernel, the per-frame sums (`accum_kernel`), the Schur products (`EEt6x6`, `Ev6x1`, `EvT6x1`) and the pose
retraction are Mojo kernels with CUDA's summation order (no float atomics: the result does not depend
on scheduling). As in the extension, the reduced sums go to the CPU once per iteration, where the pose
system is assembled and solved in float64 (Eigen's `SimplicialLLT` there, a single-threaded Mojo profile Cholesky
here; integer scatter tables are reused for identical edge topologies). Prepared calibration callers can instead
assemble and solve the same float64 system on the GPU with cuSOLVER LLT, allowing whole-step CUDA capture.
A factorisation failure gives a zero step on both paths. The
remaining tensor algebra is PyTorch on the input device, with the extension's expressions.

Every other slam_ext function raises NotImplementedError when called.
"""

from dataclasses import dataclass
from functools import lru_cache

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
    out: Tensor = torch.empty(pi.shape[0], dtype=torch.float32, device=poses.device)
    if out.numel():
        launch("frame_distance", (float(beta),), (poses, disps, intrinsics, pi, pj, qi, qj, di, out))
    return out


class _Accum:
    """`accum_cuda(data, ix, jx)`: out[j] = sum of data[e] over the edges e with ix[e] == jx[j], in edge order.

    The extension builds the edge lists from `argsort(ix)` on the CPU; CPU sorting is stable, so each list
    is in increasing edge order.
    """

    def __init__(self, ix_cpu: Tensor, jx_cpu: Tensor, device: torch.device) -> None:
        order: Tensor = torch.argsort(ix_cpu, stable=True)
        ordered: Tensor = ix_cpu[order]
        self.order: Tensor = order.to(device)
        self.starts: Tensor = torch.searchsorted(ordered, jx_cpu, right=False).to(device)
        self.ends: Tensor = torch.searchsorted(ordered, jx_cpu, right=True).to(device)

    def __call__(self, data: Tensor) -> Tensor:
        out: Tensor = torch.empty(self.starts.shape[0], data.shape[1], dtype=torch.float32, device=data.device)
        if out.numel():
            launch("accum", (), (data.contiguous(), self.order, self.starts, self.ends, out))
        return out


def _valid(pose: Tensor, num_poses: int) -> Tensor:
    return (pose >= 0) & (pose < num_poses)


class _PoseSystem:
    """The pose step of one BA iteration: immutable scatter tables from the reduced sums into (A - S, b - s),
    and the float64 solve on the CPU on one thread (`pose_solve`, mojo/slam.mojo), as the extension's Eigen step.

    Per iteration the GPU sums arrive in one float32 buffer: the BA kernel's per-edge sums (num, BA_SUMS); unless
    motion_only, the Schur blocks (pairs, 36), the Schur rhs (rows, 6) and, with the focal, its Schur column
    (target rows, 6) and the two Schur focal scalars. Poses outside [t0, t1) are dropped, as in the extension.
    """

    def __init__(self, p: int, with_scalar: bool, ii: Tensor, jj: Tensor, t0: int, schur: tuple[Tensor, Tensor, Tensor, Tensor] | None) -> None:
        self.p: int = p
        self.dim: int = 6 * p + int(with_scalar)
        dim, focal = self.dim, 6 * p
        a_parts: list[tuple[Tensor, Tensor, float]] = []
        b_parts: list[tuple[Tensor, Tensor, float]] = []
        num: int = ii.shape[0]
        edges: Tensor = torch.arange(num).view(-1, 1) * BA_SUMS
        r6: Tensor = torch.arange(6)

        # hij: the lower triangle of each edge's 12 x 12 block (poses ii, jj), both halves of the symmetric matrix.
        r12: Tensor = torch.arange(12)
        rows, cols = r12.view(-1, 1).expand(12, 12).reshape(-1), r12.view(1, -1).expand(12, 12).reshape(-1)
        tri: Tensor = torch.where(rows >= cols, rows * (rows + 1) // 2 + cols, cols * (cols + 1) // 2 + rows)
        pose_r: Tensor = torch.where(rows < 6, ii.view(-1, 1), jj.view(-1, 1)) - t0
        pose_c: Tensor = torch.where(cols < 6, ii.view(-1, 1), jj.view(-1, 1)) - t0
        keep: Tensor = _valid(pose_r, p) & _valid(pose_c, p)
        a_parts.append(((edges + tri)[keep], ((6 * pose_r + rows % 6) * dim + 6 * pose_c + cols % 6)[keep], 1.0))
        # vi, vj and, with the focal, fi, fj (its column and row) and the edge sums hf, vfocal.
        side_pose: Tensor = torch.stack([ii, jj], 1) - t0  # (num, 2)
        side_rows: Tensor = (6 * side_pose.view(num, 2, 1) + r6).view(num, 12)
        side_keep: Tensor = _valid(side_pose, p).repeat_interleave(6, 1)
        b_parts.append(((edges + HIJ + torch.arange(12))[side_keep], side_rows[side_keep], 1.0))
        if with_scalar:
            src: Tensor = (edges + HIJ + 12 + torch.arange(12))[side_keep]
            a_parts += [(src, side_rows[side_keep] * dim + focal, 1.0), (src, focal * dim + side_rows[side_keep], 1.0)]
            a_parts.append((edges.view(-1) + HIJ + 24, torch.full((num,), focal * dim + focal), 1.0))
            b_parts.append((edges.view(-1) + HIJ + 25, torch.full((num,), focal), 1.0))

        at: int = num * BA_SUMS
        if schur is not None:
            pair_i, pair_j, exp_pose, target_pose = (x - t0 for x in schur)
            pairs: int = pair_i.shape[0]
            pair_keep: Tensor = _valid(pair_i, p) & _valid(pair_j, p)
            blk: Tensor = (at + 36 * torch.arange(pairs).view(-1, 1, 1) + 6 * r6.view(1, 6, 1) + r6.view(1, 1, 6))[pair_keep]
            dst: Tensor = ((6 * pair_i.view(-1, 1, 1) + r6.view(1, 6, 1)) * dim + 6 * pair_j.view(-1, 1, 1) + r6.view(1, 1, 6))[pair_keep]
            a_parts.append((blk.reshape(-1), dst.reshape(-1), -1.0))
            at += 36 * pairs
            rows_e: int = exp_pose.shape[0]
            exp_keep: Tensor = _valid(exp_pose, p)
            b_parts.append(
                ((at + 6 * torch.arange(rows_e).view(-1, 1) + r6)[exp_keep].reshape(-1), (6 * exp_pose.view(-1, 1) + r6)[exp_keep].reshape(-1), -1.0)
            )
            at += 6 * rows_e
            if with_scalar:
                rows_t: int = target_pose.shape[0]
                t_keep: Tensor = _valid(target_pose, p)
                src = (at + 6 * torch.arange(rows_t).view(-1, 1) + r6)[t_keep].reshape(-1)
                col: Tensor = (6 * target_pose.view(-1, 1) + r6)[t_keep].reshape(-1)
                a_parts += [(src, col * dim + focal, -1.0), (src, focal * dim + col, -1.0)]
                at += 6 * rows_t
                a_parts.append((torch.tensor([at]), torch.tensor([focal * dim + focal]), -1.0))
                b_parts.append((torch.tensor([at + 1]), torch.tensor([focal]), -1.0))
                at += 2
        self.host_size: int = at

        def table(parts: list[tuple[Tensor, Tensor, float]]) -> tuple[Tensor, Tensor, Tensor]:
            src = torch.cat([s.reshape(-1) for s, _, _ in parts]).contiguous()
            dst = torch.cat([d.reshape(-1) for _, d, _ in parts]).contiguous()
            sign = torch.cat([torch.full((s.numel(),), g, dtype=torch.float32) for s, _, g in parts])
            return src, dst, sign

        self.a_table: tuple[Tensor, Tensor, Tensor] = table(a_parts)
        self.b_table: tuple[Tensor, Tensor, Tensor] = table(b_parts)
        self.device_tables: tuple[Tensor, ...] | None = None

    def prepare_device(self, device: torch.device) -> None:
        """Transfer only integer assembly metadata before CUDA capture."""
        if self.device_tables is not None and self.device_tables[0].device == device:
            return
        self.device_tables = tuple(value.to(device) for value in (*self.a_table, *self.b_table))

    def solve_device(self, reduced: Tensor, lm: float, ep: float, scalar_lm: float, scalar_ep: float) -> tuple[Tensor, Tensor]:
        """The same float64 normal equations and LLT, without a device-to-host copy.

        Failure remains a zero step. The status stays on the device, including during graph replay.
        Assembly sums float32 inputs in float64, as the CPU implementation does.
        """
        assert self.device_tables is not None
        a_src, a_dst, a_sign, b_src, b_dst, b_sign = self.device_tables
        a = torch.zeros(self.dim * self.dim, dtype=torch.float64, device=reduced.device)
        b = torch.zeros(self.dim, dtype=torch.float64, device=reduced.device)
        a.index_add_(0, a_dst, reduced[a_src].double() * a_sign)
        b.index_add_(0, b_dst, reduced[b_src].double() * b_sign)
        matrix = a.view(self.dim, self.dim)
        diagonal = matrix.diagonal()
        diagonal[:6 * self.p] += ep + lm * diagonal[:6 * self.p]
        if self.dim > 6 * self.p:
            diagonal[-1:] += scalar_ep + scalar_lm * diagonal[-1:]
        chol, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
        y = torch.linalg.solve_triangular(chol, b[:, None], upper=False)
        x = torch.linalg.solve_triangular(chol.T, y, upper=True).view(-1)
        x = torch.where(info == 0, x, 0.0)
        df = x[-1] if self.dim > 6 * self.p else torch.zeros((), dtype=torch.float64, device=reduced.device)
        return x[:6 * self.p].view(self.p, 6).float(), df

    def solve_on(
        self, reduced: Tensor, lm: float, ep: float, scalar_lm: float, scalar_ep: float, *, device_solve: bool
    ) -> tuple[Tensor, Tensor | float]:
        """(dx on ``reduced``'s device, df): :meth:`solve_device`, or :meth:`solve` on the CPU and dx copied back."""
        if device_solve:
            return self.solve_device(reduced, lm, ep, scalar_lm, scalar_ep)
        dx, df = self.solve(reduced.cpu(), lm, ep, scalar_lm, scalar_ep)
        return dx.to(reduced.device), df

    def solve(self, host: Tensor, lm: float, ep: float, scalar_lm: float, scalar_ep: float) -> tuple[Tensor, float]:
        """(dx (P, 6) float32, df) on the CPU; zeros when the Cholesky factorisation fails (Eigen's LLT)."""
        assert host.shape[0] == self.host_size, (host.shape, self.host_size)
        # Cached tables are immutable; each solve owns its scratch, including concurrent callers.
        a: Tensor = torch.empty(self.dim * self.dim, dtype=torch.float64)
        x: Tensor = torch.empty(self.dim, dtype=torch.float64)
        ok = launch(
            "pose_solve", (self.p, float(lm), float(ep), float(scalar_lm), float(scalar_ep)), (host, *self.a_table, *self.b_table, a, x)
        )
        if not ok:
            return torch.zeros(self.p, 6, dtype=torch.float32), 0.0
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


@dataclass(frozen=True, slots=True)
class BaTopology:
    """Index-only preparation shared by solves with identical ordered edges and pose ranges."""

    kx: Int64[Tensor, "k"]
    """Sorted source frames (including the optimized pose range)."""
    kk_exp: Int64[Tensor, "expanded"]
    """Expanded edge to source-group index."""
    ii_exp: Int64[Tensor, "expanded"]
    """Optimized frame range followed by edge sources."""
    accum_ii_kx: _Accum
    """Edge-to-source accumulation, in original edge order."""
    accum_ii_ts: _Accum
    """Edge-to-optimized-pose accumulation."""
    accum_exp_kx: _Accum
    """Expanded-edge-to-source accumulation."""
    pair_idx: Int64[Tensor, "pairs 3"]
    """Schur row pairs and their source group."""
    rhs_rows: Int64[Tensor, "expanded"]
    """Expanded target indices relative to the first optimized pose."""
    target_rows: Int64[Tensor, "targets"]
    """Expanded rows whose target pose is optimized."""
    pose_system: _PoseSystem
    """Immutable CPU assembly tables; each solve allocates its own scratch."""


@lru_cache(maxsize=8)
def prepare_topology(
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    t0: int,
    t1: int,
    optimize_intrinsics: bool,
    motion_only: bool,
    device: torch.device,
) -> BaTopology:
    """Build integer metadata on the CPU once, then transfer only the kernel index tables.

    Eight entries bound retained device memory. Values, order, pose bounds, solver mode and
    device all participate in the key. No poses, depths, focal estimates or residuals are cached.
    """
    ii: Int64[Tensor, "e"] = torch.tensor(sources, dtype=torch.int64)
    jj: Int64[Tensor, "e"] = torch.tensor(targets, dtype=torch.int64)
    ts: Int64[Tensor, "poses"] = torch.arange(t0, t1)
    ii_exp: Int64[Tensor, "expanded"] = torch.cat((ts, ii))
    jj_exp: Int64[Tensor, "expanded"] = torch.cat((ts, jj))
    kx, kk_exp = torch.unique(ii_exp, sorted=True, return_inverse=True)
    pair_idx, pair_i, pair_j = _schur_pairs(jj_exp, kk_exp, t0, t1)
    target_rows: Int64[Tensor, "targets"] = torch.nonzero((jj_exp >= t0) & (jj_exp < t1)).view(-1)
    pose_system: _PoseSystem = _PoseSystem(
        t1 - t0, optimize_intrinsics, ii, jj, t0,
        None if motion_only else (pair_i + t0, pair_j + t0, jj_exp, jj_exp[target_rows]),
    )
    return BaTopology(
        kx.to(device), kk_exp.to(device), ii_exp.to(device),
        _Accum(ii, kx, device), _Accum(ii, ts, device), _Accum(ii_exp, kx, device),
        pair_idx.to(device), (jj_exp - t0).contiguous().to(device), target_rows.to(device), pose_system,
    )


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
    *,
    topology: BaTopology | None = None,
    device_solve: bool = False,
) -> list[Tensor | None]:
    """`iterations` Gauss-Newton steps on poses [t0, t1), the disparities of the edges' source frames and optionally the focal.

    Updates `poses`, `disps` and (with optimize_intrinsics) `intrinsics` in place. Returns [dx, dz, energy] of the
    last iteration as the extension does (dz is None for motion_only, energy is empty unless compute_energy).
    """
    for name, t in (("poses", poses), ("disps", disps), ("intrinsics", intrinsics), ("targets", targets), ("weights", weights), ("eta", eta)):
        if not t.is_contiguous() or t.dtype != torch.float32:
            raise ValueError(f"ba_extended: {name} must be a contiguous float32 tensor")
    device: torch.device = poses.device
    f32: dict = {"dtype": torch.float32, "device": device}
    num: int = ii.shape[0]
    ht, wd = disps.shape[1], disps.shape[2]
    hw: int = ht * wd
    p: int = t1 - t0
    disps_sens = disps_sens.contiguous()
    depth_active = depth_active.contiguous()
    ii = ii.contiguous()
    jj = jj.contiguous()

    # One readback of the edge values identifies the exact topology. Tensor identity is not
    # sufficient: upstream reuses tensors and also constructs fresh tensors for unchanged edges.
    if topology is None:
        edges: list[list[int]] = torch.stack((ii, jj)).cpu().tolist()
        topology = prepare_topology(tuple(edges[0]), tuple(edges[1]), t0, t1, optimize_intrinsics, motion_only, device)
    kx, kk_exp, ii_exp = topology.kx, topology.kk_exp, topology.ii_exp
    accum_ii_kx, accum_ii_ts, accum_exp_kx = topology.accum_ii_kx, topology.accum_ii_ts, topology.accum_exp_kx
    pair_idx, rhs_rows, target_rows = topology.pair_idx, topology.rhs_rows, topology.target_rows
    num_pairs: int = pair_idx.shape[0]
    pose_system: _PoseSystem = topology.pose_system
    # Constant over the iterations.
    active: Tensor = torch.empty(0, **f32)
    mask: Tensor = torch.empty(0, **f32)
    sens: Tensor = torch.empty(0, **f32)
    if not motion_only:
        active = depth_active[kx].to(torch.float32).view(-1, 1)
        mask = (disps_sens[kx] > 0).to(torch.float32).view(-1, hw) * active
        sens = disps_sens[kx].view(-1, hw)
    depth_exp: Tensor = depth_active[ii_exp].to(torch.float32).view(-1, 1, 1)
    eta_v: Tensor = eta.view(-1, hw)

    cii: Tensor = torch.zeros(num, hw, **f32)
    wi: Tensor = torch.zeros(num, hw, **f32)
    eii: Tensor = torch.zeros(num, 6, hw, **f32)
    eij: Tensor = torch.zeros(num, 6, hw, **f32)
    efi: Tensor = torch.zeros(num, hw, **f32)
    partial: Tensor = torch.empty(num, BA_SUMS, 256, **f32)
    spp: Tensor = torch.empty(num_pairs, 36, 256, **f32)
    ev: Tensor = torch.empty(ii_exp.shape[0], 6, 256, **f32)
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
        sums_gpu: Tensor = block_reduce(partial)
        if compute_energy:
            iter_energy: Tensor = sums_gpu[:, HIJ + 26].sum()
            if not motion_only:
                iter_energy = iter_energy + (mask * alpha * torch.pow(disps[kx].view(-1, hw) - sens, 2)).sum()
            energy[itr] = iter_energy

        if motion_only:
            dx, df = pose_system.solve_on(sums_gpu.view(-1), lm, ep, intrinsics_lm, intrinsics_ep, device_solve=device_solve)
            launch("pose_retr", (t0,), (poses, dx))
            if optimize_intrinsics:
                intrinsics[:2] += df * intrinsics_scale
            continue

        c: Tensor = active * (accum_ii_kx(cii) + eta_v + mask * alpha) + (1 - active)
        w: Tensor = (active * (accum_ii_kx(wi) - mask * alpha * (disps[kx].view(-1, hw) - sens))).contiguous()
        q: Tensor = (1.0 / c).contiguous()
        ei: Tensor = accum_ii_ts(eii.view(num, 6 * hw)).view(p, 6, hw)
        e: Tensor = (torch.cat([ei, eij], 0) * depth_exp).contiguous()

        # Everything the pose system needs, reduced on the GPU and copied to the CPU at once.
        parts: list[Tensor] = [sums_gpu.view(-1)]
        if num_pairs:
            launch("eet", (), (e, q, pair_idx, spp))
            parts.append(block_reduce(spp).view(-1))
        if e.shape[0]:
            launch("ev", (), (e, q, w, kk_exp, ev))
            parts.append(block_reduce(ev).view(-1))
        ef: Tensor = torch.empty(0, **f32)
        if optimize_intrinsics:
            ef = accum_ii_kx(efi) * active
            spf: Tensor = (e[target_rows] * (q[kk_exp[target_rows]] * ef[kk_exp[target_rows]]).unsqueeze(1)).sum(2)
            parts += [spf.view(-1), (ef * q * ef).sum().view(1), (ef * q * w).sum().view(1)]
        dx, df = pose_system.solve_on(torch.cat(parts), lm, ep, intrinsics_lm, intrinsics_ep, device_solve=device_solve)
        dw: Tensor = torch.zeros(e.shape[0], hw, **f32)
        launch("evt", (), (e, dx, rhs_rows, dw))
        dz = q * (w - accum_exp_kx(dw))
        if optimize_intrinsics:
            dz = dz - q * ef * df
        launch("pose_retr", (t0,), (poses, dx))
        if optimize_intrinsics:
            intrinsics[:2] += df * intrinsics_scale
        disps.index_put_((kx,), disps[kx] + dz.view(-1, ht, wd))
    return [dx, dz, energy]
