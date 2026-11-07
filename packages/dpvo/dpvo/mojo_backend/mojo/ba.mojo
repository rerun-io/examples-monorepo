"""DPVO's bundle-adjustment kernels (`dpvo/fastba/ba_cuda.cu`), one source for CPU and CUDA.

As in corr.mojo, each `*_element` function is one CUDA thread's body, shared by the GPU kernel and
the CPU loop. The Gauss-Newton driver (unique patches, Schur complement, Cholesky) is PyTorch on the
caller's device (`dpvo.mojo_backend.ba`), as in the CUDA extension's host code.

Shapes: poses (N, 7) float32, patches (K, 3, P, P) float32, intrinsics (*, 4) float32 (row 0 is
used), target / weight (E, 2) float32, ii / jj / kk / ku / kx (E,) or (M,) int64. The normal-equation
blocks B (6n, 6n), E (6n, m), C (m,), v (6n,), u (m,) and E_lookup (num_ij, ppf, 6) are float32 and
zeroed by the caller; the Hessian kernel accumulates into them with float atomics (CUDA's
atomicAdd), so they are passed as pointers with their row lengths.

`r_total` (the CUDA kernel's float64 cost) is not computed: the extension never returns it.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.atomic import Atomic, Ordering
from std.math import isfinite, log
from std.utils.static_tuple import StaticTuple

from layout import ImmTileTensor, MutTileTensor, row_major

from corr import L1, L4, NUM_THREADS
from lie_math import sqrt_
from se3 import Vec4, Vec8, act_se3, adj_se3, rel_se3, retr_se3

comptime L2 = type_of(row_major((0, 0)))
comptime L3 = type_of(row_major((0, 0, 0)))
comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]


@always_inline
def _add(p: F32Ptr, offset: Int, value: Float32):
    # CUDA's atomicAdd: relaxed, device scope (RED.E.ADD.F32...GPU). Mojo's default on NVIDIA is sequentially
    # consistent at system scope, a fence and cache invalidation per add: 2.4x slower Hessian on the 5090.
    _ = Atomic[Float32, scope="device"].fetch_add[ordering=Ordering.RELAXED](p.unsafe_offset(offset), value)


@always_inline
def _pose(poses: ImmTileTensor[DType.float32, L2, _], i: Int) -> Tuple[Vec4, Vec4]:
    comptime assert poses.flat_rank == 2
    var t = Vec4(poses[i, 0][0], poses[i, 1][0], poses[i, 2][0], 0.0)
    var q = Vec4(poses[i, 3][0], poses[i, 4][0], poses[i, 5][0], poses[i, 6][0])
    return (t, q)


@always_inline
def reproject_element(
    n: Int,
    poses: ImmTileTensor[DType.float32, L2, _],
    patches: ImmTileTensor[DType.float32, L4, _],
    intrinsics: ImmTileTensor[DType.float32, L2, _],
    ii: ImmTileTensor[DType.int64, L1, _],
    jj: ImmTileTensor[DType.int64, L1, _],
    kk: ImmTileTensor[DType.int64, L1, _],
    coords: MutTileTensor[DType.float32, L4, _],
):
    """Project patch kk[n] from frame ii[n] into frame jj[n]: coords (E, 2, P, P) (`reproject`)."""
    comptime assert patches.flat_rank == 4 and intrinsics.flat_rank == 2 and coords.flat_rank == 4
    comptime assert ii.flat_rank == 1 and jj.flat_rank == 1 and kk.flat_rank == 1
    var fx = intrinsics[0, 0][0]
    var fy = intrinsics[0, 1][0]
    var cx = intrinsics[0, 2][0]
    var cy = intrinsics[0, 3][0]
    var kx = Int(kk[n][0])
    var ti, qi = _pose(poses, Int(ii[n][0]))
    var tj, qj = _pose(poses, Int(jj[n][0]))
    var tij, qij = rel_se3(ti, qi, tj, qj)
    for i in range(Int(patches.dim[2]())):
        for j in range(Int(patches.dim[3]())):
            var xi = Vec4((patches[kx, 0, i, j][0] - cx) / fx, (patches[kx, 1, i, j][0] - cy) / fy, 1.0, patches[kx, 2, i, j][0])
            var xj = act_se3(tij, qij, xi)
            coords[n, 0, i, j] = fx * (xj[0] / xj[2]) + cx
            coords[n, 1, i, j] = fy * (xj[1] / xj[2]) + cy


@always_inline
def hessian_element(
    n: Int,
    t0: Int,
    ppf: Int,
    poses: ImmTileTensor[DType.float32, L2, _],
    patches: ImmTileTensor[DType.float32, L4, _],
    intrinsics: ImmTileTensor[DType.float32, L2, _],
    target: ImmTileTensor[DType.float32, L2, _],
    weight: ImmTileTensor[DType.float32, L2, _],
    ij_xself: ImmTileTensor[DType.int64, L2, _],
    ii: ImmTileTensor[DType.int64, L1, _],
    jj: ImmTileTensor[DType.int64, L1, _],
    kk: ImmTileTensor[DType.int64, L1, _],
    ku: ImmTileTensor[DType.int64, L1, _],
    b: F32Ptr,
    b_row: Int,
    e: F32Ptr,
    e_row: Int,
    e_lookup: F32Ptr,
    c: F32Ptr,
    v: F32Ptr,
    u: F32Ptr,
):
    """Residuals and normal-equation blocks of edge n (`reprojection_residuals_and_hessian`).

    `ppf` > 0 selects the block-E layout (`E_lookup[ij][kk % ppf][i]`, rows by `ij_xself`) instead of
    the dense E (b_row = 6n columns of B, e_row = m columns of E).
    """
    comptime assert target.flat_rank == 2 and weight.flat_rank == 2 and ij_xself.flat_rank == 2
    comptime assert patches.flat_rank == 4 and intrinsics.flat_rank == 2
    comptime assert ii.flat_rank == 1 and jj.flat_rank == 1 and kk.flat_rank == 1 and ku.flat_rank == 1
    var eff_impl = ppf > 0
    var fx = intrinsics[0, 0][0]
    var fy = intrinsics[0, 1][0]
    var cx = intrinsics[0, 2][0]
    var cy = intrinsics[0, 3][0]

    var k = Int(ku[n][0])
    var ix = Int(ii[n][0])
    var jx = Int(jj[n][0])
    var kx = Int(kk[n][0])
    var ijx = 0
    var ijs = 0
    if eff_impl:
        ijx = Int(ij_xself[0, n][0])
        ijs = Int(ij_xself[1, n][0])

    var ti, qi = _pose(poses, ix)
    var tj, qj = _pose(poses, jx)
    var xi = Vec4((patches[kx, 0, 1, 1][0] - cx) / fx, (patches[kx, 1, 1, 1][0] - cy) / fy, 1.0, patches[kx, 2, 1, 1][0])
    var tij, qij = rel_se3(ti, qi, tj, qj)
    var xj = act_se3(tij, qij, xi)
    var X = xj[0]
    var Y = xj[1]
    var Z = xj[2]
    var W = xj[3]

    var d: Float32 = 1.0 / Z if Z >= 0.2 else 0.0
    var d2 = d * d
    var x1 = fx * (X / Z) + cx
    var y1 = fy * (Y / Z) + cy
    var rx = target[n, 0][0] - x1
    var ry = target[n, 1][0] - y1
    var in_bounds = sqrt_(rx * rx + ry * ry) < 128 and Z > 0.2 and x1 > -64 and y1 > -64 and x1 < 2 * cx + 64 and y1 < 2 * cy + 64
    var mask: Float32 = 1.0 if in_bounds else 0.0

    ix = ix - t0
    jx = jx - t0
    # Unrolled (as nvcc does) so j_i / j_j stay in registers; a runtime loop put them on the stack.
    comptime for row in range(2):
        var r: Float32
        var w: Float32
        var jz: Float32
        var j_j: Vec8
        if row == 0:
            r = target[n, 0][0] - x1
            w = mask * weight[n, 0][0]
            jz = fx * (tij[0] * d - tij[2] * (X * d2))
            j_j = Vec8(fx * W * d, 0, fx * -X * W * d2, fx * -X * Y * d2, fx * (1 + X * X * d2), fx * -Y * d, 0, 0)
        else:
            r = target[n, 1][0] - y1
            w = mask * weight[n, 1][0]
            jz = fy * (tij[1] * d - tij[2] * (Y * d2))
            j_j = Vec8(0, fy * W * d, fy * -Y * W * d2, fy * (-1 - Y * Y * d2), fy * (X * Y * d2), fy * X * d, 0, 0)
        var j_i = adj_se3(tij, qij, j_j)

        comptime for i in range(6):
            comptime for j in range(6):
                if ix >= 0:
                    _add(b, (6 * ix + i) * b_row + 6 * ix + j, w * j_i[i] * j_i[j])
                if jx >= 0:
                    _add(b, (6 * jx + i) * b_row + 6 * jx + j, w * j_j[i] * j_j[j])
                if ix >= 0 and jx >= 0:
                    _add(b, (6 * ix + i) * b_row + 6 * jx + j, -w * j_i[i] * j_j[j])
                    _add(b, (6 * jx + i) * b_row + 6 * ix + j, -w * j_j[i] * j_i[j])

        comptime for i in range(6):
            if eff_impl:
                _add(e_lookup, (ijs * ppf + kx % ppf) * 6 + i, -w * jz * j_i[i])
                _add(e_lookup, (ijx * ppf + kx % ppf) * 6 + i, w * jz * j_j[i])
            else:
                if ix >= 0:
                    _add(e, (6 * ix + i) * e_row + k, -w * jz * j_i[i])
                if jx >= 0:
                    _add(e, (6 * jx + i) * e_row + k, w * jz * j_j[i])

        comptime for i in range(6):
            if ix >= 0:
                _add(v, 6 * ix + i, -w * r * j_i[i])
            if jx >= 0:
                _add(v, 6 * jx + i, w * r * j_j[i])

        _add(c, k, w * jz * jz)
        _add(u, k, w * r * jz)


@always_inline
def depth_prior_element(
    n: Int,
    patches: ImmTileTensor[DType.float32, L4, _],
    prior: ImmTileTensor[DType.float32, L1, _],
    prior_weight: ImmTileTensor[DType.float32, L1, _],
    kx: ImmTileTensor[DType.int64, L1, _],
    prior_lambda: Float32,
    huber_delta: Float32,
    c: F32Ptr,
    u: F32Ptr,
):
    """Robust log-inverse-depth prior on unique patch n (`depth_prior_residuals_and_hessian`, MetricTrail's unary)."""
    comptime assert patches.flat_rank == 4 and prior.flat_rank == 1 and prior_weight.flat_rank == 1 and kx.flat_rank == 1
    var patch_index = Int(kx[n][0])
    if patch_index >= Int(prior.dim[0]()):
        return
    var base_weight = prior_weight[patch_index][0]
    var p = prior[patch_index][0]
    var current = patches[patch_index, 2, 1, 1][0]
    if not (base_weight > 0.0) or not (p > 0.0) or not (current > 0.0) or not isfinite(p) or not isfinite(current) or not (prior_lambda > 0.0):
        return
    # r = log(d_prior) - log(d); dr/dd = -1/d, so the right-hand side -J^T r is r/d.
    var residual = log(p) - log(current)
    var abs_residual = abs(residual)
    var robust: Float32 = 1.0 if abs_residual <= huber_delta else huber_delta / max(abs_residual, 1e-6)
    var w = prior_lambda * base_weight * robust
    var jac = 1.0 / max(current, 1e-4)
    _add(c, n, w * jac * jac)
    _add(u, n, w * residual * jac)


@always_inline
def pose_retr_element(i: Int, t0: Int, poses: MutTileTensor[DType.float32, L2, _], update: ImmTileTensor[DType.float32, L2, _]):
    """Pose t0 + i becomes Exp(update[i]) * pose (`pose_retr_kernel`)."""
    comptime assert poses.flat_rank == 2 and update.flat_rank == 2
    var t = t0 + i
    var xi = Vec8(update[i, 0][0], update[i, 1][0], update[i, 2][0], update[i, 3][0], update[i, 4][0], update[i, 5][0], 0, 0)
    var t1, q1 = retr_se3(xi, Vec4(poses[t, 0][0], poses[t, 1][0], poses[t, 2][0], 0.0), Vec4(poses[t, 3][0], poses[t, 4][0], poses[t, 5][0], poses[t, 6][0]))
    poses[t, 0] = t1[0]
    poses[t, 1] = t1[1]
    poses[t, 2] = t1[2]
    poses[t, 3] = q1[0]
    poses[t, 4] = q1[1]
    poses[t, 5] = q1[2]
    poses[t, 6] = q1[3]


@always_inline
def patch_retr_element(n: Int, index: ImmTileTensor[DType.int64, L1, _], patches: MutTileTensor[DType.float32, L4, _], update: ImmTileTensor[DType.float32, L1, _]):
    """Inverse depth of patch index[n] += update[n], reset to 1 above 20, floored at 1e-4 (`patch_retr_kernel`)."""
    comptime assert index.flat_rank == 1 and patches.flat_rank == 4 and update.flat_rank == 1
    var ix = Int(index[n][0])
    var d = patches[ix, 2, 0, 0][0] + update[n][0]
    d = 1.0 if d > 20 else d
    d = max(d, 1e-4)
    for i in range(Int(patches.dim[2]())):
        for j in range(Int(patches.dim[3]())):
            patches[ix, 2, i, j] = d


# ── block-E Schur products (`block_e.cu`, global BA) ─────────────────────────
# E is stored per (source frame i, target frame j) block and patch slot: E_lookup[ij][k % ppf][6].
# `patch_to_ku[i][k]` is the unique-patch position of slot k of frame i, or -1 for a slot with no
# patch; the CUDA kernels then read Q[-1] (and multiply it by a zero block), these skip the slot.


@always_inline
def eet_element(
    n: Int,
    t0: Int,
    ppf: Int,
    e_lookup: ImmTileTensor[DType.float32, L3, _],
    q: ImmTileTensor[DType.float32, L1, _],
    index: ImmTileTensor[DType.int64, L2, _],
    patch_to_ku: ImmTileTensor[DType.int64, L2, _],
    eet: F32Ptr,
    eet_row: Int,
):
    """E Q E^T contribution of slot k of the frame-pair row idx = (i, j1, j2, ij1, ij2) (`EEt_kernel`)."""
    comptime assert e_lookup.flat_rank == 3 and q.flat_rank == 1 and index.flat_rank == 2 and patch_to_ku.flat_rank == 2
    var k = n % ppf
    var idx = n // ppf
    var i = Int(index[idx, 0][0])
    var j1 = Int(index[idx, 1][0]) - t0
    var j2 = Int(index[idx, 2][0]) - t0
    var s1 = Int(index[idx, 3][0])
    var s2 = Int(index[idx, 4][0])
    var q_idx = Int(patch_to_ku[i, k][0])
    if j1 < 0 or j2 < 0 or q_idx < 0:
        return
    var qv = q[q_idx][0]
    for xi in range(6):
        for xj in range(6):
            _add(eet, (6 * j1 + xi) * eet_row + 6 * j2 + xj, e_lookup[s1, k, xi][0] * e_lookup[s2, k, xj][0] * qv)


@always_inline
def ev_element(
    n: Int,
    t0: Int,
    ppf: Int,
    e_lookup: ImmTileTensor[DType.float32, L3, _],
    vec: ImmTileTensor[DType.float32, L1, _],
    block_index: ImmTileTensor[DType.int64, L2, _],
    patch_to_ku: ImmTileTensor[DType.int64, L2, _],
    ev: F32Ptr,
):
    """E v contribution of slot k of block idx = (i, j) (`Ev_kernel`): ev[6 j + r] += E[idx][k][r] v[ku]."""
    comptime assert e_lookup.flat_rank == 3 and vec.flat_rank == 1 and block_index.flat_rank == 2 and patch_to_ku.flat_rank == 2
    var k = n % ppf
    var idx = n // ppf
    var i = Int(block_index[idx, 0][0])
    var j = Int(block_index[idx, 1][0]) - t0
    var q_idx = Int(patch_to_ku[i, k][0])
    if j < 0 or q_idx < 0:
        return
    var value = vec[q_idx][0]
    for r in range(6):
        _add(ev, j * 6 + r, e_lookup[idx, k, r][0] * value)


@always_inline
def etv_element(
    n: Int,
    t0: Int,
    ppf: Int,
    e_lookup: ImmTileTensor[DType.float32, L3, _],
    vec: ImmTileTensor[DType.float32, L1, _],
    block_index: ImmTileTensor[DType.int64, L2, _],
    patch_to_ku: ImmTileTensor[DType.int64, L2, _],
    etv: F32Ptr,
):
    """E^T v contribution of slot k of block idx = (i, j) (`Etv_kernel`): etv[ku] += E[idx][k][r] v[6 j + r]."""
    comptime assert e_lookup.flat_rank == 3 and vec.flat_rank == 1 and block_index.flat_rank == 2 and patch_to_ku.flat_rank == 2
    var k = n % ppf
    var idx = n // ppf
    var i = Int(block_index[idx, 0][0])
    var j = Int(block_index[idx, 1][0]) - t0
    var q_idx = Int(patch_to_ku[i, k][0])
    if j < 0 or q_idx < 0:
        return
    for r in range(6):
        _add(etv, q_idx, e_lookup[idx, k, r][0] * vec[j * 6 + r][0])


# ── GPU kernels: one thread per element, guarded by the element count ────────


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def reproject_kernel(
    poses: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    patches: ImmTileTensor[DType.float32, L4, ImmUntrackedOrigin],
    intrinsics: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    ii: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    jj: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    kk: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    coords: MutTileTensor[DType.float32, L4, MutUntrackedOrigin],
):
    var n = global_idx.x
    if n < Int(ii.dim[0]()):
        reproject_element(n, poses, patches, intrinsics, ii, jj, kk, coords)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def hessian_kernel(
    t0: Int64,
    ppf: Int64,
    b_row: Int64,
    e_row: Int64,
    poses: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    patches: ImmTileTensor[DType.float32, L4, ImmUntrackedOrigin],
    intrinsics: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    target: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    weight: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    ij_xself: ImmTileTensor[DType.int64, L2, ImmUntrackedOrigin],
    ii: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    jj: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    kk: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    ku: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    b: F32Ptr,
    e: F32Ptr,
    e_lookup: F32Ptr,
    c: F32Ptr,
    v: F32Ptr,
    u: F32Ptr,
):
    var n = global_idx.x
    if n < Int(ii.dim[0]()):
        hessian_element(
            n, Int(t0), Int(ppf), poses, patches, intrinsics, target, weight, ij_xself, ii, jj, kk, ku, b, Int(b_row), e, Int(e_row), e_lookup, c, v, u
        )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def depth_prior_kernel(
    prior_lambda: Float32,
    huber_delta: Float32,
    patches: ImmTileTensor[DType.float32, L4, ImmUntrackedOrigin],
    prior: ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin],
    prior_weight: ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin],
    kx: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    c: F32Ptr,
    u: F32Ptr,
):
    var n = global_idx.x
    if n < Int(kx.dim[0]()):
        depth_prior_element(n, patches, prior, prior_weight, kx, prior_lambda, huber_delta, c, u)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def pose_retr_kernel(t0: Int64, poses: MutTileTensor[DType.float32, L2, MutUntrackedOrigin], update: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin]):
    var i = global_idx.x
    if i < Int(update.dim[0]()):
        pose_retr_element(i, Int(t0), poses, update)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def patch_retr_kernel(
    index: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    patches: MutTileTensor[DType.float32, L4, MutUntrackedOrigin],
    update: ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin],
):
    var n = global_idx.x
    if n < Int(index.dim[0]()):
        patch_retr_element(n, index, patches, update)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def eet_kernel(
    total: Int64,
    t0: Int64,
    ppf: Int64,
    eet_row: Int64,
    e_lookup: ImmTileTensor[DType.float32, L3, ImmUntrackedOrigin],
    q: ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin],
    index: ImmTileTensor[DType.int64, L2, ImmUntrackedOrigin],
    patch_to_ku: ImmTileTensor[DType.int64, L2, ImmUntrackedOrigin],
    eet: F32Ptr,
):
    var n = global_idx.x
    if n < Int(total):
        eet_element(n, Int(t0), Int(ppf), e_lookup, q, index, patch_to_ku, eet, Int(eet_row))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def ev_kernel(
    total: Int64,
    t0: Int64,
    ppf: Int64,
    e_lookup: ImmTileTensor[DType.float32, L3, ImmUntrackedOrigin],
    vec: ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin],
    block_index: ImmTileTensor[DType.int64, L2, ImmUntrackedOrigin],
    patch_to_ku: ImmTileTensor[DType.int64, L2, ImmUntrackedOrigin],
    ev: F32Ptr,
):
    var n = global_idx.x
    if n < Int(total):
        ev_element(n, Int(t0), Int(ppf), e_lookup, vec, block_index, patch_to_ku, ev)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def etv_kernel(
    total: Int64,
    t0: Int64,
    ppf: Int64,
    e_lookup: ImmTileTensor[DType.float32, L3, ImmUntrackedOrigin],
    vec: ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin],
    block_index: ImmTileTensor[DType.int64, L2, ImmUntrackedOrigin],
    patch_to_ku: ImmTileTensor[DType.int64, L2, ImmUntrackedOrigin],
    etv: F32Ptr,
):
    var n = global_idx.x
    if n < Int(total):
        etv_element(n, Int(t0), Int(ppf), e_lookup, vec, block_index, patch_to_ku, etv)
