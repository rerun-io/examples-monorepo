"""ViPE's SLAM kernels (`csrc/slam_ext/geom_kernels.cu`): frame distance and the dense bundle adjustment.

The CUDA kernels run one 256-thread block per edge (or pair): thread t walks the pixels t, t + 256, ...
and keeps running sums, then `blockReduce` folds the 256 sums in a fixed tree (offsets 128, 64, ..., 1).
Here that is two launches with the same order: a `*_partial` element is one (block, thread) and writes
its running sums to a scratch row (blocks, sums, 256), and `block_reduce` folds each row in the same tree.
No float atomics are involved, so the results do not depend on scheduling, on any device.

The per-pixel outputs of the BA kernel (Cii, bz, Eii, Eij, Efi) are written by the thread that owns the
pixel, as in CUDA. Tensors travel as raw pointers to contiguous float32 / int64 memory.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.collections.array import Array
from std.utils.static_tuple import StaticTuple

from droid import F32Ptr, I64Ptr, NUM_THREADS
from lie_math import sqrt_
from layout import TileTensor, row_major
from se3 import Vec4, Vec8, act_se3, adj_se3, rel_se3, retr_se3

comptime THREADS = 256
comptime MIN_DEPTH: Float32 = 0.25
comptime HIJ = 78  # the lower triangle of the 12 x 12 pose-pose block
comptime BA_SUMS = HIJ + 4 * 6 + 3  # hij, vi, vj, fi, fj, hf, vfocal, residual energy


@always_inline
def _ld(p: F32Ptr, i: Int) -> Float32:
    """p[i] through a 1-D TileTensor view. Metal's air-lld crashes (segfault) on SIMD values built from raw-pointer loads."""
    return TileTensor(p, row_major((i + 1,)))[i][0]


@always_inline
def _pose(poses: F32Ptr, i: Int) -> Tuple[Vec4, Vec4]:
    var o = 7 * i
    return (
        Vec4(_ld(poses, o), _ld(poses, o + 1), _ld(poses, o + 2), 0.0),
        Vec4(_ld(poses, o + 3), _ld(poses, o + 4), _ld(poses, o + 5), _ld(poses, o + 6)),
    )


@always_inline
def frame_distance_partial_element(
    n: Int,
    height: Int,
    width: Int,
    beta: Float32,
    poses: F32Ptr,
    disps: F32Ptr,
    intrinsics: F32Ptr,
    pi: I64Ptr,
    pj: I64Ptr,
    ri: I64Ptr,
    rj: I64Ptr,
    di: I64Ptr,
    partial: F32Ptr,
):
    """Thread n % 256 of pair n // 256 of `frame_distance_kernel`: sums of beta * flow, beta and valid beta."""
    var e = n // THREADS
    var t = n % THREADS
    var dix = Int(di[unsafe_offset=e])
    var rix = Int(ri[unsafe_offset=e])
    var rjx = Int(rj[unsafe_offset=e])
    var fxi = _ld(intrinsics, 4 * rix)
    var fyi = _ld(intrinsics, 4 * rix + 1)
    var cxi = _ld(intrinsics, 4 * rix + 2)
    var cyi = _ld(intrinsics, 4 * rix + 3)
    var fxj = _ld(intrinsics, 4 * rjx)
    var fyj = _ld(intrinsics, 4 * rjx + 1)
    var cxj = _ld(intrinsics, 4 * rjx + 2)
    var cyj = _ld(intrinsics, 4 * rjx + 3)
    var ti, qi = _pose(poses, Int(pi[unsafe_offset=e]))
    var tj, qj = _pose(poses, Int(pj[unsafe_offset=e]))
    var tij, qij = rel_se3(ti, qi, tj, qj)

    var accum: Float32 = 0.0
    var valid: Float32 = 0.0
    var total: Float32 = 0.0
    var k = t
    while k < height * width:
        var u = Float32(k % width)
        var v = Float32(k // width)
        var disp = _ld(disps, dix * height * width + k)
        var xi = Vec4((u - cxi) / fxi, (v - cyi) / fyi, 1.0, disp)
        var xj = act_se3(tij, qij, xi)
        var du = fxj * (xj[0] / xj[2]) + cxj - u
        var dv = fyj * (xj[1] / xj[2]) + cyj - v
        var d = sqrt_(du * du + dv * dv)
        total += beta
        if xj[2] > MIN_DEPTH:
            accum += beta * d
            valid += beta
        # translation only
        xj = Vec4(xi[0] + xi[3] * tij[0], xi[1] + xi[3] * tij[1], xi[2] + xi[3] * tij[2], 0.0)
        du = fxj * (xj[0] / xj[2]) + cxj - u
        dv = fyj * (xj[1] / xj[2]) + cyj - v
        d = sqrt_(du * du + dv * dv)
        total += 1 - beta
        if xj[2] > MIN_DEPTH:
            accum += (1 - beta) * d
            valid += 1 - beta
        k += THREADS
    partial[unsafe_offset=(e * 3 + 0) * THREADS + t] = accum
    partial[unsafe_offset=(e * 3 + 1) * THREADS + t] = total
    partial[unsafe_offset=(e * 3 + 2) * THREADS + t] = valid


@always_inline
def block_reduce_element(n: Int, partial: F32Ptr, dst: F32Ptr):
    """`blockReduce` of row n of partial (rows, 256): the CUDA tree, offsets 128, 64, ..., 1."""
    var s = Array[Float32, THREADS](uninitialized=True)
    for t in range(THREADS):
        s[t] = _ld(partial, n * THREADS + t)
    var offset = THREADS // 2
    while offset > 0:
        for t in range(offset):
            s[t] = s[t] + s[t + offset]
        offset //= 2
    dst[unsafe_offset=n] = s[0]


@always_inline
def _jx(ji: Vec8, jj: Vec8, n: Int) -> Float32:
    return ji[n] if n < 6 else jj[n - 6]


@always_inline
def ba_partial_element(
    n: Int,
    height: Int,
    width: Int,
    optimize_intrinsics: Bool,
    intrinsics_scale: Float32,
    flow_weight: Float32,
    target: F32Ptr,
    weight: F32Ptr,
    poses: F32Ptr,
    disps: F32Ptr,
    intrinsics: F32Ptr,
    depth_active: F32Ptr,
    ii: I64Ptr,
    jj: I64Ptr,
    cii: F32Ptr,
    bz: F32Ptr,
    eii: F32Ptr,
    eij: F32Ptr,
    efi: F32Ptr,
    partial: F32Ptr,
):
    """Thread n % 256 of edge n // 256 of `projective_transform_kernel`."""
    var block = n // THREADS
    var thread = n % THREADS
    var hw = height * width
    var ix = Int(ii[unsafe_offset=block])
    var jx = Int(jj[unsafe_offset=block])
    var active_depth = _ld(depth_active, ix)
    var fx = _ld(intrinsics, 0)
    var fy = _ld(intrinsics, 1)
    var cx = _ld(intrinsics, 2)
    var cy = _ld(intrinsics, 3)
    var tij = Vec4(-0.1, 0.0, 0.0, 0.0)
    var qij = Vec4(0.0, 0.0, 0.0, 1.0)
    if ix != jx:
        var ti, qi = _pose(poses, ix)
        var tj, qj = _pose(poses, jx)
        tij, qij = rel_se3(ti, qi, tj, qj)

    var hij = Array[Float32, HIJ](fill=0.0)
    var vi = Vec8(0.0)
    var vj = Vec8(0.0)
    var fi = Vec8(0.0)
    var fj = Vec8(0.0)
    var hf: Float32 = 0.0
    var vfocal: Float32 = 0.0
    var residual_energy: Float32 = 0.0

    var k = thread
    while k < hw:
        var u = Float32(k % width)
        var v = Float32(k // width)
        var xi = Vec4((u - cx) / fx, (v - cy) / fy, 1.0, _ld(disps, ix * hw + k))
        var xj = act_se3(tij, qij, xi)
        var x = xj[0]
        var y = xj[1]
        var h = xj[3]
        var behind = xj[2] < MIN_DEPTH
        var d: Float32 = 0.0 if behind else 1.0 / xj[2]
        var d2 = d * d
        var wu: Float32 = 0.0 if behind else flow_weight * _ld(weight, (block * 2 + 0) * hw + k)
        var wv: Float32 = 0.0 if behind else flow_weight * _ld(weight, (block * 2 + 1) * hw + k)
        var jf_u: Float32 = 0.0
        var jf_v: Float32 = 0.0
        if optimize_intrinsics:
            var dxj_df = act_se3(tij, qij, Vec4(-xi[0] / fx, -xi[1] / fy, 0.0, 0.0))
            jf_u = intrinsics_scale * (fx * d * dxj_df[0] - fx * x * d2 * dxj_df[2] + x * d)
            jf_v = intrinsics_scale * (fy * d * dxj_df[1] - fy * y * d2 * dxj_df[2] + y * d)
        var ru = _ld(target, (block * 2 + 0) * hw + k) - (fx * d * x + cx)
        var rv = _ld(target, (block * 2 + 1) * hw + k) - (fy * d * y + cy)
        residual_energy += wu * ru * ru + wv * rv * rv

        # x coordinate
        var j_j = Vec8(fx * (h * d), fx * 0, fx * (-x * h * d2), fx * (-x * y * d2), fx * (1 + x * x * d2), fx * (-y * d), 0.0, 0.0)
        var jz = fx * (tij[0] * d - tij[2] * (x * d2))
        var wu_depth = wu * active_depth
        var c_k = wu_depth * jz * jz
        var b_k = wu_depth * ru * jz
        var f_k: Float32 = 0.0
        if optimize_intrinsics:
            f_k = wu_depth * jz * jf_u
        if ix == jx:
            wu = 0
        var j_i = -adj_se3(tij, qij, j_j)
        var l = 0
        for a in range(12):
            for b in range(a + 1):
                hij[l] += wu * _jx(j_i, j_j, a) * _jx(j_i, j_j, b)
                l += 1
        var e_i = Array[Float32, 6](uninitialized=True)
        var e_j = Array[Float32, 6](uninitialized=True)
        for a in range(6):
            vi[a] += wu * ru * j_i[a]
            vj[a] += wu * ru * j_j[a]
            if optimize_intrinsics:
                fi[a] += wu * j_i[a] * jf_u
                fj[a] += wu * j_j[a] * jf_u
            e_i[a] = wu_depth * jz * j_i[a]
            e_j[a] = wu_depth * jz * j_j[a]
        if optimize_intrinsics:
            hf += wu * jf_u * jf_u
            vfocal += wu * ru * jf_u

        # y coordinate
        j_j = Vec8(fy * 0, fy * (h * d), fy * (-y * h * d2), fy * (-1 - y * y * d2), fy * (x * y * d2), fy * (x * d), 0.0, 0.0)
        jz = fy * (tij[1] * d - tij[2] * (y * d2))
        var wv_depth = wv * active_depth
        c_k += wv_depth * jz * jz
        b_k += wv_depth * rv * jz
        if optimize_intrinsics:
            f_k += wv_depth * jz * jf_v
        if ix == jx:
            wv = 0
        j_i = -adj_se3(tij, qij, j_j)
        l = 0
        for a in range(12):
            for b in range(a + 1):
                hij[l] += wv * _jx(j_i, j_j, a) * _jx(j_i, j_j, b)
                l += 1
        for a in range(6):
            vi[a] += wv * rv * j_i[a]
            vj[a] += wv * rv * j_j[a]
            if optimize_intrinsics:
                fi[a] += wv * j_i[a] * jf_v
                fj[a] += wv * j_j[a] * jf_v
            e_i[a] += wv_depth * jz * j_i[a]
            e_j[a] += wv_depth * jz * j_j[a]
        if optimize_intrinsics:
            hf += wv * jf_v * jf_v
            vfocal += wv * rv * jf_v

        cii[unsafe_offset=block * hw + k] = c_k
        bz[unsafe_offset=block * hw + k] = b_k
        if optimize_intrinsics:
            efi[unsafe_offset=block * hw + k] = f_k
        for a in range(6):
            eii[unsafe_offset=(block * 6 + a) * hw + k] = e_i[a]
            eij[unsafe_offset=(block * 6 + a) * hw + k] = e_j[a]
        k += THREADS

    var row = block * BA_SUMS
    for q in range(HIJ):
        partial[unsafe_offset=(row + q) * THREADS + thread] = hij[q]
    for a in range(6):
        partial[unsafe_offset=(row + HIJ + a) * THREADS + thread] = vi[a]
        partial[unsafe_offset=(row + HIJ + 6 + a) * THREADS + thread] = vj[a]
        partial[unsafe_offset=(row + HIJ + 12 + a) * THREADS + thread] = fi[a]
        partial[unsafe_offset=(row + HIJ + 18 + a) * THREADS + thread] = fj[a]
    partial[unsafe_offset=(row + HIJ + 24) * THREADS + thread] = hf
    partial[unsafe_offset=(row + HIJ + 25) * THREADS + thread] = vfocal
    partial[unsafe_offset=(row + HIJ + 26) * THREADS + thread] = residual_energy


@always_inline
def accum_element(n: Int, width: Int, inps: F32Ptr, order: I64Ptr, starts: I64Ptr, ends: I64Ptr, dst: F32Ptr):
    """out[j, k] = sum of inps[order[i], k] over i in [starts[j], ends[j]), in that order (`accum_kernel`)."""
    var j = n // width
    var k = n % width
    var x: Float32 = 0.0
    for i in range(Int(starts[unsafe_offset=j]), Int(ends[unsafe_offset=j])):
        x += _ld(inps, Int(order[unsafe_offset=i]) * width + k)
    dst[unsafe_offset=n] = x


@always_inline
def eet_partial_element(n: Int, width: Int, e: F32Ptr, q: F32Ptr, idx: I64Ptr, partial: F32Ptr):
    """Thread n % 256 of pair n // 256 of `EEt6x6_kernel`: E[ix] Q[kx] E[jx]^T over its pixels."""
    var p = n // THREADS
    var t = n % THREADS
    var ix = Int(idx[unsafe_offset=3 * p])
    var jx = Int(idx[unsafe_offset=3 * p + 1])
    var kx = Int(idx[unsafe_offset=3 * p + 2])
    var ds = Array[Float32, 36](fill=0.0)
    var k = t
    while k < width:
        var qk = _ld(q, kx * width + k)
        var ei = Array[Float32, 6](uninitialized=True)
        var ej = Array[Float32, 6](uninitialized=True)
        for a in range(6):
            ei[a] = _ld(e, (ix * 6 + a) * width + k) * qk
            ej[a] = _ld(e, (jx * 6 + a) * width + k)
        for a in range(6):
            for b in range(6):
                ds[a * 6 + b] += ei[a] * ej[b]
        k += THREADS
    for a in range(36):
        partial[unsafe_offset=(p * 36 + a) * THREADS + t] = ds[a]


@always_inline
def ev_partial_element(n: Int, width: Int, e: F32Ptr, q: F32Ptr, w: F32Ptr, kk: I64Ptr, partial: F32Ptr):
    """Thread n % 256 of row n // 256 of `Ev6x1_kernel`: E[r] (Q w)[kk[r]] over its pixels."""
    var r = n // THREADS
    var t = n % THREADS
    var kx = Int(kk[unsafe_offset=r])
    var b = Array[Float32, 6](fill=0.0)
    var k = t
    while k < width:
        var q_w = _ld(q, kx * width + k) * _ld(w, kx * width + k)
        for a in range(6):
            b[a] += q_w * _ld(e, (r * 6 + a) * width + k)
        k += THREADS
    for a in range(6):
        partial[unsafe_offset=(r * 6 + a) * THREADS + t] = b[a]


@always_inline
def evt_element(n: Int, width: Int, num_poses: Int, e: F32Ptr, x: F32Ptr, ix: I64Ptr, dw: F32Ptr):
    """dw[r, k] = E[r, :, k] . x[ix[r]] (`EvT6x1_kernel`; rows with ix <= 0 or ix >= num_poses stay 0, as in CUDA)."""
    var r = n // width
    var k = n % width
    var i = Int(ix[unsafe_offset=r])
    if i <= 0 or i >= num_poses:
        return
    var acc: Float32 = 0.0
    for a in range(6):
        acc += _ld(e, (r * 6 + a) * width + k) * _ld(x, i * 6 + a)
    dw[unsafe_offset=n] = acc


@always_inline
def pose_retr_element(i: Int, t0: Int, poses: F32Ptr, dx: F32Ptr):
    """Pose t0 + i becomes Exp(dx[i]) * pose (`pose_retr_kernel`)."""
    var t, q = _pose(poses, t0 + i)
    var o = 6 * i
    var xi = Vec8(_ld(dx, o), _ld(dx, o + 1), _ld(dx, o + 2), _ld(dx, o + 3), _ld(dx, o + 4), _ld(dx, o + 5), 0.0, 0.0)
    var t1, q1 = retr_se3(xi, t, q)
    var p = 7 * (t0 + i)
    poses[unsafe_offset=p] = t1[0]
    poses[unsafe_offset=p + 1] = t1[1]
    poses[unsafe_offset=p + 2] = t1[2]
    poses[unsafe_offset=p + 3] = q1[0]
    poses[unsafe_offset=p + 4] = q1[1]
    poses[unsafe_offset=p + 5] = q1[2]
    poses[unsafe_offset=p + 6] = q1[3]


# ── GPU kernels: one element per thread ─────────────────────────────────────


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def frame_distance_partial_kernel(
    total: Int64, height: Int64, width: Int64, beta: Float32, poses: F32Ptr, disps: F32Ptr, intrinsics: F32Ptr,
    pi: I64Ptr, pj: I64Ptr, ri: I64Ptr, rj: I64Ptr, di: I64Ptr, partial: F32Ptr,
):  # fmt: skip
    var n = global_idx.x
    if n < Int(total):
        frame_distance_partial_element(n, Int(height), Int(width), beta, poses, disps, intrinsics, pi, pj, ri, rj, di, partial)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def block_reduce_kernel(total: Int64, partial: F32Ptr, dst: F32Ptr):
    var n = global_idx.x
    if n < Int(total):
        block_reduce_element(n, partial, dst)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def ba_partial_kernel(
    total: Int64, height: Int64, width: Int64, optimize_intrinsics: Int64, intrinsics_scale: Float32, flow_weight: Float32,
    target: F32Ptr, weight: F32Ptr, poses: F32Ptr, disps: F32Ptr, intrinsics: F32Ptr, depth_active: F32Ptr, ii: I64Ptr, jj: I64Ptr,
    cii: F32Ptr, bz: F32Ptr, eii: F32Ptr, eij: F32Ptr, efi: F32Ptr, partial: F32Ptr,
):  # fmt: skip
    var n = global_idx.x
    if n < Int(total):
        ba_partial_element(
            n, Int(height), Int(width), optimize_intrinsics != 0, intrinsics_scale, flow_weight, target, weight, poses, disps, intrinsics,
            depth_active, ii, jj, cii, bz, eii, eij, efi, partial,
        )  # fmt: skip


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def accum_kernel(total: Int64, width: Int64, inps: F32Ptr, order: I64Ptr, starts: I64Ptr, ends: I64Ptr, dst: F32Ptr):
    var n = global_idx.x
    if n < Int(total):
        accum_element(n, Int(width), inps, order, starts, ends, dst)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def eet_partial_kernel(total: Int64, width: Int64, e: F32Ptr, q: F32Ptr, idx: I64Ptr, partial: F32Ptr):
    var n = global_idx.x
    if n < Int(total):
        eet_partial_element(n, Int(width), e, q, idx, partial)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def ev_partial_kernel(total: Int64, width: Int64, e: F32Ptr, q: F32Ptr, w: F32Ptr, kk: I64Ptr, partial: F32Ptr):
    var n = global_idx.x
    if n < Int(total):
        ev_partial_element(n, Int(width), e, q, w, kk, partial)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def evt_kernel(total: Int64, width: Int64, num_poses: Int64, e: F32Ptr, x: F32Ptr, ix: I64Ptr, dw: F32Ptr):
    var n = global_idx.x
    if n < Int(total):
        evt_element(n, Int(width), Int(num_poses), e, x, ix, dw)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def pose_retr_kernel(total: Int64, t0: Int64, poses: F32Ptr, dx: F32Ptr):
    var n = global_idx.x
    if n < Int(total):
        pose_retr_element(n, Int(t0), poses, dx)
