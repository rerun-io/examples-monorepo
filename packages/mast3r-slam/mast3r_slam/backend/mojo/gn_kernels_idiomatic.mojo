"""Experimental idiomatic GN ray-step kernel.

This module intentionally coexists with `gn_kernels.mojo` instead of replacing
it. The goal is to try a cleaner kernel organization while keeping the current
validated Mojo kernel and the CUDA oracle available for direct comparison.

The kernel keeps the proven execution model:
- one thread block handles one edge-partial pair,
- threads stride across points for that edge,
- each thread accumulates scalar local terms,
- one block-level reduction writes the partial Hessian and gradient.

What changes relative to the current kernel is the organization: the shared
pose loading, relative-pose setup, and row accumulation logic are spelled out
as named helpers so the control flow is easier to audit and benchmark.
"""

from std.math import sqrt
from std.gpu import block_idx, block_dim, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation

from gn_kernels import (
    POSE_DIM,
    POSE_STRIDE,
    RAYS_HDIM,
    RAYS_THREADS,
    RAYS_WARPS,
    act_sim3_into,
    apply_sim3_adj_inv_into,
    block_reduce_sum,
    huber_scalar,
    rel_sim3_into,
    squared_norm3_components,
)


@always_inline
def load_pose_pair_shared(
    twc: UnsafePointer[Float32, MutAnyOrigin],
    ix: Int,
    jx: Int,
    tid: Int,
    pose_shared: UnsafePointer[Scalar[DType.float32], MutAnyOrigin, address_space=AddressSpace.SHARED],
):
    var pi = ix * POSE_STRIDE
    var pj = jx * POSE_STRIDE
    if tid < POSE_STRIDE:
        pose_shared[tid] = twc[pi + tid]
    if tid < POSE_STRIDE:
        pose_shared[tid + POSE_STRIDE] = twc[pj + tid]


@always_inline
def load_relative_pose_shared(
    tid: Int,
    pose_shared: UnsafePointer[Scalar[DType.float32], MutAnyOrigin, address_space=AddressSpace.SHARED],
):
    if tid == 0:
        var rel_out0: Float32 = 0.0
        var rel_out1: Float32 = 0.0
        var rel_out2: Float32 = 0.0
        var rel_out3: Float32 = 0.0
        var rel_out4: Float32 = 0.0
        var rel_out5: Float32 = 0.0
        var rel_out6: Float32 = 0.0
        var rel_out7: Float32 = 0.0
        rel_sim3_into(
            pose_shared[0][0], pose_shared[1][0], pose_shared[2][0],
            pose_shared[3][0], pose_shared[4][0], pose_shared[5][0], pose_shared[6][0],
            pose_shared[7][0],
            pose_shared[8][0], pose_shared[9][0], pose_shared[10][0],
            pose_shared[11][0], pose_shared[12][0], pose_shared[13][0], pose_shared[14][0],
            pose_shared[15][0],
            rel_out0, rel_out1, rel_out2, rel_out3, rel_out4, rel_out5, rel_out6, rel_out7,
        )
        pose_shared[16] = rel_out0
        pose_shared[17] = rel_out1
        pose_shared[18] = rel_out2
        pose_shared[19] = rel_out3
        pose_shared[20] = rel_out4
        pose_shared[21] = rel_out5
        pose_shared[22] = rel_out6
        pose_shared[23] = rel_out7


@always_inline
def accumulate_pair_row(
    err: Float32,
    w: Float32,
    ji0: Float32, ji1: Float32, ji2: Float32, ji3: Float32, ji4: Float32, ji5: Float32, ji6: Float32,
    ti0: Float32, ti1: Float32, ti2: Float32,
    qi0: Float32, qi1: Float32, qi2: Float32, qi3: Float32,
    si0: Float32,
    mut hij: InlineArray[Float32, RAYS_HDIM],
    mut vi: InlineArray[Float32, POSE_DIM],
    mut vj: InlineArray[Float32, POSE_DIM],
):
    var jadj0: Float32 = 0.0
    var jadj1: Float32 = 0.0
    var jadj2: Float32 = 0.0
    var jadj3: Float32 = 0.0
    var jadj4: Float32 = 0.0
    var jadj5: Float32 = 0.0
    var jadj6: Float32 = 0.0
    apply_sim3_adj_inv_into(
        ti0, ti1, ti2,
        qi0, qi1, qi2, qi3, si0,
        ji0, ji1, ji2, ji3, ji4, ji5, ji6,
        jadj0, jadj1, jadj2, jadj3, jadj4, jadj5, jadj6,
    )

    var row = InlineArray[Float32, 14](fill=0.0)
    row[0] = -jadj0
    row[1] = -jadj1
    row[2] = -jadj2
    row[3] = -jadj3
    row[4] = -jadj4
    row[5] = -jadj5
    row[6] = -jadj6
    row[7] = jadj0
    row[8] = jadj1
    row[9] = jadj2
    row[10] = jadj3
    row[11] = jadj4
    row[12] = jadj5
    row[13] = jadj6

    var l = 0
    for n in range(14):
        for m in range(n + 1):
            hij[l] += w * row[n] * row[m]
            l += 1
    for n in range(POSE_DIM):
        vi[n] += w * err * row[n]
        vj[n] += w * err * row[n + POSE_DIM]


def gauss_newton_rays_step_kernel_idiomatic(
    twc: UnsafePointer[Float32, MutAnyOrigin],
    xs: UnsafePointer[Float32, MutAnyOrigin],
    cs: UnsafePointer[Float32, MutAnyOrigin],
    ii: UnsafePointer[Int64, MutAnyOrigin],
    jj: UnsafePointer[Int64, MutAnyOrigin],
    idx_ii2jj: UnsafePointer[Int64, MutAnyOrigin],
    valid_match: UnsafePointer[UInt8, MutAnyOrigin],
    q_tensor: UnsafePointer[Float32, MutAnyOrigin],
    hs: UnsafePointer[Float32, MutAnyOrigin],
    gs: UnsafePointer[Float32, MutAnyOrigin],
    num_points: Int,
    num_edges: Int,
    blocks_per_edge: Int,
    sigma_ray: Float32,
    sigma_dist: Float32,
    c_thresh: Float32,
    q_thresh: Float32,
):
    var edge = Int(block_idx.x) // blocks_per_edge
    var edge_block = Int(block_idx.x) % blocks_per_edge
    var partial_edge = edge * blocks_per_edge + edge_block
    var num_partials = num_edges * blocks_per_edge
    var tid = Int(thread_idx.x)
    var block_threads = Int(block_dim.x)
    if edge >= num_edges:
        return

    var ix = Int(ii[edge])
    var jx = Int(jj[edge])
    var pose_shared = stack_allocation[
        24,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    load_pose_pair_shared(twc, ix, jx, tid, pose_shared)
    barrier()
    load_relative_pose_shared(tid, pose_shared)
    barrier()

    var ti0 = pose_shared[0][0]
    var ti1 = pose_shared[1][0]
    var ti2 = pose_shared[2][0]
    var qi0 = pose_shared[3][0]
    var qi1 = pose_shared[4][0]
    var qi2 = pose_shared[5][0]
    var qi3 = pose_shared[6][0]
    var si0 = pose_shared[7][0]
    var rel0 = pose_shared[16][0]
    var rel1 = pose_shared[17][0]
    var rel2 = pose_shared[18][0]
    var rel3 = pose_shared[19][0]
    var rel4 = pose_shared[20][0]
    var rel5 = pose_shared[21][0]
    var rel6 = pose_shared[22][0]
    var rel7 = pose_shared[23][0]

    var hij = InlineArray[Float32, RAYS_HDIM](fill=0.0)
    var vi = InlineArray[Float32, POSE_DIM](fill=0.0)
    var vj = InlineArray[Float32, POSE_DIM](fill=0.0)

    var sigma_ray_inv = 1.0 / sigma_ray
    var sigma_dist_inv = 1.0 / sigma_dist

    for k in range(edge_block * block_threads + tid, num_points, block_threads * blocks_per_edge):
        var match_ok = valid_match[edge * num_points + k] != 0
        var mapped_idx = Int(idx_ii2jj[edge * num_points + k])
        if not match_ok:
            mapped_idx = 0

        var xi_base = (ix * num_points + mapped_idx) * 3
        var xj_base = (jx * num_points + k) * 3
        var ci = cs[ix * num_points + mapped_idx]
        var cj = cs[jx * num_points + k]
        var qv = q_tensor[edge * num_points + k]

        var xi0 = xs[xi_base + 0]
        var xi1 = xs[xi_base + 1]
        var xi2 = xs[xi_base + 2]
        var xj0 = xs[xj_base + 0]
        var xj1 = xs[xj_base + 1]
        var xj2 = xs[xj_base + 2]

        var norm2_i = squared_norm3_components(xi0, xi1, xi2)
        var norm1_i = sqrt(norm2_i)
        var norm1_i_inv = 1.0 / norm1_i
        var ri0 = norm1_i_inv * xi0
        var ri1 = norm1_i_inv * xi1
        var ri2 = norm1_i_inv * xi2

        var xj_ci0: Float32 = 0.0
        var xj_ci1: Float32 = 0.0
        var xj_ci2: Float32 = 0.0
        act_sim3_into(
            rel0, rel1, rel2, rel3, rel4, rel5, rel6, rel7,
            xj0, xj1, xj2,
            xj_ci0, xj_ci1, xj_ci2,
        )
        var norm2_j = squared_norm3_components(xj_ci0, xj_ci1, xj_ci2)
        var norm1_j = sqrt(norm2_j)
        var norm1_j_inv = 1.0 / norm1_j
        var rj0 = norm1_j_inv * xj_ci0
        var rj1 = norm1_j_inv * xj_ci1
        var rj2 = norm1_j_inv * xj_ci2

        var err0 = rj0 - ri0
        var err1 = rj1 - ri1
        var err2 = rj2 - ri2
        var err3 = norm1_j - norm1_i

        var valid = match_ok and (qv > q_thresh) and (ci > c_thresh) and (cj > c_thresh)
        var sqrt_w_ray: Float32 = 0.0
        var sqrt_w_dist: Float32 = 0.0
        if valid:
            var sqrt_conf = sqrt(qv)
            sqrt_w_ray = sigma_ray_inv * sqrt_conf
            sqrt_w_dist = sigma_dist_inv * sqrt_conf

        var w0 = huber_scalar(sqrt_w_ray * err0) * sqrt_w_ray * sqrt_w_ray
        var w1 = huber_scalar(sqrt_w_ray * err1) * sqrt_w_ray * sqrt_w_ray
        var w2 = huber_scalar(sqrt_w_ray * err2) * sqrt_w_ray * sqrt_w_ray
        var w3 = huber_scalar(sqrt_w_dist * err3) * sqrt_w_dist * sqrt_w_dist

        var norm3_j_inv = norm1_j_inv / norm2_j
        var drx_dpx = norm1_j_inv - xj_ci0 * xj_ci0 * norm3_j_inv
        var dry_dpy = norm1_j_inv - xj_ci1 * xj_ci1 * norm3_j_inv
        var drz_dpz = norm1_j_inv - xj_ci2 * xj_ci2 * norm3_j_inv
        var drx_dpy = -xj_ci0 * xj_ci1 * norm3_j_inv
        var drx_dpz = -xj_ci0 * xj_ci2 * norm3_j_inv
        var dry_dpz = -xj_ci1 * xj_ci2 * norm3_j_inv

        accumulate_pair_row(err0, w0, drx_dpx, drx_dpy, drx_dpz, 0.0, rj2, -rj1, 0.0, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, hij, vi, vj)
        accumulate_pair_row(err1, w1, drx_dpy, dry_dpy, dry_dpz, -rj2, 0.0, rj0, 0.0, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, hij, vi, vj)
        accumulate_pair_row(err2, w2, drx_dpz, dry_dpz, drz_dpz, rj1, -rj0, 0.0, 0.0, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, hij, vi, vj)
        accumulate_pair_row(err3, w3, rj0, rj1, rj2, 0.0, 0.0, 0.0, norm1_j, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, hij, vi, vj)

    var warp_partials = stack_allocation[
        RAYS_WARPS,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    for n in range(POSE_DIM):
        var vi_sum = block_reduce_sum(vi[n], tid, warp_partials)
        if tid == 0:
            gs[(0 * num_partials + partial_edge) * POSE_DIM + n] = vi_sum

        var vj_sum = block_reduce_sum(vj[n], tid, warp_partials)
        if tid == 0:
            gs[(1 * num_partials + partial_edge) * POSE_DIM + n] = vj_sum

    var l = 0
    for n in range(14):
        for m in range(n + 1):
            var h_sum = block_reduce_sum(hij[l], tid, warp_partials)
            if tid == 0:
                var val = h_sum
                if n < POSE_DIM and m < POSE_DIM:
                    hs[((0 * num_partials + partial_edge) * 49) + n * 7 + m] = val
                    hs[((0 * num_partials + partial_edge) * 49) + m * 7 + n] = val
                elif n >= POSE_DIM and m < POSE_DIM:
                    hs[((1 * num_partials + partial_edge) * 49) + m * 7 + (n - POSE_DIM)] = val
                    hs[((2 * num_partials + partial_edge) * 49) + (n - POSE_DIM) * 7 + m] = val
                else:
                    hs[((3 * num_partials + partial_edge) * 49) + (n - POSE_DIM) * 7 + (m - POSE_DIM)] = val
                    hs[((3 * num_partials + partial_edge) * 49) + (m - POSE_DIM) * 7 + (n - POSE_DIM)] = val
            l += 1
