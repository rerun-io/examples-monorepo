"""Low-level GN GPU kernels for the shared-library Mojo backend.

These kernels use GPU threads for outer parallelism and keep the per-thread
math scalar. That matches the current workload better than forcing additional
SIMD structure into branchy per-point residual code, and it keeps the reduction
logic explicit at the block level.
"""

from std.math import sqrt, exp, sin, cos, abs
from std.gpu import global_idx, block_idx, block_dim, thread_idx, barrier, lane_id
from std.gpu.primitives import warp
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation

comptime POSE_DIM = 7
comptime POSE_STRIDE = 8
comptime RAYS_HDIM = 14 * (14 + 1) // 2
comptime RAYS_THREADS = 128
comptime RAYS_WARPS = RAYS_THREADS // 32


@always_inline
def block_reduce_sum[origin: Origin[mut=True], //](
    value: Float32,
    tid: Int,
    warp_partials: UnsafePointer[Scalar[DType.float32], origin, address_space=AddressSpace.SHARED],
) -> Float32:
    # First reduce within each warp, then let one lane per warp publish to
    # shared memory for the block-wide finish. This keeps the synchronization
    # surface small while preserving the same accumulation order we validated
    # against CUDA.
    var reduced = warp.sum(value)
    if Int(lane_id()) == 0:
        warp_partials[tid // 32] = reduced
    barrier()

    var block_sum: Float32 = 0.0
    if tid < RAYS_WARPS:
        block_sum = warp_partials[tid][0]
    if tid < 32:
        block_sum = warp.sum(block_sum)
    barrier()
    return block_sum


@always_inline
def quat_comp_components(
    aix: Float32, aiy: Float32, aiz: Float32, aiw: Float32,
    bjx: Float32, bjy: Float32, bjz: Float32, bjw: Float32,
) -> InlineArray[Float32, 4]:
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = aiw * bjx + aix * bjw + aiy * bjz - aiz * bjy
    out[1] = aiw * bjy - aix * bjz + aiy * bjw + aiz * bjx
    out[2] = aiw * bjz + aix * bjy - aiy * bjx + aiz * bjw
    out[3] = aiw * bjw - aix * bjx - aiy * bjy - aiz * bjz
    return out^


@always_inline
def quat_comp_into(
    aix: Float32, aiy: Float32, aiz: Float32, aiw: Float32,
    bjx: Float32, bjy: Float32, bjz: Float32, bjw: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32, mut out3: Float32,
):
    out0 = aiw * bjx + aix * bjw + aiy * bjz - aiz * bjy
    out1 = aiw * bjy - aix * bjz + aiy * bjw + aiz * bjx
    out2 = aiw * bjz + aix * bjy - aiy * bjx + aiz * bjw
    out3 = aiw * bjw - aix * bjx - aiy * bjy - aiz * bjz


@always_inline
def act_so3_components(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    x0: Float32, x1: Float32, x2: Float32,
) -> InlineArray[Float32, 3]:
    var uv0 = 2.0 * (qy * x2 - qz * x1)
    var uv1 = 2.0 * (qz * x0 - qx * x2)
    var uv2 = 2.0 * (qx * x1 - qy * x0)
    var out = InlineArray[Float32, 3](fill=0.0)
    out[0] = x0 + qw * uv0 + (qy * uv2 - qz * uv1)
    out[1] = x1 + qw * uv1 + (qz * uv0 - qx * uv2)
    out[2] = x2 + qw * uv2 + (qx * uv1 - qy * uv0)
    return out^


@always_inline
def act_so3_into(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
):
    var uv0 = 2.0 * (qy * x2 - qz * x1)
    var uv1 = 2.0 * (qz * x0 - qx * x2)
    var uv2 = 2.0 * (qx * x1 - qy * x0)
    out0 = x0 + qw * uv0 + (qy * uv2 - qz * uv1)
    out1 = x1 + qw * uv1 + (qz * uv0 - qx * uv2)
    out2 = x2 + qw * uv2 + (qx * uv1 - qy * uv0)


@always_inline
def exp_so3_components(phi0: Float32, phi1: Float32, phi2: Float32) -> InlineArray[Float32, 4]:
    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta_p4 = theta_sq * theta_sq
    var imag: Float32
    var real: Float32
    if theta_sq < 1e-6:
        imag = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
        real = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_p4
    else:
        var theta = sqrt(theta_sq)
        imag = sin(0.5 * theta) / theta
        real = cos(0.5 * theta)
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = imag * phi0
    out[1] = imag * phi1
    out[2] = imag * phi2
    out[3] = real
    return out^


@always_inline
def exp_sim3_components(
    xi0: Float32, xi1: Float32, xi2: Float32, xi3: Float32, xi4: Float32, xi5: Float32, xi6: Float32,
) -> InlineArray[Float32, 8]:
    # Convert a 7D Sim(3) tangent update [translation, rotation, log-scale]
    # into the finite transform used to retract a pose. This is the Exp map in
    # Eq. 1 of the paper.
    var tau0: Float32 = xi0
    var tau1: Float32 = xi1
    var tau2: Float32 = xi2
    var phi0: Float32 = xi3
    var phi1: Float32 = xi4
    var phi2: Float32 = xi5
    var sigma: Float32 = xi6

    var scale = exp(sigma)
    var q = exp_so3_components(phi0, phi1, phi2)

    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta = sqrt(theta_sq)

    var A: Float32
    var B: Float32
    var C: Float32
    if abs(sigma) < 1e-6:
        C = 1.0
        if abs(theta) < 1e-6:
            A = 0.5
            B = 1.0 / 6.0
        else:
            A = (1.0 - cos(theta)) / theta_sq
            B = (theta - sin(theta)) / (theta_sq * theta)
    else:
        C = (scale - 1.0) / sigma
        if abs(theta) < 1e-6:
            var sigma_sq = sigma * sigma
            A = ((sigma - 1.0) * scale + 1.0) / sigma_sq
            B = (scale * 0.5 * sigma_sq + scale - 1.0 - sigma * scale) / (sigma_sq * sigma)
        else:
            var a = scale * sin(theta)
            var b = scale * cos(theta)
            var c = theta_sq + sigma * sigma
            A = (a * sigma + (1.0 - b) * theta) / (theta * c)
            B = (C - (((b - 1.0) * sigma + a * theta) / c)) / theta_sq

    var out = InlineArray[Float32, 8](fill=0.0)
    out[0] = C * tau0
    out[1] = C * tau1
    out[2] = C * tau2

    var cx0: Float32 = phi1 * tau2 - phi2 * tau1
    var cx1: Float32 = phi2 * tau0 - phi0 * tau2
    var cx2: Float32 = phi0 * tau1 - phi1 * tau0
    out[0] += A * cx0
    out[1] += A * cx1
    out[2] += A * cx2

    var c2x0: Float32 = phi1 * cx2 - phi2 * cx1
    var c2x1: Float32 = phi2 * cx0 - phi0 * cx2
    var c2x2: Float32 = phi0 * cx1 - phi1 * cx0
    out[0] += B * c2x0
    out[1] += B * c2x1
    out[2] += B * c2x2
    out[3] = q[0]
    out[4] = q[1]
    out[5] = q[2]
    out[6] = q[3]
    out[7] = scale
    return out^


@always_inline
def quat_inv_components(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
) -> InlineArray[Float32, 4]:
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = -qx
    out[1] = -qy
    out[2] = -qz
    out[3] = qw
    return out^


@always_inline
def act_sim3_components(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
) -> InlineArray[Float32, 3]:
    var rot = act_so3_components(qx, qy, qz, qw, x0, x1, x2)
    var out = InlineArray[Float32, 3](fill=0.0)
    out[0] = rot[0] * s + tx
    out[1] = rot[1] * s + ty
    out[2] = rot[2] * s + tz
    return out^


@always_inline
def act_sim3_into(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
):
    act_so3_into(qx, qy, qz, qw, x0, x1, x2, out0, out1, out2)
    out0 = out0 * s + tx
    out1 = out1 * s + ty
    out2 = out2 * s + tz


@always_inline
def dot3_components(
    a0: Float32, a1: Float32, a2: Float32,
    b0: Float32, b1: Float32, b2: Float32,
) -> Float32:
    return a0 * b0 + a1 * b1 + a2 * b2


@always_inline
def squared_norm3_components(x0: Float32, x1: Float32, x2: Float32) -> Float32:
    return x0 * x0 + x1 * x1 + x2 * x2


@always_inline
def huber_scalar(r: Float32) -> Float32:
    var r_abs = abs(r)
    if r_abs < 1.345:
        return 1.0
    return 1.345 / r_abs


@always_inline
def rel_sim3_components(
    tix: Float32, tiy: Float32, tiz: Float32,
    qix: Float32, qiy: Float32, qiz: Float32, qiw: Float32,
    si: Float32,
    tjx: Float32, tjy: Float32, tjz: Float32,
    qjx: Float32, qjy: Float32, qjz: Float32, qjw: Float32,
    sj: Float32,
) -> InlineArray[Float32, 8]:
    # Compute T_ij = T_WC_i^{-1} * T_WC_j. The backend residuals are defined on
    # relative poses between keyframes, not directly on world poses.
    var si_inv = 1.0 / si
    var qi_inv = quat_inv_components(qix, qiy, qiz, qiw)
    var qij = quat_comp_components(qi_inv[0], qi_inv[1], qi_inv[2], qi_inv[3], qjx, qjy, qjz, qjw)
    var dt0 = tjx - tix
    var dt1 = tjy - tiy
    var dt2 = tjz - tiz
    var rot_t = act_so3_components(qi_inv[0], qi_inv[1], qi_inv[2], qi_inv[3], dt0, dt1, dt2)
    var out = InlineArray[Float32, 8](fill=0.0)
    out[0] = rot_t[0] * si_inv
    out[1] = rot_t[1] * si_inv
    out[2] = rot_t[2] * si_inv
    out[3] = qij[0]
    out[4] = qij[1]
    out[5] = qij[2]
    out[6] = qij[3]
    out[7] = si_inv * sj
    return out^


@always_inline
def rel_sim3_into(
    tix: Float32, tiy: Float32, tiz: Float32,
    qix: Float32, qiy: Float32, qiz: Float32, qiw: Float32,
    si: Float32,
    tjx: Float32, tjy: Float32, tjz: Float32,
    qjx: Float32, qjy: Float32, qjz: Float32, qjw: Float32,
    sj: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
    mut out3: Float32, mut out4: Float32, mut out5: Float32, mut out6: Float32,
    mut out7: Float32,
):
    var si_inv = 1.0 / si
    var qi_inv0 = -qix
    var qi_inv1 = -qiy
    var qi_inv2 = -qiz
    var qi_inv3 = qiw
    quat_comp_into(qi_inv0, qi_inv1, qi_inv2, qi_inv3, qjx, qjy, qjz, qjw, out3, out4, out5, out6)
    var dt0 = tjx - tix
    var dt1 = tjy - tiy
    var dt2 = tjz - tiz
    act_so3_into(qi_inv0, qi_inv1, qi_inv2, qi_inv3, dt0, dt1, dt2, out0, out1, out2)
    out0 = out0 * si_inv
    out1 = out1 * si_inv
    out2 = out2 * si_inv
    out7 = si_inv * sj


@always_inline
def apply_sim3_adj_inv_components(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    x3: Float32, x4: Float32, x5: Float32,
    x6: Float32,
) -> InlineArray[Float32, 7]:
    var s_inv = 1.0 / s
    var ra = act_so3_components(qx, qy, qz, qw, x0, x1, x2)
    var out = InlineArray[Float32, 7](fill=0.0)
    out[0] = s_inv * ra[0]
    out[1] = s_inv * ra[1]
    out[2] = s_inv * ra[2]

    var rb = act_so3_components(qx, qy, qz, qw, x3, x4, x5)
    out[3] = rb[0] + s_inv * (ty * ra[2] - tz * ra[1])
    out[4] = rb[1] + s_inv * (tz * ra[0] - tx * ra[2])
    out[5] = rb[2] + s_inv * (tx * ra[1] - ty * ra[0])
    out[6] = x6 + s_inv * dot3_components(tx, ty, tz, ra[0], ra[1], ra[2])
    return out^


@always_inline
def apply_sim3_adj_inv_into(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    x3: Float32, x4: Float32, x5: Float32,
    x6: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
    mut out3: Float32, mut out4: Float32, mut out5: Float32,
    mut out6: Float32,
):
    var s_inv = 1.0 / s
    var ra0: Float32 = 0.0
    var ra1: Float32 = 0.0
    var ra2: Float32 = 0.0
    act_so3_into(qx, qy, qz, qw, x0, x1, x2, ra0, ra1, ra2)
    out0 = s_inv * ra0
    out1 = s_inv * ra1
    out2 = s_inv * ra2

    var rb0: Float32 = 0.0
    var rb1: Float32 = 0.0
    var rb2: Float32 = 0.0
    act_so3_into(qx, qy, qz, qw, x3, x4, x5, rb0, rb1, rb2)
    out3 = rb0 + s_inv * (ty * ra2 - tz * ra1)
    out4 = rb1 + s_inv * (tz * ra0 - tx * ra2)
    out5 = rb2 + s_inv * (tx * ra1 - ty * ra0)
    out6 = x6 + s_inv * dot3_components(tx, ty, tz, ra0, ra1, ra2)


def gauss_newton_rays_step_kernel(
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
    # One thread block works on one edge-partial pair. Threads stride across
    # the points for that edge, accumulate local Jacobian/Hessian terms, then
    # reduce them once per block.
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
    var pi = ix * POSE_STRIDE
    var pj = jx * POSE_STRIDE
    var pose_shared = stack_allocation[
        24,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    # Cache both endpoint poses once in shared memory so every thread can reuse
    # them while iterating over points on this edge.
    if tid < 8:
        pose_shared[tid] = twc[pi + tid]
    if tid < 8:
        pose_shared[tid + 8] = twc[pj + tid]
    barrier()
    var ti0 = pose_shared[0][0]
    var ti1 = pose_shared[1][0]
    var ti2 = pose_shared[2][0]
    var qi0 = pose_shared[3][0]
    var qi1 = pose_shared[4][0]
    var qi2 = pose_shared[5][0]
    var qi3 = pose_shared[6][0]
    var si0 = pose_shared[7][0]
    var tj0 = pose_shared[8][0]
    var tj1 = pose_shared[9][0]
    var tj2 = pose_shared[10][0]
    var qj0 = pose_shared[11][0]
    var qj1 = pose_shared[12][0]
    var qj2 = pose_shared[13][0]
    var qj3 = pose_shared[14][0]
    var sj0 = pose_shared[15][0]

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
            ti0, ti1, ti2,
            qi0, qi1, qi2, qi3,
            si0,
            tj0, tj1, tj2,
            qj0, qj1, qj2, qj3,
            sj0,
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
    barrier()
    # pose_shared[16:24] now stores the relative Sim(3) for this edge.
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
    var jxv = InlineArray[Float32, 14](fill=0.0)

    var sigma_ray_inv = 1.0 / sigma_ray
    var sigma_dist_inv = 1.0 / sigma_dist

    for k in range(edge_block * block_threads + tid, num_points, block_threads * blocks_per_edge):
        # Each point contributes two residual families from the paper:
        # 1. ray-direction disagreement after transforming j into i,
        # 2. distance disagreement to stabilize low-parallax / pure-rotation
        #    cases where angular error alone is weak.
        var valid_match_ind = valid_match[edge * num_points + k] != 0
        var ind_xi = Int(idx_ii2jj[edge * num_points + k])
        if not valid_match_ind:
            ind_xi = 0

        var xi_base = (ix * num_points + ind_xi) * 3
        var xj_base = (jx * num_points + k) * 3
        var ci = cs[ix * num_points + ind_xi]
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
            rel0, rel1, rel2, rel3, rel4, rel5, rel6, rel7, xj0, xj1, xj2, xj_ci0, xj_ci1, xj_ci2
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

        # Confidence and validity thresholds decide whether MASt3R considers
        # this correspondence trustworthy enough to influence optimization.
        var valid = valid_match_ind and (qv > q_thresh) and (ci > c_thresh) and (cj > c_thresh)
        var conf_weight: Float32 = qv
        var sqrt_w_ray: Float32 = 0.0
        var sqrt_w_dist: Float32 = 0.0
        if valid:
            var sqrt_conf = sqrt(conf_weight)
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

        @parameter
        def accumulate_row(
            err: Float32,
            w: Float32,
            ji0: Float32, ji1: Float32, ji2: Float32, ji3: Float32, ji4: Float32, ji5: Float32, ji6: Float32,
        ):
            # Map the relative-pose Jacobian for this residual row into the two
            # world-pose blocks it touches: pose i gets the negative term and
            # pose j gets the positive term.
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
            jxv[0] = -jadj0
            jxv[1] = -jadj1
            jxv[2] = -jadj2
            jxv[3] = -jadj3
            jxv[4] = -jadj4
            jxv[5] = -jadj5
            jxv[6] = -jadj6
            jxv[7] = jadj0
            jxv[8] = jadj1
            jxv[9] = jadj2
            jxv[10] = jadj3
            jxv[11] = jadj4
            jxv[12] = jadj5
            jxv[13] = jadj6
            var l = 0
            for n in range(14):
                for m in range(n + 1):
                    hij[l] += w * jxv[n] * jxv[m]
                    l += 1
            for n in range(POSE_DIM):
                vi[n] += w * err * jxv[n]
                vj[n] += w * err * jxv[n + 7]

        accumulate_row(err0, w0, drx_dpx, drx_dpy, drx_dpz, 0.0, rj2, -rj1, 0.0)
        accumulate_row(err1, w1, drx_dpy, dry_dpy, dry_dpz, -rj2, 0.0, rj0, 0.0)
        accumulate_row(err2, w2, drx_dpz, dry_dpz, drz_dpz, rj1, -rj0, 0.0, 0.0)
        accumulate_row(err3, w3, rj0, rj1, rj2, 0.0, 0.0, 0.0, norm1_j)

    var warp_partials = stack_allocation[
        RAYS_WARPS,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    # The solver expects the same four 7x7 block layout as the CUDA backend:
    # H_ii, H_ij, H_ji, H_jj and the two gradient blocks g_i, g_j.
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
                if n < 7 and m < 7:
                    hs[((0 * num_partials + partial_edge) * 49) + n * 7 + m] = val
                    hs[((0 * num_partials + partial_edge) * 49) + m * 7 + n] = val
                elif n >= 7 and m < 7:
                    hs[((1 * num_partials + partial_edge) * 49) + m * 7 + (n - 7)] = val
                    hs[((2 * num_partials + partial_edge) * 49) + (n - 7) * 7 + m] = val
                else:
                    hs[((3 * num_partials + partial_edge) * 49) + (n - 7) * 7 + (m - 7)] = val
                    hs[((3 * num_partials + partial_edge) * 49) + (m - 7) * 7 + (n - 7)] = val
            l += 1


def pose_retr_kernel(
    poses: UnsafePointer[Float32, MutAnyOrigin],
    dx: UnsafePointer[Float32, MutAnyOrigin],
    num_fix: Int,
    num_poses: Int,
):
    # Apply the solved Sim(3) increment to every unfixed pose in place. This is
    # the final "retraction" step after solving the linearized GN system.
    var k: Int = Int(global_idx.x) + num_fix
    if k >= num_poses:
        return
    var xi = exp_sim3_components(
        dx[(k - num_fix) * POSE_DIM + 0],
        dx[(k - num_fix) * POSE_DIM + 1],
        dx[(k - num_fix) * POSE_DIM + 2],
        dx[(k - num_fix) * POSE_DIM + 3],
        dx[(k - num_fix) * POSE_DIM + 4],
        dx[(k - num_fix) * POSE_DIM + 5],
        dx[(k - num_fix) * POSE_DIM + 6],
    )
    var p = k * 8
    var rot_t = act_so3_components(
        xi[3], xi[4], xi[5], xi[6],
        poses[p + 0], poses[p + 1], poses[p + 2],
    )
    var q1 = quat_comp_components(
        xi[3], xi[4], xi[5], xi[6],
        poses[p + 3], poses[p + 4], poses[p + 5], poses[p + 6],
    )
    poses[p + 0] = rot_t[0] * xi[7] + xi[0]
    poses[p + 1] = rot_t[1] * xi[7] + xi[1]
    poses[p + 2] = rot_t[2] * xi[7] + xi[2]
    poses[p + 3] = q1[0]
    poses[p + 4] = q1[1]
    poses[p + 5] = q1[2]
    poses[p + 6] = q1[3]
    poses[p + 7] = xi[7] * poses[p + 7]
