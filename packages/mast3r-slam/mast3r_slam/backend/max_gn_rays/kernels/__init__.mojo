from compiler import register
from tensor import InputTensor, OutputTensor, foreach
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList
from std.math import sqrt

from std.gpu import global_idx

from gn_kernels import (
    POSE_DIM,
    RAYS_HDIM,
    act_sim3_into,
    apply_sim3_adj_inv_into,
    huber_scalar,
    rel_sim3_into,
    squared_norm3_components,
)


@always_inline
def accumulate_row_into(
    mut hij: InlineArray[Float32, RAYS_HDIM],
    mut vi: InlineArray[Float32, POSE_DIM],
    mut vj: InlineArray[Float32, POSE_DIM],
    ti0: Float32, ti1: Float32, ti2: Float32,
    qi0: Float32, qi1: Float32, qi2: Float32, qi3: Float32,
    si0: Float32,
    err: Float32,
    w: Float32,
    ji0: Float32, ji1: Float32, ji2: Float32, ji3: Float32, ji4: Float32, ji5: Float32, ji6: Float32,
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
    var jxv = InlineArray[Float32, 14](fill=0.0)
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


def debug_zero_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    var idx = Int(global_idx.x)
    if idx < count:
        dst[idx] = 0.0


def debug_rays_step_kernel(
    twc: UnsafePointer[Float32, MutAnyOrigin],
    xs: UnsafePointer[Float32, MutAnyOrigin],
    cs: UnsafePointer[Float32, MutAnyOrigin],
    ii: UnsafePointer[Int32, MutAnyOrigin],
    jj: UnsafePointer[Int32, MutAnyOrigin],
    idx_ii2jj: UnsafePointer[Int32, MutAnyOrigin],
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
    var partial_edge = Int(global_idx.x)
    var num_partials = num_edges * blocks_per_edge
    if partial_edge >= num_partials:
        return
    var edge = partial_edge // blocks_per_edge
    hs[((0 * num_partials + partial_edge) * 49)] = twc[0] + xs[0] + cs[0] + Float32(ii[edge]) + Float32(jj[edge]) + Float32(idx_ii2jj[edge * num_points]) + Float32(valid_match[edge * num_points]) + q_tensor[edge * num_points] + sigma_ray + sigma_dist + c_thresh + q_thresh
    gs[(0 * num_partials + partial_edge) * POSE_DIM] = hs[((0 * num_partials + partial_edge) * 49)]


@register("debug_zero_pair")
struct DebugZeroPair:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        hs_partial: OutputTensor[dtype=DType.float32, rank=4, ...],
        gs_partial: OutputTensor[dtype=DType.float32, rank=3, ...],
        twc: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        def zero_hs[
            width: Int
        ](idx: IndexList[hs_partial.rank]) -> SIMD[DType.float32, width]:
            return 0.0

        @parameter
        @always_inline
        def zero_gs[
            width: Int
        ](idx: IndexList[gs_partial.rank]) -> SIMD[DType.float32, width]:
            return 0.0

        foreach[zero_hs, target=target](hs_partial, ctx)
        foreach[zero_gs, target=target](gs_partial, ctx)


@register("debug_launch_zero_pair")
struct DebugLaunchZeroPair:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        hs_partial: OutputTensor[dtype=DType.float32, rank=4, ...],
        gs_partial: OutputTensor[dtype=DType.float32, rank=3, ...],
        twc: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("debug_launch_zero_pair only supports gpu target")

        var gpu_ctx = ctx.get_device_context()
        var hs_count = hs_partial.dim_size(0) * hs_partial.dim_size(1) * hs_partial.dim_size(2) * hs_partial.dim_size(3)
        var gs_count = gs_partial.dim_size(0) * gs_partial.dim_size(1) * gs_partial.dim_size(2)
        gpu_ctx.enqueue_function[debug_zero_kernel, debug_zero_kernel](
            hs_partial.unsafe_ptr(),
            hs_count,
            grid_dim=(hs_count + 255) // 256,
            block_dim=256,
        )
        gpu_ctx.enqueue_function[debug_zero_kernel, debug_zero_kernel](
            gs_partial.unsafe_ptr(),
            gs_count,
            grid_dim=(gs_count + 255) // 256,
            block_dim=256,
        )
        gpu_ctx.synchronize()


@register("gauss_newton_rays_step_partial")
struct GaussNewtonRaysStepPartial:
    @staticmethod
    def rays_step_partial_kernel(
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
        var partial_edge = Int(global_idx.x)
        var num_partials = num_edges * blocks_per_edge
        if partial_edge >= num_partials:
            return

        var edge = partial_edge // blocks_per_edge
        var edge_block = partial_edge % blocks_per_edge
        var ix = Int(ii[edge])
        var jx = Int(jj[edge])
        var pi = ix * 8
        var pj = jx * 8

        var ti0 = twc[pi + 0]
        var ti1 = twc[pi + 1]
        var ti2 = twc[pi + 2]
        var qi0 = twc[pi + 3]
        var qi1 = twc[pi + 4]
        var qi2 = twc[pi + 5]
        var qi3 = twc[pi + 6]
        var si0 = twc[pi + 7]
        var tj0 = twc[pj + 0]
        var tj1 = twc[pj + 1]
        var tj2 = twc[pj + 2]
        var qj0 = twc[pj + 3]
        var qj1 = twc[pj + 4]
        var qj2 = twc[pj + 5]
        var qj3 = twc[pj + 6]
        var sj0 = twc[pj + 7]

        var rel0: Float32 = 0.0
        var rel1: Float32 = 0.0
        var rel2: Float32 = 0.0
        var rel3: Float32 = 0.0
        var rel4: Float32 = 0.0
        var rel5: Float32 = 0.0
        var rel6: Float32 = 0.0
        var rel7: Float32 = 0.0
        rel_sim3_into(
            ti0, ti1, ti2,
            qi0, qi1, qi2, qi3,
            si0,
            tj0, tj1, tj2,
            qj0, qj1, qj2, qj3,
            sj0,
            rel0, rel1, rel2, rel3, rel4, rel5, rel6, rel7,
        )

        var hij = InlineArray[Float32, RAYS_HDIM](fill=0.0)
        var vi = InlineArray[Float32, POSE_DIM](fill=0.0)
        var vj = InlineArray[Float32, POSE_DIM](fill=0.0)
        var sigma_ray_inv = 1.0 / sigma_ray
        var sigma_dist_inv = 1.0 / sigma_dist

        for k in range(edge_block, num_points, blocks_per_edge):
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

            var valid = valid_match_ind and (qv > q_thresh) and (ci > c_thresh) and (cj > c_thresh)
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

            accumulate_row_into(hij, vi, vj, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, err0, w0, drx_dpx, drx_dpy, drx_dpz, 0.0, rj2, -rj1, 0.0)
            accumulate_row_into(hij, vi, vj, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, err1, w1, drx_dpy, dry_dpy, dry_dpz, -rj2, 0.0, rj0, 0.0)
            accumulate_row_into(hij, vi, vj, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, err2, w2, drx_dpz, dry_dpz, drz_dpz, rj1, -rj0, 0.0, 0.0)
            accumulate_row_into(hij, vi, vj, ti0, ti1, ti2, qi0, qi1, qi2, qi3, si0, err3, w3, rj0, rj1, rj2, 0.0, 0.0, 0.0, norm1_j)

        for n in range(POSE_DIM):
            gs[(0 * num_partials + partial_edge) * POSE_DIM + n] = vi[n]
            gs[(1 * num_partials + partial_edge) * POSE_DIM + n] = vj[n]

        var l = 0
        for n in range(14):
            for m in range(n + 1):
                var val = hij[l]
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

    @staticmethod
    def execute[
        target: StaticString,
    ](
        hs_partial: OutputTensor[dtype=DType.float32, rank=4, ...],
        gs_partial: OutputTensor[dtype=DType.float32, rank=3, ...],
        twc: InputTensor[dtype=DType.float32, rank=2, ...],
        xs: InputTensor[dtype=DType.float32, rank=3, ...],
        cs: InputTensor[dtype=DType.float32, rank=3, ...],
        ii: InputTensor[dtype=DType.int32, rank=1, ...],
        jj: InputTensor[dtype=DType.int32, rank=1, ...],
        idx_ii2jj: InputTensor[dtype=DType.int32, rank=2, ...],
        valid_match: InputTensor[dtype=DType.uint8, rank=3, ...],
        q_tensor: InputTensor[dtype=DType.float32, rank=3, ...],
        sigma_ray: InputTensor[dtype=DType.float32, rank=1, ...],
        sigma_dist: InputTensor[dtype=DType.float32, rank=1, ...],
        c_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        q_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("gauss_newton_rays_step_partial only supports gpu target")

        var num_edges = ii.dim_size(0)
        if num_edges <= 0:
            return

        var num_points = xs.dim_size(1)
        var num_partials = hs_partial.dim_size(1)
        var blocks_per_edge = num_partials // num_edges
        var zero_idx = IndexList[1](0)
        var sigma_ray_value = sigma_ray.load[1](zero_idx)[0]
        var sigma_dist_value = sigma_dist.load[1](zero_idx)[0]
        var c_thresh_value = c_thresh.load[1](zero_idx)[0]
        var q_thresh_value = q_thresh.load[1](zero_idx)[0]

        @parameter
        @always_inline
        def compute_hs[
            width: Int
        ](idx: IndexList[hs_partial.rank]) -> SIMD[DType.float32, width]:
            var block = Int(idx[0])
            var partial_edge = Int(idx[1])
            var row = Int(idx[2])
            var col = Int(idx[3])
            var edge = partial_edge // blocks_per_edge
            var edge_block = partial_edge % blocks_per_edge
            if edge >= num_edges:
                return 0.0

            var edge_idx = IndexList[1](edge)
            var ix = Int(ii.load[1](edge_idx)[0])
            var jx = Int(jj.load[1](edge_idx)[0])

            var ti0 = twc.load[1](IndexList[2](ix, 0))[0]
            var ti1 = twc.load[1](IndexList[2](ix, 1))[0]
            var ti2 = twc.load[1](IndexList[2](ix, 2))[0]
            var qi0 = twc.load[1](IndexList[2](ix, 3))[0]
            var qi1 = twc.load[1](IndexList[2](ix, 4))[0]
            var qi2 = twc.load[1](IndexList[2](ix, 5))[0]
            var qi3 = twc.load[1](IndexList[2](ix, 6))[0]
            var si0 = twc.load[1](IndexList[2](ix, 7))[0]
            var tj0 = twc.load[1](IndexList[2](jx, 0))[0]
            var tj1 = twc.load[1](IndexList[2](jx, 1))[0]
            var tj2 = twc.load[1](IndexList[2](jx, 2))[0]
            var qj0 = twc.load[1](IndexList[2](jx, 3))[0]
            var qj1 = twc.load[1](IndexList[2](jx, 4))[0]
            var qj2 = twc.load[1](IndexList[2](jx, 5))[0]
            var qj3 = twc.load[1](IndexList[2](jx, 6))[0]
            var sj0 = twc.load[1](IndexList[2](jx, 7))[0]

            var rel0: Float32 = 0.0
            var rel1: Float32 = 0.0
            var rel2: Float32 = 0.0
            var rel3: Float32 = 0.0
            var rel4: Float32 = 0.0
            var rel5: Float32 = 0.0
            var rel6: Float32 = 0.0
            var rel7: Float32 = 0.0
            rel_sim3_into(
                ti0, ti1, ti2,
                qi0, qi1, qi2, qi3,
                si0,
                tj0, tj1, tj2,
                qj0, qj1, qj2, qj3,
                sj0,
                rel0, rel1, rel2, rel3, rel4, rel5, rel6, rel7,
            )

            var sigma_ray_inv = 1.0 / sigma_ray_value
            var sigma_dist_inv = 1.0 / sigma_dist_value
            var total: Float32 = 0.0
            var sign: Float32 = 1.0
            if block == 1 or block == 2:
                sign = -1.0

            for k in range(edge_block, num_points, blocks_per_edge):
                var match_idx = IndexList[3](edge, k, 0)
                var valid_match_ind = valid_match.load[1](match_idx)[0] != 0
                var ind_xi = Int(idx_ii2jj.load[1](IndexList[2](edge, k))[0])
                if not valid_match_ind:
                    ind_xi = 0

                var ci = cs.load[1](IndexList[3](ix, ind_xi, 0))[0]
                var cj = cs.load[1](IndexList[3](jx, k, 0))[0]
                var qv = q_tensor.load[1](match_idx)[0]

                var xi0 = xs.load[1](IndexList[3](ix, ind_xi, 0))[0]
                var xi1 = xs.load[1](IndexList[3](ix, ind_xi, 1))[0]
                var xi2 = xs.load[1](IndexList[3](ix, ind_xi, 2))[0]
                var xj0 = xs.load[1](IndexList[3](jx, k, 0))[0]
                var xj1 = xs.load[1](IndexList[3](jx, k, 1))[0]
                var xj2 = xs.load[1](IndexList[3](jx, k, 2))[0]

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

                var valid = valid_match_ind and (qv > q_thresh_value) and (ci > c_thresh_value) and (cj > c_thresh_value)
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

                var h0: Float32 = 0.0
                var h1: Float32 = 0.0
                var h2: Float32 = 0.0
                var h3: Float32 = 0.0
                var h4: Float32 = 0.0
                var h5: Float32 = 0.0
                var h6: Float32 = 0.0
                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    drx_dpx, drx_dpy, drx_dpz, 0.0, rj2, -rj1, 0.0,
                    h0, h1, h2, h3, h4, h5, h6,
                )
                var h = InlineArray[Float32, POSE_DIM](fill=0.0)
                h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3; h[4] = h4; h[5] = h5; h[6] = h6
                total += sign * w0 * h[row] * h[col]

                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    drx_dpy, dry_dpy, dry_dpz, -rj2, 0.0, rj0, 0.0,
                    h0, h1, h2, h3, h4, h5, h6,
                )
                h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3; h[4] = h4; h[5] = h5; h[6] = h6
                total += sign * w1 * h[row] * h[col]

                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    drx_dpz, dry_dpz, drz_dpz, rj1, -rj0, 0.0, 0.0,
                    h0, h1, h2, h3, h4, h5, h6,
                )
                h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3; h[4] = h4; h[5] = h5; h[6] = h6
                total += sign * w2 * h[row] * h[col]

                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    rj0, rj1, rj2, 0.0, 0.0, 0.0, norm1_j,
                    h0, h1, h2, h3, h4, h5, h6,
                )
                h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3; h[4] = h4; h[5] = h5; h[6] = h6
                total += sign * w3 * h[row] * h[col]
            return total

        @parameter
        @always_inline
        def compute_gs[
            width: Int
        ](idx: IndexList[gs_partial.rank]) -> SIMD[DType.float32, width]:
            var block = Int(idx[0])
            var partial_edge = Int(idx[1])
            var row = Int(idx[2])
            var edge = partial_edge // blocks_per_edge
            var edge_block = partial_edge % blocks_per_edge
            if edge >= num_edges:
                return 0.0

            var edge_idx = IndexList[1](edge)
            var ix = Int(ii.load[1](edge_idx)[0])
            var jx = Int(jj.load[1](edge_idx)[0])

            var ti0 = twc.load[1](IndexList[2](ix, 0))[0]
            var ti1 = twc.load[1](IndexList[2](ix, 1))[0]
            var ti2 = twc.load[1](IndexList[2](ix, 2))[0]
            var qi0 = twc.load[1](IndexList[2](ix, 3))[0]
            var qi1 = twc.load[1](IndexList[2](ix, 4))[0]
            var qi2 = twc.load[1](IndexList[2](ix, 5))[0]
            var qi3 = twc.load[1](IndexList[2](ix, 6))[0]
            var si0 = twc.load[1](IndexList[2](ix, 7))[0]
            var tj0 = twc.load[1](IndexList[2](jx, 0))[0]
            var tj1 = twc.load[1](IndexList[2](jx, 1))[0]
            var tj2 = twc.load[1](IndexList[2](jx, 2))[0]
            var qj0 = twc.load[1](IndexList[2](jx, 3))[0]
            var qj1 = twc.load[1](IndexList[2](jx, 4))[0]
            var qj2 = twc.load[1](IndexList[2](jx, 5))[0]
            var qj3 = twc.load[1](IndexList[2](jx, 6))[0]
            var sj0 = twc.load[1](IndexList[2](jx, 7))[0]

            var rel0: Float32 = 0.0
            var rel1: Float32 = 0.0
            var rel2: Float32 = 0.0
            var rel3: Float32 = 0.0
            var rel4: Float32 = 0.0
            var rel5: Float32 = 0.0
            var rel6: Float32 = 0.0
            var rel7: Float32 = 0.0
            rel_sim3_into(
                ti0, ti1, ti2,
                qi0, qi1, qi2, qi3,
                si0,
                tj0, tj1, tj2,
                qj0, qj1, qj2, qj3,
                sj0,
                rel0, rel1, rel2, rel3, rel4, rel5, rel6, rel7,
            )

            var sigma_ray_inv = 1.0 / sigma_ray_value
            var sigma_dist_inv = 1.0 / sigma_dist_value
            var sign: Float32 = 1.0
            if block == 0:
                sign = -1.0
            var total: Float32 = 0.0

            for k in range(edge_block, num_points, blocks_per_edge):
                var match_idx = IndexList[3](edge, k, 0)
                var valid_match_ind = valid_match.load[1](match_idx)[0] != 0
                var ind_xi = Int(idx_ii2jj.load[1](IndexList[2](edge, k))[0])
                if not valid_match_ind:
                    ind_xi = 0

                var ci = cs.load[1](IndexList[3](ix, ind_xi, 0))[0]
                var cj = cs.load[1](IndexList[3](jx, k, 0))[0]
                var qv = q_tensor.load[1](match_idx)[0]

                var xi0 = xs.load[1](IndexList[3](ix, ind_xi, 0))[0]
                var xi1 = xs.load[1](IndexList[3](ix, ind_xi, 1))[0]
                var xi2 = xs.load[1](IndexList[3](ix, ind_xi, 2))[0]
                var xj0 = xs.load[1](IndexList[3](jx, k, 0))[0]
                var xj1 = xs.load[1](IndexList[3](jx, k, 1))[0]
                var xj2 = xs.load[1](IndexList[3](jx, k, 2))[0]

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

                var valid = valid_match_ind and (qv > q_thresh_value) and (ci > c_thresh_value) and (cj > c_thresh_value)
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

                var g0: Float32 = 0.0
                var g1: Float32 = 0.0
                var g2: Float32 = 0.0
                var g3: Float32 = 0.0
                var g4: Float32 = 0.0
                var g5: Float32 = 0.0
                var g6: Float32 = 0.0
                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    drx_dpx, drx_dpy, drx_dpz, 0.0, rj2, -rj1, 0.0,
                    g0, g1, g2, g3, g4, g5, g6,
                )
                var g = InlineArray[Float32, POSE_DIM](fill=0.0)
                g[0] = g0; g[1] = g1; g[2] = g2; g[3] = g3; g[4] = g4; g[5] = g5; g[6] = g6
                total += sign * w0 * err0 * g[row]

                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    drx_dpy, dry_dpy, dry_dpz, -rj2, 0.0, rj0, 0.0,
                    g0, g1, g2, g3, g4, g5, g6,
                )
                g[0] = g0; g[1] = g1; g[2] = g2; g[3] = g3; g[4] = g4; g[5] = g5; g[6] = g6
                total += sign * w1 * err1 * g[row]

                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    drx_dpz, dry_dpz, drz_dpz, rj1, -rj0, 0.0, 0.0,
                    g0, g1, g2, g3, g4, g5, g6,
                )
                g[0] = g0; g[1] = g1; g[2] = g2; g[3] = g3; g[4] = g4; g[5] = g5; g[6] = g6
                total += sign * w2 * err2 * g[row]

                apply_sim3_adj_inv_into(
                    ti0, ti1, ti2,
                    qi0, qi1, qi2, qi3, si0,
                    rj0, rj1, rj2, 0.0, 0.0, 0.0, norm1_j,
                    g0, g1, g2, g3, g4, g5, g6,
                )
                g[0] = g0; g[1] = g1; g[2] = g2; g[3] = g3; g[4] = g4; g[5] = g5; g[6] = g6
                total += sign * w3 * err3 * g[row]
            return total

        foreach[compute_hs, target=target](hs_partial, ctx)
        foreach[compute_gs, target=target](gs_partial, ctx)


@register("debug_rays_signature_zero")
struct DebugRaysSignatureZero:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        hs_partial: OutputTensor[dtype=DType.float32, rank=4, ...],
        gs_partial: OutputTensor[dtype=DType.float32, rank=3, ...],
        twc: InputTensor[dtype=DType.float32, rank=2, ...],
        xs: InputTensor[dtype=DType.float32, rank=3, ...],
        cs: InputTensor[dtype=DType.float32, rank=3, ...],
        ii: InputTensor[dtype=DType.int32, rank=1, ...],
        jj: InputTensor[dtype=DType.int32, rank=1, ...],
        idx_ii2jj: InputTensor[dtype=DType.int32, rank=2, ...],
        valid_match: InputTensor[dtype=DType.uint8, rank=3, ...],
        q_tensor: InputTensor[dtype=DType.float32, rank=3, ...],
        sigma_ray: InputTensor[dtype=DType.float32, rank=1, ...],
        sigma_dist: InputTensor[dtype=DType.float32, rank=1, ...],
        c_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        q_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("debug_rays_signature_zero only supports gpu target")

        var gpu_ctx = ctx.get_device_context()
        var hs_count = hs_partial.dim_size(0) * hs_partial.dim_size(1) * hs_partial.dim_size(2) * hs_partial.dim_size(3)
        var gs_count = gs_partial.dim_size(0) * gs_partial.dim_size(1) * gs_partial.dim_size(2)
        gpu_ctx.enqueue_function[debug_zero_kernel, debug_zero_kernel](
            hs_partial.unsafe_ptr(),
            hs_count,
            grid_dim=(hs_count + 255) // 256,
            block_dim=256,
        )
        gpu_ctx.enqueue_function[debug_zero_kernel, debug_zero_kernel](
            gs_partial.unsafe_ptr(),
            gs_count,
            grid_dim=(gs_count + 255) // 256,
            block_dim=256,
        )


@register("debug_rays_signature_launch")
struct DebugRaysSignatureLaunch:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        hs_partial: OutputTensor[dtype=DType.float32, rank=4, ...],
        gs_partial: OutputTensor[dtype=DType.float32, rank=3, ...],
        twc: InputTensor[dtype=DType.float32, rank=2, ...],
        xs: InputTensor[dtype=DType.float32, rank=3, ...],
        cs: InputTensor[dtype=DType.float32, rank=3, ...],
        ii: InputTensor[dtype=DType.int32, rank=1, ...],
        jj: InputTensor[dtype=DType.int32, rank=1, ...],
        idx_ii2jj: InputTensor[dtype=DType.int32, rank=2, ...],
        valid_match: InputTensor[dtype=DType.uint8, rank=3, ...],
        q_tensor: InputTensor[dtype=DType.float32, rank=3, ...],
        sigma_ray: InputTensor[dtype=DType.float32, rank=1, ...],
        sigma_dist: InputTensor[dtype=DType.float32, rank=1, ...],
        c_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        q_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("debug_rays_signature_launch only supports gpu target")

        var gpu_ctx = ctx.get_device_context()
        var num_edges = ii.dim_size(0)
        if num_edges <= 0:
            return
        var num_points = xs.dim_size(1)
        var num_partials = hs_partial.dim_size(1)
        var blocks_per_edge = num_partials // num_edges
        var zero_idx = IndexList[1](0)
        gpu_ctx.enqueue_function[
            debug_rays_step_kernel,
            debug_rays_step_kernel,
        ](
            twc.unsafe_ptr(),
            xs.unsafe_ptr(),
            cs.unsafe_ptr(),
            ii.unsafe_ptr(),
            jj.unsafe_ptr(),
            idx_ii2jj.unsafe_ptr(),
            valid_match.unsafe_ptr(),
            q_tensor.unsafe_ptr(),
            hs_partial.unsafe_ptr(),
            gs_partial.unsafe_ptr(),
            num_points,
            num_edges,
            blocks_per_edge,
            sigma_ray.load[1](zero_idx)[0],
            sigma_dist.load[1](zero_idx)[0],
            c_thresh.load[1](zero_idx)[0],
            q_thresh.load[1](zero_idx)[0],
            grid_dim=num_partials,
            block_dim=1,
        )


@register("debug_rays_foreach_probe")
struct DebugRaysForeachProbe:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        hs_partial: OutputTensor[dtype=DType.float32, rank=4, ...],
        gs_partial: OutputTensor[dtype=DType.float32, rank=3, ...],
        twc: InputTensor[dtype=DType.float32, rank=2, ...],
        xs: InputTensor[dtype=DType.float32, rank=3, ...],
        cs: InputTensor[dtype=DType.float32, rank=3, ...],
        ii: InputTensor[dtype=DType.int32, rank=1, ...],
        jj: InputTensor[dtype=DType.int32, rank=1, ...],
        idx_ii2jj: InputTensor[dtype=DType.int32, rank=2, ...],
        valid_match: InputTensor[dtype=DType.uint8, rank=3, ...],
        q_tensor: InputTensor[dtype=DType.float32, rank=3, ...],
        sigma_ray: InputTensor[dtype=DType.float32, rank=1, ...],
        sigma_dist: InputTensor[dtype=DType.float32, rank=1, ...],
        c_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        q_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        def fill_hs[
            width: Int
        ](idx: IndexList[hs_partial.rank]) -> SIMD[DType.float32, width]:
            var edge = Int(idx[1])
            if edge >= ii.dim_size(0):
                return 0.0
            var ix = Int(ii.load[1](IndexList[1](edge))[0])
            return twc.load[1](IndexList[2](ix, 0))[0] + q_tensor.load[1](IndexList[3](edge, 0, 0))[0]

        @parameter
        @always_inline
        def fill_gs[
            width: Int
        ](idx: IndexList[gs_partial.rank]) -> SIMD[DType.float32, width]:
            var edge = Int(idx[1])
            if edge >= ii.dim_size(0):
                return 0.0
            return xs.load[1](IndexList[3](0, 0, 0))[0] + cs.load[1](IndexList[3](0, 0, 0))[0] + sigma_ray.load[1](IndexList[1](0))[0]

        foreach[fill_hs, target=target](hs_partial, ctx)
        foreach[fill_gs, target=target](gs_partial, ctx)
