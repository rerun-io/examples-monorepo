"""MASt3R-SLAM matching GPU kernels registered as CustomOpLibrary ops.

CustomOpLibrary auto-compiles this file and exposes registered ops to Python.

Registered ops:
  - iter_proj: Levenberg-Marquardt iterative projection (§3.1 of the paper)
"""

from compiler import register
from tensor import InputTensor, OutputTensor
from layout import Layout, LayoutTensor
from std.runtime.asyncrt import DeviceContextPtr
from std.gpu import block_idx, block_dim, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv, sqrt, min, max, floor


# ── Compile-time constants ───────────────────────────────────────────────────
comptime RAYS_CHANNELS = 9


# ── Helpers ──────────────────────────────────────────────────────────────────

@always_inline
def bilinear_weights(du: Float32, dv: Float32) -> InlineArray[Float32, 4]:
    var w = InlineArray[Float32, 4](fill=Float32(0.0))
    w[0] = du * dv
    w[1] = (1.0 - du) * dv
    w[2] = du * (1.0 - dv)
    w[3] = (1.0 - du) * (1.0 - dv)
    return w^


@always_inline
def normalize_ray(r0: Float32, r1: Float32, r2: Float32) -> Vec3:
    var r_inv: Float32 = 1.0 / sqrt(r0 * r0 + r1 * r1 + r2 * r2)
    return Vec3(r0 * r_inv, r1 * r_inv, r2 * r_inv)


def choose_iter_proj_block(num_pts: Int) -> Int:
    if num_pts <= 4096: return 16
    if num_pts <= 16384: return 64
    return 128


def choose_refine_block(num_pts: Int) -> Int:
    if num_pts <= 1024: return 8
    return 16


# ══════════════════════════════════════════════════════════════════════════════
# iter_proj GPU kernel + launcher
# ══════════════════════════════════════════════════════════════════════════════

def iter_proj_gpu_kernel[
    rays_dtype: DType, rays_layout: Layout,
    pts_dtype: DType, pts_layout: Layout,
    pinit_dtype: DType, pinit_layout: Layout,
    pnew_dtype: DType, pnew_layout: Layout,
    conv_dtype: DType, conv_layout: Layout,
](
    rays_img: LayoutTensor[rays_dtype, rays_layout, MutAnyOrigin],
    pts_3d_norm: LayoutTensor[pts_dtype, pts_layout, MutAnyOrigin],
    p_init: LayoutTensor[pinit_dtype, pinit_layout, MutAnyOrigin],
    p_new: LayoutTensor[pnew_dtype, pnew_layout, MutAnyOrigin],
    converged: LayoutTensor[conv_dtype, conv_layout, MutAnyOrigin],
    h: Int, w: Int, num_pts: Int,
    max_iter: Int, lambda_init: Float32, cost_thresh: Float32,
):
    """LM iterative projection kernel. One thread per point."""
    var n: UInt = block_idx.x * block_dim.x + thread_idx.x
    var b: Int = Int(block_idx.y)
    if n >= UInt(num_pts):
        return
    var ni: Int = Int(n)

    var u: Float32 = p_init[b, ni, 0].cast[DType.float32]()[0]
    var v: Float32 = p_init[b, ni, 1].cast[DType.float32]()[0]
    u = min(max(u, Float32(1.0)), Float32(w - 2))
    v = min(max(v, Float32(1.0)), Float32(h - 2))

    var t0: Float32 = pts_3d_norm[b, ni, 0].cast[DType.float32]()[0]
    var t1: Float32 = pts_3d_norm[b, ni, 1].cast[DType.float32]()[0]
    var t2: Float32 = pts_3d_norm[b, ni, 2].cast[DType.float32]()[0]
    var lam: Float32 = lambda_init

    @always_inline
    def _bsample(bi: Int, vi: Int, ui: Int, wt: InlineArray[Float32, 4], ch: Int) capturing -> Float32:
        return (
            wt[0] * rays_img[bi, vi + 1, ui + 1, ch].cast[DType.float32]()[0]
            + wt[1] * rays_img[bi, vi + 1, ui, ch].cast[DType.float32]()[0]
            + wt[2] * rays_img[bi, vi, ui + 1, ch].cast[DType.float32]()[0]
            + wt[3] * rays_img[bi, vi, ui, ch].cast[DType.float32]()[0]
        )

    for _i in range(max_iter):
        var u11 = Int(floor(u))
        var v11 = Int(floor(v))
        var wt = bilinear_weights(u - Float32(u11), v - Float32(v11))

        var ray = Vec3(_bsample(b, v11, u11, wt, 0), _bsample(b, v11, u11, wt, 1), _bsample(b, v11, u11, wt, 2))
        var gx = Vec3(_bsample(b, v11, u11, wt, 3), _bsample(b, v11, u11, wt, 4), _bsample(b, v11, u11, wt, 5))
        var gy = Vec3(_bsample(b, v11, u11, wt, 6), _bsample(b, v11, u11, wt, 7), _bsample(b, v11, u11, wt, 8))

        var r = normalize_ray(ray.x, ray.y, ray.z)
        var e = Vec3(r.x - t0, r.y - t1, r.z - t2)
        var cost = e.squared_norm()

        var A00 = gx.squared_norm() + lam
        var A01 = gx.dot(gy)
        var A11 = gy.squared_norm() + lam
        var b0 = -e.dot(gx)
        var b1 = -e.dot(gy)

        var det_inv = 1.0 / (A00 * A11 - A01 * A01)
        var delta_u = det_inv * (A11 * b0 - A01 * b1)
        var delta_v = det_inv * (-A01 * b0 + A00 * b1)

        var u_new = min(max(u + delta_u, Float32(1.0)), Float32(w - 2))
        var v_new = min(max(v + delta_v, Float32(1.0)), Float32(h - 2))

        var u11n = Int(floor(u_new))
        var v11n = Int(floor(v_new))
        var wtn = bilinear_weights(u_new - Float32(u11n), v_new - Float32(v11n))

        var nr_raw = Vec3(_bsample(b, v11n, u11n, wtn, 0), _bsample(b, v11n, u11n, wtn, 1), _bsample(b, v11n, u11n, wtn, 2))
        var nr = normalize_ray(nr_raw.x, nr_raw.y, nr_raw.z)
        var ne = Vec3(nr.x - t0, nr.y - t1, nr.z - t2)
        var new_cost = ne.squared_norm()

        if new_cost < cost:
            u = u_new
            v = v_new
            lam *= 0.1
            converged[b, ni] = Scalar[conv_dtype](1) if new_cost < cost_thresh else Scalar[conv_dtype](0)
        else:
            lam *= 10.0
            converged[b, ni] = Scalar[conv_dtype](1) if cost < cost_thresh else Scalar[conv_dtype](0)

    p_new[b, ni, 0] = Scalar[pnew_dtype](u)
    p_new[b, ni, 1] = Scalar[pnew_dtype](v)


def iter_proj_launch(
    ctx: DeviceContext,
    rays_img: LayoutTensor,
    pts_3d_norm: LayoutTensor,
    p_init: LayoutTensor,
    p_new: LayoutTensor,
    converged: LayoutTensor,
    h: Int, w: Int, num_pts: Int, batch: Int,
    max_iter: Int, lambda_init: Float32, cost_thresh: Float32,
) raises:
    """Launch iter_proj kernel with compile-time specialized types."""
    comptime kernel = iter_proj_gpu_kernel[
        rays_img.dtype, rays_img.layout,
        pts_3d_norm.dtype, pts_3d_norm.layout,
        p_init.dtype, p_init.layout,
        p_new.dtype, p_new.layout,
        converged.dtype, converged.layout,
    ]
    var block_size = choose_iter_proj_block(num_pts)
    ctx.enqueue_function_experimental[kernel](
        rays_img, pts_3d_norm, p_init, p_new, converged,
        h, w, num_pts, max_iter, lambda_init, cost_thresh,
        grid_dim=(ceildiv(num_pts, block_size), batch),
        block_dim=block_size,
    )


@register("iter_proj")
struct IterProj[max_iter: Int, lambda_init_x1e8: Int, cost_thresh_x1e6: Int]:
    """LM iterative projection (§3.1 of the paper).

    All scalar params are compile-time (CustomOpLibrary only supports tensor args).
    Float scalars are encoded as scaled integers to avoid Float compile-time param issues:
      lambda_init = lambda_init_x1e8 * 1e-8  (e.g. 1 → 1e-8)
      cost_thresh = cost_thresh_x1e6 * 1e-6  (e.g. 1 → 1e-6)
    """

    @staticmethod
    def execute[target: StaticString](
        p_new: OutputTensor[dtype=DType.float32, rank=3, ...],
        converged: OutputTensor[dtype=DType.uint8, rank=2, ...],
        rays_img: InputTensor[dtype=DType.float32, rank=4, ...],
        pts_3d_norm: InputTensor[dtype=DType.float32, rank=3, ...],
        p_init: InputTensor[dtype=DType.float32, rank=3, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var rays_lt = rays_img.to_layout_tensor()
        var pts_lt = pts_3d_norm.to_layout_tensor()
        var pinit_lt = p_init.to_layout_tensor()
        var pnew_lt = p_new.to_layout_tensor()
        var conv_lt = converged.to_layout_tensor()

        var batch = Int(rays_lt.dim[0]())
        var h = Int(rays_lt.dim[1]())
        var w = Int(rays_lt.dim[2]())
        var num_pts = Int(pts_lt.dim[1]())

        # Decode scaled-integer compile-time params back to Float32
        var lambda_init = Float32(Self.lambda_init_x1e8) * 1e-8
        var cost_thresh = Float32(Self.cost_thresh_x1e6) * 1e-6

        var dev_ctx = ctx.get_device_context()
        iter_proj_launch(
            dev_ctx, rays_lt, pts_lt, pinit_lt, pnew_lt, conv_lt,
            h, w, num_pts, batch,
            Self.max_iter, lambda_init, cost_thresh,
        )


# ══════════════════════════════════════════════════════════════════════════════
# refine_matches: coarse-to-fine descriptor search
# ══════════════════════════════════════════════════════════════════════════════

def refine_matches_f16_gpu_kernel[
    p1_dtype: DType, p1_layout: Layout,
    p1_new_dtype: DType, p1_new_layout: Layout,
](
    D11: UnsafePointer[Float16, MutAnyOrigin],
    D21: UnsafePointer[Float16, MutAnyOrigin],
    p1: LayoutTensor[p1_dtype, p1_layout, MutAnyOrigin],
    p1_new: LayoutTensor[p1_new_dtype, p1_new_layout, MutAnyOrigin],
    h: Int, w: Int, num_pts: Int, fdim: Int, radius: Int, dilation_max: Int,
):
    """Descriptor-based match refinement kernel (f16). SIMD-8 vectorized dot product."""
    var n: UInt = block_idx.x * block_dim.x + thread_idx.x
    var b: Int = Int(block_idx.y)
    if n >= UInt(num_pts):
        return
    var ni: Int = Int(n)

    var d11_stride_b: Int = h * w * fdim
    var d11_stride_v: Int = w * fdim
    var d21_stride_b: Int = num_pts * fdim

    var u0: Int = Int(p1[b, ni, 0])
    var v0: Int = Int(p1[b, ni, 1])
    var d21_base: Int = b * d21_stride_b + ni * fdim

    var max_score: Float32 = -3.4028235e+38
    var u_new: Int = u0
    var v_new: Int = v0

    for d in range(dilation_max, 0, -1):
        var rd: Int = radius * d
        var diam: Int = 2 * rd + 1
        var i: Int = 0
        while i < diam:
            var j: Int = 0
            while j < diam:
                var u_cand: Int = u0 - rd + i
                var v_cand: Int = v0 - rd + j
                if v_cand >= 0 and v_cand < h and u_cand >= 0 and u_cand < w:
                    var score: Float32 = 0.0
                    var d11_base: Int = b * d11_stride_b + v_cand * d11_stride_v + u_cand * fdim
                    var k: Int = 0
                    while k + 8 <= fdim:
                        var v21 = D21.load[width=8](d21_base + k)
                        var v11 = D11.load[width=8](d11_base + k)
                        score += (v21.cast[DType.float32]() * v11.cast[DType.float32]()).reduce_add()
                        k += 8
                    while k < fdim:
                        score += Float32(D21[d21_base + k]) * Float32(D11[d11_base + k])
                        k += 1
                    if score > max_score:
                        max_score = score
                        u_new = u_cand
                        v_new = v_cand
                j += d
            i += d
        u0 = u_new
        v0 = v_new

    p1_new[b, ni, 0] = Scalar[p1_new_dtype](u_new)
    p1_new[b, ni, 1] = Scalar[p1_new_dtype](v_new)


def refine_matches_launch(
    ctx: DeviceContext,
    D11: LayoutTensor,
    D21: LayoutTensor,
    p1: LayoutTensor,
    p1_new: LayoutTensor,
    h: Int, w: Int, num_pts: Int, fdim: Int, batch: Int,
    radius: Int, dilation_max: Int,
) raises:
    """Launch refine_matches with compile-time specialized types."""
    # Extract raw f16 pointers for SIMD loads in the kernel
    var d11_ptr = D11.ptr.bitcast[Float16]()
    var d21_ptr = D21.ptr.bitcast[Float16]()

    comptime kernel = refine_matches_f16_gpu_kernel[
        p1.dtype, p1.layout,
        p1_new.dtype, p1_new.layout,
    ]
    var block_size = choose_refine_block(num_pts)
    ctx.enqueue_function_experimental[kernel](
        d11_ptr, d21_ptr, p1, p1_new,
        h, w, num_pts, fdim, radius, dilation_max,
        grid_dim=(ceildiv(num_pts, block_size), batch),
        block_dim=block_size,
    )


@register("refine_matches")
struct RefineMatches[radius: Int, dilation_max: Int]:
    """Coarse-to-fine descriptor refinement (§3.2 of the paper).

    Compile-time params:
        radius: search radius in pixels (typically 3)
        dilation_max: max dilation level (typically 5)
    """

    @staticmethod
    def execute[target: StaticString](
        p1_new: OutputTensor[dtype=DType.int64, rank=3, ...],
        D11: InputTensor[dtype=DType.float16, rank=4, ...],
        D21: InputTensor[dtype=DType.float16, rank=3, ...],
        p1: InputTensor[dtype=DType.int64, rank=3, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var d11_lt = D11.to_layout_tensor()
        var d21_lt = D21.to_layout_tensor()
        var p1_lt = p1.to_layout_tensor()
        var p1_new_lt = p1_new.to_layout_tensor()

        var batch = Int(d11_lt.dim[0]())
        var h = Int(d11_lt.dim[1]())
        var w = Int(d11_lt.dim[2]())
        var fdim = Int(d11_lt.dim[3]())
        var num_pts = Int(p1_lt.dim[1]())

        var dev_ctx = ctx.get_device_context()
        refine_matches_launch(
            dev_ctx, d11_lt, d21_lt, p1_lt, p1_new_lt,
            h, w, num_pts, fdim, batch,
            Self.radius, Self.dilation_max,
        )
