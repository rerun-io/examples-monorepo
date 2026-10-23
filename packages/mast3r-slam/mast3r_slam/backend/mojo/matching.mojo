"""Matching kernels of the MASt3R-SLAM frontend: a Mojo port of matching_kernels.cu (§3.1-3.2 of arXiv 2412.12392).

Two stages, one thread per query point:
  1. `iter_proj_kernel`: Levenberg-Marquardt search over the pixel (u, v) of image i whose ray, bilinearly sampled from
     the ray field, best aligns with a target 3D direction from image j.
  2. `refine_matches_kernel_f16*`: coarse-to-fine grid search in descriptor space around that pixel.

Tensors are row-major (PyTorch). `rays_img` is [B, H, W, 9]: ray (0..2), d ray/du (3..5), d ray/dv (6..8).
"""

from std.math import ceildiv, floor, max, min, sqrt
from std.python import Python, PythonObject
from std.sys import align_of
from std.utils.numerics import min_finite
from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.memory import AddressSpace
from layout import Idx, TileTensor, row_major, stack_allocation

from gn_kernels import Vec3
from python_interop import cached_context, torch_ptr

comptime RAYS_CHANNELS = 9

comptime RaysLayout = type_of(row_major((0, 0, 0, Idx[RAYS_CHANNELS])))  # [B, H, W, 9]
comptime Points3Layout = type_of(row_major((0, 0, Idx[3])))  # [B, N, 3]
comptime Pixels2Layout = type_of(row_major((0, 0, Idx[2])))  # [B, N, 2]
comptime PerPointLayout = type_of(row_major((0, 0)))  # [B, N]
comptime DescRefLayout = type_of(row_major((0, 0, 0, 0)))  # [B, H, W, fdim]
comptime DescQueryLayout = type_of(row_major((0, 0, 0)))  # [B, N, fdim]
comptime DescRefStaticLayout[fdim: Int] = type_of(row_major((0, 0, 0, Idx[fdim])))
comptime DescQueryStaticLayout[fdim: Int] = type_of(row_major((0, 0, Idx[fdim])))

# Descriptor loads keep the scalar alignment of the old pointer loads: a torch view can start at any element, and
# TileTensor.load would otherwise assume align_of[SIMD[float16, 8]].
comptime F16_ALIGN = align_of[Float16]()


def choose_iter_proj_block(num_pts: Int) -> Int:
    """Small blocks waste fewer threads on small problems, large blocks raise occupancy on large ones."""
    if num_pts <= 4096:
        return 16
    if num_pts <= 16384:
        return 64
    return 128


def choose_refine_block(num_pts: Int) -> Int:
    if num_pts <= 1024:
        return 8
    return 16


# ── Kernel helpers ───────────────────────────────────────────────────────────


@inline(.always)
def bilinear_weights(du: Float32, dv: Float32) -> SIMD[.float32, 4]:
    """[du dv, (1-du) dv, du (1-dv), (1-du) (1-dv)]: each pixel gets the opposite area (as the CUDA kernel)."""
    return SIMD[.float32, 4](du * dv, (1.0 - du) * dv, du * (1.0 - dv), (1.0 - du) * (1.0 - dv))


@inline(.always)
def bilinear_sample(
    rays_img: TileTensor[.float32, RaysLayout, ImmutAnyOrigin], b: Int, v: Int, u: Int, wt: SIMD[.float32, 4], ch: Int
) -> Float32:
    return (
        wt[0] * rays_img[b, v + 1, u + 1, ch]
        + wt[1] * rays_img[b, v + 1, u, ch]
        + wt[2] * rays_img[b, v, u + 1, ch]
        + wt[3] * rays_img[b, v, u, ch]
    )


@inline(.always)
def normalize_ray(r0: Float32, r1: Float32, r2: Float32) -> Vec3:
    var r_inv: Float32 = 1.0 / sqrt(r0 * r0 + r1 * r1 + r2 * r2)
    return Vec3(r0 * r_inv, r1 * r_inv, r2 * r_inv)


# ── Kernels ──────────────────────────────────────────────────────────────────


def iter_proj_kernel(
    rays_img: TileTensor[.float32, RaysLayout, ImmutAnyOrigin],
    pts_3d_norm: TileTensor[.float32, Points3Layout, ImmutAnyOrigin],
    p_init: TileTensor[.float32, Pixels2Layout, ImmutAnyOrigin],
    p_new: TileTensor[.float32, Pixels2Layout, MutAnyOrigin],
    converged: TileTensor[.uint8, PerPointLayout, MutAnyOrigin],
    max_iter: Int32,
    lambda_init: Float32,
    cost_thresh: Float32,
):
    """Levenberg-Marquardt over one pixel per thread: sample and normalise the ray, solve the 2x2 system, keep the step if
    the cost drops (the paper's projective search, Fig. 2)."""
    var n = global_idx.x
    var b = block_idx.y
    if n >= Int(pts_3d_norm.dim[1]()):
        return
    var h = Int(rays_img.dim[1]())
    var w = Int(rays_img.dim[2]())

    # min(max(..)) rather than clamp: a NaN step must land on the lower bound, as CUDA's fmin(fmax(..)) does.
    var u = min(max(p_init[b, n, 0], Float32(1.0)), Float32(w - 2))
    var v = min(max(p_init[b, n, 1], Float32(1.0)), Float32(h - 2))
    var t0 = pts_3d_norm[b, n, 0]
    var t1 = pts_3d_norm[b, n, 1]
    var t2 = pts_3d_norm[b, n, 2]
    var lam = lambda_init

    for _ in range(max_iter):
        var u11 = Int(floor(u))
        var v11 = Int(floor(v))
        var wt = bilinear_weights(u - Float32(u11), v - Float32(v11))
        var ray = Vec3(
            bilinear_sample(rays_img, b, v11, u11, wt, 0),
            bilinear_sample(rays_img, b, v11, u11, wt, 1),
            bilinear_sample(rays_img, b, v11, u11, wt, 2),
        )
        var gx = Vec3(
            bilinear_sample(rays_img, b, v11, u11, wt, 3),
            bilinear_sample(rays_img, b, v11, u11, wt, 4),
            bilinear_sample(rays_img, b, v11, u11, wt, 5),
        )
        var gy = Vec3(
            bilinear_sample(rays_img, b, v11, u11, wt, 6),
            bilinear_sample(rays_img, b, v11, u11, wt, 7),
            bilinear_sample(rays_img, b, v11, u11, wt, 8),
        )
        var r = normalize_ray(ray.x, ray.y, ray.z)
        var e = Vec3(r.x - t0, r.y - t1, r.z - t2)
        var cost = e.squared_norm()

        # 2x2 LM system in (u, v), solved by Cramer's rule.
        var A00 = gx.squared_norm() + lam
        var A01 = gx.dot(gy)
        var A11 = gy.squared_norm() + lam
        var b0 = -e.dot(gx)
        var b1 = -e.dot(gy)
        var det_inv: Float32 = 1.0 / (A00 * A11 - A01 * A01)
        var delta_u = det_inv * (A11 * b0 - A01 * b1)
        var delta_v = det_inv * (-A01 * b0 + A00 * b1)
        var u_new = min(max(u + delta_u, Float32(1.0)), Float32(w - 2))
        var v_new = min(max(v + delta_v, Float32(1.0)), Float32(h - 2))

        # Cost at the candidate pixel.
        var u11n = Int(floor(u_new))
        var v11n = Int(floor(v_new))
        var wtn = bilinear_weights(u_new - Float32(u11n), v_new - Float32(v11n))
        var nr = normalize_ray(
            bilinear_sample(rays_img, b, v11n, u11n, wtn, 0),
            bilinear_sample(rays_img, b, v11n, u11n, wtn, 1),
            bilinear_sample(rays_img, b, v11n, u11n, wtn, 2),
        )
        var ne = Vec3(nr.x - t0, nr.y - t1, nr.z - t2)
        var new_cost = ne.squared_norm()

        if new_cost < cost:
            u = u_new
            v = v_new
            lam *= 0.1
            converged[b, n] = UInt8(1) if new_cost < cost_thresh else UInt8(0)
        else:
            lam *= 10.0
            converged[b, n] = UInt8(1) if cost < cost_thresh else UInt8(0)

    p_new[b, n, 0] = u
    p_new[b, n, 1] = v


def refine_matches_kernel_f16(
    D11: TileTensor[.float16, DescRefLayout, ImmutAnyOrigin],
    D21: TileTensor[.float16, DescQueryLayout, ImmutAnyOrigin],
    p1: TileTensor[.int64, Pixels2Layout, ImmutAnyOrigin],
    p1_new: TileTensor[.int64, Pixels2Layout, MutAnyOrigin],
    radius: Int32,
    dilation_max: Int32,
):
    """Coarse-to-fine descriptor search, any radius / dilation / fdim; float32 dot products of float16 descriptors."""
    var n = global_idx.x
    var b = block_idx.y
    if n >= Int(p1.dim[1]()):
        return
    var h = Int(D11.dim[1]())
    var w = Int(D11.dim[2]())
    var fdim = Int(D11.dim[3]())
    var vec_end = fdim - fdim % 8

    var u0 = Int(p1[b, n, 0])
    var v0 = Int(p1[b, n, 1])
    var max_score = min_finite[DType.float32]()
    var u_new = u0
    var v_new = v0
    for d in range(Int(dilation_max), 0, -1):
        var rd = Int(radius) * d
        var diam = 2 * rd + 1
        for i in range(0, diam, d):
            for j in range(0, diam, d):
                var u_cand = u0 - rd + i
                var v_cand = v0 - rd + j
                if v_cand >= 0 and v_cand < h and u_cand >= 0 and u_cand < w:
                    var score: Float32 = 0.0
                    for k in range(0, vec_end, 8):
                        var v21 = D21.load[width=8, alignment=F16_ALIGN]((b, n, k))
                        var v11 = D11.load[width=8, alignment=F16_ALIGN]((b, v_cand, u_cand, k))
                        score += (v21.cast[.float32]() * v11.cast[.float32]()).reduce_add()
                    for k in range(vec_end, fdim):
                        score += Float32(D21[b, n, k]) * Float32(D11[b, v_cand, u_cand, k])
                    if score > max_score:
                        max_score = score
                        u_new = u_cand
                        v_new = v_cand
        u0 = u_new
        v0 = v_new

    p1_new[b, n, 0] = Int64(u_new)
    p1_new[b, n, 1] = Int64(v_new)


def refine_matches_kernel_f16_cached[FDIM: Int, BLOCK_SIZE: Int](
    D11: TileTensor[.float16, DescRefStaticLayout[FDIM], ImmutAnyOrigin],
    D21: TileTensor[.float16, DescQueryStaticLayout[FDIM], ImmutAnyOrigin],
    p1: TileTensor[.int64, Pixels2Layout, ImmutAnyOrigin],
    p1_new: TileTensor[.int64, Pixels2Layout, MutAnyOrigin],
):
    """The default MASt3R-SLAM case (radius 2, dilation 2): static fdim, unrolled dot product, and the query descriptor
    read once into shared memory."""
    comptime assert FDIM % 8 == 0, "FDIM must be a multiple of 8"
    comptime CHUNKS = FDIM // 8
    var n = global_idx.x
    var b = block_idx.y
    if n >= Int(p1.dim[1]()):
        return
    var h = Int(D11.dim[1]())
    var w = Int(D11.dim[2]())

    # Each thread writes and reads only its own row, so no barrier (threads past num_pts have already returned).
    var q_shared = stack_allocation[.float16, address_space=AddressSpace.SHARED, alignment=align_of[SIMD[.float16, 8]]()](
        row_major[BLOCK_SIZE, FDIM]()
    )
    var tid = thread_idx.x
    comptime for chunk in range(CHUNKS):
        q_shared.store((tid, chunk * 8), D21.load[width=8, alignment=F16_ALIGN]((b, n, chunk * 8)))

    var u0 = Int(p1[b, n, 0])
    var v0 = Int(p1[b, n, 1])
    var max_score = min_finite[DType.float32]()
    var u_new = u0
    var v_new = v0
    for d in range(2, 0, -1):
        var rd = 2 * d
        var diam = 2 * rd + 1
        for i in range(0, diam, d):
            for j in range(0, diam, d):
                var u_cand = u0 - rd + i
                var v_cand = v0 - rd + j
                if v_cand >= 0 and v_cand < h and u_cand >= 0 and u_cand < w:
                    var score: Float32 = 0.0
                    comptime for chunk in range(CHUNKS):
                        var q = q_shared.load[width=8]((tid, chunk * 8))
                        var v11 = D11.load[width=8, alignment=F16_ALIGN]((b, v_cand, u_cand, chunk * 8))
                        score += (q.cast[.float32]() * v11.cast[.float32]()).reduce_add()
                    if score > max_score:
                        max_score = score
                        u_new = u_cand
                        v_new = v_cand
        u0 = u_new
        v0 = v_new

    p1_new[b, n, 0] = Int64(u_new)
    p1_new[b, n, 1] = Int64(v_new)


# ── Python entry points (same signatures as mast3r_slam._backends) ────────────


def iter_proj_py(
    rays_img_obj: PythonObject,
    pts_3d_norm_obj: PythonObject,
    p_init_obj: PythonObject,
    max_iter_obj: PythonObject,
    lambda_init_obj: PythonObject,
    cost_thresh_obj: PythonObject,
) raises -> PythonObject:
    """Returns (p_new [B, N, 2] float32, converged [B, N] bool)."""
    var torch = Python.import_module("torch")
    var rays_img = rays_img_obj.contiguous()
    var pts_3d_norm = pts_3d_norm_obj.contiguous()
    var p_init = p_init_obj.contiguous()
    var batch = Int(py=rays_img.shape[0])
    var h = Int(py=rays_img.shape[1])
    var w = Int(py=rays_img.shape[2])
    var num_pts = Int(py=pts_3d_norm.shape[1])

    # The kernel writes every element. torch stores bool as one byte, so it writes 0/1 straight into a bool tensor.
    var p_new = torch.empty(Python.list(batch, num_pts, 2), device=rays_img.device, dtype=torch.float32)
    var converged = torch.empty(Python.list(batch, num_pts), device=rays_img.device, dtype=torch.bool)

    var block_size = choose_iter_proj_block(num_pts)
    cached_context()[].enqueue_function[iter_proj_kernel](
        TileTensor(torch_ptr[.float32](rays_img), row_major((batch, h, w, Idx[RAYS_CHANNELS]))),
        TileTensor(torch_ptr[.float32](pts_3d_norm), row_major((batch, num_pts, Idx[3]))),
        TileTensor(torch_ptr[.float32](p_init), row_major((batch, num_pts, Idx[2]))),
        TileTensor(torch_ptr[.float32](p_new), row_major((batch, num_pts, Idx[2]))),
        TileTensor(torch_ptr[.uint8](converged), row_major((batch, num_pts))),
        Int32(Int(py=max_iter_obj)),
        Float32(py=lambda_init_obj),
        Float32(py=cost_thresh_obj),
        grid_dim=(ceildiv(num_pts, block_size), batch),
        block_dim=block_size,
    )
    # Device-wide sync: torch's caching allocator must not recycle the outputs before the kernel is done.
    torch.cuda.synchronize()
    return Python.tuple(p_new, converged)


def refine_matches_py(
    D11_obj: PythonObject,
    D21_obj: PythonObject,
    p1_obj: PythonObject,
    radius_obj: PythonObject,
    dilation_max_obj: PythonObject,
) raises -> PythonObject:
    """Returns (p1_new [B, N, 2] int64,). Descriptors are searched as float16 (the pipeline passes .half())."""
    var torch = Python.import_module("torch")
    var p1 = p1_obj.contiguous()
    var D11 = D11_obj.contiguous().half()
    var D21 = D21_obj.contiguous().half()
    var batch = Int(py=D11.shape[0])
    var h = Int(py=D11.shape[1])
    var w = Int(py=D11.shape[2])
    var fdim = Int(py=D11.shape[3])
    var num_pts = Int(py=p1.shape[1])
    var radius = Int(py=radius_obj)
    var dilation_max = Int(py=dilation_max_obj)

    var p1_new = torch.empty(Python.list(batch, num_pts, 2), device=p1.device, dtype=p1.dtype)
    var p1_t = TileTensor(torch_ptr[.int64](p1), row_major((batch, num_pts, Idx[2])))
    var p1_new_t = TileTensor(torch_ptr[.int64](p1_new), row_major((batch, num_pts, Idx[2])))
    var ctx = cached_context()
    var block_size = choose_refine_block(num_pts)
    var grid = (ceildiv(num_pts, block_size), batch)
    # The hot MASt3R-SLAM configurations get a kernel specialised on fdim and the block size.
    if radius == 2 and dilation_max == 2 and block_size == 8 and fdim == 16:
        comptime kernel = refine_matches_kernel_f16_cached[16, 8]
        ctx[].enqueue_function[kernel](
            TileTensor(torch_ptr[.float16](D11), row_major((batch, h, w, Idx[16]))),
            TileTensor(torch_ptr[.float16](D21), row_major((batch, num_pts, Idx[16]))),
            p1_t,
            p1_new_t,
            grid_dim=grid,
            block_dim=block_size,
        )
    elif radius == 2 and dilation_max == 2 and block_size == 16 and fdim == 128:
        comptime kernel = refine_matches_kernel_f16_cached[128, 16]
        ctx[].enqueue_function[kernel](
            TileTensor(torch_ptr[.float16](D11), row_major((batch, h, w, Idx[128]))),
            TileTensor(torch_ptr[.float16](D21), row_major((batch, num_pts, Idx[128]))),
            p1_t,
            p1_new_t,
            grid_dim=grid,
            block_dim=block_size,
        )
    else:
        ctx[].enqueue_function[refine_matches_kernel_f16](
            TileTensor(torch_ptr[.float16](D11), row_major((batch, h, w, fdim))),
            TileTensor(torch_ptr[.float16](D21), row_major((batch, num_pts, fdim))),
            p1_t,
            p1_new_t,
            Int32(radius),
            Int32(dilation_max),
            grid_dim=grid,
            block_dim=block_size,
        )
    # Device-wide sync: torch's caching allocator must not recycle the output before the kernel is done.
    torch.cuda.synchronize()
    return Python.tuple(p1_new)
