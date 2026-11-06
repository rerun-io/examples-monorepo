"""Mojo matching kernels used by the MASt3R-SLAM frontend.

This file is a **Mojo port of matching_kernels.cu** — the CUDA kernels that
implement the frontend correspondence pipeline from §3.1–3.2 of the MASt3R-SLAM
paper (arXiv 2412.12392).

The paper's frontend establishes dense correspondences between pairs of keyframes
in two stages:

  1. **Iterative projection** (`iter_proj_kernel`):
     Given a 3D point from view j, find the pixel in view i whose ray best aligns
     with it. This is a per-point Levenberg-Marquardt (LM) optimisation over 2D
     pixel coordinates — conceptually an inverse camera projection using the
     MASt3R ray field instead of a parametric camera model.

  2. **Descriptor refinement** (`refine_matches_kernel`):
     After the projective search gives an approximate pixel, do a local
     grid search in descriptor space to snap to the best-matching pixel within
     a small neighbourhood. This corrects for ray-field noise and discretisation.

Both stages are embarrassingly parallel — each thread handles one query point.
No shared memory or inter-thread communication is needed within a block (except
for the optimised cached variant that preloads query descriptors).

Tensor layouts
--------------
All tensors are row-major (PyTorch default). The ray-field tensor `rays_img` has
shape [batch, H, W, 9] where the 9 channels are:
  [0..2] = ray direction (unnormalised)
  [3..5] = ∂ray/∂u (gradient in the horizontal pixel direction)
  [6..8] = ∂ray/∂v (gradient in the vertical pixel direction)
"""

from std.math import ceildiv, clamp, floor, sqrt
from std.sys import align_of
from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from std.python import Python, PythonObject
from layout import Idx, ImmTileTensor, MutTileTensor, TensorLayout, row_major, stack_allocation
from python_interop import get_cached_context_ptr, torch_imm_tensor, torch_tensor
from gn_kernels import Vec3

# ── Compile-time constants and tensor layouts ────────────────────────────────
comptime RAYS_CHANNELS = 9  # ray(3) + ∂ray/∂u(3) + ∂ray/∂v(3)
comptime DESC_WIDTH = 8  # descriptor SIMD width: 8 × f16 = 128 bits
comptime DescVec = SIMD[.float16, DESC_WIDTH]
# torch storage offsets only guarantee element alignment for descriptor rows.
comptime DESC_ALIGN = align_of[Float16]()

# A `0` extent is a runtime dimension; `Idx[N]` keeps a compile-time extent static.
comptime RAYS_LT = type_of(row_major((0, 0, 0, Idx[RAYS_CHANNELS])))  # [B, H, W, 9]
comptime PTS3_LT = type_of(row_major((0, 0, Idx[3])))                 # [B, N, 3]
comptime PIX2_LT = type_of(row_major((0, 0, Idx[2])))                 # [B, N, 2]
comptime CONV_LT = type_of(row_major((0, 0)))                         # [B, N]
comptime DESC_REF_LT = type_of(row_major((0, 0, 0, 0)))               # [B, H, W, fdim]
comptime DESC_QRY_LT = type_of(row_major((0, 0, 0)))                  # [B, N, fdim]
comptime DescRefLT[fdim: Int] = type_of(row_major((0, 0, 0, Idx[fdim])))  # [B, H, W, FDIM]
comptime DescQryLT[fdim: Int] = type_of(row_major((0, 0, Idx[fdim])))     # [B, N, FDIM]


# ── Block-size heuristics ─────────────────────────────────────────────────────
# These choose the GPU block size based on the problem size. Smaller blocks
# waste fewer threads when num_pts isn't a multiple of the block size; larger
# blocks improve occupancy when there's plenty of work.

def choose_iter_proj_block(num_pts: Int) -> Int:
    """Pick block size for the iterative projection kernel."""
    if num_pts <= 4096:
        return 16
    if num_pts <= 16384:
        return 64
    return 128


def choose_refine_block(num_pts: Int) -> Int:
    """Pick block size for the descriptor refinement kernel."""
    if num_pts <= 1024:
        return 8
    return 16


# ─── GPU Kernel Helpers ───────────────────────────────────────────────────────


@inline(.always)
def bilinear_weights(du: Float32, dv: Float32) -> Array[Float32, 4]:
    """Compute bilinear interpolation weights from fractional pixel offsets.

    Returns [w11, w12, w21, w22] = [du*dv, (1-du)*dv, du*(1-dv), (1-du)*(1-dv)].
    Note: pixels are opposite the area weights (same convention as CUDA kernel).
    """
    var w = Array[Float32, 4](fill=0.0)
    w[0] = du * dv
    w[1] = (1.0 - du) * dv
    w[2] = du * (1.0 - dv)
    w[3] = (1.0 - du) * (1.0 - dv)
    return w^


@inline(.always)
def normalize_ray(ray: Vec3) -> Vec3:
    """Normalize a 3D ray direction to unit length."""
    var r_inv: Float32 = 1.0 / sqrt(ray.squared_norm())
    return ray * r_inv


# ─── GPU Kernels ──────────────────────────────────────────────────────────────

def iter_proj_kernel(
    rays_img: ImmTileTensor[.float32, RAYS_LT, ImmUntrackedOrigin],
    pts_3d_norm: ImmTileTensor[.float32, PTS3_LT, ImmUntrackedOrigin],
    p_init: ImmTileTensor[.float32, PIX2_LT, ImmUntrackedOrigin],
    p_new: MutTileTensor[.float32, PIX2_LT, MutUntrackedOrigin],
    converged: MutTileTensor[.uint8, CONV_LT, MutUntrackedOrigin],
    h_arg: Int64,
    w_arg: Int64,
    num_pts_arg: Int64,
    max_iter_arg: Int64,
    lambda_init: Float32,
    cost_thresh: Float32,
):
    """Levenberg-Marquardt iterative projection kernel.

    Each thread processes one point: bilinear-interpolates rays at the current
    pixel, normalises, computes error against the target 3D direction, builds a
    2x2 LM system, solves for a pixel update, and accepts/rejects based on cost.

    Conceptually this is the paper's local projective search from Fig. 2:
    starting from an initial pixel, repeatedly adjust the pixel so the sampled
    ray in image i aligns with the target 3D direction coming from image j.
    """
    comptime assert rays_img.flat_rank == 4 and pts_3d_norm.flat_rank == 3
    comptime assert p_init.flat_rank == 3 and p_new.flat_rank == 3 and converged.flat_rank == 2
    var n = global_idx.x
    var b = block_idx.y
    if n >= Int(num_pts_arg):
        return
    var u_max = Float32(w_arg - 2)
    var v_max = Float32(h_arg - 2)

    # ── Load initial pixel and clamp to valid bilinear range ──
    var u = clamp(p_init[b, n, 0][0], 1.0, u_max)
    var v = clamp(p_init[b, n, 1][0], 1.0, v_max)

    # ── Load target point once (loop-invariant) ──
    var target = Vec3(pts_3d_norm[b, n, 0][0], pts_3d_norm[b, n, 1][0], pts_3d_norm[b, n, 2][0])

    var lam: Float32 = lambda_init

    @inline(.always)
    def bsample(vi: Int, ui: Int, wt: Array[Float32, 4], ch: Int) {imm rays_img, imm b} -> Float32:
        """Bilinear-sample one channel of `rays_img`."""
        return (
            wt[0] * rays_img[b, vi + 1, ui + 1, ch][0]
            + wt[1] * rays_img[b, vi + 1, ui, ch][0]
            + wt[2] * rays_img[b, vi, ui + 1, ch][0]
            + wt[3] * rays_img[b, vi, ui, ch][0]
        )

    for _ in range(Int(max_iter_arg)):
        # ── Bilinear interpolation at current pixel ──
        var u11 = Int(floor(u))
        var v11 = Int(floor(v))
        var wt = bilinear_weights(u - Float32(u11), v - Float32(v11))

        # Interpolate ray (ch 0-2), gx (ch 3-5), gy (ch 6-8)
        var ray = Vec3(bsample(v11, u11, wt, 0), bsample(v11, u11, wt, 1), bsample(v11, u11, wt, 2))
        var gx = Vec3(bsample(v11, u11, wt, 3), bsample(v11, u11, wt, 4), bsample(v11, u11, wt, 5))
        var gy = Vec3(bsample(v11, u11, wt, 6), bsample(v11, u11, wt, 7), bsample(v11, u11, wt, 8))

        # ── Error and cost ──
        var e = normalize_ray(ray) - target
        var cost = e.squared_norm()

        # Build the tiny 2x2 LM system for pixel coordinates (u, v).
        var A00 = gx.squared_norm() + lam
        var A01 = gx.dot(gy)
        var A11 = gy.squared_norm() + lam
        var b0 = -e.dot(gx)
        var b1 = -e.dot(gy)

        # ── Solve 2x2 system via Cramer's rule ──
        var det_inv = 1.0 / (A00 * A11 - A01 * A01)
        var delta_u = det_inv * (A11 * b0 - A01 * b1)
        var delta_v = det_inv * (-A01 * b0 + A00 * b1)

        var u_new = clamp(u + delta_u, 1.0, u_max)
        var v_new = clamp(v + delta_v, 1.0, v_max)

        # ── Evaluate new cost at candidate position ──
        var u11n = Int(floor(u_new))
        var v11n = Int(floor(v_new))
        var wtn = bilinear_weights(u_new - Float32(u11n), v_new - Float32(v11n))

        var nr_raw = Vec3(bsample(v11n, u11n, wtn, 0), bsample(v11n, u11n, wtn, 1), bsample(v11n, u11n, wtn, 2))
        var new_cost = (normalize_ray(nr_raw) - target).squared_norm()

        # ── Accept/reject step (Levenberg-Marquardt) ──
        if new_cost < cost:
            u = u_new
            v = v_new
            lam *= 0.1
            converged[b, n] = UInt8(1) if new_cost < cost_thresh else UInt8(0)
        else:
            lam *= 10.0
            converged[b, n] = UInt8(1) if cost < cost_thresh else UInt8(0)

    # ── Write final pixel ──
    p_new[b, n, 0] = u
    p_new[b, n, 1] = v


def refine_matches_kernel_f16(
    D11: ImmTileTensor[.float16, DESC_REF_LT, ImmUntrackedOrigin],
    D21: ImmTileTensor[.float16, DESC_QRY_LT, ImmUntrackedOrigin],
    p1: ImmTileTensor[.int64, PIX2_LT, ImmUntrackedOrigin],
    p1_new: MutTileTensor[.int64, PIX2_LT, MutUntrackedOrigin],
    h_arg: Int64,
    w_arg: Int64,
    num_pts_arg: Int64,
    fdim_arg: Int64,
    radius_arg: Int64,
    dilation_max_arg: Int64,
):
    """Descriptor-based match refinement kernel (float16 path).

    Reads float16 descriptors with SIMD-8 loads and accumulates dot products
    in float32 for precision.
    """
    comptime assert D11.flat_rank == 4 and D21.flat_rank == 3
    comptime assert p1.flat_rank == 3 and p1_new.flat_rank == 3
    var n = global_idx.x
    var b = block_idx.y
    if n >= Int(num_pts_arg):
        return
    var h = Int(h_arg)
    var w = Int(w_arg)
    var fdim = Int(fdim_arg)
    var radius = Int(radius_arg)

    var u0 = Int(p1[b, n, 0])
    var v0 = Int(p1[b, n, 1])

    var max_score = Float32.MIN_FINITE
    var u_new = u0
    var v_new = v0

    for d in range(Int(dilation_max_arg), 0, -1):
        var rd = radius * d
        var diam = 2 * rd + 1
        for i in range(0, diam, d):
            for j in range(0, diam, d):
                var u_cand = u0 - rd + i
                var v_cand = v0 - rd + j

                if v_cand >= 0 and v_cand < h and u_cand >= 0 and u_cand < w:
                    var score: Float32 = 0.0
                    var k = 0
                    while k + DESC_WIDTH <= fdim:
                        var v21 = D21.load[width=DESC_WIDTH, alignment=DESC_ALIGN]((b, n, k))
                        var v11 = D11.load[width=DESC_WIDTH, alignment=DESC_ALIGN]((b, v_cand, u_cand, k))
                        score += (v21.cast[.float32]() * v11.cast[.float32]()).reduce_add()
                        k += DESC_WIDTH
                    while k < fdim:
                        score += D21[b, n, k][0].cast[.float32]() * D11[b, v_cand, u_cand, k][0].cast[.float32]()
                        k += 1

                    if score > max_score:
                        max_score = score
                        u_new = u_cand
                        v_new = v_cand

        u0 = u_new
        v0 = v_new

    p1_new[b, n, 0] = Int64(u_new)
    p1_new[b, n, 1] = Int64(v_new)


def refine_matches_kernel_f16_cached[
    FDIM: Int,
    BLOCK_SIZE: Int,
](
    D11: ImmTileTensor[.float16, DescRefLT[FDIM], ImmUntrackedOrigin],
    D21: ImmTileTensor[.float16, DescQryLT[FDIM], ImmUntrackedOrigin],
    p1: ImmTileTensor[.int64, PIX2_LT, ImmUntrackedOrigin],
    p1_new: MutTileTensor[.int64, PIX2_LT, MutUntrackedOrigin],
    h_arg: Int64,
    w_arg: Int64,
    num_pts_arg: Int64,
):
    """Specialized f16 refinement kernel for the common radius=2, dilation=2 case.

    Fast path for MASt3R-SLAM's default config (16- or 128-dim descriptors in
    half precision). Caches each thread's query descriptor in shared memory,
    compile-time-unrolls the inner dot product.
    """
    comptime assert D11.flat_rank == 4 and D21.flat_rank == 3
    comptime assert p1.flat_rank == 3 and p1_new.flat_rank == 3
    comptime assert FDIM % DESC_WIDTH == 0, "FDIM must be a multiple of the descriptor SIMD width"
    var n = global_idx.x
    var b = block_idx.y
    if n >= Int(num_pts_arg):
        return
    var h = Int(h_arg)
    var w = Int(w_arg)
    var tid = thread_idx.x

    comptime CHUNKS = FDIM // DESC_WIDTH
    comptime SHARED_ALIGN = align_of[DescVec]()

    var q_shared = stack_allocation[.float16, address_space=.SHARED, alignment=SHARED_ALIGN](
        row_major[BLOCK_SIZE, FDIM]()
    )

    var u0 = Int(p1[b, n, 0])
    var v0 = Int(p1[b, n, 1])

    # Cache query descriptor in shared memory (one load per thread)
    comptime for chunk in range(CHUNKS):
        q_shared.store[alignment=SHARED_ALIGN](
            (tid, chunk * DESC_WIDTH),
            D21.load[width=DESC_WIDTH, alignment=DESC_ALIGN]((b, n, chunk * DESC_WIDTH)),
        )
    barrier()

    var max_score = Float32.MIN_FINITE
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
                        var q = q_shared.load[width=DESC_WIDTH, alignment=SHARED_ALIGN]((tid, chunk * DESC_WIDTH))
                        var v11 = D11.load[width=DESC_WIDTH, alignment=DESC_ALIGN]((b, v_cand, u_cand, chunk * DESC_WIDTH))
                        score += (q.cast[.float32]() * v11.cast[.float32]()).reduce_add()

                    if score > max_score:
                        max_score = score
                        u_new = u_cand
                        v_new = v_cand

        u0 = u_new
        v0 = v_new

    p1_new[b, n, 0] = Int64(u_new)
    p1_new[b, n, 1] = Int64(v_new)


# ─── Python-facing wrapper functions ──────────────────────────────────────────

def iter_proj_py(
    rays_img_obj: PythonObject,
    pts_3d_norm_obj: PythonObject,
    p_init_obj: PythonObject,
    max_iter_obj: PythonObject,
    lambda_init_obj: PythonObject,
    cost_thresh_obj: PythonObject,
) raises -> PythonObject:
    """Drop-in replacement for mast3r_slam_backends.iter_proj.

    Python owns tensor allocation and shape handling; Mojo owns the hot inner
    loop that updates one pixel hypothesis per query point.
    """
    var torch = Python.import_module("torch")

    # Ensure inputs are contiguous
    var rays_img = rays_img_obj.contiguous()
    var pts_3d_norm = pts_3d_norm_obj.contiguous()
    var p_init = p_init_obj.contiguous()

    # Extract shapes
    var batch = Int(py=rays_img.shape[0])
    var h = Int(py=rays_img.shape[1])
    var w = Int(py=rays_img.shape[2])
    var num_pts = Int(py=pts_3d_norm.shape[1])
    var max_iter = Int(py=max_iter_obj)
    var lambda_init = Float64(py=lambda_init_obj)
    var cost_thresh = Float64(py=cost_thresh_obj)

    # Allocate output tensors via torch.empty (kernel writes all positions)
    var p_new = torch.empty(
        Python.list(batch, num_pts, 2),
        device=rays_img.device,
        dtype=torch.float32,
    )
    # Allocate directly as bool — PyTorch stores bool as uint8 internally,
    # so the kernel can write UInt8 (0/1) to it without a .to(bool) cast.
    var converged = torch.empty(
        Python.list(batch, num_pts),
        device=rays_img.device,
        dtype=torch.bool,
    )

    var ctx_ptr = get_cached_context_ptr()
    var block_size = choose_iter_proj_block(num_pts)
    ctx_ptr[].enqueue_function[iter_proj_kernel](
        torch_imm_tensor[.float32](rays_img, row_major((batch, h, w, Idx[RAYS_CHANNELS]))),
        torch_imm_tensor[.float32](pts_3d_norm, row_major((batch, num_pts, Idx[3]))),
        torch_imm_tensor[.float32](p_init, row_major((batch, num_pts, Idx[2]))),
        torch_tensor[.float32](p_new, row_major((batch, num_pts, Idx[2]))),
        torch_tensor[.uint8](converged, row_major((batch, num_pts))),
        Int64(h),
        Int64(w),
        Int64(num_pts),
        Int64(max_iter),
        Float32(lambda_init),
        Float32(cost_thresh),
        grid_dim=(ceildiv(num_pts, block_size), batch),
        block_dim=block_size,
    )
    # Device-wide sync to ensure Mojo's kernel completes before PyTorch's
    # caching allocator can recycle the output buffer on a subsequent call.
    torch.cuda.synchronize()

    return Python.tuple(p_new, converged)


def refine_matches_py(
    D11_obj: PythonObject,
    D21_obj: PythonObject,
    p1_obj: PythonObject,
    radius_obj: PythonObject,
    dilation_max_obj: PythonObject,
) raises -> PythonObject:
    """Drop-in replacement for mast3r_slam_backends.refine_matches.

    Dispatches to one of two kernel variants:
      1. `refine_matches_kernel_f16_cached` — fast path for radius=2, dilation=2
         with compile-time-specialised fdim (16 or 128).
      2. `refine_matches_kernel_f16` — generic path for other configurations.
    Descriptors are cast to float16 first if they arrive in another dtype.

    All variants implement the same algorithm: coarse-to-fine local descriptor
    search centred on the projective-search result.

    Returns a tuple containing the refined pixel locations tensor.
    """
    var torch = Python.import_module("torch")

    var p1 = p1_obj.contiguous()

    # Shapes from original (possibly half) inputs
    var batch = Int(py=D11_obj.shape[0])
    var h = Int(py=D11_obj.shape[1])
    var w = Int(py=D11_obj.shape[2])
    var fdim = Int(py=D11_obj.shape[3])
    var num_pts = Int(py=p1.shape[1])
    var radius = Int(py=radius_obj)
    var dilation_max = Int(py=dilation_max_obj)

    # Allocate output
    var p1_new = torch.empty(
        Python.list(batch, num_pts, 2),
        device=p1.device,
        dtype=p1.dtype,
    )
    var p1_t = torch_imm_tensor[.int64](p1, row_major((batch, num_pts, Idx[2])))
    var p1_new_t = torch_tensor[.int64](p1_new, row_major((batch, num_pts, Idx[2])))

    var ctx_ptr = get_cached_context_ptr()
    var block_size = choose_refine_block(num_pts)
    var grid = (ceildiv(num_pts, block_size), batch)

    # The production pipeline always passes .half() descriptors. We only
    # support f16 — casting to f16 here if needed for robustness.
    var D11 = D11_obj.contiguous().half()
    var D21 = D21_obj.contiguous().half()

    # Specialize the hot MASt3R-SLAM path so the compiler can unroll the
    # inner dot product and we only read each query descriptor once.
    if radius == 2 and dilation_max == 2 and block_size == 8 and fdim == 16:
        ctx_ptr[].enqueue_function[refine_matches_kernel_f16_cached[16, 8]](
            torch_imm_tensor[.float16](D11, row_major((batch, h, w, Idx[16]))),
            torch_imm_tensor[.float16](D21, row_major((batch, num_pts, Idx[16]))),
            p1_t,
            p1_new_t,
            Int64(h), Int64(w), Int64(num_pts),
            grid_dim=grid,
            block_dim=block_size,
        )
    elif radius == 2 and dilation_max == 2 and block_size == 16 and fdim == 128:
        ctx_ptr[].enqueue_function[refine_matches_kernel_f16_cached[128, 16]](
            torch_imm_tensor[.float16](D11, row_major((batch, h, w, Idx[128]))),
            torch_imm_tensor[.float16](D21, row_major((batch, num_pts, Idx[128]))),
            p1_t,
            p1_new_t,
            Int64(h), Int64(w), Int64(num_pts),
            grid_dim=grid,
            block_dim=block_size,
        )
    else:
        ctx_ptr[].enqueue_function[refine_matches_kernel_f16](
            torch_imm_tensor[.float16](D11, row_major((batch, h, w, fdim))),
            torch_imm_tensor[.float16](D21, row_major((batch, num_pts, fdim))),
            p1_t,
            p1_new_t,
            Int64(h), Int64(w), Int64(num_pts), Int64(fdim), Int64(radius), Int64(dilation_max),
            grid_dim=grid,
            block_dim=block_size,
        )

    # Device-wide sync to ensure Mojo's kernel completes before PyTorch's
    # caching allocator can recycle the output buffer on a subsequent call.
    torch.cuda.synchronize()
    return Python.tuple(p1_new)
