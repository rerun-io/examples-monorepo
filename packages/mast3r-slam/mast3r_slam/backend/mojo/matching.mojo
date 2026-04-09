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

from std.math import ceildiv, sqrt, min, max, floor
from std.gpu import block_idx, block_dim, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.python import Python, PythonObject
from python_interop import (
    get_cached_context_ptr,
    torch_float16_ptr,
    torch_float32_ptr,
    torch_int64_ptr,
    torch_uint8_ptr,
)


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


def choose_refine_block(num_pts: Int, fdim: Int) -> Int:
    """Pick block size for the descriptor refinement kernel."""
    if num_pts <= 1024:
        return 8
    return 16


# ─── GPU Kernel Helpers ───────────────────────────────────────────────────────


@always_inline
def bilinear_sample(
    data: UnsafePointer[Float32, MutAnyOrigin],
    off11: Int, off12: Int, off21: Int, off22: Int,
    w11: Float32, w12: Float32, w21: Float32, w22: Float32,
    ch: Int,
) -> Float32:
    """Sample a single channel from four bilinear corners.

    The corners are named by their bilinear weight, NOT their spatial position
    (pixels are opposite the area weights — same convention as the CUDA kernel).
    """
    return (
        w11 * data[off11 + ch]
        + w12 * data[off12 + ch]
        + w21 * data[off21 + ch]
        + w22 * data[off22 + ch]
    )


@always_inline
def bilinear_corners(
    base_b: Int, rays_stride_v: Int,
    v11: Int, u11: Int,
) -> InlineArray[Int, 4]:
    """Compute the four corner offsets for bilinear interpolation.

    Returns [bottom-right, bottom-left, top-right, top-left] offsets.
    NOTE: pixels opposite the area weights (same as CUDA kernel).
    """
    var offsets = InlineArray[Int, 4](fill=0)
    offsets[0] = base_b + (v11 + 1) * rays_stride_v + (u11 + 1) * 9  # bottom-right
    offsets[1] = base_b + (v11 + 1) * rays_stride_v + u11 * 9        # bottom-left
    offsets[2] = base_b + v11 * rays_stride_v + (u11 + 1) * 9        # top-right
    offsets[3] = base_b + v11 * rays_stride_v + u11 * 9              # top-left
    return offsets^


@always_inline
def bilinear_weights(du: Float32, dv: Float32) -> InlineArray[Float32, 4]:
    """Compute bilinear interpolation weights from fractional pixel offsets."""
    var w = InlineArray[Float32, 4](fill=Float32(0.0))
    w[0] = du * dv
    w[1] = (1.0 - du) * dv
    w[2] = du * (1.0 - dv)
    w[3] = (1.0 - du) * (1.0 - dv)
    return w^


@always_inline
def normalize_ray(
    r0: Float32, r1: Float32, r2: Float32,
) -> InlineArray[Float32, 3]:
    """Normalize a 3D ray direction to unit length."""
    var r_norm: Float32 = sqrt(r0 * r0 + r1 * r1 + r2 * r2)
    var r_inv: Float32 = 1.0 / r_norm
    var out = InlineArray[Float32, 3](fill=Float32(0.0))
    out[0] = r0 * r_inv
    out[1] = r1 * r_inv
    out[2] = r2 * r_inv
    return out^


# ─── GPU Kernels ──────────────────────────────────────────────────────────────

def iter_proj_kernel(
    rays_img: UnsafePointer[Float32, MutAnyOrigin],
    pts_3d_norm: UnsafePointer[Float32, MutAnyOrigin],
    p_init: UnsafePointer[Float32, MutAnyOrigin],
    p_new: UnsafePointer[Float32, MutAnyOrigin],
    converged: UnsafePointer[UInt8, MutAnyOrigin],
    h: Int,
    w: Int,
    num_pts: Int,
    max_iter: Int,
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
    # ── Thread indexing ──
    var n: UInt = block_idx.x * block_dim.x + thread_idx.x
    var b: UInt = block_idx.y

    if n >= UInt(num_pts):
        return

    # ── Strides for row-major tensors ──
    var rays_stride_b: Int = h * w * 9
    var rays_stride_v: Int = w * 9
    var pts_stride_b: Int = num_pts * 3
    var p_stride_b: Int = num_pts * 2
    var conv_stride_b: Int = num_pts

    # ── Batch-invariant base offsets (hoisted out of loop) ──
    var base_b: Int = Int(b) * rays_stride_b
    var conv_idx: Int = Int(b) * conv_stride_b + Int(n)

    # ── Load initial pixel and clamp to valid bilinear range ──
    var u: Float32 = p_init[Int(b) * p_stride_b + Int(n) * 2 + 0]
    var v: Float32 = p_init[Int(b) * p_stride_b + Int(n) * 2 + 1]
    u = min(max(u, Float32(1.0)), Float32(w - 2))
    v = min(max(v, Float32(1.0)), Float32(h - 2))

    # ── Load target point once (loop-invariant) ──
    var pts_base: Int = Int(b) * pts_stride_b + Int(n) * 3
    var t0: Float32 = pts_3d_norm[pts_base + 0]
    var t1: Float32 = pts_3d_norm[pts_base + 1]
    var t2: Float32 = pts_3d_norm[pts_base + 2]

    var lam: Float32 = lambda_init

    for _i in range(max_iter):
        # ── Bilinear interpolation at current pixel ──
        var u11: Int = Int(floor(u))
        var v11: Int = Int(floor(v))
        var wt = bilinear_weights(u - Float32(u11), v - Float32(v11))
        var off = bilinear_corners(base_b, rays_stride_v, v11, u11)

        # Interpolate ray (ch 0-2), gx (ch 3-5), gy (ch 6-8)
        var r0: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 0)
        var r1: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 1)
        var r2: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 2)

        var gx0: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 3)
        var gx1: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 4)
        var gx2: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 5)

        var gy0: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 6)
        var gy1: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 7)
        var gy2: Float32 = bilinear_sample(rays_img, off[0], off[1], off[2], off[3], wt[0], wt[1], wt[2], wt[3], 8)

        # ── Normalize ray ──
        var r = normalize_ray(r0, r1, r2)
        r0 = r[0]
        r1 = r[1]
        r2 = r[2]

        # ── Error and cost ──
        var e0: Float32 = r0 - t0
        var e1: Float32 = r1 - t1
        var e2: Float32 = r2 - t2
        var cost: Float32 = e0 * e0 + e1 * e1 + e2 * e2

        # Build the tiny 2x2 LM system for pixel coordinates (u, v). This is
        # a per-point optimization, not the global GN pose solve used later in
        # the backend.
        var A00: Float32 = gx0 * gx0 + gx1 * gx1 + gx2 * gx2 + lam
        var A01: Float32 = gx0 * gy0 + gx1 * gy1 + gx2 * gy2
        var A11: Float32 = gy0 * gy0 + gy1 * gy1 + gy2 * gy2 + lam
        var b0: Float32 = -(e0 * gx0 + e1 * gx1 + e2 * gx2)
        var b1: Float32 = -(e0 * gy0 + e1 * gy1 + e2 * gy2)

        # ── Solve 2x2 system via Cramer's rule ──
        var det_inv: Float32 = 1.0 / (A00 * A11 - A01 * A01)
        var delta_u: Float32 = det_inv * (A11 * b0 - A01 * b1)
        var delta_v: Float32 = det_inv * (-A01 * b0 + A00 * b1)

        var u_new: Float32 = min(max(u + delta_u, Float32(1.0)), Float32(w - 2))
        var v_new: Float32 = min(max(v + delta_v, Float32(1.0)), Float32(h - 2))

        # ── Evaluate new cost at candidate position ──
        var u11n: Int = Int(floor(u_new))
        var v11n: Int = Int(floor(v_new))
        var wtn = bilinear_weights(u_new - Float32(u11n), v_new - Float32(v11n))
        var offn = bilinear_corners(base_b, rays_stride_v, v11n, u11n)

        var nr0: Float32 = bilinear_sample(rays_img, offn[0], offn[1], offn[2], offn[3], wtn[0], wtn[1], wtn[2], wtn[3], 0)
        var nr1: Float32 = bilinear_sample(rays_img, offn[0], offn[1], offn[2], offn[3], wtn[0], wtn[1], wtn[2], wtn[3], 1)
        var nr2: Float32 = bilinear_sample(rays_img, offn[0], offn[1], offn[2], offn[3], wtn[0], wtn[1], wtn[2], wtn[3], 2)
        var nr = normalize_ray(nr0, nr1, nr2)

        var ne0: Float32 = nr[0] - t0
        var ne1: Float32 = nr[1] - t1
        var ne2: Float32 = nr[2] - t2
        var new_cost: Float32 = ne0 * ne0 + ne1 * ne1 + ne2 * ne2

        # ── Accept/reject step (Levenberg-Marquardt) ──
        if new_cost < cost:
            u = u_new
            v = v_new
            lam *= 0.1
            converged[conv_idx] = UInt8(1) if new_cost < cost_thresh else UInt8(0)
        else:
            lam *= 10.0
            converged[conv_idx] = UInt8(1) if cost < cost_thresh else UInt8(0)

    # ── Write final pixel ──
    var out_idx: Int = Int(b) * p_stride_b + Int(n) * 2
    p_new[out_idx + 0] = u
    p_new[out_idx + 1] = v


def refine_matches_kernel(
    D11: UnsafePointer[Float32, MutAnyOrigin],
    D21: UnsafePointer[Float32, MutAnyOrigin],
    p1: UnsafePointer[Int64, MutAnyOrigin],
    p1_new: UnsafePointer[Int64, MutAnyOrigin],
    h: Int,
    w: Int,
    num_pts: Int,
    fdim: Int,
    radius: Int,
    dilation_max: Int,
):
    """Descriptor-based match refinement kernel (float32 path).

    After projective search gives a plausible pixel, this kernel does a local
    descriptor search around that pixel and keeps the best-scoring candidate.

    The search proceeds in a coarse-to-fine manner:
      - Start at the highest dilation (largest step size), scanning a grid of
        `(2·radius·d + 1)²` candidates spaced `d` pixels apart.
      - Re-centre on the winner, then decrease `d` by 1 and repeat.
      - At dilation=1 the search is pixel-dense within the radius.

    This multi-scale approach matches the CUDA `refine_matches_kernel<scalar_t>`.

    Args:
        D11 — reference image descriptors, shape [B, H, W, fdim]
        D21 — query point descriptors, shape [B, N_pts, fdim]
        p1  — initial pixel locations from iterative projection, [B, N_pts, 2]
        p1_new — OUTPUT: refined pixel locations, [B, N_pts, 2]
    """
    var n: UInt = block_idx.x * block_dim.x + thread_idx.x
    var b: UInt = block_idx.y

    if n >= UInt(num_pts):
        return

    var d11_stride_b: Int = h * w * fdim
    var d11_stride_v: Int = w * fdim
    var d21_stride_b: Int = num_pts * fdim
    var p_stride_b: Int = num_pts * 2

    var p_base: Int = Int(b) * p_stride_b + Int(n) * 2
    var u0: Int = Int(p1[p_base + 0])
    var v0: Int = Int(p1[p_base + 1])

    # Pre-compute query descriptor base (loop-invariant)
    var d21_base: Int = Int(b) * d21_stride_b + Int(n) * fdim

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
                    var d11_base: Int = Int(b) * d11_stride_b + v_cand * d11_stride_v + u_cand * fdim
                    # SIMD-4 vectorized dot product
                    var k: Int = 0
                    while k + 4 <= fdim:
                        var v21 = D21.load[width=4](d21_base + k)
                        var v11 = D11.load[width=4](d11_base + k)
                        score += (v21 * v11).reduce_add()
                        k += 4
                    # Scalar remainder
                    while k < fdim:
                        score += D21[d21_base + k] * D11[d11_base + k]
                        k += 1

                    if score > max_score:
                        max_score = score
                        u_new = u_cand
                        v_new = v_cand

                j += d
            i += d

        u0 = u_new
        v0 = v_new

    p1_new[p_base + 0] = Int64(u_new)
    p1_new[p_base + 1] = Int64(v_new)


def refine_matches_kernel_f16(
    D11: UnsafePointer[Float16, MutAnyOrigin],
    D21: UnsafePointer[Float16, MutAnyOrigin],
    p1: UnsafePointer[Int64, MutAnyOrigin],
    p1_new: UnsafePointer[Int64, MutAnyOrigin],
    h: Int,
    w: Int,
    num_pts: Int,
    fdim: Int,
    radius: Int,
    dilation_max: Int,
):
    """Descriptor-based match refinement kernel (float16 path).

    Reads float16 descriptors directly, accumulates dot products in float32.
    """
    var n: UInt = block_idx.x * block_dim.x + thread_idx.x
    var b: UInt = block_idx.y

    if n >= UInt(num_pts):
        return

    var d11_stride_b: Int = h * w * fdim
    var d11_stride_v: Int = w * fdim
    var d21_stride_b: Int = num_pts * fdim
    var p_stride_b: Int = num_pts * 2

    var p_base: Int = Int(b) * p_stride_b + Int(n) * 2
    var u0: Int = Int(p1[p_base + 0])
    var v0: Int = Int(p1[p_base + 1])

    var d21_base: Int = Int(b) * d21_stride_b + Int(n) * fdim

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
                    # SIMD-8 vectorized dot product (8 × f16 = 128 bits),
                    # accumulated in float32 for precision.
                    var score: Float32 = 0.0
                    var d11_base: Int = Int(b) * d11_stride_b + v_cand * d11_stride_v + u_cand * fdim
                    var k: Int = 0
                    while k + 8 <= fdim:
                        var v21 = D21.load[width=8](d21_base + k)
                        var v11 = D11.load[width=8](d11_base + k)
                        score += (v21.cast[DType.float32]() * v11.cast[DType.float32]()).reduce_add()
                        k += 8
                    # Scalar remainder
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

    p1_new[p_base + 0] = Int64(u_new)
    p1_new[p_base + 1] = Int64(v_new)


def refine_matches_kernel_f16_cached[
    FDIM: Int,
    BLOCK_SIZE: Int,
](
    D11: UnsafePointer[Float16, MutAnyOrigin],
    D21: UnsafePointer[Float16, MutAnyOrigin],
    p1: UnsafePointer[Int64, MutAnyOrigin],
    p1_new: UnsafePointer[Int64, MutAnyOrigin],
    h: Int,
    w: Int,
    num_pts: Int,
):
    """Specialized f16 refinement kernel for the common radius=2, dilation=2 case.

    This is the **fast path** for MASt3R-SLAM's default configuration (16-dim
    or 128-dim descriptors in half precision). Two compile-time optimisations:

    1. **Shared-memory caching**: each thread loads its query descriptor into
       shared memory once (FDIM/8 chunks of 8×f16), then reuses it across all
       ~25 candidate dot products. This eliminates repeated global reads of D21.

    2. **Compile-time unrolling**: `comptime for` over the descriptor chunks
       lets the compiler fully unroll the inner dot product, enabling better
       instruction scheduling.

    The `FDIM` and `BLOCK_SIZE` compile-time parameters are specialised for
    the two common configurations: (16, 8) and (128, 16).
    """
    var n: UInt = block_idx.x * block_dim.x + thread_idx.x
    var b: UInt = block_idx.y

    if n >= UInt(num_pts):
        return

    comptime CHUNKS = FDIM // 8

    var q_shared = stack_allocation[
        BLOCK_SIZE * CHUNKS,
        SIMD[DType.float16, 8],
        address_space=AddressSpace.SHARED,
    ]()

    var d11_stride_b: Int = h * w * FDIM
    var d11_stride_v: Int = w * FDIM
    var d21_stride_b: Int = num_pts * FDIM
    var p_stride_b: Int = num_pts * 2

    var p_base: Int = Int(b) * p_stride_b + Int(n) * 2
    var u0: Int = Int(p1[p_base + 0])
    var v0: Int = Int(p1[p_base + 1])
    var d21_base: Int = Int(b) * d21_stride_b + Int(n) * FDIM
    var q_shared_base: Int = Int(thread_idx.x) * CHUNKS

    comptime for chunk in range(CHUNKS):
        q_shared[q_shared_base + chunk] = D21.load[width=8](
            d21_base + chunk * 8
        )
    barrier()

    var max_score: Float32 = -3.4028235e+38
    var u_new: Int = u0
    var v_new: Int = v0

    for d in range(2, 0, -1):
        var rd: Int = 2 * d
        var diam: Int = 2 * rd + 1
        var i: Int = 0
        while i < diam:
            var j: Int = 0
            while j < diam:
                var u_cand: Int = u0 - rd + i
                var v_cand: Int = v0 - rd + j

                if v_cand >= 0 and v_cand < h and u_cand >= 0 and u_cand < w:
                    var score: Float32 = 0.0
                    var d11_base: Int = Int(b) * d11_stride_b + v_cand * d11_stride_v + u_cand * FDIM
                    comptime for chunk in range(CHUNKS):
                        var q = q_shared[q_shared_base + chunk]
                        var v11 = D11.load[width=8](
                            d11_base + chunk * 8
                        )
                        score += (
                            q.cast[DType.float32]() * v11.cast[DType.float32]()
                        ).reduce_add()

                    if score > max_score:
                        max_score = score
                        u_new = u_cand
                        v_new = v_cand

                j += d
            i += d

        u0 = u_new
        v0 = v_new

    p1_new[p_base + 0] = Int64(u_new)
    p1_new[p_base + 1] = Int64(v_new)


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
    var rays_img: PythonObject = rays_img_obj.contiguous()
    var pts_3d_norm: PythonObject = pts_3d_norm_obj.contiguous()
    var p_init: PythonObject = p_init_obj.contiguous()

    # Extract shapes
    var batch: Int = Int(py=rays_img.shape[0])
    var h: Int = Int(py=rays_img.shape[1])
    var w: Int = Int(py=rays_img.shape[2])
    var num_pts: Int = Int(py=pts_3d_norm.shape[1])
    var max_iter: Int = Int(py=max_iter_obj)
    var lambda_init: Float64 = Float64(py=lambda_init_obj)
    var cost_thresh: Float64 = Float64(py=cost_thresh_obj)

    # No torch.cuda.synchronize() — CUDA stream ordering handles dependencies.

    # Allocate output tensors via torch.empty (kernel writes all positions)
    var p_new: PythonObject = torch.empty(
        Python.list(batch, num_pts, 2),
        device=rays_img.device,
        dtype=torch.float32,
    )
    # Allocate directly as bool — PyTorch stores bool as uint8 internally,
    # so the kernel can write UInt8 (0/1) to it without a .to(bool) cast.
    var converged: PythonObject = torch.empty(
        Python.list(batch, num_pts),
        device=rays_img.device,
        dtype=torch.bool,
    )

    var rays_uptr = torch_float32_ptr(rays_img)
    var pts_uptr = torch_float32_ptr(pts_3d_norm)
    var pinit_uptr = torch_float32_ptr(p_init)
    var pnew_uptr = torch_float32_ptr(p_new)
    var conv_uptr = torch_uint8_ptr(converged)

    var ctx_ptr = get_cached_context_ptr()
    var block_size: Int = choose_iter_proj_block(num_pts)
    var num_blocks_x: Int = ceildiv(num_pts, block_size)
    ctx_ptr[].enqueue_function[iter_proj_kernel, iter_proj_kernel](
        rays_uptr,
        pts_uptr,
        pinit_uptr,
        pnew_uptr,
        conv_uptr,
        h,
        w,
        num_pts,
        max_iter,
        Float32(lambda_init),
        Float32(cost_thresh),
        grid_dim=(num_blocks_x, batch),
        block_dim=block_size,
    )
    # Device-wide sync to ensure Mojo's kernel completes before PyTorch's
    # caching allocator can recycle the output buffer on a subsequent call.
    Python.import_module("torch").cuda.synchronize()
    # PyTorch ops on the same default stream will wait automatically.

    return Python.tuple(p_new, converged)


def refine_matches_py(
    D11_obj: PythonObject,
    D21_obj: PythonObject,
    p1_obj: PythonObject,
    radius_obj: PythonObject,
    dilation_max_obj: PythonObject,
) raises -> PythonObject:
    """Drop-in replacement for mast3r_slam_backends.refine_matches.

    Dispatches to one of three kernel variants based on dtype and parameters:
      1. `refine_matches_kernel_f16_cached` — fast path for f16, radius=2, dilation=2
         with compile-time-specialised fdim (16 or 128).
      2. `refine_matches_kernel_f16` — generic f16 path for other configurations.
      3. `refine_matches_kernel` — float32 fallback (casts from any other dtype).

    All variants implement the same algorithm: coarse-to-fine local descriptor
    search centred on the projective-search result.

    Returns a tuple containing the refined pixel locations tensor.
    """
    var torch = Python.import_module("torch")

    var p1: PythonObject = p1_obj.contiguous()

    # Shapes from original (possibly half) inputs
    var batch: Int = Int(py=D11_obj.shape[0])
    var h: Int = Int(py=D11_obj.shape[1])
    var w: Int = Int(py=D11_obj.shape[2])
    var fdim: Int = Int(py=D11_obj.shape[3])
    var num_pts: Int = Int(py=p1.shape[1])
    var radius: Int = Int(py=radius_obj)
    var dilation_max: Int = Int(py=dilation_max_obj)

    # Allocate output
    var p1_new: PythonObject = torch.empty(
        Python.list(batch, num_pts, 2),
        device=p1.device,
        dtype=p1.dtype,
    )

    var p1_uptr = torch_int64_ptr(p1)
    var p1_new_uptr = torch_int64_ptr(p1_new)

    var ctx_ptr = get_cached_context_ptr()
    var block_size: Int = choose_refine_block(num_pts, fdim)
    var num_blocks_x: Int = ceildiv(num_pts, block_size)

    # Dispatch float16 vs float32 to avoid half→float copy overhead (~7 µs)
    var is_half: Bool = Bool(D11_obj.dtype == torch.float16)

    if is_half:
        var D11: PythonObject = D11_obj.contiguous()
        var D21: PythonObject = D21_obj.contiguous()
        var d11_uptr = torch_float16_ptr(D11)
        var d21_uptr = torch_float16_ptr(D21)

        # Specialize the hot MASt3R-SLAM path so the compiler can unroll the
        # inner dot product and we only read each query descriptor once.
        if radius == 2 and dilation_max == 2 and block_size == 8 and fdim == 16:
            comptime kernel = refine_matches_kernel_f16_cached[16, 8]
            ctx_ptr[].enqueue_function[kernel, kernel](
                d11_uptr,
                d21_uptr,
                p1_uptr,
                p1_new_uptr,
                h, w, num_pts,
                grid_dim=(num_blocks_x, batch),
                block_dim=block_size,
            )
        elif radius == 2 and dilation_max == 2 and block_size == 16 and fdim == 128:
            comptime kernel = refine_matches_kernel_f16_cached[128, 16]
            ctx_ptr[].enqueue_function[kernel, kernel](
                d11_uptr,
                d21_uptr,
                p1_uptr,
                p1_new_uptr,
                h, w, num_pts,
                grid_dim=(num_blocks_x, batch),
                block_dim=block_size,
            )
        else:
            ctx_ptr[].enqueue_function[refine_matches_kernel_f16, refine_matches_kernel_f16](
                d11_uptr,
                d21_uptr,
                p1_uptr,
                p1_new_uptr,
                h, w, num_pts, fdim, radius, dilation_max,
                grid_dim=(num_blocks_x, batch),
                block_dim=block_size,
            )
    else:
        var D11: PythonObject = D11_obj.contiguous().float()
        var D21: PythonObject = D21_obj.contiguous().float()
        var d11_uptr = torch_float32_ptr(D11)
        var d21_uptr = torch_float32_ptr(D21)

        ctx_ptr[].enqueue_function[refine_matches_kernel, refine_matches_kernel](
            d11_uptr,
            d21_uptr,
            p1_uptr,
            p1_new_uptr,
            h, w, num_pts, fdim, radius, dilation_max,
            grid_dim=(num_blocks_x, batch),
            block_dim=block_size,
        )

    # Device-wide sync to ensure Mojo's kernel completes before PyTorch's
    # caching allocator can recycle the output buffer on a subsequent call.
    Python.import_module("torch").cuda.synchronize()
    return Python.tuple(p1_new)
