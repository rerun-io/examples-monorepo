"""Image-GS's 2D Gaussian kernels in hexel-kernel (`gsplat/cuda/csrc`: `foward2d.cu`, `backward2d.cu`, `forward.cu`
`nd_rasterize_forward_topk_norm`, `backward.cu` `nd_rasterize_backward_topk_norm_kernel`), which the `imagegs-2d`
baseline (`hexels-fit-2d --variant imagegs`) runs every step.

Projection: a Gaussian (center in [0, 1]^2, scales, rotation) becomes a pixel center, the inverse covariance
(conic), a 3-sigma radius and the number of 16x16 tiles it touches. Rasterization: per pixel, the 10 Gaussians of
largest exp(-sigma) among the tile's, normalized by their sum + 1e-4. Projection and the forward pass write every
float op as nvcc's fast-math PTX, in its order (`fastmath`; glm's 2x2 products, multiplications by zero included),
so they match the CUDA build bit for bit; the projection backward has no atomics and matches too. The rasterizer
backward scatters with float atomics (scheduling-order sums).
"""

from layout import row_major
from layout import stack_allocation as tile_allocation
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from std.math import ceil, ceildiv, clamp, isinf, isnan
from std.memory import stack_allocation
from std.utils.static_tuple import StaticTuple

import fastmath as fm
from raster2d import BLOCK_SIZE, EPS, MAX_CHANNELS, TILE, TOP_K
from tensors import Grid, In, Out, Rows, atomic_add

comptime POINT_THREADS = 256


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def project_forward_kernel(
    num_points: Int32,
    width: Int32,
    height: Int32,
    tiles_x: Int32,
    tiles_y: Int32,
    means: In[.float32, Rows[2]],
    scales: In[.float32, Rows[2]],
    rotation: In[.float32, Rows[1]],
    xys: Out[.float32, Rows[2]],
    radii: Out[.int32, Rows[1]],
    conics: Out[.float32, Rows[3]],
    num_tiles_hit: Out[.int32, Rows[1]],
):
    var idx = global_idx.x
    if idx >= Int(num_points):
        return
    radii[idx, 0] = 0
    num_tiles_hit[idx, 0] = 0
    var cx = fm.mul(means[idx, 0][0], Float32(width))
    var cy = fm.mul(means[idx, 1][0], Float32(height))
    var rot = rotation[idx, 0][0]
    var c = fm.cos(rot)
    var s = fm.sin(rot)
    var sx = scales[idx, 0][0]
    var sy = scales[idx, 1][0]
    # M = R S (glm, column-major), then cov = M M^T.
    var s0 = fm.mul(s, 0.0)
    var m00 = fm.fma(c, sx, s0)
    var c0 = fm.mul(c, 0.0)
    var m01 = fm.sub(c0, fm.mul(s, sx))
    var m10 = fm.fma(s, sy, c0)
    var m11 = fm.sub(fm.mul(c, sy), s0)
    var cov_x = fm.fma(m00, m00, fm.mul(m10, m10))
    var cov_y = fm.fma(m01, m00, fm.mul(m11, m10))
    var cov_z = fm.fma(m01, m01, fm.mul(m11, m11))
    var det = fm.sub(fm.mul(cov_x, cov_z), fm.mul(cov_y, cov_y))
    if det == 0.0:
        return
    var inv_det = fm.rcp(det)
    var b = fm.mul(fm.add(cov_x, cov_z), 0.5)
    var root = fm.sqrt(max(fm.sub(fm.mul(b, b), det), Float32(0.1)))
    var radius = ceil(fm.mul(fm.sqrt(max(fm.add(b, root), fm.sub(b, root))), 3.0))
    conics[idx, 0] = fm.mul(cov_z, inv_det)
    conics[idx, 1] = fm.mul(inv_det, -cov_y)
    conics[idx, 2] = fm.mul(cov_x, inv_det)
    xys[idx, 0] = cx
    xys[idx, 1] = cy
    radii[idx, 0] = Int32(radius)
    var tcx = fm.div(cx, Float32(TILE))
    var tcy = fm.div(cy, Float32(TILE))
    var tr = fm.div(radius, Float32(TILE))
    var x0 = clamp(Int(fm.sub(tcx, tr)), 0, Int(tiles_x))
    var x1 = clamp(Int(fm.add(fm.add(tcx, tr), 1.0)), 0, Int(tiles_x))
    var y0 = clamp(Int(fm.sub(tcy, tr)), 0, Int(tiles_y))
    var y1 = clamp(Int(fm.add(fm.add(tcy, tr), 1.0)), 0, Int(tiles_y))
    var area = (x1 - x0) * (y1 - y0)
    if area > 0:
        num_tiles_hit[idx, 0] = Int32(area)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def project_backward_kernel(
    num_points: Int32,
    width: Int32,
    height: Int32,
    scales: In[.float32, Rows[2]],
    rotation: In[.float32, Rows[1]],
    radii: In[.int32, Rows[1]],
    conics: In[.float32, Rows[3]],
    v_xy: In[.float32, Rows[2]],
    v_conic: In[.float32, Rows[3]],
    v_cov2d: Out[.float32, Rows[3]],
    v_mean: Out[.float32, Rows[2]],
    v_scale: Out[.float32, Rows[2]],
    v_rot: Out[.float32, Rows[1]],
):
    """Op for op as nvcc's PTX of `project_gaussians_2d_scale_rot_backward_kernel` (register names in comments)."""
    var idx = global_idx.x
    if idx >= Int(num_points) or radii[idx, 0][0] <= 0:
        return
    # v_Sigma = -X G X with X the conic, G = dL/dconic (cov2d_to_conic_vjp).
    var f1 = conics[idx, 0][0]
    var f2 = conics[idx, 1][0]
    var f3 = conics[idx, 2][0]
    var f4 = v_conic[idx, 0][0]
    var f5 = v_conic[idx, 1][0]
    var f6 = v_conic[idx, 2][0]
    var f11 = fm.sub(fm.mul(f5, -f2), fm.mul(f1, f4))
    var f14 = fm.sub(fm.mul(f5, -f3), fm.mul(f2, f4))
    var f17 = fm.sub(fm.mul(f6, -f2), fm.mul(f1, f5))
    var f20 = fm.sub(fm.mul(f6, -f3), fm.mul(f2, f5))
    var f22 = fm.fma(f1, f11, fm.mul(f2, f17))
    var f24 = fm.fma(f1, f14, fm.mul(f2, f20))
    var f26 = fm.fma(f2, f11, fm.mul(f3, f17))
    var f28 = fm.fma(f2, f14, fm.mul(f3, f20))
    var f29 = fm.add(f26, f24)
    v_cov2d[idx, 0] = f22
    v_cov2d[idx, 1] = f29
    v_cov2d[idx, 2] = f28
    # Rotation and scale gradients through M = R S (glm's full 2x2 products).
    var rot = rotation[idx, 0][0]
    var f31 = fm.cos(rot)
    var f32 = fm.sin(rot)
    var f34 = scales[idx, 0][0]
    var f35 = scales[idx, 1][0]
    var f36 = fm.mul(f31, f34)
    var f37 = fm.fma(f32, 0.0, f36)
    var f38 = fm.mul(f31, 0.0)
    var f40 = fm.sub(f38, fm.mul(f32, f34))
    var f41 = fm.mul(f32, f35)
    var f42 = fm.add(f38, f41)
    var f43 = fm.mul(f32, -0.0)
    var f44 = fm.fma(f31, f35, f43)
    var f45 = fm.sub(f43, f36)
    var f47 = fm.sub(fm.mul(f31, -0.0), f41)
    var f49 = fm.fma(f40, f37, fm.mul(f44, f42))
    var f51 = fm.fma(f45, f37, fm.mul(f47, f42))
    var f53 = fm.fma(f45, f40, fm.mul(f47, f44))
    var f55 = fm.fma(f34, f37, fm.mul(f42, 0.0))
    var f57 = fm.fma(f34, f40, fm.mul(f44, 0.0))
    var f59 = fm.fma(f37, 0.0, fm.mul(f35, f42))
    var f61 = fm.fma(f40, 0.0, fm.mul(f35, f44))
    var f64 = fm.sub(fm.mul(f31, f59), fm.mul(f32, f55))
    var f67 = fm.sub(fm.mul(f31, f61), fm.mul(f32, f57))
    var f70 = fm.sub(fm.mul(f61, -f32), fm.mul(f31, f57))
    var f71 = fm.add(f49, f64)
    var f72 = fm.add(f51, f67)
    var f73 = fm.add(f53, f70)
    var f74 = fm.add(f34, f34)
    var f75 = fm.add(f35, f35)
    var f77 = fm.fma(f32, 0.0, fm.mul(f31, f74))
    var f79 = fm.sub(f38, fm.mul(f32, f74))
    var f80 = fm.fma(f32, 0.0, f38)
    var f81 = fm.add(f38, f43)
    var f83 = fm.fma(f32, f80, fm.mul(f31, f77))
    var f84 = fm.mul(f32, f81)
    var f85 = fm.fma(f31, f79, f84)
    var f86 = fm.mul(f31, f81)
    var f88 = fm.sub(f86, fm.mul(f32, f79))
    var f89 = fm.fma(f32, f75, f38)
    var f90 = fm.fma(f31, f75, f43)
    var f92 = fm.fma(f31, f80, fm.mul(f32, f89))
    var f93 = fm.fma(f32, f90, f86)
    var f95 = fm.sub(fm.mul(f31, f90), f84)
    var f96 = fm.add(f29, f29)
    v_scale[idx, 0] = fm.fma(f28, f88, fm.fma(f22, f83, fm.mul(f96, f85)))
    v_scale[idx, 1] = fm.fma(f28, f95, fm.fma(f22, f92, fm.mul(f96, f93)))
    v_rot[idx, 0] = fm.fma(f28, f73, fm.fma(f22, f71, fm.mul(f96, f72)))
    v_mean[idx, 0] = fm.mul(v_xy[idx, 0][0], Float32(width))
    v_mean[idx, 1] = fm.mul(v_xy[idx, 1][0], Float32(height))


@inline(.always)
def _sigma(cx: Float32, cy: Float32, cz: Float32, dx: Float32, dy: Float32) -> Float32:
    """0.5 (cx dx^2 + cz dy^2) + cy dx dy, as nvcc fuses it."""
    return fm.fma(fm.mul(cy, dx), dy, fm.mul(fm.fma(dx, fm.mul(cx, dx), fm.mul(dy, fm.mul(cz, dy))), 0.5))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def forward_kernel(
    tiles_x: Int32,
    width: Int32,
    height: Int32,
    ids_sorted: In[.int32, Rows[1]],
    tile_bins: In[.int32, Rows[2]],
    xys: In[.float32, Rows[2]],
    conics: In[.float32, Rows[3]],
    colors: In[.float32, Grid],
    out_img: Out[.float32, Grid],
    pixel_topk: Out[.int32, Rows[TOP_K]],
):
    var w = Int(width)
    var ch = Int(colors.dim[1]())
    var tile_id = block_idx.y * Int(tiles_x) + block_idx.x
    var i = block_idx.y * TILE + thread_idx.y
    var j = block_idx.x * TILE + thread_idx.x
    var px = Float32(j)
    var py = Float32(i)
    var pix = i * w + j
    var inside = i < Int(height) and j < w
    var start = Int(tile_bins[tile_id, 0][0])
    var end = Int(tile_bins[tile_id, 1][0])
    var id_batch = tile_allocation[.int32, address_space=.SHARED](row_major[BLOCK_SIZE]())
    var param_batch = tile_allocation[.float32, address_space=.SHARED](row_major[BLOCK_SIZE, 5]())  # x, y, conic
    var tr = thread_idx.y * TILE + thread_idx.x
    var topk_ids = stack_allocation[TOP_K, Int32]()
    var topk_val = stack_allocation[TOP_K, Float32]()
    comptime for k in range(TOP_K):
        topk_ids[unsafe_offset=k] = -1
        topk_val[unsafe_offset=k] = 0.0
    var num_batches = ceildiv(end - start, BLOCK_SIZE)
    for b in range(num_batches):
        barrier()
        var batch_start = start + BLOCK_SIZE * b
        var idx = batch_start + tr
        if idx < end:
            var g = Int(ids_sorted[idx, 0][0])
            id_batch[tr] = Int32(g)
            param_batch[tr, 0] = xys[g, 0][0]
            param_batch[tr, 1] = xys[g, 1][0]
            param_batch[tr, 2] = conics[g, 0][0]
            param_batch[tr, 3] = conics[g, 1][0]
            param_batch[tr, 4] = conics[g, 2][0]
        barrier()
        if not inside:
            continue
        for t in range(min(BLOCK_SIZE, end - batch_start)):
            var sigma = _sigma(
                param_batch[t, 2][0], param_batch[t, 3][0], param_batch[t, 4][0], fm.sub(param_batch[t, 0][0], px), fm.sub(param_batch[t, 1][0], py)
            )
            if not (sigma < 0.0 or isnan(sigma) or isinf(sigma)):  # an if, not continue: lanes reconverge per Gaussian
                var alpha = fm.ex2(fm.mul(sigma, -fm.LOG2E))
                var min_id = -1
                var min_val: Float32 = 1e30
                for k in range(TOP_K):
                    if topk_ids[unsafe_offset=k] < 0:
                        min_id = k
                        min_val = -1.0
                        break
                    var v = topk_val[unsafe_offset=k]
                    if v < min_val:
                        min_id = k
                        min_val = v
                if alpha > min_val:
                    topk_ids[unsafe_offset=min_id] = id_batch[t][0]
                    topk_val[unsafe_offset=min_id] = alpha
    if not inside:
        return
    var sum_val: Float32 = 0.0
    comptime for k in range(TOP_K):
        if topk_ids[unsafe_offset=k] >= 0:
            sum_val = fm.add(sum_val, topk_val[unsafe_offset=k])
    var denom = Float32(Float64(sum_val) + EPS)
    var pix_out = stack_allocation[MAX_CHANNELS, Float32]()
    for c in range(ch):
        pix_out[unsafe_offset=c] = 0.0
    for c in range(ch):
        comptime for k in range(TOP_K):
            var g = Int(topk_ids[unsafe_offset=k])
            if g >= 0:
                var vis = fm.div(topk_val[unsafe_offset=k], denom)
                pix_out[unsafe_offset=c] = fm.fma(vis, colors[g, c][0], pix_out[unsafe_offset=c])
    for c in range(ch):
        out_img[pix, c] = pix_out[unsafe_offset=c]
    comptime for k in range(TOP_K):
        pixel_topk[pix, k] = topk_ids[unsafe_offset=k]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def backward_kernel(
    width: Int32,
    height: Int32,
    xys: In[.float32, Rows[2]],
    conics: In[.float32, Rows[3]],
    colors: In[.float32, Grid],
    v_output: In[.float32, Grid],
    pixel_topk: In[.int32, Rows[TOP_K]],
    v_xy: Out[.float32, Rows[2]],
    v_conic: Out[.float32, Rows[3]],
    v_colors: Out[.float32, Grid],
):
    var w = Int(width)
    var ch = Int(colors.dim[1]())
    var i = global_idx.y
    var j = global_idx.x
    if i >= Int(height) or j >= w:
        return
    var px = Float32(j)
    var py = Float32(i)
    var pix = i * w + j
    var topk = SIMD[.int32, 16](-1)
    comptime for k in range(TOP_K):
        topk[k] = pixel_topk[pix, k][0]
    var d_local = SIMD[.float32, 16](0.0)
    var denom = Float32(EPS)
    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var d = fm.expf(
                -_sigma(conics[g, 0][0], conics[g, 1][0], conics[g, 2][0], xys[g, 0][0] - px,
                        xys[g, 1][0] - py)
            )  # fmt: skip
            denom += d
            d_local[k] = d
    var v_d_local = SIMD[.float32, 16](0.0)
    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var d = d_local[k]
            var norm_d = fm.div(d, denom)
            var v_norm_d: Float32 = 0.0
            for c in range(ch):
                var vo = v_output[pix, c][0]
                atomic_add(v_colors, g, c, norm_d * vo)
                v_norm_d += colors[g, c][0] * vo
            var tmp = fm.div(-d, denom * denom) * v_norm_d
            comptime for l in range(TOP_K):
                comptime if l == k:
                    v_d_local[l] += fm.div(v_norm_d, denom) + tmp
                else:
                    v_d_local[l] += tmp
    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var v_sigma = v_d_local[k] * -d_local[k]
            var cx = conics[g, 0][0]
            var cy = conics[g, 1][0]
            var cz = conics[g, 2][0]
            var dx = xys[g, 0][0] - px
            var dy = xys[g, 1][0] - py
            atomic_add(v_conic, g, 0, 0.5 * v_sigma * dx * dx)
            atomic_add(v_conic, g, 1, v_sigma * dx * dy)
            atomic_add(v_conic, g, 2, 0.5 * v_sigma * dy * dy)
            atomic_add(v_xy, g, 0, v_sigma * (cx * dx + cy * dy))
            atomic_add(v_xy, g, 1, v_sigma * (cy * dx + cz * dy))
