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

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import ceil, isinf, isnan
from std.memory import stack_allocation
from std.utils.static_tuple import StaticTuple

import fastmath as fm
from raster2d import BLOCK_SIZE, EPS, F32Ptr, I32Ptr, MAX_CHANNELS, TILE, TOP_K, _atomic_add

comptime POINT_THREADS = 256


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def project_forward_kernel(
    num_points: Int32,
    width: Int32,
    height: Int32,
    tiles_x: Int32,
    tiles_y: Int32,
    means: F32Ptr,
    scales: F32Ptr,
    rotation: F32Ptr,
    xys: F32Ptr,
    radii: I32Ptr,
    conics: F32Ptr,
    num_tiles_hit: I32Ptr,
):
    var idx = global_idx.x
    if idx >= Int(num_points):
        return
    radii[unsafe_offset=idx] = 0
    num_tiles_hit[unsafe_offset=idx] = 0
    var cx = fm.mul(means.unsafe_load(2 * idx), Float32(Int(width)))
    var cy = fm.mul(means.unsafe_load(2 * idx + 1), Float32(Int(height)))
    var rot = rotation.unsafe_load(idx)
    var c = fm.cos(rot)
    var s = fm.sin(rot)
    var sx = scales.unsafe_load(2 * idx)
    var sy = scales.unsafe_load(2 * idx + 1)
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
    conics[unsafe_offset=3 * idx] = fm.mul(cov_z, inv_det)
    conics[unsafe_offset=3 * idx + 1] = fm.mul(inv_det, -cov_y)
    conics[unsafe_offset=3 * idx + 2] = fm.mul(cov_x, inv_det)
    xys[unsafe_offset=2 * idx] = cx
    xys[unsafe_offset=2 * idx + 1] = cy
    radii[unsafe_offset=idx] = Int32(Int(radius))
    var tcx = fm.div(cx, Float32(TILE))
    var tcy = fm.div(cy, Float32(TILE))
    var tr = fm.div(radius, Float32(TILE))
    var x0 = min(max(Int(fm.sub(tcx, tr)), 0), Int(tiles_x))
    var x1 = min(max(Int(fm.add(fm.add(tcx, tr), 1.0)), 0), Int(tiles_x))
    var y0 = min(max(Int(fm.sub(tcy, tr)), 0), Int(tiles_y))
    var y1 = min(max(Int(fm.add(fm.add(tcy, tr), 1.0)), 0), Int(tiles_y))
    var area = (x1 - x0) * (y1 - y0)
    if area > 0:
        num_tiles_hit[unsafe_offset=idx] = Int32(area)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def project_backward_kernel(
    num_points: Int32,
    width: Int32,
    height: Int32,
    scales: F32Ptr,
    rotation: F32Ptr,
    radii: I32Ptr,
    conics: F32Ptr,
    v_xy: F32Ptr,
    v_conic: F32Ptr,
    v_cov2d: F32Ptr,
    v_mean: F32Ptr,
    v_scale: F32Ptr,
    v_rot: F32Ptr,
):
    """Op for op as nvcc's PTX of `project_gaussians_2d_scale_rot_backward_kernel` (register names in comments)."""
    var idx = global_idx.x
    if idx >= Int(num_points) or radii.unsafe_load(idx) <= 0:
        return
    # v_Sigma = -X G X with X the conic, G = dL/dconic (cov2d_to_conic_vjp).
    var f1 = conics.unsafe_load(3 * idx)
    var f2 = conics.unsafe_load(3 * idx + 1)
    var f3 = conics.unsafe_load(3 * idx + 2)
    var f4 = v_conic.unsafe_load(3 * idx)
    var f5 = v_conic.unsafe_load(3 * idx + 1)
    var f6 = v_conic.unsafe_load(3 * idx + 2)
    var f11 = fm.sub(fm.mul(f5, -f2), fm.mul(f1, f4))
    var f14 = fm.sub(fm.mul(f5, -f3), fm.mul(f2, f4))
    var f17 = fm.sub(fm.mul(f6, -f2), fm.mul(f1, f5))
    var f20 = fm.sub(fm.mul(f6, -f3), fm.mul(f2, f5))
    var f22 = fm.fma(f1, f11, fm.mul(f2, f17))
    var f24 = fm.fma(f1, f14, fm.mul(f2, f20))
    var f26 = fm.fma(f2, f11, fm.mul(f3, f17))
    var f28 = fm.fma(f2, f14, fm.mul(f3, f20))
    var f29 = fm.add(f26, f24)
    v_cov2d[unsafe_offset=3 * idx] = f22
    v_cov2d[unsafe_offset=3 * idx + 1] = f29
    v_cov2d[unsafe_offset=3 * idx + 2] = f28
    # Rotation and scale gradients through M = R S (glm's full 2x2 products).
    var rot = rotation.unsafe_load(idx)
    var f31 = fm.cos(rot)
    var f32 = fm.sin(rot)
    var f34 = scales.unsafe_load(2 * idx)
    var f35 = scales.unsafe_load(2 * idx + 1)
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
    v_scale[unsafe_offset=2 * idx] = fm.fma(f28, f88, fm.fma(f22, f83, fm.mul(f96, f85)))
    v_scale[unsafe_offset=2 * idx + 1] = fm.fma(f28, f95, fm.fma(f22, f92, fm.mul(f96, f93)))
    v_rot[unsafe_offset=idx] = fm.fma(f28, f73, fm.fma(f22, f71, fm.mul(f96, f72)))
    v_mean[unsafe_offset=2 * idx] = fm.mul(v_xy.unsafe_load(2 * idx), Float32(Int(width)))
    v_mean[unsafe_offset=2 * idx + 1] = fm.mul(v_xy.unsafe_load(2 * idx + 1), Float32(Int(height)))


@always_inline
def _sigma(cx: Float32, cy: Float32, cz: Float32, dx: Float32, dy: Float32) -> Float32:
    """0.5 (cx dx^2 + cz dy^2) + cy dx dy, as nvcc fuses it."""
    return fm.fma(fm.mul(cy, dx), dy, fm.mul(fm.fma(dx, fm.mul(cx, dx), fm.mul(dy, fm.mul(cz, dy))), 0.5))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def forward_kernel(
    tiles_x: Int32,
    width: Int32,
    height: Int32,
    channels: Int32,
    ids_sorted: I32Ptr,
    tile_bins: I32Ptr,
    xys: F32Ptr,
    conics: F32Ptr,
    colors: F32Ptr,
    out_img: F32Ptr,
    pixel_topk: I32Ptr,
):
    var w = Int(width)
    var ch = Int(channels)
    var tile_id = block_idx.y * Int(tiles_x) + block_idx.x
    var i = block_idx.y * TILE + thread_idx.y
    var j = block_idx.x * TILE + thread_idx.x
    var px = Float32(j)
    var py = Float32(i)
    var pix = i * w + j
    var inside = i < Int(height) and j < w
    var start = Int(tile_bins.unsafe_load(2 * tile_id))
    var end = Int(tile_bins.unsafe_load(2 * tile_id + 1))
    var id_batch = stack_allocation[BLOCK_SIZE, Int32, address_space=AddressSpace.SHARED]()
    var param_batch = stack_allocation[BLOCK_SIZE * 5, Float32, address_space=AddressSpace.SHARED]()
    var tr = thread_idx.y * TILE + thread_idx.x
    var topk_ids = stack_allocation[TOP_K, Int32]()
    var topk_val = stack_allocation[TOP_K, Float32]()
    comptime for k in range(TOP_K):
        topk_ids[unsafe_offset=k] = -1
        topk_val[unsafe_offset=k] = 0.0
    var num_batches = (end - start + BLOCK_SIZE - 1) // BLOCK_SIZE
    for b in range(num_batches):
        barrier()
        var batch_start = start + BLOCK_SIZE * b
        var idx = batch_start + tr
        if idx < end:
            var g = Int(ids_sorted.unsafe_load(idx))
            id_batch[unsafe_offset=tr] = Int32(g)
            var p = param_batch.unsafe_offset(5 * tr)
            p[unsafe_offset=0] = xys.unsafe_load(2 * g)
            p[unsafe_offset=1] = xys.unsafe_load(2 * g + 1)
            p[unsafe_offset=2] = conics.unsafe_load(3 * g)
            p[unsafe_offset=3] = conics.unsafe_load(3 * g + 1)
            p[unsafe_offset=4] = conics.unsafe_load(3 * g + 2)
        barrier()
        if not inside:
            continue
        for t in range(min(BLOCK_SIZE, end - batch_start)):
            var p = param_batch.unsafe_offset(5 * t)
            var sigma = _sigma(
                p.unsafe_load(2), p.unsafe_load(3), p.unsafe_load(4), fm.sub(p.unsafe_load(0), px), fm.sub(p.unsafe_load(1), py)
            )
            if not (sigma < 0.0 or isnan(sigma) or isinf(sigma)):  # an if, not continue: lanes reconverge per Gaussian
                var alpha = fm.ex2(fm.mul(sigma, -fm.LOG2E))
                var min_id = -1
                var min_val: Float32 = 1e30
                for k in range(TOP_K):
                    if topk_ids.unsafe_load(k) < 0:
                        min_id = k
                        min_val = -1.0
                        break
                    var v = topk_val.unsafe_load(k)
                    if v < min_val:
                        min_id = k
                        min_val = v
                if alpha > min_val:
                    topk_ids[unsafe_offset=min_id] = id_batch.unsafe_load(t)
                    topk_val[unsafe_offset=min_id] = alpha
    if not inside:
        return
    var sum_val: Float32 = 0.0
    comptime for k in range(TOP_K):
        if topk_ids.unsafe_load(k) >= 0:
            sum_val = fm.add(sum_val, topk_val.unsafe_load(k))
    var denom = Float32(Float64(sum_val) + EPS)
    var pix_out = stack_allocation[MAX_CHANNELS, Float32]()
    for c in range(ch):
        pix_out[unsafe_offset=c] = 0.0
    for c in range(ch):
        comptime for k in range(TOP_K):
            var g = Int(topk_ids.unsafe_load(k))
            if g >= 0:
                var vis = fm.div(topk_val.unsafe_load(k), denom)
                pix_out[unsafe_offset=c] = fm.fma(vis, colors.unsafe_load(g * ch + c), pix_out.unsafe_load(c))
    for c in range(ch):
        out_img[unsafe_offset=pix * ch + c] = pix_out.unsafe_load(c)
    comptime for k in range(TOP_K):
        pixel_topk[unsafe_offset=pix * TOP_K + k] = topk_ids.unsafe_load(k)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def backward_kernel(
    width: Int32,
    height: Int32,
    channels: Int32,
    xys: F32Ptr,
    conics: F32Ptr,
    colors: F32Ptr,
    v_output: F32Ptr,
    pixel_topk: I32Ptr,
    v_xy: F32Ptr,
    v_conic: F32Ptr,
    v_colors: F32Ptr,
):
    var w = Int(width)
    var ch = Int(channels)
    var i = global_idx.y
    var j = global_idx.x
    if i >= Int(height) or j >= w:
        return
    var px = Float32(j)
    var py = Float32(i)
    var pix = i * w + j
    var topk = SIMD[DType.int32, 16](-1)
    comptime for k in range(TOP_K):
        topk[k] = pixel_topk.unsafe_load(pix * TOP_K + k)
    var d_local = SIMD[DType.float32, 16](0.0)
    var denom = Float32(EPS)
    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var d = fm.expf(
                -_sigma(conics.unsafe_load(3 * g), conics.unsafe_load(3 * g + 1), conics.unsafe_load(3 * g + 2), xys.unsafe_load(2 * g) - px,
                        xys.unsafe_load(2 * g + 1) - py)
            )  # fmt: skip
            denom += d
            d_local[k] = d
    var v_d_local = SIMD[DType.float32, 16](0.0)
    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var d = d_local[k]
            var norm_d = fm.div(d, denom)
            var v_norm_d: Float32 = 0.0
            for c in range(ch):
                var vo = v_output.unsafe_load(pix * ch + c)
                _atomic_add(v_colors, g * ch + c, norm_d * vo)
                v_norm_d += colors.unsafe_load(g * ch + c) * vo
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
            var cx = conics.unsafe_load(3 * g)
            var cy = conics.unsafe_load(3 * g + 1)
            var cz = conics.unsafe_load(3 * g + 2)
            var dx = xys.unsafe_load(2 * g) - px
            var dy = xys.unsafe_load(2 * g + 1) - py
            _atomic_add(v_conic, 3 * g, 0.5 * v_sigma * dx * dx)
            _atomic_add(v_conic, 3 * g + 1, v_sigma * dx * dy)
            _atomic_add(v_conic, 3 * g + 2, 0.5 * v_sigma * dy * dy)
            _atomic_add(v_xy, 2 * g, v_sigma * (cx * dx + cy * dy))
            _atomic_add(v_xy, 2 * g + 1, v_sigma * (cy * dx + cz * dy))
