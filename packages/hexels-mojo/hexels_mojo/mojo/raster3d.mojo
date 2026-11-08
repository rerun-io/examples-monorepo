"""hexel-kernel's 3D hexel rasterizer, forward (`gsplat/cuda_hexels3d/csrc/hexels3d_forward.cu`).

Covered: `preprocess_hexels3d` (camera transform, the 3x3 splat-to-pixel transform T, legacy AABB and hexagon tile
bounds, conservative bounds), `duplicate_with_keys` (one (tile << 32 | depth bits) key per touched tile), and
`render_hexels3d<kReturnSurface=false, kPrefilter=false, kReturnOwner=false>` (front-to-back compositing of the
ray-hexel intersections; flat / linear / quadratic / quadratic-bounded color, view gradients, specular lobe).
Tile ranges reuse `raster2d.tile_bin_edges_kernel`; PyTorch's stable sort reproduces cub's radix sort. The transform T,
its bounds and the render's float ops follow nvcc's fast-math PTX op for op (`fastmath`), so every output matches the
CUDA build bit for bit. Not ported: prefiltering, surface / owner / contribution / overdraw outputs, the backward.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import ceil
from std.memory import bitcast, stack_allocation
from std.sys._assembly import inlined_assembly
from std.sys.info import is_nvidia_gpu
from std.utils.static_tuple import StaticTuple

import fastmath as fm
from raster2d import BLOCK_SIZE, F32Ptr, I32Ptr, I64Ptr, TILE

comptime POINT_THREADS = 256
comptime PARAMS = 13  # floats per staged hexel in render_kernel
comptime NEAR: Float32 = 0.2
comptime FILTER_SIZE: Float32 = 0.707106
comptime HEX_RADIUS: Float32 = 1.15470053838
comptime LOG254: Float32 = 5.537334267
comptime FLT_MAX = Float32.MAX_FINITE
comptime FLT_EPSILON: Float32 = 1.1920929e-07
comptime Vec3 = SIMD[DType.float32, 4]


@always_inline
def _dot(a: Vec3, b: Vec3) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@always_inline
def _rect(px: Float32, py: Float32, rx: Int, ry: Int, tiles_x: Int, tiles_y: Int) -> Tuple[Int, Int, Int, Int]:
    """`get_rect`: (min x, min y, max x, max y) tiles of the square around (px, py)."""
    var x0 = min(tiles_x, max(0, Int(fm.div(px - Float32(rx), Float32(TILE)))))
    var y0 = min(tiles_y, max(0, Int(fm.div(py - Float32(ry), Float32(TILE)))))
    var x1 = min(tiles_x, max(0, Int(fm.div(px + Float32(rx) + Float32(TILE) - 1.0, Float32(TILE)))))
    var y1 = min(tiles_y, max(0, Int(fm.div(py + Float32(ry) + Float32(TILE) - 1.0, Float32(TILE)))))
    return (x0, y0, x1, y1)


@always_inline
def _hex_sdf(u: Float32, v: Float32) -> Tuple[Float32, Int]:
    var d0 = abs(v)
    var t1 = 0.8660254 * u + 0.5 * v
    var t2 = 0.8660254 * u - 0.5 * v
    var max_slant = abs(t1)
    var edge = 1
    if abs(t2) >= max_slant:
        max_slant = abs(t2)
        edge = 2
    if d0 > max_slant:
        return (d0 - 1.0, 0)
    return (max_slant - 1.0, edge)


@always_inline
def _hex_bounds(t0: Vec3, t1: Vec3, t2: Vec3, apothem_: Float32, w: Int, h: Int) -> Tuple[Bool, Float32, Float32, Int, Int]:
    """`compute_hex_bounds`: (ok, center x, center y, radius x, radius y) of the hexagon's screen bounds."""
    var apothem = apothem_ + 4e-5 * max(Float32(1.0), abs(apothem_))
    if apothem <= 0.0:
        return (False, 0.0, 0.0, 0, 0)
    var r = HEX_RADIUS * apothem
    var vx = SIMD[DType.float32, 8](r, 0.5 * r, -0.5 * r, -r, -0.5 * r, 0.5 * r, 0.0, 0.0)
    var vy = SIMD[DType.float32, 8](0.0, apothem, apothem, 0.0, -apothem, -apothem, 0.0, 0.0)
    var lo_x = FLT_MAX
    var lo_y = FLT_MAX
    var hi_x = -FLT_MAX
    var hi_y = -FLT_MAX
    comptime for i in range(6):
        var a = Vec3(vx[i], vy[i], 1.0, 0.0)
        var b = Vec3(vx[(i + 1) % 6], vy[(i + 1) % 6], 1.0, 0.0)
        var za = _dot(a, t2)
        var zb = _dot(b, t2)
        if za >= NEAR:
            var px = fm.div(_dot(a, t0), za)
            var py = fm.div(_dot(a, t1), za)
            lo_x = min(lo_x, px)
            lo_y = min(lo_y, py)
            hi_x = max(hi_x, px)
            hi_y = max(hi_y, py)
        if (za < NEAR) != (zb < NEAR):
            var edge = a + fm.div(NEAR - za, zb - za) * (b - a)
            var px = fm.div(_dot(edge, t0), NEAR)
            var py = fm.div(_dot(edge, t1), NEAR)
            lo_x = min(lo_x, px)
            lo_y = min(lo_y, py)
            hi_x = max(hi_x, px)
            hi_y = max(hi_y, py)
    if lo_x > Float32(w) or lo_y > Float32(h) or hi_x < -1.0 or hi_y < -1.0:
        return (False, 0.0, 0.0, 0, 0)
    lo_x = max(lo_x - 1.0, Float32(-TILE))
    lo_y = max(lo_y - 1.0, Float32(-TILE))
    hi_x = min(hi_x + 1.0, Float32(w + TILE))
    hi_y = min(hi_y + 1.0, Float32(h + TILE))
    return (True, (lo_x + hi_x) * 0.5, (lo_y + hi_y) * 0.5, Int(ceil((hi_x - lo_x) * 0.5)) + 1, Int(ceil((hi_y - lo_y) * 0.5)) + 1)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def preprocess_kernel(
    num_points: Int32,
    width: Int32,
    height: Int32,
    scale_modifier: Float32,
    softness_default: Float32,
    softness_stride: Int32,
    has_softness: Int32,
    conservative: Int32,
    want_normals: Int32,
    means: F32Ptr,
    scales: F32Ptr,
    rotations: F32Ptr,
    opacities: F32Ptr,
    softnesses: F32Ptr,
    viewmatrix: F32Ptr,
    projmatrix: F32Ptr,
    campos: F32Ptr,
    radii: I32Ptr,
    tile_radii: I32Ptr,
    points_xy: F32Ptr,
    depths: F32Ptr,
    transmats: F32Ptr,
    tiles_touched: I32Ptr,
    opacities_out: F32Ptr,
    softnesses_out: F32Ptr,
    view_dirs: F32Ptr,
    normals: F32Ptr,
):
    var idx = global_idx.x
    if idx >= Int(num_points):
        return
    var w = Int(width)
    var h = Int(height)
    var tiles_x = (w + TILE - 1) // TILE
    var tiles_y = (h + TILE - 1) // TILE
    radii[unsafe_offset=idx] = 0
    tile_radii[unsafe_offset=2 * idx] = 0
    tile_radii[unsafe_offset=2 * idx + 1] = 0
    tiles_touched[unsafe_offset=idx] = 0
    var p = Vec3(means.unsafe_load(3 * idx), means.unsafe_load(3 * idx + 1), means.unsafe_load(3 * idx + 2), 0.0)
    var d = p - Vec3(campos.unsafe_load(0), campos.unsafe_load(1), campos.unsafe_load(2), 0.0)
    var inv_norm = fm.rsqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 1e-8)
    comptime for k in range(3):
        view_dirs[unsafe_offset=3 * idx + k] = d[k] * inv_norm
    var m = viewmatrix
    var z_view = fm.add(m.unsafe_load(14), fm.fma(p[2], m.unsafe_load(10), fm.fma(p[0], m.unsafe_load(2), fm.mul(p[1], m.unsafe_load(6)))))
    if not z_view > NEAR:
        return

    # T = S2W^T world2ndc ndc2pix with glm's products, op for op as nvcc's PTX (multiplications by zero included).
    var qa = rotations.unsafe_load(4 * idx)
    var qb = rotations.unsafe_load(4 * idx + 1)
    var qc = rotations.unsafe_load(4 * idx + 2)
    var qd = rotations.unsafe_load(4 * idx + 3)
    var s = fm.rsqrt(fm.fma(qc, qc, fm.fma(qb, qb, fm.fma(qa, qa, fm.mul(qd, qd)))))
    var qw = fm.mul(qa, s)
    var qx = fm.mul(qb, s)
    var qy = fm.mul(qc, s)
    var qz = fm.mul(qd, s)
    var yy = fm.mul(qy, qy)
    var zz = fm.mul(qz, qz)
    var t214 = fm.add(yy, zz)
    var r00 = fm.sub(1.0, fm.add(t214, t214))
    var xy = fm.mul(qx, qy)
    var wz = fm.mul(qw, qz)
    var t220 = fm.add(xy, wz)
    var r01 = fm.add(t220, t220)
    var xz = fm.mul(qx, qz)
    var wy = fm.mul(qw, qy)
    var t224 = fm.sub(xz, wy)
    var r02 = fm.add(t224, t224)
    var t226 = fm.sub(xy, wz)
    var r10 = fm.add(t226, t226)
    var t228 = fm.fma(qx, qx, zz)
    var r11 = fm.sub(1.0, fm.add(t228, t228))
    var yz = fm.mul(qy, qz)
    var wx = fm.mul(qw, qx)
    var t233 = fm.add(yz, wx)
    var r12 = fm.add(t233, t233)
    var t235 = fm.add(xz, wy)
    var r20 = fm.add(t235, t235)
    var t236 = fm.sub(yz, wx)
    var r21 = fm.add(t236, t236)
    var t237 = fm.fma(qx, qx, yy)
    var r22 = fm.sub(1.0, fm.add(t237, t237))
    var r2 = Vec3(r20, r21, r22, 0.0)
    var sx = fm.mul(scale_modifier, scales.unsafe_load(2 * idx))
    var sy = fm.mul(scale_modifier, scales.unsafe_load(2 * idx + 1))
    var l0x = fm.fma(r20, 0.0, fm.fma(r10, 0.0, fm.mul(sx, r00)))
    var l0y = fm.fma(r21, 0.0, fm.fma(sx, r01, fm.mul(r11, 0.0)))
    var l0z = fm.fma(r22, 0.0, fm.fma(sx, r02, fm.mul(r12, 0.0)))
    var l1x = fm.fma(r20, 0.0, fm.fma(sy, r10, fm.mul(r00, 0.0)))
    var l1y = fm.fma(r21, 0.0, fm.fma(r01, 0.0, fm.mul(sy, r11)))
    var l1z = fm.fma(r22, 0.0, fm.fma(r02, 0.0, fm.mul(sy, r12)))
    var half_w = fm.div(Float32(w), 2.0)
    var off_w = fm.div(fm.add(Float32(w), -1.0), 2.0)
    var half_h = fm.div(Float32(h), 2.0)
    var off_h = fm.div(fm.add(Float32(h), -1.0), 2.0)
    # M1[j] = S2W^T world2ndc, column j of world2ndc being (pm[j], pm[4 + j], pm[8 + j], pm[12 + j]).
    var m1 = SIMD[DType.float32, 16](0.0)  # m1[4 j + row]
    comptime for j in range(4):
        var c0 = projmatrix.unsafe_load(j)
        var c1 = projmatrix.unsafe_load(4 + j)
        var c2 = projmatrix.unsafe_load(8 + j)
        var c3 = projmatrix.unsafe_load(12 + j)
        m1[4 * j] = fm.fma(c3, 0.0, fm.fma(c2, l0z, fm.fma(c0, l0x, fm.mul(c1, l0y))))
        m1[4 * j + 1] = fm.fma(c3, 0.0, fm.fma(c2, l1z, fm.fma(c0, l1x, fm.mul(c1, l1y))))
        m1[4 * j + 2] = fm.add(c3, fm.fma(p[2], c2, fm.fma(p[0], c0, fm.mul(p[1], c1))))
    var t0 = Vec3(0.0)
    var t1 = Vec3(0.0)
    var t2 = Vec3(0.0)
    comptime for r in range(3):
        var zero1 = fm.mul(m1[4 + r], 0.0)
        t0[r] = fm.fma(m1[12 + r], off_w, fm.fma(m1[8 + r], 0.0, fm.fma(m1[r], half_w, zero1)))
        t1[r] = fm.fma(m1[12 + r], off_h, fm.fma(m1[8 + r], 0.0, fm.fma(m1[r], 0.0, fm.mul(m1[4 + r], half_h))))
        t2[r] = fm.add(fm.fma(m1[8 + r], 0.0, fm.fma(m1[r], 0.0, zero1)), m1[12 + r])

    var s0 = softness_default
    var s1 = softness_default
    var s2 = softness_default
    if has_softness != 0:
        if softness_stride == 3:
            s0 = softnesses.unsafe_load(3 * idx)
            s1 = softnesses.unsafe_load(3 * idx + 1)
            s2 = softnesses.unsafe_load(3 * idx + 2)
        else:
            s0 = softnesses.unsafe_load(idx)
            s1 = s0
            s2 = s0
    s0 = max(s0, 1e-4)
    s1 = max(s1, 1e-4)
    s2 = max(s2, 1e-4)
    var softness_min = min(s0, min(s1, s2))
    var cutoff = fm.mul(fm.add(fm.div(LOG254, softness_min), 1.0), HEX_RADIUS)

    # Legacy AABB (compute_aabb), op for op.
    var cut2 = fm.mul(cutoff, cutoff)
    var dd = fm.sub(fm.fma(fm.mul(t2[0], t2[0]), cut2, fm.mul(fm.mul(t2[1], t2[1]), cut2)), fm.mul(t2[2], t2[2]))
    var bounded = dd < 0.0
    var old_bound = dd != 0.0
    var pix_x: Float32 = 0.0
    var pix_y: Float32 = 0.0
    var ext_x: Float32 = 0.0
    var ext_y: Float32 = 0.0
    if old_bound:
        var inv_d = fm.rcp(dd)
        var fxy = fm.mul(cut2, inv_d)
        pix_x = fm.sub(fm.fma(fm.mul(t2[0], t0[0]), fxy, fm.mul(fm.mul(t2[1], t0[1]), fxy)), fm.mul(fm.mul(t2[2], t0[2]), inv_d))
        pix_y = fm.sub(fm.fma(fm.mul(t2[0], t1[0]), fxy, fm.mul(fm.mul(t2[1], t1[1]), fxy)), fm.mul(fm.mul(t2[2], t1[2]), inv_d))
        var neg_dot0 = fm.sub(fm.mul(fm.mul(t0[2], t0[2]), inv_d), fm.fma(fm.mul(t0[0], t0[0]), fxy, fm.mul(fm.mul(t0[1], t0[1]), fxy)))
        var neg_dot1 = fm.sub(fm.mul(fm.mul(t1[2], t1[2]), inv_d), fm.fma(fm.mul(t1[0], t1[0]), fxy, fm.mul(fm.mul(t1[1], t1[1]), fxy)))
        ext_x = fm.sqrt(max(fm.fma(pix_x, pix_x, neg_dot0), Float32(1e-4)))
        ext_y = fm.sqrt(max(fm.fma(pix_y, pix_y, neg_dot1), Float32(1e-4)))
    var radius = Int(ceil(max(max(ext_x, ext_y), fm.mul(cutoff, FILTER_SIZE)))) if old_bound else 0
    var x0, y0, x1, y1 = _rect(pix_x, pix_y, radius, radius, tiles_x, tiles_y)
    var old_visible = old_bound and x0 < x1 and y0 < y1
    var corrected = conservative != 0
    if not old_visible and not corrected:
        return
    radii[unsafe_offset=idx] = Int32(radius) if old_visible else 0
    var trx: Int
    var try_: Int
    var opacity = opacities.unsafe_load(idx)
    if bounded and opacity <= 1.0:
        trx = min(radius, Int(ceil(max(ext_x, cutoff * FILTER_SIZE))) + 1)
        try_ = min(radius, Int(ceil(max(ext_y, cutoff * FILTER_SIZE))) + 1)
        x0, y0, x1, y1 = _rect(pix_x, pix_y, trx, try_, tiles_x, tiles_y)
    var lx0 = x0
    var ly0 = y0
    var lx1 = x1
    var ly1 = y1
    if opacity < 1.0 / 255.0:
        return
    var margin = fm.div(LOG254, softness_min)
    if opacity > 1.0 / 255.0:
        var log_cutoff = fm.lg2((255.0 * (1.0 + 8.0 * FLT_EPSILON)) * opacity - 1.0) * fm.LN2
        var softness_max = max(s0, max(s1, s2))
        margin = fm.div(log_cutoff, softness_max if log_cutoff < 0.0 else softness_min)
    var ok, cx, cy, hrx, hry = _hex_bounds(t0, t1, t2, 1.0 + margin, w, h)
    if not ok:
        return
    pix_x = cx
    pix_y = cy
    trx = hrx
    try_ = hry
    x0, y0, x1, y1 = _rect(pix_x, pix_y, trx, try_, tiles_x, tiles_y)
    if not corrected:
        x0 = max(x0, lx0)
        y0 = max(y0, ly0)
        x1 = min(x1, lx1)
        y1 = min(y1, ly1)
        if x0 >= x1 or y0 >= y1:
            return
        pix_x = Float32(x0 + x1) * Float32(TILE // 2)
        pix_y = Float32(y0 + y1) * Float32(TILE // 2)
        trx = (x1 - x0) * (TILE // 2)
        try_ = (y1 - y0) * (TILE // 2)
    if corrected and not old_visible:
        radii[unsafe_offset=idx] = Int32(max(trx, try_))
    tile_radii[unsafe_offset=2 * idx] = Int32(trx)
    tile_radii[unsafe_offset=2 * idx + 1] = Int32(try_)
    depths[unsafe_offset=idx] = z_view
    points_xy[unsafe_offset=2 * idx] = pix_x
    points_xy[unsafe_offset=2 * idx + 1] = pix_y
    opacities_out[unsafe_offset=idx] = opacity
    if softness_stride == 3:
        softnesses_out[unsafe_offset=3 * idx] = s0
        softnesses_out[unsafe_offset=3 * idx + 1] = s1
        softnesses_out[unsafe_offset=3 * idx + 2] = s2
    else:
        softnesses_out[unsafe_offset=idx] = s0
    tiles_touched[unsafe_offset=idx] = Int32((y1 - y0) * (x1 - x0))
    comptime for k in range(3):
        transmats[unsafe_offset=9 * idx + k] = t0[k]
        transmats[unsafe_offset=9 * idx + 3 + k] = t1[k]
        transmats[unsafe_offset=9 * idx + 6 + k] = t2[k]
    if want_normals != 0:
        comptime for k in range(3):
            normals[unsafe_offset=3 * idx + k] = r2[k]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def duplicate_kernel(
    num_points: Int32, width: Int32, height: Int32, points_xy: F32Ptr, depths: F32Ptr, offsets: I32Ptr, radii: I32Ptr, tile_radii: I32Ptr,
    keys: I64Ptr, values: I32Ptr,
):  # fmt: skip
    var idx = global_idx.x
    if idx >= Int(num_points):
        return
    var rx = Int(tile_radii.unsafe_load(2 * idx))
    var ry = Int(tile_radii.unsafe_load(2 * idx + 1))
    if radii.unsafe_load(idx) <= 0 or rx <= 0 or ry <= 0:
        return
    var tiles_x = (Int(width) + TILE - 1) // TILE
    var tiles_y = (Int(height) + TILE - 1) // TILE
    var off = 0 if idx == 0 else Int(offsets.unsafe_load(idx - 1))
    var x0, y0, x1, y1 = _rect(points_xy.unsafe_load(2 * idx), points_xy.unsafe_load(2 * idx + 1), rx, ry, tiles_x, tiles_y)
    var depth_bits = Int64(bitcast[DType.uint32](depths.unsafe_load(idx)))
    for y in range(y0, y1):
        for x in range(x0, x1):
            keys[unsafe_offset=off] = (Int64(y * tiles_x + x) << 32) | depth_bits
            values[unsafe_offset=off] = Int32(idx)
            off += 1


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
@always_inline
def _block_count(pred: Bool) -> Int:
    """CUDA's `__syncthreads_count(pred)`: a block barrier that returns how many threads passed a true `pred`."""
    comptime if is_nvidia_gpu():
        return Int(
            inlined_assembly[
                "{ .reg .pred p; setp.ne.s32 p, $1, 0; bar.red.popc.u32 $0, 0, p; }", Int32, constraints="=r,r", has_side_effect=True
            ](Int32(1) if pred else Int32(0))
        )
    else:
        barrier()
        return 0


def render_kernel[color_model: Int, use_view_grads: Bool, specular: Bool](
    width: Int32,
    height: Int32,
    ranges: I32Ptr,
    point_list: I32Ptr,
    colors: F32Ptr,
    color_grads: F32Ptr,
    color_quads: F32Ptr,
    view_grads: F32Ptr,
    specular_ks: F32Ptr,
    specular_log_s: F32Ptr,
    view_dirs: F32Ptr,
    normals: F32Ptr,
    transmats: F32Ptr,
    opacities: F32Ptr,
    softnesses: F32Ptr,
    softness_stride: Int32,
    background: F32Ptr,
    out_color: F32Ptr,
    final_transmittance: F32Ptr,
    last_index: I32Ptr,
):
    var w = Int(width)
    var h = Int(height)
    var tiles_x = (w + TILE - 1) // TILE
    var px = block_idx.x * TILE + thread_idx.x
    var py = block_idx.y * TILE + thread_idx.y
    var inside = px < w and py < h
    var pixf_x = Float32(px)
    var pixf_y = Float32(py)
    var tile = block_idx.y * tiles_x + block_idx.x
    var start = Int(ranges.unsafe_load(2 * tile))
    var end = Int(ranges.unsafe_load(2 * tile + 1))
    var tr = thread_idx.y * TILE + thread_idx.x
    # Per-hexel batch in shared memory: opacity, softness x3, T (9). Colors and gradients are read from global memory
    # per contributor (CUDA stages them too; here that costs occupancy: 14 KB of shared memory per block instead of 45 KB).
    var ids = stack_allocation[BLOCK_SIZE, Int32, address_space=AddressSpace.SHARED]()
    var params = stack_allocation[BLOCK_SIZE * PARAMS, Float32, address_space=AddressSpace.SHARED]()
    var stride = Int(softness_stride)
    var done = not inside
    var trans: Float32 = 1.0
    var c_acc = Vec3(0.0)
    var n_visited = 0
    var rounds = (end - start + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(rounds):
        # Stop loading batches once every pixel of the tile is done.
        if _block_count(done) == BLOCK_SIZE:
            break
        var progress = start + i * BLOCK_SIZE + tr
        if progress < end:
            var g = Int(point_list.unsafe_load(progress))
            ids[unsafe_offset=tr] = Int32(g)
            var q = params.unsafe_offset(PARAMS * tr)
            q[unsafe_offset=0] = opacities.unsafe_load(g)
            if stride == 3:
                q[unsafe_offset=1] = softnesses.unsafe_load(3 * g)
                q[unsafe_offset=2] = softnesses.unsafe_load(3 * g + 1)
                q[unsafe_offset=3] = softnesses.unsafe_load(3 * g + 2)
            else:
                var sv = softnesses.unsafe_load(g)
                q[unsafe_offset=1] = sv
                q[unsafe_offset=2] = sv
                q[unsafe_offset=3] = sv
            comptime for k in range(9):
                q[unsafe_offset=4 + k] = transmats.unsafe_load(9 * g + k)
        barrier()
        if done:
            continue
        var batch = min(BLOCK_SIZE, end - start - i * BLOCK_SIZE)
        # Nested ifs, not continue / break: every lane of a warp then reconverges after each hexel (as nvcc's
        # build does), so the warp reads one shared slot at a time instead of drifting apart.
        var j = 0
        while j < batch and not done:
            var q = params.unsafe_offset(PARAMS * j)
            # Op for op as nvcc's PTX of render_hexels3d<false, false, false>.
            var k0 = fm.sub(fm.mul(q.unsafe_load(10), pixf_x), q.unsafe_load(4))
            var k1 = fm.sub(fm.mul(q.unsafe_load(11), pixf_x), q.unsafe_load(5))
            var k2 = fm.sub(fm.mul(q.unsafe_load(12), pixf_x), q.unsafe_load(6))
            var l0 = fm.sub(fm.mul(q.unsafe_load(10), pixf_y), q.unsafe_load(7))
            var l1 = fm.sub(fm.mul(q.unsafe_load(11), pixf_y), q.unsafe_load(8))
            var l2 = fm.sub(fm.mul(q.unsafe_load(12), pixf_y), q.unsafe_load(9))
            var p_x = fm.sub(fm.mul(k1, l2), fm.mul(l1, k2))
            var p_y = fm.sub(fm.mul(l0, k2), fm.mul(k0, l2))
            var p_z = fm.sub(fm.mul(k0, l1), fm.mul(l0, k1))
            if p_z != 0.0:
                var su = fm.div(p_x, p_z)
                var sv = fm.div(p_y, p_z)
                var depth = fm.add(q.unsafe_load(12), fm.fma(q.unsafe_load(10), su, fm.mul(q.unsafe_load(11), sv)))
                if not depth < NEAR:
                    var d0 = abs(sv)
                    var ua = fm.mul(su, 0.8660254)
                    var vb = fm.mul(sv, 0.5)
                    var d1 = abs(fm.add(ua, vb))
                    var d2 = abs(fm.sub(ua, vb))
                    var slant_is_1 = d2 < d1
                    var slant = d1 if slant_is_1 else d2
                    var not_edge0 = d0 <= slant
                    var dist = fm.add(slant if not_edge0 else d0, -1.0)
                    var soft = (q.unsafe_load(2) if slant_is_1 else q.unsafe_load(3)) if not_edge0 else q.unsafe_load(1)
                    var boundary = fm.rcp(fm.add(fm.ex2(fm.mul(fm.mul(dist, soft), fm.LOG2E)), 1.0))
                    var alpha = min(fm.mul(q.unsafe_load(0), boundary), Float32(0.99))
                    if not alpha < 1.0 / 255.0:
                        var test_t = fm.mul(trans, fm.sub(1.0, alpha))
                        if test_t < 0.0001:
                            done = True
                        else:
                            var wgt = fm.mul(trans, alpha)
                            var g = Int(ids.unsafe_load(j))
                            var c = Vec3(colors.unsafe_load(3 * g), colors.unsafe_load(3 * g + 1), colors.unsafe_load(3 * g + 2), 0.0)
                            comptime if color_model >= 1:
                                comptime for ch in range(3):
                                    c[ch] = fm.add(c[ch], fm.fma(su, color_grads.unsafe_load(6 * g + ch), fm.mul(sv, color_grads.unsafe_load(6 * g + 3 + ch))))
                            comptime if color_model >= 2:
                                var qu = su
                                var qv = sv
                                comptime if color_model == 3:
                                    var rho = fm.add(dist, 1.0)
                                    var inv = fm.rcp(rho) if rho > 1.0 else Float32(1.0)
                                    qu = fm.mul(su, inv)
                                    qv = fm.mul(sv, inv)
                                var u2 = fm.mul(qu, qu)
                                var uv = fm.mul(qu, qv)
                                var v2 = fm.mul(qv, qv)
                                comptime for ch in range(3):
                                    var quad = fm.fma(
                                        v2, color_quads.unsafe_load(9 * g + 6 + ch),
                                        fm.fma(uv, color_quads.unsafe_load(9 * g + 3 + ch), fm.mul(u2, color_quads.unsafe_load(9 * g + ch))),
                                    )  # fmt: skip
                                    c[ch] = fm.add(quad, c[ch])
                            var vd = Vec3(view_dirs.unsafe_load(3 * g), view_dirs.unsafe_load(3 * g + 1), view_dirs.unsafe_load(3 * g + 2), 0.0)
                            comptime if use_view_grads:
                                comptime for ch in range(3):
                                    var vg = view_grads.unsafe_offset(9 * g + 3 * ch)
                                    var term = fm.fma(vd[2], vg.unsafe_load(2), fm.fma(vd[0], vg.unsafe_load(0), fm.mul(vd[1], vg.unsafe_load(1))))
                                    c[ch] = fm.add(term, c[ch])
                            comptime if specular:
                                var n = normals.unsafe_offset(3 * g)
                                var ndv = fm.fma(vd[2], n.unsafe_load(2), fm.fma(vd[0], n.unsafe_load(0), fm.mul(vd[1], n.unsafe_load(1))))
                                var cos_nv = max(-ndv, Float32(0.0))
                                var spec: Float32 = 0.0
                                if cos_nv > 0.0:
                                    var s_exp = fm.ex2(fm.mul(min(max(specular_log_s.unsafe_load(g), Float32(0.0)), Float32(5.0)), fm.LOG2E))
                                    spec = fm.ex2(fm.mul(s_exp, fm.lg2(cos_nv)))
                                comptime for ch in range(3):
                                    c[ch] = fm.fma(specular_ks.unsafe_load(3 * g + ch), spec, c[ch])
                            comptime for ch in range(3):
                                c_acc[ch] = fm.fma(wgt, c[ch], c_acc[ch])
                            trans = test_t
                            n_visited = i * BLOCK_SIZE + j + 1
            j += 1

    if inside:
        var pix = py * w + px
        comptime for ch in range(3):
            out_color[unsafe_offset=ch * h * w + pix] = fm.fma(trans, background.unsafe_load(ch), c_acc[ch])
        final_transmittance[unsafe_offset=pix] = trans
        last_index[unsafe_offset=pix] = Int32(n_visited)
