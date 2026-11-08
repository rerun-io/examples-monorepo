"""hexel-kernel's 3D hexel rasterizer, forward (`gsplat/cuda_hexels3d/csrc/hexels3d_forward.cu`).

Covered: `preprocess_hexels3d` (camera transform, the 3x3 splat-to-pixel transform T, legacy AABB and hexagon tile
bounds, conservative bounds), `duplicate_with_keys` (one (tile << 32 | depth bits) key per touched tile), and
`render_hexels3d<kReturnSurface=false, kPrefilter=false, kReturnOwner=false>` (front-to-back compositing of the
ray-hexel intersections; flat / linear / quadratic / quadratic-bounded color, view gradients, specular lobe).
Tile ranges reuse `raster2d.tile_bin_edges_kernel`; PyTorch's stable sort reproduces cub's radix sort. The transform T,
its bounds and the render's float ops follow nvcc's fast-math PTX op for op (`fastmath`), so every output matches the
CUDA build bit for bit. Not ported: prefiltering, surface / owner / contribution / overdraw outputs, the backward.
"""

from layout import row_major
from layout import stack_allocation as tile_allocation
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from std.math import ceil, ceildiv, clamp
from std.memory import bitcast
from std.sys._assembly import inlined_assembly
from std.sys.info import is_nvidia_gpu
from std.utils.static_tuple import StaticTuple

import fastmath as fm
from raster2d import BLOCK_SIZE, TILE
from tensors import Grid, In, Out, Planes, Rows

comptime POINT_THREADS = 256
comptime PARAMS = 13  # floats per staged hexel in render_kernel
comptime NEAR: Float32 = 0.2
comptime FILTER_SIZE: Float32 = 0.707106
comptime HEX_RADIUS: Float32 = 1.15470053838
comptime LOG254: Float32 = 5.537334267
comptime FLT_MAX = Float32.MAX_FINITE
comptime FLT_EPSILON: Float32 = 1.1920929e-07
comptime Vec3 = SIMD[.float32, 4]


@inline(.always)
def _dot(a: Vec3, b: Vec3) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@inline(.always)
def _rect(px: Float32, py: Float32, rx: Int, ry: Int, tiles_x: Int, tiles_y: Int) -> Tuple[Int, Int, Int, Int]:
    """`get_rect`: (min x, min y, max x, max y) tiles of the square around (px, py)."""
    var x0 = min(tiles_x, max(0, Int(fm.div(px - Float32(rx), Float32(TILE)))))
    var y0 = min(tiles_y, max(0, Int(fm.div(py - Float32(ry), Float32(TILE)))))
    var x1 = min(tiles_x, max(0, Int(fm.div(px + Float32(rx) + Float32(TILE) - 1.0, Float32(TILE)))))
    var y1 = min(tiles_y, max(0, Int(fm.div(py + Float32(ry) + Float32(TILE) - 1.0, Float32(TILE)))))
    return (x0, y0, x1, y1)


@inline(.always)
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


@inline(.always)
def _hex_bounds(t0: Vec3, t1: Vec3, t2: Vec3, apothem_: Float32, w: Int, h: Int) -> Tuple[Bool, Float32, Float32, Int, Int]:
    """`compute_hex_bounds`: (ok, center x, center y, radius x, radius y) of the hexagon's screen bounds."""
    var apothem = apothem_ + 4e-5 * max(Float32(1.0), abs(apothem_))
    if apothem <= 0.0:
        return (False, 0.0, 0.0, 0, 0)
    var r = HEX_RADIUS * apothem
    var vx = SIMD[.float32, 8](r, 0.5 * r, -0.5 * r, -r, -0.5 * r, 0.5 * r, 0.0, 0.0)
    var vy = SIMD[.float32, 8](0.0, apothem, apothem, 0.0, -apothem, -apothem, 0.0, 0.0)
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
    means: In[.float32, Rows[3]],
    scales: In[.float32, Rows[2]],
    rotations: In[.float32, Rows[4]],
    opacities: In[.float32, Rows[1]],
    softnesses: In[.float32, Grid],
    viewmatrix: In[.float32, Rows[16]],
    projmatrix: In[.float32, Rows[16]],
    campos: In[.float32, Rows[3]],
    radii: Out[.int32, Rows[1]],
    tile_radii: Out[.int32, Rows[2]],
    points_xy: Out[.float32, Rows[2]],
    depths: Out[.float32, Rows[1]],
    transmats: Out[.float32, Rows[9]],
    tiles_touched: Out[.int32, Rows[1]],
    opacities_out: Out[.float32, Rows[1]],
    softnesses_out: Out[.float32, Grid],
    view_dirs: Out[.float32, Rows[3]],
    normals: Out[.float32, Rows[3]],
):
    var idx = global_idx.x
    if idx >= Int(num_points):
        return
    var w = Int(width)
    var h = Int(height)
    var tiles_x = ceildiv(w, TILE)
    var tiles_y = ceildiv(h, TILE)
    radii[idx, 0] = 0
    tile_radii[idx, 0] = 0
    tile_radii[idx, 1] = 0
    tiles_touched[idx, 0] = 0
    var p = Vec3(means[idx, 0][0], means[idx, 1][0], means[idx, 2][0], 0.0)
    var d = p - Vec3(campos[0, 0][0], campos[0, 1][0], campos[0, 2][0], 0.0)
    var inv_norm = fm.rsqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 1e-8)
    comptime for k in range(3):
        view_dirs[idx, k] = d[k] * inv_norm
    var z_view = fm.add(viewmatrix[0, 14][0], fm.fma(p[2], viewmatrix[0, 10][0], fm.fma(p[0], viewmatrix[0, 2][0], fm.mul(p[1], viewmatrix[0, 6][0]))))
    if not z_view > NEAR:
        return

    # T = S2W^T world2ndc ndc2pix with glm's products, op for op as nvcc's PTX (multiplications by zero included).
    var qa = rotations[idx, 0][0]
    var qb = rotations[idx, 1][0]
    var qc = rotations[idx, 2][0]
    var qd = rotations[idx, 3][0]
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
    var sx = fm.mul(scale_modifier, scales[idx, 0][0])
    var sy = fm.mul(scale_modifier, scales[idx, 1][0])
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
    var m1 = SIMD[.float32, 16](0.0)  # m1[4 j + row]
    comptime for j in range(4):
        var c0 = projmatrix[0, j][0]
        var c1 = projmatrix[0, 4 + j][0]
        var c2 = projmatrix[0, 8 + j][0]
        var c3 = projmatrix[0, 12 + j][0]
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
            s0 = softnesses[idx, 0][0]
            s1 = softnesses[idx, 1][0]
            s2 = softnesses[idx, 2][0]
        else:
            s0 = softnesses[idx, 0][0]
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
    radii[idx, 0] = Int32(radius) if old_visible else 0
    var trx: Int
    var try_: Int
    var opacity = opacities[idx, 0][0]
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
        radii[idx, 0] = Int32(max(trx, try_))
    tile_radii[idx, 0] = Int32(trx)
    tile_radii[idx, 1] = Int32(try_)
    depths[idx, 0] = z_view
    points_xy[idx, 0] = pix_x
    points_xy[idx, 1] = pix_y
    opacities_out[idx, 0] = opacity
    if softness_stride == 3:
        softnesses_out[idx, 0] = s0
        softnesses_out[idx, 1] = s1
        softnesses_out[idx, 2] = s2
    else:
        softnesses_out[idx, 0] = s0
    tiles_touched[idx, 0] = Int32((y1 - y0) * (x1 - x0))
    comptime for k in range(3):
        transmats[idx, k] = t0[k]
        transmats[idx, 3 + k] = t1[k]
        transmats[idx, 6 + k] = t2[k]
    if want_normals != 0:
        comptime for k in range(3):
            normals[idx, k] = r2[k]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def duplicate_kernel(
    width: Int32,
    height: Int32,
    points_xy: In[.float32, Rows[2]],
    depths: In[.float32, Rows[1]],
    offsets: In[.int32, Rows[1]],
    radii: In[.int32, Rows[1]],
    tile_radii: In[.int32, Rows[2]],
    keys: Out[.int64, Rows[1]],
    values: Out[.int32, Rows[1]],
):
    var idx = global_idx.x
    if idx >= Int(points_xy.dim[0]()):
        return
    var rx = Int(tile_radii[idx, 0][0])
    var ry = Int(tile_radii[idx, 1][0])
    if radii[idx, 0][0] <= 0 or rx <= 0 or ry <= 0:
        return
    var tiles_x = ceildiv(Int(width), TILE)
    var tiles_y = ceildiv(Int(height), TILE)
    var off = 0 if idx == 0 else Int(offsets[idx - 1, 0][0])
    var x0, y0, x1, y1 = _rect(points_xy[idx, 0][0], points_xy[idx, 1][0], rx, ry, tiles_x, tiles_y)
    var depth_bits = Int64(bitcast[.uint32](depths[idx, 0][0]))
    for y in range(y0, y1):
        for x in range(x0, x1):
            keys[off, 0] = (Int64(y * tiles_x + x) << 32) | depth_bits
            values[off, 0] = Int32(idx)
            off += 1


@inline(.always)
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


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def render_kernel[color_model: Int, use_view_grads: Bool, specular: Bool](
    ranges: In[.int32, Rows[2]],
    point_list: In[.int32, Rows[1]],
    colors: In[.float32, Rows[3]],
    color_grads: In[.float32, Rows[6]],
    color_quads: In[.float32, Rows[9]],
    view_grads: In[.float32, Rows[9]],
    specular_ks: In[.float32, Rows[3]],
    specular_log_s: In[.float32, Rows[1]],
    view_dirs: In[.float32, Rows[3]],
    normals: In[.float32, Rows[3]],
    transmats: In[.float32, Rows[9]],
    opacities: In[.float32, Rows[1]],
    softnesses: In[.float32, Grid],
    softness_stride: Int32,
    background: In[.float32, Rows[3]],
    out_color: Out[.float32, Planes],
    final_transmittance: Out[.float32, Rows[1]],
    last_index: Out[.int32, Rows[1]],
):
    """out_color is (3, H, W); final_transmittance and last_index one value per pixel."""
    var h = Int(out_color.dim[1]())
    var w = Int(out_color.dim[2]())
    var tiles_x = ceildiv(w, TILE)
    var px = block_idx.x * TILE + thread_idx.x
    var py = block_idx.y * TILE + thread_idx.y
    var inside = px < w and py < h
    var pixf_x = Float32(px)
    var pixf_y = Float32(py)
    var tile = block_idx.y * tiles_x + block_idx.x
    var start = Int(ranges[tile, 0][0])
    var end = Int(ranges[tile, 1][0])
    var tr = thread_idx.y * TILE + thread_idx.x
    # Per-hexel batch in shared memory: opacity, softness x3, T (9). Colors and gradients are read from global memory
    # per contributor (CUDA stages them too; here that costs occupancy: 14 KB of shared memory per block instead of 45 KB).
    var ids = tile_allocation[.int32, address_space=.SHARED](row_major[BLOCK_SIZE]())
    var params = tile_allocation[.float32, address_space=.SHARED](row_major[BLOCK_SIZE, PARAMS]())
    var stride = Int(softness_stride)
    var done = not inside
    var trans: Float32 = 1.0
    var c_acc = Vec3(0.0)
    var n_visited = 0
    var rounds = ceildiv(end - start, BLOCK_SIZE)
    for i in range(rounds):
        # Stop loading batches once every pixel of the tile is done.
        if _block_count(done) == BLOCK_SIZE:
            break
        var progress = start + i * BLOCK_SIZE + tr
        if progress < end:
            var g = Int(point_list[progress, 0][0])
            ids[tr] = Int32(g)
            params[tr, 0] = opacities[g, 0][0]
            if stride == 3:
                params[tr, 1] = softnesses[g, 0][0]
                params[tr, 2] = softnesses[g, 1][0]
                params[tr, 3] = softnesses[g, 2][0]
            else:
                var sv = softnesses[g, 0][0]
                params[tr, 1] = sv
                params[tr, 2] = sv
                params[tr, 3] = sv
            comptime for k in range(9):
                params[tr, 4 + k] = transmats[g, k][0]
        barrier()
        if done:
            continue
        var batch = min(BLOCK_SIZE, end - start - i * BLOCK_SIZE)
        # Nested ifs, not continue / break: every lane of a warp then reconverges after each hexel (as nvcc's
        # build does), so the warp reads one shared slot at a time instead of drifting apart.
        var j = 0
        while j < batch and not done:
            # Op for op as nvcc's PTX of render_hexels3d<false, false, false>.
            var k0 = fm.sub(fm.mul(params[j, 10][0], pixf_x), params[j, 4][0])
            var k1 = fm.sub(fm.mul(params[j, 11][0], pixf_x), params[j, 5][0])
            var k2 = fm.sub(fm.mul(params[j, 12][0], pixf_x), params[j, 6][0])
            var l0 = fm.sub(fm.mul(params[j, 10][0], pixf_y), params[j, 7][0])
            var l1 = fm.sub(fm.mul(params[j, 11][0], pixf_y), params[j, 8][0])
            var l2 = fm.sub(fm.mul(params[j, 12][0], pixf_y), params[j, 9][0])
            var p_x = fm.sub(fm.mul(k1, l2), fm.mul(l1, k2))
            var p_y = fm.sub(fm.mul(l0, k2), fm.mul(k0, l2))
            var p_z = fm.sub(fm.mul(k0, l1), fm.mul(l0, k1))
            if p_z != 0.0:
                var su = fm.div(p_x, p_z)
                var sv = fm.div(p_y, p_z)
                var depth = fm.add(params[j, 12][0], fm.fma(params[j, 10][0], su, fm.mul(params[j, 11][0], sv)))
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
                    var soft = (params[j, 2][0] if slant_is_1 else params[j, 3][0]) if not_edge0 else params[j, 1][0]
                    var boundary = fm.rcp(fm.add(fm.ex2(fm.mul(fm.mul(dist, soft), fm.LOG2E)), 1.0))
                    var alpha = min(fm.mul(params[j, 0][0], boundary), Float32(0.99))
                    if not alpha < 1.0 / 255.0:
                        var test_t = fm.mul(trans, fm.sub(1.0, alpha))
                        if test_t < 0.0001:
                            done = True
                        else:
                            var wgt = fm.mul(trans, alpha)
                            var g = Int(ids[j][0])
                            var c = Vec3(colors[g, 0][0], colors[g, 1][0], colors[g, 2][0], 0.0)
                            comptime if color_model >= 1:
                                comptime for ch in range(3):
                                    c[ch] = fm.add(c[ch], fm.fma(su, color_grads[g, ch][0], fm.mul(sv, color_grads[g, 3 + ch][0])))
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
                                        v2, color_quads[g, 6 + ch][0],
                                        fm.fma(uv, color_quads[g, 3 + ch][0], fm.mul(u2, color_quads[g, ch][0])),
                                    )  # fmt: skip
                                    c[ch] = fm.add(quad, c[ch])
                            var vd = Vec3(view_dirs[g, 0][0], view_dirs[g, 1][0], view_dirs[g, 2][0], 0.0)
                            comptime if use_view_grads:
                                comptime for ch in range(3):
                                    var term = fm.fma(vd[2], view_grads[g, 3 * ch + 2][0], fm.fma(vd[0], view_grads[g, 3 * ch + 0][0], fm.mul(vd[1], view_grads[g, 3 * ch + 1][0])))
                                    c[ch] = fm.add(term, c[ch])
                            comptime if specular:
                                var ndv = fm.fma(vd[2], normals[g, 2][0], fm.fma(vd[0], normals[g, 0][0], fm.mul(vd[1], normals[g, 1][0])))
                                var cos_nv = max(-ndv, Float32(0.0))
                                var spec: Float32 = 0.0
                                if cos_nv > 0.0:
                                    var s_exp = fm.ex2(fm.mul(clamp(specular_log_s[g, 0][0], 0.0, 5.0), fm.LOG2E))
                                    spec = fm.ex2(fm.mul(s_exp, fm.lg2(cos_nv)))
                                comptime for ch in range(3):
                                    c[ch] = fm.fma(specular_ks[g, ch][0], spec, c[ch])
                            comptime for ch in range(3):
                                c_acc[ch] = fm.fma(wgt, c[ch], c_acc[ch])
                            trans = test_t
                            n_visited = i * BLOCK_SIZE + j + 1
            j += 1

    if inside:
        var pix = py * w + px
        comptime for ch in range(3):
            out_color[ch, py, px] = fm.fma(trans, background[0, ch][0], c_acc[ch])
        final_transmittance[pix, 0] = trans
        last_index[pix, 0] = Int32(n_visited)
