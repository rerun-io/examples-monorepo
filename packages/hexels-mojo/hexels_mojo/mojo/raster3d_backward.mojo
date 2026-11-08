"""hexel-kernel's 3D hexel rasterizer, cached backward (`gsplat/cuda_hexels3d/csrc/hexels3d_forward.cu`).

Covered: `render_hexels3d_backward<kHasSurfaceGrads=false, kPrefilter=false>` with the forward's saved final
transmittance and last index (every color model, view gradients, specular lobe, the optional |d pixel| statistic of
(P, 4) means2d; the tile walks its list together and sums each hexel's gradients across the warp before one atomic,
3.5x faster than CUDA's one-pixel-per-thread walk), `transmat_to_means2d_backward` and `transmat_to_geometry_backward` (the splat-to-pixel transform's
gradient back to means, scales and quaternions). The ray-hexel intersection and alpha follow the forward render's
fast-math ops (`raster3d.render_kernel`), so the backward revisits exactly the hexels the forward composited. Float
atomics reorder sums, so the gradients match the CUDA build to a tolerance, not bit for bit. Not ported: surface /
overdraw gradients and prefiltering (they stay CUDA).
"""

from layout import row_major
from layout import stack_allocation as tile_allocation
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, lane_id, thread_idx
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from std.atomic import Atomic, Ordering
from std.math import ceildiv, clamp
from std.utils.static_tuple import StaticTuple

import fastmath as fm
from raster2d import BLOCK_SIZE, TILE
from raster3d import NEAR, POINT_THREADS, Vec3
from tensors import Grid, In, Out, Planes, Rows, atomic_add

comptime PARAMS = 13  # floats per staged hexel: T (9), opacity, softness x3

comptime HEX_A: Float32 = 0.8660254  # 0.5 * sqrt(3)
comptime HEX_B: Float32 = 0.5


@inline(.always)
def _hex_sdf_grad(u: Float32, v: Float32) -> Tuple[Float32, Float32, Float32, Int]:
    """`hex_sdf_grad`: (distance, d/du, d/dv, edge id) of the pointy-top hexagon's 3-way max."""
    var d0 = abs(v)
    var t1 = HEX_A * u + HEX_B * v
    var t2 = HEX_A * u - HEX_B * v
    var max_slant = abs(t1)
    var edge = 1
    if abs(t2) >= max_slant:
        max_slant = abs(t2)
        edge = 2
    if d0 > max_slant:
        return (d0 - 1.0, Float32(0.0), Float32(1.0) if v >= 0.0 else Float32(-1.0), 0)
    if edge == 1:
        var s = Float32(1.0) if t1 >= 0.0 else Float32(-1.0)
        return (max_slant - 1.0, HEX_A * s, HEX_B * s, 1)
    var s = Float32(1.0) if t2 >= 0.0 else Float32(-1.0)
    return (max_slant - 1.0, HEX_A * s, -HEX_B * s, 2)


@inline(.always)
def _bounded(u: Float32, v: Float32, dist: Float32) -> Tuple[Float32, Float32]:
    """`bounded_quadratic_coordinates`: (u, v) scaled back onto the hexagon when outside it (rho = dist + 1)."""
    var rho = dist + 1.0
    var inv = 1.0 / rho if rho > 1.0 else Float32(1.0)
    return (u * inv, v * inv)


@inline(.always)
def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.0)


@inline(.always)
def _dot3(a: Vec3, b: Vec3) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def render_backward_kernel[color_model: Int, use_view_grads: Bool, specular: Bool](
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
    has_softness_grad: Int32,
    abs_means2d: Int32,
    background: In[.float32, Rows[3]],
    grad_out_color: In[.float32, Planes],
    final_transmittance: In[.float32, Rows[1]],
    last_index: In[.int32, Rows[1]],
    grad_colors: Out[.float32, Rows[3]],
    grad_color_grads: Out[.float32, Rows[6]],
    grad_color_quads: Out[.float32, Rows[9]],
    grad_view_grads: Out[.float32, Rows[9]],
    grad_specular_ks: Out[.float32, Rows[3]],
    grad_specular_log_s: Out[.float32, Rows[1]],
    grad_opacities: Out[.float32, Rows[1]],
    grad_softnesses: Out[.float32, Grid],
    grad_transmats: Out[.float32, Rows[9]],
    grad_means2d: Out[.float32, Grid],
    grad_view_dirs: Out[.float32, Rows[3]],
    grad_normals: Out[.float32, Rows[3]],
):
    """The tile walks its hexel list back to front together (CUDA walks each pixel's list on its own): every lane of a
    warp then holds the same hexel, so each gradient is summed across the warp and written by one atomic instead of
    up to 32. Pixel by pixel the arithmetic is CUDA's; only the order of the sums changes."""
    var h = Int(grad_out_color.dim[1]())
    var w = Int(grad_out_color.dim[2]())
    var px = block_idx.x * TILE + thread_idx.x
    var py = block_idx.y * TILE + thread_idx.y
    var inside = px < w and py < h
    var tile = block_idx.y * ceildiv(w, TILE) + block_idx.x
    var start = Int(ranges[tile, 0][0])
    if Int(ranges[tile, 1][0]) <= start:
        return  # block-uniform
    var tr = thread_idx.y * TILE + thread_idx.x
    var pix = py * w + px if inside else 0
    var pixf_x = Float32(px)
    var pixf_y = Float32(py)
    var v_out = Vec3(0.0)
    var t_after: Float32 = 1.0
    var my_last = 0
    if inside:
        v_out = Vec3(grad_out_color[0, py, px][0], grad_out_color[1, py, px][0], grad_out_color[2, py, px][0], 0.0)
        t_after = final_transmittance[pix, 0][0]
        my_last = Int(last_index[pix, 0][0])
    var behind = Vec3(background[0, 0][0], background[0, 1][0], background[0, 2][0], 0.0)
    var stride = Int(softness_stride)
    var ids = tile_allocation[.int32, address_space=.SHARED](row_major[BLOCK_SIZE]())
    var params = tile_allocation[.float32, address_space=.SHARED](row_major[BLOCK_SIZE, PARAMS]())
    var tile_last = tile_allocation[.int32, address_space=.SHARED](row_major[1]())
    if tr == 0:
        tile_last[0] = 0
    barrier()
    _ = Atomic[Int32, scope="block"].max[ordering=Ordering.RELAXED](tile_last.unsafe_ptr(), Int32(my_last))
    barrier()
    var n = Int(tile_last[0][0])
    var rounds = ceildiv(n, BLOCK_SIZE)
    for b in range(rounds):
        var top = n - 1 - b * BLOCK_SIZE  # slot j holds list position top - j
        barrier()
        if top - tr >= 0:
            var g = Int(point_list[start + top - tr, 0][0])
            ids[tr] = Int32(g)
            comptime for k in range(9):
                params[tr, k] = transmats[g, k][0]
            params[tr, 9] = opacities[g, 0][0]
            if stride == 3:
                params[tr, 10] = softnesses[g, 0][0]
                params[tr, 11] = softnesses[g, 1][0]
                params[tr, 12] = softnesses[g, 2][0]
            else:
                var sv = softnesses[g, 0][0]
                params[tr, 10] = sv
                params[tr, 11] = sv
                params[tr, 12] = sv
        barrier()
        for j in range(min(BLOCK_SIZE, top + 1)):
            # This lane's share of hexel j's gradients (zero unless it composited the hexel).
            var contributes = False
            var g_color = Vec3(0.0)
            var g_cgrad = SIMD[.float32, 8](0.0)
            var g_quad = SIMD[.float32, 16](0.0)
            var g_vgrad = SIMD[.float32, 16](0.0)
            var g_vdir = Vec3(0.0)
            var g_ks = Vec3(0.0)
            var g_logs: Float32 = 0.0
            var g_normal = Vec3(0.0)
            var g_opacity: Float32 = 0.0
            var g_soft = Vec3(0.0)
            var g_tm = SIMD[.float32, 16](0.0)
            var g_abs: Float32 = 0.0
            if top - j < my_last:
                var tu = Vec3(params[j, 0][0], params[j, 1][0], params[j, 2][0], 0.0)
                var tv = Vec3(params[j, 3][0], params[j, 4][0], params[j, 5][0], 0.0)
                var tw = Vec3(params[j, 6][0], params[j, 7][0], params[j, 8][0], 0.0)
                # The forward render's intersection and alpha, op for op.
                var k0 = fm.sub(fm.mul(tw[0], pixf_x), tu[0])
                var k1 = fm.sub(fm.mul(tw[1], pixf_x), tu[1])
                var k2 = fm.sub(fm.mul(tw[2], pixf_x), tu[2])
                var l0 = fm.sub(fm.mul(tw[0], pixf_y), tv[0])
                var l1 = fm.sub(fm.mul(tw[1], pixf_y), tv[1])
                var l2 = fm.sub(fm.mul(tw[2], pixf_y), tv[2])
                var p_x = fm.sub(fm.mul(k1, l2), fm.mul(l1, k2))
                var p_y = fm.sub(fm.mul(l0, k2), fm.mul(k0, l2))
                var p_z = fm.sub(fm.mul(k0, l1), fm.mul(l0, k1))
                if p_z != 0.0:
                    var u = fm.div(p_x, p_z)
                    var v = fm.div(p_y, p_z)
                    var depth = fm.add(tw[2], fm.fma(tw[0], u, fm.mul(tw[1], v)))
                    if not depth < NEAR:
                        var s0 = params[j, 10][0]
                        var s1 = params[j, 11][0]
                        var s2 = params[j, 12][0]
                        var d0 = abs(v)
                        var ua = fm.mul(u, 0.8660254)
                        var vb = fm.mul(v, 0.5)
                        var d1 = abs(fm.add(ua, vb))
                        var d2 = abs(fm.sub(ua, vb))
                        var slant_is_1 = d2 < d1
                        var slant = d1 if slant_is_1 else d2
                        var not_edge0 = d0 <= slant
                        var dist = fm.add(slant if not_edge0 else d0, -1.0)
                        var edge = (1 if slant_is_1 else 2) if not_edge0 else 0
                        var soft = (s1 if slant_is_1 else s2) if not_edge0 else s0
                        var boundary = fm.rcp(fm.add(fm.ex2(fm.mul(fm.mul(dist, soft), fm.LOG2E)), 1.0))
                        var opacity = params[j, 9][0]
                        var alpha_raw = fm.mul(opacity, boundary)
                        var clamped = alpha_raw > 0.99
                        var alpha = Float32(0.99) if clamped else alpha_raw
                        if not alpha < 1.0 / 255.0:
                            contributes = True
                            var g = Int(ids[j][0])
                            var t_before = t_after / (1.0 - alpha)
                            var wgt = alpha * t_before
                            var c = Vec3(colors[g, 0][0], colors[g, 1][0], colors[g, 2][0], 0.0)
                            var v_u: Float32 = 0.0
                            var v_v: Float32 = 0.0
                            g_color = v_out * wgt
                            comptime if color_model >= 1:
                                var gu = Vec3(0.0)
                                var gv = Vec3(0.0)
                                comptime for ch in range(3):
                                    gu[ch] = color_grads[g, ch][0]
                                    gv[ch] = color_grads[g, 3 + ch][0]
                                    c[ch] += u * gu[ch] + v * gv[ch]
                                    g_cgrad[ch] = wgt * v_out[ch] * u
                                    g_cgrad[3 + ch] = wgt * v_out[ch] * v
                                v_u += wgt * _dot3(v_out, gu)
                                v_v += wgt * _dot3(v_out, gv)
                            comptime if color_model >= 2:
                                var quu = Vec3(0.0)
                                var quv = Vec3(0.0)
                                var qvv = Vec3(0.0)
                                comptime for ch in range(3):
                                    quu[ch] = color_quads[g, ch][0]
                                    quv[ch] = color_quads[g, 3 + ch][0]
                                    qvv[ch] = color_quads[g, 6 + ch][0]
                                var qu = u
                                var qv = v
                                comptime if color_model == 3:
                                    qu, qv = _bounded(u, v, dist)
                                var u2 = qu * qu
                                var uv = qu * qv
                                var v2 = qv * qv
                                comptime for ch in range(3):
                                    c[ch] += u2 * quu[ch] + uv * quv[ch] + v2 * qvv[ch]
                                    g_quad[ch] = wgt * v_out[ch] * u2
                                    g_quad[3 + ch] = wgt * v_out[ch] * uv
                                    g_quad[6 + ch] = wgt * v_out[ch] * v2
                                var du_q = Vec3(0.0)
                                var dv_q = Vec3(0.0)
                                comptime for ch in range(3):
                                    du_q[ch] = 2.0 * qu * quu[ch] + qv * quv[ch]
                                    dv_q[ch] = qu * quv[ch] + 2.0 * qv * qvv[ch]
                                var v_qu = wgt * _dot3(v_out, du_q)
                                var v_qv = wgt * _dot3(v_out, dv_q)
                                comptime if color_model == 3:
                                    # bounded_quadratic_coordinate_vjp
                                    var gdist, d_rho_du, d_rho_dv, _ = _hex_sdf_grad(u, v)
                                    var rho = gdist + 1.0
                                    if rho <= 1.0:
                                        v_u += v_qu
                                        v_v += v_qv
                                    else:
                                        var inv_rho = 1.0 / rho
                                        var radial = (v_qu * u + v_qv * v) * inv_rho * inv_rho
                                        v_u += v_qu * inv_rho - radial * d_rho_du
                                        v_v += v_qv * inv_rho - radial * d_rho_dv
                                else:
                                    v_u += v_qu
                                    v_v += v_qv
                            var vd = Vec3(0.0)
                            comptime if use_view_grads or specular:
                                vd = Vec3(view_dirs[g, 0][0], view_dirs[g, 1][0], view_dirs[g, 2][0], 0.0)
                            comptime if use_view_grads:
                                var v_vd = Vec3(0.0)
                                comptime for ch in range(3):
                                    var vg = Vec3(view_grads[g, 3 * ch][0], view_grads[g, 3 * ch + 1][0], view_grads[g, 3 * ch + 2][0], 0.0)
                                    c[ch] += _dot3(vd, vg)
                                    comptime for k in range(3):
                                        g_vgrad[3 * ch + k] = wgt * v_out[ch] * vd[k]
                                    v_vd += vg * v_out[ch]
                                g_vdir += v_vd * wgt
                            comptime if specular:
                                var nv = Vec3(normals[g, 0][0], normals[g, 1][0], normals[g, 2][0], 0.0)
                                var ks = Vec3(specular_ks[g, 0][0], specular_ks[g, 1][0], specular_ks[g, 2][0], 0.0)
                                var log_s = specular_log_s[g, 0][0]
                                var cos_nv = max(-_dot3(nv, vd), Float32(0.0))
                                var log_s_is_clamped = log_s < 0.0 or log_s > 5.0
                                var s_exp = fm.ex2(clamp(log_s, 0.0, 5.0) * fm.LOG2E)
                                var spec: Float32 = 0.0
                                if cos_nv > 0.0:
                                    spec = fm.ex2(s_exp * fm.lg2(cos_nv))
                                comptime for ch in range(3):
                                    c[ch] += ks[ch] * spec
                                g_ks = v_out * (wgt * spec)
                                var v_spec = wgt * _dot3(v_out, ks)
                                if not log_s_is_clamped and cos_nv > 0.0:
                                    g_logs = v_spec * (spec * (fm.lg2(cos_nv) * fm.LN2) * s_exp)
                                if v_spec != 0.0 and cos_nv > 0.0:
                                    var v_cos = v_spec * (s_exp * spec / cos_nv)
                                    g_vdir -= nv * v_cos
                                    g_normal = vd * (-v_cos)
                            var v_alpha = t_before * _dot3(v_out, c - behind)
                            if not clamped:
                                g_opacity = v_alpha * boundary
                                var v_softness = v_alpha * opacity * (-dist) * boundary * (1.0 - boundary)
                                g_soft[edge if stride == 3 else 0] = v_softness
                                var _gd, d_dist_du, d_dist_dv, edge_g = _hex_sdf_grad(u, v)
                                var soft_g = (s1 if edge_g == 1 else s2) if edge_g != 0 else s0
                                var v_alpha_dist = v_alpha * opacity * (-soft_g * boundary * (1.0 - boundary))
                                v_u += v_alpha_dist * d_dist_du
                                v_v += v_alpha_dist * d_dist_dv
                            # d(u, v) / dT through p = k x l.
                            var kv = Vec3(pixf_x * tw[0] - tu[0], pixf_x * tw[1] - tu[1], pixf_x * tw[2] - tu[2], 0.0)
                            var lv = Vec3(pixf_y * tw[0] - tv[0], pixf_y * tw[1] - tv[1], pixf_y * tw[2] - tv[2], 0.0)
                            var pz = kv[0] * lv[1] - kv[1] * lv[0]
                            if pz != 0.0:
                                var inv_pz = 1.0 / pz
                                var dsx = v_u * inv_pz
                                var dsy = v_v * inv_pz
                                var v_p = Vec3(dsx, dsy, -(dsx * u + dsy * v), 0.0)
                                var v_k = _cross(lv, v_p)
                                var v_l = _cross(v_p, kv)
                                var gx = -v_k[2] * tw[2] * 0.5 * Float32(w)
                                var gy = -v_l[2] * tw[2] * 0.5 * Float32(h)
                                g_abs = fm.sqrt(gx * gx + gy * gy)
                                comptime for k in range(3):
                                    g_tm[k] = -v_k[k]
                                    g_tm[3 + k] = -v_l[k]
                                    g_tm[6 + k] = pixf_x * v_k[k] + pixf_y * v_l[k]
                            behind = c * alpha + behind * (1.0 - alpha)
                            t_after = t_before
            # Reconverged: sum hexel j's gradients across the warp; lane 0 writes them.
            if warp.max(Int32(1) if contributes else Int32(0)) != 0:
                var g = Int(ids[j][0])
                var lead = lane_id() == 0
                comptime for ch in range(3):
                    var t = warp.sum(g_color[ch])
                    if lead:
                        atomic_add(grad_colors, g, ch, t)
                comptime if color_model >= 1:
                    comptime for k in range(6):
                        var t = warp.sum(g_cgrad[k])
                        if lead:
                            atomic_add(grad_color_grads, g, k, t)
                comptime if color_model >= 2:
                    comptime for k in range(9):
                        var t = warp.sum(g_quad[k])
                        if lead:
                            atomic_add(grad_color_quads, g, k, t)
                comptime if use_view_grads:
                    comptime for k in range(9):
                        var t = warp.sum(g_vgrad[k])
                        if lead:
                            atomic_add(grad_view_grads, g, k, t)
                comptime if use_view_grads or specular:
                    comptime for k in range(3):
                        var t = warp.sum(g_vdir[k])
                        if lead:
                            atomic_add(grad_view_dirs, g, k, t)
                comptime if specular:
                    var tl = warp.sum(g_logs)
                    if lead:
                        atomic_add(grad_specular_log_s, g, 0, tl)
                    comptime for k in range(3):
                        var t = warp.sum(g_ks[k])
                        var tn = warp.sum(g_normal[k])
                        if lead:
                            atomic_add(grad_specular_ks, g, k, t)
                            atomic_add(grad_normals, g, k, tn)
                var t_op = warp.sum(g_opacity)
                if lead:
                    atomic_add(grad_opacities, g, 0, t_op)
                if has_softness_grad != 0:
                    if stride == 3:
                        comptime for k in range(3):
                            var t = warp.sum(g_soft[k])
                            if lead:
                                atomic_add(grad_softnesses, g, k, t)
                    else:
                        var t = warp.sum(g_soft[0])
                        if lead:
                            atomic_add(grad_softnesses, g, 0, t)
                comptime for k in range(9):
                    var t = warp.sum(g_tm[k])
                    if lead:
                        atomic_add(grad_transmats, g, k, t)
                if abs_means2d != 0:
                    var t = warp.sum(g_abs)
                    if lead:
                        atomic_add(grad_means2d, g, 3, t)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def means2d_backward_kernel(
    width: Int32,
    height: Int32,
    radii: In[.int32, Rows[1]],
    transmats: In[.float32, Rows[9]],
    grad_transmats: In[.float32, Rows[9]],
    grad_means2d: Out[.float32, Grid],
):
    """`transmat_to_means2d_backward`: the screen-space gradient densification reads from means2d.grad."""
    var idx = global_idx.x
    if idx >= Int(radii.dim[0]()) or radii[idx, 0][0] <= 0:
        return
    var depth = transmats[idx, 8][0]
    grad_means2d[idx, 0] = grad_transmats[idx, 2][0] * depth * 0.5 * Float32(width)
    grad_means2d[idx, 1] = grad_transmats[idx, 5][0] * depth * 0.5 * Float32(height)
    grad_means2d[idx, 2] = 0.0


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def geometry_backward_kernel(
    width: Int32,
    height: Int32,
    scale_modifier: Float32,
    has_normals: Int32,
    means: In[.float32, Rows[3]],
    scales: In[.float32, Rows[2]],
    rotations: In[.float32, Rows[4]],
    projmatrix: In[.float32, Rows[16]],
    campos: In[.float32, Rows[3]],
    radii: In[.int32, Rows[1]],
    grad_transmats: In[.float32, Rows[9]],
    grad_view_dirs: In[.float32, Rows[3]],
    grad_normals: In[.float32, Rows[3]],
    grad_means: Out[.float32, Rows[3]],
    grad_scales: Out[.float32, Rows[2]],
    grad_rotations: Out[.float32, Rows[4]],
):
    """`transmat_to_geometry_backward`. T = S2W^T P with S2W = [L0 | L1 | (mean, 1)] (4x3, L = R S) and P = world2ndc
    ndc2pix (4x3), so dL/dS2W = P dL/dT^T: its columns are the gradients of L0, L1 and the mean."""
    var idx = global_idx.x
    if idx >= Int(radii.dim[0]()) or radii[idx, 0][0] <= 0:
        return
    var hw = Float32(width) / 2.0
    var ow = (Float32(width) - 1.0) / 2.0
    var hh = Float32(height) / 2.0
    var oh = (Float32(height) - 1.0) / 2.0
    var v_m = SIMD[.float32, 16](0.0)  # v_m[4 c + r] = dL/dS2W[r][c], rows 0..2; grad_transmats[idx, 3 c + r] = dL/dT[r][c]
    comptime for r in range(3):
        var pm0 = projmatrix[0, 4 * r][0] * hw + projmatrix[0, 4 * r + 3][0] * ow
        var pm1 = projmatrix[0, 4 * r + 1][0] * hh + projmatrix[0, 4 * r + 3][0] * oh
        var pm2 = projmatrix[0, 4 * r + 3][0]
        comptime for c in range(3):
            v_m[4 * c + r] = pm0 * grad_transmats[idx, c][0] + pm1 * grad_transmats[idx, 3 + c][0] + pm2 * grad_transmats[idx, 6 + c][0]
    # dnormvdv3: the view direction's normalization.
    var d = Vec3(means[idx, 0][0], means[idx, 1][0], means[idx, 2][0], 0.0) - Vec3(
        campos[0, 0][0], campos[0, 1][0], campos[0, 2][0], 0.0
    )
    var dv = Vec3(grad_view_dirs[idx, 0][0], grad_view_dirs[idx, 1][0], grad_view_dirs[idx, 2][0], 0.0)
    var sum2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 1e-8
    var invsum32 = fm.rsqrt(sum2 * sum2 * sum2)
    grad_means[idx, 0] = v_m[8] + ((sum2 - d[0] * d[0]) * dv[0] - d[1] * d[0] * dv[1] - d[2] * d[0] * dv[2]) * invsum32
    grad_means[idx, 1] = v_m[9] + (-d[0] * d[1] * dv[0] + (sum2 - d[1] * d[1]) * dv[1] - d[2] * d[1] * dv[2]) * invsum32
    grad_means[idx, 2] = v_m[10] + (-d[0] * d[2] * dv[0] - d[1] * d[2] * dv[1] + (sum2 - d[2] * d[2]) * dv[2]) * invsum32

    # Quaternion stored (w, x, y, z); R's columns as quat_to_rotmat.
    var q = SIMD[.float32, 4](rotations[idx, 0][0], rotations[idx, 1][0], rotations[idx, 2][0], rotations[idx, 3][0])
    var s = fm.rsqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    var qw = q[0] * s
    var qx = q[1] * s
    var qy = q[2] * s
    var qz = q[3] * s
    var r0 = Vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy), 0.0)
    var r1 = Vec3(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx), 0.0)
    var sx = scales[idx, 0][0]
    var sy = scales[idx, 1][0]
    var v_rs0 = Vec3(v_m[0], v_m[1], v_m[2], 0.0)
    var v_rs1 = Vec3(v_m[4], v_m[5], v_m[6], 0.0)
    var vr0 = v_rs0 * (scale_modifier * sx)  # v_R columns
    var vr1 = v_rs1 * (scale_modifier * sy)
    var vr2 = Vec3(0.0)
    if has_normals != 0:
        vr2 = Vec3(grad_normals[idx, 0][0], grad_normals[idx, 1][0], grad_normals[idx, 2][0], 0.0)
    # quat_to_rotmat_vjp, v_R[i][j] = column i, row j.
    var vq = SIMD[.float32, 4](
        2.0 * (qx * (vr1[2] - vr2[1]) + qy * (vr2[0] - vr0[2]) + qz * (vr0[1] - vr1[0])),
        2.0 * (-2.0 * qx * (vr1[1] + vr2[2]) + qy * (vr0[1] + vr1[0]) + qz * (vr0[2] + vr2[0]) + qw * (vr1[2] - vr2[1])),
        2.0 * (qx * (vr0[1] + vr1[0]) - 2.0 * qy * (vr0[0] + vr2[2]) + qz * (vr1[2] + vr2[1]) + qw * (vr2[0] - vr0[2])),
        2.0 * (qx * (vr0[2] + vr2[0]) + qy * (vr1[2] + vr2[1]) - 2.0 * qz * (vr0[0] + vr1[1]) + qw * (vr0[1] - vr1[0])),
    )
    var q_unit = q * s
    var g_rot = (vq - q_unit * (q_unit * vq).reduce_add()) * s
    comptime for k in range(4):
        grad_rotations[idx, k] = g_rot[k]
    grad_scales[idx, 0] = _dot3(v_rs0, r0) * scale_modifier
    grad_scales[idx, 1] = _dot3(v_rs1, r1) * scale_modifier
