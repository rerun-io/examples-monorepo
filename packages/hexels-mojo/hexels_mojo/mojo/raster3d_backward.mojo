"""hexel-kernel's 3D hexel rasterizer, cached backward (`gsplat/cuda_hexels3d/csrc/hexels3d_forward.cu`).

Covered: `render_hexels3d_backward<kHasSurfaceGrads=false, kPrefilter=false>` with the forward's saved final
transmittance and last index (one thread per pixel walks its tile list back to front and scatters float atomics into
the per-hexel gradients; every color model, view gradients, specular lobe, the optional |d pixel| statistic of
(P, 4) means2d), `transmat_to_means2d_backward` and `transmat_to_geometry_backward` (the splat-to-pixel transform's
gradient back to means, scales and quaternions). The ray-hexel intersection and alpha follow the forward render's
fast-math ops (`raster3d.render_kernel`), so the backward revisits exactly the hexels the forward composited. Float
atomics reorder sums, so the gradients match the CUDA build to a tolerance, not bit for bit. Not ported: surface /
overdraw gradients and prefiltering (they stay CUDA).
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from std.utils.static_tuple import StaticTuple

import fastmath as fm
from raster2d import F32Ptr, I32Ptr, TILE, _atomic_add
from raster3d import NEAR, POINT_THREADS, Vec3

comptime HEX_A: Float32 = 0.8660254  # 0.5 * sqrt(3)
comptime HEX_B: Float32 = 0.5


@always_inline
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


@always_inline
def _bounded(u: Float32, v: Float32, dist: Float32) -> Tuple[Float32, Float32]:
    """`bounded_quadratic_coordinates`: (u, v) scaled back onto the hexagon when outside it (rho = dist + 1)."""
    var rho = dist + 1.0
    var inv = 1.0 / rho if rho > 1.0 else Float32(1.0)
    return (u * inv, v * inv)


@always_inline
def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.0)


@always_inline
def _dot3(a: Vec3, b: Vec3) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def render_backward_kernel[color_model: Int, use_view_grads: Bool, specular: Bool](
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
    has_softness_grad: Int32,
    abs_means2d: Int32,
    background: F32Ptr,
    grad_out_color: F32Ptr,
    final_transmittance: F32Ptr,
    last_index: I32Ptr,
    grad_colors: F32Ptr,
    grad_color_grads: F32Ptr,
    grad_color_quads: F32Ptr,
    grad_view_grads: F32Ptr,
    grad_specular_ks: F32Ptr,
    grad_specular_log_s: F32Ptr,
    grad_opacities: F32Ptr,
    grad_softnesses: F32Ptr,
    grad_transmats: F32Ptr,
    grad_means2d: F32Ptr,
    grad_view_dirs: F32Ptr,
    grad_normals: F32Ptr,
):
    var w = Int(width)
    var h = Int(height)
    var px = block_idx.x * TILE + thread_idx.x
    var py = block_idx.y * TILE + thread_idx.y
    if px >= w or py >= h:
        return
    var tile = block_idx.y * ((w + TILE - 1) // TILE) + block_idx.x
    var start = Int(ranges.unsafe_load(2 * tile))
    if Int(ranges.unsafe_load(2 * tile + 1)) <= start:
        return
    var pix = py * w + px
    var pixf_x = Float32(px)
    var pixf_y = Float32(py)
    var v_out = Vec3(grad_out_color.unsafe_load(pix), grad_out_color.unsafe_load(h * w + pix), grad_out_color.unsafe_load(2 * h * w + pix), 0.0)
    var t_after = final_transmittance.unsafe_load(pix)
    var behind = Vec3(background.unsafe_load(0), background.unsafe_load(1), background.unsafe_load(2), 0.0)
    var stride = Int(softness_stride)
    var idx = Int(last_index.unsafe_load(pix)) - 1
    # Nested ifs, not continue: the lanes of a warp reconverge after every hexel (see raster3d.render_kernel).
    while idx >= 0:
        var g = Int(point_list.unsafe_load(start + idx))
        var tm = transmats.unsafe_offset(9 * g)
        var tu = Vec3(tm.unsafe_load(0), tm.unsafe_load(1), tm.unsafe_load(2), 0.0)
        var tv = Vec3(tm.unsafe_load(3), tm.unsafe_load(4), tm.unsafe_load(5), 0.0)
        var tw = Vec3(tm.unsafe_load(6), tm.unsafe_load(7), tm.unsafe_load(8), 0.0)
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
                var s0: Float32
                var s1: Float32
                var s2: Float32
                if stride == 3:
                    s0 = softnesses.unsafe_load(3 * g)
                    s1 = softnesses.unsafe_load(3 * g + 1)
                    s2 = softnesses.unsafe_load(3 * g + 2)
                else:
                    s0 = softnesses.unsafe_load(g)
                    s1 = s0
                    s2 = s0
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
                var opacity = opacities.unsafe_load(g)
                var alpha_raw = fm.mul(opacity, boundary)
                var clamped = alpha_raw > 0.99
                var alpha = Float32(0.99) if clamped else alpha_raw
                if not alpha < 1.0 / 255.0:
                    var t_before = t_after / (1.0 - alpha)
                    var wgt = alpha * t_before
                    var c = Vec3(colors.unsafe_load(3 * g), colors.unsafe_load(3 * g + 1), colors.unsafe_load(3 * g + 2), 0.0)
                    var gu = Vec3(0.0)
                    var gv = Vec3(0.0)
                    var quu = Vec3(0.0)
                    var quv = Vec3(0.0)
                    var qvv = Vec3(0.0)
                    var v_u: Float32 = 0.0
                    var v_v: Float32 = 0.0
                    comptime for ch in range(3):
                        _atomic_add(grad_colors, 3 * g + ch, wgt * v_out[ch])
                    comptime if color_model >= 1:
                        comptime for ch in range(3):
                            gu[ch] = color_grads.unsafe_load(6 * g + ch)
                            gv[ch] = color_grads.unsafe_load(6 * g + 3 + ch)
                            c[ch] += u * gu[ch] + v * gv[ch]
                            _atomic_add(grad_color_grads, 6 * g + ch, wgt * v_out[ch] * u)
                            _atomic_add(grad_color_grads, 6 * g + 3 + ch, wgt * v_out[ch] * v)
                        v_u += wgt * _dot3(v_out, gu)
                        v_v += wgt * _dot3(v_out, gv)
                    comptime if color_model >= 2:
                        comptime for ch in range(3):
                            quu[ch] = color_quads.unsafe_load(9 * g + ch)
                            quv[ch] = color_quads.unsafe_load(9 * g + 3 + ch)
                            qvv[ch] = color_quads.unsafe_load(9 * g + 6 + ch)
                        var qu = u
                        var qv = v
                        comptime if color_model == 3:
                            qu, qv = _bounded(u, v, dist)
                        var u2 = qu * qu
                        var uv = qu * qv
                        var v2 = qv * qv
                        comptime for ch in range(3):
                            c[ch] += u2 * quu[ch] + uv * quv[ch] + v2 * qvv[ch]
                            _atomic_add(grad_color_quads, 9 * g + ch, wgt * v_out[ch] * u2)
                            _atomic_add(grad_color_quads, 9 * g + 3 + ch, wgt * v_out[ch] * uv)
                            _atomic_add(grad_color_quads, 9 * g + 6 + ch, wgt * v_out[ch] * v2)
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
                        vd = Vec3(view_dirs.unsafe_load(3 * g), view_dirs.unsafe_load(3 * g + 1), view_dirs.unsafe_load(3 * g + 2), 0.0)
                    comptime if use_view_grads:
                        var v_vd = Vec3(0.0)
                        comptime for ch in range(3):
                            var vg = Vec3(view_grads.unsafe_load(9 * g + 3 * ch), view_grads.unsafe_load(9 * g + 3 * ch + 1), view_grads.unsafe_load(9 * g + 3 * ch + 2), 0.0)
                            c[ch] += _dot3(vd, vg)
                            comptime for k in range(3):
                                _atomic_add(grad_view_grads, 9 * g + 3 * ch + k, wgt * v_out[ch] * vd[k])
                            v_vd += vg * v_out[ch]
                        comptime for k in range(3):
                            _atomic_add(grad_view_dirs, 3 * g + k, wgt * v_vd[k])
                    comptime if specular:
                        var n = Vec3(normals.unsafe_load(3 * g), normals.unsafe_load(3 * g + 1), normals.unsafe_load(3 * g + 2), 0.0)
                        var ks = Vec3(specular_ks.unsafe_load(3 * g), specular_ks.unsafe_load(3 * g + 1), specular_ks.unsafe_load(3 * g + 2), 0.0)
                        var log_s = specular_log_s.unsafe_load(g)
                        var cos_nv = max(-_dot3(n, vd), Float32(0.0))
                        var log_s_is_clamped = log_s < 0.0 or log_s > 5.0
                        var s_exp = fm.ex2(min(max(log_s, Float32(0.0)), Float32(5.0)) * fm.LOG2E)
                        var spec: Float32 = 0.0
                        if cos_nv > 0.0:
                            spec = fm.ex2(s_exp * fm.lg2(cos_nv))
                        comptime for ch in range(3):
                            c[ch] += ks[ch] * spec
                            _atomic_add(grad_specular_ks, 3 * g + ch, wgt * v_out[ch] * spec)
                        var v_spec = wgt * _dot3(v_out, ks)
                        if not log_s_is_clamped and cos_nv > 0.0:
                            _atomic_add(grad_specular_log_s, g, v_spec * (spec * (fm.lg2(cos_nv) * fm.LN2) * s_exp))
                        if v_spec != 0.0 and cos_nv > 0.0:
                            var v_cos = v_spec * (s_exp * spec / cos_nv)
                            comptime for k in range(3):
                                _atomic_add(grad_view_dirs, 3 * g + k, -v_cos * n[k])
                                _atomic_add(grad_normals, 3 * g + k, -v_cos * vd[k])
                    var v_alpha = t_before * _dot3(v_out, c - behind)
                    if not clamped:
                        _atomic_add(grad_opacities, g, v_alpha * boundary)
                        if has_softness_grad != 0:
                            var v_softness = v_alpha * opacity * (-dist) * boundary * (1.0 - boundary)
                            if v_softness != 0.0:
                                _atomic_add(grad_softnesses, 3 * g + edge if stride == 3 else g, v_softness)
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
                        if abs_means2d != 0:
                            var gx = -v_k[2] * tw[2] * 0.5 * Float32(w)
                            var gy = -v_l[2] * tw[2] * 0.5 * Float32(h)
                            _atomic_add(grad_means2d, 4 * g + 3, fm.sqrt(gx * gx + gy * gy))
                        comptime for k in range(3):
                            _atomic_add(grad_transmats, 9 * g + k, -v_k[k])
                            _atomic_add(grad_transmats, 9 * g + 3 + k, -v_l[k])
                            _atomic_add(grad_transmats, 9 * g + 6 + k, pixf_x * v_k[k] + pixf_y * v_l[k])
                    behind = c * alpha + behind * (1.0 - alpha)
                    t_after = t_before
        idx -= 1


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def means2d_backward_kernel(
    num_points: Int32, width: Int32, height: Int32, stride: Int32, radii: I32Ptr, transmats: F32Ptr, grad_transmats: F32Ptr, grad_means2d: F32Ptr
):
    """`transmat_to_means2d_backward`: the screen-space gradient densification reads from means2d.grad."""
    var idx = global_idx.x
    if idx >= Int(num_points) or radii.unsafe_load(idx) <= 0:
        return
    var depth = transmats.unsafe_load(9 * idx + 8)
    var s = Int(stride)
    grad_means2d[unsafe_offset=s * idx] = grad_transmats.unsafe_load(9 * idx + 2) * depth * 0.5 * Float32(Int(width))
    grad_means2d[unsafe_offset=s * idx + 1] = grad_transmats.unsafe_load(9 * idx + 5) * depth * 0.5 * Float32(Int(height))
    grad_means2d[unsafe_offset=s * idx + 2] = 0.0


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POINT_THREADS))
def geometry_backward_kernel(
    num_points: Int32,
    width: Int32,
    height: Int32,
    scale_modifier: Float32,
    has_normals: Int32,
    means: F32Ptr,
    scales: F32Ptr,
    rotations: F32Ptr,
    projmatrix: F32Ptr,
    campos: F32Ptr,
    radii: I32Ptr,
    grad_transmats: F32Ptr,
    grad_view_dirs: F32Ptr,
    grad_normals: F32Ptr,
    grad_means: F32Ptr,
    grad_scales: F32Ptr,
    grad_rotations: F32Ptr,
):
    """`transmat_to_geometry_backward`. T = S2W^T P with S2W = [L0 | L1 | (mean, 1)] (4x3, L = R S) and P = world2ndc
    ndc2pix (4x3), so dL/dS2W = P dL/dT^T: its columns are the gradients of L0, L1 and the mean."""
    var idx = global_idx.x
    if idx >= Int(num_points) or radii.unsafe_load(idx) <= 0:
        return
    var hw = Float32(Int(width)) / 2.0
    var ow = (Float32(Int(width)) - 1.0) / 2.0
    var hh = Float32(Int(height)) / 2.0
    var oh = (Float32(Int(height)) - 1.0) / 2.0
    var gt = grad_transmats.unsafe_offset(9 * idx)  # gt[3 c + r] = dL/dT[r][c]
    var v_m = SIMD[DType.float32, 16](0.0)  # v_m[4 c + r] = dL/dS2W[r][c], rows 0..2
    comptime for r in range(3):
        var pm0 = projmatrix.unsafe_load(4 * r) * hw + projmatrix.unsafe_load(4 * r + 3) * ow
        var pm1 = projmatrix.unsafe_load(4 * r + 1) * hh + projmatrix.unsafe_load(4 * r + 3) * oh
        var pm2 = projmatrix.unsafe_load(4 * r + 3)
        comptime for c in range(3):
            v_m[4 * c + r] = pm0 * gt.unsafe_load(c) + pm1 * gt.unsafe_load(3 + c) + pm2 * gt.unsafe_load(6 + c)
    # dnormvdv3: the view direction's normalization.
    var d = Vec3(means.unsafe_load(3 * idx), means.unsafe_load(3 * idx + 1), means.unsafe_load(3 * idx + 2), 0.0) - Vec3(
        campos.unsafe_load(0), campos.unsafe_load(1), campos.unsafe_load(2), 0.0
    )
    var dv = Vec3(grad_view_dirs.unsafe_load(3 * idx), grad_view_dirs.unsafe_load(3 * idx + 1), grad_view_dirs.unsafe_load(3 * idx + 2), 0.0)
    var sum2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 1e-8
    var invsum32 = fm.rsqrt(sum2 * sum2 * sum2)
    grad_means[unsafe_offset=3 * idx] = v_m[8] + ((sum2 - d[0] * d[0]) * dv[0] - d[1] * d[0] * dv[1] - d[2] * d[0] * dv[2]) * invsum32
    grad_means[unsafe_offset=3 * idx + 1] = v_m[9] + (-d[0] * d[1] * dv[0] + (sum2 - d[1] * d[1]) * dv[1] - d[2] * d[1] * dv[2]) * invsum32
    grad_means[unsafe_offset=3 * idx + 2] = v_m[10] + (-d[0] * d[2] * dv[0] - d[1] * d[2] * dv[1] + (sum2 - d[2] * d[2]) * dv[2]) * invsum32

    # Quaternion stored (w, x, y, z); R's columns as quat_to_rotmat.
    var q = SIMD[DType.float32, 4](rotations.unsafe_load(4 * idx), rotations.unsafe_load(4 * idx + 1), rotations.unsafe_load(4 * idx + 2), rotations.unsafe_load(4 * idx + 3))
    var s = fm.rsqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    var qw = q[0] * s
    var qx = q[1] * s
    var qy = q[2] * s
    var qz = q[3] * s
    var r0 = Vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy), 0.0)
    var r1 = Vec3(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx), 0.0)
    var sx = scales.unsafe_load(2 * idx)
    var sy = scales.unsafe_load(2 * idx + 1)
    var v_rs0 = Vec3(v_m[0], v_m[1], v_m[2], 0.0)
    var v_rs1 = Vec3(v_m[4], v_m[5], v_m[6], 0.0)
    var vr0 = v_rs0 * (scale_modifier * sx)  # v_R columns
    var vr1 = v_rs1 * (scale_modifier * sy)
    var vr2 = Vec3(0.0)
    if has_normals != 0:
        vr2 = Vec3(grad_normals.unsafe_load(3 * idx), grad_normals.unsafe_load(3 * idx + 1), grad_normals.unsafe_load(3 * idx + 2), 0.0)
    # quat_to_rotmat_vjp, v_R[i][j] = column i, row j.
    var vq = SIMD[DType.float32, 4](
        2.0 * (qx * (vr1[2] - vr2[1]) + qy * (vr2[0] - vr0[2]) + qz * (vr0[1] - vr1[0])),
        2.0 * (-2.0 * qx * (vr1[1] + vr2[2]) + qy * (vr0[1] + vr1[0]) + qz * (vr0[2] + vr2[0]) + qw * (vr1[2] - vr2[1])),
        2.0 * (qx * (vr0[1] + vr1[0]) - 2.0 * qy * (vr0[0] + vr2[2]) + qz * (vr1[2] + vr2[1]) + qw * (vr2[0] - vr0[2])),
        2.0 * (qx * (vr0[2] + vr2[0]) + qy * (vr1[2] + vr2[1]) - 2.0 * qz * (vr0[0] + vr1[1]) + qw * (vr0[1] - vr1[0])),
    )
    var q_unit = q * s
    var g_rot = (vq - q_unit * (q_unit * vq).reduce_add()) * s
    comptime for k in range(4):
        grad_rotations[unsafe_offset=4 * idx + k] = g_rot[k]
    grad_scales[unsafe_offset=2 * idx] = _dot3(v_rs0, r0) * scale_modifier
    grad_scales[unsafe_offset=2 * idx + 1] = _dot3(v_rs1, r1) * scale_modifier
