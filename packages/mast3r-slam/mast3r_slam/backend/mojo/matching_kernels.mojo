"""Mojo GPU implementations of mast3r-slam matching kernels.

Provides `iter_proj` and `refine_matches` as drop-in replacements
for the CUDA implementations in mast3r_slam_backends.

Build: mojo build --emit shared-lib -o mast3r_slam_mojo_backends.so \
           mast3r_slam/backend/mojo/matching_kernels.mojo
"""

from gn_kernels import GN_IMPORT_SENTINEL

from std.os import abort
from std.math import ceildiv, sqrt, min, max, floor, exp, sin, cos, abs
from std.sys import has_accelerator
from std.gpu import global_idx, block_idx, block_dim, thread_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.memory import alloc
from std.memory import stack_allocation
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

# ─── Launch heuristics ────────────────────────────────────────────────────────
# The CUDA reference uses 16 threads. Mojo benefits from larger blocks on large
# workloads, but that hurts the smaller 64x64 iter_proj benchmark where wrapper
# overhead dominates. Pick block sizes per workload instead of forcing a single
# compile-time choice for every case.
def choose_iter_proj_block(num_pts: Int) -> Int:
    if num_pts <= 4096:
        return 16
    if num_pts <= 16384:
        return 64
    return 128


def choose_refine_block(num_pts: Int, fdim: Int) -> Int:
    if num_pts <= 1024:
        return 8
    return 16


def get_cached_context_ptr() raises -> UnsafePointer[DeviceContext, MutAnyOrigin]:
    """Return the process-lifetime DeviceContext storage for this extension."""
    var module = Python.import_module("mast3r_slam_mojo_backends")
    var ctx_addr = Int(py=module._ctx_addr)
    return UnsafePointer[DeviceContext, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )


def get_torch_module() raises -> PythonObject:
    return Python.import_module("torch")


def get_cuda_backend_module() raises -> PythonObject:
    return Python.import_module("mast3r_slam._backends")


def get_lietorch_module() raises -> PythonObject:
    return Python.import_module("lietorch")


comptime POSE_DIM = 7
comptime POSE_STRIDE = 8
comptime RAYS_HDIM = 14 * (14 + 1) // 2
comptime RAYS_THREADS = 64


def get_unique_kf_idx(ii: PythonObject, jj: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    return torch.unique(torch.cat(Python.list(ii, jj)), sorted=True)


def create_inds(
    unique_kf_idx: PythonObject,
    pin: Int,
    ii: PythonObject,
    jj: PythonObject,
) raises -> PythonObject:
    var torch = get_torch_module()
    var ii_ind = torch.searchsorted(unique_kf_idx, ii) - pin
    var jj_ind = torch.searchsorted(unique_kf_idx, jj) - pin
    return Python.tuple(ii_ind, jj_ind)


def assemble_dense_system(
    hs: PythonObject,
    gs: PythonObject,
    ii_opt: PythonObject,
    jj_opt: PythonObject,
    n_vars: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    var flat_h = torch.zeros(
        Python.list(n_vars * n_vars, POSE_DIM * POSE_DIM),
        device=hs.device,
        dtype=hs.dtype,
    )
    var block_terms = torch.cat(Python.list(hs[0], hs[1], hs[2], hs[3]), dim=0).reshape(-1, POSE_DIM * POSE_DIM)
    var rows = torch.cat(Python.list(ii_opt, ii_opt, jj_opt, jj_opt), dim=0)
    var cols = torch.cat(Python.list(ii_opt, jj_opt, ii_opt, jj_opt), dim=0)
    var valid_h = (rows >= 0) & (cols >= 0)
    if Bool(py=valid_h.any().item()):
        var block_indices = rows[valid_h] * n_vars + cols[valid_h]
        flat_h.index_add_(0, block_indices, block_terms[valid_h])
    var h_dense = flat_h.view(n_vars, n_vars, POSE_DIM, POSE_DIM).permute(0, 2, 1, 3).reshape(
        n_vars * POSE_DIM,
        n_vars * POSE_DIM,
    )

    var b_dense = torch.zeros(
        Python.list(n_vars, POSE_DIM),
        device=gs.device,
        dtype=gs.dtype,
    )
    var rhs_terms = torch.cat(Python.list(gs[0], gs[1]), dim=0)
    var rhs_rows = torch.cat(Python.list(ii_opt, jj_opt), dim=0)
    var valid_b = rhs_rows >= 0
    if Bool(py=valid_b.any().item()):
        b_dense.index_add_(0, rhs_rows[valid_b], rhs_terms[valid_b])

    return Python.tuple(h_dense, b_dense)


def solve_dense_system(
    h_dense: PythonObject,
    b_dense: PythonObject,
    n_vars: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    if n_vars <= 0:
        return torch.zeros(Python.list(0, POSE_DIM), device=h_dense.device, dtype=h_dense.dtype)

    var h64 = h_dense.to(dtype=torch.float64)
    var b64 = b_dense.to(dtype=torch.float64)
    var rhs = -b64.reshape(n_vars * POSE_DIM, 1)
    var eye = torch.eye(n_vars * POSE_DIM, device=h_dense.device, dtype=torch.float64)
    var chol_h = h64 + 1e-9 * eye
    var chol = torch.linalg.cholesky_ex(chol_h, upper=False)
    var l_factor = chol[0]
    var info = chol[1]
    if Int(py=info.max().item()) != 0:
        return torch.zeros(Python.list(n_vars, POSE_DIM), device=h_dense.device, dtype=h_dense.dtype)
    var dx_vec = torch.cholesky_solve(rhs, l_factor, upper=False)
    return dx_vec.to(dtype=h_dense.dtype).view(n_vars, POSE_DIM)


def assemble_h_dense_kernel(
    hs: UnsafePointer[Float32, MutAnyOrigin],
    ii_opt: UnsafePointer[Int64, MutAnyOrigin],
    jj_opt: UnsafePointer[Int64, MutAnyOrigin],
    h_dense: UnsafePointer[Float64, MutAnyOrigin],
    num_edges: Int,
    n_vars: Int,
):
    var row: Int = Int(global_idx.x)
    var col: Int = Int(global_idx.y)
    var dim: Int = n_vars * POSE_DIM
    if row >= dim or col >= dim:
        return

    var bi = row // POSE_DIM
    var bj = col // POSE_DIM
    var ki = row % POSE_DIM
    var kj = col % POSE_DIM

    var acc: Float64 = 0.0
    for e in range(num_edges):
        var ii = Int(ii_opt[e])
        var jj = Int(jj_opt[e])
        if ii >= 0 and bj == ii and bi == ii:
            acc += Float64(hs[((0 * num_edges + e) * POSE_DIM + ki) * POSE_DIM + kj])
        if ii >= 0 and jj >= 0 and bi == ii and bj == jj:
            acc += Float64(hs[((1 * num_edges + e) * POSE_DIM + ki) * POSE_DIM + kj])
        if ii >= 0 and jj >= 0 and bi == jj and bj == ii:
            acc += Float64(hs[((2 * num_edges + e) * POSE_DIM + ki) * POSE_DIM + kj])
        if jj >= 0 and bi == jj and bj == jj:
            acc += Float64(hs[((3 * num_edges + e) * POSE_DIM + ki) * POSE_DIM + kj])
    h_dense[row * dim + col] = acc


def assemble_b_dense_kernel(
    gs: UnsafePointer[Float32, MutAnyOrigin],
    ii_opt: UnsafePointer[Int64, MutAnyOrigin],
    jj_opt: UnsafePointer[Int64, MutAnyOrigin],
    b_dense: UnsafePointer[Float64, MutAnyOrigin],
    num_edges: Int,
    n_vars: Int,
):
    var row: Int = Int(global_idx.x)
    var col: Int = Int(global_idx.y)
    if row >= n_vars or col >= POSE_DIM:
        return

    var acc: Float64 = 0.0
    for e in range(num_edges):
        var ii = Int(ii_opt[e])
        var jj = Int(jj_opt[e])
        if ii >= 0 and row == ii:
            acc += Float64(gs[(0 * num_edges + e) * POSE_DIM + col])
        if jj >= 0 and row == jj:
            acc += Float64(gs[(1 * num_edges + e) * POSE_DIM + col])
    b_dense[row * POSE_DIM + col] = acc


@always_inline
def quat_comp_components(
    aix: Float32, aiy: Float32, aiz: Float32, aiw: Float32,
    bjx: Float32, bjy: Float32, bjz: Float32, bjw: Float32,
) -> InlineArray[Float32, 4]:
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = aiw * bjx + aix * bjw + aiy * bjz - aiz * bjy
    out[1] = aiw * bjy - aix * bjz + aiy * bjw + aiz * bjx
    out[2] = aiw * bjz + aix * bjy - aiy * bjx + aiz * bjw
    out[3] = aiw * bjw - aix * bjx - aiy * bjy - aiz * bjz
    return out^


@always_inline
def act_so3_components(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    x0: Float32, x1: Float32, x2: Float32,
) -> InlineArray[Float32, 3]:
    var uv0 = 2.0 * (qy * x2 - qz * x1)
    var uv1 = 2.0 * (qz * x0 - qx * x2)
    var uv2 = 2.0 * (qx * x1 - qy * x0)
    var out = InlineArray[Float32, 3](fill=0.0)
    out[0] = x0 + qw * uv0 + (qy * uv2 - qz * uv1)
    out[1] = x1 + qw * uv1 + (qz * uv0 - qx * uv2)
    out[2] = x2 + qw * uv2 + (qx * uv1 - qy * uv0)
    return out^


@always_inline
def exp_so3_components(phi0: Float32, phi1: Float32, phi2: Float32) -> InlineArray[Float32, 4]:
    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta_p4 = theta_sq * theta_sq
    var imag: Float32
    var real: Float32
    if theta_sq < 1e-6:
        imag = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
        real = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_p4
    else:
        var theta = sqrt(theta_sq)
        imag = sin(0.5 * theta) / theta
        real = cos(0.5 * theta)
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = imag * phi0
    out[1] = imag * phi1
    out[2] = imag * phi2
    out[3] = real
    return out^


@always_inline
def exp_sim3_components(
    xi0: Float32, xi1: Float32, xi2: Float32, xi3: Float32, xi4: Float32, xi5: Float32, xi6: Float32,
) -> InlineArray[Float32, 8]:
    var tau0: Float32 = xi0
    var tau1: Float32 = xi1
    var tau2: Float32 = xi2
    var phi0: Float32 = xi3
    var phi1: Float32 = xi4
    var phi2: Float32 = xi5
    var sigma: Float32 = xi6
    var scale: Float32 = exp(sigma)
    var q = exp_so3_components(phi0, phi1, phi2)

    var theta_sq: Float32 = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta: Float32 = sqrt(theta_sq)
    var A: Float32
    var B: Float32
    var C: Float32
    if abs(sigma) < Float32(1e-6):
        C = 1.0
        if abs(theta) < Float32(1e-6):
            A = 0.5
            B = 1.0 / 6.0
        else:
            A = (1.0 - cos(theta)) / theta_sq
            B = (theta - sin(theta)) / (theta_sq * theta)
    else:
        C = (scale - 1.0) / sigma
        if abs(theta) < Float32(1e-6):
            var sigma_sq: Float32 = sigma * sigma
            A = ((sigma - 1.0) * scale + 1.0) / sigma_sq
            B = (scale * 0.5 * sigma_sq + scale - 1.0 - sigma * scale) / (sigma_sq * sigma)
        else:
            var a: Float32 = scale * sin(theta)
            var b: Float32 = scale * cos(theta)
            var c: Float32 = theta_sq + sigma * sigma
            A = (a * sigma + (1.0 - b) * theta) / (theta * c)
            B = (C - (((b - 1.0) * sigma + a * theta) / c)) / theta_sq

    var out = InlineArray[Float32, 8](fill=0.0)
    out[0] = C * tau0
    out[1] = C * tau1
    out[2] = C * tau2

    var cx0: Float32 = phi1 * tau2 - phi2 * tau1
    var cx1: Float32 = phi2 * tau0 - phi0 * tau2
    var cx2: Float32 = phi0 * tau1 - phi1 * tau0
    out[0] += A * cx0
    out[1] += A * cx1
    out[2] += A * cx2

    var c2x0: Float32 = phi1 * cx2 - phi2 * cx1
    var c2x1: Float32 = phi2 * cx0 - phi0 * cx2
    var c2x2: Float32 = phi0 * cx1 - phi1 * cx0
    out[0] += B * c2x0
    out[1] += B * c2x1
    out[2] += B * c2x2
    out[3] = q[0]
    out[4] = q[1]
    out[5] = q[2]
    out[6] = q[3]
    out[7] = scale
    return out^


@always_inline
def quat_inv_components(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
) -> InlineArray[Float32, 4]:
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = -qx
    out[1] = -qy
    out[2] = -qz
    out[3] = qw
    return out^


@always_inline
def act_sim3_components(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
) -> InlineArray[Float32, 3]:
    var rot = act_so3_components(qx, qy, qz, qw, x0, x1, x2)
    var out = InlineArray[Float32, 3](fill=0.0)
    out[0] = rot[0] * s + tx
    out[1] = rot[1] * s + ty
    out[2] = rot[2] * s + tz
    return out^


@always_inline
def dot3_components(
    a0: Float32, a1: Float32, a2: Float32,
    b0: Float32, b1: Float32, b2: Float32,
) -> Float32:
    return a0 * b0 + a1 * b1 + a2 * b2


@always_inline
def squared_norm3_components(x0: Float32, x1: Float32, x2: Float32) -> Float32:
    return x0 * x0 + x1 * x1 + x2 * x2


@always_inline
def huber_scalar(r: Float32) -> Float32:
    var r_abs = abs(r)
    if r_abs < 1.345:
        return 1.0
    return 1.345 / r_abs


@always_inline
def rel_sim3_components(
    tix: Float32, tiy: Float32, tiz: Float32,
    qix: Float32, qiy: Float32, qiz: Float32, qiw: Float32,
    si: Float32,
    tjx: Float32, tjy: Float32, tjz: Float32,
    qjx: Float32, qjy: Float32, qjz: Float32, qjw: Float32,
    sj: Float32,
) -> InlineArray[Float32, 8]:
    var si_inv = 1.0 / si
    var qi_inv = quat_inv_components(qix, qiy, qiz, qiw)
    var qij = quat_comp_components(qi_inv[0], qi_inv[1], qi_inv[2], qi_inv[3], qjx, qjy, qjz, qjw)
    var dt0 = tjx - tix
    var dt1 = tjy - tiy
    var dt2 = tjz - tiz
    var rot_t = act_so3_components(qi_inv[0], qi_inv[1], qi_inv[2], qi_inv[3], dt0, dt1, dt2)
    var out = InlineArray[Float32, 8](fill=0.0)
    out[0] = rot_t[0] * si_inv
    out[1] = rot_t[1] * si_inv
    out[2] = rot_t[2] * si_inv
    out[3] = qij[0]
    out[4] = qij[1]
    out[5] = qij[2]
    out[6] = qij[3]
    out[7] = si_inv * sj
    return out^


@always_inline
def apply_sim3_adj_inv_components(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    x3: Float32, x4: Float32, x5: Float32,
    x6: Float32,
) -> InlineArray[Float32, 7]:
    var s_inv = 1.0 / s
    var ra = act_so3_components(qx, qy, qz, qw, x0, x1, x2)
    var out = InlineArray[Float32, 7](fill=0.0)
    out[0] = s_inv * ra[0]
    out[1] = s_inv * ra[1]
    out[2] = s_inv * ra[2]

    var rb = act_so3_components(qx, qy, qz, qw, x3, x4, x5)
    out[3] = rb[0] + s_inv * (ty * ra[2] - tz * ra[1])
    out[4] = rb[1] + s_inv * (tz * ra[0] - tx * ra[2])
    out[5] = rb[2] + s_inv * (tx * ra[1] - ty * ra[0])
    out[6] = x6 + s_inv * dot3_components(tx, ty, tz, ra[0], ra[1], ra[2])
    return out^


def gauss_newton_rays_step_kernel(
    twc: UnsafePointer[Float32, MutAnyOrigin],
    xs: UnsafePointer[Float32, MutAnyOrigin],
    cs: UnsafePointer[Float32, MutAnyOrigin],
    ii: UnsafePointer[Int64, MutAnyOrigin],
    jj: UnsafePointer[Int64, MutAnyOrigin],
    idx_ii2jj: UnsafePointer[Int64, MutAnyOrigin],
    valid_match: UnsafePointer[UInt8, MutAnyOrigin],
    q_tensor: UnsafePointer[Float32, MutAnyOrigin],
    hs: UnsafePointer[Float32, MutAnyOrigin],
    gs: UnsafePointer[Float32, MutAnyOrigin],
    num_points: Int,
    num_edges: Int,
    sigma_ray: Float32,
    sigma_dist: Float32,
    c_thresh: Float32,
    q_thresh: Float32,
):
    var edge = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var block_threads = Int(block_dim.x)
    if edge >= num_edges:
        return

    var ix = Int(ii[edge])
    var jx = Int(jj[edge])
    var pi = ix * POSE_STRIDE
    var pj = jx * POSE_STRIDE

    var rel = rel_sim3_components(
        twc[pi + 0], twc[pi + 1], twc[pi + 2],
        twc[pi + 3], twc[pi + 4], twc[pi + 5], twc[pi + 6],
        twc[pi + 7],
        twc[pj + 0], twc[pj + 1], twc[pj + 2],
        twc[pj + 3], twc[pj + 4], twc[pj + 5], twc[pj + 6],
        twc[pj + 7],
    )

    var hij = InlineArray[Float32, RAYS_HDIM](fill=0.0)
    var vi = InlineArray[Float32, POSE_DIM](fill=0.0)
    var vj = InlineArray[Float32, POSE_DIM](fill=0.0)
    var jxv = InlineArray[Float32, 14](fill=0.0)

    var sigma_ray_inv = 1.0 / sigma_ray
    var sigma_dist_inv = 1.0 / sigma_dist

    for k in range(tid, num_points, block_threads):
        var valid_match_ind = valid_match[edge * num_points + k] != 0
        var ind_xi = Int(idx_ii2jj[edge * num_points + k])
        if not valid_match_ind:
            ind_xi = 0

        var xi_base = (ix * num_points + ind_xi) * 3
        var xj_base = (jx * num_points + k) * 3
        var ci = cs[ix * num_points + ind_xi]
        var cj = cs[jx * num_points + k]
        var qv = q_tensor[edge * num_points + k]

        var xi0 = xs[xi_base + 0]
        var xi1 = xs[xi_base + 1]
        var xi2 = xs[xi_base + 2]
        var xj0 = xs[xj_base + 0]
        var xj1 = xs[xj_base + 1]
        var xj2 = xs[xj_base + 2]

        var norm2_i = squared_norm3_components(xi0, xi1, xi2)
        var norm1_i = sqrt(norm2_i)
        var norm1_i_inv = 1.0 / norm1_i
        var ri0 = norm1_i_inv * xi0
        var ri1 = norm1_i_inv * xi1
        var ri2 = norm1_i_inv * xi2

        var xj_ci = act_sim3_components(
            rel[0], rel[1], rel[2], rel[3], rel[4], rel[5], rel[6], rel[7], xj0, xj1, xj2
        )
        var norm2_j = squared_norm3_components(xj_ci[0], xj_ci[1], xj_ci[2])
        var norm1_j = sqrt(norm2_j)
        var norm1_j_inv = 1.0 / norm1_j
        var rj0 = norm1_j_inv * xj_ci[0]
        var rj1 = norm1_j_inv * xj_ci[1]
        var rj2 = norm1_j_inv * xj_ci[2]

        var err0 = rj0 - ri0
        var err1 = rj1 - ri1
        var err2 = rj2 - ri2
        var err3 = norm1_j - norm1_i

        var valid = valid_match_ind and (qv > q_thresh) and (ci > c_thresh) and (cj > c_thresh)
        var conf_weight: Float32 = qv
        var sqrt_w_ray: Float32 = 0.0
        var sqrt_w_dist: Float32 = 0.0
        if valid:
            var sqrt_conf = sqrt(conf_weight)
            sqrt_w_ray = sigma_ray_inv * sqrt_conf
            sqrt_w_dist = sigma_dist_inv * sqrt_conf

        var w0 = huber_scalar(sqrt_w_ray * err0) * sqrt_w_ray * sqrt_w_ray
        var w1 = huber_scalar(sqrt_w_ray * err1) * sqrt_w_ray * sqrt_w_ray
        var w2 = huber_scalar(sqrt_w_ray * err2) * sqrt_w_ray * sqrt_w_ray
        var w3 = huber_scalar(sqrt_w_dist * err3) * sqrt_w_dist * sqrt_w_dist

        var norm3_j_inv = norm1_j_inv / norm2_j
        var drx_dpx = norm1_j_inv - xj_ci[0] * xj_ci[0] * norm3_j_inv
        var dry_dpy = norm1_j_inv - xj_ci[1] * xj_ci[1] * norm3_j_inv
        var drz_dpz = norm1_j_inv - xj_ci[2] * xj_ci[2] * norm3_j_inv
        var drx_dpy = -xj_ci[0] * xj_ci[1] * norm3_j_inv
        var drx_dpz = -xj_ci[0] * xj_ci[2] * norm3_j_inv
        var dry_dpz = -xj_ci[1] * xj_ci[2] * norm3_j_inv

        @parameter
        def accumulate_row(
            err: Float32,
            w: Float32,
            ji0: Float32, ji1: Float32, ji2: Float32, ji3: Float32, ji4: Float32, ji5: Float32, ji6: Float32,
        ):
            var jadj = apply_sim3_adj_inv_components(
                twc[pi + 0], twc[pi + 1], twc[pi + 2],
                twc[pi + 3], twc[pi + 4], twc[pi + 5], twc[pi + 6], twc[pi + 7],
                ji0, ji1, ji2, ji3, ji4, ji5, ji6,
            )
            for n in range(POSE_DIM):
                jxv[n + 7] = jadj[n]
                jxv[n] = -jadj[n]
            var l = 0
            for n in range(14):
                for m in range(n + 1):
                    hij[l] += w * jxv[n] * jxv[m]
                    l += 1
            for n in range(POSE_DIM):
                vi[n] += w * err * jxv[n]
                vj[n] += w * err * jxv[n + 7]

        accumulate_row(err0, w0, drx_dpx, drx_dpy, drx_dpz, 0.0, rj2, -rj1, 0.0)
        accumulate_row(err1, w1, drx_dpy, dry_dpy, dry_dpz, -rj2, 0.0, rj0, 0.0)
        accumulate_row(err2, w2, drx_dpz, dry_dpz, drz_dpz, rj1, -rj0, 0.0, 0.0)
        accumulate_row(err3, w3, rj0, rj1, rj2, 0.0, 0.0, 0.0, norm1_j)

    var sdata = stack_allocation[
        RAYS_THREADS,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    for n in range(POSE_DIM):
        sdata[tid] = vi[n]
        barrier()
        var active = block_threads // 2
        while active > 0:
            if tid < active:
                sdata[tid] += sdata[tid + active]
            barrier()
            active = active // 2
        if tid == 0:
            gs[(0 * num_edges + edge) * POSE_DIM + n] = sdata[0][0]

        sdata[tid] = vj[n]
        barrier()
        active = block_threads // 2
        while active > 0:
            if tid < active:
                sdata[tid] += sdata[tid + active]
            barrier()
            active = active // 2
        if tid == 0:
            gs[(1 * num_edges + edge) * POSE_DIM + n] = sdata[0][0]

    var l = 0
    for n in range(14):
        for m in range(n + 1):
            sdata[tid] = hij[l]
            barrier()
            var active_h = block_threads // 2
            while active_h > 0:
                if tid < active_h:
                    sdata[tid] += sdata[tid + active_h]
                barrier()
                active_h = active_h // 2
            if tid == 0:
                var val = sdata[0][0]
                if n < 7 and m < 7:
                    hs[((0 * num_edges + edge) * 49) + n * 7 + m] = val
                    hs[((0 * num_edges + edge) * 49) + m * 7 + n] = val
                elif n >= 7 and m < 7:
                    hs[((1 * num_edges + edge) * 49) + m * 7 + (n - 7)] = val
                    hs[((2 * num_edges + edge) * 49) + (n - 7) * 7 + m] = val
                else:
                    hs[((3 * num_edges + edge) * 49) + (n - 7) * 7 + (m - 7)] = val
                    hs[((3 * num_edges + edge) * 49) + (m - 7) * 7 + (n - 7)] = val
            l += 1


def pose_retr_kernel(
    poses: UnsafePointer[Float32, MutAnyOrigin],
    dx: UnsafePointer[Float32, MutAnyOrigin],
    num_fix: Int,
    num_poses: Int,
):
    var k: Int = Int(global_idx.x) + num_fix
    if k >= num_poses:
        return
    var xi = exp_sim3_components(
        dx[(k - num_fix) * POSE_DIM + 0],
        dx[(k - num_fix) * POSE_DIM + 1],
        dx[(k - num_fix) * POSE_DIM + 2],
        dx[(k - num_fix) * POSE_DIM + 3],
        dx[(k - num_fix) * POSE_DIM + 4],
        dx[(k - num_fix) * POSE_DIM + 5],
        dx[(k - num_fix) * POSE_DIM + 6],
    )
    var p = k * 8
    var rot_t = act_so3_components(
        xi[3], xi[4], xi[5], xi[6],
        poses[p + 0], poses[p + 1], poses[p + 2],
    )
    var q1 = quat_comp_components(
        xi[3], xi[4], xi[5], xi[6],
        poses[p + 3], poses[p + 4], poses[p + 5], poses[p + 6],
    )
    poses[p + 0] = rot_t[0] * xi[7] + xi[0]
    poses[p + 1] = rot_t[1] * xi[7] + xi[1]
    poses[p + 2] = rot_t[2] * xi[7] + xi[2]
    poses[p + 3] = q1[0]
    poses[p + 4] = q1[1]
    poses[p + 5] = q1[2]
    poses[p + 6] = q1[3]
    poses[p + 7] = xi[7] * poses[p + 7]


def tensor_col(x: PythonObject, idx: Int) raises -> PythonObject:
    return x.select(1, idx).unsqueeze(1)


def assemble_dense_system_gpu(
    hs: PythonObject,
    gs: PythonObject,
    ii_opt: PythonObject,
    jj_opt: PythonObject,
    n_vars: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    var num_edges = Int(py=hs.shape[1])
    var dim = n_vars * POSE_DIM
    var h_dense = torch.zeros(
        Python.list(dim, dim),
        device=hs.device,
        dtype=torch.float64,
    )
    var b_dense = torch.zeros(
        Python.list(n_vars, POSE_DIM),
        device=gs.device,
        dtype=torch.float64,
    )
    if n_vars <= 0:
        return Python.tuple(h_dense, b_dense)

    var hs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=hs.data_ptr()))
    var gs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=gs.data_ptr()))
    var ii_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=ii_opt.contiguous().data_ptr()))
    var jj_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=jj_opt.contiguous().data_ptr()))
    var h_ptr = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=h_dense.data_ptr()))
    var b_ptr = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=b_dense.data_ptr()))

    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[assemble_h_dense_kernel, assemble_h_dense_kernel](
        hs_ptr, ii_ptr, jj_ptr, h_ptr, num_edges, n_vars,
        grid_dim=(ceildiv(dim, 16), ceildiv(dim, 16)),
        block_dim=(16, 16),
    )
    ctx_ptr[].enqueue_function[assemble_b_dense_kernel, assemble_b_dense_kernel](
        gs_ptr, ii_ptr, jj_ptr, b_ptr, num_edges, n_vars,
        grid_dim=(ceildiv(n_vars, 16), 1),
        block_dim=(16, POSE_DIM),
    )
    ctx_ptr[].synchronize()
    return Python.tuple(h_dense, b_dense)


def quat_comp_tensor(qi: PythonObject, qj: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    var qix = tensor_col(qi, 0)
    var qiy = tensor_col(qi, 1)
    var qiz = tensor_col(qi, 2)
    var qiw = tensor_col(qi, 3)
    var qjx = tensor_col(qj, 0)
    var qjy = tensor_col(qj, 1)
    var qjz = tensor_col(qj, 2)
    var qjw = tensor_col(qj, 3)
    var x = qiw * qjx + qix * qjw + qiy * qjz - qiz * qjy
    var y = qiw * qjy - qix * qjz + qiy * qjw + qiz * qjx
    var z = qiw * qjz + qix * qjy - qiy * qjx + qiz * qjw
    var w = qiw * qjw - qix * qjx - qiy * qjy - qiz * qjz
    return torch.cat(Python.list(x, y, z, w), dim=1)


def act_so3_tensor(q: PythonObject, X: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    var qx = tensor_col(q, 0)
    var qy = tensor_col(q, 1)
    var qz = tensor_col(q, 2)
    var qw = tensor_col(q, 3)
    var Xx = tensor_col(X, 0)
    var Xy = tensor_col(X, 1)
    var Xz = tensor_col(X, 2)

    var uv0 = 2.0 * (qy * Xz - qz * Xy)
    var uv1 = 2.0 * (qz * Xx - qx * Xz)
    var uv2 = 2.0 * (qx * Xy - qy * Xx)

    var Y0 = Xx + qw * uv0 + (qy * uv2 - qz * uv1)
    var Y1 = Xy + qw * uv1 + (qz * uv0 - qx * uv2)
    var Y2 = Xz + qw * uv2 + (qx * uv1 - qy * uv0)
    return torch.cat(Python.list(Y0, Y1, Y2), dim=1)


def exp_so3_tensor(phi: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    var phi0 = tensor_col(phi, 0)
    var phi1 = tensor_col(phi, 1)
    var phi2 = tensor_col(phi, 2)
    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta_p4 = theta_sq * theta_sq
    var theta = torch.sqrt(theta_sq)
    var small = theta_sq < 1e-6

    var imag_small = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
    var imag_large = torch.sin(0.5 * theta) / theta
    var real_small = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_p4
    var real_large = torch.cos(0.5 * theta)

    var imag = torch.where(small, imag_small, imag_large)
    var real = torch.where(small, real_small, real_large)
    return torch.cat(Python.list(imag * phi0, imag * phi1, imag * phi2, real), dim=1)


def exp_sim3_tensor(xi: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    var tau = xi.narrow(1, 0, 3)
    var phi = xi.narrow(1, 3, 3)
    var sigma = tensor_col(xi, 6)
    var scale = torch.exp(sigma)

    var q = exp_so3_tensor(phi)

    var phi0 = tensor_col(phi, 0)
    var phi1 = tensor_col(phi, 1)
    var phi2 = tensor_col(phi, 2)
    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta = torch.sqrt(theta_sq)
    var sigma_sq = sigma * sigma
    var c = theta_sq + sigma_sq
    var small_sigma = torch.abs(sigma) < 1e-6
    var small_theta = torch.abs(theta) < 1e-6

    var C = torch.where(small_sigma, torch.ones_like(sigma), (scale - 1.0) / sigma)

    var A_small_sigma = torch.where(
        small_theta,
        torch.full_like(theta, 0.5),
        (1.0 - torch.cos(theta)) / theta_sq,
    )
    var B_small_sigma = torch.where(
        small_theta,
        torch.full_like(theta, 1.0 / 6.0),
        (theta - torch.sin(theta)) / (theta_sq * theta),
    )

    var A_large_sigma = torch.where(
        small_theta,
        ((sigma - 1.0) * scale + 1.0) / sigma_sq,
        (scale * torch.sin(theta) * sigma + (1.0 - scale * torch.cos(theta)) * theta) / (theta * c),
    )
    var B_large_sigma = torch.where(
        small_theta,
        (scale * 0.5 * sigma_sq + scale - 1.0 - sigma * scale) / (sigma_sq * sigma),
        (C - (((scale * torch.cos(theta) - 1.0) * sigma + scale * torch.sin(theta) * theta) / c)) / theta_sq,
    )

    var A = torch.where(small_sigma, A_small_sigma, A_large_sigma)
    var B = torch.where(small_sigma, B_small_sigma, B_large_sigma)

    var tau1 = torch.cross(phi, tau, dim=1)
    var tau2 = torch.cross(phi, tau1, dim=1)
    var t = C * tau + A * tau1 + B * tau2
    return Python.tuple(t, q, scale)


def pose_retr_py(
    poses_obj: PythonObject,
    dx_obj: PythonObject,
    num_fix_obj: PythonObject,
) raises -> PythonObject:
    var num_fix = Int(py=num_fix_obj)
    var num_poses = Int(py=poses_obj.shape[0])
    if num_fix >= num_poses:
        return Python.none()
    var poses = poses_obj.contiguous().float()
    var dx = dx_obj.contiguous().float()
    var poses_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=poses.data_ptr()))
    var dx_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=dx.data_ptr()))
    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[pose_retr_kernel, pose_retr_kernel](
        poses_ptr,
        dx_ptr,
        num_fix,
        num_poses,
        grid_dim=ceildiv(max(num_poses - num_fix, 0), 256),
        block_dim=256,
    )
    return Python.none()


def gauss_newton_rays_step_py(args_obj: PythonObject) raises -> PythonObject:
    var twc = args_obj[0].contiguous().float()
    var xs = args_obj[1].contiguous().float()
    var cs = args_obj[2].contiguous().float()
    var ii = args_obj[3].contiguous()
    var jj = args_obj[4].contiguous()
    var idx_ii2jj = args_obj[5].contiguous()
    var valid_match = args_obj[6].contiguous()
    var q_tensor = args_obj[7].contiguous().float()
    var sigma_ray = Float32(py=args_obj[8])
    var sigma_dist = Float32(py=args_obj[9])
    var c_thresh = Float32(py=args_obj[10])
    var q_thresh = Float32(py=args_obj[11])

    var torch = get_torch_module()
    var num_edges = Int(py=ii.shape[0])
    var num_points = Int(py=xs.shape[1])
    var hs = torch.zeros(Python.list(4, num_edges, POSE_DIM, POSE_DIM), device=twc.device, dtype=twc.dtype)
    var gs = torch.zeros(Python.list(2, num_edges, POSE_DIM), device=twc.device, dtype=twc.dtype)

    var twc_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=twc.data_ptr()))
    var xs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=xs.data_ptr()))
    var cs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=cs.data_ptr()))
    var ii_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=ii.data_ptr()))
    var jj_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=jj.data_ptr()))
    var idx_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=idx_ii2jj.data_ptr()))
    var valid_ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(py=valid_match.data_ptr()))
    var q_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=q_tensor.data_ptr()))
    var hs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=hs.data_ptr()))
    var gs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=gs.data_ptr()))

    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[gauss_newton_rays_step_kernel, gauss_newton_rays_step_kernel](
        twc_ptr, xs_ptr, cs_ptr, ii_ptr, jj_ptr, idx_ptr, valid_ptr, q_ptr, hs_ptr, gs_ptr,
        num_points, num_edges, sigma_ray, sigma_dist, c_thresh, q_thresh,
        grid_dim=num_edges,
        block_dim=RAYS_THREADS,
    )
    return Python.tuple(hs, gs)


def gauss_newton_impl(
    args_obj: PythonObject,
    step_name: String,
    step_argc: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    var cuda_be = get_cuda_backend_module()
    var Twc = args_obj[0]
    var ii = args_obj[3]
    var jj = args_obj[4]
    var max_iter = Int(py=args_obj[step_argc])
    var delta_thresh = Float64(py=args_obj[step_argc + 1])

    var num_poses = Int(py=Twc.shape[0])
    var num_fix = 1
    var unique_kf_idx = get_unique_kf_idx(ii, jj)
    var inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj)
    var ii_opt = inds_opt[0]
    var jj_opt = inds_opt[1]
    var n_vars = num_poses - num_fix

    var dx = torch.zeros(
        Python.list(max(n_vars, 0), POSE_DIM),
        device=Twc.device,
        dtype=Twc.dtype,
    )

    for _i in range(max_iter):
        var step_out: PythonObject
        if step_name == "gauss_newton_points_step":
            step_out = cuda_be.gauss_newton_points_step(
                args_obj[0],
                args_obj[1],
                args_obj[2],
                args_obj[3],
                args_obj[4],
                args_obj[5],
                args_obj[6],
                args_obj[7],
                args_obj[8],
                args_obj[9],
                args_obj[10],
            )
        elif step_name == "gauss_newton_rays_step":
            step_out = gauss_newton_rays_step_py(
                Python.tuple(
                    args_obj[0],
                    args_obj[1],
                    args_obj[2],
                    args_obj[3],
                    args_obj[4],
                    args_obj[5],
                    args_obj[6],
                    args_obj[7],
                    args_obj[8],
                    args_obj[9],
                    args_obj[10],
                    args_obj[11],
                )
            )
        else:
            step_out = cuda_be.gauss_newton_calib_step(
                args_obj[0],
                args_obj[1],
                args_obj[2],
                args_obj[3],
                args_obj[4],
                args_obj[5],
                args_obj[6],
                args_obj[7],
                args_obj[8],
                args_obj[9],
                args_obj[10],
                args_obj[11],
                args_obj[12],
                args_obj[13],
                args_obj[14],
                args_obj[15],
            )
        var hs = step_out[0]
        var gs = step_out[1]
        var sys = assemble_dense_system_gpu(hs, gs, ii_opt, jj_opt, n_vars)
        var h_dense = sys[0]
        var b_dense = sys[1]
        dx = solve_dense_system(h_dense, b_dense, n_vars).to(dtype=Twc.dtype)
        _ = pose_retr_py(Twc, dx, PythonObject(num_fix))
        var delta_norm = torch.linalg.norm(dx)
        if Float64(py=delta_norm.item()) < delta_thresh:
            break

    return Python.tuple(dx)
 

def gauss_newton_points_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_points_step", 11)


def gauss_newton_rays_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_rays_step", 12)


def gauss_newton_calib_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_calib_step", 16)


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

    Mirrors the CUDA kernel in matching_kernels.cu:iter_proj_kernel (lines 119-275).
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

        # ── Build normal equations: J^T J + lambda * I, -J^T r ──
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
    """Descriptor-based match refinement kernel (float32 path)."""
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

    Each thread copies its query descriptor into shared memory once, then reuses
    it across all candidate scores. This reduces repeated global reads of D21.
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
    """Drop-in replacement for mast3r_slam_backends.iter_proj."""
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

    # Get raw CUDA pointers
    var rays_ptr: Int = Int(py=rays_img.data_ptr())
    var pts_ptr: Int = Int(py=pts_3d_norm.data_ptr())
    var pinit_ptr: Int = Int(py=p_init.data_ptr())
    var pnew_ptr: Int = Int(py=p_new.data_ptr())
    var conv_ptr: Int = Int(py=converged.data_ptr())

    var rays_uptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=rays_ptr)
    var pts_uptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=pts_ptr)
    var pinit_uptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=pinit_ptr)
    var pnew_uptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=pnew_ptr)
    var conv_uptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=conv_ptr)

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
    """Drop-in replacement for mast3r_slam_backends.refine_matches."""
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

    var p1_ptr: Int = Int(py=p1.data_ptr())
    var p1_new_ptr: Int = Int(py=p1_new.data_ptr())
    var p1_uptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=p1_ptr)
    var p1_new_uptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=p1_new_ptr)

    var ctx_ptr = get_cached_context_ptr()
    var block_size: Int = choose_refine_block(num_pts, fdim)
    var num_blocks_x: Int = ceildiv(num_pts, block_size)

    # Dispatch float16 vs float32 to avoid half→float copy overhead (~7 µs)
    var is_half: Bool = Bool(D11_obj.dtype == torch.float16)

    if is_half:
        var D11: PythonObject = D11_obj.contiguous()
        var D21: PythonObject = D21_obj.contiguous()
        var d11_ptr: Int = Int(py=D11.data_ptr())
        var d21_ptr: Int = Int(py=D21.data_ptr())
        var d11_uptr = UnsafePointer[Float16, MutAnyOrigin](unsafe_from_address=d11_ptr)
        var d21_uptr = UnsafePointer[Float16, MutAnyOrigin](unsafe_from_address=d21_ptr)

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
        var d11_ptr: Int = Int(py=D11.data_ptr())
        var d21_ptr: Int = Int(py=D21.data_ptr())
        var d11_uptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=d11_ptr)
        var d21_uptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=d21_ptr)

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


# ─── Python extension module registration ─────────────────────────────────────

@export
def PyInit_mast3r_slam_mojo_backends() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mast3r_slam_mojo_backends")
        m.def_function[iter_proj_py]("iter_proj")
        m.def_function[refine_matches_py]("refine_matches")
        m.def_function[pose_retr_py]("pose_retr")
        m.def_function[gauss_newton_rays_step_py]("gauss_newton_rays_step")
        m.def_function[gauss_newton_points_impl]("gauss_newton_points_impl")
        m.def_function[gauss_newton_rays_impl]("gauss_newton_rays_impl")
        m.def_function[gauss_newton_calib_impl]("gauss_newton_calib_impl")
        var module = m.finalize()

        # Keep one owning DeviceContext alive for the life of the extension so
        # wrappers can reuse it instead of constructing a fresh context per call.
        var ctx_storage = alloc[DeviceContext](1)
        var cached_ctx = DeviceContext()
        ctx_storage.init_pointee_move(cached_ctx^)
        Python.add_object(module, "_ctx_addr", PythonObject(Int(ctx_storage)))
        return module
    except e:
        abort(String("Failed to create mast3r_slam_mojo_backends module: ", e))
