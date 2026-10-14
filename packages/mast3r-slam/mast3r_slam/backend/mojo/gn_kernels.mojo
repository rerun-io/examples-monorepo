"""Gauss-Newton GPU kernels of the MASt3R-SLAM backend: a Mojo port of gn_kernels.cu (§3.3 of arXiv 2412.12392).

Each keyframe pose is a Sim(3) element stored as [tx, ty, tz, qx, qy, qz, qw, s]; its tangent space has 7 DOF. Per edge,
`gauss_newton_rays_step_kernel` linearises the ray + distance residuals of all matched points and writes the 7x7 Hessian
blocks (H_ii, H_ij, H_ji, H_jj) and gradients (g_i, g_j). The caller assembles and solves the dense normal equations
(mast3r_slam/ops/optim.py) and `pose_retr_kernel` applies the increment: T_new = Exp(dx) * T_old.

One thread block handles one (edge, partial) pair: threads stride over the edge's points, accumulate locally, then reduce
across the block (warp shuffles + shared memory), like the CUDA kernel. No atomics.
"""

from std.math import abs, cos, exp, sin, sqrt
from max.gpu import WARP_SIZE, barrier, block_dim, block_idx, global_idx, lane_id, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from layout import Idx, TileTensor, row_major, stack_allocation

comptime POSE_DIM = 7  # tangent DOF per Sim(3) pose (tx, ty, tz, phi_x, phi_y, phi_z, log_s)
comptime POSE_STRIDE = 8  # stored floats per pose (tx, ty, tz, qx, qy, qz, qw, s)
comptime RAYS_HDIM = 14 * (14 + 1) // 2  # lower triangle of the 14x14 edge Hessian
comptime RAYS_THREADS = 128
comptime RAYS_WARPS = RAYS_THREADS // WARP_SIZE

# Tensor layouts; a plain 0 marks a dimension known only at run time.
comptime PosesLayout = type_of(row_major((0, Idx[POSE_STRIDE])))  # [N_kf, 8]
comptime DxLayout = type_of(row_major((0, Idx[POSE_DIM])))  # [N_vars, 7]
comptime PointsLayout = type_of(row_major((0, 0, Idx[3])))  # [N_kf, N_pts, 3]
comptime PerPointLayout = type_of(row_major((0, 0)))  # [N_kf, N_pts] confidences, [N_edges, N_pts] matches
comptime EdgesLayout = type_of(row_major((0,)))  # [N_edges]
comptime HsLayout = type_of(row_major((Idx[4], 0, Idx[POSE_DIM], Idx[POSE_DIM])))  # [4, N_partials, 7, 7]
comptime GsLayout = type_of(row_major((Idx[2], 0, Idx[POSE_DIM])))  # [2, N_partials, 7]
comptime SharedPosesLayout = type_of(row_major[3 * POSE_STRIDE]())  # pose_i, pose_j, relative pose
comptime WarpPartialsLayout = type_of(row_major[RAYS_WARPS]())


# ── Value types (live in registers) ──────────────────────────────────────────


@fieldwise_init
struct Vec3(TrivialRegisterPassable):
    var x: Float32
    var y: Float32
    var z: Float32

    @inline(.always)
    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    @inline(.always)
    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    @inline(.always)
    def __mul__(self, s: Float32) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    @inline(.always)
    def dot(self, other: Vec3) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @inline(.always)
    def squared_norm(self) -> Float32:
        return self.x * self.x + self.y * self.y + self.z * self.z

    @inline(.always)
    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@fieldwise_init
struct Quat(TrivialRegisterPassable):
    """Unit quaternion (x, y, z, w), w the scalar part (CUDA backend and Eigen layout)."""

    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    @inline(.always)
    def __mul__(self, other: Quat) -> Quat:
        """Hamilton product (CUDA `quat_comp`)."""
        return Quat(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    @inline(.always)
    def conjugate(self) -> Quat:
        return Quat(-self.x, -self.y, -self.z, self.w)

    @inline(.always)
    def rotate(self, v: Vec3) -> Vec3:
        """R(q) v by the double cross product: uv = 2 (q_xyz x v), v + w uv + q_xyz x uv (CUDA `actSO3`)."""
        var uv = Vec3(
            2.0 * (self.y * v.z - self.z * v.y),
            2.0 * (self.z * v.x - self.x * v.z),
            2.0 * (self.x * v.y - self.y * v.x),
        )
        return Vec3(
            v.x + self.w * uv.x + (self.y * uv.z - self.z * uv.y),
            v.y + self.w * uv.y + (self.z * uv.x - self.x * uv.z),
            v.z + self.w * uv.z + (self.x * uv.y - self.y * uv.x),
        )


@fieldwise_init
struct Sim3Pose(TrivialRegisterPassable):
    var t: Vec3
    var q: Quat
    var s: Float32

    @inline(.always)
    def act(self, pt: Vec3) -> Vec3:
        """s R(q) x + t (CUDA `actSIM3`)."""
        return self.q.rotate(pt) * self.s + self.t

    @inline(.always)
    def relative(self, other: Sim3Pose) -> Sim3Pose:
        """self^-1 * other (CUDA `relSIM3`); all residuals use relative poses."""
        var si_inv: Float32 = 1.0 / self.s
        var qi_inv = self.q.conjugate()
        var qij = qi_inv * other.q
        var rot_dt = qi_inv.rotate(other.t - self.t)
        return Sim3Pose(rot_dt * si_inv, qij, si_inv * other.s)

    @inline(.always)
    def compose(self, other: Sim3Pose) -> Sim3Pose:
        """self * other; the retraction is Exp(dx) * T_old."""
        var rot_t = self.q.rotate(other.t)
        return Sim3Pose(
            Vec3(rot_t.x * self.s + self.t.x, rot_t.y * self.s + self.t.y, rot_t.z * self.s + self.t.z),
            self.q * other.q,
            self.s * other.s,
        )


@fieldwise_init
struct TangentVec7(TrivialRegisterPassable):
    """Sim(3) tangent vector (tau_x, tau_y, tau_z, phi_x, phi_y, phi_z, sigma)."""

    var v0: Float32
    var v1: Float32
    var v2: Float32
    var v3: Float32
    var v4: Float32
    var v5: Float32
    var v6: Float32


# ── Block reduction and shared-pose access ───────────────────────────────────


@inline(.always)
def block_reduce_sum(
    value: Float32,
    tid: Int,
    warp_partials: TileTensor[.float32, WarpPartialsLayout, MutAnyOrigin, address_space=AddressSpace.SHARED],
) -> Float32:
    """Sum `value` over the block; only thread 0 holds the total."""
    var reduced = warp.sum(value)
    if lane_id() == 0:
        warp_partials[tid // WARP_SIZE] = reduced
    barrier()
    var block_sum: Float32 = 0.0
    if tid < RAYS_WARPS:
        block_sum = warp_partials[tid]
    if tid < WARP_SIZE:
        block_sum = warp.sum(block_sum)
    barrier()
    return block_sum


@inline(.always)
def shared_pose(
    poses: TileTensor[.float32, SharedPosesLayout, MutAnyOrigin, address_space=AddressSpace.SHARED], offset: Int
) -> Sim3Pose:
    """The pose stored at poses[offset : offset + 8]."""
    return Sim3Pose(
        Vec3(poses[offset], poses[offset + 1], poses[offset + 2]),
        Quat(poses[offset + 3], poses[offset + 4], poses[offset + 5], poses[offset + 6]),
        poses[offset + 7],
    )


# ── Lie-group exponential maps ───────────────────────────────────────────────


@inline(.always)
def exp_so3(phi: Vec3) -> Quat:
    """Axis-angle -> unit quaternion; Taylor series near theta = 0 (CUDA `expSO3`)."""
    var theta_sq = phi.squared_norm()
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
    return Quat(imag * phi.x, imag * phi.y, imag * phi.z, real)


@inline(.always)
def exp_sim3(xi: TangentVec7) -> Sim3Pose:
    """Tangent vector (tau, phi, sigma) -> Sim(3); the translation goes through the V matrix (CUDA `expSIM3`)."""
    var tau = Vec3(xi.v0, xi.v1, xi.v2)
    var phi = Vec3(xi.v3, xi.v4, xi.v5)
    var sigma: Float32 = xi.v6

    var scale = exp(sigma)
    var q_rot = exp_so3(phi)
    var theta_sq = phi.squared_norm()
    var theta = sqrt(theta_sq)

    # t = (C I + A [phi]x + B [phi]x^2) tau
    var A: Float32
    var B: Float32
    var C: Float32
    if abs(sigma) < 1e-6:
        C = 1.0
        if abs(theta) < 1e-6:
            A = 0.5
            B = 1.0 / 6.0
        else:
            A = (1.0 - cos(theta)) / theta_sq
            B = (theta - sin(theta)) / (theta_sq * theta)
    else:
        C = (scale - 1.0) / sigma
        if abs(theta) < 1e-6:
            var sigma_sq = sigma * sigma
            A = ((sigma - 1.0) * scale + 1.0) / sigma_sq
            B = (scale * 0.5 * sigma_sq + scale - 1.0 - sigma * scale) / (sigma_sq * sigma)
        else:
            var a = scale * sin(theta)
            var b = scale * cos(theta)
            var c = theta_sq + sigma * sigma
            A = (a * sigma + (1.0 - b) * theta) / (theta * c)
            B = (C - (((b - 1.0) * sigma + a * theta) / c)) / theta_sq

    var t_out = tau * C
    var cx = phi.cross(tau)
    t_out = t_out + cx * A
    var c2x = phi.cross(cx)
    t_out = t_out + c2x * B
    return Sim3Pose(t_out, q_rot, scale)


# ── Adjoint and robust weighting ─────────────────────────────────────────────


@inline(.always)
def apply_sim3_adj_inv(pose: Sim3Pose, xi: TangentVec7) -> TangentVec7:
    """Inverse adjoint of `pose` applied to a tangent vector: relative-frame -> world-frame Jacobian (CUDA)."""
    var s_inv: Float32 = 1.0 / pose.s
    var ra = pose.q.rotate(Vec3(xi.v0, xi.v1, xi.v2))
    var rb = pose.q.rotate(Vec3(xi.v3, xi.v4, xi.v5))
    var t = pose.t
    return TangentVec7(
        s_inv * ra.x,
        s_inv * ra.y,
        s_inv * ra.z,
        rb.x + s_inv * (t.y * ra.z - t.z * ra.y),
        rb.y + s_inv * (t.z * ra.x - t.x * ra.z),
        rb.z + s_inv * (t.x * ra.y - t.y * ra.x),
        xi.v6 + s_inv * t.dot(ra),
    )


@inline(.always)
def huber_scalar(r: Float32) -> Float32:
    """Huber weight: 1 inside |r| < 1.345, 1.345 / |r| outside."""
    var r_abs = abs(r)
    if r_abs < 1.345:
        return 1.0
    return 1.345 / r_abs


# ── Kernels ──────────────────────────────────────────────────────────────────


def gauss_newton_rays_step_kernel(
    twc: TileTensor[.float32, PosesLayout, ImmutAnyOrigin],
    xs: TileTensor[.float32, PointsLayout, ImmutAnyOrigin],
    cs: TileTensor[.float32, PerPointLayout, ImmutAnyOrigin],
    ii: TileTensor[.int64, EdgesLayout, ImmutAnyOrigin],
    jj: TileTensor[.int64, EdgesLayout, ImmutAnyOrigin],
    idx_ii2jj: TileTensor[.int64, PerPointLayout, ImmutAnyOrigin],
    valid_match: TileTensor[.uint8, PerPointLayout, ImmutAnyOrigin],
    q_tensor: TileTensor[.float32, PerPointLayout, ImmutAnyOrigin],
    hs: TileTensor[.float32, HsLayout, MutAnyOrigin],
    gs: TileTensor[.float32, GsLayout, MutAnyOrigin],
    blocks_per_edge: Int32,
    sigma_ray: Float32,
    sigma_dist: Float32,
    c_thresh: Float32,
    q_thresh: Float32,
):
    """One linearisation of the ray cost (CUDA `gauss_newton_rays_step`): per block, J^T W J and J^T W r of one edge part."""
    var edge, edge_block = divmod(block_idx.x, Int(blocks_per_edge))
    var partial_edge = block_idx.x
    var tid = thread_idx.x
    if edge >= Int(ii.dim[0]()):
        return
    var num_points = Int(xs.dim[1]())

    # Both endpoint poses, then their relative pose (computed once by thread 0), go through shared memory.
    var ix = Int(ii[edge])
    var jx = Int(jj[edge])
    var pose_shared = stack_allocation[.float32, address_space=AddressSpace.SHARED](row_major[3 * POSE_STRIDE]())
    if tid < POSE_STRIDE:
        pose_shared[tid] = twc[ix, tid]
        pose_shared[tid + POSE_STRIDE] = twc[jx, tid]
    barrier()
    var pose_i = shared_pose(pose_shared, 0)
    var pose_j = shared_pose(pose_shared, POSE_STRIDE)
    comptime REL = 2 * POSE_STRIDE
    if tid == 0:
        var rel_tmp = pose_i.relative(pose_j)
        pose_shared[REL] = rel_tmp.t.x
        pose_shared[REL + 1] = rel_tmp.t.y
        pose_shared[REL + 2] = rel_tmp.t.z
        pose_shared[REL + 3] = rel_tmp.q.x
        pose_shared[REL + 4] = rel_tmp.q.y
        pose_shared[REL + 5] = rel_tmp.q.z
        pose_shared[REL + 6] = rel_tmp.q.w
        pose_shared[REL + 7] = rel_tmp.s
    barrier()
    var rel = shared_pose(pose_shared, REL)

    # Per-thread accumulators.
    var hij = Array[Float32, RAYS_HDIM](fill=0.0)
    var vi = Array[Float32, POSE_DIM](fill=0.0)
    var vj = Array[Float32, POSE_DIM](fill=0.0)
    var jxv = Array[Float32, 14](fill=0.0)
    var sigma_ray_inv = 1.0 / sigma_ray
    var sigma_dist_inv = 1.0 / sigma_dist

    def accumulate_row(residual: Float32, w: Float32, ji: TangentVec7) {mut hij, mut vi, mut vj, mut jxv, imm pose_i}:
        """Map one relative-pose Jacobian row to the two world poses and accumulate H and g."""
        var jadj = apply_sim3_adj_inv(pose_i, ji)
        jxv[0] = -jadj.v0
        jxv[1] = -jadj.v1
        jxv[2] = -jadj.v2
        jxv[3] = -jadj.v3
        jxv[4] = -jadj.v4
        jxv[5] = -jadj.v5
        jxv[6] = -jadj.v6
        jxv[7] = jadj.v0
        jxv[8] = jadj.v1
        jxv[9] = jadj.v2
        jxv[10] = jadj.v3
        jxv[11] = jadj.v4
        jxv[12] = jadj.v5
        jxv[13] = jadj.v6
        var l = 0
        for n in range(14):
            for m in range(n + 1):
                hij[l] += w * jxv[n] * jxv[m]
                l += 1
        for n in range(POSE_DIM):
            vi[n] += w * residual * jxv[n]
            vj[n] += w * residual * jxv[n + 7]

    var block_threads = block_dim.x
    for k in range(edge_block * block_threads + tid, num_points, block_threads * Int(blocks_per_edge)):
        var valid_match_ind = valid_match[edge, k] != 0
        var ind_xi = Int(idx_ii2jj[edge, k])
        if not valid_match_ind:
            ind_xi = 0

        var ci = cs[ix, ind_xi]
        var cj = cs[jx, k]
        var qv = q_tensor[edge, k]
        var pt_i = Vec3(xs[ix, ind_xi, 0], xs[ix, ind_xi, 1], xs[ix, ind_xi, 2])
        var pt_j = Vec3(xs[jx, k, 0], xs[jx, k, 1], xs[jx, k, 2])

        # Unit ray of the point in frame i, and of point j moved into frame i.
        var norm1_i = sqrt(pt_i.squared_norm())
        var ri = pt_i * (1.0 / norm1_i)
        var xj_ci = rel.act(pt_j)
        var norm2_j = xj_ci.squared_norm()
        var norm1_j = sqrt(norm2_j)
        var norm1_j_inv = 1.0 / norm1_j
        var rj = xj_ci * norm1_j_inv

        # Residuals: ray direction (3) and distance (1).
        var err = rj - ri
        var err3 = norm1_j - norm1_i

        var valid = valid_match_ind and (qv > q_thresh) and (ci > c_thresh) and (cj > c_thresh)
        var sqrt_w_ray: Float32 = 0.0
        var sqrt_w_dist: Float32 = 0.0
        if valid:
            var sqrt_conf = sqrt(qv)
            sqrt_w_ray = sigma_ray_inv * sqrt_conf
            sqrt_w_dist = sigma_dist_inv * sqrt_conf

        # IRLS (Huber) weights.
        var w0 = huber_scalar(sqrt_w_ray * err.x) * sqrt_w_ray * sqrt_w_ray
        var w1 = huber_scalar(sqrt_w_ray * err.y) * sqrt_w_ray * sqrt_w_ray
        var w2 = huber_scalar(sqrt_w_ray * err.z) * sqrt_w_ray * sqrt_w_ray
        var w3 = huber_scalar(sqrt_w_dist * err3) * sqrt_w_dist * sqrt_w_dist

        # Jacobian of the normalisation: d(p/|p|)/dp = (I - r r^T) / |p|.
        var norm3_j_inv = norm1_j_inv / norm2_j
        var drx_dpx = norm1_j_inv - xj_ci.x * xj_ci.x * norm3_j_inv
        var dry_dpy = norm1_j_inv - xj_ci.y * xj_ci.y * norm3_j_inv
        var drz_dpz = norm1_j_inv - xj_ci.z * xj_ci.z * norm3_j_inv
        var drx_dpy = -xj_ci.x * xj_ci.y * norm3_j_inv
        var drx_dpz = -xj_ci.x * xj_ci.z * norm3_j_inv
        var dry_dpz = -xj_ci.y * xj_ci.z * norm3_j_inv

        accumulate_row(err.x, w0, TangentVec7(drx_dpx, drx_dpy, drx_dpz, 0.0, rj.z, -rj.y, 0.0))
        accumulate_row(err.y, w1, TangentVec7(drx_dpy, dry_dpy, dry_dpz, -rj.z, 0.0, rj.x, 0.0))
        accumulate_row(err.z, w2, TangentVec7(drx_dpz, dry_dpz, drz_dpz, rj.y, -rj.x, 0.0, 0.0))
        accumulate_row(err3, w3, TangentVec7(rj.x, rj.y, rj.z, 0.0, 0.0, 0.0, norm1_j))

    # Block-reduce every accumulator; thread 0 writes the edge part's blocks.
    var warp_partials = stack_allocation[.float32, address_space=AddressSpace.SHARED](row_major[RAYS_WARPS]())
    for n in range(POSE_DIM):
        var vi_sum = block_reduce_sum(vi[n], tid, warp_partials)
        if tid == 0:
            gs[0, partial_edge, n] = vi_sum
        var vj_sum = block_reduce_sum(vj[n], tid, warp_partials)
        if tid == 0:
            gs[1, partial_edge, n] = vj_sum

    var l = 0
    for n in range(14):
        for m in range(n + 1):
            var h_sum = block_reduce_sum(hij[l], tid, warp_partials)
            if tid == 0:
                if n < 7 and m < 7:
                    hs[0, partial_edge, n, m] = h_sum  # H_ii
                    hs[0, partial_edge, m, n] = h_sum
                elif n >= 7 and m < 7:
                    hs[1, partial_edge, m, n - 7] = h_sum  # H_ij
                    hs[2, partial_edge, n - 7, m] = h_sum  # H_ji
                else:
                    hs[3, partial_edge, n - 7, m - 7] = h_sum  # H_jj
                    hs[3, partial_edge, m - 7, n - 7] = h_sum
            l += 1


def pose_retr_kernel(
    poses: TileTensor[.float32, PosesLayout, MutAnyOrigin],
    dx: TileTensor[.float32, DxLayout, ImmutAnyOrigin],
    num_fix: Int32,
):
    """T_new = Exp(dx) * T_old for every pose after the first `num_fix`, one thread per pose (CUDA `pose_retr_kernel`)."""
    var dx_row = global_idx.x
    var k = dx_row + Int(num_fix)
    if k >= Int(poses.dim[0]()):
        return
    var xi_tangent = TangentVec7(
        dx[dx_row, 0], dx[dx_row, 1], dx[dx_row, 2], dx[dx_row, 3], dx[dx_row, 4], dx[dx_row, 5], dx[dx_row, 6]
    )
    var old_pose = Sim3Pose(
        Vec3(poses[k, 0], poses[k, 1], poses[k, 2]),
        Quat(poses[k, 3], poses[k, 4], poses[k, 5], poses[k, 6]),
        poses[k, 7],
    )
    var new_pose = exp_sim3(xi_tangent).compose(old_pose)
    poses[k, 0] = new_pose.t.x
    poses[k, 1] = new_pose.t.y
    poses[k, 2] = new_pose.t.z
    poses[k, 3] = new_pose.q.x
    poses[k, 4] = new_pose.q.y
    poses[k, 5] = new_pose.q.z
    poses[k, 6] = new_pose.q.w
    poses[k, 7] = new_pose.s
