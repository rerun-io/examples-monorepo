"""Low-level Gauss-Newton GPU kernels for the MASt3R-SLAM Mojo backend.

This file is a **Mojo port of gn_kernels.cu** — the CUDA backend that
implements the second-order pose optimisation described in §3.3 of the
MASt3R-SLAM paper (arXiv 2412.12392).

Background
----------
MASt3R-SLAM represents each keyframe pose as a Sim(3) element — a similarity
transform with 7 degrees of freedom: 3 translation, 3 rotation (stored as a
unit quaternion, 4 components), and 1 log-scale. Pairs of keyframes connected
by an edge in the pose graph contribute residuals measuring how well their
pointmaps (3D predictions from MASt3R) agree after being transformed into a
common frame.

The optimiser is a standard Gauss-Newton (GN) loop:
  1. For every edge, compute residuals and Jacobians per matched point.
  2. Accumulate per-edge 7×7 Hessian blocks (H) and 7-element gradient
     vectors (g) via a GPU reduction.
  3. Assemble these into a dense normal-equation system  H dx = −g  and
     solve it with Cholesky factorisation (done on the CPU in gn.mojo).
  4. Retract the solved Sim(3) increment back onto the poses via the Exp map.

Parallelism strategy
--------------------
Each GPU **thread block** handles one (edge, partial-block) pair. Threads
within the block stride across the matched points on that edge, each
accumulating its own local Hessian/gradient contribution. A block-level
reduction (warp shuffle + shared memory) combines these before writing the
per-edge output. This mirrors the CUDA kernel's structure and avoids atomics.

Mojo-specific notes
-------------------
- `@register_passable("trivial")` structs (Vec3, Quat, Sim3Pose) replace the
  CUDA `float[N]` arrays and the old scalar-parameter pattern. They live in
  registers and are returned by value with zero overhead.
- `stack_allocation[..., address_space=AddressSpace.SHARED]` replaces `__shared__`.
- `@always_inline` replaces CUDA's `__forceinline__ __device__`.
- `warp.sum()` replaces CUDA's `__shfl_down_sync` reduction.
- `barrier()` replaces `__syncthreads()`.
"""

from std.math import sqrt, exp, sin, cos, abs
from std.gpu import global_idx, block_idx, block_dim, thread_idx, barrier, lane_id
from std.gpu.primitives import warp
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import Index
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE

# ── Compile-time constants ───────────────────────────────────────────────────
comptime POSE_DIM = 7       # DOF per Sim(3) pose (tx, ty, tz, qx, qy, qz, log_s)
comptime POSE_STRIDE = 8    # in-memory stride per pose (tx, ty, tz, qx, qy, qz, qw, s)
comptime RAYS_HDIM = 14 * (14 + 1) // 2  # lower-triangle of 14×14 block Hessian
comptime RAYS_THREADS = 128  # block size for the rays step kernel (4 warps)
comptime RAYS_WARPS = RAYS_THREADS // 32

# ── Layout aliases for LayoutTensor ──────────────────────────────────────────
# UNKNOWN_VALUE marks dimensions whose size is determined at runtime.
# These replace UnsafePointer in kernel signatures, giving typed multi-dim indexing.
comptime POSES_LT = Layout.row_major(UNKNOWN_VALUE, POSE_STRIDE)     # [N_kf, 8]
comptime DX_LT = Layout.row_major(UNKNOWN_VALUE, POSE_DIM)           # [N_vars, 7]
comptime XS_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 3)   # [N_kf, N_pts, 3]
comptime CS_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)      # [N_kf, N_pts]
comptime EDGES_LT = Layout.row_major(UNKNOWN_VALUE)                   # [N_edges]
comptime IDX_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)     # [N_edges, N_pts]
comptime HS_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, POSE_DIM, POSE_DIM)  # [4, N_partials, 7, 7]
comptime GS_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, POSE_DIM)             # [2, N_partials, 7]


# ══════════════════════════════════════════════════════════════════════════════
# Value types: Vec3, Quat, Sim3Pose, TangentVec7
# ══════════════════════════════════════════════════════════════════════════════
#
# These replace the old pattern of passing 8-16 individual Float32 scalars.
# `@register_passable("trivial")` means they live entirely in registers and
# are returned by value with zero overhead — identical codegen to scalars.

@fieldwise_init
struct Vec3(TrivialRegisterPassable):
    """3D vector stored as three Float32 scalars."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    @always_inline
    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    @always_inline
    def __mul__(self, s: Float32) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    @always_inline
    def dot(self, other: Vec3) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @always_inline
    def squared_norm(self) -> Float32:
        return self.x * self.x + self.y * self.y + self.z * self.z

    @always_inline
    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@fieldwise_init
struct Quat(TrivialRegisterPassable):
    """Unit quaternion in (x, y, z, w) convention where w is the scalar part.

    Matches the CUDA backend and Eigen's quaternion layout.
    A unit quaternion q = (sin(θ/2)·n̂, cos(θ/2)) encodes rotation by θ about n̂.
    """
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    @always_inline
    def __mul__(self, other: Quat) -> Quat:
        """Hamilton product: composes two rotations. Equivalent to CUDA's `quat_comp()`."""
        return Quat(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    @always_inline
    def conjugate(self) -> Quat:
        """Quaternion conjugate (= inverse for unit quaternions): q⁻¹ = (−x, −y, −z, w)."""
        return Quat(-self.x, -self.y, -self.z, self.w)

    @always_inline
    def rotate(self, v: Vec3) -> Vec3:
        """Rotate a 3D point: y = R(q) · v.

        Uses Rodrigues' rotation via the quaternion double-cross-product formula
        (avoids building a 3×3 matrix):  uv = 2·(q_xyz × v), y = v + w·uv + (q_xyz × uv).
        Equivalent to CUDA's `actSO3()`.
        """
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
    """Sim(3) similarity transform: translation + rotation + scale.

    Memory layout matches the [tx, ty, tz, qx, qy, qz, qw, s] convention
    used by the CUDA backend and the PyTorch pose tensor.
    """
    var t: Vec3
    var q: Quat
    var s: Float32

    @always_inline
    def act(self, pt: Vec3) -> Vec3:
        """Apply Sim(3) to a point: y = s·R(q)·x + t. Equivalent to CUDA's `actSIM3()`."""
        return self.q.rotate(pt) * self.s + self.t

    @always_inline
    def relative(self, other: Sim3Pose) -> Sim3Pose:
        """Compute T_ij = self⁻¹ · other. Equivalent to CUDA's `relSIM3()`.

        All backend residuals are formulated in terms of relative poses between
        keyframes, not world poses directly.
        """
        var si_inv: Float32 = 1.0 / self.s
        var qi_inv = self.q.conjugate()
        var qij = qi_inv * other.q
        var dt = other.t - self.t
        var rot_dt = qi_inv.rotate(dt)
        return Sim3Pose(rot_dt * si_inv, qij, si_inv * other.s)

    @always_inline
    def compose(self, other: Sim3Pose) -> Sim3Pose:
        """Left-multiply: T_new = self · other.

        Used for pose retraction: T_new = Exp(dx) · T_old.
        """
        var rot_t = self.q.rotate(other.t)
        return Sim3Pose(
            Vec3(rot_t.x * self.s + self.t.x, rot_t.y * self.s + self.t.y, rot_t.z * self.s + self.t.z),
            self.q * other.q,
            self.s * other.s,
        )



@fieldwise_init
struct TangentVec7(TrivialRegisterPassable):
    """7D Sim(3) tangent vector: (tau_x, tau_y, tau_z, phi_x, phi_y, phi_z, sigma).

    Used for Lie algebra increments and adjoint operations.
    """
    var v0: Float32
    var v1: Float32
    var v2: Float32
    var v3: Float32
    var v4: Float32
    var v5: Float32
    var v6: Float32


# ══════════════════════════════════════════════════════════════════════════════
# Block-level reduction
# ══════════════════════════════════════════════════════════════════════════════

@always_inline
def block_reduce_sum[origin: Origin[mut=True], //](
    value: Float32,
    tid: Int,
    # UnsafePointer is required here: this is GPU shared memory from stack_allocation,
    # which returns UnsafePointer. Mojo's safe pointer types (Pointer, OwnedPointer)
    # only support single values, not array-like access or AddressSpace parameters.
    warp_partials: UnsafePointer[Scalar[DType.float32], origin, address_space=AddressSpace.SHARED],
) -> Float32:
    """Sum `value` across all threads in the block. Only thread 0 gets the correct total."""
    var reduced = warp.sum(value)
    if Int(lane_id()) == 0:
        warp_partials[tid // 32] = reduced
    barrier()
    var block_sum: Float32 = 0.0
    if tid < RAYS_WARPS:
        block_sum = warp_partials[tid][0]
    if tid < 32:
        block_sum = warp.sum(block_sum)
    barrier()
    return block_sum


# ══════════════════════════════════════════════════════════════════════════════
# Lie-group exponential maps
# ══════════════════════════════════════════════════════════════════════════════

@always_inline
def exp_so3(phi: Vec3) -> Quat:
    """SO(3) exponential map: axis-angle vector φ → unit quaternion.

    For small θ we use a Taylor expansion to avoid the sin(θ/2)/θ singularity.
    Equivalent to CUDA's `expSO3()`.
    """
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


@always_inline
def exp_sim3(xi: TangentVec7) -> Sim3Pose:
    """Sim(3) exponential map: 7D tangent vector ξ → finite similarity transform.

    ξ = (τ, φ, σ) where τ is translational velocity, φ is axis-angle rotation,
    and σ is log-scale. The translation includes the "V matrix" coupling.
    Equivalent to CUDA's `expSIM3()`.
    """
    var tau = Vec3(xi.v0, xi.v1, xi.v2)
    var phi = Vec3(xi.v3, xi.v4, xi.v5)
    var sigma: Float32 = xi.v6

    var scale = exp(sigma)
    var q_rot = exp_so3(phi)
    var theta_sq = phi.squared_norm()
    var theta = sqrt(theta_sq)

    # V-matrix coefficients: t = (C·I + A·[φ]× + B·[φ]×²) · τ
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

    # t = C·τ + A·(φ × τ) + B·(φ × (φ × τ))
    var t_out = tau * C
    var cx = phi.cross(tau)
    t_out = t_out + cx * A
    var c2x = phi.cross(cx)
    t_out = t_out + c2x * B
    return Sim3Pose(t_out, q_rot, scale)


# ══════════════════════════════════════════════════════════════════════════════
# Adjoint and robust weighting
# ══════════════════════════════════════════════════════════════════════════════

@always_inline
def apply_sim3_adj_inv(pose: Sim3Pose, xi: TangentVec7) -> TangentVec7:
    """Apply the inverse adjoint of a Sim(3) element to a 7D tangent vector.

    Maps a relative-frame Jacobian column into the world-frame Jacobian.
    Equivalent to CUDA's `apply_sim3_adj_inv()`.
    """
    var s_inv: Float32 = 1.0 / pose.s
    var x_trans = Vec3(xi.v0, xi.v1, xi.v2)
    var x_rot = Vec3(xi.v3, xi.v4, xi.v5)
    var ra = pose.q.rotate(x_trans)
    var rb = pose.q.rotate(x_rot)
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


@always_inline
def huber_scalar(r: Float32) -> Float32:
    """Huber robust weight. Returns 1.0 inside threshold (1.345), decreasing outside."""
    var r_abs = abs(r)
    if r_abs < 1.345:
        return 1.0
    return 1.345 / r_abs


# ══════════════════════════════════════════════════════════════════════════════
# Main Gauss-Newton ray-cost kernel
# ══════════════════════════════════════════════════════════════════════════════

def gauss_newton_rays_step_kernel(
    twc: LayoutTensor[DType.float32, POSES_LT, MutAnyOrigin],
    xs: LayoutTensor[DType.float32, XS_LT, MutAnyOrigin],
    cs: LayoutTensor[DType.float32, CS_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGES_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGES_LT, MutAnyOrigin],
    idx_ii2jj: LayoutTensor[DType.int64, IDX_LT, MutAnyOrigin],
    valid_match: LayoutTensor[DType.uint8, IDX_LT, MutAnyOrigin],
    q_tensor: LayoutTensor[DType.float32, IDX_LT, MutAnyOrigin],
    hs: LayoutTensor[DType.float32, HS_LT, MutAnyOrigin],
    gs: LayoutTensor[DType.float32, GS_LT, MutAnyOrigin],
    num_points: Int,
    num_edges: Int,
    blocks_per_edge: Int,
    sigma_ray: Float32,
    sigma_dist: Float32,
    c_thresh: Float32,
    q_thresh: Float32,
):
    """One linearisation step of the ray-based Gauss-Newton backend (§3.3).

    Each GPU block handles one (edge, partial) pair. Threads stride across
    points, accumulate local J^T W J and J^T W r, then block-reduce.
    Equivalent to CUDA's `gauss_newton_rays_step()`.
    """
    # ── Thread/block indexing ──
    var edge = Int(block_idx.x) // blocks_per_edge
    var edge_block = Int(block_idx.x) % blocks_per_edge
    var partial_edge = edge * blocks_per_edge + edge_block
    var num_partials = num_edges * blocks_per_edge
    var tid = Int(thread_idx.x)
    var block_threads = Int(block_dim.x)
    if edge >= num_edges:
        return

    # ── Cache both endpoint Sim(3) poses in shared memory ──
    var ix = Int(ii[edge])
    var jx = Int(jj[edge])
    var pose_shared = stack_allocation[
        24, Scalar[DType.float32], address_space=AddressSpace.SHARED,
    ]()
    if tid < 8:
        pose_shared[tid] = twc[ix, tid][0]       # pose_i → shared[0..7]
    if tid < 8:
        pose_shared[tid + 8] = twc[jx, tid][0]   # pose_j → shared[8..15]
    barrier()

    # Read cached poses into register structs
    var pose_i = Sim3Pose(
        Vec3(pose_shared[0][0], pose_shared[1][0], pose_shared[2][0]),
        Quat(pose_shared[3][0], pose_shared[4][0], pose_shared[5][0], pose_shared[6][0]),
        pose_shared[7][0],
    )
    var pose_j = Sim3Pose(
        Vec3(pose_shared[8][0], pose_shared[9][0], pose_shared[10][0]),
        Quat(pose_shared[11][0], pose_shared[12][0], pose_shared[13][0], pose_shared[14][0]),
        pose_shared[15][0],
    )

    # Thread 0 computes the relative pose and publishes to shared memory
    if tid == 0:
        var rel_tmp = pose_i.relative(pose_j)
        pose_shared[16] = rel_tmp.t.x
        pose_shared[17] = rel_tmp.t.y
        pose_shared[18] = rel_tmp.t.z
        pose_shared[19] = rel_tmp.q.x
        pose_shared[20] = rel_tmp.q.y
        pose_shared[21] = rel_tmp.q.z
        pose_shared[22] = rel_tmp.q.w
        pose_shared[23] = rel_tmp.s
    barrier()

    # All threads read the relative pose
    var rel = Sim3Pose(
        Vec3(pose_shared[16][0], pose_shared[17][0], pose_shared[18][0]),
        Quat(pose_shared[19][0], pose_shared[20][0], pose_shared[21][0], pose_shared[22][0]),
        pose_shared[23][0],
    )

    # ── Per-thread accumulators ──
    var hij = InlineArray[Float32, RAYS_HDIM](fill=0.0)
    var vi = InlineArray[Float32, POSE_DIM](fill=0.0)
    var vj = InlineArray[Float32, POSE_DIM](fill=0.0)
    var jxv = InlineArray[Float32, 14](fill=0.0)
    var sigma_ray_inv = 1.0 / sigma_ray
    var sigma_dist_inv = 1.0 / sigma_dist

    # ── Main point loop ──
    for k in range(edge_block * block_threads + tid, num_points, block_threads * blocks_per_edge):
        var valid_match_ind = valid_match[edge, k] != 0
        var ind_xi = Int(idx_ii2jj[edge, k])
        if not valid_match_ind:
            ind_xi = 0

        # LayoutTensor multi-dim indexing — [0] extracts scalar from SIMD[dtype, 1]
        var ci = cs[ix, ind_xi][0]
        var cj = cs[jx, k][0]
        var qv = q_tensor[edge, k][0]

        var pt_i = Vec3(xs[ix, ind_xi, 0][0], xs[ix, ind_xi, 1][0], xs[ix, ind_xi, 2][0])
        var pt_j = Vec3(xs[jx, k, 0][0], xs[jx, k, 1][0], xs[jx, k, 2][0])

        # Unit ray for point in frame i: r̂_i = x_i / ‖x_i‖
        var norm1_i = sqrt(pt_i.squared_norm())
        var ri = pt_i * (1.0 / norm1_i)

        # Transform point j into frame i, then normalise
        var xj_ci = rel.act(pt_j)
        var norm2_j = xj_ci.squared_norm()
        var norm1_j = sqrt(norm2_j)
        var norm1_j_inv = 1.0 / norm1_j
        var rj = xj_ci * norm1_j_inv

        # Residuals: ray direction (3D) + distance (1D)
        var err = rj - ri
        var err3 = norm1_j - norm1_i

        # Confidence gating
        var valid = valid_match_ind and (qv > q_thresh) and (ci > c_thresh) and (cj > c_thresh)
        var sqrt_w_ray: Float32 = 0.0
        var sqrt_w_dist: Float32 = 0.0
        if valid:
            var sqrt_conf = sqrt(qv)
            sqrt_w_ray = sigma_ray_inv * sqrt_conf
            sqrt_w_dist = sigma_dist_inv * sqrt_conf

        # IRLS weights (Huber robust)
        var w0 = huber_scalar(sqrt_w_ray * err.x) * sqrt_w_ray * sqrt_w_ray
        var w1 = huber_scalar(sqrt_w_ray * err.y) * sqrt_w_ray * sqrt_w_ray
        var w2 = huber_scalar(sqrt_w_ray * err.z) * sqrt_w_ray * sqrt_w_ray
        var w3 = huber_scalar(sqrt_w_dist * err3) * sqrt_w_dist * sqrt_w_dist

        # Jacobian of ray normalisation: ∂r̂/∂p = (I − r̂r̂ᵀ) / ‖p‖
        var norm3_j_inv = norm1_j_inv / norm2_j
        var drx_dpx = norm1_j_inv - xj_ci.x * xj_ci.x * norm3_j_inv
        var dry_dpy = norm1_j_inv - xj_ci.y * xj_ci.y * norm3_j_inv
        var drz_dpz = norm1_j_inv - xj_ci.z * xj_ci.z * norm3_j_inv
        var drx_dpy = -xj_ci.x * xj_ci.y * norm3_j_inv
        var drx_dpz = -xj_ci.x * xj_ci.z * norm3_j_inv
        var dry_dpz = -xj_ci.y * xj_ci.z * norm3_j_inv

        @parameter
        def accumulate_row(
            residual: Float32,
            w: Float32,
            ji: TangentVec7,
        ):
            """Map relative-pose Jacobian to world-pose blocks and accumulate H, g."""
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

        # Accumulate all four residual rows
        accumulate_row(err.x, w0, TangentVec7(drx_dpx, drx_dpy, drx_dpz, 0.0, rj.z, -rj.y, 0.0))
        accumulate_row(err.y, w1, TangentVec7(drx_dpy, dry_dpy, dry_dpz, -rj.z, 0.0, rj.x, 0.0))
        accumulate_row(err.z, w2, TangentVec7(drx_dpz, dry_dpz, drz_dpz, rj.y, -rj.x, 0.0, 0.0))
        accumulate_row(err3, w3, TangentVec7(rj.x, rj.y, rj.z, 0.0, 0.0, 0.0, norm1_j))

    # ── Block-wide reduction and output ──
    var warp_partials = stack_allocation[
        RAYS_WARPS, Scalar[DType.float32], address_space=AddressSpace.SHARED,
    ]()

    # ── Scatter reduced H/G blocks via LayoutTensor ──
    # gs[block_idx, partial_edge, dim] and hs[block_idx, partial_edge, row, col]
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
                var val = h_sum
                if n < 7 and m < 7:
                    hs[0, partial_edge, n, m] = val  # H_ii
                    hs[0, partial_edge, m, n] = val
                elif n >= 7 and m < 7:
                    hs[1, partial_edge, m, n - 7] = val  # H_ij
                    hs[2, partial_edge, n - 7, m] = val  # H_ji
                else:
                    hs[3, partial_edge, n - 7, m - 7] = val  # H_jj
                    hs[3, partial_edge, m - 7, n - 7] = val
            l += 1


# ══════════════════════════════════════════════════════════════════════════════
# Pose retraction kernel
# ══════════════════════════════════════════════════════════════════════════════

def pose_retr_kernel(
    poses: LayoutTensor[DType.float32, POSES_LT, MutAnyOrigin],
    dx: LayoutTensor[DType.float32, DX_LT, MutAnyOrigin],
    num_fix: Int,
    num_poses: Int,
):
    """Apply Sim(3) tangent update to every unfixed pose: T_new = Exp(dx) · T_old.

    One thread per unfixed pose. Uses LayoutTensor for structured [kf, dim]
    access instead of raw pointer arithmetic.
    Equivalent to CUDA's `pose_retr_kernel()`.
    """
    var k: Int = Int(global_idx.x) + num_fix
    if k >= num_poses:
        return
    # Load the 7D tangent increment dx[k-fix, :] via LayoutTensor indexing
    var dx_row: Int = k - num_fix
    var xi_tangent = TangentVec7(
        dx[dx_row, 0][0], dx[dx_row, 1][0], dx[dx_row, 2][0],
        dx[dx_row, 3][0], dx[dx_row, 4][0], dx[dx_row, 5][0], dx[dx_row, 6][0],
    )
    # Load the current pose, compose with Exp(dx), write back
    var old_pose = Sim3Pose(
        Vec3(poses[k, 0][0], poses[k, 1][0], poses[k, 2][0]),
        Quat(poses[k, 3][0], poses[k, 4][0], poses[k, 5][0], poses[k, 6][0]),
        poses[k, 7][0],
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
