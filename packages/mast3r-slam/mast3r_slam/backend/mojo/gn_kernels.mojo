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
- `TrivialRegisterPassable` structs (Vec3, Quat, Sim3Pose) replace the
  CUDA `float[N]` arrays and the old scalar-parameter pattern. They live in
  registers and are returned by value with zero overhead.
- Kernels take `TileTensor` views and index them by coordinate; read-only
  inputs are `ImmTileTensor`, outputs `MutTileTensor`.
- Kernel scalar arguments are `Int64`: 1.2.0 rejects `Int` as a kernel argument.
- `layout.stack_allocation[..., address_space=.SHARED]` replaces `__shared__`.
- `@inline(.always)` replaces CUDA's `__forceinline__ __device__`.
- `warp.sum()` replaces CUDA's `__shfl_down_sync` reduction.
- `barrier()` replaces `__syncthreads()`.
"""

from std.math import abs, sqrt
from std.utils.static_tuple import StaticTuple
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, block_dim, block_idx, global_idx, lane_id, thread_idx
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import Idx, ImmTileTensor, MutTileTensor, TensorLayout, row_major, stack_allocation

# ── Compile-time constants ───────────────────────────────────────────────────
comptime POSE_DIM = 7       # DOF per Sim(3) pose (tx, ty, tz, qx, qy, qz, log_s)
comptime POSE_STRIDE = 8    # in-memory stride per pose (tx, ty, tz, qx, qy, qz, qw, s)
comptime RAYS_HDIM = 14 * (14 + 1) // 2  # lower-triangle of 14×14 block Hessian
comptime RAYS_THREADS = 128  # block size for the rays step kernel
comptime RAYS_WARPS = RAYS_THREADS // WARP_SIZE

# ── Tensor layouts ───────────────────────────────────────────────────────────
# A `0` extent is a runtime dimension (set when the view is built in gn.mojo);
# `Idx[N]` keeps a compile-time extent static.
comptime POSES_LT = type_of(row_major((0, Idx[POSE_STRIDE])))                   # [N_kf, 8]
comptime DX_LT = type_of(row_major((0, Idx[POSE_DIM])))                         # [N_vars, 7]
comptime XS_LT = type_of(row_major((0, 0, Idx[3])))                             # [N_kf, N_pts, 3]
comptime CS_LT = type_of(row_major((0, 0)))                                     # [N_kf, N_pts]
comptime EDGES_LT = type_of(row_major((0,)))                                    # [N_edges]
comptime IDX_LT = type_of(row_major((0, 0)))                                    # [N_edges, N_pts]
comptime HS_LT = type_of(row_major((Idx[4], 0, Idx[POSE_DIM], Idx[POSE_DIM])))  # [4, N_partials, 7, 7]
comptime GS_LT = type_of(row_major((Idx[2], 0, Idx[POSE_DIM])))                 # [2, N_partials, 7]


# ══════════════════════════════════════════════════════════════════════════════
# Value types: Vec3, Quat, Sim3Pose, TangentVec7
# ══════════════════════════════════════════════════════════════════════════════
#
# These replace the old pattern of passing 8-16 individual Float32 scalars.
# `TrivialRegisterPassable` means they live entirely in registers and are
# returned by value with zero overhead — identical codegen to scalars.

@fieldwise_init
struct Vec3(TrivialRegisterPassable):
    """3D vector stored as three Float32 scalars."""
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
    """Unit quaternion in (x, y, z, w) convention where w is the scalar part.

    Matches the CUDA backend and Eigen's quaternion layout.
    A unit quaternion q = (sin(θ/2)·n̂, cos(θ/2)) encodes rotation by θ about n̂.
    """
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    @inline(.always)
    def __mul__(self, other: Quat) -> Quat:
        """Hamilton product: composes two rotations. Equivalent to CUDA's `quat_comp()`."""
        return Quat(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    @inline(.always)
    def conjugate(self) -> Quat:
        """Quaternion conjugate (= inverse for unit quaternions): q⁻¹ = (−x, −y, −z, w)."""
        return Quat(-self.x, -self.y, -self.z, self.w)

    @inline(.always)
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

    @inline(.always)
    def act(self, pt: Vec3) -> Vec3:
        """Apply Sim(3) to a point: y = s·R(q)·x + t. Equivalent to CUDA's `actSIM3()`."""
        return self.q.rotate(pt) * self.s + self.t

    @inline(.always)
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

@inline(.always)
def block_reduce_sum[LT: TensorLayout](
    value: Float32,
    tid: Int,
    warp_partials: MutTileTensor[.float32, LT, _, address_space=.SHARED],
) -> Float32:
    """Sum `value` across all threads in the block. Only thread 0 gets the correct total."""
    comptime assert warp_partials.flat_rank == 1
    var reduced = warp.sum(value)
    if lane_id() == 0:
        warp_partials[tid // WARP_SIZE] = reduced
    barrier()
    var block_sum: Float32 = 0.0
    if tid < RAYS_WARPS:
        block_sum = warp_partials[tid][0]
    if tid < WARP_SIZE:
        block_sum = warp.sum(block_sum)
    barrier()
    return block_sum


# ══════════════════════════════════════════════════════════════════════════════
# Adjoint and robust weighting
# ══════════════════════════════════════════════════════════════════════════════

@inline(.always)
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


@inline(.always)
def huber_scalar(r: Float32) -> Float32:
    """Huber robust weight. Returns 1.0 inside threshold (1.345), decreasing outside."""
    var r_abs = abs(r)
    if r_abs < 1.345:
        return 1.0
    return 1.345 / r_abs


# ══════════════════════════════════════════════════════════════════════════════
# Main Gauss-Newton ray-cost kernel
# ══════════════════════════════════════════════════════════════════════════════

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](RAYS_THREADS))
def gauss_newton_rays_step_kernel(
    twc: ImmTileTensor[.float32, POSES_LT, ImmUntrackedOrigin],
    xs: ImmTileTensor[.float32, XS_LT, ImmUntrackedOrigin],
    cs: ImmTileTensor[.float32, CS_LT, ImmUntrackedOrigin],
    ii: ImmTileTensor[.int64, EDGES_LT, ImmUntrackedOrigin],
    jj: ImmTileTensor[.int64, EDGES_LT, ImmUntrackedOrigin],
    idx_ii2jj: ImmTileTensor[.int64, IDX_LT, ImmUntrackedOrigin],
    valid_match: ImmTileTensor[.uint8, IDX_LT, ImmUntrackedOrigin],
    q_tensor: ImmTileTensor[.float32, IDX_LT, ImmUntrackedOrigin],
    hs: MutTileTensor[.float32, HS_LT, MutUntrackedOrigin],
    gs: MutTileTensor[.float32, GS_LT, MutUntrackedOrigin],
    num_points_arg: Int64,
    num_edges_arg: Int64,
    blocks_per_edge_arg: Int64,
    sigma_ray: Float32,
    sigma_dist: Float32,
    c_thresh: Float32,
    q_thresh: Float32,
):
    """One linearisation step of the ray-based Gauss-Newton backend (§3.3).

    Each GPU block handles one (edge, partial) pair. Threads stride across
    points, accumulate local J^T W J and J^T W r, then block-reduce.
    Must be launched with `block_dim=RAYS_THREADS`.
    Equivalent to CUDA's `gauss_newton_rays_step()`.
    """
    comptime assert twc.flat_rank == 2 and xs.flat_rank == 3 and cs.flat_rank == 2
    comptime assert ii.flat_rank == 1 and jj.flat_rank == 1
    comptime assert idx_ii2jj.flat_rank == 2 and valid_match.flat_rank == 2 and q_tensor.flat_rank == 2
    comptime assert hs.flat_rank == 4 and gs.flat_rank == 3
    var num_points = Int(num_points_arg)
    var num_edges = Int(num_edges_arg)
    var blocks_per_edge = Int(blocks_per_edge_arg)

    # ── Thread/block indexing ──
    var edge, edge_block = divmod(block_idx.x, blocks_per_edge)
    var partial_edge = edge * blocks_per_edge + edge_block
    var tid = thread_idx.x
    var block_threads = block_dim.x
    if edge >= num_edges:
        return

    # ── Cache both endpoint Sim(3) poses in shared memory ──
    # Row 0 = pose_i, row 1 = pose_j, row 2 = relative pose T_ij.
    var ix = Int(ii[edge])
    var jx = Int(jj[edge])
    var pose_shared = stack_allocation[.float32, address_space=.SHARED](
        row_major[3, POSE_STRIDE]()
    )
    if tid < POSE_STRIDE:
        pose_shared[0, tid] = twc[ix, tid]
        pose_shared[1, tid] = twc[jx, tid]
    barrier()

    # Read cached poses into register structs
    var pose_i = Sim3Pose(
        Vec3(pose_shared[0, 0][0], pose_shared[0, 1][0], pose_shared[0, 2][0]),
        Quat(pose_shared[0, 3][0], pose_shared[0, 4][0], pose_shared[0, 5][0], pose_shared[0, 6][0]),
        pose_shared[0, 7][0],
    )
    var pose_j = Sim3Pose(
        Vec3(pose_shared[1, 0][0], pose_shared[1, 1][0], pose_shared[1, 2][0]),
        Quat(pose_shared[1, 3][0], pose_shared[1, 4][0], pose_shared[1, 5][0], pose_shared[1, 6][0]),
        pose_shared[1, 7][0],
    )

    # Thread 0 computes the relative pose and publishes to shared memory
    if tid == 0:
        var rel_tmp = pose_i.relative(pose_j)
        pose_shared[2, 0] = rel_tmp.t.x
        pose_shared[2, 1] = rel_tmp.t.y
        pose_shared[2, 2] = rel_tmp.t.z
        pose_shared[2, 3] = rel_tmp.q.x
        pose_shared[2, 4] = rel_tmp.q.y
        pose_shared[2, 5] = rel_tmp.q.z
        pose_shared[2, 6] = rel_tmp.q.w
        pose_shared[2, 7] = rel_tmp.s
    barrier()

    # All threads read the relative pose
    var rel = Sim3Pose(
        Vec3(pose_shared[2, 0][0], pose_shared[2, 1][0], pose_shared[2, 2][0]),
        Quat(pose_shared[2, 3][0], pose_shared[2, 4][0], pose_shared[2, 5][0], pose_shared[2, 6][0]),
        pose_shared[2, 7][0],
    )

    # ── Per-thread accumulators ──
    var hij = Array[Float32, RAYS_HDIM](fill=0.0)
    var vi = Array[Float32, POSE_DIM](fill=0.0)
    var vj = Array[Float32, POSE_DIM](fill=0.0)
    var jxv = Array[Float32, 14](fill=0.0)
    var sigma_ray_inv = 1.0 / sigma_ray
    var sigma_dist_inv = 1.0 / sigma_dist

    def accumulate_row(residual: Float32, w: Float32, ji: TangentVec7) {mut hij, mut vi, mut vj, mut jxv, imm pose_i}:
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

    # ── Main point loop ──
    for k in range(edge_block * block_threads + tid, num_points, block_threads * blocks_per_edge):
        var valid_match_ind = valid_match[edge, k] != 0
        var ind_xi = Int(idx_ii2jj[edge, k])
        if not valid_match_ind:
            ind_xi = 0

        # TileTensor elements are SIMD[dtype, 1]; `[0]` extracts the scalar.
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

        # Accumulate all four residual rows
        accumulate_row(err.x, w0, TangentVec7(drx_dpx, drx_dpy, drx_dpz, 0.0, rj.z, -rj.y, 0.0))
        accumulate_row(err.y, w1, TangentVec7(drx_dpy, dry_dpy, dry_dpz, -rj.z, 0.0, rj.x, 0.0))
        accumulate_row(err.z, w2, TangentVec7(drx_dpz, dry_dpz, drz_dpz, rj.y, -rj.x, 0.0, 0.0))
        accumulate_row(err3, w3, TangentVec7(rj.x, rj.y, rj.z, 0.0, 0.0, 0.0, norm1_j))

    # ── Block-wide reduction and output ──
    var warp_partials = stack_allocation[.float32, address_space=.SHARED](
        row_major[RAYS_WARPS]()
    )

    # gs[side, partial_edge, dim] and hs[block, partial_edge, row, col]
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
