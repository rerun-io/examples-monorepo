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
- Mojo has no CUDA `__shared__` keyword; instead we use
  `stack_allocation[..., address_space=AddressSpace.SHARED]`.
- `@always_inline` replaces CUDA's `__forceinline__ __device__`.
- `warp.sum()` replaces CUDA's `__shfl_down_sync` reduction.
- `barrier()` replaces `__syncthreads()`.
- There is no SIMD vectorisation inside the kernel — the per-point residual
  code is too branchy to benefit, so threads stay scalar and we rely on
  occupancy for throughput. This matches the CUDA version.
"""

from std.math import sqrt, exp, sin, cos, abs
from std.gpu import global_idx, block_idx, block_dim, thread_idx, barrier, lane_id
from std.gpu.primitives import warp
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation

# ── Compile-time constants ───────────────────────────────────────────────────
#
# POSE_DIM   = 7   — degrees of freedom per Sim(3) pose (tx, ty, tz, qx, qy, qz, log_s).
#                     Note: the quaternion's w-component is implicit (unit constraint).
# POSE_STRIDE = 8  — in-memory stride per pose (tx, ty, tz, qx, qy, qz, qw, s).
#                     The stored representation is the *finite* transform, not the Lie algebra.
# RAYS_HDIM       — number of unique elements in the lower-triangle of the symmetric
#                     14×14 block Hessian (7 DOF for pose i + 7 DOF for pose j).
# RAYS_THREADS    — block size for the rays step kernel (128 threads = 4 warps).
# RAYS_WARPS      — number of warps per block, used for the shared-memory reduction buffer.
comptime POSE_DIM = 7
comptime POSE_STRIDE = 8
comptime RAYS_HDIM = 14 * (14 + 1) // 2
comptime RAYS_THREADS = 128
comptime RAYS_WARPS = RAYS_THREADS // 32


# ══════════════════════════════════════════════════════════════════════════════
# Block-level reduction
# ══════════════════════════════════════════════════════════════════════════════

@always_inline
def block_reduce_sum[origin: Origin[mut=True], //](
    value: Float32,
    tid: Int,
    warp_partials: UnsafePointer[Scalar[DType.float32], origin, address_space=AddressSpace.SHARED],
) -> Float32:
    """Sum `value` across all threads in the block and return the result.

    Algorithm (two-phase warp reduction, same as the CUDA `blockReduce`):
      Phase 1 — Intra-warp: each warp reduces its 32 lanes with `warp.sum`.
                Lane 0 of each warp writes the partial to shared memory.
      Phase 2 — Cross-warp: the first warp loads the per-warp partials and
                does a final `warp.sum` to produce the block total.

    Only thread 0 holds the correct total; all other threads get an
    undefined value. The caller must guard writes with `if tid == 0`.
    """
    # Phase 1: reduce within each warp (32 threads) using hardware shuffles.
    var reduced = warp.sum(value)
    if Int(lane_id()) == 0:
        warp_partials[tid // 32] = reduced
    barrier()  # wait for all warps to publish

    # Phase 2: first warp reads the per-warp partials and reduces them.
    var block_sum: Float32 = 0.0
    if tid < RAYS_WARPS:
        block_sum = warp_partials[tid][0]
    if tid < 32:
        block_sum = warp.sum(block_sum)
    barrier()  # ensure the result is visible before the next reduction call
    return block_sum


# ══════════════════════════════════════════════════════════════════════════════
# Quaternion arithmetic
# ══════════════════════════════════════════════════════════════════════════════
#
# Quaternion convention: (x, y, z, w) where w is the scalar part.
# This matches the CUDA backend and Eigen's quaternion layout.
# A unit quaternion q = (x, y, z, w) encodes rotation by angle θ about axis n̂:
#   q = (sin(θ/2)·n̂, cos(θ/2))
#
# Two variants exist for most operations:
#   *_components  → returns InlineArray (allocates a small stack buffer)
#   *_into        → writes results into caller-owned `mut` variables (zero-copy)
# The `_into` versions avoid stack traffic in the innermost loops.

@always_inline
def quat_comp_components(
    aix: Float32, aiy: Float32, aiz: Float32, aiw: Float32,
    bjx: Float32, bjy: Float32, bjz: Float32, bjw: Float32,
) -> InlineArray[Float32, 4]:
    """Hamilton product of two quaternions: q_out = q_a ⊗ q_b.

    This is the standard quaternion multiplication that composes two rotations.
    Equivalent to CUDA's `quat_comp()`.
    """
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = aiw * bjx + aix * bjw + aiy * bjz - aiz * bjy
    out[1] = aiw * bjy - aix * bjz + aiy * bjw + aiz * bjx
    out[2] = aiw * bjz + aix * bjy - aiy * bjx + aiz * bjw
    out[3] = aiw * bjw - aix * bjx - aiy * bjy - aiz * bjz
    return out^


@always_inline
def quat_comp_into(
    aix: Float32, aiy: Float32, aiz: Float32, aiw: Float32,
    bjx: Float32, bjy: Float32, bjz: Float32, bjw: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32, mut out3: Float32,
):
    """Hamilton product written into caller-owned scalars (zero-copy variant)."""
    out0 = aiw * bjx + aix * bjw + aiy * bjz - aiz * bjy
    out1 = aiw * bjy - aix * bjz + aiy * bjw + aiz * bjx
    out2 = aiw * bjz + aix * bjy - aiy * bjx + aiz * bjw
    out3 = aiw * bjw - aix * bjx - aiy * bjy - aiz * bjz


# ══════════════════════════════════════════════════════════════════════════════
# SO(3) / Sim(3) group actions
# ══════════════════════════════════════════════════════════════════════════════

@always_inline
def act_so3_components(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    x0: Float32, x1: Float32, x2: Float32,
) -> InlineArray[Float32, 3]:
    """Rotate a 3D point by a unit quaternion: y = R(q) · x.

    Uses Rodrigues' rotation via the quaternion double-cross-product formula
    (avoids building a 3×3 matrix):
        uv = 2 · (q_xyz × x)
        y  = x + w·uv + (q_xyz × uv)

    Equivalent to CUDA's `actSO3()`.
    """
    var uv0 = 2.0 * (qy * x2 - qz * x1)
    var uv1 = 2.0 * (qz * x0 - qx * x2)
    var uv2 = 2.0 * (qx * x1 - qy * x0)
    var out = InlineArray[Float32, 3](fill=0.0)
    out[0] = x0 + qw * uv0 + (qy * uv2 - qz * uv1)
    out[1] = x1 + qw * uv1 + (qz * uv0 - qx * uv2)
    out[2] = x2 + qw * uv2 + (qx * uv1 - qy * uv0)
    return out^


@always_inline
def act_so3_into(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
):
    """Rotate a 3D point by a unit quaternion (zero-copy variant)."""
    var uv0 = 2.0 * (qy * x2 - qz * x1)
    var uv1 = 2.0 * (qz * x0 - qx * x2)
    var uv2 = 2.0 * (qx * x1 - qy * x0)
    out0 = x0 + qw * uv0 + (qy * uv2 - qz * uv1)
    out1 = x1 + qw * uv1 + (qz * uv0 - qx * uv2)
    out2 = x2 + qw * uv2 + (qx * uv1 - qy * uv0)


# ══════════════════════════════════════════════════════════════════════════════
# Lie-group exponential maps
# ══════════════════════════════════════════════════════════════════════════════

@always_inline
def exp_so3_components(phi0: Float32, phi1: Float32, phi2: Float32) -> InlineArray[Float32, 4]:
    """SO(3) exponential map: axis-angle vector φ → unit quaternion.

    Given φ = θ·n̂ (rotation of angle θ about unit axis n̂):
        q = (sin(θ/2)/θ · φ,  cos(θ/2))

    For small θ (< ~1e-3 rad) we use a Taylor expansion to avoid
    the 0/0 singularity in sin(θ/2)/θ.

    Equivalent to CUDA's `expSO3()`.
    """
    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta_p4 = theta_sq * theta_sq
    var imag: Float32
    var real: Float32
    if theta_sq < 1e-6:
        # Taylor: sin(θ/2)/θ ≈ 1/2 − θ²/48 + θ⁴/3840
        imag = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
        # Taylor: cos(θ/2) ≈ 1 − θ²/8 + θ⁴/384
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
    """Sim(3) exponential map: 7D tangent vector ξ → finite similarity transform.

    The tangent vector ξ = (τ, φ, σ) encodes:
      τ = (xi0, xi1, xi2) — translational velocity
      φ = (xi3, xi4, xi5) — rotational velocity (axis-angle)
      σ = xi6             — log-scale change

    Returns an 8-element array [tx, ty, tz, qx, qy, qz, qw, scale] that can
    be composed with an existing pose via left-multiplication.

    The translation component includes the "V matrix" coupling between rotation,
    scale, and translation that arises from the closed-form Sim(3) Exp map.
    Coefficients A, B, C handle four limit cases (small σ, small θ, both small,
    neither small). See Eq. 1 of the paper and Strasdat's PhD thesis §2.4.

    Equivalent to CUDA's `expSIM3()`.
    """
    var tau0: Float32 = xi0  # translational velocity x
    var tau1: Float32 = xi1  # translational velocity y
    var tau2: Float32 = xi2  # translational velocity z
    var phi0: Float32 = xi3  # rotation axis-angle x
    var phi1: Float32 = xi4  # rotation axis-angle y
    var phi2: Float32 = xi5  # rotation axis-angle z
    var sigma: Float32 = xi6 # log-scale

    var scale = exp(sigma)                                # finite scale factor e^σ
    var q = exp_so3_components(phi0, phi1, phi2)          # rotation quaternion

    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta = sqrt(theta_sq)

    # A, B, C are the "V-matrix" coefficients that couple rotation/scale into
    # the translation part:  t = (C·I + A·[φ]× + B·[φ]×²) · τ
    # Four cases handle the numerical limits of θ→0 and/or σ→0.
    var A: Float32
    var B: Float32
    var C: Float32
    if abs(sigma) < 1e-6:
        C = 1.0                                           # lim σ→0 of (e^σ − 1)/σ = 1
        if abs(theta) < 1e-6:
            A = 0.5                                       # both small: constant approx
            B = 1.0 / 6.0
        else:
            A = (1.0 - cos(theta)) / theta_sq             # small σ, finite θ
            B = (theta - sin(theta)) / (theta_sq * theta)
    else:
        C = (scale - 1.0) / sigma                         # (e^σ − 1) / σ
        if abs(theta) < 1e-6:
            var sigma_sq = sigma * sigma
            A = ((sigma - 1.0) * scale + 1.0) / sigma_sq  # finite σ, small θ
            B = (scale * 0.5 * sigma_sq + scale - 1.0 - sigma * scale) / (sigma_sq * sigma)
        else:
            # General case: both σ and θ are non-negligible.
            var a = scale * sin(theta)
            var b = scale * cos(theta)
            var c = theta_sq + sigma * sigma
            A = (a * sigma + (1.0 - b) * theta) / (theta * c)
            B = (C - (((b - 1.0) * sigma + a * theta) / c)) / theta_sq

    # Build the translation output:  t = C·τ + A·(φ × τ) + B·(φ × (φ × τ))
    var out = InlineArray[Float32, 8](fill=0.0)
    out[0] = C * tau0
    out[1] = C * tau1
    out[2] = C * tau2

    # First cross product: φ × τ
    var cx0: Float32 = phi1 * tau2 - phi2 * tau1
    var cx1: Float32 = phi2 * tau0 - phi0 * tau2
    var cx2: Float32 = phi0 * tau1 - phi1 * tau0
    out[0] += A * cx0
    out[1] += A * cx1
    out[2] += A * cx2

    # Second cross product: φ × (φ × τ)
    var c2x0: Float32 = phi1 * cx2 - phi2 * cx1
    var c2x1: Float32 = phi2 * cx0 - phi0 * cx2
    var c2x2: Float32 = phi0 * cx1 - phi1 * cx0
    out[0] += B * c2x0
    out[1] += B * c2x1
    out[2] += B * c2x2
    out[3] = q[0]   # quaternion x
    out[4] = q[1]   # quaternion y
    out[5] = q[2]   # quaternion z
    out[6] = q[3]   # quaternion w
    out[7] = scale   # finite scale
    return out^


@always_inline
def quat_inv_components(
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
) -> InlineArray[Float32, 4]:
    """Quaternion conjugate (= inverse for unit quaternions): q⁻¹ = (−x, −y, −z, w)."""
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
    """Apply a Sim(3) transform to a 3D point: y = s·R(q)·x + t.

    This is the fundamental operation that projects a point from one keyframe
    into another. The paper parameterises each world-from-camera pose as
    T_WC = (t, q, s) ∈ Sim(3).

    Equivalent to CUDA's `actSIM3()`.
    """
    var rot = act_so3_components(qx, qy, qz, qw, x0, x1, x2)
    var out = InlineArray[Float32, 3](fill=0.0)
    out[0] = rot[0] * s + tx
    out[1] = rot[1] * s + ty
    out[2] = rot[2] * s + tz
    return out^


@always_inline
def act_sim3_into(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
):
    """Apply a Sim(3) transform to a 3D point (zero-copy variant)."""
    act_so3_into(qx, qy, qz, qw, x0, x1, x2, out0, out1, out2)
    out0 = out0 * s + tx
    out1 = out1 * s + ty
    out2 = out2 * s + tz


# ── Small vector helpers ─────────────────────────────────────────────────────

@always_inline
def dot3_components(
    a0: Float32, a1: Float32, a2: Float32,
    b0: Float32, b1: Float32, b2: Float32,
) -> Float32:
    """3D dot product: a · b."""
    return a0 * b0 + a1 * b1 + a2 * b2


@always_inline
def squared_norm3_components(x0: Float32, x1: Float32, x2: Float32) -> Float32:
    """Squared Euclidean norm: ‖x‖²."""
    return x0 * x0 + x1 * x1 + x2 * x2


@always_inline
def huber_scalar(r: Float32) -> Float32:
    """Huber robust weight for a scalar residual.

    Returns 1.0 inside the Huber threshold (1.345) and decreases linearly
    outside it, which down-weights outlier correspondences that would
    otherwise dominate the least-squares cost. The threshold 1.345 makes the
    estimator 95% as efficient as OLS on Gaussian data.

    This is used as a multiplicative weight on the squared residual: the
    caller computes  w = huber(√w · r) · w,  giving the iteratively
    reweighted least-squares (IRLS) form of Huber loss.
    """
    var r_abs = abs(r)
    if r_abs < 1.345:
        return 1.0
    return 1.345 / r_abs


# ══════════════════════════════════════════════════════════════════════════════
# Relative-pose and adjoint operations
# ══════════════════════════════════════════════════════════════════════════════

@always_inline
def rel_sim3_components(
    tix: Float32, tiy: Float32, tiz: Float32,
    qix: Float32, qiy: Float32, qiz: Float32, qiw: Float32,
    si: Float32,
    tjx: Float32, tjy: Float32, tjz: Float32,
    qjx: Float32, qjy: Float32, qjz: Float32, qjw: Float32,
    sj: Float32,
) -> InlineArray[Float32, 8]:
    """Compute the relative Sim(3) transform: T_ij = T_WC_i⁻¹ · T_WC_j.

    All backend residuals are formulated in terms of *relative* poses between
    pairs of keyframes (edges in the pose graph), not in the world frame
    directly. This function extracts that relative transform from two world
    poses.

    Returns [tx, ty, tz, qx, qy, qz, qw, s] of the relative transform.
    Equivalent to CUDA's `relSIM3()`.
    """
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
def rel_sim3_into(
    tix: Float32, tiy: Float32, tiz: Float32,
    qix: Float32, qiy: Float32, qiz: Float32, qiw: Float32,
    si: Float32,
    tjx: Float32, tjy: Float32, tjz: Float32,
    qjx: Float32, qjy: Float32, qjz: Float32, qjw: Float32,
    sj: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
    mut out3: Float32, mut out4: Float32, mut out5: Float32, mut out6: Float32,
    mut out7: Float32,
):
    """Relative Sim(3) transform (zero-copy variant)."""
    var si_inv = 1.0 / si
    var qi_inv0 = -qix
    var qi_inv1 = -qiy
    var qi_inv2 = -qiz
    var qi_inv3 = qiw
    quat_comp_into(qi_inv0, qi_inv1, qi_inv2, qi_inv3, qjx, qjy, qjz, qjw, out3, out4, out5, out6)
    var dt0 = tjx - tix
    var dt1 = tjy - tiy
    var dt2 = tjz - tiz
    act_so3_into(qi_inv0, qi_inv1, qi_inv2, qi_inv3, dt0, dt1, dt2, out0, out1, out2)
    out0 = out0 * si_inv
    out1 = out1 * si_inv
    out2 = out2 * si_inv
    out7 = si_inv * sj


@always_inline
def apply_sim3_adj_inv_components(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    x3: Float32, x4: Float32, x5: Float32,
    x6: Float32,
) -> InlineArray[Float32, 7]:
    """Apply the inverse adjoint of a Sim(3) element to a 7D tangent vector.

    When the residual Jacobian is computed with respect to the *relative* pose
    T_ij, but we need it with respect to the *world* poses T_WC_i and T_WC_j,
    the chain rule introduces the adjoint representation of Sim(3). This
    function computes  Ad(T_WC_i)⁻¹ · ξ  to map a relative-frame Jacobian
    column into the world-frame Jacobian for pose i (and its negation gives
    the Jacobian for pose j).

    Input ξ = (x0..x2, x3..x5, x6) is a 7D sim(3) tangent vector.
    Output is the transformed 7D tangent vector.

    Equivalent to CUDA's `apply_sim3_adj_inv()`.
    """
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


@always_inline
def apply_sim3_adj_inv_into(
    tx: Float32, ty: Float32, tz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    s: Float32,
    x0: Float32, x1: Float32, x2: Float32,
    x3: Float32, x4: Float32, x5: Float32,
    x6: Float32,
    mut out0: Float32, mut out1: Float32, mut out2: Float32,
    mut out3: Float32, mut out4: Float32, mut out5: Float32,
    mut out6: Float32,
):
    """Inverse adjoint of Sim(3) applied to a tangent vector (zero-copy variant)."""
    var s_inv = 1.0 / s
    var ra0: Float32 = 0.0
    var ra1: Float32 = 0.0
    var ra2: Float32 = 0.0
    act_so3_into(qx, qy, qz, qw, x0, x1, x2, ra0, ra1, ra2)
    out0 = s_inv * ra0
    out1 = s_inv * ra1
    out2 = s_inv * ra2

    var rb0: Float32 = 0.0
    var rb1: Float32 = 0.0
    var rb2: Float32 = 0.0
    act_so3_into(qx, qy, qz, qw, x3, x4, x5, rb0, rb1, rb2)
    out3 = rb0 + s_inv * (ty * ra2 - tz * ra1)
    out4 = rb1 + s_inv * (tz * ra0 - tx * ra2)
    out5 = rb2 + s_inv * (tx * ra1 - ty * ra0)
    out6 = x6 + s_inv * dot3_components(tx, ty, tz, ra0, ra1, ra2)


# ══════════════════════════════════════════════════════════════════════════════
# Main Gauss-Newton ray-cost kernel
# ══════════════════════════════════════════════════════════════════════════════

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
    blocks_per_edge: Int,
    sigma_ray: Float32,
    sigma_dist: Float32,
    c_thresh: Float32,
    q_thresh: Float32,
):
    """One linearisation step of the ray-based Gauss-Newton backend (§3.3).

    This is the hot inner kernel — it runs on the GPU and processes every
    matched point across every edge in the pose graph. Its outputs are
    per-edge Hessian and gradient blocks that get assembled into the normal
    equations on the CPU.

    Arguments (all GPU pointers)
    ----------------------------
    twc          — world-from-camera Sim(3) poses, shape [N_kf, 8].
                   Layout per pose: [tx, ty, tz, qx, qy, qz, qw, scale].
    xs           — per-keyframe 3D pointmaps from MASt3R, shape [N_kf, N_pts, 3].
    cs           — per-point confidence from MASt3R, shape [N_kf, N_pts].
    ii, jj       — edge endpoint indices, shape [N_edges]. Edge k connects
                   keyframe ii[k] to keyframe jj[k].
    idx_ii2jj    — per-edge per-point mapping: for each point in view j, the
                   index of its matched point in view i. Shape [N_edges, N_pts].
    valid_match  — bool mask, shape [N_edges, N_pts]. 1 if the match is valid.
    q_tensor     — match quality score from the frontend, shape [N_edges, N_pts].
    hs           — OUTPUT: per-edge 7×7 Hessian blocks (4 sub-blocks for the
                   2×2 block structure: H_ii, H_ij, H_ji, H_jj).
    gs           — OUTPUT: per-edge 7-element gradient blocks (2 sub-blocks:
                   g_i, g_j).
    sigma_ray    — standard deviation for the ray-direction residual.
    sigma_dist   — standard deviation for the distance residual.
    c_thresh     — minimum MASt3R confidence to include a correspondence.
    q_thresh     — minimum match quality to include a correspondence.

    Residuals (per matched point pair)
    ----------------------------------
    The paper defines two residual families for each valid correspondence:

    1. **Ray residual** (3D):  r_ray = r̂_j→i − r̂_i
       where r̂ = x/‖x‖ is the unit ray direction after transforming
       point j into frame i via the relative Sim(3).

    2. **Distance residual** (1D):  r_dist = ‖x_j→i‖ − ‖x_i‖
       This stabilises the system when parallax is low (pure rotation),
       because angular error alone provides no depth constraint.

    Both residuals are weighted by √(confidence · quality) and passed through
    a Huber robust kernel to reduce the influence of outlier matches.

    Block structure
    ---------------
    Each GPU block handles one (edge, partial) pair. Threads stride across
    points, accumulate local J^T W J and J^T W r contributions, then do a
    block-wide reduction to produce the per-edge Hessian/gradient output.

    Equivalent to CUDA's `gauss_newton_rays_step()`.
    """
    # ── Thread/block indexing ──
    # Grid is 1D with (num_edges × blocks_per_edge) total blocks.
    # Each block works on one (edge, partial-chunk) pair.
    var edge = Int(block_idx.x) // blocks_per_edge       # which edge in the pose graph
    var edge_block = Int(block_idx.x) % blocks_per_edge   # which chunk of this edge
    var partial_edge = edge * blocks_per_edge + edge_block # linear index into partial arrays
    var num_partials = num_edges * blocks_per_edge
    var tid = Int(thread_idx.x)      # thread ID within this block (0..RAYS_THREADS-1)
    var block_threads = Int(block_dim.x)
    if edge >= num_edges:
        return

    # ── Load edge endpoints ──
    var ix = Int(ii[edge])  # keyframe index i (left endpoint)
    var jx = Int(jj[edge])  # keyframe index j (right endpoint)
    var pi = ix * POSE_STRIDE  # byte offset into twc[] for pose i
    var pj = jx * POSE_STRIDE  # byte offset into twc[] for pose j

    # ── Cache both endpoint Sim(3) poses in shared memory ──
    # Layout: [0..7] = pose_i, [8..15] = pose_j, [16..23] = rel_sim3
    # Each pose is 8 floats: [tx, ty, tz, qx, qy, qz, qw, scale]
    var pose_shared = stack_allocation[
        24,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    if tid < 8:
        pose_shared[tid] = twc[pi + tid]      # pose i → shared[0..7]
    if tid < 8:
        pose_shared[tid + 8] = twc[pj + tid]  # pose j → shared[8..15]
    barrier()  # all threads must see both poses
    var ti0 = pose_shared[0][0]
    var ti1 = pose_shared[1][0]
    var ti2 = pose_shared[2][0]
    var qi0 = pose_shared[3][0]
    var qi1 = pose_shared[4][0]
    var qi2 = pose_shared[5][0]
    var qi3 = pose_shared[6][0]
    var si0 = pose_shared[7][0]
    var tj0 = pose_shared[8][0]
    var tj1 = pose_shared[9][0]
    var tj2 = pose_shared[10][0]
    var qj0 = pose_shared[11][0]
    var qj1 = pose_shared[12][0]
    var qj2 = pose_shared[13][0]
    var qj3 = pose_shared[14][0]
    var sj0 = pose_shared[15][0]

    if tid == 0:
        var rel_out0: Float32 = 0.0
        var rel_out1: Float32 = 0.0
        var rel_out2: Float32 = 0.0
        var rel_out3: Float32 = 0.0
        var rel_out4: Float32 = 0.0
        var rel_out5: Float32 = 0.0
        var rel_out6: Float32 = 0.0
        var rel_out7: Float32 = 0.0
        rel_sim3_into(
            ti0, ti1, ti2,
            qi0, qi1, qi2, qi3,
            si0,
            tj0, tj1, tj2,
            qj0, qj1, qj2, qj3,
            sj0,
            rel_out0, rel_out1, rel_out2, rel_out3, rel_out4, rel_out5, rel_out6, rel_out7,
        )
        pose_shared[16] = rel_out0
        pose_shared[17] = rel_out1
        pose_shared[18] = rel_out2
        pose_shared[19] = rel_out3
        pose_shared[20] = rel_out4
        pose_shared[21] = rel_out5
        pose_shared[22] = rel_out6
        pose_shared[23] = rel_out7
    barrier()
    # pose_shared[16:24] now stores the relative Sim(3) for this edge.
    var rel0 = pose_shared[16][0]
    var rel1 = pose_shared[17][0]
    var rel2 = pose_shared[18][0]
    var rel3 = pose_shared[19][0]
    var rel4 = pose_shared[20][0]
    var rel5 = pose_shared[21][0]
    var rel6 = pose_shared[22][0]
    var rel7 = pose_shared[23][0]

    # ── Per-thread accumulators ──
    # hij  — lower triangle of the 14×14 block Hessian (J^T W J) for this edge,
    #         stored as a flat array of RAYS_HDIM = 105 unique elements.
    # vi   — gradient block for pose i (J_i^T W r), 7 elements.
    # vj   — gradient block for pose j (J_j^T W r), 7 elements.
    # jxv  — scratch space for the 14-element Jacobian row [J_i | J_j].
    var hij = InlineArray[Float32, RAYS_HDIM](fill=0.0)
    var vi = InlineArray[Float32, POSE_DIM](fill=0.0)
    var vj = InlineArray[Float32, POSE_DIM](fill=0.0)
    var jxv = InlineArray[Float32, 14](fill=0.0)

    var sigma_ray_inv = 1.0 / sigma_ray    # precision (1/σ) for ray residual
    var sigma_dist_inv = 1.0 / sigma_dist   # precision (1/σ) for distance residual

    # ── Main point loop ──
    # Each thread strides across points with step = (block_threads × blocks_per_edge)
    # so all threads collectively cover all num_points for this edge.
    for k in range(edge_block * block_threads + tid, num_points, block_threads * blocks_per_edge):
        # Each point contributes two residual families from the paper:
        # 1. ray-direction disagreement after transforming j into i,
        # 2. distance disagreement to stabilize low-parallax / pure-rotation
        #    cases where angular error alone is weak.
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

        # ── Compute unit ray for point in frame i ──
        # r̂_i = x_i / ‖x_i‖
        var norm2_i = squared_norm3_components(xi0, xi1, xi2)
        var norm1_i = sqrt(norm2_i)
        var norm1_i_inv = 1.0 / norm1_i
        var ri0 = norm1_i_inv * xi0
        var ri1 = norm1_i_inv * xi1
        var ri2 = norm1_i_inv * xi2

        # ── Transform point j into frame i using the relative Sim(3) ──
        # x_j→i = T_ij · x_j = s_ij · R_ij · x_j + t_ij
        var xj_ci0: Float32 = 0.0
        var xj_ci1: Float32 = 0.0
        var xj_ci2: Float32 = 0.0
        act_sim3_into(
            rel0, rel1, rel2, rel3, rel4, rel5, rel6, rel7, xj0, xj1, xj2, xj_ci0, xj_ci1, xj_ci2
        )

        # ── Compute unit ray for transformed point j ──
        # r̂_j→i = x_j→i / ‖x_j→i‖
        var norm2_j = squared_norm3_components(xj_ci0, xj_ci1, xj_ci2)
        var norm1_j = sqrt(norm2_j)
        var norm1_j_inv = 1.0 / norm1_j
        var rj0 = norm1_j_inv * xj_ci0
        var rj1 = norm1_j_inv * xj_ci1
        var rj2 = norm1_j_inv * xj_ci2

        # ── Residuals ──
        # err0..err2: ray direction disagreement (3D)
        # err3: depth (distance) disagreement (1D)
        var err0 = rj0 - ri0
        var err1 = rj1 - ri1
        var err2 = rj2 - ri2
        var err3 = norm1_j - norm1_i

        # Confidence and validity thresholds decide whether MASt3R considers
        # this correspondence trustworthy enough to influence optimization.
        var valid = valid_match_ind and (qv > q_thresh) and (ci > c_thresh) and (cj > c_thresh)
        var conf_weight: Float32 = qv
        var sqrt_w_ray: Float32 = 0.0
        var sqrt_w_dist: Float32 = 0.0
        if valid:
            var sqrt_conf = sqrt(conf_weight)
            sqrt_w_ray = sigma_ray_inv * sqrt_conf
            sqrt_w_dist = sigma_dist_inv * sqrt_conf

        # ── IRLS weights (Huber + confidence + precision) ──
        # w = huber(√w·r) · w, where w = conf / σ². This is the iteratively
        # reweighted least-squares form of Huber-robust estimation.
        var w0 = huber_scalar(sqrt_w_ray * err0) * sqrt_w_ray * sqrt_w_ray
        var w1 = huber_scalar(sqrt_w_ray * err1) * sqrt_w_ray * sqrt_w_ray
        var w2 = huber_scalar(sqrt_w_ray * err2) * sqrt_w_ray * sqrt_w_ray
        var w3 = huber_scalar(sqrt_w_dist * err3) * sqrt_w_dist * sqrt_w_dist

        # ── Jacobian of the ray-normalisation step ──
        # ∂r̂/∂p = (I − r̂ r̂^T) / ‖p‖, i.e. the projection onto the plane
        # perpendicular to the ray, divided by the distance. Only the upper
        # triangle is computed (the matrix is symmetric).
        var norm3_j_inv = norm1_j_inv / norm2_j
        var drx_dpx = norm1_j_inv - xj_ci0 * xj_ci0 * norm3_j_inv
        var dry_dpy = norm1_j_inv - xj_ci1 * xj_ci1 * norm3_j_inv
        var drz_dpz = norm1_j_inv - xj_ci2 * xj_ci2 * norm3_j_inv
        var drx_dpy = -xj_ci0 * xj_ci1 * norm3_j_inv
        var drx_dpz = -xj_ci0 * xj_ci2 * norm3_j_inv
        var dry_dpz = -xj_ci1 * xj_ci2 * norm3_j_inv

        @parameter
        def accumulate_row(
            err: Float32,
            w: Float32,
            ji0: Float32, ji1: Float32, ji2: Float32, ji3: Float32, ji4: Float32, ji5: Float32, ji6: Float32,
        ):
            # Map the relative-pose Jacobian for this residual row into the two
            # world-pose blocks it touches: pose i gets the negative term and
            # pose j gets the positive term.
            var jadj0: Float32 = 0.0
            var jadj1: Float32 = 0.0
            var jadj2: Float32 = 0.0
            var jadj3: Float32 = 0.0
            var jadj4: Float32 = 0.0
            var jadj5: Float32 = 0.0
            var jadj6: Float32 = 0.0
            apply_sim3_adj_inv_into(
                ti0, ti1, ti2,
                qi0, qi1, qi2, qi3, si0,
                ji0, ji1, ji2, ji3, ji4, ji5, ji6,
                jadj0, jadj1, jadj2, jadj3, jadj4, jadj5, jadj6,
            )
            jxv[0] = -jadj0
            jxv[1] = -jadj1
            jxv[2] = -jadj2
            jxv[3] = -jadj3
            jxv[4] = -jadj4
            jxv[5] = -jadj5
            jxv[6] = -jadj6
            jxv[7] = jadj0
            jxv[8] = jadj1
            jxv[9] = jadj2
            jxv[10] = jadj3
            jxv[11] = jadj4
            jxv[12] = jadj5
            jxv[13] = jadj6
            var l = 0
            for n in range(14):
                for m in range(n + 1):
                    hij[l] += w * jxv[n] * jxv[m]
                    l += 1
            for n in range(POSE_DIM):
                vi[n] += w * err * jxv[n]
                vj[n] += w * err * jxv[n + 7]

        # ── Accumulate all four residual rows ──
        # Each call processes one scalar residual, providing the 7-element
        # Jacobian row [∂r/∂(translation), ∂r/∂(rotation), ∂r/∂(scale)]
        # with respect to the relative pose. The accumulate_row closure
        # maps these into world-frame Jacobians and updates H and g.
        #
        # Rows 0-2: ray-direction residuals (drx_dp* are the normalisation
        #           Jacobian; the rotation columns come from the cross-product
        #           form of ∂(Rx)/∂φ = -[Rx]×).
        # Row 3:    distance residual (Jacobian is just the ray direction
        #           for translation, and ‖x‖ for scale).
        accumulate_row(err0, w0, drx_dpx, drx_dpy, drx_dpz, 0.0, rj2, -rj1, 0.0)
        accumulate_row(err1, w1, drx_dpy, dry_dpy, dry_dpz, -rj2, 0.0, rj0, 0.0)
        accumulate_row(err2, w2, drx_dpz, dry_dpz, drz_dpz, rj1, -rj0, 0.0, 0.0)
        accumulate_row(err3, w3, rj0, rj1, rj2, 0.0, 0.0, 0.0, norm1_j)

    var warp_partials = stack_allocation[
        RAYS_WARPS,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    # The solver expects the same four 7x7 block layout as the CUDA backend:
    # H_ii, H_ij, H_ji, H_jj and the two gradient blocks g_i, g_j.
    for n in range(POSE_DIM):
        var vi_sum = block_reduce_sum(vi[n], tid, warp_partials)
        if tid == 0:
            gs[(0 * num_partials + partial_edge) * POSE_DIM + n] = vi_sum

        var vj_sum = block_reduce_sum(vj[n], tid, warp_partials)
        if tid == 0:
            gs[(1 * num_partials + partial_edge) * POSE_DIM + n] = vj_sum

    var l = 0
    for n in range(14):
        for m in range(n + 1):
            var h_sum = block_reduce_sum(hij[l], tid, warp_partials)
            if tid == 0:
                var val = h_sum
                if n < 7 and m < 7:
                    hs[((0 * num_partials + partial_edge) * 49) + n * 7 + m] = val
                    hs[((0 * num_partials + partial_edge) * 49) + m * 7 + n] = val
                elif n >= 7 and m < 7:
                    hs[((1 * num_partials + partial_edge) * 49) + m * 7 + (n - 7)] = val
                    hs[((2 * num_partials + partial_edge) * 49) + (n - 7) * 7 + m] = val
                else:
                    hs[((3 * num_partials + partial_edge) * 49) + (n - 7) * 7 + (m - 7)] = val
                    hs[((3 * num_partials + partial_edge) * 49) + (m - 7) * 7 + (n - 7)] = val
            l += 1


# ══════════════════════════════════════════════════════════════════════════════
# Pose retraction kernel
# ══════════════════════════════════════════════════════════════════════════════

def pose_retr_kernel(
    poses: UnsafePointer[Float32, MutAnyOrigin],
    dx: UnsafePointer[Float32, MutAnyOrigin],
    num_fix: Int,
    num_poses: Int,
):
    """Apply the solved Sim(3) tangent update to every unfixed pose in place.

    After the GN system  H dx = −g  has been solved, `dx` contains a 7D Lie
    algebra vector for each unfixed pose. This kernel:
      1. Converts each dx to a finite Sim(3) element via `exp_sim3`.
      2. Left-multiplies it onto the existing pose: T_new = Exp(dx) · T_old.

    The first `num_fix` poses (the gauge anchor) are skipped.
    One thread per unfixed pose — trivially parallel, no shared memory needed.

    Equivalent to CUDA's `pose_retr_kernel()`.
    """
    var k: Int = Int(global_idx.x) + num_fix
    if k >= num_poses:
        return
    # Step 1: Convert the 7D tangent vector dx into a finite Sim(3) element.
    # xi = Exp(dx) = [t_new, q_new, s_new]
    var xi = exp_sim3_components(
        dx[(k - num_fix) * POSE_DIM + 0],
        dx[(k - num_fix) * POSE_DIM + 1],
        dx[(k - num_fix) * POSE_DIM + 2],
        dx[(k - num_fix) * POSE_DIM + 3],
        dx[(k - num_fix) * POSE_DIM + 4],
        dx[(k - num_fix) * POSE_DIM + 5],
        dx[(k - num_fix) * POSE_DIM + 6],
    )
    # Step 2: Left-multiply onto existing pose: T_new = Exp(dx) · T_old
    #   translation: t_new = s_xi · R_xi · t_old + t_xi
    #   rotation:    q_new = q_xi ⊗ q_old
    #   scale:       s_new = s_xi · s_old
    var p = k * 8  # offset into the flat pose array
    var rot_t = act_so3_components(            # R_xi · t_old
        xi[3], xi[4], xi[5], xi[6],
        poses[p + 0], poses[p + 1], poses[p + 2],
    )
    var q1 = quat_comp_components(             # q_xi ⊗ q_old
        xi[3], xi[4], xi[5], xi[6],
        poses[p + 3], poses[p + 4], poses[p + 5], poses[p + 6],
    )
    poses[p + 0] = rot_t[0] * xi[7] + xi[0]  # t_new.x = s_xi · (R_xi · t_old).x + t_xi.x
    poses[p + 1] = rot_t[1] * xi[7] + xi[1]  # t_new.y
    poses[p + 2] = rot_t[2] * xi[7] + xi[2]  # t_new.z
    poses[p + 3] = q1[0]                       # q_new.x
    poses[p + 4] = q1[1]                       # q_new.y
    poses[p + 5] = q1[2]                       # q_new.z
    poses[p + 6] = q1[3]                       # q_new.w
    poses[p + 7] = xi[7] * poses[p + 7]       # s_new = s_xi · s_old
