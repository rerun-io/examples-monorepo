"""The linearisation step of MASt3R-SLAM's Gauss-Newton backends (paper §3.3; `gn_kernels.cu`'s
`ray_align_kernel` and `calib_proj_kernel`), one kernel for both costs.

Each keyframe pose is a Sim(3) element stored as (tx, ty, tz, qx, qy, qz, qw, s). An edge (i, j) compares the
points of keyframe j, moved into keyframe i by T_ij = T_i⁻¹ T_j, with their matches in keyframe i:
- rays: the ray direction (3 rows) and the distance (1 row) of each point;
- calib: the pixel of the projected point (2 rows) and its log-depth (1 row), with intrinsics K.

Every row adds w Jᵀ J and w r Jᵀ for the 14 pose variables [T_i, T_j] of its edge. A thread keeps the sums of
its points in registers: the 105 entries of the lower triangle of the 14 × 14 block, then the 7 + 7 gradient
entries (`SUMS` = 119, in that order). `comptime for` unrolls the accumulation so the sums stay in registers
(a runtime loop over them puts them on the stack). Grid (blocks per edge, edges), `THREADS` per block; each
block reduces its sums by warp and adds them to `sums[edge]` with relaxed device-scope atomics (CUDA's
atomicAdd), so `sums` must be zeroed before the launch. The CUDA kernel uses one block per edge; several
blocks per edge keep the GPU busy when a graph has few edges.

Points with zero weight (invalid match, confidence below threshold, behind the camera) are skipped: in the
CUDA kernel they add exact zeros.
"""

from std.atomic import Atomic, Ordering
from std.math import abs, log, sqrt
from std.utils.static_tuple import StaticTuple

from layout import row_major, stack_allocation
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.primitives import warp
from max.gpu.sync import barrier

comptime THREADS = 128
comptime WARPS = THREADS // WARP_SIZE
comptime H_DIM = 14 * (14 + 1) // 2
comptime SUMS = H_DIM + 14
comptime POSE_STRIDE = 8

comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime I64Ptr = MutPointer[Int64, MutUntrackedOrigin]
comptime U8Ptr = MutPointer[UInt8, MutUntrackedOrigin]


@fieldwise_init
struct Vec3(TrivialRegisterPassable):
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
        return self.dot(self)


@fieldwise_init
struct Quat(TrivialRegisterPassable):
    """Unit quaternion (x, y, z, w), w the scalar part."""

    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    @always_inline
    def __mul__(self, other: Quat) -> Quat:
        return Quat(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    @always_inline
    def conjugate(self) -> Quat:
        return Quat(-self.x, -self.y, -self.z, self.w)

    @always_inline
    def rotate(self, v: Vec3) -> Vec3:
        """R(q) v by the double cross product (CUDA's `actSO3`)."""
        var uv = Vec3(2.0 * (self.y * v.z - self.z * v.y), 2.0 * (self.z * v.x - self.x * v.z), 2.0 * (self.x * v.y - self.y * v.x))
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

    @always_inline
    def act(self, pt: Vec3) -> Vec3:
        """s R(q) pt + t (CUDA's `actSim3`)."""
        return self.q.rotate(pt) * self.s + self.t

    @always_inline
    def relative(self, other: Sim3Pose) -> Sim3Pose:
        """self⁻¹ other (CUDA's `relSim3`)."""
        var s_inv: Float32 = 1.0 / self.s
        var q_inv = self.q.conjugate()
        return Sim3Pose(q_inv.rotate(other.t - self.t) * s_inv, q_inv * other.q, s_inv * other.s)


@always_inline
def _pose(twc: F32Ptr, i: Int) -> Sim3Pose:
    var p = i * POSE_STRIDE
    return Sim3Pose(
        Vec3(twc[unsafe_offset=p], twc[unsafe_offset=p + 1], twc[unsafe_offset=p + 2]),
        Quat(twc[unsafe_offset=p + 3], twc[unsafe_offset=p + 4], twc[unsafe_offset=p + 5], twc[unsafe_offset=p + 6]),
        twc[unsafe_offset=p + 7],
    )


@always_inline
def _point(xs: F32Ptr, row: Int) -> Vec3:
    return Vec3(xs[unsafe_offset=3 * row], xs[unsafe_offset=3 * row + 1], xs[unsafe_offset=3 * row + 2])


@always_inline
def _huber(r: Float32) -> Float32:
    var r_abs = abs(r)
    return 1.0 if r_abs < 1.345 else 1.345 / r_abs


@always_inline
def _add_row(mut acc: Array[Float32, SUMS], pose_i: Sim3Pose, w: Float32, r: Float32, j: SIMD[DType.float32, 8]):
    """Add one residual row: J_j = Ad⁻¹(T_i) j (CUDA's `apply_Sim3_adj_inv`), J_i = -J_j, J = [J_i, J_j]."""
    var s_inv: Float32 = 1.0 / pose_i.s
    var ra = pose_i.q.rotate(Vec3(j[0], j[1], j[2]))
    var rb = pose_i.q.rotate(Vec3(j[3], j[4], j[5]))
    var t = pose_i.t
    var jj = SIMD[DType.float32, 8](
        s_inv * ra.x,
        s_inv * ra.y,
        s_inv * ra.z,
        rb.x + s_inv * (t.y * ra.z - t.z * ra.y),
        rb.y + s_inv * (t.z * ra.x - t.x * ra.z),
        rb.z + s_inv * (t.x * ra.y - t.y * ra.x),
        j[6] + s_inv * t.dot(ra),
        0.0,
    )
    var jx = (-jj).join(jj)  # [J_i (7), 0, J_j (7), 0]
    comptime for n in range(14):
        comptime xn = n + n // 7
        var wn = w * jx[xn]
        comptime for m in range(n + 1):
            acc[n * (n + 1) // 2 + m] += wn * jx[m + m // 7]
    var wr = w * r
    comptime for n in range(14):
        acc[H_DIM + n] += wr * jx[n + n // 7]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](THREADS))
def gn_step_kernel[calib: Bool](
    num_points: Int64, blocks_per_edge: Int64, height: Int64, width: Int64, pixel_border: Int64,
    sigma_a: Float32, sigma_b: Float32, c_thresh: Float32, q_thresh: Float32, z_eps: Float32,
    twc: F32Ptr, xs: F32Ptr, cs: F32Ptr, k_mat: F32Ptr, ii: I64Ptr, jj: I64Ptr, idx: I64Ptr, valid: U8Ptr, q: F32Ptr,
    sums: F32Ptr,
):  # fmt: skip
    """sigma_a / sigma_b: sigma_ray / sigma_dist (rays) or sigma_pixel / sigma_depth (calib). Shapes: twc (P, 8),
    xs (P, n, 3), cs (P, n, 1), k_mat (3, 3) (calib only), ii / jj (E,) rows of twc, idx (E, n), valid (E, n, 1)
    bool, q (E, n, 1), sums (E, SUMS)."""
    var n = Int(num_points)
    var e = block_idx.y
    var t = thread_idx.x
    var ix = Int(ii[unsafe_offset=e])
    var jx = Int(jj[unsafe_offset=e])
    var pose_i = _pose(twc, ix)
    var rel = pose_i.relative(_pose(twc, jx))
    var inv_a: Float32 = 1.0 / sigma_a
    var inv_b: Float32 = 1.0 / sigma_b
    var acc = Array[Float32, SUMS](fill=0.0)

    var stride = THREADS * Int(blocks_per_edge)
    for k in range(block_idx.x * THREADS + t, n, stride):
        var m = e * n + k
        var matched = valid[unsafe_offset=m] != 0
        var ind = Int(idx[unsafe_offset=m]) if matched else 0
        var qv = q[unsafe_offset=m]
        if not (matched and qv > q_thresh and cs[unsafe_offset=ix * n + ind] > c_thresh and cs[unsafe_offset=jx * n + k] > c_thresh):
            continue
        var xi = _point(xs, ix * n + ind)
        var xj = rel.act(_point(xs, jx * n + k))
        var sqrt_q = sqrt(qv)
        comptime if calib:
            if not (xj.z > z_eps and xi.z > z_eps):
                continue
            var fx = k_mat[unsafe_offset=0]
            var fy = k_mat[unsafe_offset=4]
            var z_inv: Float32 = 1.0 / xj.z
            var x_z = xj.x * z_inv
            var y_z = xj.y * z_inv
            var u = fx * x_z + k_mat[unsafe_offset=2]
            var v = fy * y_z + k_mat[unsafe_offset=5]
            var border = Float32(pixel_border)
            if not (u > border and u < Float32(width - 1 - pixel_border) and v > border and v < Float32(height - 1 - pixel_border)):
                continue
            var err_u = u - Float32(ind % Int(width))
            var err_v = v - Float32(ind // Int(width))
            var err_z = log(xj.z) - log(xi.z)
            var sw_pixel = inv_a * sqrt_q
            var sw_depth = inv_b * sqrt_q
            var w_u = _huber(sw_pixel * err_u) * sw_pixel * sw_pixel
            var w_v = _huber(sw_pixel * err_v) * sw_pixel * sw_pixel
            var w_z = _huber(sw_depth * err_z) * sw_depth * sw_depth
            _add_row(acc, pose_i, w_u, err_u, SIMD[DType.float32, 8](fx * z_inv, 0.0, -fx * x_z * z_inv, -fx * x_z * y_z, fx * (1.0 + x_z * x_z), -fx * y_z, 0.0, 0.0))
            _add_row(acc, pose_i, w_v, err_v, SIMD[DType.float32, 8](0.0, fy * z_inv, -fy * y_z * z_inv, -fy * (1.0 + y_z * y_z), fy * x_z * y_z, fy * x_z, 0.0, 0.0))
            _add_row(acc, pose_i, w_z, err_z, SIMD[DType.float32, 8](0.0, 0.0, z_inv, y_z, -x_z, 0.0, 1.0, 0.0))
        else:
            var norm_i = sqrt(xi.squared_norm())
            var ri = xi * (1.0 / norm_i)
            var norm2_j = xj.squared_norm()
            var norm_j = sqrt(norm2_j)
            var norm_j_inv: Float32 = 1.0 / norm_j
            var rj = xj * norm_j_inv
            var err = rj - ri
            var err_d = norm_j - norm_i
            var sw_ray = inv_a * sqrt_q
            var sw_dist = inv_b * sqrt_q
            # d(p / |p|) / dp = (I - r rᵀ) / |p|
            var n3_inv = norm_j_inv / norm2_j
            var dxx = norm_j_inv - xj.x * xj.x * n3_inv
            var dyy = norm_j_inv - xj.y * xj.y * n3_inv
            var dzz = norm_j_inv - xj.z * xj.z * n3_inv
            var dxy = -xj.x * xj.y * n3_inv
            var dxz = -xj.x * xj.z * n3_inv
            var dyz = -xj.y * xj.z * n3_inv
            _add_row(acc, pose_i, _huber(sw_ray * err.x) * sw_ray * sw_ray, err.x, SIMD[DType.float32, 8](dxx, dxy, dxz, 0.0, rj.z, -rj.y, 0.0, 0.0))
            _add_row(acc, pose_i, _huber(sw_ray * err.y) * sw_ray * sw_ray, err.y, SIMD[DType.float32, 8](dxy, dyy, dyz, -rj.z, 0.0, rj.x, 0.0, 0.0))
            _add_row(acc, pose_i, _huber(sw_ray * err.z) * sw_ray * sw_ray, err.z, SIMD[DType.float32, 8](dxz, dyz, dzz, rj.y, -rj.x, 0.0, 0.0, 0.0))
            _add_row(acc, pose_i, _huber(sw_dist * err_d) * sw_dist * sw_dist, err_d, SIMD[DType.float32, 8](rj.x, rj.y, rj.z, 0.0, 0.0, 0.0, norm_j, 0.0))

    var partial = stack_allocation[DType.float32, address_space=.SHARED](row_major[WARPS, SUMS]())
    comptime for s in range(SUMS):
        var r = warp.sum(acc[s])
        if lane_id() == 0:
            partial[t // WARP_SIZE, s] = r
    barrier()
    if t < SUMS:
        var total: Float32 = 0.0
        comptime for w in range(WARPS):
            total += partial[w, t][0]
        # CUDA's atomicAdd: relaxed, device scope (Mojo's default is sequentially consistent at system scope).
        _ = Atomic[Float32, scope="device"].fetch_add[ordering=Ordering.RELAXED](sums.unsafe_offset(e * SUMS + t), total)
