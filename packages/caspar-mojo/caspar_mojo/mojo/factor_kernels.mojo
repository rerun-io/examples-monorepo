"""Per-observation kernels of the reprojection factor, generic over the camera model and the refined groups.

One thread per observation. The factor joins up to four nodes: the frame's `rig_from_world` (pose), the
camera's `sensor_from_rig` (rig), the camera's focal-and-extra (calib) and the point. A node with tangent
offset -1 is fixed: its blocks are skipped at run time. `REFINE_RIG` / `REFINE_INTRINSICS` remove the rig and
calib Jacobians at compile time. Jacobians are cached per observation, column-major over observations
(`jac[column, observation]`), so the matrix-free products need no camera model.

Scatter-adds into the gradient, the block diagonal and the products use relaxed device-scope float atomics.
"""

from std.atomic import Atomic, Ordering
from max.gpu import global_idx, lane_id
from max.gpu.primitives import warp

from layout import ImmTileTensor, MutTileTensor, row_major

from camera_models import CameraModel, Params, Projection
from geometry import Mat3, Pose, Vec3, apply, cross, rotation, translation

comptime L1 = type_of(row_major((0,)))
comptime L2 = type_of(row_major((0, 0)))
comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime F64Ptr = MutPointer[Float64, MutUntrackedOrigin]
comptime F32In = ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin]
comptime I32In = ImmTileTensor[DType.int32, L1, ImmUntrackedOrigin]
comptime VecIn = ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin]
comptime Jac = MutTileTensor[DType.float32, L2, MutUntrackedOrigin]
comptime JacIn = ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin]

# Jacobian cache columns: pose 2x6, point 2x3, then rig 2x6 and calib 2xK when refined.
comptime J_POSE = 0
comptime J_POINT = 12
comptime J_RIG = 18


def jac_width[K: Int, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool]() -> Int:
    return 18 + (12 if REFINE_RIG else 0) + (2 * K if REFINE_INTRINSICS else 0)


def calib_column[REFINE_RIG: Bool]() -> Int:
    return 30 if REFINE_RIG else 18


def tri(d: Int) -> Int:
    """Packed upper-triangle size of a d x d block."""
    return d * (d + 1) // 2


def tri_index[D: Int](i: Int, j: Int) -> Int:
    return i * D - i * (i - 1) // 2 + (j - i)


@always_inline
def atomic_add(p: F32Ptr, offset: Int, value: Float32):
    # CUDA's atomicAdd (relaxed, device scope); Mojo's default ordering adds a fence per add, about 3x slower.
    _ = Atomic[Float32, scope="device"].fetch_add[ordering=Ordering.relaxed](p.unsafe_offset(offset), value)


@always_inline
def warp_uniform(key: Int) -> Bool:
    """Whether every lane of the warp holds the same key. Observations come sorted by camera then image, so a
    warp usually shares its pose (and its camera); one atomic per warp then replaces 32 on one address."""
    var k = Int32(key)
    return warp.min(k) == warp.max(k)


@always_inline
def scatter_add(dst: F32Ptr, offset: Int, value: Float32, uniform: Bool, active: Bool):
    """dst[offset] += value; `uniform` (warp-uniform) sums the warp first. Every lane must call it."""
    if uniform:
        var total = warp.sum(value)
        if active and lane_id() == 0:
            atomic_add(dst, offset, total)
    elif active:
        atomic_add(dst, offset, value)


@always_inline
def reduce_add(p: F64Ptr, slot: Int, value: Float32):
    """Sum over the warp, then one float64 atomic per warp. Every lane must call it.

    Float64 slots keep the cost free of the accumulation-order noise that would otherwise decide the
    accept / reject of the near-converged trials (their cost changes are ~1e-6 relative).
    """
    var total = warp.sum(value)
    if lane_id() == 0:
        _ = Atomic[Float64, scope="device"].fetch_add[ordering=Ordering.relaxed](p.unsafe_offset(slot), Float64(total))


@always_inline
def load_pose(t: F32In, i: Int) -> Pose:
    comptime assert t.flat_rank == 2
    return Pose(t[i, 0][0], t[i, 1][0], t[i, 2][0], t[i, 3][0], t[i, 4][0], t[i, 5][0], t[i, 6][0])


@always_inline
def load_params(t: F32In, i: Int) -> Params:
    comptime assert t.flat_rank == 2
    var p = Params(fill=0.0)
    comptime for k in range(12):
        p[k] = t[i, k][0]
    return p^


@fieldwise_init
struct Observed[K: Int](Movable):
    """One observation projected at the current state, with what its Jacobians need."""

    var e0: Float32
    var e1: Float32
    var world: Vec3
    var rig_point: Vec3
    var r_rig: Mat3
    var r_sensor: Mat3
    var proj: Projection[Self.K]


@always_inline
def observe[Model: CameraModel](
    n: Int, poses: F32In, rigs: F32In, intrinsics: F32In, points: F32In, frame: Int, camera: Int, point: Int, pixels: F32In
) -> Observed[Model.NUM_CALIB]:
    comptime assert points.flat_rank == 2 and pixels.flat_rank == 2
    var pose = load_pose(poses, frame)
    var rig = load_pose(rigs, camera)
    var world = Vec3(points[point, 0][0], points[point, 1][0], points[point, 2][0])
    var r_rig = rotation(pose)
    var r_sensor = rotation(rig)
    var rig_point = apply(r_rig, translation(pose), world)
    var cam = apply(r_sensor, translation(rig), rig_point)
    var proj = Model.project(cam.x, cam.y, cam.z, load_params(intrinsics, camera))
    var e0 = proj.u - pixels[n, 0][0]
    var e1 = proj.v - pixels[n, 1][0]
    return Observed[Model.NUM_CALIB](e0, e1, world, rig_point, r_rig, r_sensor, proj^)


@always_inline
def accumulate_block[D: Int](
    j: Array[Float32, 2 * D], jac: Jac, n: Int, valid: Bool, col: Int, e0: Float32, e1: Float32, v: Int, h: Int, g: F32Ptr, hblk: F32Ptr
):
    """Cache J in jac[col:col+2D, n]; for a free node (v >= 0), g[v:v+D] -= J^T e and J^T J into hblk[h:].
    Every lane must call it; `valid` is False past the last observation."""
    comptime assert jac.flat_rank == 2
    if valid:
        comptime for i in range(2 * D):
            jac[col + i, n] = j[i]
    var key = v if valid else -1
    var uniform = warp_uniform(key)
    var active = key >= 0
    comptime for i in range(D):
        scatter_add(g, v + i, -(j[i] * e0 + j[D + i] * e1), uniform, active)
        comptime for k in range(i, D):
            scatter_add(hblk, h + tri_index[D](i, k), j[i] * j[k] + j[D + i] * j[D + k], uniform, active)


@always_inline
def h_offset[K: Int](v: Int, n6: Int, n_calib: Int) -> Int:
    """Packed block-diagonal offset of the block whose tangent slice starts at `v` (same block order)."""
    var calib_v = 6 * n6
    var point_v = calib_v + K * n_calib
    if v < calib_v:
        return tri(6) * (v // 6)
    if v < point_v:
        return tri(6) * n6 + tri(K) * ((v - calib_v) // K)
    return tri(6) * n6 + tri(K) * n_calib + tri(3) * ((v - point_v) // 3)


def linearize_kernel[Model: CameraModel, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool](
    poses: F32In,
    rigs: F32In,
    intrinsics: F32In,
    points: F32In,
    pose_off: I32In,
    rig_off: I32In,
    calib_off: I32In,
    point_off: I32In,
    obs_frame: I32In,
    obs_camera: I32In,
    obs_point: I32In,
    pixels: F32In,
    jac: Jac,
    g: F32Ptr,
    hblk: F32Ptr,
    scalars: F64Ptr,
    cost_slot: Int32,
    n6: Int32,
    n_calib: Int32,
):
    """Residual, Jacobians (cached), -J^T e and the block diagonal of J^T J, and the cost 1/2 sum |e|^2."""
    comptime assert obs_frame.flat_rank == 1 and obs_camera.flat_rank == 1 and obs_point.flat_rank == 1
    comptime assert pose_off.flat_rank == 1 and rig_off.flat_rank == 1 and calib_off.flat_rank == 1
    comptime assert point_off.flat_rank == 1 and jac.flat_rank == 2
    comptime K = Model.NUM_CALIB
    # Lanes past the end recompute the last observation and add nothing, so every lane reaches the warp sums.
    var n_obs = Int(obs_frame.dim[0]())
    var valid = global_idx.x < n_obs
    var n = min(global_idx.x, n_obs - 1)
    var frame = Int(obs_frame[n][0])
    var camera = Int(obs_camera[n][0])
    var point = Int(obs_point[n][0])
    var o = observe[Model](n, poses, rigs, intrinsics, points, frame, camera, point, pixels)
    var cost: Float32 = 0.5 * (o.e0 * o.e0 + o.e1 * o.e1) if valid else 0.0
    var j_pose = Array[Float32, 12](fill=0.0)
    var j_point = Array[Float32, 6](fill=0.0)
    var j_rig = Array[Float32, 12](fill=0.0)
    comptime for r in range(2):
        var dp = Vec3(o.proj.d_point[3 * r], o.proj.d_point[3 * r + 1], o.proj.d_point[3 * r + 2])
        # a = d pixel / d camera point * R_sensor (w.r.t. the rig-frame point), b = a * R_rig (w.r.t. the world point).
        var a = Vec3(
            dp.x * o.r_sensor[0, 0] + dp.y * o.r_sensor[1, 0] + dp.z * o.r_sensor[2, 0],
            dp.x * o.r_sensor[0, 1] + dp.y * o.r_sensor[1, 1] + dp.z * o.r_sensor[2, 1],
            dp.x * o.r_sensor[0, 2] + dp.y * o.r_sensor[1, 2] + dp.z * o.r_sensor[2, 2],
        )
        var b = Vec3(
            a.x * o.r_rig[0, 0] + a.y * o.r_rig[1, 0] + a.z * o.r_rig[2, 0],
            a.x * o.r_rig[0, 1] + a.y * o.r_rig[1, 1] + a.z * o.r_rig[2, 1],
            a.x * o.r_rig[0, 2] + a.y * o.r_rig[1, 2] + a.z * o.r_rig[2, 2],
        )
        # Right perturbation R exp(w): d/dw = -J R [X]x, whose row is X x (J R); likewise P_rig x a for the rig.
        var rot = cross(o.world, b)
        j_pose[6 * r + 0] = rot.x
        j_pose[6 * r + 1] = rot.y
        j_pose[6 * r + 2] = rot.z
        j_pose[6 * r + 3] = a.x
        j_pose[6 * r + 4] = a.y
        j_pose[6 * r + 5] = a.z
        j_point[3 * r + 0] = b.x
        j_point[3 * r + 1] = b.y
        j_point[3 * r + 2] = b.z
        var rig_rot = cross(o.rig_point, a)
        j_rig[6 * r + 0] = rig_rot.x
        j_rig[6 * r + 1] = rig_rot.y
        j_rig[6 * r + 2] = rig_rot.z
        j_rig[6 * r + 3] = dp.x
        j_rig[6 * r + 4] = dp.y
        j_rig[6 * r + 5] = dp.z
    var nn6 = Int(n6)
    var nc = Int(n_calib)
    var po = Int(pose_off[frame][0])
    accumulate_block[6](j_pose, jac, n, valid, J_POSE, o.e0, o.e1, po, h_offset[K](po, nn6, nc), g, hblk)
    var pto = Int(point_off[point][0])
    accumulate_block[3](j_point, jac, n, valid, J_POINT, o.e0, o.e1, pto, h_offset[K](pto, nn6, nc), g, hblk)
    comptime if REFINE_RIG:
        var ro = Int(rig_off[camera][0])
        accumulate_block[6](j_rig, jac, n, valid, J_RIG, o.e0, o.e1, ro, h_offset[K](ro, nn6, nc), g, hblk)
    comptime if REFINE_INTRINSICS:
        var co = Int(calib_off[camera][0])
        accumulate_block[K](o.proj.d_calib, jac, n, valid, calib_column[REFINE_RIG](), o.e0, o.e1, co, h_offset[K](co, nn6, nc), g, hblk)
    reduce_add(scalars, Int(cost_slot), cost)


def score_kernel[Model: CameraModel](
    poses: F32In,
    rigs: F32In,
    intrinsics: F32In,
    points: F32In,
    obs_frame: I32In,
    obs_camera: I32In,
    obs_point: I32In,
    pixels: F32In,
    scalars: F64Ptr,
    cost_slot: Int32,
):
    """The cost 1/2 sum |e|^2 at a (trial) state."""
    comptime assert obs_frame.flat_rank == 1 and obs_camera.flat_rank == 1 and obs_point.flat_rank == 1
    var n = global_idx.x
    var cost: Float32 = 0.0
    if n < Int(obs_frame.dim[0]()):
        var o = observe[Model](
            n, poses, rigs, intrinsics, points, Int(obs_frame[n][0]), Int(obs_camera[n][0]), Int(obs_point[n][0]), pixels
        )
        cost = 0.5 * (o.e0 * o.e0 + o.e1 * o.e1)
    reduce_add(scalars, Int(cost_slot), cost)


@always_inline
def block_times[D: Int](jac: JacIn, n: Int, col: Int, x: VecIn, v: Int) -> Tuple[Float32, Float32]:
    """J x[v:v+D] for one 2xD block."""
    comptime assert jac.flat_rank == 2 and x.flat_rank == 1
    var u0: Float32 = 0.0
    var u1: Float32 = 0.0
    comptime for i in range(D):
        var xi = x[v + i][0]
        u0 += jac[col + i, n][0] * xi
        u1 += jac[col + D + i, n][0] * xi
    return (u0, u1)


@always_inline
def add_block_transpose[D: Int](jac: JacIn, n: Int, col: Int, u0: Float32, u1: Float32, dst: F32Ptr, key: Int):
    """dst[key:key+D] += J^T u for one 2xD block; key -1 adds nothing. Every lane must call it."""
    comptime assert jac.flat_rank == 2
    var uniform = warp_uniform(key)
    comptime for i in range(D):
        scatter_add(dst, key + i, jac[col + i, n][0] * u0 + jac[col + D + i, n][0] * u1, uniform, key >= 0)


def cross_cameras_kernel[K: Int, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool](
    jac: JacIn,
    pose_off: I32In,
    rig_off: I32In,
    calib_off: I32In,
    point_off: I32In,
    obs_frame: I32In,
    obs_camera: I32In,
    obs_point: I32In,
    x: VecIn,
    dst: F32Ptr,
):
    """First Schur pass, x zero on the points: dst_cam += cross camera-camera terms, dst_point += H_pc x_cam."""
    comptime assert obs_frame.flat_rank == 1 and obs_camera.flat_rank == 1 and obs_point.flat_rank == 1
    comptime assert pose_off.flat_rank == 1 and rig_off.flat_rank == 1 and calib_off.flat_rank == 1
    comptime assert point_off.flat_rank == 1
    var n_obs = Int(obs_frame.dim[0]())
    var valid = global_idx.x < n_obs
    var n = min(global_idx.x, n_obs - 1)
    var camera = Int(obs_camera[n][0])
    var po = Int(pose_off[Int(obs_frame[n][0])][0]) if valid else -1
    var ro = -1
    var co = -1
    comptime if REFINE_RIG:
        ro = Int(rig_off[camera][0]) if valid else -1
    comptime if REFINE_INTRINSICS:
        co = Int(calib_off[camera][0]) if valid else -1
    var up = (Float32(0.0), Float32(0.0))
    var ur = (Float32(0.0), Float32(0.0))
    var uc = (Float32(0.0), Float32(0.0))
    if po >= 0:
        up = block_times[6](jac, n, J_POSE, x, po)
    if ro >= 0:
        ur = block_times[6](jac, n, J_RIG, x, ro)
    if co >= 0:
        uc = block_times[K](jac, n, calib_column[REFINE_RIG](), x, co)
    var u0 = up[0] + ur[0] + uc[0]
    var u1 = up[1] + ur[1] + uc[1]
    comptime if REFINE_RIG or REFINE_INTRINSICS:
        add_block_transpose[6](jac, n, J_POSE, u0 - up[0], u1 - up[1], dst, po)
        comptime if REFINE_RIG:
            add_block_transpose[6](jac, n, J_RIG, u0 - ur[0], u1 - ur[1], dst, ro)
        comptime if REFINE_INTRINSICS:
            add_block_transpose[K](jac, n, calib_column[REFINE_RIG](), u0 - uc[0], u1 - uc[1], dst, co)
    var pto = Int(point_off[Int(obs_point[n][0])][0]) if valid else -1
    add_block_transpose[3](jac, n, J_POINT, u0, u1, dst, pto)


def cross_points_kernel[K: Int, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool](
    jac: JacIn,
    pose_off: I32In,
    rig_off: I32In,
    calib_off: I32In,
    point_off: I32In,
    obs_frame: I32In,
    obs_camera: I32In,
    obs_point: I32In,
    x: VecIn,
    dst: F32Ptr,
):
    """Second Schur pass: dst_cam += H_cp x_point."""
    comptime assert obs_frame.flat_rank == 1 and obs_camera.flat_rank == 1 and obs_point.flat_rank == 1
    comptime assert pose_off.flat_rank == 1 and rig_off.flat_rank == 1 and calib_off.flat_rank == 1
    comptime assert point_off.flat_rank == 1
    var n_obs = Int(obs_frame.dim[0]())
    var valid = global_idx.x < n_obs
    var n = min(global_idx.x, n_obs - 1)
    var pto = Int(point_off[Int(obs_point[n][0])][0]) if valid else -1
    var u = (Float32(0.0), Float32(0.0))
    if pto >= 0:
        u = block_times[3](jac, n, J_POINT, x, pto)
    # A fixed point adds nothing: its lanes join the warp sums with key -1.
    var po = Int(pose_off[Int(obs_frame[n][0])][0]) if pto >= 0 else -1
    add_block_transpose[6](jac, n, J_POSE, u[0], u[1], dst, po)
    var camera = Int(obs_camera[n][0])
    comptime if REFINE_RIG:
        var ro = Int(rig_off[camera][0]) if pto >= 0 else -1
        add_block_transpose[6](jac, n, J_RIG, u[0], u[1], dst, ro)
    comptime if REFINE_INTRINSICS:
        var co = Int(calib_off[camera][0]) if pto >= 0 else -1
        add_block_transpose[K](jac, n, calib_column[REFINE_RIG](), u[0], u[1], dst, co)
