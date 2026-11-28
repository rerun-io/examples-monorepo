"""Per-observation kernels of the reprojection factor, generic over the camera model and the refined groups.

The factor joins up to four nodes: the frame's `rig_from_world` (pose), the
camera's `sensor_from_rig` (rig), the camera's focal-and-extra (calib) and the point. A node with tangent
offset -1 is fixed: its blocks are skipped at run time. `REFINE_RIG` / `REFINE_INTRINSICS` remove the rig and
calib Jacobians at compile time. The matrix-free products recompute the Jacobians from the linearisation state
instead of reading a cache: a cache streams 18-50 floats per observation per pass, the state is a few hundred KB
that stays in L2, and the arithmetic is far cheaper than the memory traffic on every GPU.

Sums into camera-side nodes walk warp tiles (up to WARP_SIZE observations of ONE image each, in camera-then-frame
order, WARPS per block tile), so every warp shares its nodes whatever the image sizes; they reach memory through
`Running` and relaxed device-scope float atomics. Sums into points run POINT_LANES lanes per point
over a point-sorted copy of the observations and need no atomics.
"""

from std.atomic import Atomic, Ordering
from std.bit import log2_floor
from std.math import ceildiv, sqrt
from max.gpu import WARP_SIZE, block_idx, global_idx, lane_id, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation
from max.gpu.primitives import warp

from layout import ImmTileTensor, MutTileTensor, row_major

from camera_models import CameraModel, Params
from geometry import Pose, Vec3, apply, cross, rotation, translation

comptime L1 = type_of(row_major((0,)))
comptime L2 = type_of(row_major((0, 0)))
comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime F64Ptr = MutPointer[Float64, MutUntrackedOrigin]
comptime F32In = ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin]
comptime Vec = MutTileTensor[DType.float32, L1, MutUntrackedOrigin]
comptime I32In = ImmTileTensor[DType.int32, L1, ImmUntrackedOrigin]
comptime TilesIn = ImmTileTensor[DType.int32, L2, ImmUntrackedOrigin]
comptime I32Out = MutTileTensor[DType.int32, L1, MutUntrackedOrigin]
comptime F32Out = MutTileTensor[DType.float32, L2, MutUntrackedOrigin]
comptime VecIn = ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin]

def tri(d: Int) -> Int:
    """Packed upper-triangle size of a d x d block."""
    return d * (d + 1) // 2


def tri_index[D: Int](i: Int, j: Int) -> Int:
    return i * D - i * (i - 1) // 2 + (j - i)


@always_inline
def damped_solve[D: Int](hblk: VecIn, h: Int, rhs: Array[Float32, D], lam: Float32) -> Array[Float32, D]:
    comptime assert hblk.flat_rank == 1
    var a = Array[Float32, D * D](fill=0.0)
    comptime for i in range(D):
        comptime for j in range(i, D):
            var hij = hblk[h + tri_index[D](i, j)][0]
            a[i * D + j] = hij
            a[j * D + i] = hij
        a[i * D + i] = a[i * D + i] * (1.0 + lam) + lam * 1e-6
    # In-place Cholesky, lower triangle.
    comptime for j in range(D):
        var d = a[j * D + j]
        comptime for k in range(j):
            d -= a[j * D + k] * a[j * D + k]
        var l = sqrt(d)
        a[j * D + j] = l
        comptime for i in range(j + 1, D):
            var s = a[i * D + j]
            comptime for k in range(j):
                s -= a[i * D + k] * a[j * D + k]
            a[i * D + j] = s / l
    var y = Array[Float32, D](fill=0.0)
    comptime for i in range(D):
        var s = rhs[i]
        comptime for k in range(i):
            s -= a[i * D + k] * y[k]
        y[i] = s / a[i * D + i]
    comptime for ii in range(D):
        comptime i = D - 1 - ii
        var s = y[i]
        comptime for k in range(i + 1, D):
            s -= a[k * D + i] * y[k]
        y[i] = s / a[i * D + i]
    return y^


@always_inline
def atomic_add(p: F32Ptr, offset: Int, value: Float32):
    # CUDA's atomicAdd (relaxed, device scope); Mojo's default ordering adds a fence per add, about 3x slower.
    _ = Atomic[Float32, scope="device"].fetch_add[ordering=Ordering.relaxed](p.unsafe_offset(offset), value)


comptime BLOCK = 256
"""Threads per block of every per-observation kernel (the block reductions below assume it)."""
comptime WARPS = BLOCK // WARP_SIZE


@fieldwise_init
struct Running(TrivialRegisterPassable):
    """A block's running sum for one scatter site, carried over its tiles: thread t < D holds component t of the
    node block at `key`. Observations come sorted by camera then frame, so consecutive warps of a block mostly share
    the node: the sum reaches memory once per key change, and the same-address atomics (which serialise across
    blocks; one camera's intrinsics are shared by every observation) fall by the warps per block."""

    var key: Int
    var value: Float32

    @staticmethod
    def empty() -> Self:
        return Self(-1, 0.0)

    @always_inline
    def flush[D: Int](mut self, dst: F32Ptr):
        if thread_idx.x < D and self.key >= 0:
            atomic_add(dst, self.key + thread_idx.x, self.value)
        self.key = -1
        self.value = 0.0


@always_inline
def scatter_values[D: Int](dst: F32Ptr, key: Int, values: Array[Float32, D], mut running: Running):
    """dst[key:key+D] += values: summed over the warp, whose lanes share `key` (warp tiles hold one image), then the
    block's warps merged in order into `running`; key -1 adds nothing. Every thread must call it."""
    comptime assert D <= BLOCK, "one thread per component"
    var sums = stack_allocation[WARPS * D, Float32, address_space=AddressSpace.SHARED]()
    var keys = stack_allocation[WARPS, Int32, address_space=AddressSpace.SHARED]()
    var warp_index = thread_idx.x // WARP_SIZE
    comptime for i in range(D):
        var total = warp.sum(values[i])
        if lane_id() == 0:
            sums[unsafe_offset=warp_index * D + i] = total
    if lane_id() == 0:
        keys[unsafe_offset=warp_index] = Int32(key)
    barrier()
    var t = thread_idx.x
    if t < D:
        comptime for w in range(WARPS):
            var k = Int(keys[unsafe_offset=w])
            if k != running.key:
                running.flush[D](dst)
                running.key = k
            running.value += sums[unsafe_offset=w * D + t]
    barrier()


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
struct Linearized[K: Int](Movable):
    """One observation at a state: its projection and the 2xD Jacobian of each node block, row-major."""

    var u: Float32
    var v: Float32
    var pose: Array[Float32, 12]
    var point: Array[Float32, 6]
    var rig: Array[Float32, 12]
    var calib: Array[Float32, 2 * Self.K]


@always_inline
def project_observation[Model: CameraModel](
    poses: F32In, rigs: F32In, intrinsics: F32In, points: F32In, frame: Int, camera: Int, point: Int
) -> Linearized[Model.NUM_CALIB]:
    comptime assert points.flat_rank == 2
    var pose = load_pose(poses, frame)
    var rig = load_pose(rigs, camera)
    var world = Vec3(points[point, 0][0], points[point, 1][0], points[point, 2][0])
    var r_rig = rotation(pose)
    var r_sensor = rotation(rig)
    var rig_point = apply(r_rig, translation(pose), world)
    var cam = apply(r_sensor, translation(rig), rig_point)
    var proj = Model.project(cam.x, cam.y, cam.z, load_params(intrinsics, camera))
    var j_pose = Array[Float32, 12](fill=0.0)
    var j_point = Array[Float32, 6](fill=0.0)
    var j_rig = Array[Float32, 12](fill=0.0)
    comptime for r in range(2):
        var dp = Vec3(proj.d_point[3 * r], proj.d_point[3 * r + 1], proj.d_point[3 * r + 2])
        # a = d pixel / d camera point * R_sensor (w.r.t. the rig-frame point), b = a * R_rig (w.r.t. the world point).
        var a = Vec3(
            dp.x * r_sensor[0, 0] + dp.y * r_sensor[1, 0] + dp.z * r_sensor[2, 0],
            dp.x * r_sensor[0, 1] + dp.y * r_sensor[1, 1] + dp.z * r_sensor[2, 1],
            dp.x * r_sensor[0, 2] + dp.y * r_sensor[1, 2] + dp.z * r_sensor[2, 2],
        )
        var b = Vec3(
            a.x * r_rig[0, 0] + a.y * r_rig[1, 0] + a.z * r_rig[2, 0],
            a.x * r_rig[0, 1] + a.y * r_rig[1, 1] + a.z * r_rig[2, 1],
            a.x * r_rig[0, 2] + a.y * r_rig[1, 2] + a.z * r_rig[2, 2],
        )
        # Right perturbation R exp(w): d/dw = -J R [X]x, whose row is X x (J R); likewise P_rig x a for the rig.
        var rot = cross(world, b)
        j_pose[6 * r + 0] = rot.x
        j_pose[6 * r + 1] = rot.y
        j_pose[6 * r + 2] = rot.z
        j_pose[6 * r + 3] = a.x
        j_pose[6 * r + 4] = a.y
        j_pose[6 * r + 5] = a.z
        j_point[3 * r + 0] = b.x
        j_point[3 * r + 1] = b.y
        j_point[3 * r + 2] = b.z
        var rig_rot = cross(rig_point, a)
        j_rig[6 * r + 0] = rig_rot.x
        j_rig[6 * r + 1] = rig_rot.y
        j_rig[6 * r + 2] = rig_rot.z
        j_rig[6 * r + 3] = dp.x
        j_rig[6 * r + 4] = dp.y
        j_rig[6 * r + 5] = dp.z
    return Linearized[Model.NUM_CALIB](proj.u, proj.v, j_pose^, j_point^, j_rig^, proj.d_calib.copy())


@always_inline
def accumulate_block[D: Int](
    j: Array[Float32, 2 * D],
    valid: Bool,
    e0: Float32,
    e1: Float32,
    v: Int,
    h: Int,
    g: F32Ptr,
    hblk: F32Ptr,
    mut run_g: Running,
    mut run_h: Running,
):
    """For a free node (v >= 0, warp-uniform), g[v:v+D] -= J^T e and J^T J into hblk[h:]. A lane without an
    observation (`valid` False) adds zeros. Every lane must call it."""
    var grad = Array[Float32, D](fill=0.0)
    var hess = Array[Float32, tri(D)](fill=0.0)
    if valid:
        comptime for i in range(D):
            grad[i] = -(j[i] * e0 + j[D + i] * e1)
            comptime for k in range(i, D):
                hess[tri_index[D](i, k)] = j[i] * j[k] + j[D + i] * j[D + k]
    scatter_values[D](g, v, grad, run_g)
    scatter_values[tri(D)](hblk, h if v >= 0 else -1, hess, run_h)


@fieldwise_init
struct WarpSlot(TrivialRegisterPassable):
    """This lane's observation in a warp tile: `n` is always a real row (idle lanes repeat the tile's last one, so
    every lane computes and reaches the warp sums), `valid` says the lane adds it, `live` that the warp holds a tile."""

    var n: Int
    var valid: Bool
    var live: Bool


@always_inline
def warp_slot(warp_tiles: TilesIn, tile: Int) -> WarpSlot:
    """Block tile `tile`'s warp tile for this warp: rows (first observation, count <= WARP_SIZE) of one image each."""
    comptime assert warp_tiles.flat_rank == 2
    var n_tiles = Int(warp_tiles.dim[0]())
    var row = tile * WARPS + thread_idx.x // WARP_SIZE
    var live = row < n_tiles
    row = min(row, n_tiles - 1)
    var count = Int(warp_tiles[row, 1][0])
    return WarpSlot(Int(warp_tiles[row, 0][0]) + min(lane_id(), count - 1), live and lane_id() < count, live)


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
    warp_tiles: TilesIn,
    g: F32Ptr,
    hblk: F32Ptr,
    scalars: F64Ptr,
    cost_slot: Int32,
    n6: Int32,
    n_calib: Int32,
    tiles: Int32,
):
    """Residual, the camera blocks of -J^T e and of the block diagonal of J^T J, and the cost 1/2 sum |e|^2, over
    `tiles` consecutive block tiles (WARPS warp tiles) per block. The point blocks come from `point_linearize_kernel`."""
    comptime assert obs_frame.flat_rank == 1 and obs_camera.flat_rank == 1 and obs_point.flat_rank == 1
    comptime assert pose_off.flat_rank == 1 and rig_off.flat_rank == 1 and calib_off.flat_rank == 1
    comptime assert point_off.flat_rank == 1 and pixels.flat_rank == 2
    comptime K = Model.NUM_CALIB
    var n_block_tiles = ceildiv(Int(warp_tiles.dim[0]()), WARPS)
    var nn6 = Int(n6)
    var nc = Int(n_calib)
    var first = block_idx.x * Int(tiles)
    var cost: Float32 = 0.0
    var pose_g = Running.empty()
    var pose_h = Running.empty()
    var rig_g = Running.empty()
    var rig_h = Running.empty()
    var calib_g = Running.empty()
    var calib_h = Running.empty()
    for tile in range(first, min(first + Int(tiles), n_block_tiles)):
        var slot = warp_slot(warp_tiles, tile)
        var n = slot.n
        var frame = Int(obs_frame[n][0])
        var camera = Int(obs_camera[n][0])
        var o = project_observation[Model](poses, rigs, intrinsics, points, frame, camera, Int(obs_point[n][0]))
        var e0 = o.u - pixels[n, 0][0]
        var e1 = o.v - pixels[n, 1][0]
        if slot.valid:
            cost += 0.5 * (e0 * e0 + e1 * e1)
        var po = Int(pose_off[frame][0]) if slot.live else -1
        accumulate_block[6](o.pose, slot.valid, e0, e1, po, h_offset[K](po, nn6, nc), g, hblk, pose_g, pose_h)
        comptime if REFINE_RIG:
            var ro = Int(rig_off[camera][0]) if slot.live else -1
            accumulate_block[6](o.rig, slot.valid, e0, e1, ro, h_offset[K](ro, nn6, nc), g, hblk, rig_g, rig_h)
        comptime if REFINE_INTRINSICS:
            var co = Int(calib_off[camera][0]) if slot.live else -1
            accumulate_block[K](o.calib, slot.valid, e0, e1, co, h_offset[K](co, nn6, nc), g, hblk, calib_g, calib_h)
    pose_g.flush[6](g)
    pose_h.flush[tri(6)](hblk)
    rig_g.flush[6](g)
    rig_h.flush[tri(6)](hblk)
    calib_g.flush[K](g)
    calib_h.flush[tri(K)](hblk)
    reduce_add(scalars, Int(cost_slot), cost)


comptime POINT_LANES = 4
"""Lanes that share one point in the per-point kernels. Measured on the 3060 (1, 4, 16 lanes): one lane per point
is 1.5x slower on 60 observations per point, 4 and 16 tie there and 4 is ahead on 15 per point."""


@always_inline
def lane_group_sum(value: Float32) -> Float32:
    """Sum over the POINT_LANES lanes of this lane's group (xor offsets stay inside the group). Every lane of the
    warp must call it: the shuffles span the whole warp."""
    var total = value
    comptime for i in range(log2_floor(POINT_LANES)):
        total += warp.shuffle_xor(total, UInt32(POINT_LANES >> (i + 1)))
    return total


def point_linearize_kernel[Model: CameraModel](
    poses: F32In,
    rigs: F32In,
    intrinsics: F32In,
    points: F32In,
    point_off: I32In,
    point_start: I32In,
    by_point_frame: I32In,
    by_point_camera: I32In,
    by_point_pixels: F32In,
    g: Vec,
    hblk: Vec,
    point_v: Int32,
    point_h: Int32,
):
    """The point blocks of -J^T e and of the block diagonal, POINT_LANES lanes per point over its observations (no atomics)."""
    comptime assert point_off.flat_rank == 1 and point_start.flat_rank == 1 and by_point_pixels.flat_rank == 2
    comptime assert by_point_frame.flat_rank == 1 and by_point_camera.flat_rank == 1
    comptime assert g.flat_rank == 1 and hblk.flat_rank == 1
    # No early return: a warp holds several points and the shuffles need every lane. Idle lanes loop over nothing.
    var j = global_idx.x // POINT_LANES
    var lane = global_idx.x % POINT_LANES
    var v = Int(point_off[j][0]) if j < Int(point_off.dim[0]()) else -1
    var begin = Int(point_start[j][0]) if v >= 0 else 0
    var end = Int(point_start[j + 1][0]) if v >= 0 else 0
    var grad = Array[Float32, 3](fill=0.0)
    var hess = Array[Float32, tri(3)](fill=0.0)
    for i in range(begin + lane, end, POINT_LANES):
        var o = project_observation[Model](poses, rigs, intrinsics, points, Int(by_point_frame[i][0]), Int(by_point_camera[i][0]), j)
        var e0 = o.u - by_point_pixels[i, 0][0]
        var e1 = o.v - by_point_pixels[i, 1][0]
        comptime for a in range(3):
            grad[a] -= o.point[a] * e0 + o.point[3 + a] * e1
            comptime for b in range(a, 3):
                hess[tri_index[3](a, b)] += o.point[a] * o.point[b] + o.point[3 + a] * o.point[3 + b]
    comptime for a in range(3):
        grad[a] = lane_group_sum(grad[a])
    comptime for t in range(tri(3)):
        hess[t] = lane_group_sum(hess[t])
    if lane != 0 or v < 0:
        return
    var h = Int(point_h) + tri(3) * ((v - Int(point_v)) // 3)
    comptime for a in range(3):
        g[v + a] = grad[a]
    comptime for t in range(tri(3)):
        hblk[h + t] = hess[t]


def gather_rows_kernel(
    rows: I32In,
    frame: I32In,
    camera: I32In,
    pixels: F32In,
    out_frame: I32Out,
    out_camera: I32Out,
    out_pixels: F32Out,
):
    """out[i] = column[rows[i]] for the frame, camera and pixel columns."""
    comptime assert rows.flat_rank == 1 and frame.flat_rank == 1 and camera.flat_rank == 1 and pixels.flat_rank == 2
    comptime assert out_frame.flat_rank == 1 and out_camera.flat_rank == 1 and out_pixels.flat_rank == 2
    var i = global_idx.x
    if i >= Int(rows.dim[0]()):
        return
    var n = Int(rows[i][0])
    out_frame[i] = frame[n]
    out_camera[i] = camera[n]
    out_pixels[i, 0] = pixels[n, 0]
    out_pixels[i, 1] = pixels[n, 1]


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
    comptime assert points.flat_rank == 2 and pixels.flat_rank == 2
    var n = global_idx.x
    var cost: Float32 = 0.0
    if n < Int(obs_frame.dim[0]()):
        var camera = Int(obs_camera[n][0])
        var pose = load_pose(poses, Int(obs_frame[n][0]))
        var rig = load_pose(rigs, camera)
        var point = Int(obs_point[n][0])
        var world = Vec3(points[point, 0][0], points[point, 1][0], points[point, 2][0])
        var cam = apply(rotation(rig), translation(rig), apply(rotation(pose), translation(pose), world))
        var proj = Model.project(cam.x, cam.y, cam.z, load_params(intrinsics, camera))
        var e0 = proj.u - pixels[n, 0][0]
        var e1 = proj.v - pixels[n, 1][0]
        cost = 0.5 * (e0 * e0 + e1 * e1)
    reduce_add(scalars, Int(cost_slot), cost)


@always_inline
def block_times[D: Int](j: Array[Float32, 2 * D], x: VecIn, v: Int) -> Tuple[Float32, Float32]:
    """J x[v:v+D] for one 2xD block."""
    comptime assert x.flat_rank == 1
    var u0: Float32 = 0.0
    var u1: Float32 = 0.0
    comptime for i in range(D):
        var xi = x[v + i][0]
        u0 += j[i] * xi
        u1 += j[D + i] * xi
    return (u0, u1)


@always_inline
def add_block_transpose[D: Int](j: Array[Float32, 2 * D], u0: Float32, u1: Float32, dst: F32Ptr, key: Int, mut running: Running):
    """dst[key:key+D] += J^T u for one 2xD block; key -1 adds nothing. Every lane must call it."""
    var values = Array[Float32, D](fill=0.0)
    comptime for i in range(D):
        values[i] = j[i] * u0 + j[D + i] * u1
    scatter_values[D](dst, key, values, running)


comptime GATHER_NEGATE = 0
"""dst_p = -M_pp^-1 H_pc x_cam: the point half of the Schur matvec."""
comptime GATHER_BACKSUBSTITUTE = 1
"""dst_p = M_pp^-1 (g_p - H_pc x_cam): the point step after the PCG."""


def point_gather_kernel[Model: CameraModel, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool, MODE: Int](
    poses: F32In,
    rigs: F32In,
    intrinsics: F32In,
    points: F32In,
    pose_off: I32In,
    rig_off: I32In,
    calib_off: I32In,
    point_off: I32In,
    point_start: I32In,
    by_point_frame: I32In,
    by_point_camera: I32In,
    x: VecIn,
    g: VecIn,
    dst: Vec,
    hblk: VecIn,
    point_v: Int32,
    point_h: Int32,
    lam: Float32,
):
    """POINT_LANES lanes per point gather its observations (point-sorted copies of frame and camera): H_pc x_cam
    summed over the lanes, then the damped 3x3 point solve. No atomics: the camera-sorted order scatters a point's terms."""
    comptime assert point_off.flat_rank == 1 and point_start.flat_rank == 1 and pose_off.flat_rank == 1
    comptime assert by_point_frame.flat_rank == 1 and by_point_camera.flat_rank == 1
    comptime assert rig_off.flat_rank == 1 and calib_off.flat_rank == 1
    comptime assert g.flat_rank == 1 and dst.flat_rank == 1
    comptime K = Model.NUM_CALIB
    # No early return: a warp holds several points and the shuffles need every lane. Idle lanes loop over nothing.
    var j = global_idx.x // POINT_LANES
    var lane = global_idx.x % POINT_LANES
    var v = Int(point_off[j][0]) if j < Int(point_off.dim[0]()) else -1
    var begin = Int(point_start[j][0]) if v >= 0 else 0
    var end = Int(point_start[j + 1][0]) if v >= 0 else 0
    var w = Array[Float32, 3](fill=0.0)
    for i in range(begin + lane, end, POINT_LANES):
        var frame = Int(by_point_frame[i][0])
        var camera = Int(by_point_camera[i][0])
        var o = project_observation[Model](poses, rigs, intrinsics, points, frame, camera, j)
        var u0: Float32 = 0.0
        var u1: Float32 = 0.0
        var po = Int(pose_off[frame][0])
        if po >= 0:
            var up = block_times[6](o.pose, x, po)
            u0 += up[0]
            u1 += up[1]
        comptime if REFINE_RIG:
            var ro = Int(rig_off[camera][0])
            if ro >= 0:
                var ur = block_times[6](o.rig, x, ro)
                u0 += ur[0]
                u1 += ur[1]
        comptime if REFINE_INTRINSICS:
            var co = Int(calib_off[camera][0])
            if co >= 0:
                var uc = block_times[K](o.calib, x, co)
                u0 += uc[0]
                u1 += uc[1]
        comptime for k in range(3):
            w[k] += o.point[k] * u0 + o.point[3 + k] * u1
    comptime for k in range(3):
        w[k] = lane_group_sum(w[k])
    if lane != 0 or v < 0:
        return
    var rhs = Array[Float32, 3](fill=0.0)
    comptime for k in range(3):
        comptime if MODE == GATHER_NEGATE:
            rhs[k] = -w[k]
        else:
            rhs[k] = g[v + k][0] - w[k]
    var sol = damped_solve[3](hblk, Int(point_h) + tri(3) * ((v - Int(point_v)) // 3), rhs, lam)
    comptime for k in range(3):
        dst[v + k] = sol[k]


def cross_points_kernel[Model: CameraModel, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool, CAMERA_TERMS: Bool](
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
    warp_tiles: TilesIn,
    x_cam: VecIn,
    x_point: VecIn,
    dst: F32Ptr,
    tiles: Int32,
):
    """Camera side of the Schur products at the linearisation state: dst_cam += H_cp x_point and, with
    `CAMERA_TERMS`, the cross camera-camera terms of H_cc x_cam (pose-rig-calib of one observation; the in-block
    terms come from M p)."""
    comptime assert obs_frame.flat_rank == 1 and obs_camera.flat_rank == 1 and obs_point.flat_rank == 1
    comptime assert pose_off.flat_rank == 1 and rig_off.flat_rank == 1 and calib_off.flat_rank == 1
    comptime assert point_off.flat_rank == 1
    comptime K = Model.NUM_CALIB
    comptime CROSS = CAMERA_TERMS and (REFINE_RIG or REFINE_INTRINSICS)
    var n_block_tiles = ceildiv(Int(warp_tiles.dim[0]()), WARPS)
    var first = block_idx.x * Int(tiles)
    var pose_sum = Running.empty()
    var rig_sum = Running.empty()
    var calib_sum = Running.empty()
    for tile in range(first, min(first + Int(tiles), n_block_tiles)):
        var slot = warp_slot(warp_tiles, tile)
        var n = slot.n
        var frame = Int(obs_frame[n][0])
        var camera = Int(obs_camera[n][0])
        var point = Int(obs_point[n][0])
        var o = project_observation[Model](poses, rigs, intrinsics, points, frame, camera, point)
        var pto = Int(point_off[point][0]) if slot.valid else -1
        var u = (Float32(0.0), Float32(0.0))
        if pto >= 0:
            u = block_times[3](o.point, x_point, pto)
        var po = Int(pose_off[frame][0]) if slot.live else -1
        var ro = -1
        var co = -1
        comptime if REFINE_RIG:
            ro = Int(rig_off[camera][0]) if slot.live else -1
        comptime if REFINE_INTRINSICS:
            co = Int(calib_off[camera][0]) if slot.live else -1
        var up = (Float32(0.0), Float32(0.0))
        var ur = (Float32(0.0), Float32(0.0))
        var uc = (Float32(0.0), Float32(0.0))
        comptime if CROSS:
            if slot.valid:
                if po >= 0:
                    up = block_times[6](o.pose, x_cam, po)
                if ro >= 0:
                    ur = block_times[6](o.rig, x_cam, ro)
                if co >= 0:
                    uc = block_times[K](o.calib, x_cam, co)
        var u0 = u[0] + up[0] + ur[0] + uc[0]
        var u1 = u[1] + up[1] + ur[1] + uc[1]
        # The sums need no masking: an idle lane's or a fixed point's u is zero.
        add_block_transpose[6](o.pose, u0 - up[0], u1 - up[1], dst, po, pose_sum)
        comptime if REFINE_RIG:
            add_block_transpose[6](o.rig, u0 - ur[0], u1 - ur[1], dst, ro, rig_sum)
        comptime if REFINE_INTRINSICS:
            add_block_transpose[K](o.calib, u0 - uc[0], u1 - uc[1], dst, co, calib_sum)
    pose_sum.flush[6](dst)
    rig_sum.flush[6](dst)
    calib_sum.flush[K](dst)
