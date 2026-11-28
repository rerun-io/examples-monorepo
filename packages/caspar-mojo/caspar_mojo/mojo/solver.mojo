"""Host side of the solver: CASPAR's `GraphSolver::solve` (solver.cc:5284-5577) with the reduced camera system
(`RcsReduceRhs` / `RcsMatvec` / `RcsBackSubstitute`, solver.cc:11859-11921), over the generic kernels.

Free of Python types: the problem arrives as host pointers and sizes, so a C-ABI export can wrap `solve` as the
Python extension does. Host arrays are f32 / i32 / u8 (masks), row-major.
"""

from std.math import ceildiv, isfinite
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu import WARP_SIZE
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext, HostBuffer
from layout import TileTensor, row_major

from camera_models import CameraModel, FullOpenCV, MAX_PARAMS, OpenCVFisheye, Pinhole, SimpleRadial
from factor_kernels import (
    BLOCK,
    WARPS,
    GATHER_BACKSUBSTITUTE,
    GATHER_NEGATE,
    POINT_LANES,
    cross_points_kernel,
    gather_rows_kernel,
    linearize_kernel,
    point_gather_kernel,
    point_linearize_kernel,
    score_kernel,
    tri,
)
from node_kernels import (
    axpy_kernel,
    dot_kernel,
    pcg_direction_kernel,
    pcg_precond_kernel,
    point_solve_kernel,
    retract_kernel,
)

comptime F32HostIn = ImmPointer[Float32, ImmUntrackedOrigin]
comptime I32HostIn = ImmPointer[Int32, ImmUntrackedOrigin]
comptime U8HostIn = ImmPointer[UInt8, ImmUntrackedOrigin]
comptime F32HostOut = MutPointer[Float32, MutUntrackedOrigin]

# scalars[]: the costs and the predicted decrease, then one (rho_k, p_k.w_k) pair per PCG iteration.
comptime SLOT_COST_LINEARIZE = 0
comptime SLOT_COST_TRIAL = 1
comptime SLOT_PRED = 2
comptime SLOT_PCG = 4

comptime COST_DECREASE_FLOOR: Float64 = 1e-6
"""Smallest relative cost decrease a trial must show to be accepted. CASPAR's float32 cost sum cannot resolve a
smaller one (its noise is about 1e-6 relative); our float64 sum can, and polishing below it costs up to 2.7x the solve
for no change in the result. 1e-6 is also Ceres's default function tolerance."""

comptime BLOCKS_PER_SM = 6
"""Resident blocks per SM the tiled observation kernels aim for; each block then walks consecutive tiles."""

comptime EXIT_MAX_ITERATIONS = 0
comptime EXIT_CONVERGED_DIAG_EXIT = 1
comptime EXIT_CONVERGED_SCORE_THRESHOLD = 2


@fieldwise_init
struct HostProblem(ImplicitlyCopyable):
    var n_frames: Int
    var n_cameras: Int
    var n_points: Int
    var n_obs: Int
    var intrinsics: F32HostIn
    """[n_cameras, 12]"""
    var intrinsics_free: U8HostIn
    var sensor_from_rig: F32HostIn
    """[n_cameras, 7]"""
    var sensor_free: U8HostIn
    var rig_from_world: F32HostIn
    """[n_frames, 7]"""
    var pose_free: U8HostIn
    var points: F32HostIn
    """[n_points, 3]"""
    var point_free: U8HostIn
    var pixels: F32HostIn
    """[n_obs, 2]"""
    var obs_frame: I32HostIn
    var obs_camera: I32HostIn
    var obs_point: I32HostIn


@fieldwise_init
struct HostState(ImplicitlyCopyable):
    """Output arrays, same shapes as the problem's state."""

    var intrinsics: F32HostOut
    var sensor_from_rig: F32HostOut
    var rig_from_world: F32HostOut
    var points: F32HostOut


@fieldwise_init
struct Options(ImplicitlyCopyable):
    var solver_iter_max: Int
    var diag_init: Float32
    var diag_min: Float32
    var diag_scaling_up: Float32
    var diag_scaling_down: Float32
    var diag_exit_value: Float32
    var score_exit_value: Float32
    var solver_rel_decrease_min: Float32
    var reduced_pcg_iter_max: Int
    var reduced_pcg_eta: Float64


@fieldwise_init
struct Trial(ImplicitlyCopyable):
    """One LM trial, the fields of CASPAR's print_progress line."""

    var cost_tried: Float32
    var cost_best: Float32
    var damping: Float32
    var pcg_iterations: Int
    var accepted: Bool


@fieldwise_init
struct Stats(Movable):
    var initial_cost: Float32
    var final_cost: Float32
    var iterations: Int
    var pcg_iterations: Int
    var exit_reason: Int
    var solve_seconds: Float64
    var launches: Int
    var trace: List[Trial]


@fieldwise_init
struct TangentLayout(ImplicitlyCopyable):
    """Sizes of the flat tangent vector `[poses 6 | rigs 6 | calib K | points 3]` and of the packed block diagonal."""

    var n6: Int
    var n_calib: Int
    var n_free_points: Int
    var n_cam: Int
    var n_tan: Int
    var n_h_cam: Int
    var n_h: Int


def tangent_layout[K: Int](problem: HostProblem) -> TangentLayout:
    var n6 = 0
    for i in range(problem.n_frames):
        n6 += Int(problem.pose_free[unsafe_offset=i])
    var n_calib = 0
    for c in range(problem.n_cameras):
        n6 += Int(problem.sensor_free[unsafe_offset=c])
        n_calib += Int(problem.intrinsics_free[unsafe_offset=c])
    var n_points = 0
    for j in range(problem.n_points):
        n_points += Int(problem.point_free[unsafe_offset=j])
    var n_cam = 6 * n6 + K * n_calib
    var n_h_cam = tri(6) * n6 + tri(K) * n_calib
    return TangentLayout(n6, n_calib, n_points, n_cam, n_cam + 3 * n_points, n_h_cam, n_h_cam + tri(3) * n_points)


def _offsets(free: U8HostIn, n: Int, start: Int, width: Int, mut next: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    for i in range(n):
        if free[unsafe_offset=i] != 0:
            out.append(Int32(start + width * next))
            next += 1
        else:
            out.append(-1)
    return out^


struct ByPoint(Movable):
    """Observations regrouped by point (a counting sort): point j's observations are `rows[start[j]:start[j+1]]`."""

    var start: List[Int32]
    var rows: HostBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, problem: HostProblem) raises:
        self.start = List[Int32](length=problem.n_points + 1, fill=0)
        for n in range(problem.n_obs):
            self.start[Int(problem.obs_point[unsafe_offset=n]) + 1] += 1
        for j in range(problem.n_points):
            self.start[j + 1] += self.start[j]
        var next = self.start.copy()
        # Pinned, and the device gathers the point-ordered columns through it: the host scatters one value per
        # observation instead of four (the scatter, not the arithmetic, is the cost: 17 ms on clip 007).
        self.rows = ctx.enqueue_create_host_buffer[DType.int32](max(problem.n_obs, 1))
        ctx.synchronize()
        var rows = self.rows.unsafe_ptr()
        for n in range(problem.n_obs):
            var j = Int(problem.obs_point[unsafe_offset=n])
            rows[unsafe_offset=Int(next[j])] = Int32(n)
            next[j] += 1


@fieldwise_init
struct ImageOrder(Movable):
    """Observation columns re-sorted by (camera, frame), for a caller whose rows come in another order."""

    var frame: List[Int32]
    var camera: List[Int32]
    var point: List[Int32]
    var pixels: List[Float32]


def _in_image_order(problem: HostProblem) -> Bool:
    for n in range(1, problem.n_obs):
        var camera = problem.obs_camera[unsafe_offset=n]
        var previous = problem.obs_camera[unsafe_offset=n - 1]
        if camera < previous or (camera == previous and problem.obs_frame[unsafe_offset=n] < problem.obs_frame[unsafe_offset=n - 1]):
            return False
    return True


def _counting_order(keys: I32HostIn, n_keys: Int, rows: List[Int]) -> List[Int]:
    """`rows` stably reordered by keys[row] (a counting sort)."""
    var start = List[Int](length=n_keys + 1, fill=0)
    for row in rows:
        start[Int(keys[unsafe_offset=row]) + 1] += 1
    for k in range(n_keys):
        start[k + 1] += start[k]
    var out = List[Int](length=len(rows), fill=0)
    for row in rows:
        var k = Int(keys[unsafe_offset=row])
        out[start[k]] = row
        start[k] += 1
    return out^


def _image_order(problem: HostProblem) -> ImageOrder:
    """Two stable counting sorts (frame, then camera): O(observations + frames + cameras)."""
    var rows = List[Int](capacity=problem.n_obs)
    for n in range(problem.n_obs):
        rows.append(n)
    rows = _counting_order(problem.obs_frame, problem.n_frames, rows)
    rows = _counting_order(problem.obs_camera, problem.n_cameras, rows)
    var order = ImageOrder(
        List[Int32](capacity=problem.n_obs), List[Int32](capacity=problem.n_obs), List[Int32](capacity=problem.n_obs),
        List[Float32](capacity=2 * problem.n_obs),
    )
    for row in rows:
        order.frame.append(problem.obs_frame[unsafe_offset=row])
        order.camera.append(problem.obs_camera[unsafe_offset=row])
        order.point.append(problem.obs_point[unsafe_offset=row])
        order.pixels.append(problem.pixels[unsafe_offset=2 * row])
        order.pixels.append(problem.pixels[unsafe_offset=2 * row + 1])
    return order^


def _warp_tiles(problem: HostProblem) -> List[Int32]:
    """Rows (first observation, count) of up to WARP_SIZE consecutive observations of one image each, flattened."""
    var tiles = List[Int32](capacity=2 * (ceildiv(problem.n_obs, WARP_SIZE) + problem.n_frames))
    var begin = 0
    for n in range(1, problem.n_obs + 1):
        var boundary = n == problem.n_obs
        if not boundary:
            boundary = (
                problem.obs_camera[unsafe_offset=n] != problem.obs_camera[unsafe_offset=begin]
                or problem.obs_frame[unsafe_offset=n] != problem.obs_frame[unsafe_offset=begin]
                or n - begin == WARP_SIZE
            )
        if boundary:
            tiles.append(Int32(begin))
            tiles.append(Int32(n - begin))
            begin = n
    return tiles^


def _grid(n: Int) -> Int:
    return max(ceildiv(n, BLOCK), 1)


def _upload[dtype: DType](ctx: DeviceContext, src: ImmPointer[Scalar[dtype], ImmUntrackedOrigin], n: Int) raises -> DeviceBuffer[dtype]:
    var buf = ctx.enqueue_create_buffer[dtype](max(n, 1))
    if n > 0:
        ctx.enqueue_copy(buf.create_sub_buffer[dtype](0, n), src)
    return buf^


def _upload_list[dtype: DType](ctx: DeviceContext, values: List[Scalar[dtype]]) raises -> DeviceBuffer[dtype]:
    var buf = ctx.enqueue_create_buffer[dtype](max(len(values), 1))
    if len(values) > 0:
        ctx.enqueue_copy(buf.create_sub_buffer[dtype](0, len(values)), values.unsafe_ptr())
    return buf^


def _download(ctx: DeviceContext, dst: F32HostOut, src: DeviceBuffer[DType.float32], n: Int) raises:
    if n > 0:
        ctx.enqueue_copy(dst, src.create_sub_buffer[DType.float32](0, n))


struct KernelClock(Movable):
    """With CASPAR_MOJO_PROF=1, a sync after every kernel and its time charged to its name, printed at the end.
    The syncs serialise the stream, so a profiled solve is never a timed one."""

    var on: Bool
    var names: List[StaticString]
    var nanos: List[Int]
    var last: Int

    def __init__(out self):
        self.on = getenv("CASPAR_MOJO_PROF") == "1"
        self.names = List[StaticString]()
        self.nanos = List[Int]()
        self.last = perf_counter_ns()

    def tick(mut self, ctx: DeviceContext, name: StaticString) raises:
        if not self.on:
            return
        ctx.synchronize()
        var now = perf_counter_ns()
        for i in range(len(self.names)):
            if self.names[i] == name:
                self.nanos[i] += now - self.last
                self.last = now
                return
        self.names.append(name)
        self.nanos.append(now - self.last)
        self.last = now

    def report(self):
        if not self.on:
            return
        var line = String("CASPAR_MOJO_PROF")
        for i in range(len(self.names)):
            line.write(" ", self.names[i], "=", self.nanos[i] // 1000, "us")
        print(line)


def _usable(x: Float64) -> Bool:
    return x > 0.0 and isfinite(x)


def solve[Model: CameraModel, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool](
    ctx: DeviceContext,
    var problem: HostProblem,
    options: Options,
    result: HostState,
    linearize_gradient: F32HostOut,
    linearize_hblk: F32HostOut,
    linearize_only: Bool,
) raises -> Stats:
    """Run the LM solve and write the accepted state into `result`, or with `linearize_only` linearise once at the
    given state and write the gradient -J^T e and the packed block diagonal instead."""
    comptime K = Model.NUM_CALIB
    var image_order = Optional[ImageOrder]()
    if not _in_image_order(problem):
        # The warp tiles hold one image each: any order is correct, (camera, frame) order makes them full.
        image_order = _image_order(problem)
        problem.obs_frame = I32HostIn(unsafe_from_address=Int(image_order.value().frame.unsafe_ptr()))
        problem.obs_camera = I32HostIn(unsafe_from_address=Int(image_order.value().camera.unsafe_ptr()))
        problem.obs_point = I32HostIn(unsafe_from_address=Int(image_order.value().point.unsafe_ptr()))
        problem.pixels = F32HostIn(unsafe_from_address=Int(image_order.value().pixels.unsafe_ptr()))
    var nf = problem.n_frames
    var nc = problem.n_cameras
    var np = problem.n_points
    var no = problem.n_obs
    var lay = tangent_layout[K](problem)
    var next = 0
    var pose_off = _offsets(problem.pose_free, nf, 0, 6, next)
    var rig_off = _offsets(problem.sensor_free, nc, 0, 6, next)
    next = 0
    var calib_off = _offsets(problem.intrinsics_free, nc, 6 * lay.n6, K, next)
    next = 0
    var point_off = _offsets(problem.point_free, np, lay.n_cam, 3, next)

    var d_intr = _upload[DType.float32](ctx, problem.intrinsics, nc * MAX_PARAMS)
    var d_rigs = _upload[DType.float32](ctx, problem.sensor_from_rig, nc * 7)
    var d_poses = _upload[DType.float32](ctx, problem.rig_from_world, nf * 7)
    var d_points = _upload[DType.float32](ctx, problem.points, np * 3)
    var t_intr = ctx.enqueue_create_buffer[DType.float32](max(nc * MAX_PARAMS, 1))
    var t_rigs = ctx.enqueue_create_buffer[DType.float32](max(nc * 7, 1))
    var t_poses = ctx.enqueue_create_buffer[DType.float32](max(nf * 7, 1))
    var t_points = ctx.enqueue_create_buffer[DType.float32](max(np * 3, 1))
    var d_pixels = _upload[DType.float32](ctx, problem.pixels, no * 2)
    var d_frame = _upload[DType.int32](ctx, problem.obs_frame, no)
    var d_camera = _upload[DType.int32](ctx, problem.obs_camera, no)
    var d_point = _upload[DType.int32](ctx, problem.obs_point, no)
    var d_pose_off = _upload_list(ctx, pose_off)
    var d_rig_off = _upload_list(ctx, rig_off)
    var d_calib_off = _upload_list(ctx, calib_off)
    var d_point_off = _upload_list(ctx, point_off)
    var by_point = ByPoint(ctx, problem)
    var d_point_start = _upload_list(ctx, by_point.start)
    var d_rows = ctx.enqueue_create_buffer[DType.int32](max(no, 1))
    ctx.enqueue_copy(d_rows, by_point.rows)
    var d_by_point_frame = ctx.enqueue_create_buffer[DType.int32](max(no, 1))
    var d_by_point_camera = ctx.enqueue_create_buffer[DType.int32](max(no, 1))
    var d_by_point_pixels = ctx.enqueue_create_buffer[DType.float32](max(2 * no, 1))
    var tile_rows = _warp_tiles(problem)
    var n_warp_tiles = len(tile_rows) // 2
    var d_warp_tiles = _upload_list(ctx, tile_rows)
    var nt = max(lay.n_tan, 1)
    var d_g = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_r = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_z = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_p = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_w = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_mp = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_s = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_h = ctx.enqueue_create_buffer[DType.float32](max(lay.n_h, 1))
    var n_scalars = SLOT_PCG + 2 * (options.reduced_pcg_iter_max + 1)
    var d_scalars = ctx.enqueue_create_buffer[DType.float64](n_scalars)
    var host2 = ctx.enqueue_create_host_buffer[DType.float64](2)
    ctx.enqueue_memset(d_p, 0.0)
    ctx.enqueue_memset(d_mp, 0.0)
    ctx.enqueue_memset(d_s, 0.0)
    ctx.enqueue_memset(d_z, 0.0)
    ctx.enqueue_memset(d_w, 0.0)

    # Tensor views that never change; the state views follow the current/trial buffers.
    var pixels = TileTensor(d_pixels, row_major((no, 2))).as_imm()
    var obs_frame = TileTensor(d_frame, row_major((no,))).as_imm()
    var obs_camera = TileTensor(d_camera, row_major((no,))).as_imm()
    var obs_point = TileTensor(d_point, row_major((no,))).as_imm()
    var t_pose_off = TileTensor(d_pose_off, row_major((nf,))).as_imm()
    var t_rig_off = TileTensor(d_rig_off, row_major((nc,))).as_imm()
    var t_calib_off = TileTensor(d_calib_off, row_major((nc,))).as_imm()
    var t_point_off = TileTensor(d_point_off, row_major((np,))).as_imm()
    var point_start = TileTensor(d_point_start, row_major((np + 1,))).as_imm()
    var by_point_frame = TileTensor(d_by_point_frame, row_major((no,))).as_imm()
    var by_point_camera = TileTensor(d_by_point_camera, row_major((no,))).as_imm()
    var by_point_pixels = TileTensor(d_by_point_pixels, row_major((no, 2))).as_imm()
    var warp_tiles = TileTensor(d_warp_tiles, row_major((n_warp_tiles, 2))).as_imm()
    ctx.enqueue_function[gather_rows_kernel](
        TileTensor(d_rows, row_major((no,))).as_imm(), obs_frame, obs_camera, pixels, TileTensor(d_by_point_frame, row_major((no,))),
        TileTensor(d_by_point_camera, row_major((no,))), TileTensor(d_by_point_pixels, row_major((no, 2))), grid_dim=_grid(no), block_dim=BLOCK,
    )
    var g = TileTensor(d_g, row_major((nt,)))
    var r = TileTensor(d_r, row_major((nt,)))
    var z = TileTensor(d_z, row_major((nt,)))
    var p = TileTensor(d_p, row_major((nt,)))
    var w = TileTensor(d_w, row_major((nt,)))
    var mp = TileTensor(d_mp, row_major((nt,)))
    var s = TileTensor(d_s, row_major((nt,)))
    var hblk = TileTensor(d_h, row_major((max(lay.n_h, 1),))).as_imm()
    var scalars = d_scalars.unsafe_ptr()
    var n6 = Int32(lay.n6)
    var ncal = Int32(lay.n_calib)
    var point_v = Int32(lay.n_cam)
    var point_h = Int32(lay.n_h_cam)
    var n_free_points = Int32(lay.n_free_points)

    comptime linearize = linearize_kernel[Model, REFINE_INTRINSICS, REFINE_RIG]
    comptime score = score_kernel[Model]
    comptime point_linearize = point_linearize_kernel[Model]
    comptime cross_points = cross_points_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, False]
    comptime cross_all = cross_points_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, True]
    comptime gather_negate = point_gather_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, GATHER_NEGATE]
    comptime gather_backsubstitute = point_gather_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, GATHER_BACKSUBSTITUTE]
    comptime precond = pcg_precond_kernel[K]
    comptime retract = retract_kernel[Model]

    # Kernel launches only; memsets and device copies are not counted.
    var launches = 0
    var clock = KernelClock()
    var obs_grid = _grid(no)
    var n_tiles = ceildiv(n_warp_tiles, WARPS)
    var tiles = max(ceildiv(n_tiles, BLOCKS_PER_SM * ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)), 1)
    var tile_grid = max(ceildiv(n_tiles, tiles), 1)
    var cam_blocks = lay.n6 + lay.n_calib

    ctx.synchronize()
    _ = image_order^  # the uploads above read its lists through raw addresses
    _ = by_point^  # and the pinned rows asynchronously
    var t0 = perf_counter_ns()
    clock.last = perf_counter_ns()

    # Linearise at the current state: g = -J^T e, the block diagonal, the J cache, the cost into its slot.
    ctx.enqueue_memset(d_g, 0.0)
    ctx.enqueue_memset(d_h, 0.0)
    ctx.enqueue_memset(d_scalars, 0.0)
    ctx.enqueue_function[linearize](
        TileTensor(d_poses, row_major((nf, 7))).as_imm(), TileTensor(d_rigs, row_major((nc, 7))).as_imm(),
        TileTensor(d_intr, row_major((nc, MAX_PARAMS))).as_imm(), TileTensor(d_points, row_major((np, 3))).as_imm(),
        t_pose_off, t_rig_off, t_calib_off, t_point_off, obs_frame, obs_camera, obs_point, pixels,
        warp_tiles, d_g.unsafe_ptr(), d_h.unsafe_ptr(), scalars, Int32(SLOT_COST_LINEARIZE), n6, ncal, Int32(tiles),
        grid_dim=tile_grid, block_dim=BLOCK,
    )
    clock.tick(ctx, "linearize")
    ctx.enqueue_function[point_linearize](
        TileTensor(d_poses, row_major((nf, 7))).as_imm(), TileTensor(d_rigs, row_major((nc, 7))).as_imm(),
        TileTensor(d_intr, row_major((nc, MAX_PARAMS))).as_imm(), TileTensor(d_points, row_major((np, 3))).as_imm(),
        t_point_off, point_start, by_point_frame, by_point_camera, by_point_pixels, g, TileTensor(d_h, row_major((max(lay.n_h, 1),))),
        point_v, point_h, grid_dim=_grid(np * POINT_LANES), block_dim=BLOCK,
    )
    clock.tick(ctx, "point_linearize")
    launches += 2
    ctx.enqueue_copy(host2, d_scalars.create_sub_buffer[DType.float64](SLOT_COST_LINEARIZE, 2))
    ctx.synchronize()
    var score_best = Float32(host2[0])
    var initial_cost = score_best
    if linearize_only:
        _download(ctx, linearize_gradient, d_g, lay.n_tan)
        _download(ctx, linearize_hblk, d_h, lay.n_h)
        ctx.synchronize()
        return Stats(initial_cost, initial_cost, 0, 0, EXIT_MAX_ITERATIONS, 0.0, launches, List[Trial]())

    var trace = List[Trial](capacity=options.solver_iter_max)
    var diag = options.diag_init
    var up_scale = options.diag_scaling_up
    var exit_reason = EXIT_MAX_ITERATIONS
    var pcg_total = 0
    # CASPAR's iteration_count: the index of the trial that ended the solve, or solver_iter_max.
    var trials = options.solver_iter_max
    # Whether the linearisation state sits in the trial buffers: after an accept that the next iteration does not
    # re-linearise (CASPAR skips it on the last allowed iteration), the products must still see the old state.
    var lin_in_trial = False
    for it in range(options.solver_iter_max):
        if it != 0 and it < options.solver_iter_max - 1:
            lin_in_trial = False
            ctx.enqueue_memset(d_g, 0.0)
            ctx.enqueue_memset(d_h, 0.0)
            ctx.enqueue_function[linearize](
                TileTensor(d_poses, row_major((nf, 7))).as_imm(), TileTensor(d_rigs, row_major((nc, 7))).as_imm(),
                TileTensor(d_intr, row_major((nc, MAX_PARAMS))).as_imm(), TileTensor(d_points, row_major((np, 3))).as_imm(),
                t_pose_off, t_rig_off, t_calib_off, t_point_off, obs_frame, obs_camera, obs_point, pixels,
                warp_tiles, d_g.unsafe_ptr(), d_h.unsafe_ptr(), scalars, Int32(SLOT_COST_LINEARIZE), n6, ncal, Int32(tiles),
                grid_dim=tile_grid, block_dim=BLOCK,
            )
            clock.tick(ctx, "linearize")
            ctx.enqueue_function[point_linearize](
                TileTensor(d_poses, row_major((nf, 7))).as_imm(), TileTensor(d_rigs, row_major((nc, 7))).as_imm(),
                TileTensor(d_intr, row_major((nc, MAX_PARAMS))).as_imm(), TileTensor(d_points, row_major((np, 3))).as_imm(),
                t_point_off, point_start, by_point_frame, by_point_camera, by_point_pixels, g, TileTensor(d_h, row_major((max(lay.n_h, 1),))),
                point_v, point_h, grid_dim=_grid(np * POINT_LANES), block_dim=BLOCK,
            )
            clock.tick(ctx, "point_linearize")
            launches += 2
        var lin_poses = TileTensor(t_poses if lin_in_trial else d_poses, row_major((nf, 7))).as_imm()
        var lin_rigs = TileTensor(t_rigs if lin_in_trial else d_rigs, row_major((nc, 7))).as_imm()
        var lin_intr = TileTensor(t_intr if lin_in_trial else d_intr, row_major((nc, MAX_PARAMS))).as_imm()
        var lin_points = TileTensor(t_points if lin_in_trial else d_points, row_major((np, 3))).as_imm()
        ctx.enqueue_memset(d_scalars, 0.0)
        # RcsReduceRhs: r_cam = g_cam - H_cp H_pp^-1 g_p.
        ctx.enqueue_copy(dst_buf=d_r, src_buf=d_g)
        ctx.enqueue_function[point_solve_kernel](
            g.as_imm(), z, hblk, point_v, point_h, n_free_points, diag, grid_dim=_grid(lay.n_free_points), block_dim=BLOCK
        )
        clock.tick(ctx, "point_solve")
        ctx.enqueue_function[cross_points](
            lin_poses, lin_rigs, lin_intr, lin_points, t_pose_off, t_rig_off, t_calib_off, t_point_off, obs_frame, obs_camera, obs_point,
            warp_tiles, z.as_imm(), z.as_imm(), d_r.unsafe_ptr(), Int32(tiles), grid_dim=tile_grid, block_dim=BLOCK,
        )
        clock.tick(ctx, "cross_points")
        launches += 2

        var pcg_q: Float64 = 0.0
        var alpha: Float32 = 0.0
        var pending = False
        var k = 0
        while k < options.reduced_pcg_iter_max:
            var first = Int32(1 if k == 0 else 0)
            var rho_slot = Int32(SLOT_PCG + 2 * k)
            ctx.enqueue_function[precond](
                r, z, s, p.as_imm(), w.as_imm(), hblk, n6, ncal, diag, alpha, first, scalars, rho_slot,
                grid_dim=_grid(cam_blocks), block_dim=BLOCK,
            )
            clock.tick(ctx, "precond")
            pending = False
            ctx.enqueue_function[pcg_direction_kernel](
                r.as_imm(), z.as_imm(), p, mp, w, Int32(lay.n_cam), first, scalars, rho_slot, grid_dim=_grid(lay.n_tan), block_dim=BLOCK
            )
            clock.tick(ctx, "pcg_direction")
            # RcsMatvec: w_cam = M_cc p + H_cc' p - H_cp H_pp^-1 H_pc p (w enters holding M p).
            ctx.enqueue_function[gather_negate](
                lin_poses, lin_rigs, lin_intr, lin_points, t_pose_off, t_rig_off, t_calib_off, t_point_off, point_start, by_point_frame,
                by_point_camera, p.as_imm(), g.as_imm(), z, hblk, point_v, point_h, diag, grid_dim=_grid(np * POINT_LANES), block_dim=BLOCK,
            )
            clock.tick(ctx, "point_gather")
            ctx.enqueue_function[cross_all](
                lin_poses, lin_rigs, lin_intr, lin_points, t_pose_off, t_rig_off, t_calib_off, t_point_off, obs_frame, obs_camera, obs_point,
                warp_tiles, p.as_imm(), z.as_imm(), d_w.unsafe_ptr(), Int32(tiles), grid_dim=tile_grid, block_dim=BLOCK,
            )
            clock.tick(ctx, "cross_points")
            ctx.enqueue_function[dot_kernel](
                p.as_imm(), w.as_imm(), Int32(lay.n_cam), scalars, rho_slot + 1, grid_dim=_grid(lay.n_cam), block_dim=BLOCK
            )
            clock.tick(ctx, "dot")
            launches += 5
            ctx.enqueue_copy(host2, d_scalars.create_sub_buffer[DType.float64](Int(rho_slot), 2))
            ctx.synchronize()
            var rho = Float32(host2[0])
            alpha = rho / Float32(host2[1])
            k += 1
            # A rho or alpha that is not finite and positive ends the PCG before its step (k = 1: a zero step).
            if not _usable(Float64(rho)) or not _usable(Float64(alpha)):
                break
            pending = True
            var decrease = 0.5 * Float64(alpha) * Float64(rho)
            pcg_q -= decrease
            if pcg_q < 0.0 and Float64(k) * decrease < -pcg_q * options.reduced_pcg_eta:
                break
        pcg_total += k
        if pending:
            ctx.enqueue_function[axpy_kernel](s, p.as_imm(), alpha, Int32(lay.n_cam), grid_dim=_grid(lay.n_cam), block_dim=BLOCK)
            clock.tick(ctx, "axpy")
            launches += 1
        # RcsBackSubstitute: s_p = H_pp^-1 (g_p - H_pc s_cam), reading s_cam from a copy (w is free now).
        ctx.enqueue_copy(dst_buf=d_w, src_buf=d_s)
        ctx.enqueue_function[gather_backsubstitute](
            lin_poses, lin_rigs, lin_intr, lin_points, t_pose_off, t_rig_off, t_calib_off, t_point_off, point_start, by_point_frame,
            by_point_camera, w.as_imm(), g.as_imm(), s, hblk, point_v, point_h, diag, grid_dim=_grid(np * POINT_LANES), block_dim=BLOCK,
        )
        clock.tick(ctx, "point_backsubstitute")
        # Retract into the trial state (with s.(g + lambda D s) fused), then score it.
        ctx.enqueue_function[retract](
            TileTensor(d_poses, row_major((nf, 7))).as_imm(), TileTensor(d_rigs, row_major((nc, 7))).as_imm(),
            TileTensor(d_intr, row_major((nc, MAX_PARAMS))).as_imm(), TileTensor(d_points, row_major((np, 3))).as_imm(),
            TileTensor(t_poses, row_major((nf, 7))), TileTensor(t_rigs, row_major((nc, 7))),
            TileTensor(t_intr, row_major((nc, MAX_PARAMS))), TileTensor(t_points, row_major((np, 3))),
            t_pose_off, t_rig_off, t_calib_off, t_point_off, s.as_imm(), g.as_imm(), hblk, n6, ncal, diag, scalars, Int32(SLOT_PRED),
            grid_dim=_grid(nf + nc + np), block_dim=BLOCK,
        )
        clock.tick(ctx, "retract")
        ctx.enqueue_function[score](
            TileTensor(t_poses, row_major((nf, 7))).as_imm(), TileTensor(t_rigs, row_major((nc, 7))).as_imm(),
            TileTensor(t_intr, row_major((nc, MAX_PARAMS))).as_imm(), TileTensor(t_points, row_major((np, 3))).as_imm(),
            obs_frame, obs_camera, obs_point, pixels, scalars, Int32(SLOT_COST_TRIAL),
            grid_dim=obs_grid, block_dim=BLOCK,
        )
        clock.tick(ctx, "score")
        launches += 3
        ctx.enqueue_copy(host2, d_scalars.create_sub_buffer[DType.float64](SLOT_COST_TRIAL, 2))
        ctx.synchronize()
        var score_trial = Float32(host2[0])
        var damping = diag
        var accepted = Float64(score_trial) < Float64(score_best) * Float64(options.solver_rel_decrease_min) * (1.0 - COST_DECREASE_FLOOR)
        if accepted:
            var quality = (score_best - score_trial) / Float32(0.5 * host2[1])
            var q = 2.0 * quality - 1.0
            var scale = max(options.diag_scaling_down, 1.0 - q * q * q)
            diag = max(options.diag_min, diag * scale)
            up_scale = options.diag_scaling_up
            score_best = score_trial
            swap(d_poses, t_poses)
            swap(d_rigs, t_rigs)
            swap(d_intr, t_intr)
            swap(d_points, t_points)
            lin_in_trial = not lin_in_trial
        else:
            diag = diag * up_scale
            up_scale *= 2.0
        trace.append(Trial(score_trial, score_best, damping, k, accepted))
        if not accepted and diag > options.diag_exit_value:
            exit_reason = EXIT_CONVERGED_DIAG_EXIT
            trials = it
            break
        if score_best <= options.score_exit_value:
            exit_reason = EXIT_CONVERGED_SCORE_THRESHOLD
            trials = it
            break
    ctx.synchronize()
    var solve_seconds = Float64(perf_counter_ns() - t0) * 1e-9
    clock.report()
    _download(ctx, result.intrinsics, d_intr, nc * MAX_PARAMS)
    _download(ctx, result.sensor_from_rig, d_rigs, nc * 7)
    _download(ctx, result.rig_from_world, d_poses, nf * 7)
    _download(ctx, result.points, d_points, np * 3)
    ctx.synchronize()
    return Stats(initial_cost, score_best, trials, pcg_total, exit_reason, solve_seconds, launches, trace^)


def _any(mask: U8HostIn, n: Int) -> Bool:
    for i in range(n):
        if mask[unsafe_offset=i] != 0:
            return True
    return False


def _solve_flags[Model: CameraModel](
    ctx: DeviceContext,
    problem: HostProblem,
    options: Options,
    result: HostState,
    linearize_gradient: F32HostOut,
    linearize_hblk: F32HostOut,
    linearize_only: Bool,
) raises -> Stats:
    match (_any(problem.intrinsics_free, problem.n_cameras), _any(problem.sensor_free, problem.n_cameras)):
        case (True, True):
            return solve[Model, True, True](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)
        case (True, False):
            return solve[Model, True, False](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)
        case (False, True):
            return solve[Model, False, True](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)
        case (False, False):
            return solve[Model, False, False](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)


def solve_model(
    ctx: DeviceContext,
    model: Int,
    problem: HostProblem,
    options: Options,
    result: HostState,
    linearize_gradient: F32HostOut,
    linearize_hblk: F32HostOut,
    linearize_only: Bool,
) raises -> Stats:
    """`solve` for a runtime camera model (the `CameraModel` value of `caspar_mojo.problem`) and the refined groups the
    masks free."""
    match model:
        case 0:
            return _solve_flags[Pinhole](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)
        case 1:
            return _solve_flags[SimpleRadial](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)
        case 2:
            return _solve_flags[OpenCVFisheye](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)
        case 3:
            return _solve_flags[FullOpenCV](ctx, problem, options, result, linearize_gradient, linearize_hblk, linearize_only)
        case _:
            raise Error(t"unknown camera model {model}")
