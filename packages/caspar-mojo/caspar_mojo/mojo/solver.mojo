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

from blocks import FRAME_WIDTH, POINT_WIDTH, Packed, tri
from camera_models import CameraModel, FullOpenCV, MAX_PARAMS, OpenCVFisheye, Pinhole, SimpleRadial
from factor_kernels import (
    F32In,
    F32Out,
    BLOCK,
    WARPS,
    POINT_LANES,
    cross_points_kernel,
    gather_rows_kernel,
    linearize_kernel,
    point_gather_kernel,
    point_linearize_kernel,
    score_kernel,
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
struct LinearizeOut(ImplicitlyCopyable):
    """Caller-owned outputs of `linearize_model`, capacities in floats."""

    var gradient: F32HostOut
    var gradient_capacity: Int
    var hblk: F32HostOut
    var hblk_capacity: Int

    def check(self, n_tan: Int, n_h: Int) raises:
        if self.gradient_capacity < n_tan or self.hblk_capacity < n_h:
            raise Error(t"linearize outputs hold {self.gradient_capacity} and {self.hblk_capacity} floats, need {n_tan} and {n_h}")


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


def _count(mask: U8HostIn, n: Int) -> Int:
    var count = 0
    for i in range(n):
        count += Int(mask[unsafe_offset=i] != 0)
    return count


@fieldwise_init
struct GroupCounts(ImplicitlyCopyable):
    """Free blocks per group, in the packed order (`blocks.mojo`)."""

    var poses: Int
    var sensors: Int
    var calibs: Int
    var points: Int

    def __init__(out self, problem: HostProblem):
        self.poses = _count(problem.pose_free, problem.n_frames)
        self.sensors = _count(problem.sensor_free, problem.n_cameras)
        self.calibs = _count(problem.intrinsics_free, problem.n_cameras)
        self.points = _count(problem.point_free, problem.n_points)

    def packed[K: Int](self) -> Packed[K]:
        # Poses then sensors share the frame group.
        return Packed[K](self.poses + self.sensors, self.calibs)


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
    return _upload[dtype](ctx, ImmPointer[Scalar[dtype], ImmUntrackedOrigin](unsafe_from_address=Int(values.unsafe_ptr())), len(values))


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


def _copy(dst: F32HostOut, src: F32HostIn, n: Int):
    for i in range(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


def _zero(dst: F32HostOut, n: Int):
    for i in range(n):
        dst[unsafe_offset=i] = 0.0


comptime INDEX_LIMIT = 2**31 - 1
"""Device indices are int32."""


def _check(problem: HostProblem, options: Options) raises:
    """The limits the buffer sizing and the int32 device indices rely on, before any buffer exists."""
    if options.solver_iter_max < 1 or options.reduced_pcg_iter_max < 1:
        raise Error(t"solver_iter_max {options.solver_iter_max} and reduced_pcg_iter_max {options.reduced_pcg_iter_max} must be >= 1")
    if not (0.0 < options.reduced_pcg_eta < 1.0):
        raise Error(t"reduced_pcg_eta {options.reduced_pcg_eta} must lie in (0, 1)")
    if min(problem.n_frames, problem.n_cameras, problem.n_points, problem.n_obs) < 0:
        raise Error("negative problem size")
    var packed = tri(3) * problem.n_points + tri(6) * (problem.n_frames + problem.n_cameras) + tri(MAX_PARAMS) * problem.n_cameras
    if 2 * problem.n_obs > INDEX_LIMIT or packed > INDEX_LIMIT:
        raise Error(t"problem too large for int32 indices: {problem.n_obs} observations, {packed} packed block entries")


@fieldwise_init
struct StateViews(ImplicitlyCopyable):
    var poses: F32In
    var rigs: F32In
    var intrinsics: F32In
    var points: F32In


@fieldwise_init
struct StateOuts(ImplicitlyCopyable):
    var poses: F32Out
    var rigs: F32Out
    var intrinsics: F32Out
    var points: F32Out


struct DeviceState(Movable):
    """The solver state on the device (rig_from_world, sensor_from_rig, intrinsics, points), owned as one group so the
    current and trial states swap as a whole."""

    var poses: DeviceBuffer[DType.float32]
    var rigs: DeviceBuffer[DType.float32]
    var intrinsics: DeviceBuffer[DType.float32]
    var points: DeviceBuffer[DType.float32]
    var n_frames: Int
    var n_cameras: Int
    var n_points: Int

    def __init__(out self, ctx: DeviceContext, problem: HostProblem, upload: Bool) raises:
        """The problem's state, or (`upload` False) an uninitialised state of the same shape."""
        self.n_frames = problem.n_frames
        self.n_cameras = problem.n_cameras
        self.n_points = problem.n_points
        if upload:
            self.poses = _upload[DType.float32](ctx, problem.rig_from_world, self.n_frames * 7)
            self.rigs = _upload[DType.float32](ctx, problem.sensor_from_rig, self.n_cameras * 7)
            self.intrinsics = _upload[DType.float32](ctx, problem.intrinsics, self.n_cameras * MAX_PARAMS)
            self.points = _upload[DType.float32](ctx, problem.points, self.n_points * 3)
        else:
            self.poses = ctx.enqueue_create_buffer[DType.float32](max(self.n_frames * 7, 1))
            self.rigs = ctx.enqueue_create_buffer[DType.float32](max(self.n_cameras * 7, 1))
            self.intrinsics = ctx.enqueue_create_buffer[DType.float32](max(self.n_cameras * MAX_PARAMS, 1))
            self.points = ctx.enqueue_create_buffer[DType.float32](max(self.n_points * 3, 1))

    def views(self) -> StateViews:
        return StateViews(
            TileTensor(self.poses, row_major((self.n_frames, 7))).as_imm(),
            TileTensor(self.rigs, row_major((self.n_cameras, 7))).as_imm(),
            TileTensor(self.intrinsics, row_major((self.n_cameras, MAX_PARAMS))).as_imm(),
            TileTensor(self.points, row_major((self.n_points, 3))).as_imm(),
        )

    def outs(self) -> StateOuts:
        return StateOuts(
            TileTensor(self.poses, row_major((self.n_frames, 7))),
            TileTensor(self.rigs, row_major((self.n_cameras, 7))),
            TileTensor(self.intrinsics, row_major((self.n_cameras, MAX_PARAMS))),
            TileTensor(self.points, row_major((self.n_points, 3))),
        )

    def download(self, ctx: DeviceContext, result: HostState) raises:
        _download(ctx, result.intrinsics, self.intrinsics, self.n_cameras * MAX_PARAMS)
        _download(ctx, result.sensor_from_rig, self.rigs, self.n_cameras * 7)
        _download(ctx, result.rig_from_world, self.poses, self.n_frames * 7)
        _download(ctx, result.points, self.points, self.n_points * 3)


@fieldwise_init
struct Role(Equatable, TrivialRegisterPassable):
    """Which of the two states a role (here: the linearisation point) refers to."""

    var value: Int
    comptime CURRENT = Self(0)
    comptime TRIAL = Self(1)

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def swapped(self) -> Self:
        """The same buffers after `current` and `trial` swap."""
        return Self(1 - self.value)


def _usable(x: Float64) -> Bool:
    return x > 0.0 and isfinite(x)


def solve[Model: CameraModel, REFINE_INTRINSICS: Bool, REFINE_RIG: Bool](
    ctx: DeviceContext,
    var problem: HostProblem,
    options: Options,
    result: HostState,
    collect_trace: Bool,
    linearize_hook: Optional[LinearizeOut],
) raises -> Stats:
    """Run the LM solve and write the accepted state into `result`, or with `linearize_hook` linearise once at the
    given state and write the gradient -J^T e and the packed block diagonal instead."""
    comptime K = Model.NUM_CALIB
    _check(problem, options)
    if problem.n_obs == 0:
        # No residual: cost 0, nothing moves (CASPAR ends such a solve on its score exit).
        if case .Some(hook) = linearize_hook:
            var counts = GroupCounts(problem)
            var lay = counts.packed[K]()
            hook.check(lay.n_tan(counts.points), lay.n_h(counts.points))
            _zero(hook.gradient, lay.n_tan(counts.points))
            _zero(hook.hblk, lay.n_h(counts.points))
        else:
            _copy(result.intrinsics, problem.intrinsics, problem.n_cameras * MAX_PARAMS)
            _copy(result.sensor_from_rig, problem.sensor_from_rig, problem.n_cameras * 7)
            _copy(result.rig_from_world, problem.rig_from_world, problem.n_frames * 7)
            _copy(result.points, problem.points, problem.n_points * 3)
        return Stats(0.0, 0.0, 0, 0, EXIT_CONVERGED_SCORE_THRESHOLD, 0.0, 0, List[Trial]())
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
    var counts = GroupCounts(problem)
    var lay = counts.packed[K]()
    var n_free_points = counts.points
    var n_cam = lay.point_start()
    var n_tan = lay.n_tan(n_free_points)
    var n_h = lay.n_h(n_free_points)
    if case .Some(hook) = linearize_hook:
        hook.check(n_tan, n_h)
    var next = 0
    var pose_off = _offsets(problem.pose_free, nf, 0, 6, next)
    var rig_off = _offsets(problem.sensor_free, nc, 0, 6, next)
    next = 0
    var calib_off = _offsets(problem.intrinsics_free, nc, lay.calib_start(), K, next)
    next = 0
    var point_off = _offsets(problem.point_free, np, lay.point_start(), 3, next)

    var current = DeviceState(ctx, problem, True)
    var trial = DeviceState(ctx, problem, False)
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
    var nt = max(n_tan, 1)
    var d_g = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_r = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_z = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_p = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_w = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_mp = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_s = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_h = ctx.enqueue_create_buffer[DType.float32](max(n_h, 1))
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
    var hblk = TileTensor(d_h, row_major((max(n_h, 1),))).as_imm()
    var scalars = d_scalars.unsafe_ptr()
    var n6 = Int32(lay.n6)
    var ncal = Int32(lay.n_calib)

    comptime linearize = linearize_kernel[Model, REFINE_INTRINSICS, REFINE_RIG]
    comptime score = score_kernel[Model]
    comptime point_linearize = point_linearize_kernel[Model]
    comptime cross_points = cross_points_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, False]
    comptime cross_all = cross_points_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, True]
    comptime gather_negate = point_gather_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, False]
    comptime gather_backsubstitute = point_gather_kernel[Model, REFINE_INTRINSICS, REFINE_RIG, True]
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

    var trace = List[Trial](capacity=options.solver_iter_max if collect_trace else 0)
    var diag = options.diag_init
    var up_scale = options.diag_scaling_up
    var exit_reason = EXIT_MAX_ITERATIONS
    var pcg_total = 0
    # CASPAR's iteration_count: the index of the trial that ended the solve, or solver_iter_max.
    var trials = options.solver_iter_max
    # The state the products differentiate at: `current` after each relinearisation; after an accept that the next
    # trial does not relinearise (CASPAR's last allowed trial), the state the step was solved at, by then `trial`.
    var linearised_at = Role.CURRENT
    var score_best: Float32 = 0.0
    var initial_cost: Float32 = 0.0
    for it in range(options.solver_iter_max):
        # CASPAR linearises first (res_jac_first) and again at every trial but the last allowed one.
        if it == 0 or it < options.solver_iter_max - 1:
            linearised_at = Role.CURRENT
            ctx.enqueue_memset(d_g, 0.0)
            ctx.enqueue_memset(d_h, 0.0)
            if it == 0:
                ctx.enqueue_memset(d_scalars, 0.0)
            var state = current.views()
            ctx.enqueue_function[linearize](
                state.poses, state.rigs, state.intrinsics, state.points,
                t_pose_off, t_rig_off, t_calib_off, t_point_off, obs_frame, obs_camera, obs_point, pixels,
                warp_tiles, d_g.unsafe_ptr(), d_h.unsafe_ptr(), scalars, Int32(SLOT_COST_LINEARIZE if it == 0 else -1), n6, ncal, Int32(tiles),
                grid_dim=tile_grid, block_dim=BLOCK,
            )
            clock.tick(ctx, "linearize")
            ctx.enqueue_function[point_linearize](
                state.poses, state.rigs, state.intrinsics, state.points,
                t_point_off, point_start, by_point_frame, by_point_camera, by_point_pixels, g, TileTensor(d_h, row_major((max(n_h, 1),))),
                n6, ncal, grid_dim=_grid(np * POINT_LANES), block_dim=BLOCK,
            )
            clock.tick(ctx, "point_linearize")
            launches += 2
            if it == 0:
                ctx.enqueue_copy(host2, d_scalars.create_sub_buffer[DType.float64](SLOT_COST_LINEARIZE, 2))
                ctx.synchronize()
                score_best = Float32(host2[0])
                initial_cost = score_best
                if case .Some(hook) = linearize_hook:
                    _download(ctx, hook.gradient, d_g, n_tan)
                    _download(ctx, hook.hblk, d_h, n_h)
                    ctx.synchronize()
                    return Stats(initial_cost, initial_cost, 0, 0, EXIT_MAX_ITERATIONS, 0.0, launches, List[Trial]())
        var lin = trial.views() if linearised_at == Role.TRIAL else current.views()
        ctx.enqueue_memset(d_scalars, 0.0)
        # RcsReduceRhs: r_cam = g_cam - H_cp H_pp^-1 g_p.
        ctx.enqueue_copy(dst_buf=d_r, src_buf=d_g)
        ctx.enqueue_function[point_solve_kernel[K]](
            g.as_imm(), z, hblk, n6, ncal, Int32(n_free_points), diag, grid_dim=_grid(n_free_points), block_dim=BLOCK
        )
        clock.tick(ctx, "point_solve")
        ctx.enqueue_function[cross_points](
            lin.poses, lin.rigs, lin.intrinsics, lin.points, t_pose_off, t_rig_off, t_calib_off, t_point_off, obs_frame, obs_camera, obs_point,
            warp_tiles, z.as_imm(), z.as_imm(), d_r.unsafe_ptr(), Int32(tiles), grid_dim=tile_grid, block_dim=BLOCK,
        )
        clock.tick(ctx, "cross_points")
        launches += 2

        var pcg_q: Float64 = 0.0
        var alpha: Float32 = 0.0
        var pending = False
        var k = 0
        while k < options.reduced_pcg_iter_max:
            var first = Int32(1 if k == 0 else 0)  # Bool is not DevicePassable
            var rho_slot = Int32(SLOT_PCG + 2 * k)
            ctx.enqueue_function[precond](
                r, z, s, p.as_imm(), w.as_imm(), hblk, n6, ncal, diag, alpha, first, scalars, rho_slot,
                grid_dim=_grid(cam_blocks), block_dim=BLOCK,
            )
            clock.tick(ctx, "precond")
            pending = False
            ctx.enqueue_function[pcg_direction_kernel](
                r.as_imm(), z.as_imm(), p, mp, w, Int32(n_cam), first, scalars, rho_slot, grid_dim=_grid(n_tan), block_dim=BLOCK
            )
            clock.tick(ctx, "pcg_direction")
            # RcsMatvec: w_cam = M_cc p + H_cc' p - H_cp H_pp^-1 H_pc p (w enters holding M p).
            ctx.enqueue_function[gather_negate](
                lin.poses, lin.rigs, lin.intrinsics, lin.points, t_pose_off, t_rig_off, t_calib_off, t_point_off, point_start, by_point_frame,
                by_point_camera, p.as_imm(), g.as_imm(), z, hblk, n6, ncal, diag, grid_dim=_grid(np * POINT_LANES), block_dim=BLOCK,
            )
            clock.tick(ctx, "point_gather")
            ctx.enqueue_function[cross_all](
                lin.poses, lin.rigs, lin.intrinsics, lin.points, t_pose_off, t_rig_off, t_calib_off, t_point_off, obs_frame, obs_camera, obs_point,
                warp_tiles, p.as_imm(), z.as_imm(), d_w.unsafe_ptr(), Int32(tiles), grid_dim=tile_grid, block_dim=BLOCK,
            )
            clock.tick(ctx, "cross_points")
            ctx.enqueue_function[dot_kernel](
                p.as_imm(), w.as_imm(), Int32(n_cam), scalars, rho_slot + 1, grid_dim=_grid(n_cam), block_dim=BLOCK
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
            ctx.enqueue_function[axpy_kernel](s, p.as_imm(), alpha, Int32(n_cam), grid_dim=_grid(n_cam), block_dim=BLOCK)
            clock.tick(ctx, "axpy")
            launches += 1
        # RcsBackSubstitute: s_p = H_pp^-1 (g_p - H_pc s_cam), reading s_cam from a copy (w is free now).
        ctx.enqueue_copy(dst_buf=d_w, src_buf=d_s)
        ctx.enqueue_function[gather_backsubstitute](
            lin.poses, lin.rigs, lin.intrinsics, lin.points, t_pose_off, t_rig_off, t_calib_off, t_point_off, point_start, by_point_frame,
            by_point_camera, w.as_imm(), g.as_imm(), s, hblk, n6, ncal, diag, grid_dim=_grid(np * POINT_LANES), block_dim=BLOCK,
        )
        clock.tick(ctx, "point_backsubstitute")
        # Retract into the trial state (with s.(g + lambda D s) fused), then score it.
        var start = current.views()
        var into = trial.outs()
        ctx.enqueue_function[retract](
            start.poses, start.rigs, start.intrinsics, start.points, into.poses, into.rigs, into.intrinsics, into.points,
            t_pose_off, t_rig_off, t_calib_off, t_point_off, s.as_imm(), g.as_imm(), hblk, n6, ncal, diag, scalars, Int32(SLOT_PRED),
            grid_dim=_grid(nf + nc + np), block_dim=BLOCK,
        )
        clock.tick(ctx, "retract")
        var tried = trial.views()
        ctx.enqueue_function[score](
            tried.poses, tried.rigs, tried.intrinsics, tried.points,
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
            swap(current, trial)
            linearised_at = linearised_at.swapped()
        else:
            diag = diag * up_scale
            up_scale *= 2.0
        if collect_trace:
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
    current.download(ctx, result)
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
    collect_trace: Bool,
    linearize_hook: Optional[LinearizeOut],
) raises -> Stats:
    match (_any(problem.intrinsics_free, problem.n_cameras), _any(problem.sensor_free, problem.n_cameras)):
        case (True, True):
            return solve[Model, True, True](ctx, problem, options, result, collect_trace, linearize_hook)
        case (True, False):
            return solve[Model, True, False](ctx, problem, options, result, collect_trace, linearize_hook)
        case (False, True):
            return solve[Model, False, True](ctx, problem, options, result, collect_trace, linearize_hook)
        case (False, False):
            return solve[Model, False, False](ctx, problem, options, result, collect_trace, linearize_hook)


def supports_model(model: Int) -> Bool:
    """Whether `solve_model` has a specialisation for this `CameraModel` value (same arms)."""
    match model:
        case 0 | 1 | 2 | 3:
            return True
        case _:
            return False


def _dispatch(
    ctx: DeviceContext,
    model: Int,
    problem: HostProblem,
    options: Options,
    result: HostState,
    collect_trace: Bool,
    linearize_hook: Optional[LinearizeOut],
) raises -> Stats:
    match model:
        case 0:
            return _solve_flags[Pinhole](ctx, problem, options, result, collect_trace, linearize_hook)
        case 1:
            return _solve_flags[SimpleRadial](ctx, problem, options, result, collect_trace, linearize_hook)
        case 2:
            return _solve_flags[OpenCVFisheye](ctx, problem, options, result, collect_trace, linearize_hook)
        case 3:
            return _solve_flags[FullOpenCV](ctx, problem, options, result, collect_trace, linearize_hook)
        case _:
            raise Error(t"unknown camera model {model}")


def calib_width(model: Int) raises -> Int:
    """Tangent width of a calibration block (`CameraModel.NUM_CALIB`), through the same arms as `_dispatch`."""
    match model:
        case 0:
            return Pinhole.NUM_CALIB
        case 1:
            return SimpleRadial.NUM_CALIB
        case 2:
            return OpenCVFisheye.NUM_CALIB
        case 3:
            return FullOpenCV.NUM_CALIB
        case _:
            raise Error(t"unknown camera model {model}")


def packed_groups(model: Int, problem: HostProblem) raises -> List[Tuple[Int, Int]]:
    """(free blocks, block width) per group in the packed order of `blocks.mojo`: poses, sensors, calibrations, points."""
    var counts = GroupCounts(problem)
    return [(counts.poses, FRAME_WIDTH), (counts.sensors, FRAME_WIDTH), (counts.calibs, calib_width(model)), (counts.points, POINT_WIDTH)]


def solve_model(ctx: DeviceContext, model: Int, problem: HostProblem, options: Options, result: HostState, collect_trace: Bool) raises -> Stats:
    """`solve` for a runtime camera model (the `CameraModel` value of `caspar_mojo.problem`) and the refined groups the
    masks free; `collect_trace` fills `Stats.trace` (the C door does not read it)."""
    return _dispatch(ctx, model, problem, options, result, collect_trace, Optional[LinearizeOut](None))


def linearize_model(ctx: DeviceContext, model: Int, problem: HostProblem, hook: LinearizeOut) raises -> Float32:
    """Test hook: one linearisation at the problem's state (CASPAR's `res_jac_first`) into `hook`; returns the cost."""
    var options = Options(1, 1.0, 0.0, 2.0, 0.333333, 1e3, 0.0, 1.0, 1, 0.1)
    var unused = HostState(hook.gradient, hook.gradient, hook.gradient, hook.gradient)
    return _dispatch(ctx, model, problem, options, unused, False, hook).initial_cost
