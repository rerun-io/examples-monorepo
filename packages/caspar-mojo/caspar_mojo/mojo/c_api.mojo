"""C entry point (`libcaspar_mojo.so`, header `include/caspar_mojo.h`): COLMAP's C++ calls the solver without Python.

A thin wrapper over the same `solve` the Python extension calls. Every C struct field is 8 bytes wide, so these
structs match the header field for field. Errors never cross the ABI: they become a return code and a message in
the caller's buffer.
"""

from std.ffi import c_char
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from solver import F32HostIn, F32HostOut, HostProblem, HostState, I32HostIn, Options, Stats, U8HostIn, solve_model, supports_model

comptime ABI_VERSION: Int32 = 1
comptime OK: Int32 = 0
comptime ERROR: Int32 = 1
comptime UNSUPPORTED: Int32 = 2


@fieldwise_init
struct CProblem(TrivialRegisterPassable):
    var camera_model: Int64
    var num_frames: Int64
    var num_cameras: Int64
    var num_points: Int64
    var num_observations: Int64
    var intrinsics: F32HostIn
    var intrinsics_free: U8HostIn
    var sensor_from_rig: F32HostIn
    var sensor_free: U8HostIn
    var rig_from_world: F32HostIn
    var pose_free: U8HostIn
    var points: F32HostIn
    var point_free: U8HostIn
    var pixels: F32HostIn
    var observation_frame: I32HostIn
    var observation_camera: I32HostIn
    var observation_point: I32HostIn


@fieldwise_init
struct COptions(TrivialRegisterPassable):
    var device_id: Int64
    var solver_iter_max: Int64
    var diag_init: Float64
    var diag_min: Float64
    var diag_scaling_up: Float64
    var diag_scaling_down: Float64
    var diag_exit_value: Float64
    var score_exit_value: Float64
    var solver_rel_decrease_min: Float64
    var reduced_pcg_iter_max: Int64
    var reduced_pcg_eta: Float64


@fieldwise_init
struct CState(TrivialRegisterPassable):
    var intrinsics: F32HostOut
    var sensor_from_rig: F32HostOut
    var rig_from_world: F32HostOut
    var points: F32HostOut


@fieldwise_init
struct CStats(TrivialRegisterPassable):
    var initial_cost: Float64
    var final_cost: Float64
    var iterations: Int64
    var pcg_iterations: Int64
    var exit_reason: Int64
    var solve_seconds: Float64
    var total_seconds: Float64
    var launches: Int64


def _set_error(message: String, buffer: MutPointer[c_char, MutUntrackedOrigin], capacity: Int64):
    if capacity <= 0:
        return
    var bytes = message.as_bytes()
    var n = min(len(bytes), Int(capacity) - 1)
    for i in range(n):
        buffer[unsafe_offset=i] = c_char(bytes[i])
    buffer[unsafe_offset=n] = 0


def _host_problem(p: CProblem) -> HostProblem:
    return HostProblem(
        n_frames=Int(p.num_frames),
        n_cameras=Int(p.num_cameras),
        n_points=Int(p.num_points),
        n_obs=Int(p.num_observations),
        intrinsics=p.intrinsics,
        intrinsics_free=p.intrinsics_free,
        sensor_from_rig=p.sensor_from_rig,
        sensor_free=p.sensor_free,
        rig_from_world=p.rig_from_world,
        pose_free=p.pose_free,
        points=p.points,
        point_free=p.point_free,
        pixels=p.pixels,
        obs_frame=p.observation_frame,
        obs_camera=p.observation_camera,
        obs_point=p.observation_point,
    )


def _solve(p: CProblem, o: COptions, s: CState) raises -> Stats:
    var problem = _host_problem(p)
    var options = Options(
        solver_iter_max=Int(o.solver_iter_max),
        diag_init=Float32(o.diag_init),
        diag_min=Float32(o.diag_min),
        diag_scaling_up=Float32(o.diag_scaling_up),
        diag_scaling_down=Float32(o.diag_scaling_down),
        diag_exit_value=Float32(o.diag_exit_value),
        score_exit_value=Float32(o.score_exit_value),
        solver_rel_decrease_min=Float32(o.solver_rel_decrease_min),
        reduced_pcg_iter_max=Int(o.reduced_pcg_iter_max),
        reduced_pcg_eta=o.reduced_pcg_eta,
    )
    var result = HostState(s.intrinsics, s.sensor_from_rig, s.rig_from_world, s.points)
    return solve_model(DeviceContext(device_id=Int(o.device_id)), Int(p.camera_model), problem, options, result, False)


@export
def caspar_mojo_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def caspar_mojo_supports(camera_model: Int64, refine_intrinsics: Int64, refine_rig: Int64) abi("C") -> Int32:
    _ = refine_intrinsics, refine_rig  # every flag combination is compiled for a supported model
    return 1 if supports_model(Int(camera_model)) else 0


@export
def caspar_mojo_solve(
    problem: ImmPointer[CProblem, ImmUntrackedOrigin],
    options: ImmPointer[COptions, ImmUntrackedOrigin],
    result: ImmPointer[CState, ImmUntrackedOrigin],
    stats: MutPointer[CStats, MutUntrackedOrigin],
    error_message: MutPointer[c_char, MutUntrackedOrigin],
    error_capacity: Int64,
) abi("C") -> Int32:
    var start = perf_counter_ns()
    if not supports_model(Int(problem[].camera_model)):
        _set_error(String(t"camera model {problem[].camera_model} is not compiled into this library"), error_message, error_capacity)
        return UNSUPPORTED
    try:
        var out = _solve(problem[], options[], result[])
        stats[] = CStats(
            initial_cost=Float64(out.initial_cost),
            final_cost=Float64(out.final_cost),
            iterations=Int64(out.iterations),
            pcg_iterations=Int64(out.pcg_iterations),
            exit_reason=Int64(out.exit_reason),
            solve_seconds=out.solve_seconds,
            total_seconds=Float64(perf_counter_ns() - start) * 1e-9,
            launches=Int64(out.launches),
        )
        return OK
    except e:
        _set_error(String(e), error_message, error_capacity)
        return ERROR
