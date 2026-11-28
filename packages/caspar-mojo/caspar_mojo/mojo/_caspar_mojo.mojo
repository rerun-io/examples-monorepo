"""CPython entry point (`caspar_mojo._caspar_mojo`): one call per solve.

Every value is read by name: arrays from the `_Packet` / `_State` dataclasses of `caspar_mojo.solver` (C-contiguous
numpy arrays, f32 state, i32 indices, 0/1 u8 masks, checked there), options from `BaOptions`. A renamed or missing
field raises instead of shifting meaning. The solver copies the arrays to device buffers it owns and writes the
result into the `_State` arrays.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from solver import F32HostIn, F32HostOut, HostProblem, HostState, I32HostIn, LinearizeOut, Options, Stats, U8HostIn, linearize_model, packed_groups, solve_model


def _address(array: PythonObject) raises -> Int:
    return Int(py=array.ctypes.data)


def _rows(array: PythonObject) raises -> Int:
    return Int(py=array.shape[0])


def _f32_out(array: PythonObject) raises -> F32HostOut:
    return F32HostOut(unsafe_from_address=_address(array))


def _problem(packet: PythonObject) raises -> HostProblem:
    return HostProblem(
        n_frames=_rows(packet.rig_from_world),
        n_cameras=_rows(packet.intrinsics),
        n_points=_rows(packet.points),
        n_obs=_rows(packet.pixels),
        intrinsics=F32HostIn(unsafe_from_address=_address(packet.intrinsics)),
        intrinsics_free=U8HostIn(unsafe_from_address=_address(packet.intrinsics_free)),
        sensor_from_rig=F32HostIn(unsafe_from_address=_address(packet.sensor_from_rig)),
        sensor_free=U8HostIn(unsafe_from_address=_address(packet.sensor_free)),
        rig_from_world=F32HostIn(unsafe_from_address=_address(packet.rig_from_world)),
        pose_free=U8HostIn(unsafe_from_address=_address(packet.pose_free)),
        points=F32HostIn(unsafe_from_address=_address(packet.points)),
        point_free=U8HostIn(unsafe_from_address=_address(packet.point_free)),
        pixels=F32HostIn(unsafe_from_address=_address(packet.pixels)),
        obs_frame=I32HostIn(unsafe_from_address=_address(packet.observation_frame)),
        obs_camera=I32HostIn(unsafe_from_address=_address(packet.observation_camera)),
        obs_point=I32HostIn(unsafe_from_address=_address(packet.observation_point)),
    )


def _stats(stats: Stats) raises -> PythonObject:
    var trace = Python.list()
    for t in stats.trace:
        trace.append(Python.tuple(Float64(t.cost_tried), Float64(t.cost_best), Float64(t.damping), t.pcg_iterations, t.accepted))
    return Python.dict(
        initial_cost=PythonObject(Float64(stats.initial_cost)),
        final_cost=PythonObject(Float64(stats.final_cost)),
        iterations=PythonObject(stats.iterations),
        pcg_iterations=PythonObject(stats.pcg_iterations),
        exit_reason=PythonObject(stats.exit_reason),
        solve_seconds=PythonObject(stats.solve_seconds),
        launches=PythonObject(stats.launches),
        trace=trace,
    )


def py_solve(packet: PythonObject, options: PythonObject, result: PythonObject) raises -> PythonObject:
    """`solve(packet, options: BaOptions, result) -> NativeStats` (a dict, `caspar_mojo.solver.NativeStats`)."""
    var opts = Options(
        solver_iter_max=Int(py=options.solver_iter_max),
        diag_init=Float32(py=options.diag_init),
        diag_min=Float32(py=options.diag_min),
        diag_scaling_up=Float32(py=options.diag_scaling_up),
        diag_scaling_down=Float32(py=options.diag_scaling_down),
        diag_exit_value=Float32(py=options.diag_exit_value),
        score_exit_value=Float32(py=options.score_exit_value),
        solver_rel_decrease_min=Float32(py=options.solver_rel_decrease_min),
        reduced_pcg_iter_max=Int(py=options.reduced_pcg_iter_max),
        reduced_pcg_eta=Float64(py=options.reduced_pcg_eta),
    )
    var state = HostState(_f32_out(result.intrinsics), _f32_out(result.sensor_from_rig), _f32_out(result.rig_from_world), _f32_out(result.points))
    return _stats(solve_model(DeviceContext(), Int(py=packet.camera_model), _problem(packet), opts, state, True))


def py_linearize(packet: PythonObject, gradient: PythonObject, hblk: PythonObject) raises -> PythonObject:
    """`linearize(packet, gradient, hblk) -> cost`: one linearisation at the packet's state, outputs written in place."""
    var hook = LinearizeOut(_f32_out(gradient), Int(py=gradient.size), _f32_out(hblk), Int(py=hblk.size))
    return PythonObject(Float64(linearize_model(DeviceContext(), Int(py=packet.camera_model), _problem(packet), hook)))


def py_packed_groups(packet: PythonObject) raises -> PythonObject:
    """`packed_groups(packet) -> [(blocks, width), ...]`: the linearize outputs' groups, in their packed order."""
    var groups = Python.list()
    for group in packed_groups(Int(py=packet.camera_model), _problem(packet)):
        groups.append(Python.tuple(group[0], group[1]))
    return groups


def py_device_count() raises -> PythonObject:
    """`device_count() -> int`: GPUs the runtime sees; the integration tests skip on 0 instead of guessing from errors."""
    return PythonObject(DeviceContext.number_of_devices())


@export
def PyInit__caspar_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_caspar_mojo")
        m.def_function[py_solve]("solve")
        m.def_function[py_linearize]("linearize")
        m.def_function[py_packed_groups]("packed_groups")
        m.def_function[py_device_count]("device_count")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
