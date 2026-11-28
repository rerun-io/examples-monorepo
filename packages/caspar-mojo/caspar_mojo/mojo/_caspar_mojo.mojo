"""CPython entry point (`caspar_mojo._caspar_mojo`): one call per solve.

Arrays cross as `(address, *shape)` tuples of contiguous numpy arrays (f32 state, i32 indices, u8 masks), built
and checked by `caspar_mojo.solver`. The solver copies them to device buffers it owns and writes the result
into the output arrays.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from solver import F32HostIn, F32HostOut, HostProblem, HostState, I32HostIn, Options, Stats, U8HostIn, solve_model


def _f32_in(spec: PythonObject) raises -> F32HostIn:
    return F32HostIn(unsafe_from_address=Int(py=spec[0]))


def _i32_in(spec: PythonObject) raises -> I32HostIn:
    return I32HostIn(unsafe_from_address=Int(py=spec[0]))


def _u8_in(spec: PythonObject) raises -> U8HostIn:
    return U8HostIn(unsafe_from_address=Int(py=spec[0]))


def _f32_out(spec: PythonObject) raises -> F32HostOut:
    return F32HostOut(unsafe_from_address=Int(py=spec[0]))


def _problem(inputs: PythonObject) raises -> HostProblem:
    return HostProblem(
        n_frames=Int(py=inputs[5][1]),
        n_cameras=Int(py=inputs[1][1]),
        n_points=Int(py=inputs[7][1]),
        n_obs=Int(py=inputs[9][1]),
        intrinsics=_f32_in(inputs[1]),
        intrinsics_free=_u8_in(inputs[2]),
        sensor_from_rig=_f32_in(inputs[3]),
        sensor_free=_u8_in(inputs[4]),
        rig_from_world=_f32_in(inputs[5]),
        pose_free=_u8_in(inputs[6]),
        points=_f32_in(inputs[7]),
        point_free=_u8_in(inputs[8]),
        pixels=_f32_in(inputs[9]),
        obs_frame=_i32_in(inputs[10]),
        obs_camera=_i32_in(inputs[11]),
        obs_point=_i32_in(inputs[12]),
    )


def _run(
    inputs: PythonObject, options: Options, result: HostState, gradient: F32HostOut, hblk: F32HostOut, linearize_only: Bool
) raises -> Stats:
    return solve_model(DeviceContext(), Int(py=inputs[0]), _problem(inputs), options, result, gradient, hblk, linearize_only)


def _stats(stats: Stats) raises -> PythonObject:
    var trace = Python.list()
    for t in stats.trace:
        trace.append(Python.tuple(Float64(t.cost_tried), Float64(t.cost_best), Float64(t.damping), t.pcg_iterations, t.accepted))
    return Python.tuple(
        Float64(stats.initial_cost),
        Float64(stats.final_cost),
        stats.iterations,
        stats.pcg_iterations,
        stats.exit_reason,
        stats.solve_seconds,
        stats.launches,
        trace,
    )


def py_solve(inputs: PythonObject, options: PythonObject, outputs: PythonObject) raises -> PythonObject:
    """`solve(inputs, options, outputs) -> (initial_cost, final_cost, iterations, pcg_iterations, exit, seconds, launches,
    trace)`, trace = [(cost_tried, cost_best, damping, pcg_iterations, accepted), ...]."""
    var opts = Options(
        solver_iter_max=Int(py=options[0]),
        diag_init=Float32(py=options[1]),
        diag_min=Float32(py=options[2]),
        diag_scaling_up=Float32(py=options[3]),
        diag_scaling_down=Float32(py=options[4]),
        diag_exit_value=Float32(py=options[5]),
        score_exit_value=Float32(py=options[6]),
        solver_rel_decrease_min=Float32(py=options[7]),
        reduced_pcg_iter_max=Int(py=options[8]),
        reduced_pcg_eta=Float64(py=options[9]),
    )
    var result = HostState(_f32_out(outputs[0]), _f32_out(outputs[1]), _f32_out(outputs[2]), _f32_out(outputs[3]))
    var unused = result.points
    return _stats(_run(inputs, opts, result, unused, unused, False))


def py_linearize(inputs: PythonObject, gradient: PythonObject, hblk: PythonObject) raises -> PythonObject:
    """`linearize(inputs, gradient, hblk) -> cost`: one linearisation at the given state, outputs written in place."""
    var opts = Options(1, 1.0, 0.0, 2.0, 0.333333, 1e3, 0.0, 1.0, 1, 0.1)
    var none = _f32_out(gradient)
    var result = HostState(none, none, none, none)
    return PythonObject(Float64(_run(inputs, opts, result, _f32_out(gradient), _f32_out(hblk), True).initial_cost))


@export
def PyInit__caspar_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_caspar_mojo")
        m.def_function[py_solve]("solve")
        m.def_function[py_linearize]("linearize")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
