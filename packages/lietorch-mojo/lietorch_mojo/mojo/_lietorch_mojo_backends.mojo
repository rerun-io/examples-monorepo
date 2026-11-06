"""CPython entry point of the Mojo backend (`lietorch_mojo._lietorch_mojo_backends`).

Python passes plain integers and the runtime objects created here:

- `launch_cuda(runtime, stream, spec, n, tensors)` and
  `launch_cpu(runtime, spec, n, tensors)` run op `spec` over `n` rows.
  `spec` = op | group_id << 8 | is_float64 << 16 (op codes in
  `group_ops.mojo`; group ids as in lietorch: SO3=1, RxSO3=2, SE3=3, Sim3=4).
  `runtime` is a `GPURuntime` / `CPURuntime`, `stream` PyTorch's current
  `cudaStream_t`, and `tensors` the op's `data_ptr()`s, inputs then outputs.
- `row_widths(spec)` lists the width of each of those tensors; Python checks
  every tensor's shape against it before it launches.

This file turns the runtime integers into compile-time parameters. SE3,
RxSO3 and Sim3 are instantiated; a group plugs in here once it implements
`LieGroup`.

- `pose_retr_cuda(runtime, stream, poses, dx, num_poses, num_fix, is_float64)`
  and `pose_retr_cpu(runtime, poses, dx, num_poses, num_fix, is_float64)`: the
  in-place Sim3 retraction of `pose_retr.mojo` (Python checks the tensors).
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from group_ops import BINARY, NUM_OPS, UNARY, UNARY_BACKWARD, family, num_tensors, row_widths
from launch import (
    CPURuntime,
    GPURuntime,
    binary_backward_cpu,
    binary_backward_cuda,
    binary_cpu,
    binary_cuda,
    unary_backward_cpu,
    unary_backward_cuda,
    unary_cpu,
    unary_cuda,
)
from lie_group import LieGroup
from pose_retr import pose_retr_cpu, pose_retr_cuda
from rxso3 import RxSO3
from se3 import SE3
from sim3 import Sim3

comptime GROUP_RXSO3 = 2
comptime GROUP_SE3 = 3
comptime GROUP_SIM3 = 4


def _check_tensor_count(op: Int, tensors: PythonObject) raises:
    var expected = num_tensors(op)
    var got = len(tensors)
    if got != expected:
        raise Error(t"lietorch_mojo: op {op} takes {expected} tensors, got {got}")


def _launch_cuda[G: LieGroup](mut rt: GPURuntime, stream: Int, op: Int, n: Int, tensors: PythonObject) raises:
    comptime for o in range(NUM_OPS):
        if op == o:
            _check_tensor_count(o, tensors)
            comptime match family(o):
                case == UNARY:
                    unary_cuda[G, o](rt, stream, n, tensors)
                case == UNARY_BACKWARD:
                    unary_backward_cuda[G, o](rt, stream, n, tensors)
                case == BINARY:
                    binary_cuda[G, o](rt, stream, n, tensors)
                case _:
                    binary_backward_cuda[G, o](rt, stream, n, tensors)
            return
    raise Error(t"lietorch_mojo: unknown op code {op}")


def _launch_cpu[G: LieGroup](rt: CPURuntime, op: Int, n: Int, tensors: PythonObject) raises:
    comptime for o in range(NUM_OPS):
        if op == o:
            _check_tensor_count(o, tensors)
            comptime match family(o):
                case == UNARY:
                    unary_cpu[G, o](rt, n, tensors)
                case == UNARY_BACKWARD:
                    unary_backward_cpu[G, o](rt, n, tensors)
                case == BINARY:
                    binary_cpu[G, o](rt, n, tensors)
                case _:
                    binary_backward_cpu[G, o](rt, n, tensors)
            return
    raise Error(t"lietorch_mojo: unknown op code {op}")


def _row_widths[G: LieGroup](op: Int) raises -> PythonObject:
    comptime for o in range(NUM_OPS):
        if op == o:
            var widths = Python.list()
            comptime for k in range(num_tensors(o)):
                comptime width = row_widths[G, o]()[k]
                widths.append(width)
            return widths
    raise Error(t"lietorch_mojo: unknown op code {op}")


def _unpack_spec(spec_obj: PythonObject) raises -> Tuple[Int, Int, Bool]:
    """(group id, op, is_float64) of a spec; raises for groups that are not implemented."""
    var spec = Int(py=spec_obj)
    var group_id = (spec >> 8) & 0xFF
    if group_id != GROUP_SE3 and group_id != GROUP_RXSO3 and group_id != GROUP_SIM3:
        raise Error(t"lietorch_mojo: group id {group_id} is not implemented")
    return (group_id, spec & 0xFF, ((spec >> 16) & 1) == 1)


def launch_cuda_py(
    runtime: PythonObject, stream_obj: PythonObject, spec_obj: PythonObject, n_obj: PythonObject, tensors: PythonObject
) raises -> PythonObject:
    var group_id, op, is_f64 = _unpack_spec(spec_obj)
    var n = Int(py=n_obj)
    if n > 0:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var stream = Int(py=stream_obj)
        if group_id == GROUP_SE3:
            if is_f64:
                _launch_cuda[SE3[.float64]](rt[], stream, op, n, tensors)
            else:
                _launch_cuda[SE3[.float32]](rt[], stream, op, n, tensors)
        elif group_id == GROUP_RXSO3:
            if is_f64:
                _launch_cuda[RxSO3[.float64]](rt[], stream, op, n, tensors)
            else:
                _launch_cuda[RxSO3[.float32]](rt[], stream, op, n, tensors)
        elif group_id == GROUP_SIM3:
            if is_f64:
                _launch_cuda[Sim3[.float64]](rt[], stream, op, n, tensors)
            else:
                _launch_cuda[Sim3[.float32]](rt[], stream, op, n, tensors)
    return Python.none()


def launch_cpu_py(runtime: PythonObject, spec_obj: PythonObject, n_obj: PythonObject, tensors: PythonObject) raises -> PythonObject:
    var group_id, op, is_f64 = _unpack_spec(spec_obj)
    var n = Int(py=n_obj)
    if n > 0:
        var rt = runtime.downcast_value_ptr[CPURuntime]()
        if group_id == GROUP_SE3:
            if is_f64:
                _launch_cpu[SE3[.float64]](rt[], op, n, tensors)
            else:
                _launch_cpu[SE3[.float32]](rt[], op, n, tensors)
        elif group_id == GROUP_RXSO3:
            if is_f64:
                _launch_cpu[RxSO3[.float64]](rt[], op, n, tensors)
            else:
                _launch_cpu[RxSO3[.float32]](rt[], op, n, tensors)
        elif group_id == GROUP_SIM3:
            if is_f64:
                _launch_cpu[Sim3[.float64]](rt[], op, n, tensors)
            else:
                _launch_cpu[Sim3[.float32]](rt[], op, n, tensors)
    return Python.none()


def row_widths_py(spec_obj: PythonObject) raises -> PythonObject:
    var group_id, op, _ = _unpack_spec(spec_obj)
    # Row widths do not depend on the dtype.
    if group_id == GROUP_RXSO3:
        return _row_widths[RxSO3[.float32]](op)
    if group_id == GROUP_SIM3:
        return _row_widths[Sim3[.float32]](op)
    return _row_widths[SE3[.float32]](op)


def pose_retr_cuda_py(
    runtime: PythonObject,
    stream_obj: PythonObject,
    poses: PythonObject,
    dx: PythonObject,
    num_poses_obj: PythonObject,
    num_fix_obj: PythonObject,
    is_f64_obj: PythonObject,
) raises -> PythonObject:
    """Sim3 pose retraction in place on the GPU (lietorch_mojo.pose_retr checked the tensors)."""
    var num_poses = Int(py=num_poses_obj)
    var num_fix = Int(py=num_fix_obj)
    if num_fix < 0:
        raise Error(t"lietorch_mojo: num_fix must be >= 0, got {num_fix}")
    if num_fix < num_poses:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var stream = Int(py=stream_obj)
        if Bool(py=is_f64_obj):
            pose_retr_cuda[.float64](rt[], stream, poses, dx, num_poses, num_fix)
        else:
            pose_retr_cuda[.float32](rt[], stream, poses, dx, num_poses, num_fix)
    return Python.none()


def pose_retr_cpu_py(
    runtime: PythonObject, poses: PythonObject, dx: PythonObject, num_poses_obj: PythonObject, num_fix_obj: PythonObject, is_f64_obj: PythonObject
) raises -> PythonObject:
    """Sim3 pose retraction in place on the CPU."""
    var num_poses = Int(py=num_poses_obj)
    var num_fix = Int(py=num_fix_obj)
    if num_fix < 0:
        raise Error(t"lietorch_mojo: num_fix must be >= 0, got {num_fix}")
    if num_fix < num_poses:
        var rt = runtime.downcast_value_ptr[CPURuntime]()
        if Bool(py=is_f64_obj):
            pose_retr_cpu[.float64](rt[], poses, dx, num_poses, num_fix)
        else:
            pose_retr_cpu[.float32](rt[], poses, dx, num_poses, num_fix)
    return Python.none()


def create_gpu_runtime_py() raises -> PythonObject:
    """The CUDA context of cuda:0 plus the cached PyTorch-stream wrapper (Python creates it on the first CUDA call)."""
    return PythonObject(alloc=GPURuntime())


def create_cpu_runtime_py() raises -> PythonObject:
    """The CPU context of the Mojo thread pool (Python creates it on the first CPU call)."""
    return PythonObject(alloc=CPURuntime())


@export
def PyInit__lietorch_mojo_backends() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_lietorch_mojo_backends")
        _ = m.add_type[GPURuntime]("GPURuntime")
        _ = m.add_type[CPURuntime]("CPURuntime")
        m.def_function[launch_cuda_py]("launch_cuda")
        m.def_function[launch_cpu_py]("launch_cpu")
        m.def_function[row_widths_py]("row_widths")
        m.def_function[pose_retr_cuda_py]("pose_retr_cuda")
        m.def_function[pose_retr_cpu_py]("pose_retr_cpu")
        m.def_function[create_gpu_runtime_py]("create_gpu_runtime")
        m.def_function[create_cpu_runtime_py]("create_cpu_runtime")
        return m.finalize()
    except e:
        abort(t"failed to create _lietorch_mojo_backends: {e}")
