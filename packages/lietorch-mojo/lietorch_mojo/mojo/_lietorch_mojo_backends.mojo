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

This file turns the runtime integers into compile-time parameters. Only SE3
is instantiated in this step; the other groups plug in here once they
implement `LieGroup`.
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
from se3 import SE3

comptime GROUP_SE3 = 3


def _check_tensor_count(op: Int, tensors: PythonObject) raises:
    var expected = num_tensors(op)
    var got = len(tensors)
    if got != expected:
        raise Error(t"lietorch_mojo: op {op} takes {expected} tensors, got {got}")


def _launch_cuda[G: LieGroup](mut rt: GPURuntime, stream: Int, op: Int, n: Int, tensors: PythonObject) raises:
    comptime for o in range(NUM_OPS):
        if op == o:
            _check_tensor_count(o, tensors)
            comptime if family(o) == UNARY:
                unary_cuda[G, o](rt, stream, n, tensors)
            elif family(o) == UNARY_BACKWARD:
                unary_backward_cuda[G, o](rt, stream, n, tensors)
            elif family(o) == BINARY:
                binary_cuda[G, o](rt, stream, n, tensors)
            else:
                binary_backward_cuda[G, o](rt, stream, n, tensors)
            return
    raise Error(t"lietorch_mojo: unknown op code {op}")


def _launch_cpu[G: LieGroup](rt: CPURuntime, op: Int, n: Int, tensors: PythonObject) raises:
    comptime for o in range(NUM_OPS):
        if op == o:
            _check_tensor_count(o, tensors)
            comptime if family(o) == UNARY:
                unary_cpu[G, o](rt, n, tensors)
            elif family(o) == UNARY_BACKWARD:
                unary_backward_cpu[G, o](rt, n, tensors)
            elif family(o) == BINARY:
                binary_cpu[G, o](rt, n, tensors)
            else:
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
    """(op, group id, is_float64) of a spec; raises for groups that are not implemented."""
    var spec = Int(py=spec_obj)
    var group_id = (spec >> 8) & 0xFF
    if group_id != GROUP_SE3:
        raise Error(t"lietorch_mojo: group id {group_id} is not implemented")
    return (spec & 0xFF, group_id, ((spec >> 16) & 1) == 1)


def launch_cuda_py(
    runtime: PythonObject, stream_obj: PythonObject, spec_obj: PythonObject, n_obj: PythonObject, tensors: PythonObject
) raises -> PythonObject:
    var op, _, is_f64 = _unpack_spec(spec_obj)
    var n = Int(py=n_obj)
    if n > 0:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var stream = Int(py=stream_obj)
        if is_f64:
            _launch_cuda[SE3[DType.float64]](rt[], stream, op, n, tensors)
        else:
            _launch_cuda[SE3[DType.float32]](rt[], stream, op, n, tensors)
    return Python.none()


def launch_cpu_py(runtime: PythonObject, spec_obj: PythonObject, n_obj: PythonObject, tensors: PythonObject) raises -> PythonObject:
    var op, _, is_f64 = _unpack_spec(spec_obj)
    var n = Int(py=n_obj)
    if n > 0:
        var rt = runtime.downcast_value_ptr[CPURuntime]()
        if is_f64:
            _launch_cpu[SE3[DType.float64]](rt[], op, n, tensors)
        else:
            _launch_cpu[SE3[DType.float32]](rt[], op, n, tensors)
    return Python.none()


def row_widths_py(spec_obj: PythonObject) raises -> PythonObject:
    var op, _, is_f64 = _unpack_spec(spec_obj)
    if is_f64:
        return _row_widths[SE3[DType.float64]](op)
    return _row_widths[SE3[DType.float32]](op)



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
        m.def_function[create_gpu_runtime_py]("create_gpu_runtime")
        m.def_function[create_cpu_runtime_py]("create_cpu_runtime")
        return m.finalize()
    except e:
        abort(t"failed to create _lietorch_mojo_backends: {e}")
