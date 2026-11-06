"""Test-only Python extension: the scalar math the group code uses, on CPU and GPU.

`tests/test_math.py` builds it with `-I lietorch_mojo/mojo`; it is not part of
the production extension. `probe_cpu` / `probe_cuda` map x (n, 1) to
(n, 4) = [sin, cos, atan, sqrt(|x|)] in x's dtype: `sincos_` and `sqrt_` from
`lie_math.mojo`, `atan` from `std.math`. `x` and `out` are `data_ptr()`s of
contiguous tensors on that device. The CUDA path runs on its own stream and
synchronises before it returns.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from max.gpu.host import DeviceContext
from std.math import atan, ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.utils.static_tuple import StaticTuple

from layout import ImmTileTensor, MutTileTensor

from launch import NUM_THREADS, InRows, OutRows, in_rows, out_rows
from lie_math import sincos_, sqrt_


@inline(.always)
def probe_row[dtype: DType, //](i: Int, x: ImmTileTensor[dtype, ...], y: MutTileTensor[dtype, ...]):
    comptime assert x.flat_rank == 2 and y.flat_rank == 2, "expected (batch, width) tensors"
    comptime assert dtype.is_floating_point()
    var v = x.load((i, 0))[0]
    var s, c = sincos_(v)
    y[i, 0] = s
    y[i, 1] = c
    y[i, 2] = atan(v)
    y[i, 3] = sqrt_(abs(v))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def probe_kernel[dtype: DType](x: InRows[dtype, 1], y: OutRows[dtype, 4]):
    var i = global_idx.x
    if i < Int(x.dim[0]()):
        probe_row(i, x, y)


def _probe_cuda[dtype: DType](x_addr: PythonObject, y_addr: PythonObject, n: Int) raises:
    var ctx = DeviceContext()
    ctx.enqueue_function[probe_kernel[dtype]](
        in_rows[dtype, 1](x_addr, n), out_rows[dtype, 4](y_addr, n), grid_dim=ceildiv(n, NUM_THREADS), block_dim=NUM_THREADS
    )
    ctx.synchronize()


def _probe_cpu[dtype: DType](x_addr: PythonObject, y_addr: PythonObject, n: Int) raises:
    var x = in_rows[dtype, 1](x_addr, n)
    var y = out_rows[dtype, 4](y_addr, n)
    for i in range(n):
        probe_row(i, x, y)


def probe_cuda_py(is_f64: PythonObject, n_obj: PythonObject, x_addr: PythonObject, y_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=n_obj)
    if n > 0:
        if Bool(py=is_f64):
            _probe_cuda[.float64](x_addr, y_addr, n)
        else:
            _probe_cuda[.float32](x_addr, y_addr, n)
    return Python.none()


def probe_cpu_py(is_f64: PythonObject, n_obj: PythonObject, x_addr: PythonObject, y_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=n_obj)
    if n > 0:
        if Bool(py=is_f64):
            _probe_cpu[.float64](x_addr, y_addr, n)
        else:
            _probe_cpu[.float32](x_addr, y_addr, n)
    return Python.none()


@export
def PyInit_math_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("math_probe")
        m.def_function[probe_cuda_py]("probe_cuda")
        m.def_function[probe_cpu_py]("probe_cpu")
        return m.finalize()
    except e:
        abort(t"failed to create math_probe: {e}")
