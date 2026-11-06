"""Test-only Python extension: the scalar math of `lie_math.mojo` on CPU and GPU.

`tests/test_math.py` builds it with `-I lietorch_mojo/mojo`; it is not part of
the production extension. `probe_cpu` / `probe_cuda` map x (n, 1) to
(n, 4) = [sin, cos, atan, sqrt(|x|)] in x's dtype; `x` and `out` are
`data_ptr()`s of contiguous tensors on that device. The CUDA path runs on its
own stream and synchronises before it returns.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from layout import TileTensor

from launch import InRows, NUM_THREADS, OutRows, in_rows, out_rows
from lie_math import atan_, sincos_, sqrt_


@always_inline
def probe_row[dtype: DType](i: Int, x: TileTensor[mut=False, dtype, ...], y: TileTensor[mut=True, dtype, ...]):
    comptime assert x.flat_rank == 2 and y.flat_rank == 2, "expected (batch, width) tensors"
    var v = x.load[width=1]((i, 0))[0]
    var sc = sincos_(v)
    y[i, 0] = sc[0]
    y[i, 1] = sc[1]
    y[i, 2] = atan_(v)
    y[i, 3] = sqrt_(abs(v))


def probe_kernel[dtype: DType](x: InRows[dtype, 1], y: OutRows[dtype, 4], n: Int):
    var i = global_idx.x
    if i < n:
        probe_row[dtype](i, x, y)


def _probe_cuda[dtype: DType](x_addr: PythonObject, y_addr: PythonObject, n: Int) raises:
    var ctx = DeviceContext()
    var func = ctx.compile_function[probe_kernel[dtype]]()
    ctx.enqueue_function(
        func, in_rows[dtype, 1](x_addr, n), out_rows[dtype, 4](y_addr, n), n, grid_dim=ceildiv(n, NUM_THREADS), block_dim=NUM_THREADS
    )
    ctx.synchronize()


def _probe_cpu[dtype: DType](x_addr: PythonObject, y_addr: PythonObject, n: Int) raises:
    var x = in_rows[dtype, 1](x_addr, n)
    var y = out_rows[dtype, 4](y_addr, n)
    for i in range(n):
        probe_row[dtype](i, x, y)


def probe_cuda_py(is_f64: PythonObject, n_obj: PythonObject, x_addr: PythonObject, y_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=n_obj)
    if n > 0:
        if Bool(py=is_f64):
            _probe_cuda[DType.float64](x_addr, y_addr, n)
        else:
            _probe_cuda[DType.float32](x_addr, y_addr, n)
    return Python.none()


def probe_cpu_py(is_f64: PythonObject, n_obj: PythonObject, x_addr: PythonObject, y_addr: PythonObject) raises -> PythonObject:
    var n = Int(py=n_obj)
    if n > 0:
        if Bool(py=is_f64):
            _probe_cpu[DType.float64](x_addr, y_addr, n)
        else:
            _probe_cpu[DType.float32](x_addr, y_addr, n)
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
