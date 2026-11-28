# Python-loadable GPU smoke: proves a Mojo DeviceContext and kernel run inside the same process as CASPAR (pycolmap).
from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, ImmTileTensor, MutTileTensor, row_major

comptime BLOCK = 256
comptime L1 = type_of(row_major((0,)))


def axpy(
    a: Float32,
    x: ImmTileTensor[.float32, L1, ImmUntrackedOrigin],
    y: MutTileTensor[.float32, L1, MutUntrackedOrigin],
    n: Int32,
):
    comptime assert x.flat_rank == 1 and y.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        y[i] = a * x[i] + y[i]


def axpy_max_error(py_n: PythonObject) raises -> PythonObject:
    """Runs y = 2 x + 1 on the GPU and returns (device name, max abs error)."""
    var n = Int(py=py_n)
    var ctx = DeviceContext()
    var x_buf = ctx.enqueue_create_buffer[.float32](n)
    var y_buf = ctx.enqueue_create_buffer[.float32](n)
    with x_buf.map_to_host() as xh, y_buf.map_to_host() as yh:
        for i in range(n):
            xh[i] = Float32(i)
            yh[i] = 1.0
    ctx.enqueue_function[axpy](
        Float32(2.0), TileTensor(x_buf, row_major(n)).as_imm(), TileTensor(y_buf, row_major(n)), Int32(n),
        grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK,
    )
    var err: Float32 = 0.0
    with y_buf.map_to_host() as yh:
        for i in range(n):
            err = max(err, abs(yh[i] - (2.0 * Float32(i) + 1.0)))
    return Python.tuple(ctx.device().get_name(), Float64(err))


@export
def PyInit_mojo_smoke_ext() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_smoke_ext")
        m.def_function[axpy_max_error]("axpy_max_error")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
