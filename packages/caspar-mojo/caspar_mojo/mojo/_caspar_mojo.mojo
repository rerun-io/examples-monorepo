from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import ImmTileTensor, MutTileTensor, TileTensor, row_major

comptime L1 = type_of(row_major((0,)))


def scale_kernel(
    src: ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin],
    dst: MutTileTensor[DType.float32, L1, MutUntrackedOrigin],
    factor: Float32,
):
    comptime assert src.flat_rank == 1 and dst.flat_rank == 1
    var i = global_idx.x
    if i < Int(src.dim[0]()):
        dst[i] = src[i] * factor


def smoke(spec: PythonObject, factor: PythonObject) raises -> PythonObject:
    var n = Int(py=spec[1])
    var host = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](n)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(a, host)
    var ta = TileTensor(a, row_major((n,)))
    var tb = TileTensor(b, row_major((n,)))
    ctx.enqueue_function[scale_kernel](ta.as_imm(), tb, Float32(py=factor), grid_dim=ceildiv(n, 256), block_dim=256)
    ctx.enqueue_copy(host, b)
    ctx.synchronize()
    return PythonObject(n)


@export
def PyInit__caspar_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_caspar_mojo")
        m.def_function[smoke]("smoke")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
