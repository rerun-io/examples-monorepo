"""Run one op over a batch: GPU kernels and the CPU loop over the `group_ops` row functions.

The tensors arrive as integer addresses from Python (`backends._run`
checked their shapes against `row_widths`); this file wraps each in a
(batch, width) `TileTensor` and runs the op's family function on every row.

GPU strategy: the kernel runs on PyTorch's current CUDA stream, so it is
ordered with the PyTorch work around it exactly as a native PyTorch op is, and
nothing synchronises. Python passes the raw `cudaStream_t` of
`torch.cuda.current_stream()`; `GPURuntime` wraps it in a non-owning
`DeviceStream` and rebuilds the wrapper only when the handle changes. Outputs
are allocated by PyTorch on that stream, so the caching allocator's
stream-ordered block reuse stays correct.
"""

from std.algorithm import parallelize
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.gpu.host import DeviceContext, DeviceStream
from std.math import ceildiv
from std.python import PythonObject
from std.runtime.asyncrt import parallelism_level
from std.utils.static_tuple import StaticTuple

from layout import Idx, TileTensor, row_major

from group_ops import binary_backward_row, binary_row, row_widths, unary_backward_row, unary_row
from lie_group import LieGroup

comptime NUM_THREADS = 256
# CPU: batches up to this size run on the calling thread; larger ones are
# split into chunks of this size across the Mojo runtime's worker threads.
comptime CPU_CHUNK = 2048

comptime RowLayout[width: Int] = type_of(row_major((0, Idx[width])))
comptime InRows[dtype: DType, width: Int] = TileTensor[dtype, RowLayout[width], ImmutUntrackedOrigin]
comptime OutRows[dtype: DType, width: Int] = TileTensor[dtype, RowLayout[width], MutUntrackedOrigin]


@always_inline
def in_rows[dtype: DType, width: Int](address: PythonObject, n: Int) raises -> InRows[dtype, width]:
    var p = UnsafePointer[Scalar[dtype], ImmutUntrackedOrigin](unsafe_from_address=Int(py=address))
    return TileTensor(p, row_major((n, Idx[width])))


@always_inline
def out_rows[dtype: DType, width: Int](address: PythonObject, n: Int) raises -> OutRows[dtype, width]:
    var p = UnsafePointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=address))
    return TileTensor(p, row_major((n, Idx[width])))


@always_inline
def _stream_ptr(handle: Int) -> Optional[UnsafePointer[NoneType, MutUntrackedOrigin]]:
    """`cudaStream_t` from its integer value; 0 (the legacy default stream) is `None`."""
    if handle == 0:
        return None
    return UnsafePointer[NoneType, MutUntrackedOrigin](unsafe_from_address=handle)


struct GPURuntime(Movable, Writable):
    """Process-lifetime CUDA state: the context of cuda:0 and a wrapper of PyTorch's current stream."""

    var ctx: DeviceContext
    var stream_handle: Int
    var stream: DeviceStream

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.stream_handle = 0
        self.stream = self.ctx.create_external_stream(_stream_ptr(0))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GPURuntime(cuda:0)")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("GPURuntime(cuda:0)")

    @always_inline
    def torch_stream(mut self, handle: Int) raises -> DeviceStream:
        """Non-owning wrapper of the PyTorch stream `handle` (0 = legacy default stream), cached while it repeats."""
        if handle != self.stream_handle:
            self.stream = self.ctx.create_external_stream(_stream_ptr(handle))
            self.stream_handle = handle
        return self.stream


struct CPURuntime(Movable, Writable):
    """Process-lifetime CPU state: the context of the Mojo thread pool."""

    var ctx: DeviceContext

    def __init__(out self) raises:
        self.ctx = DeviceContext(api="cpu")

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CPURuntime")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("CPURuntime")

    def for_each_row[F: def(Int) -> None](self, row: F, n: Int):
        """`row(i)` for i in [0, n): inline for small batches, else in chunks on the thread pool."""
        if n <= CPU_CHUNK:
            for i in range(n):
                row(i)
            return

        def chunk(c: Int) {read}:
            for i in range(c * CPU_CHUNK, min((c + 1) * CPU_CHUNK, n)):
                row(i)

        var chunks = ceildiv(n, CPU_CHUNK)
        parallelize(chunk, chunks, min(chunks, parallelism_level(self.ctx)), self.ctx)


# ── GPU kernels, one per op family ───────────────────────────────────────────


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def unary_kernel[G: LieGroup, op: Int, Wx: Int, Wy: Int](x: InRows[G.dtype, Wx], y: OutRows[G.dtype, Wy], n: Int):
    var i = global_idx.x
    if i < n:
        unary_row[G, op](i, x, y)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def unary_backward_kernel[
    G: LieGroup, op: Int, Wg: Int, Wx: Int
](grad: InRows[G.dtype, Wg], x: InRows[G.dtype, Wx], dx: OutRows[G.dtype, Wx], n: Int):
    var i = global_idx.x
    if i < n:
        unary_backward_row[G, op](i, grad, x, dx)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def binary_kernel[
    G: LieGroup, op: Int, Wb: Int
](X: InRows[G.dtype, G.N], operand: InRows[G.dtype, Wb], y: OutRows[G.dtype, Wb], n: Int):
    var i = global_idx.x
    if i < n:
        binary_row[G, op](i, X, operand, y)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def binary_backward_kernel[
    G: LieGroup, op: Int, Wb: Int
](
    grad: InRows[G.dtype, Wb],
    X: InRows[G.dtype, G.N],
    operand: InRows[G.dtype, Wb],
    dX: OutRows[G.dtype, G.N],
    d_operand: OutRows[G.dtype, Wb],
    n: Int,
):
    var i = global_idx.x
    if i < n:
        binary_backward_row[G, op](i, grad, X, operand, dX, d_operand)


# ── launchers: tensors are the op's addresses, inputs then outputs ───────────


def unary_cuda[G: LieGroup, op: Int](mut rt: GPURuntime, stream: Int, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var func = rt.ctx.compile_function[unary_kernel[G, op, W[0], W[1]]]()  # looks up the cubin embedded at build time
    rt.torch_stream(stream).enqueue_function(
        func,
        in_rows[G.dtype, W[0]](tensors[0], n),
        out_rows[G.dtype, W[1]](tensors[1], n),
        n,
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )


def unary_cpu[G: LieGroup, op: Int](rt: CPURuntime, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var x = in_rows[G.dtype, W[0]](tensors[0], n)
    var y = out_rows[G.dtype, W[1]](tensors[1], n)

    def row(i: Int) {read}:
        unary_row[G, op](i, x, y)

    rt.for_each_row(row, n)


def unary_backward_cuda[G: LieGroup, op: Int](mut rt: GPURuntime, stream: Int, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var func = rt.ctx.compile_function[unary_backward_kernel[G, op, W[0], W[1]]]()
    rt.torch_stream(stream).enqueue_function(
        func,
        in_rows[G.dtype, W[0]](tensors[0], n),
        in_rows[G.dtype, W[1]](tensors[1], n),
        out_rows[G.dtype, W[2]](tensors[2], n),
        n,
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )


def unary_backward_cpu[G: LieGroup, op: Int](rt: CPURuntime, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var grad = in_rows[G.dtype, W[0]](tensors[0], n)
    var x = in_rows[G.dtype, W[1]](tensors[1], n)
    var dx = out_rows[G.dtype, W[2]](tensors[2], n)

    def row(i: Int) {read}:
        unary_backward_row[G, op](i, grad, x, dx)

    rt.for_each_row(row, n)


def binary_cuda[G: LieGroup, op: Int](mut rt: GPURuntime, stream: Int, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var func = rt.ctx.compile_function[binary_kernel[G, op, W[1]]]()
    rt.torch_stream(stream).enqueue_function(
        func,
        in_rows[G.dtype, W[0]](tensors[0], n),
        in_rows[G.dtype, W[1]](tensors[1], n),
        out_rows[G.dtype, W[2]](tensors[2], n),
        n,
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )


def binary_cpu[G: LieGroup, op: Int](rt: CPURuntime, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var X = in_rows[G.dtype, W[0]](tensors[0], n)
    var operand = in_rows[G.dtype, W[1]](tensors[1], n)
    var y = out_rows[G.dtype, W[2]](tensors[2], n)

    def row(i: Int) {read}:
        binary_row[G, op](i, X, operand, y)

    rt.for_each_row(row, n)


def binary_backward_cuda[G: LieGroup, op: Int](mut rt: GPURuntime, stream: Int, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var func = rt.ctx.compile_function[binary_backward_kernel[G, op, W[0]]]()
    rt.torch_stream(stream).enqueue_function(
        func,
        in_rows[G.dtype, W[0]](tensors[0], n),
        in_rows[G.dtype, W[1]](tensors[1], n),
        in_rows[G.dtype, W[2]](tensors[2], n),
        out_rows[G.dtype, W[3]](tensors[3], n),
        out_rows[G.dtype, W[4]](tensors[4], n),
        n,
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )


def binary_backward_cpu[G: LieGroup, op: Int](rt: CPURuntime, n: Int, tensors: PythonObject) raises:
    comptime W = row_widths[G, op]()
    var grad = in_rows[G.dtype, W[0]](tensors[0], n)
    var X = in_rows[G.dtype, W[1]](tensors[1], n)
    var operand = in_rows[G.dtype, W[2]](tensors[2], n)
    var dX = out_rows[G.dtype, W[3]](tensors[3], n)
    var d_operand = out_rows[G.dtype, W[4]](tensors[4], n)

    def row(i: Int) {read}:
        binary_backward_row[G, op](i, grad, X, operand, dX, d_operand)

    rt.for_each_row(row, n)

