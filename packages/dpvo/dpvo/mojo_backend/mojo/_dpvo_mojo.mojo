"""CPython entry point of DPVO's Mojo kernels (`dpvo.mojo_backend._dpvo_mojo`).

Every op is a pair `<op>_gpu(runtime, scalars, tensors)` / `<op>_cpu(runtime, scalars, tensors)`:
- `runtime`: lietorch-mojo's `GPURuntime` (CUDA) or `CPURuntime` object, from
  `lietorch_mojo.backends._runtime`. This module is built against the same Mojo structs
  (`-I <lietorch-mojo>/lietorch_mojo/mojo`) but must not register them as Python types again: a second
  registration of one Mojo type aborts the import. So both extensions share one runtime per device.
- `scalars`: a tuple, first the raw CUDA stream for GPU calls (see launch.mojo), then the
  op's integer or float arguments.
- `tensors`: a tuple of `(address, *shape)` tuples, contiguous tensors in the op's order: `data_ptr()`
  on CPU and CUDA. `dpvo.mojo_backend._launch`
  builds them and owns the dtype and shape checks.

GPU calls are asynchronous (PyTorch's current CUDA stream).
"""

from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from layout import ImmTileTensor, MutTileTensor, TileTensor, row_major

from ba import (
    F32Ptr,
    L2,
    eet_element,
    eet_kernel,
    etv_element,
    etv_kernel,
    ev_element,
    ev_kernel,
    depth_prior_element,
    depth_prior_kernel,
    hessian_element,
    hessian_kernel,
    patch_retr_element,
    patch_retr_kernel,
    pose_retr_element,
    pose_retr_kernel,
    reproject_element,
    reproject_kernel,
)
from corr import L1, L3, L4, L5, L6, NUM_THREADS, corr_element, corr_kernel, patchify_element, patchify_kernel
from launch import CPURuntime, GPURuntime
from pgo import (
    pgo_apply_element,
    pgo_apply_kernel,
    pgo_blocks_element,
    pgo_blocks_kernel,
    pgo_chain_factor,
    pgo_chain_factor_kernel,
    pgo_chain_solve,
    pgo_chain_solve_kernel,
)
from projective import flow_magnitude_element, flow_magnitude_kernel

# ── tensor views from (address, *shape) tuples ──────────────────────────────


@always_inline
def _dim(spec: PythonObject, axis: Int) raises -> Int:
    return Int(py=spec[axis + 1])


@always_inline
def _imm[dtype: DType](spec: PythonObject) raises -> ImmPointer[Scalar[dtype], ImmUntrackedOrigin]:
    return ImmPointer[Scalar[dtype], ImmUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))


@always_inline
def _mut[dtype: DType](spec: PythonObject) raises -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    return MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))


@always_inline
def in1[dtype: DType](s: PythonObject) raises -> ImmTileTensor[dtype, L1, ImmUntrackedOrigin]:
    return TileTensor(_imm[dtype](s), row_major((_dim(s, 0),)))


@always_inline
def in2[dtype: DType](s: PythonObject) raises -> ImmTileTensor[dtype, L2, ImmUntrackedOrigin]:
    return TileTensor(_imm[dtype](s), row_major((_dim(s, 0), _dim(s, 1))))


@always_inline
def in3[dtype: DType](s: PythonObject) raises -> ImmTileTensor[dtype, L3, ImmUntrackedOrigin]:
    return TileTensor(_imm[dtype](s), row_major((_dim(s, 0), _dim(s, 1), _dim(s, 2))))


@always_inline
def in3f(s: PythonObject) raises -> ImmTileTensor[DType.float32, L3, ImmUntrackedOrigin]:
    return in3[DType.float32](s)


@always_inline
def in4[dtype: DType](s: PythonObject) raises -> ImmTileTensor[dtype, L4, ImmUntrackedOrigin]:
    return TileTensor(_imm[dtype](s), row_major((_dim(s, 0), _dim(s, 1), _dim(s, 2), _dim(s, 3))))


@always_inline
def in5[dtype: DType](s: PythonObject) raises -> ImmTileTensor[dtype, L5, ImmUntrackedOrigin]:
    return TileTensor(_imm[dtype](s), row_major((_dim(s, 0), _dim(s, 1), _dim(s, 2), _dim(s, 3), _dim(s, 4))))


@always_inline
def out1[dtype: DType](s: PythonObject) raises -> MutTileTensor[dtype, L1, MutUntrackedOrigin]:
    return TileTensor(_mut[dtype](s), row_major((_dim(s, 0),)))


@always_inline
def out2[dtype: DType](s: PythonObject) raises -> MutTileTensor[dtype, L2, MutUntrackedOrigin]:
    return TileTensor(_mut[dtype](s), row_major((_dim(s, 0), _dim(s, 1))))


@always_inline
def out3[dtype: DType](s: PythonObject) raises -> MutTileTensor[dtype, L3, MutUntrackedOrigin]:
    return TileTensor(_mut[dtype](s), row_major((_dim(s, 0), _dim(s, 1), _dim(s, 2))))


@always_inline
def out4[dtype: DType](s: PythonObject) raises -> MutTileTensor[dtype, L4, MutUntrackedOrigin]:
    return TileTensor(_mut[dtype](s), row_major((_dim(s, 0), _dim(s, 1), _dim(s, 2), _dim(s, 3))))


@always_inline
def out5[dtype: DType](s: PythonObject) raises -> MutTileTensor[dtype, L5, MutUntrackedOrigin]:
    return TileTensor(_mut[dtype](s), row_major((_dim(s, 0), _dim(s, 1), _dim(s, 2), _dim(s, 3), _dim(s, 4))))


@always_inline
def out6[dtype: DType](s: PythonObject) raises -> MutTileTensor[dtype, L6, MutUntrackedOrigin]:
    return TileTensor(_mut[dtype](s), row_major((_dim(s, 0), _dim(s, 1), _dim(s, 2), _dim(s, 3), _dim(s, 4), _dim(s, 5))))


@always_inline
def _f32(s: PythonObject) raises -> F32Ptr:
    return _mut[DType.float32](s)


@always_inline
def _numel(s: PythonObject) raises -> Int:
    var n = 1
    for axis in range(len(s) - 1):
        n *= _dim(s, axis)
    return n


# ── correlation and patchify ─────────────────────────────────────────────────


def _corr_gpu[dtype: DType](mut rt: GPURuntime, stream: Int, radius: Int, t: PythonObject) raises:
    var out = out6[dtype](t[5])
    var total = _numel(t[5])
    var threads = total
    comptime if dtype == DType.float16:
        threads //= 2  # CUDA half2 handles two neighboring search columns per thread.
    var func = rt.ctx.compile_function[corr_kernel[dtype]]()
    rt.torch_stream(stream).enqueue_function(
        func,
        Int64(total),
        Int64(radius),
        in5[dtype](t[0]),
        in5[dtype](t[1]),
        in5[DType.float32](t[2]),
        in1[DType.int64](t[3]),
        in1[DType.int64](t[4]),
        out,
        grid_dim=ceildiv(threads, NUM_THREADS),
        block_dim=NUM_THREADS,
    )


def _corr_cpu[dtype: DType](rt: CPURuntime, radius: Int, t: PythonObject) raises:
    var fmap1 = in5[dtype](t[0])
    var fmap2 = in5[dtype](t[1])
    var coords = in5[DType.float32](t[2])
    var us = in1[DType.int64](t[3])
    var vs = in1[DType.int64](t[4])
    var out = out6[dtype](t[5])

    def element(n: Int) {imm}:
        corr_element(n, radius, fmap1, fmap2, coords, us, vs, out)

    rt.for_each_row(element, _numel(t[5]))


def corr_gpu_py(runtime: PythonObject, scalars: PythonObject, tensors: PythonObject) raises -> PythonObject:
    """scalars = (stream, is_half, radius); tensors = (fmap1, fmap2, coords, us, vs, corr)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    if Bool(py=scalars[1]):
        _corr_gpu[DType.float16](rt[], Int(py=scalars[0]), Int(py=scalars[2]), tensors)
    else:
        _corr_gpu[DType.float32](rt[], Int(py=scalars[0]), Int(py=scalars[2]), tensors)
    return Python.none()


def corr_cpu_py(runtime: PythonObject, scalars: PythonObject, tensors: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    if Bool(py=scalars[1]):
        _corr_cpu[DType.float16](rt[], Int(py=scalars[2]), tensors)
    else:
        _corr_cpu[DType.float32](rt[], Int(py=scalars[2]), tensors)
    return Python.none()


def _patchify_gpu[dtype: DType](mut rt: GPURuntime, stream: Int, radius: Int, t: PythonObject) raises:
    var d = 2 * radius + 2
    var total = _dim(t[1], 0) * _dim(t[1], 1) * d * d
    var func = rt.ctx.compile_function[patchify_kernel[dtype]]()
    rt.torch_stream(stream).enqueue_function(
        func,
        Int64(total),
        Int64(radius),
        in4[dtype](t[0]),
        in3[DType.float32](t[1]),
        out5[dtype](t[2]),
        grid_dim=ceildiv(total, NUM_THREADS),
        block_dim=NUM_THREADS,
    )


def _patchify_cpu[dtype: DType](rt: CPURuntime, radius: Int, t: PythonObject) raises:
    var d = 2 * radius + 2
    var net = in4[dtype](t[0])
    var coords = in3[DType.float32](t[1])
    var patches = out5[dtype](t[2])

    def element(n: Int) {imm}:
        patchify_element(n, radius, net, coords, patches)

    rt.for_each_row(element, _dim(t[1], 0) * _dim(t[1], 1) * d * d)


def patchify_gpu_py(runtime: PythonObject, scalars: PythonObject, tensors: PythonObject) raises -> PythonObject:
    """scalars = (stream, is_half, radius); tensors = (net, coords, patches)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    if Bool(py=scalars[1]):
        _patchify_gpu[DType.float16](rt[], Int(py=scalars[0]), Int(py=scalars[2]), tensors)
    else:
        _patchify_gpu[DType.float32](rt[], Int(py=scalars[0]), Int(py=scalars[2]), tensors)
    return Python.none()


def patchify_cpu_py(runtime: PythonObject, scalars: PythonObject, tensors: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    if Bool(py=scalars[1]):
        _patchify_cpu[DType.float16](rt[], Int(py=scalars[2]), tensors)
    else:
        _patchify_cpu[DType.float32](rt[], Int(py=scalars[2]), tensors)
    return Python.none()


# ── bundle adjustment ────────────────────────────────────────────────────────


def flow_magnitude_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """Scalars = (stream, beta); tensors = (poses, patches, intrinsics, source, target, patch, flow)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[3], 0)
    var func = rt[].ctx.compile_function[flow_magnitude_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        Float32(Float64(py=scalars[1])),
        in2[DType.float32](t[0]),
        in4[DType.float32](t[1]),
        in2[DType.float32](t[2]),
        in1[DType.int64](t[3]),
        in1[DType.int64](t[4]),
        in1[DType.int64](t[5]),
        out3[DType.float32](t[6]),
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )
    return Python.none()


def flow_magnitude_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var beta = Float32(Float64(py=scalars[1]))
    var poses = in2[DType.float32](t[0])
    var patches = in4[DType.float32](t[1])
    var intrinsics = in2[DType.float32](t[2])
    var source = in1[DType.int64](t[3])
    var target = in1[DType.int64](t[4])
    var patch = in1[DType.int64](t[5])
    var flow = out3[DType.float32](t[6])

    def element(n: Int) {imm}:
        flow_magnitude_element(n, beta, poses, patches, intrinsics, source, target, patch, flow)

    rt[].for_each_row(element, _dim(t[3], 0))
    return Python.none()


def reproject_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (poses, patches, intrinsics, ii, jj, kk, coords)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[3], 0)
    var func = rt[].ctx.compile_function[reproject_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        in2[DType.float32](t[0]),
        in4[DType.float32](t[1]),
        in2[DType.float32](t[2]),
        in1[DType.int64](t[3]),
        in1[DType.int64](t[4]),
        in1[DType.int64](t[5]),
        out4[DType.float32](t[6]),
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )
    return Python.none()


def reproject_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var poses = in2[DType.float32](t[0])
    var patches = in4[DType.float32](t[1])
    var intrinsics = in2[DType.float32](t[2])
    var ii = in1[DType.int64](t[3])
    var jj = in1[DType.int64](t[4])
    var kk = in1[DType.int64](t[5])
    var coords = out4[DType.float32](t[6])

    def element(n: Int) {imm}:
        reproject_element(n, poses, patches, intrinsics, ii, jj, kk, coords)

    rt[].for_each_row(element, _dim(t[3], 0))
    return Python.none()


def hessian_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, t0, ppf); tensors = (poses, patches, intrinsics, target, weight, ij_xself, ii, jj, kk, ku, B, E, E_lookup, C, v, u)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[6], 0)
    var func = rt[].ctx.compile_function[hessian_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        Int64(Int(py=scalars[1])),
        Int64(Int(py=scalars[2])),
        Int64(_dim(t[10], 1)),
        Int64(_dim(t[11], 1)),
        in2[DType.float32](t[0]),
        in4[DType.float32](t[1]),
        in2[DType.float32](t[2]),
        in2[DType.float32](t[3]),
        in2[DType.float32](t[4]),
        in2[DType.int64](t[5]),
        in1[DType.int64](t[6]),
        in1[DType.int64](t[7]),
        in1[DType.int64](t[8]),
        in1[DType.int64](t[9]),
        _f32(t[10]),
        _f32(t[11]),
        _f32(t[12]),
        _f32(t[13]),
        _f32(t[14]),
        _f32(t[15]),
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )
    return Python.none()


def hessian_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var t0 = Int(py=scalars[1])
    var ppf = Int(py=scalars[2])
    var b_row = _dim(t[10], 1)
    var e_row = _dim(t[11], 1)
    var poses = in2[DType.float32](t[0])
    var patches = in4[DType.float32](t[1])
    var intrinsics = in2[DType.float32](t[2])
    var target = in2[DType.float32](t[3])
    var weight = in2[DType.float32](t[4])
    var ij_xself = in2[DType.int64](t[5])
    var ii = in1[DType.int64](t[6])
    var jj = in1[DType.int64](t[7])
    var kk = in1[DType.int64](t[8])
    var ku = in1[DType.int64](t[9])
    var b = _f32(t[10])
    var e = _f32(t[11])
    var e_lookup = _f32(t[12])
    var c = _f32(t[13])
    var v = _f32(t[14])
    var u = _f32(t[15])

    def element(n: Int) {imm}:
        hessian_element(n, t0, ppf, poses, patches, intrinsics, target, weight, ij_xself, ii, jj, kk, ku, b, b_row, e, e_row, e_lookup, c, v, u)

    rt[].for_each_row(element, _dim(t[6], 0))
    return Python.none()


def depth_prior_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, lambda, huber); tensors = (patches, prior, prior_weight, kx, C, u)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[3], 0)
    var func = rt[].ctx.compile_function[depth_prior_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        Float32(Float64(py=scalars[1])),
        Float32(Float64(py=scalars[2])),
        in4[DType.float32](t[0]),
        in1[DType.float32](t[1]),
        in1[DType.float32](t[2]),
        in1[DType.int64](t[3]),
        _f32(t[4]),
        _f32(t[5]),
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )
    return Python.none()


def depth_prior_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var lam = Float32(Float64(py=scalars[1]))
    var huber = Float32(Float64(py=scalars[2]))
    var patches = in4[DType.float32](t[0])
    var prior = in1[DType.float32](t[1])
    var prior_weight = in1[DType.float32](t[2])
    var kx = in1[DType.int64](t[3])
    var c = _f32(t[4])
    var u = _f32(t[5])

    def element(n: Int) {imm}:
        depth_prior_element(n, patches, prior, prior_weight, kx, lam, huber, c, u)

    rt[].for_each_row(element, _dim(t[3], 0))
    return Python.none()


def pose_retr_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, t0); tensors = (poses, update (t1 - t0, 6))."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[1], 0)
    var func = rt[].ctx.compile_function[pose_retr_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func, Int64(Int(py=scalars[1])), out2[DType.float32](t[0]), in2[DType.float32](t[1]), grid_dim=ceildiv(n, NUM_THREADS), block_dim=NUM_THREADS
    )
    return Python.none()


def pose_retr_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var t0 = Int(py=scalars[1])
    var poses = out2[DType.float32](t[0])
    var update = in2[DType.float32](t[1])

    def element(i: Int) {imm}:
        pose_retr_element(i, t0, poses, update)

    rt[].for_each_row(element, _dim(t[1], 0))
    return Python.none()


def patch_retr_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (index, patches, update)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[0], 0)
    var func = rt[].ctx.compile_function[patch_retr_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func, in1[DType.int64](t[0]), out4[DType.float32](t[1]), in1[DType.float32](t[2]), grid_dim=ceildiv(n, NUM_THREADS), block_dim=NUM_THREADS
    )
    return Python.none()


def patch_retr_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var index = in1[DType.int64](t[0])
    var patches = out4[DType.float32](t[1])
    var update = in1[DType.float32](t[2])

    def element(n: Int) {imm}:
        patch_retr_element(n, index, patches, update)

    rt[].for_each_row(element, _dim(t[0], 0))
    return Python.none()


# ── block-E products: scalars = (stream, t0, ppf); tensors = (E_lookup, vec_or_Q, index, patch_to_ku, out) ──


def _block_e_gpu[which: Int](runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """which = 0: EEt (index rows (i, j1, j2, ij1, ij2), out (6n, 6n)); 1: Ev (out 6n); 2: Etv (out m)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var t0 = Int64(Int(py=scalars[1]))
    var ppf = Int64(Int(py=scalars[2]))
    var total = _dim(t[2], 0) * Int(ppf)
    var stream = rt[].torch_stream(Int(py=scalars[0]))
    comptime if which == 0:
        var func = rt[].ctx.compile_function[eet_kernel]()
        stream.enqueue_function(
            func, Int64(total), t0, ppf, Int64(_dim(t[4], 1)), in3f(t[0]), in1[DType.float32](t[1]), in2[DType.int64](t[2]), in2[DType.int64](t[3]), _f32(t[4]),
            grid_dim=ceildiv(total, NUM_THREADS), block_dim=NUM_THREADS,
        )
    elif which == 1:
        var func = rt[].ctx.compile_function[ev_kernel]()
        stream.enqueue_function(
            func, Int64(total), t0, ppf, in3f(t[0]), in1[DType.float32](t[1]), in2[DType.int64](t[2]), in2[DType.int64](t[3]), _f32(t[4]),
            grid_dim=ceildiv(total, NUM_THREADS), block_dim=NUM_THREADS,
        )
    else:
        var func = rt[].ctx.compile_function[etv_kernel]()
        stream.enqueue_function(
            func, Int64(total), t0, ppf, in3f(t[0]), in1[DType.float32](t[1]), in2[DType.int64](t[2]), in2[DType.int64](t[3]), _f32(t[4]),
            grid_dim=ceildiv(total, NUM_THREADS), block_dim=NUM_THREADS,
        )
    return Python.none()


def _block_e_cpu[which: Int](runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var t0 = Int(py=scalars[1])
    var ppf = Int(py=scalars[2])
    var e_lookup = in3f(t[0])
    var vec = in1[DType.float32](t[1])
    var index = in2[DType.int64](t[2])
    var patch_to_ku = in2[DType.int64](t[3])
    var out = _f32(t[4])
    var row = _dim(t[4], len(t[4]) - 2)

    def element(n: Int) {imm}:
        comptime if which == 0:
            eet_element(n, t0, ppf, e_lookup, vec, index, patch_to_ku, out, row)
        elif which == 1:
            ev_element(n, t0, ppf, e_lookup, vec, index, patch_to_ku, out)
        else:
            etv_element(n, t0, ppf, e_lookup, vec, index, patch_to_ku, out)

    rt[].for_each_row(element, _dim(t[2], 0) * ppf)
    return Python.none()


# ── loop closure's pose graph (matrix-free normal equations, float64) ──────────


def pgo_blocks_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (Ji, Jj, res, offsets, incident, blocks, grad)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[6], 0)
    var func = rt[].ctx.compile_function[pgo_blocks_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        in3[DType.float64](t[0]),
        in3[DType.float64](t[1]),
        in2[DType.float64](t[2]),
        in1[DType.int64](t[3]),
        in1[DType.int64](t[4]),
        out3[DType.float64](t[5]),
        out2[DType.float64](t[6]),
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )
    return Python.none()


def pgo_blocks_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var Ji = in3[DType.float64](t[0])
    var Jj = in3[DType.float64](t[1])
    var res = in2[DType.float64](t[2])
    var offsets = in1[DType.int64](t[3])
    var incident = in1[DType.int64](t[4])
    var blocks = out3[DType.float64](t[5])
    var grad = out2[DType.float64](t[6])

    def element(k: Int) {imm}:
        pgo_blocks_element(k, Ji, Jj, res, offsets, incident, blocks, grad)

    rt[].for_each_row(element, _dim(t[6], 0))
    return Python.none()


def pgo_apply_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (Ji, Jj, ii, jj, offsets, incident, damp, x, y)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var n = _dim(t[8], 0)
    var func = rt[].ctx.compile_function[pgo_apply_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        in3[DType.float64](t[0]),
        in3[DType.float64](t[1]),
        in1[DType.int64](t[2]),
        in1[DType.int64](t[3]),
        in1[DType.int64](t[4]),
        in1[DType.int64](t[5]),
        in2[DType.float64](t[6]),
        in2[DType.float64](t[7]),
        out2[DType.float64](t[8]),
        grid_dim=ceildiv(n, NUM_THREADS),
        block_dim=NUM_THREADS,
    )
    return Python.none()


def pgo_apply_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    var rt = runtime.downcast_value_ptr[CPURuntime]()
    var Ji = in3[DType.float64](t[0])
    var Jj = in3[DType.float64](t[1])
    var ii = in1[DType.int64](t[2])
    var jj = in1[DType.int64](t[3])
    var offsets = in1[DType.int64](t[4])
    var incident = in1[DType.int64](t[5])
    var damp = in2[DType.float64](t[6])
    var x = in2[DType.float64](t[7])
    var y = out2[DType.float64](t[8])

    def element(k: Int) {imm}:
        pgo_apply_element(k, Ji, Jj, ii, jj, offsets, incident, damp, x, y)

    rt[].for_each_row(element, _dim(t[8], 0))
    return Python.none()


def pgo_chain_factor_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """Scalars = (stream,); tensors = (diag, sub, l_diag, l_sub); one thread (a sequential sweep)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var func = rt[].ctx.compile_function[pgo_chain_factor_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        in3[DType.float64](t[0]),
        in3[DType.float64](t[1]),
        out3[DType.float64](t[2]),
        out3[DType.float64](t[3]),
        grid_dim=1,
        block_dim=1,
    )
    return Python.none()


def pgo_chain_factor_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    pgo_chain_factor(in3[DType.float64](t[0]), in3[DType.float64](t[1]), out3[DType.float64](t[2]), out3[DType.float64](t[3]))
    return Python.none()


def pgo_chain_solve_gpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    """Scalars = (stream,); tensors = (l_diag, l_sub, rhs, out); one thread (a sequential sweep)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var func = rt[].ctx.compile_function[pgo_chain_solve_kernel]()
    rt[].torch_stream(Int(py=scalars[0])).enqueue_function(
        func,
        in3[DType.float64](t[0]),
        in3[DType.float64](t[1]),
        in2[DType.float64](t[2]),
        out2[DType.float64](t[3]),
        grid_dim=1,
        block_dim=1,
    )
    return Python.none()


def pgo_chain_solve_cpu_py(runtime: PythonObject, scalars: PythonObject, t: PythonObject) raises -> PythonObject:
    pgo_chain_solve(in3[DType.float64](t[0]), in3[DType.float64](t[1]), in2[DType.float64](t[2]), out2[DType.float64](t[3]))
    return Python.none()


@export
def PyInit__dpvo_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_dpvo_mojo")
        m.def_function[corr_gpu_py]("corr_gpu")
        m.def_function[corr_cpu_py]("corr_cpu")
        m.def_function[patchify_gpu_py]("patchify_gpu")
        m.def_function[patchify_cpu_py]("patchify_cpu")
        m.def_function[reproject_gpu_py]("reproject_gpu")
        m.def_function[reproject_cpu_py]("reproject_cpu")
        m.def_function[flow_magnitude_gpu_py]("flow_magnitude_gpu")
        m.def_function[flow_magnitude_cpu_py]("flow_magnitude_cpu")
        m.def_function[hessian_gpu_py]("hessian_gpu")
        m.def_function[hessian_cpu_py]("hessian_cpu")
        m.def_function[depth_prior_gpu_py]("depth_prior_gpu")
        m.def_function[depth_prior_cpu_py]("depth_prior_cpu")
        m.def_function[pose_retr_gpu_py]("pose_retr_gpu")
        m.def_function[pose_retr_cpu_py]("pose_retr_cpu")
        m.def_function[patch_retr_gpu_py]("patch_retr_gpu")
        m.def_function[patch_retr_cpu_py]("patch_retr_cpu")
        m.def_function[_block_e_gpu[0]]("eet_gpu")
        m.def_function[_block_e_cpu[0]]("eet_cpu")
        m.def_function[_block_e_gpu[1]]("ev_gpu")
        m.def_function[_block_e_cpu[1]]("ev_cpu")
        m.def_function[_block_e_gpu[2]]("etv_gpu")
        m.def_function[_block_e_cpu[2]]("etv_cpu")
        m.def_function[pgo_blocks_gpu_py]("pgo_blocks_gpu")
        m.def_function[pgo_blocks_cpu_py]("pgo_blocks_cpu")
        m.def_function[pgo_apply_gpu_py]("pgo_apply_gpu")
        m.def_function[pgo_apply_cpu_py]("pgo_apply_cpu")
        m.def_function[pgo_chain_factor_gpu_py]("pgo_chain_factor_gpu")
        m.def_function[pgo_chain_factor_cpu_py]("pgo_chain_factor_cpu")
        m.def_function[pgo_chain_solve_gpu_py]("pgo_chain_solve_gpu")
        m.def_function[pgo_chain_solve_cpu_py]("pgo_chain_solve_cpu")
        return m.finalize()
    except e:
        abort(t"failed to create _dpvo_mojo: {e}")
