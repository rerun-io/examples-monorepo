"""CPython entry point of ViPE's kernels in Mojo (`vipe_mojo._vipe_mojo`).

Every op is a pair `<op>_gpu(runtime, scalars, tensors)` / `<op>_cpu(runtime, scalars, tensors)`, as in
dpvo.mojo_backend:
- `runtime`: lietorch-mojo's `GPURuntime` (CUDA) or `CPURuntime` object, from
  `lietorch_mojo.backends._runtime`. This module is built against the same Mojo structs
  (`-I <lietorch-mojo>/lietorch_mojo/mojo`) but must not register them as Python types again: a second
  registration of one Mojo type aborts the import.
- `scalars`: a tuple, first the raw CUDA stream for GPU calls (0 on CPU), then the op's
  integer or float arguments.
- `tensors`: a tuple of `(address, *shape)` tuples of contiguous tensors in the op's order. `vipe_mojo._launch`
  builds them and the Python wrappers own the dtype and shape checks.

GPU calls are asynchronous (PyTorch's current CUDA stream).
"""

from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from droid import F32Ptr, I64Ptr, NUM_THREADS, altcorr_index_element, altcorr_index_kernel, corr_index_element, corr_index_kernel
from launch import CPURuntime, GPURuntime
from slam import (
    THREADS,
    accum_element,
    accum_kernel,
    ba_partial_element,
    ba_partial_kernel,
    block_reduce_element,
    block_reduce_kernel,
    eet_partial_element,
    eet_partial_kernel,
    ev_partial_element,
    ev_partial_kernel,
    evt_element,
    evt_kernel,
    frame_distance_element,
    frame_distance_kernel,
    pose_retr_element,
    pose_solve,
    pose_retr_kernel,
)


@always_inline
def _dim(spec: PythonObject, axis: Int) raises -> Int:
    return Int(py=spec[axis + 1])


@always_inline
def _ptr[dtype: DType](spec: PythonObject) raises -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    return MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))


@always_inline
def _f(spec: PythonObject) raises -> F32Ptr:
    return _ptr[DType.float32](spec)


@always_inline
def _i(spec: PythonObject) raises -> I64Ptr:
    return _ptr[DType.int64](spec)


@always_inline
def _int(scalars: PythonObject, i: Int) raises -> Int:
    return Int(py=scalars[i])


@always_inline
def _float(scalars: PythonObject, i: Int) raises -> Float32:
    return Float32(Float64(py=scalars[i]))


@always_inline
def _double(scalars: PythonObject, i: Int) raises -> Float64:
    return Float64(py=scalars[i])


@always_inline
def _grid(total: Int) -> Int:
    return max(ceildiv(total, NUM_THREADS), 1)


# ── correlation ──────────────────────────────────────────────────────────────


def _corr_index[dtype: DType, gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises:
    """scalars = (stream, is_half, radius); tensors = (volume (B, H1, W1, H2, W2), coords (B, 2, H1, W1), corr)."""
    var radius = _int(s, 2)
    var h1 = _dim(t[0], 1)
    var w1 = _dim(t[0], 2)
    var h2 = _dim(t[0], 3)
    var w2 = _dim(t[0], 4)
    var volume = _ptr[dtype](t[0])
    var coords = _f(t[1])
    var corr = _ptr[dtype](t[2])
    var total = _dim(t[0], 0) * h1 * w1
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[corr_index_kernel[dtype]]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(
            func, Int64(total), Int64(radius), Int64(h1), Int64(w1), Int64(h2), Int64(w2), volume, coords, corr,
            grid_dim=_grid(total), block_dim=NUM_THREADS,
        )  # fmt: skip
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            corr_index_element(n, radius, h1, w1, h2, w2, volume, coords, corr)

        rt[].for_each_row(element, total)


def _corr_index_py[gpu: Bool](runtime: PythonObject, scalars: PythonObject, tensors: PythonObject) raises -> PythonObject:
    if Bool(py=scalars[1]):
        _corr_index[DType.float16, gpu](runtime, scalars, tensors)
    else:
        _corr_index[DType.float32, gpu](runtime, scalars, tensors)
    return Python.none()


def _altcorr_index[dtype: DType, pix: Bool, gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises:
    """scalars = (stream, is_half, pix, radius, level_offset, coord_scale);
    tensors = (fmap1 (B, N1, Hf, Wf, C), fmap2 (B, N2, H2, W2, C), coords (B, M, H1, W1, 2), ii, jj, corr (B, M, CH, H1, W1)).
    """
    var radius = _int(s, 3)
    var level_offset = _int(s, 4)
    var coord_scale = _float(s, 5)
    var num_edges = _dim(t[2], 1)
    var height1 = _dim(t[2], 2)
    var width1 = _dim(t[2], 3)
    var frames1 = _dim(t[0], 1)
    var hf = _dim(t[0], 2)
    var wf = _dim(t[0], 3)
    var channels = _dim(t[0], 4)
    var frames2 = _dim(t[1], 1)
    var height2 = _dim(t[1], 2)
    var width2 = _dim(t[1], 3)
    var corr_channels = _dim(t[5], 2)
    var fmap1 = _ptr[dtype](t[0])
    var fmap2 = _ptr[dtype](t[1])
    var coords = _f(t[2])
    var ii = _i(t[3])
    var jj = _i(t[4])
    var corr = _f(t[5])
    var total = _dim(t[2], 0) * num_edges * height1 * width1
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[altcorr_index_kernel[dtype, pix]]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(
            func, Int64(total), Int64(radius), Int64(level_offset), coord_scale, Int64(num_edges), Int64(height1), Int64(width1),
            Int64(frames1), Int64(hf), Int64(wf), Int64(frames2), Int64(height2), Int64(width2), Int64(channels), Int64(corr_channels),
            fmap1, fmap2, coords, ii, jj, corr, grid_dim=_grid(total), block_dim=NUM_THREADS,
        )  # fmt: skip
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            altcorr_index_element[dtype, pix](
                n, radius, level_offset, coord_scale, num_edges, height1, width1, frames1, hf, wf, frames2, height2, width2, channels,
                corr_channels, fmap1, fmap2, coords, ii, jj, corr,
            )  # fmt: skip

        rt[].for_each_row(element, total)


def _altcorr_index_py[gpu: Bool](runtime: PythonObject, scalars: PythonObject, tensors: PythonObject) raises -> PythonObject:
    if Bool(py=scalars[1]):
        if Bool(py=scalars[2]):
            _altcorr_index[DType.float16, True, gpu](runtime, scalars, tensors)
        else:
            _altcorr_index[DType.float16, False, gpu](runtime, scalars, tensors)
    else:
        _altcorr_index[DType.float32, False, gpu](runtime, scalars, tensors)
    return Python.none()


# ── SLAM ─────────────────────────────────────────────────────────────────────


def _frame_distance_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, beta); tensors = (poses, disps (ND, H, W), intrinsics (Q, 4), pi, pj, qi, qj, di, dist (M,))."""
    var beta = _float(s, 1)
    var height = _dim(t[1], 1)
    var width = _dim(t[1], 2)
    var poses = _f(t[0])
    var disps = _f(t[1])
    var intrinsics = _f(t[2])
    var pi = _i(t[3])
    var pj = _i(t[4])
    var ri = _i(t[5])
    var rj = _i(t[6])
    var di = _i(t[7])
    var dist = _f(t[8])
    var pairs = _dim(t[3], 0)
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[frame_distance_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(
            func, Int64(height), Int64(width), beta, poses, disps, intrinsics, pi, pj, ri, rj, di, dist, grid_dim=pairs, block_dim=THREADS,
        )  # fmt: skip
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            frame_distance_element(n, height, width, beta, poses, disps, intrinsics, pi, pj, ri, rj, di, dist)

        rt[].for_each_row(element, pairs)
    return Python.none()


def _block_reduce_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (partial (rows, 256), out (rows,))."""
    var partial = _f(t[0])
    var dst = _f(t[1])
    var total = _dim(t[0], 0)
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[block_reduce_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(func, partial, dst, grid_dim=total, block_dim=THREADS)
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            block_reduce_element(n, partial, dst)

        rt[].for_each_row(element, total)
    return Python.none()


def _ba_partial_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, optimize_intrinsics, intrinsics_scale, flow_weight);
    tensors = (targets, weights, poses, disps, intrinsics, depth_active, ii, jj, Cii, bz, Eii, Eij, Efi, partial).
    """
    var optimize_intrinsics = _int(s, 1)
    var intrinsics_scale = _float(s, 2)
    var flow_weight = _float(s, 3)
    var height = _dim(t[3], 1)
    var width = _dim(t[3], 2)
    var target = _f(t[0])
    var weight = _f(t[1])
    var poses = _f(t[2])
    var disps = _f(t[3])
    var intrinsics = _f(t[4])
    var depth_active = _f(t[5])
    var ii = _i(t[6])
    var jj = _i(t[7])
    var cii = _f(t[8])
    var bz = _f(t[9])
    var eii = _f(t[10])
    var eij = _f(t[11])
    var efi = _f(t[12])
    var partial = _f(t[13])
    var total = _dim(t[6], 0) * THREADS
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[ba_partial_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(
            func, Int64(total), Int64(height), Int64(width), Int64(optimize_intrinsics), intrinsics_scale, flow_weight, target, weight, poses,
            disps, intrinsics, depth_active, ii, jj, cii, bz, eii, eij, efi, partial, grid_dim=_grid(total), block_dim=NUM_THREADS,
        )  # fmt: skip
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            ba_partial_element(
                n, height, width, optimize_intrinsics != 0, intrinsics_scale, flow_weight, target, weight, poses, disps, intrinsics,
                depth_active, ii, jj, cii, bz, eii, eij, efi, partial,
            )  # fmt: skip

        rt[].for_each_row(element, total)
    return Python.none()


def _accum_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (inps (R, D), order (R',), starts (J,), ends (J,), out (J, D))."""
    var width = _dim(t[0], 1)
    var inps = _f(t[0])
    var order = _i(t[1])
    var starts = _i(t[2])
    var ends = _i(t[3])
    var dst = _f(t[4])
    var total = _dim(t[4], 0) * width
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[accum_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(
            func, Int64(total), Int64(width), inps, order, starts, ends, dst, grid_dim=_grid(total), block_dim=NUM_THREADS
        )
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            accum_element(n, width, inps, order, starts, ends, dst)

        rt[].for_each_row(element, total)
    return Python.none()


def _eet_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (E (R, 6, D), Q (K, D), idx (P, 3), partial (P, 36, 256))."""
    var width = _dim(t[0], 2)
    var e = _f(t[0])
    var q = _f(t[1])
    var idx = _i(t[2])
    var partial = _f(t[3])
    var total = _dim(t[2], 0) * THREADS
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[eet_partial_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(func, Int64(total), Int64(width), e, q, idx, partial, grid_dim=_grid(total), block_dim=NUM_THREADS)
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            eet_partial_element(n, width, e, q, idx, partial)

        rt[].for_each_row(element, total)
    return Python.none()


def _ev_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (E (R, 6, D), Q (K, D), w (K, D), kk (R,), partial (R, 6, 256))."""
    var width = _dim(t[0], 2)
    var e = _f(t[0])
    var q = _f(t[1])
    var w = _f(t[2])
    var kk = _i(t[3])
    var partial = _f(t[4])
    var total = _dim(t[3], 0) * THREADS
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[ev_partial_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(func, Int64(total), Int64(width), e, q, w, kk, partial, grid_dim=_grid(total), block_dim=NUM_THREADS)
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            ev_partial_element(n, width, e, q, w, kk, partial)

        rt[].for_each_row(element, total)
    return Python.none()


def _evt_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (E (R, 6, D), x (P, 6), ix (R,), dw (R, D))."""
    var width = _dim(t[0], 2)
    var num_poses = _dim(t[1], 0)
    var e = _f(t[0])
    var x = _f(t[1])
    var ix = _i(t[2])
    var dw = _f(t[3])
    var total = _dim(t[3], 0) * width
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[evt_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(
            func, Int64(total), Int64(width), Int64(num_poses), e, x, ix, dw, grid_dim=_grid(total), block_dim=NUM_THREADS
        )
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            evt_element(n, width, num_poses, e, x, ix, dw)

        rt[].for_each_row(element, total)
    return Python.none()


def _pose_retr_py[gpu: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, t0); tensors = (poses (N, 7), dx (P, 6)): poses t0 .. t0 + P - 1."""
    var t0 = _int(s, 1)
    var poses = _f(t[0])
    var dx = _f(t[1])
    var total = _dim(t[1], 0)
    comptime if gpu:
        var rt = runtime.downcast_value_ptr[GPURuntime]()
        var func = rt[].ctx.compile_function[pose_retr_kernel]()
        rt[].torch_stream(_int(s, 0)).enqueue_function(func, Int64(total), Int64(t0), poses, dx, grid_dim=_grid(total), block_dim=NUM_THREADS)
    else:
        var rt = runtime.downcast_value_ptr[CPURuntime]()

        def element(n: Int) {imm}:
            pose_retr_element(n, t0, poses, dx)

        rt[].for_each_row(element, total)
    return Python.none()


def _pose_solve_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """CPU only. scalars = (0, poses, lm, ep, scalar_lm, scalar_ep); tensors = (host f32, a_src, a_dst, a_sign f32,
    b_src, b_dst, b_sign f32, a f64 (dim * dim) workspace, x f64 (dim)). Returns whether the factorisation succeeded."""
    var ok = pose_solve(
        _dim(t[8], 0), _int(s, 1), _double(s, 2), _double(s, 3), _double(s, 4), _double(s, 5), _f(t[0]),
        _i(t[1]), _i(t[2]), _f(t[3]), _dim(t[1], 0), _i(t[4]), _i(t[5]), _f(t[6]), _dim(t[4], 0),
        _ptr[DType.float64](t[7]), _ptr[DType.float64](t[8]),
    )  # fmt: skip
    return PythonObject(ok)


@export
def PyInit__vipe_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_vipe_mojo")
        m.def_function[_corr_index_py[True]]("corr_index_gpu")
        m.def_function[_corr_index_py[False]]("corr_index_cpu")
        m.def_function[_altcorr_index_py[True]]("altcorr_index_gpu")
        m.def_function[_altcorr_index_py[False]]("altcorr_index_cpu")
        m.def_function[_frame_distance_py[True]]("frame_distance_gpu")
        m.def_function[_frame_distance_py[False]]("frame_distance_cpu")
        m.def_function[_block_reduce_py[True]]("block_reduce_gpu")
        m.def_function[_block_reduce_py[False]]("block_reduce_cpu")
        m.def_function[_ba_partial_py[True]]("ba_partial_gpu")
        m.def_function[_ba_partial_py[False]]("ba_partial_cpu")
        m.def_function[_accum_py[True]]("accum_gpu")
        m.def_function[_accum_py[False]]("accum_cpu")
        m.def_function[_eet_py[True]]("eet_gpu")
        m.def_function[_eet_py[False]]("eet_cpu")
        m.def_function[_ev_py[True]]("ev_gpu")
        m.def_function[_ev_py[False]]("ev_cpu")
        m.def_function[_evt_py[True]]("evt_gpu")
        m.def_function[_evt_py[False]]("evt_cpu")
        m.def_function[_pose_retr_py[True]]("pose_retr_gpu")
        m.def_function[_pose_retr_py[False]]("pose_retr_cpu")
        m.def_function[_pose_solve_py]("pose_solve_cpu")
        return m.finalize()
    except e:
        abort(t"failed to create _vipe_mojo: {e}")
