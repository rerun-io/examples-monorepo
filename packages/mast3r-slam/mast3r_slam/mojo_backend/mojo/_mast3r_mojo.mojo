"""CPython entry point of MASt3R-SLAM's Mojo kernels (`mast3r_slam.mojo_backend._mast3r_mojo`).

Every op is `<op>_gpu(runtime, scalars, tensors)` (CUDA) or `<op>_cpu(runtime, scalars, tensors)`:
- `runtime`: lietorch-mojo's `GPURuntime` or `CPURuntime` object (`lietorch_mojo.backends._runtime`). This module
  is built against the same Mojo structs (`-I <lietorch-mojo>/lietorch_mojo/mojo`) and must not register them as
  Python types again, so both extensions share one runtime per device.
- `scalars`: a tuple, first the raw CUDA stream (0 for CPU ops), then the op's integer or float arguments.
- `tensors`: a tuple of `(address, *shape)` tuples of contiguous tensors in the op's order, built by
  `mast3r_slam.mojo_backend._launch`, which owns the device checks; the Python ops own dtypes and shapes.

GPU calls are asynchronous (PyTorch's current CUDA stream).
"""

from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from gn import THREADS, gn_step_kernel
from launch import GPURuntime
from matching import BLOCK, iter_proj_kernel, refine_matches_kernel
from solve import gn_solve


@always_inline
def _dim(spec: PythonObject, axis: Int) raises -> Int:
    return Int(py=spec[axis + 1])


@always_inline
def _ptr[dtype: DType](spec: PythonObject) raises -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    return MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))


@always_inline
def _int(scalars: PythonObject, i: Int) raises -> Int64:
    return Int64(Int(py=scalars[i]))


@always_inline
def _float(scalars: PythonObject, i: Int) raises -> Float32:
    return Float32(Float64(py=scalars[i]))


def iter_proj_gpu_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """Scalars = (stream, max_iter, lambda_init, cost_thresh); tensors = (rays, pts, p_init, p_new, converged)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var batch = _dim(t[0], 0)
    var num_points = _dim(t[1], 1)
    var func = rt[].ctx.compile_function[iter_proj_kernel]()
    rt[].torch_stream(Int(py=s[0])).enqueue_function(
        func, Int64(num_points), Int64(_dim(t[0], 1)), Int64(_dim(t[0], 2)), _int(s, 1), _float(s, 2), _float(s, 3),
        _ptr[DType.float32](t[0]), _ptr[DType.float32](t[1]), _ptr[DType.float32](t[2]), _ptr[DType.float32](t[3]),
        _ptr[DType.uint8](t[4]),
        grid_dim=(ceildiv(num_points, BLOCK), batch), block_dim=BLOCK,
    )  # fmt: skip
    return Python.none()


def _refine_matches[fdim: Int](mut rt: GPURuntime, s: PythonObject, t: PythonObject) raises:
    var batch = _dim(t[0], 0)
    var num_points = _dim(t[1], 1)
    var func = rt.ctx.compile_function[refine_matches_kernel[fdim]]()
    rt.torch_stream(Int(py=s[0])).enqueue_function(
        func, Int64(num_points), Int64(_dim(t[0], 1)), Int64(_dim(t[0], 2)), Int64(_dim(t[0], 3)), _int(s, 1), _int(s, 2),
        _ptr[DType.float16](t[0]), _ptr[DType.float16](t[1]), _ptr[DType.int64](t[2]), _ptr[DType.int64](t[3]),
        grid_dim=(ceildiv(num_points, BLOCK), batch), block_dim=BLOCK,
    )  # fmt: skip


def refine_matches_gpu_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """Scalars = (stream, radius, dilation_max, aligned); tensors = (D11, D21, p1, p_new). MASt3R's descriptors
    are 24 wide: that width has its own kernel when both descriptor arrays are 16-byte aligned."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    if _dim(t[0], 3) == 24 and Bool(py=s[3]):
        _refine_matches[24](rt[], s, t)
    else:
        _refine_matches[0](rt[], s, t)
    return Python.none()


def _gn_step[calib: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """Scalars = (stream, blocks_per_edge, height, width, pixel_border, sigma_a, sigma_b, C_thresh, Q_thresh, z_eps);
    tensors = (Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q, sums). Rays ignore height, width, pixel_border,
    z_eps and K."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var num_edges = _dim(t[4], 0)
    var blocks_per_edge = Int(py=s[1])
    var func = rt[].ctx.compile_function[gn_step_kernel[calib]]()
    rt[].torch_stream(Int(py=s[0])).enqueue_function(
        func, Int64(_dim(t[1], 1)), Int64(blocks_per_edge), _int(s, 2), _int(s, 3), _int(s, 4),
        _float(s, 5), _float(s, 6), _float(s, 7), _float(s, 8), _float(s, 9),
        _ptr[DType.float32](t[0]), _ptr[DType.float32](t[1]), _ptr[DType.float32](t[2]), _ptr[DType.float32](t[3]),
        _ptr[DType.int64](t[4]), _ptr[DType.int64](t[5]), _ptr[DType.int64](t[6]), _ptr[DType.uint8](t[7]),
        _ptr[DType.float32](t[8]), _ptr[DType.float32](t[9]),
        grid_dim=(blocks_per_edge, num_edges), block_dim=THREADS,
    )  # fmt: skip
    return Python.none()


def gn_solve_cpu_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """Scalars = (0,); tensors = (sums (E, 119) f32, ii (E,) i64, jj (E,) i64, dx (P, 7) f32), all on the CPU.
    Returns whether the factorisation succeeded (dx is zero when it did not)."""
    var ok = gn_solve(
        _dim(t[3], 0), _dim(t[1], 0), _ptr[DType.float32](t[0]), _ptr[DType.int64](t[1]), _ptr[DType.int64](t[2]),
        _ptr[DType.float32](t[3]),
    )  # fmt: skip
    return PythonObject(ok)


@export
def PyInit__mast3r_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mast3r_mojo")
        m.def_function[iter_proj_gpu_py]("iter_proj_gpu")
        m.def_function[refine_matches_gpu_py]("refine_matches_gpu")
        m.def_function[_gn_step[False]]("gn_rays_step_gpu")
        m.def_function[_gn_step[True]]("gn_calib_step_gpu")
        m.def_function[gn_solve_cpu_py]("gn_solve_cpu")
        return m.finalize()
    except e:
        abort(t"failed to create _mast3r_mojo: {e}")

