"""CPython entry point of the Hexels CUDA extensions in Mojo (`hexels_mojo._hexels_mojo`).

Every op is `<op>(runtime, scalars, tensors)`, as in vipe_mojo:
- `runtime`: lietorch-mojo's `GPURuntime`, from `lietorch_mojo.backends._runtime(True)`. This module is
  built against the same Mojo structs (`-I <lietorch-mojo>/lietorch_mojo/mojo`) and must not register
  them as Python types again: a second registration of one Mojo type aborts the import.
- `scalars`: a tuple, first the raw CUDA stream, then the op's integer or float arguments.
- `tensors`: a tuple of `(address, *shape)` tuples of contiguous CUDA tensors in the op's order. The
  Python wrappers own the dtype and shape checks.

Calls are asynchronous on PyTorch's current CUDA stream. The upstream extensions are CUDA-only, so
there is no CPU path.
"""

from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from knn import BOX_SIZE, MORTON_THREADS, box_bounds_kernel, box_mean_dist_kernel, morton_kernel
from launch import GPURuntime
from prune import BLOCK, F32Ptr, removal_scores_kernel
from raster2d import BINNING_THREADS, TILE, backward_kernel, forward_kernel, map_to_intersects_kernel, tile_bin_edges_kernel
from ssim import BX, BY, fused_ssim_backward_kernel, fused_ssim_kernel


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
def _int(scalars: PythonObject, i: Int) raises -> Int:
    return Int(py=scalars[i])


def _removal_scores_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, rich); tensors = (xy (N, 2), scales (N, 2), rotations (N, 1), colors (N, 3),
    gradients (N, 2, 3), softness (N), image (H, W, 4), target (H, W, 3), selected i32 (H, W, 10),
    scores (N) zeroed, max_error u32 () zeroed)."""
    if Bool(py=s[1]):
        _removal_scores[True](runtime, s, t)
    else:
        _removal_scores[False](runtime, s, t)
    return Python.none()


def _removal_scores[rich: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises:
    var height = _dim(t[6], 0)
    var width = _dim(t[6], 1)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var func = rt[].ctx.compile_function[removal_scores_kernel[rich]]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        func, Int32(width), Int32(height), _f(t[0]), _f(t[1]), _f(t[2]), _f(t[3]), _f(t[4]),
        _f(t[5]), _f(t[6]), _f(t[7]), _ptr[DType.int32](t[8]), _f(t[9]), _ptr[DType.uint32](t[10]),
        grid_dim=(ceildiv(width, BLOCK), ceildiv(height, BLOCK)), block_dim=(BLOCK, BLOCK),
    )  # fmt: skip


def _fused_ssim_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, c1, c2, train); tensors = (img1 (B, C, H, W), img2, ssim_map, dm_dmu1, dm_dsigma1_sq,
    dm_dsigma12); the three derivatives are only written when train (pass any CUDA tensor otherwise)."""
    if Bool(py=s[3]):
        _fused_ssim[True](runtime, s, t)
    else:
        _fused_ssim[False](runtime, s, t)
    return Python.none()


def _fused_ssim[train: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises:
    var batch = _dim(t[0], 0)
    var channels = _dim(t[0], 1)
    var height = _dim(t[0], 2)
    var width = _dim(t[0], 3)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var func = rt[].ctx.compile_function[fused_ssim_kernel[train]]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        func, Int32(height), Int32(width), Int32(channels), Float32(Float64(py=s[1])), Float32(Float64(py=s[2])),
        _f(t[0]), _f(t[1]), _f(t[2]), _f(t[3]), _f(t[4]), _f(t[5]),
        grid_dim=(ceildiv(width, BX), ceildiv(height, BY), batch), block_dim=(BX, BY),
    )  # fmt: skip


def _fused_ssim_backward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (img1 (B, C, H, W), img2, dL_dmap, dL_dimg1, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)."""
    var batch = _dim(t[0], 0)
    var channels = _dim(t[0], 1)
    var height = _dim(t[0], 2)
    var width = _dim(t[0], 3)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var func = rt[].ctx.compile_function[fused_ssim_backward_kernel]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        func, Int32(height), Int32(width), Int32(channels), _f(t[0]), _f(t[1]), _f(t[2]), _f(t[3]), _f(t[4]), _f(t[5]), _f(t[6]),
        grid_dim=(ceildiv(width, BX), ceildiv(height, BY), batch), block_dim=(BX, BY),
    )  # fmt: skip
    return Python.none()


def _knn_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, stage); stage 0: tensors = (points (P, 3), bounds (6), codes i64 (P));
    stage 1: tensors = (points, order i64 (P), boxes (ceil(P / 1024), 6), dists (P))."""
    var count = _dim(t[0], 0)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var stream = rt[].torch_stream(_int(s, 0))
    if _int(s, 1) == 0:
        var func = rt[].ctx.compile_function[morton_kernel]()
        stream.enqueue_function(
            func, Int32(count), _f(t[0]), _f(t[1]), _ptr[DType.int64](t[2]), grid_dim=ceildiv(count, MORTON_THREADS), block_dim=MORTON_THREADS
        )
        return Python.none()
    var boxes = ceildiv(count, BOX_SIZE)
    var bounds = rt[].ctx.compile_function[box_bounds_kernel]()
    stream.enqueue_function(bounds, Int32(count), _f(t[0]), _ptr[DType.int64](t[1]), _f(t[2]), grid_dim=boxes, block_dim=BOX_SIZE)
    var mean_dist = rt[].ctx.compile_function[box_mean_dist_kernel]()
    stream.enqueue_function(
        mean_dist, Int32(count), _f(t[0]), _ptr[DType.int64](t[1]), _f(t[2]), _f(t[3]), grid_dim=boxes, block_dim=BOX_SIZE
    )
    return Python.none()


def _i32(spec: PythonObject) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return _ptr[DType.int32](spec)


def _raster2d_binning_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, stage, tiles_x, tiles_y); stage 0: tensors = (xys (N, 2), radii i32 (N), cum_tiles_hit i32 (N),
    isect_ids i64 (M), gaussian_ids i32 (M)); stage 1: tensors = (isect_ids_sorted i64 (M), tile_bins i32 (T, 2) zeroed)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var stream = rt[].torch_stream(_int(s, 0))
    if _int(s, 1) == 0:
        var n = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[map_to_intersects_kernel](), Int32(n), Int32(_int(s, 2)), Int32(_int(s, 3)), _f(t[0]), _i32(t[1]),
            _i32(t[2]), _ptr[DType.int64](t[3]), _i32(t[4]), grid_dim=ceildiv(n, BINNING_THREADS), block_dim=BINNING_THREADS,
        )  # fmt: skip
    else:
        var m = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[tile_bin_edges_kernel](), Int32(m), _ptr[DType.int64](t[0]), _i32(t[1]),
            grid_dim=ceildiv(m, BINNING_THREADS), block_dim=BINNING_THREADS,
        )  # fmt: skip
    return Python.none()


def _raster2d_forward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, color_grad_scale, aa, softness_stride, softness_default, has_sharpness, has_emission);
    tensors = (ids_sorted i32, tile_bins i32 (T, 2), xys, scales, rotation, colors (N, C), color_grads, softnesses,
    sharpness, emission, out_img (H, W, C), pixel_topk i32 (H, W, 10), weights_sum (H, W), entropy (H, W))."""
    var height = _dim(t[10], 0)
    var width = _dim(t[10], 1)
    var channels = _dim(t[10], 2)
    var tiles_x = ceildiv(width, TILE)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[forward_kernel](), Int32(tiles_x), Int32(width), Int32(height), Int32(channels), _i32(t[0]), _i32(t[1]),
        _f(t[2]), _f(t[3]), _f(t[4]), _f(t[5]), _f(t[6]), Float32(Float64(py=s[1])), Float32(Float64(py=s[2])), _f(t[7]),
        Int32(_int(s, 3)), Float32(Float64(py=s[4])), Int32(_int(s, 5)), _f(t[8]), Int32(_int(s, 6)), _f(t[9]), _f(t[10]), _i32(t[11]),
        _f(t[12]), _f(t[13]), grid_dim=(tiles_x, ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip
    return Python.none()


def _raster2d_backward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, color_grad_scale, aa, softness_stride, softness_default, has_sharpness, has_emission);
    tensors = (xys, scales, rotation, colors (N, C), color_grads, emission, v_output (H, W, C), v_weights_sum, v_entropy,
    pixel_topk i32, softnesses, sharpness, then the zeroed outputs v_xy, v_scale, v_rot, v_colors, v_color_grads,
    v_emission, v_sharpness, v_softnesses)."""
    var height = _dim(t[6], 0)
    var width = _dim(t[6], 1)
    var channels = _dim(t[6], 2)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[backward_kernel](), Int32(width), Int32(height), Int32(channels), _f(t[0]), _f(t[1]), _f(t[2]), _f(t[3]),
        _f(t[4]), Float32(Float64(py=s[1])), Int32(_int(s, 6)), _f(t[5]), _f(t[6]), _f(t[7]), _f(t[8]), _i32(t[9]),
        Float32(Float64(py=s[2])), _f(t[10]), Int32(_int(s, 3)), Float32(Float64(py=s[4])), Int32(_int(s, 5)), _f(t[11]), _f(t[12]),
        _f(t[13]), _f(t[14]), _f(t[15]), _f(t[16]), _f(t[17]), _f(t[18]), _f(t[19]),
        grid_dim=(ceildiv(width, TILE), ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip
    return Python.none()


@export
def PyInit__hexels_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_hexels_mojo")
        m.def_function[_removal_scores_py]("removal_scores")
        m.def_function[_fused_ssim_py]("fused_ssim")
        m.def_function[_fused_ssim_backward_py]("fused_ssim_backward")
        m.def_function[_knn_py]("knn")
        m.def_function[_raster2d_binning_py]("raster2d_binning")
        m.def_function[_raster2d_forward_py]("raster2d_forward")
        m.def_function[_raster2d_backward_py]("raster2d_backward")
        return m.finalize()
    except e:
        abort(t"failed to create _hexels_mojo: {e}")
