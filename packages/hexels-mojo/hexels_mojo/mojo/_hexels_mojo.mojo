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

from std.itertools import product
from std.math import ceildiv
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

import gaussians2d
import raster3d
import raster3d_backward
from tensors import Grid, In, Out, Planes, Rows, StackedPlanes
from knn import BOX_SIZE, MORTON_THREADS, box_bounds_kernel, box_mean_dist_kernel, morton_kernel
from launch import GPURuntime
from layout import Idx, TileTensor, row_major
from prune import BLOCK, removal_scores_kernel
from raster2d import BINNING_THREADS, TILE, TOP_K, backward_kernel, forward_kernel, map_to_intersects_kernel, tile_bin_edges_kernel
from ssim import BX, BY, fused_ssim_backward_kernel, fused_ssim_kernel


@inline(.always)
def _dim(spec: PythonObject, axis: Int) raises -> Int:
    return Int(py=spec[axis + 1])


def _rows(spec: PythonObject, width: Int) raises -> Int:
    """Rows of `width` elements in the contiguous tensor `spec`."""
    var numel = 1
    for axis in range(len(spec) - 1):
        numel *= _dim(spec, axis)
    if numel % width != 0:
        raise Error(t"expected rows of {width} elements, got {numel} elements")
    return numel // width


def _in[dtype: DType, width: Int = 1](spec: PythonObject) raises -> In[dtype, Rows[width]]:
    """`spec` as an immutable (rows, width) tensor."""
    var data = ImmPointer[Scalar[dtype], ImmUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major((_rows(spec, width), Idx[width])))


def _out[dtype: DType, width: Int = 1](spec: PythonObject) raises -> Out[dtype, Rows[width]]:
    """`spec` as a mutable (rows, width) tensor."""
    var data = MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major((_rows(spec, width), Idx[width])))


def _planes(spec: PythonObject) raises -> Tuple[Int, Int, Int]:
    """(leading dims flattened, H, W) of a (..., H, W) tensor; (0, 0, 0) when empty."""
    var rank = len(spec) - 1
    if rank < 3:
        return (0, 0, 0)
    var planes = 1
    for axis in range(rank - 2):
        planes *= _dim(spec, axis)
    return (planes, _dim(spec, rank - 2), _dim(spec, rank - 1))


def _in_planes(spec: PythonObject) raises -> In[.float32, Planes]:
    var data = ImmPointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major(_planes(spec)))


def _out_planes(spec: PythonObject) raises -> Out[.float32, Planes]:
    var data = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major(_planes(spec)))


def _stacked(spec: PythonObject) raises -> Tuple[Int, Int, Int, Int]:
    """(K, middle dims flattened, H, W) of a (K, ..., H, W) tensor."""
    var rank = len(spec) - 1
    var planes = 1
    for axis in range(1, rank - 2):
        planes *= _dim(spec, axis)
    return (_dim(spec, 0), planes, _dim(spec, rank - 2), _dim(spec, rank - 1))


def _in_stacked(spec: PythonObject) raises -> In[.float32, StackedPlanes]:
    var data = ImmPointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major(_stacked(spec)))


def _out_stacked(spec: PythonObject) raises -> Out[.float32, StackedPlanes]:
    var data = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major(_stacked(spec)))


def _grid(spec: PythonObject) raises -> Tuple[Int, Int]:
    """(leading dims flattened, last dim) of a tensor; a 1-D tensor is one column."""
    var rank = len(spec) - 1
    if rank == 1:
        return (_dim(spec, 0), 1)
    var rows = 1
    for axis in range(rank - 1):
        rows *= _dim(spec, axis)
    return (rows, _dim(spec, rank - 1))


def _in_grid(spec: PythonObject) raises -> In[.float32, Grid]:
    var data = ImmPointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major(_grid(spec)))


def _out_grid(spec: PythonObject) raises -> Out[.float32, Grid]:
    var data = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=spec[0]))
    return TileTensor(data, row_major(_grid(spec)))


@inline(.always)
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
        func, Int32(width), Int32(height), _in[.float32, 2](t[0]), _in[.float32, 2](t[1]), _in[.float32](t[2]), _in[.float32, 3](t[3]),
        _in[.float32, 6](t[4]), _in[.float32](t[5]), _in[.float32, 4](t[6]), _in[.float32, 3](t[7]), _in[.int32, TOP_K](t[8]),
        _out[.float32](t[9]), _out[.uint32](t[10]),
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
        func, Int32(channels), Float32(Float64(py=s[1])), Float32(Float64(py=s[2])), _in_planes(t[0]), _in_planes(t[1]),
        _out_stacked(t[2]),
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
        func, Int32(channels), _in_planes(t[0]), _in_planes(t[1]), _in_planes(t[2]), _out_planes(t[3]), _in_stacked(t[4]),
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
            func, _in[.float32, 3](t[0]), _in[.float32, 3](t[1]), _out[.int64](t[2]), grid_dim=ceildiv(count, MORTON_THREADS), block_dim=MORTON_THREADS
        )
        return Python.none()
    var boxes = ceildiv(count, BOX_SIZE)
    var bounds = rt[].ctx.compile_function[box_bounds_kernel]()
    stream.enqueue_function(bounds, _in[.float32, 3](t[0]), _in[.int64](t[1]), _out[.float32, 6](t[2]), grid_dim=boxes, block_dim=BOX_SIZE)
    var mean_dist = rt[].ctx.compile_function[box_mean_dist_kernel]()
    stream.enqueue_function(
        mean_dist, _in[.float32, 3](t[0]), _in[.int64](t[1]), _in[.float32, 6](t[2]), _out[.float32](t[3]), grid_dim=boxes, block_dim=BOX_SIZE
    )
    return Python.none()


def _raster2d_binning_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, stage, tiles_x, tiles_y); stage 0: tensors = (xys (N, 2), radii i32 (N), cum_tiles_hit i32 (N),
    isect_ids i64 (M), gaussian_ids i32 (M)); stage 1: tensors = (isect_ids_sorted i64 (M), tile_bins i32 (T, 2) zeroed)."""
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var stream = rt[].torch_stream(_int(s, 0))
    if _int(s, 1) == 0:
        var n = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[map_to_intersects_kernel](), Int32(_int(s, 2)), Int32(_int(s, 3)), _in[.float32, 2](t[0]),
            _in[.int32](t[1]), _in[.int32](t[2]), _out[.int64](t[3]), _out[.int32](t[4]), grid_dim=ceildiv(n, BINNING_THREADS),
            block_dim=BINNING_THREADS,
        )  # fmt: skip
    else:
        var m = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[tile_bin_edges_kernel](), _in[.int64](t[0]), _out[.int32, 2](t[1]),
            grid_dim=ceildiv(m, BINNING_THREADS), block_dim=BINNING_THREADS,
        )  # fmt: skip
    return Python.none()


def _raster2d_forward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, color_grad_scale, aa, softness_stride, softness_default, has_sharpness, has_emission, color_grad);
    tensors = (ids_sorted i32, tile_bins i32 (T, 2), xys, scales, rotation, colors (N, C), color_grads, softnesses,
    sharpness, emission, out_img (H, W, C), pixel_topk i32 (H, W, 10), weights_sum (H, W), entropy (H, W))."""
    if Bool(py=s[7]):
        _raster2d_forward[True](runtime, s, t)
    else:
        _raster2d_forward[False](runtime, s, t)
    return Python.none()


def _raster2d_forward[color_grad: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises:
    var height = _dim(t[10], 0)
    var width = _dim(t[10], 1)
    var tiles_x = ceildiv(width, TILE)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[forward_kernel[color_grad]](), Int32(tiles_x), Int32(width), Int32(height), _in[.int32](t[0]),
        _in[.int32, 2](t[1]), _in[.float32, 2](t[2]), _in[.float32, 2](t[3]), _in[.float32](t[4]), _in_grid(t[5]), _in_grid(t[6]),
        Float32(Float64(py=s[1])), Float32(Float64(py=s[2])), _in_grid(t[7]), Int32(_int(s, 3)), Float32(Float64(py=s[4])), Int32(_int(s, 5)),
        _in[.float32](t[8]), Int32(_int(s, 6)), _in_grid(t[9]), _out_grid(t[10]), _out[.int32, TOP_K](t[11]), _out[.float32](t[12]),
        _out[.float32](t[13]), grid_dim=(tiles_x, ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip


def _raster2d_backward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, color_grad_scale, aa, softness_stride, softness_default, has_sharpness, has_emission, color_grad);
    tensors = (xys, scales, rotation, colors (N, C), color_grads, emission, v_output (H, W, C), v_weights_sum, v_entropy,
    pixel_topk i32, softnesses, sharpness, then the zeroed outputs v_xy, v_scale, v_rot, v_colors, v_color_grads,
    v_emission, v_sharpness, v_softnesses)."""
    if Bool(py=s[7]):
        _raster2d_backward[True](runtime, s, t)
    else:
        _raster2d_backward[False](runtime, s, t)
    return Python.none()


def _raster2d_backward[color_grad: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises:
    var height = _dim(t[6], 0)
    var width = _dim(t[6], 1)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[backward_kernel[color_grad]](), Int32(width), Int32(height), _in[.float32, 2](t[0]), _in[.float32, 2](t[1]),
        _in[.float32](t[2]), _in_grid(t[3]), _in_grid(t[4]), Float32(Float64(py=s[1])), Int32(_int(s, 6)), _in_grid(t[5]), _in_grid(t[6]),
        _in[.float32](t[7]), _in[.float32](t[8]), _in[.int32, TOP_K](t[9]), Float32(Float64(py=s[2])), _in_grid(t[10]), Int32(_int(s, 3)),
        Float32(Float64(py=s[4])), Int32(_int(s, 5)), _in[.float32](t[11]), _out[.float32, 2](t[12]), _out[.float32, 2](t[13]),
        _out[.float32](t[14]), _out_grid(t[15]), _out_grid(t[16]), _out_grid(t[17]), _out[.float32](t[18]), _out_grid(t[19]),
        grid_dim=(ceildiv(width, TILE), ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip


def _gaussians2d_project_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, stage, width, height); stage 0 (forward): tensors = (means (N, 2), scales (N, 2), rotation (N, 1),
    xys, radii i32 (N), conics (N, 3), num_tiles_hit i32 (N)), outputs zeroed; stage 1 (backward): tensors = (scales,
    rotation, radii i32, conics, v_xy (N, 2), v_conic (N, 3), v_cov2d, v_mean, v_scale, v_rot), outputs zeroed."""
    var width = _int(s, 2)
    var height = _int(s, 3)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var stream = rt[].torch_stream(_int(s, 0))
    var n = _dim(t[0], 0)
    var grid = ceildiv(n, gaussians2d.POINT_THREADS)
    if _int(s, 1) == 0:
        stream.enqueue_function(
            rt[].ctx.compile_function[gaussians2d.project_forward_kernel](), Int32(n), Int32(width), Int32(height),
            Int32(ceildiv(width, TILE)), Int32(ceildiv(height, TILE)), _in[.float32, 2](t[0]), _in[.float32, 2](t[1]), _in[.float32](t[2]),
            _out[.float32, 2](t[3]), _out[.int32](t[4]), _out[.float32, 3](t[5]), _out[.int32](t[6]), grid_dim=grid,
            block_dim=gaussians2d.POINT_THREADS,
        )  # fmt: skip
    else:
        stream.enqueue_function(
            rt[].ctx.compile_function[gaussians2d.project_backward_kernel](), Int32(n), Int32(width), Int32(height), _in[.float32, 2](t[0]),
            _in[.float32](t[1]), _in[.int32](t[2]), _in[.float32, 3](t[3]), _in[.float32, 2](t[4]), _in[.float32, 3](t[5]),
            _out[.float32, 3](t[6]), _out[.float32, 2](t[7]), _out[.float32, 2](t[8]), _out[.float32](t[9]),
            grid_dim=grid, block_dim=gaussians2d.POINT_THREADS,
        )  # fmt: skip
    return Python.none()


def _gaussians2d_forward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (ids_sorted i32, tile_bins i32 (T, 2), xys, conics (N, 3), colors (N, C),
    out_img (H, W, C), pixel_topk i32 (H, W, 10))."""
    var height = _dim(t[5], 0)
    var width = _dim(t[5], 1)
    var tiles_x = ceildiv(width, TILE)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[gaussians2d.forward_kernel](), Int32(tiles_x), Int32(width), Int32(height), _in[.int32](t[0]),
        _in[.int32, 2](t[1]), _in[.float32, 2](t[2]), _in[.float32, 3](t[3]), _in_grid(t[4]), _out_grid(t[5]), _out[.int32, TOP_K](t[6]),
        grid_dim=(tiles_x, ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip
    return Python.none()


def _gaussians2d_backward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream,); tensors = (xys, conics, colors (N, C), v_output (H, W, C), pixel_topk i32, v_xy, v_conic, v_colors)."""
    var height = _dim(t[3], 0)
    var width = _dim(t[3], 1)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[gaussians2d.backward_kernel](), Int32(width), Int32(height), _in[.float32, 2](t[0]), _in[.float32, 3](t[1]),
        _in_grid(t[2]), _in_grid(t[3]), _in[.int32, TOP_K](t[4]), _out[.float32, 2](t[5]), _out[.float32, 3](t[6]), _out_grid(t[7]),
        grid_dim=(ceildiv(width, TILE), ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip
    return Python.none()


def _raster3d_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, stage, width, height, ...). Stage 0 preprocess: (scale_modifier, softness_default, softness_stride,
    has_softness, conservative, want_normals); tensors = (means, scales, rotations, opacities, softnesses, viewmatrix,
    projmatrix, campos, radii, tile_radii, points_xy, depths, transmats, tiles_touched, opacities_out, softnesses_out,
    view_dirs, normals). Stage 1 duplicate: tensors = (points_xy, depths, offsets, radii, tile_radii, keys i64, values i32).
    Stage 2 render: (color_model, view_model, use_view_grads, softness_stride); tensors = (ranges, point_list, colors,
    color_grads, color_quads, view_grads, specular_ks, specular_log_s, view_dirs, normals, transmats, opacities, softnesses,
    background, out_color (3, H, W), final_transmittance (H, W), last_index i32 (H, W))."""
    var stage = _int(s, 1)
    var width = _int(s, 2)
    var height = _int(s, 3)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var stream = rt[].torch_stream(_int(s, 0))
    if stage == 0:
        var n = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[raster3d.preprocess_kernel](), Int32(n), Int32(width), Int32(height), Float32(Float64(py=s[4])),
            Float32(Float64(py=s[5])), Int32(_int(s, 6)), Int32(_int(s, 7)), Int32(_int(s, 8)), Int32(_int(s, 9)), _in[.float32, 3](t[0]),
            _in[.float32, 2](t[1]), _in[.float32, 4](t[2]), _in[.float32](t[3]), _in_grid(t[4]), _in[.float32, 16](t[5]),
            _in[.float32, 16](t[6]), _in[.float32, 3](t[7]), _out[.int32](t[8]), _out[.int32, 2](t[9]), _out[.float32, 2](t[10]),
            _out[.float32](t[11]), _out[.float32, 9](t[12]), _out[.int32](t[13]), _out[.float32](t[14]), _out_grid(t[15]),
            _out[.float32, 3](t[16]), _out[.float32, 3](t[17]),
            grid_dim=ceildiv(n, raster3d.POINT_THREADS), block_dim=raster3d.POINT_THREADS,
        )  # fmt: skip
    elif stage == 1:
        var n = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[raster3d.duplicate_kernel](), Int32(width), Int32(height), _in[.float32, 2](t[0]), _in[.float32](t[1]),
            _in[.int32](t[2]), _in[.int32](t[3]), _in[.int32, 2](t[4]), _out[.int64](t[5]), _out[.int32](t[6]),
            grid_dim=ceildiv(n, raster3d.POINT_THREADS), block_dim=raster3d.POINT_THREADS,
        )  # fmt: skip
    else:
        # One kernel per (color model, view gradients, specular lobe): the flags are compile-time, as hot-loop branches.
        var color_model = _int(s, 4)
        var use_view_grads = _int(s, 6) != 0
        var specular = _int(s, 5) == 2
        comptime for cm, vg, spec in product(range(4), range(2), range(2)):
            if color_model == cm and use_view_grads == (vg == 1) and specular == (spec == 1):
                _raster3d_render[cm, vg == 1, spec == 1](runtime, s, t)
    return Python.none()


def _raster3d_render[color_model: Int, use_view_grads: Bool, specular: Bool](runtime: PythonObject, s: PythonObject, t: PythonObject) raises:
    var width = _int(s, 2)
    var height = _int(s, 3)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[raster3d.render_kernel[color_model, use_view_grads, specular]](), _in[.int32, 2](t[0]), _in[.int32](t[1]),
        _in[.float32, 3](t[2]), _in[.float32, 6](t[3]), _in[.float32, 9](t[4]), _in[.float32, 9](t[5]), _in[.float32, 3](t[6]),
        _in[.float32](t[7]), _in[.float32, 3](t[8]), _in[.float32, 3](t[9]), _in[.float32, 9](t[10]), _in[.float32](t[11]), _in_grid(t[12]),
        Int32(_int(s, 7)), _in[.float32, 3](t[13]), _out_planes(t[14]), _out[.float32](t[15]), _out[.int32](t[16]),
        grid_dim=(ceildiv(width, TILE), ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip



def _raster3d_backward_py(runtime: PythonObject, s: PythonObject, t: PythonObject) raises -> PythonObject:
    """scalars = (stream, stage, width, height, ...). Stage 0 render: (color_model, view_model, use_view_grads,
    softness_stride, has_softness_grad, abs_means2d); tensors = (ranges, point_list, colors, color_grads, color_quads,
    view_grads, specular_ks, specular_log_s, view_dirs, normals, transmats, opacities, softnesses, background,
    grad_out_color, final_transmittance, last_index, then the gradients of colors, color_grads, color_quads, view_grads,
    specular_ks, specular_log_s, opacities, softnesses, transmats, means2d, view_dirs, normals). Stage 1 means2d:
    (stride,); tensors = (radii, transmats, grad_transmats, grad_means2d). Stage 2 geometry: (scale_modifier,
    has_normals); tensors = (means, scales, rotations, projmatrix, campos, radii, grad_transmats, grad_view_dirs,
    grad_normals, grad_means, grad_scales, grad_rotations)."""
    var stage = _int(s, 1)
    var width = _int(s, 2)
    var height = _int(s, 3)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    var stream = rt[].torch_stream(_int(s, 0))
    if stage == 0:
        var color_model = _int(s, 4)
        var use_view_grads = _int(s, 6) != 0
        var specular = _int(s, 5) == 2
        comptime for cm, vg, spec in product(range(4), range(2), range(2)):
            if color_model == cm and use_view_grads == (vg == 1) and specular == (spec == 1):
                _raster3d_render_backward[cm, vg == 1, spec == 1](runtime, s, t)
    elif stage == 1:
        var n = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[raster3d_backward.means2d_backward_kernel](), Int32(width), Int32(height), _in[.int32](t[0]),
            _in[.float32, 9](t[1]), _in[.float32, 9](t[2]), _out_grid(t[3]),
            grid_dim=ceildiv(n, raster3d.POINT_THREADS), block_dim=raster3d.POINT_THREADS,
        )  # fmt: skip
    else:
        var n = _dim(t[0], 0)
        stream.enqueue_function(
            rt[].ctx.compile_function[raster3d_backward.geometry_backward_kernel](), Int32(width), Int32(height), Float32(Float64(py=s[4])),
            Int32(_int(s, 5)), _in[.float32, 3](t[0]), _in[.float32, 2](t[1]), _in[.float32, 4](t[2]), _in[.float32, 16](t[3]),
            _in[.float32, 3](t[4]), _in[.int32](t[5]), _in[.float32, 9](t[6]), _in[.float32, 3](t[7]), _in[.float32, 3](t[8]),
            _out[.float32, 3](t[9]), _out[.float32, 2](t[10]), _out[.float32, 4](t[11]),
            grid_dim=ceildiv(n, raster3d.POINT_THREADS), block_dim=raster3d.POINT_THREADS,
        )  # fmt: skip
    return Python.none()


def _raster3d_render_backward[color_model: Int, use_view_grads: Bool, specular: Bool](
    runtime: PythonObject, s: PythonObject, t: PythonObject
) raises:
    var width = _int(s, 2)
    var height = _int(s, 3)
    var rt = runtime.downcast_value_ptr[GPURuntime]()
    rt[].torch_stream(_int(s, 0)).enqueue_function(
        rt[].ctx.compile_function[raster3d_backward.render_backward_kernel[color_model, use_view_grads, specular]](), _in[.int32, 2](t[0]),
        _in[.int32](t[1]), _in[.float32, 3](t[2]), _in[.float32, 6](t[3]), _in[.float32, 9](t[4]), _in[.float32, 9](t[5]),
        _in[.float32, 3](t[6]), _in[.float32](t[7]), _in[.float32, 3](t[8]), _in[.float32, 3](t[9]), _in[.float32, 9](t[10]),
        _in[.float32](t[11]), _in_grid(t[12]), Int32(_int(s, 7)), Int32(_int(s, 8)), Int32(_int(s, 9)), _in[.float32, 3](t[13]),
        _in_planes(t[14]), _in[.float32](t[15]), _in[.int32](t[16]), _out[.float32, 3](t[17]), _out[.float32, 6](t[18]),
        _out[.float32, 9](t[19]), _out[.float32, 9](t[20]), _out[.float32, 3](t[21]), _out[.float32](t[22]), _out[.float32](t[23]),
        _out_grid(t[24]), _out[.float32, 9](t[25]), _out_grid(t[26]), _out[.float32, 3](t[27]), _out[.float32, 3](t[28]),
        grid_dim=(ceildiv(width, TILE), ceildiv(height, TILE)), block_dim=(TILE, TILE),
    )  # fmt: skip

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
        m.def_function[_gaussians2d_project_py]("gaussians2d_project")
        m.def_function[_gaussians2d_forward_py]("gaussians2d_forward")
        m.def_function[_gaussians2d_backward_py]("gaussians2d_backward")
        m.def_function[_raster3d_py]("raster3d")
        m.def_function[_raster3d_backward_py]("raster3d_backward")
        return m.finalize()
    except e:
        abort(t"failed to create _hexels_mojo: {e}")
