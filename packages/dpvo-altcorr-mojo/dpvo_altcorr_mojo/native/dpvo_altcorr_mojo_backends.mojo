from std.atomic import Atomic
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv, floor
from std.memory import alloc
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.utils.index import Index
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE


# Mojo implementation of DPVO's alternative correlation kernels.
#
# DPVO tracks a sparse patch graph rather than dense optical flow. The altcorr op
# does two things in that graph:
#   1. `patchify_*` crops small descriptor patches around selected keypoints.
#   2. `corr_*` evaluates dot-product correlation on a local search grid around
#      each patch reprojection.
#
# The DPVO and DPV-SLAM papers describe these correlation features as inputs to
# the recurrent update operator before bundle adjustment. Keeping these kernels
# numerically close to the CUDA extension is therefore more important than only
# matching shapes.


# Tensor layouts are row-major because every wrapper calls `.contiguous()` on the
# incoming PyTorch tensors before exposing storage to Mojo. UNKNOWN_VALUE marks
# runtime dimensions while preserving typed multi-dimensional indexing inside
# kernels.
comptime NET4_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)  # [B, C, H, W]
comptime PATCHES5_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)  # [B, N, C, D, D]
comptime PATCH_COORDS3_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 2)  # [B, N, 2]
comptime FMAP5_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)  # [B, F, C, H, W]
comptime EDGE_COORDS5_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 2, UNKNOWN_VALUE, UNKNOWN_VALUE)  # [B, E, 2, Ph, Pw]
comptime EDGE1_LT = Layout.row_major(UNKNOWN_VALUE)  # [E]
comptime CORR6_LT = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)  # [B, E, D, D, Ph, Pw]


# ── Python/PyTorch Interop ────────────────────────────────────────────────────

def _install_cached_context(module: PythonObject) raises:
    var ctx_storage = alloc[DeviceContext](1)
    var cached_ctx = DeviceContext()
    ctx_storage.init_pointee_move(cached_ctx^)
    Python.add_object(module, "_ctx_addr", PythonObject(Int(ctx_storage)))


def _get_cached_context_ptr() raises -> UnsafePointer[DeviceContext, MutAnyOrigin]:
    # `Pointer` cannot be reconstructed from a Python integer address. The
    # process-lifetime DeviceContext is stored on the extension module and
    # recovered as an UnsafePointer only at this interop boundary.
    var module = Python.import_module("dpvo_altcorr_mojo_backends")
    var ctx_addr = Int(py=module._ctx_addr)
    return UnsafePointer[DeviceContext, MutAnyOrigin](unsafe_from_address=ctx_addr)


@always_inline
def _torch_float32_ptr(tensor: PythonObject) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    # PyTorch exposes tensor storage as an integer data_ptr(). We immediately
    # wrap these raw addresses in LayoutTensors at call sites when the kernel can
    # use structured indexing. Safe Pointer is for a single initialized value,
    # not an externally-owned contiguous tensor buffer.
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=tensor.data_ptr()))


@always_inline
def _torch_int64_ptr(tensor: PythonObject) raises -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=tensor.data_ptr()))


@always_inline
def _within_bounds(row: Int, col: Int, height: Int, width: Int) -> Bool:
    return row >= 0 and row < height and col >= 0 and col < width


# ── Patch Extraction ──────────────────────────────────────────────────────────

def patchify_forward_kernel(
    patches: LayoutTensor[DType.float32, PATCHES5_LT, MutAnyOrigin],
    net: LayoutTensor[DType.float32, NET4_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, PATCH_COORDS3_LT, MutAnyOrigin],
    total: Int,
    num_patches: Int,
    channels: Int,
    height: Int,
    width: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var patch_col = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_row = linear_index % diameter
        linear_index = linear_index // diameter
        var channel = linear_index % channels
        linear_index = linear_index // channels
        var patch_id = linear_index % num_patches
        linear_index = linear_index // num_patches
        var batch = linear_index

        # One thread writes one descriptor sample from net into:
        # patches[batch, patch_id, channel, patch_row, patch_col].
        var x = coords[batch, patch_id, 0][0]
        var y = coords[batch, patch_id, 1][0]
        var src_row = Int(floor(y)) + patch_row - radius
        var src_col = Int(floor(x)) + patch_col - radius

        var value: Float32 = 0.0
        if _within_bounds(src_row, src_col, height, width):
            value = net[batch, channel, src_row, src_col][0]

        patches[batch, patch_id, channel, patch_row, patch_col] = value


def patchify_backward_kernel(
    net_grad: UnsafePointer[Float32, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, PATCH_COORDS3_LT, MutAnyOrigin],
    patch_grad: LayoutTensor[DType.float32, PATCHES5_LT, MutAnyOrigin],
    total: Int,
    num_patches: Int,
    channels: Int,
    height: Int,
    width: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var patch_col = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_row = linear_index % diameter
        linear_index = linear_index // diameter
        var channel = linear_index % channels
        linear_index = linear_index // channels
        var patch_id = linear_index % num_patches
        linear_index = linear_index // num_patches
        var batch = linear_index

        var x = coords[batch, patch_id, 0][0]
        var y = coords[batch, patch_id, 1][0]
        var dst_row = Int(floor(y)) + patch_row - radius
        var dst_col = Int(floor(x)) + patch_col - radius

        if _within_bounds(dst_row, dst_col, height, width):
            # Atomic scatter accumulates many patch samples into the same
            # feature-map location. Atomic.fetch_add still needs a raw element
            # address, so only this destination buffer stays as UnsafePointer.
            var out_offset = ((batch * channels + channel) * height + dst_row) * width + dst_col
            _ = Atomic.fetch_add(
                net_grad + out_offset,
                patch_grad[batch, patch_id, channel, patch_row, patch_col][0],
            )


# ── Integer-grid Correlation ──────────────────────────────────────────────────

def corr_forward_raw_kernel(
    corr: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    fmap1: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    fmap2: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    channels: Int,
    patch_h: Int,
    patch_w: Int,
    fmap1_frames: Int,
    fmap2_frames: Int,
    fmap2_h: Int,
    fmap2_w: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var search_col = linear_index % diameter
        linear_index = linear_index // diameter
        var search_row = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        # Correlate one source patch pixel against one destination search-grid
        # location. This is Eq. 4 in the DPVO/DPV-SLAM descriptions: an inner
        # product between patch features and destination-frame features.
        var src_frame = Int(ii[edge])
        var dst_frame = Int(jj[edge])
        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var tgt_row = Int(floor(y)) + search_row - radius
        var tgt_col = Int(floor(x)) + search_col - radius

        var acc: Float32 = 0.0
        if _within_bounds(tgt_row, tgt_col, fmap2_h, fmap2_w):
            for channel in range(channels):
                acc += (
                    fmap1[batch, src_frame, channel, patch_row, patch_col][0]
                    * fmap2[batch, dst_frame, channel, tgt_row, tgt_col][0]
                )

        corr[batch, edge, search_row, search_col, patch_row, patch_col] = acc


def corr_forward_raw_kernel_c128(
    corr: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    fmap1: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    fmap2: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    patch_h: Int,
    patch_w: Int,
    fmap1_frames: Int,
    fmap2_frames: Int,
    fmap2_h: Int,
    fmap2_w: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var search_col = linear_index % diameter
        linear_index = linear_index // diameter
        var search_row = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        var src_frame = Int(ii[edge])
        var dst_frame = Int(jj[edge])
        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var tgt_row = Int(floor(y)) + search_row - radius
        var tgt_col = Int(floor(x)) + search_col - radius

        var acc: Float32 = 0.0
        if _within_bounds(tgt_row, tgt_col, fmap2_h, fmap2_w):
            # The DPVO model uses 128-channel correlation features in the hot
            # path. Keeping the channel loop compile-time unrolled preserves the
            # CUDA extension's specialization while LayoutTensor removes the
            # hand-written stride math.
            comptime for channel in range(128):
                acc += (
                    fmap1[batch, src_frame, channel, patch_row, patch_col][0]
                    * fmap2[batch, dst_frame, channel, tgt_row, tgt_col][0]
                )

        corr[batch, edge, search_row, search_col, patch_row, patch_col] = acc


# ── Correlation Backward Helpers ──────────────────────────────────────────────

def corr_backward_raw_kernel(
    fmap1_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap2_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap1: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    fmap2: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    corr_grad: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    channels: Int,
    patch_h: Int,
    patch_w: Int,
    fmap1_frames: Int,
    fmap2_frames: Int,
    fmap2_h: Int,
    fmap2_w: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var channel = linear_index % channels
        linear_index = linear_index // channels
        var search_col = linear_index % diameter
        linear_index = linear_index // diameter
        var search_row = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        var src_frame = Int(ii[edge])
        var dst_frame = Int(jj[edge])
        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var tgt_row = Int(floor(y)) + search_row - radius
        var tgt_col = Int(floor(x)) + search_col - radius

        if _within_bounds(tgt_row, tgt_col, fmap2_h, fmap2_w):
            var grad = corr_grad[batch, edge, search_row, search_col, patch_row, patch_col][0]
            var f1_offset = ((((batch * fmap1_frames + src_frame) * channels + channel) * patch_h + patch_row) * patch_w + patch_col)
            var f2_offset = ((((batch * fmap2_frames + dst_frame) * channels + channel) * fmap2_h + tgt_row) * fmap2_w + tgt_col)
            _ = Atomic.fetch_add(
                fmap1_grad + f1_offset,
                grad * fmap2[batch, dst_frame, channel, tgt_row, tgt_col][0],
            )
            _ = Atomic.fetch_add(
                fmap2_grad + f2_offset,
                grad * fmap1[batch, src_frame, channel, patch_row, patch_col][0],
            )


def corr_backward_raw_kernel_loop_channels_c128(
    fmap1_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap2_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap1: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    fmap2: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    corr_grad: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    patch_h: Int,
    patch_w: Int,
    fmap1_frames: Int,
    fmap2_frames: Int,
    fmap2_h: Int,
    fmap2_w: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var search_col = linear_index % diameter
        linear_index = linear_index // diameter
        var search_row = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        var src_frame = Int(ii[edge])
        var dst_frame = Int(jj[edge])
        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var tgt_row = Int(floor(y)) + search_row - radius
        var tgt_col = Int(floor(x)) + search_col - radius

        if _within_bounds(tgt_row, tgt_col, fmap2_h, fmap2_w):
            var grad = corr_grad[batch, edge, search_row, search_col, patch_row, patch_col][0]
            comptime for channel in range(128):
                var f1_offset = ((((batch * fmap1_frames + src_frame) * 128 + channel) * patch_h + patch_row) * patch_w + patch_col)
                var f2_offset = ((((batch * fmap2_frames + dst_frame) * 128 + channel) * fmap2_h + tgt_row) * fmap2_w + tgt_col)
                _ = Atomic.fetch_add(
                    fmap1_grad + f1_offset,
                    grad * fmap2[batch, dst_frame, channel, tgt_row, tgt_col][0],
                )
                _ = Atomic.fetch_add(
                    fmap2_grad + f2_offset,
                    grad * fmap1[batch, src_frame, channel, patch_row, patch_col][0],
                )


def corr_backward_raw_kernel_loop_channels(
    fmap1_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap2_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap1: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    fmap2: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    corr_grad: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    channels: Int,
    patch_h: Int,
    patch_w: Int,
    fmap1_frames: Int,
    fmap2_frames: Int,
    fmap2_h: Int,
    fmap2_w: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var search_col = linear_index % diameter
        linear_index = linear_index // diameter
        var search_row = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        var src_frame = Int(ii[edge])
        var dst_frame = Int(jj[edge])
        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var tgt_row = Int(floor(y)) + search_row - radius
        var tgt_col = Int(floor(x)) + search_col - radius

        if _within_bounds(tgt_row, tgt_col, fmap2_h, fmap2_w):
            var grad = corr_grad[batch, edge, search_row, search_col, patch_row, patch_col][0]
            for channel in range(channels):
                var f1_offset = ((((batch * fmap1_frames + src_frame) * channels + channel) * patch_h + patch_row) * patch_w + patch_col)
                var f2_offset = ((((batch * fmap2_frames + dst_frame) * channels + channel) * fmap2_h + tgt_row) * fmap2_w + tgt_col)
                _ = Atomic.fetch_add(
                    fmap1_grad + f1_offset,
                    grad * fmap2[batch, dst_frame, channel, tgt_row, tgt_col][0],
                )
                _ = Atomic.fetch_add(
                    fmap2_grad + f2_offset,
                    grad * fmap1[batch, src_frame, channel, patch_row, patch_col][0],
                )


def corr_interpolate_kernel(
    output: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    corr: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    patch_h: Int,
    patch_w: Int,
    diameter: Int,
):
    var tid = global_idx.x
    if tid < total:
        var d_out = diameter - 1
        var linear_index = tid
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var search_y = linear_index % d_out
        linear_index = linear_index // d_out
        var search_x = linear_index % d_out
        linear_index = linear_index // d_out
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        # Bilinear interpolation turns a D x D integer-grid correlation volume
        # into the (D-1) x (D-1) feature block consumed by DPVO's update operator.
        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var dx = x - floor(x)
        var dy = y - floor(y)

        var value = (
            (1.0 - dx) * (1.0 - dy)
            * corr[batch, edge, search_y, search_x, patch_row, patch_col][0]
            + dx
            * (1.0 - dy)
            * corr[batch, edge, search_y, search_x + 1, patch_row, patch_col][0]
            + (1.0 - dx)
            * dy
            * corr[batch, edge, search_y + 1, search_x, patch_row, patch_col][0]
            + dx
            * dy
            * corr[batch, edge, search_y + 1, search_x + 1, patch_row, patch_col][0]
        )

        output[batch, edge, search_x, search_y, patch_row, patch_col] = value


# ── Interpolation Backward Helpers ────────────────────────────────────────────

def corr_expand_grad_kernel(
    corr_grad: UnsafePointer[Float32, MutAnyOrigin],
    grad: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    patch_h: Int,
    patch_w: Int,
    diameter: Int,
):
    var tid = global_idx.x
    if tid < total:
        var d_out = diameter - 1
        var linear_index = tid
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var search_y = linear_index % d_out
        linear_index = linear_index // d_out
        var search_x = linear_index % d_out
        linear_index = linear_index // d_out
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var dx = x - floor(x)
        var dy = y - floor(y)

        var value = grad[batch, edge, search_x, search_y, patch_row, patch_col][0]

        # This helper mirrors the backward of corr_interpolate_kernel. It is not
        # used in the fused hot path below, but keeping it documented makes the
        # math easy to compare against the two-stage CUDA implementation.
        var c00 = (((((batch * edges + edge) * diameter + search_y) * diameter + search_x) * patch_h + patch_row) * patch_w + patch_col)
        var c01 = (((((batch * edges + edge) * diameter + search_y) * diameter + search_x + 1) * patch_h + patch_row) * patch_w + patch_col)
        var c10 = (((((batch * edges + edge) * diameter + search_y + 1) * diameter + search_x) * patch_h + patch_row) * patch_w + patch_col)
        var c11 = (((((batch * edges + edge) * diameter + search_y + 1) * diameter + search_x + 1) * patch_h + patch_row) * patch_w + patch_col)
        _ = Atomic.fetch_add(corr_grad + c00, (1.0 - dx) * (1.0 - dy) * value)
        _ = Atomic.fetch_add(corr_grad + c01, dx * (1.0 - dy) * value)
        _ = Atomic.fetch_add(corr_grad + c10, (1.0 - dx) * dy * value)
        _ = Atomic.fetch_add(corr_grad + c11, dx * dy * value)


# ── Fused Hot Backward Path ───────────────────────────────────────────────────

def corr_backward_interpolated_kernel_loop_channels(
    fmap1_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap2_grad: UnsafePointer[Float32, MutAnyOrigin],
    fmap1: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    fmap2: LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin],
    coords: LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    grad: LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin],
    total: Int,
    edges: Int,
    channels: Int,
    patch_h: Int,
    patch_w: Int,
    fmap1_frames: Int,
    fmap2_frames: Int,
    fmap2_h: Int,
    fmap2_w: Int,
    diameter: Int,
    radius: Int,
):
    var tid = global_idx.x
    if tid < total:
        var d_out = diameter - 1
        var linear_index = tid
        var search_col = linear_index % diameter
        linear_index = linear_index // diameter
        var search_row = linear_index % diameter
        linear_index = linear_index // diameter
        var patch_col = linear_index % patch_w
        linear_index = linear_index // patch_w
        var patch_row = linear_index % patch_h
        linear_index = linear_index // patch_h
        var edge = linear_index % edges
        linear_index = linear_index // edges
        var batch = linear_index

        var src_frame = Int(ii[edge])
        var dst_frame = Int(jj[edge])
        var x = coords[batch, edge, 0, patch_row, patch_col][0]
        var y = coords[batch, edge, 1, patch_row, patch_col][0]
        var tgt_row = Int(floor(y)) + search_row - radius
        var tgt_col = Int(floor(x)) + search_col - radius

        if _within_bounds(tgt_row, tgt_col, fmap2_h, fmap2_w):
            var dx = x - floor(x)
            var dy = y - floor(y)
            var corr_grad_value: Float32 = 0.0

            # Fused interpolation-backward: each raw D x D correlation sample is
            # influenced by up to four neighboring output samples. Computing the
            # weighted gradient here avoids materializing the full corr_grad
            # tensor and keeps the no-fallback speed gate within tolerance.
            if search_row < d_out and search_col < d_out:
                corr_grad_value += (
                    (1.0 - dx)
                    * (1.0 - dy)
                    * grad[batch, edge, search_col, search_row, patch_row, patch_col][0]
                )
            if search_row < d_out and search_col > 0:
                corr_grad_value += (
                    dx
                    * (1.0 - dy)
                    * grad[batch, edge, search_col - 1, search_row, patch_row, patch_col][0]
                )
            if search_row > 0 and search_col < d_out:
                corr_grad_value += (
                    (1.0 - dx)
                    * dy
                    * grad[batch, edge, search_col, search_row - 1, patch_row, patch_col][0]
                )
            if search_row > 0 and search_col > 0:
                corr_grad_value += (
                    dx
                    * dy
                    * grad[batch, edge, search_col - 1, search_row - 1, patch_row, patch_col][0]
                )

            for channel in range(channels):
                var f1_offset = ((((batch * fmap1_frames + src_frame) * channels + channel) * patch_h + patch_row) * patch_w + patch_col)
                var f2_offset = ((((batch * fmap2_frames + dst_frame) * channels + channel) * fmap2_h + tgt_row) * fmap2_w + tgt_col)
                _ = Atomic.fetch_add(
                    fmap1_grad + f1_offset,
                    corr_grad_value * fmap2[batch, dst_frame, channel, tgt_row, tgt_col][0],
                )
                _ = Atomic.fetch_add(
                    fmap2_grad + f2_offset,
                    corr_grad_value * fmap1[batch, src_frame, channel, patch_row, patch_col][0],
                )


# ── Python Module Wrappers ────────────────────────────────────────────────────

def patchify_forward(
    net_obj: PythonObject,
    coords_obj: PythonObject,
    radius_obj: PythonObject,
) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var net = net_obj.contiguous()
    var coords = coords_obj.contiguous()

    var batch: Int = Int(py=net.shape[0])
    var channels: Int = Int(py=net.shape[1])
    var height: Int = Int(py=net.shape[2])
    var width: Int = Int(py=net.shape[3])
    var num_patches: Int = Int(py=coords.shape[1])
    var radius: Int = Int(py=radius_obj)
    var diameter: Int = 2 * radius + 2
    var total: Int = batch * num_patches * channels * diameter * diameter

    var patches: PythonObject = torch.empty(
        Python.list(batch, num_patches, channels, diameter, diameter),
        device=net.device,
        dtype=torch.float32,
    )

    # PyTorch owns the storage; Mojo receives a non-owning view with the same
    # row-major shape. This is the only unsafe pointer boundary for this kernel.
    var patches_lt = LayoutTensor[DType.float32, PATCHES5_LT, MutAnyOrigin](
        _torch_float32_ptr(patches),
        RuntimeLayout[PATCHES5_LT].row_major(Index(batch, num_patches, channels, diameter, diameter)),
    )
    var net_lt = LayoutTensor[DType.float32, NET4_LT, MutAnyOrigin](
        _torch_float32_ptr(net),
        RuntimeLayout[NET4_LT].row_major(Index(batch, channels, height, width)),
    )
    var coords_lt = LayoutTensor[DType.float32, PATCH_COORDS3_LT, MutAnyOrigin](
        _torch_float32_ptr(coords),
        RuntimeLayout[PATCH_COORDS3_LT].row_major(Index(batch, num_patches, 2)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    ctx_ptr[].enqueue_function[patchify_forward_kernel, patchify_forward_kernel](
        patches_lt,
        net_lt,
        coords_lt,
        total,
        num_patches,
        channels,
        height,
        width,
        diameter,
        radius,
        grid_dim=ceildiv(total, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    torch.cuda.synchronize(device=net.device)
    return Python.tuple(patches)


def patchify_backward(
    net_obj: PythonObject,
    coords_obj: PythonObject,
    grad_obj: PythonObject,
    radius_obj: PythonObject,
) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var net = net_obj.contiguous()
    var coords = coords_obj.contiguous()
    var grad = grad_obj.contiguous()

    var batch: Int = Int(py=net.shape[0])
    var channels: Int = Int(py=net.shape[1])
    var height: Int = Int(py=net.shape[2])
    var width: Int = Int(py=net.shape[3])
    var num_patches: Int = Int(py=coords.shape[1])
    var radius: Int = Int(py=radius_obj)
    var diameter: Int = 2 * radius + 2
    var total: Int = batch * num_patches * channels * diameter * diameter

    var net_grad: PythonObject = torch.zeros_like(net)

    var coords_lt = LayoutTensor[DType.float32, PATCH_COORDS3_LT, MutAnyOrigin](
        _torch_float32_ptr(coords),
        RuntimeLayout[PATCH_COORDS3_LT].row_major(Index(batch, num_patches, 2)),
    )
    var grad_lt = LayoutTensor[DType.float32, PATCHES5_LT, MutAnyOrigin](
        _torch_float32_ptr(grad),
        RuntimeLayout[PATCHES5_LT].row_major(Index(batch, num_patches, channels, diameter, diameter)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 512
    ctx_ptr[].enqueue_function[patchify_backward_kernel, patchify_backward_kernel](
        _torch_float32_ptr(net_grad),
        coords_lt,
        grad_lt,
        total,
        num_patches,
        channels,
        height,
        width,
        diameter,
        radius,
        grid_dim=ceildiv(total, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    torch.cuda.synchronize(device=net.device)
    return Python.tuple(net_grad)


def corr_forward_raw(
    fmap1_obj: PythonObject,
    fmap2_obj: PythonObject,
    coords_obj: PythonObject,
    ii_obj: PythonObject,
    jj_obj: PythonObject,
    radius_obj: PythonObject,
) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var fmap1 = fmap1_obj.contiguous()
    var fmap2 = fmap2_obj.contiguous()
    var coords = coords_obj.contiguous()
    var ii = ii_obj.contiguous()
    var jj = jj_obj.contiguous()

    var batch: Int = Int(py=coords.shape[0])
    var edges: Int = Int(py=coords.shape[1])
    var channels: Int = Int(py=fmap1.shape[2])
    var patch_h: Int = Int(py=coords.shape[3])
    var patch_w: Int = Int(py=coords.shape[4])
    var fmap1_frames: Int = Int(py=fmap1.shape[1])
    var fmap2_frames: Int = Int(py=fmap2.shape[1])
    var fmap2_h: Int = Int(py=fmap2.shape[3])
    var fmap2_w: Int = Int(py=fmap2.shape[4])
    var radius: Int = Int(py=radius_obj)
    var diameter: Int = 2 * radius + 2
    var total: Int = batch * edges * patch_h * patch_w * diameter * diameter

    var corr: PythonObject = torch.empty(
        Python.list(batch, edges, diameter, diameter, patch_h, patch_w),
        device=fmap1.device,
        dtype=torch.float32,
    )

    var corr_lt = LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin](
        _torch_float32_ptr(corr),
        RuntimeLayout[CORR6_LT].row_major(Index(batch, edges, diameter, diameter, patch_h, patch_w)),
    )
    var fmap1_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap1),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap1_frames, channels, patch_h, patch_w)),
    )
    var fmap2_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap2),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap2_frames, channels, fmap2_h, fmap2_w)),
    )
    var coords_lt = LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin](
        _torch_float32_ptr(coords),
        RuntimeLayout[EDGE_COORDS5_LT].row_major(Index(batch, edges, 2, patch_h, patch_w)),
    )
    var ii_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(ii),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var jj_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(jj),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    if channels == 128:
        ctx_ptr[].enqueue_function[corr_forward_raw_kernel_c128, corr_forward_raw_kernel_c128](
            corr_lt,
            fmap1_lt,
            fmap2_lt,
            coords_lt,
            ii_lt,
            jj_lt,
            total,
            edges,
            patch_h,
            patch_w,
            fmap1_frames,
            fmap2_frames,
            fmap2_h,
            fmap2_w,
            diameter,
            radius,
            grid_dim=ceildiv(total, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
    else:
        ctx_ptr[].enqueue_function[corr_forward_raw_kernel, corr_forward_raw_kernel](
            corr_lt,
            fmap1_lt,
            fmap2_lt,
            coords_lt,
            ii_lt,
            jj_lt,
            total,
            edges,
            channels,
            patch_h,
            patch_w,
            fmap1_frames,
            fmap2_frames,
            fmap2_h,
            fmap2_w,
            diameter,
            radius,
            grid_dim=ceildiv(total, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
    torch.cuda.synchronize(device=fmap1.device)
    return Python.tuple(corr)


def corr_forward(
    fmap1_obj: PythonObject,
    fmap2_obj: PythonObject,
    coords_obj: PythonObject,
    ii_obj: PythonObject,
    jj_obj: PythonObject,
    radius_obj: PythonObject,
) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var fmap1 = fmap1_obj.contiguous()
    var fmap2 = fmap2_obj.contiguous()
    var coords = coords_obj.contiguous()
    var ii = ii_obj.contiguous()
    var jj = jj_obj.contiguous()

    var batch: Int = Int(py=coords.shape[0])
    var edges: Int = Int(py=coords.shape[1])
    var channels: Int = Int(py=fmap1.shape[2])
    var patch_h: Int = Int(py=coords.shape[3])
    var patch_w: Int = Int(py=coords.shape[4])
    var fmap1_frames: Int = Int(py=fmap1.shape[1])
    var fmap2_frames: Int = Int(py=fmap2.shape[1])
    var fmap2_h: Int = Int(py=fmap2.shape[3])
    var fmap2_w: Int = Int(py=fmap2.shape[4])
    var radius: Int = Int(py=radius_obj)
    var diameter: Int = 2 * radius + 2
    var d_out: Int = diameter - 1
    var corr_total: Int = batch * edges * patch_h * patch_w * diameter * diameter
    var out_total: Int = batch * edges * patch_h * patch_w * d_out * d_out

    var corr: PythonObject = torch.empty(
        Python.list(batch, edges, diameter, diameter, patch_h, patch_w),
        device=fmap1.device,
        dtype=torch.float32,
    )
    var output: PythonObject = torch.empty(
        Python.list(batch, edges, d_out, d_out, patch_h, patch_w),
        device=fmap1.device,
        dtype=torch.float32,
    )

    var corr_lt = LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin](
        _torch_float32_ptr(corr),
        RuntimeLayout[CORR6_LT].row_major(Index(batch, edges, diameter, diameter, patch_h, patch_w)),
    )
    var output_lt = LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin](
        _torch_float32_ptr(output),
        RuntimeLayout[CORR6_LT].row_major(Index(batch, edges, d_out, d_out, patch_h, patch_w)),
    )
    var fmap1_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap1),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap1_frames, channels, patch_h, patch_w)),
    )
    var fmap2_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap2),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap2_frames, channels, fmap2_h, fmap2_w)),
    )
    var coords_lt = LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin](
        _torch_float32_ptr(coords),
        RuntimeLayout[EDGE_COORDS5_LT].row_major(Index(batch, edges, 2, patch_h, patch_w)),
    )
    var ii_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(ii),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var jj_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(jj),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    if channels == 128:
        ctx_ptr[].enqueue_function[corr_forward_raw_kernel_c128, corr_forward_raw_kernel_c128](
            corr_lt,
            fmap1_lt,
            fmap2_lt,
            coords_lt,
            ii_lt,
            jj_lt,
            corr_total,
            edges,
            patch_h,
            patch_w,
            fmap1_frames,
            fmap2_frames,
            fmap2_h,
            fmap2_w,
            diameter,
            radius,
            grid_dim=ceildiv(corr_total, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
    else:
        ctx_ptr[].enqueue_function[corr_forward_raw_kernel, corr_forward_raw_kernel](
            corr_lt,
            fmap1_lt,
            fmap2_lt,
            coords_lt,
            ii_lt,
            jj_lt,
            corr_total,
            edges,
            channels,
            patch_h,
            patch_w,
            fmap1_frames,
            fmap2_frames,
            fmap2_h,
            fmap2_w,
            diameter,
            radius,
            grid_dim=ceildiv(corr_total, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
    ctx_ptr[].enqueue_function[corr_interpolate_kernel, corr_interpolate_kernel](
        output_lt,
        corr_lt,
        coords_lt,
        out_total,
        edges,
        patch_h,
        patch_w,
        diameter,
        grid_dim=ceildiv(out_total, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    torch.cuda.synchronize(device=fmap1.device)
    return Python.tuple(output)


def corr_backward_raw(args_obj: PythonObject) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var fmap1 = args_obj[0].contiguous()
    var fmap2 = args_obj[1].contiguous()
    var coords = args_obj[2].contiguous()
    var ii = args_obj[3].contiguous()
    var jj = args_obj[4].contiguous()
    var corr_grad = args_obj[5].contiguous()

    var batch: Int = Int(py=coords.shape[0])
    var edges: Int = Int(py=coords.shape[1])
    var channels: Int = Int(py=fmap1.shape[2])
    var patch_h: Int = Int(py=coords.shape[3])
    var patch_w: Int = Int(py=coords.shape[4])
    var fmap1_frames: Int = Int(py=fmap1.shape[1])
    var fmap2_frames: Int = Int(py=fmap2.shape[1])
    var fmap2_h: Int = Int(py=fmap2.shape[3])
    var fmap2_w: Int = Int(py=fmap2.shape[4])
    var radius: Int = Int(py=args_obj[6])
    var diameter: Int = 2 * radius + 2
    var total: Int = batch * edges * patch_h * patch_w * diameter * diameter

    var fmap1_grad: PythonObject = torch.zeros_like(fmap1)
    var fmap2_grad: PythonObject = torch.zeros_like(fmap2)

    var fmap1_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap1),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap1_frames, channels, patch_h, patch_w)),
    )
    var fmap2_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap2),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap2_frames, channels, fmap2_h, fmap2_w)),
    )
    var coords_lt = LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin](
        _torch_float32_ptr(coords),
        RuntimeLayout[EDGE_COORDS5_LT].row_major(Index(batch, edges, 2, patch_h, patch_w)),
    )
    var ii_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(ii),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var jj_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(jj),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var corr_grad_lt = LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin](
        _torch_float32_ptr(corr_grad),
        RuntimeLayout[CORR6_LT].row_major(Index(batch, edges, diameter, diameter, patch_h, patch_w)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    ctx_ptr[].enqueue_function[corr_backward_raw_kernel_loop_channels, corr_backward_raw_kernel_loop_channels](
        _torch_float32_ptr(fmap1_grad),
        _torch_float32_ptr(fmap2_grad),
        fmap1_lt,
        fmap2_lt,
        coords_lt,
        ii_lt,
        jj_lt,
        corr_grad_lt,
        total,
        edges,
        channels,
        patch_h,
        patch_w,
        fmap1_frames,
        fmap2_frames,
        fmap2_h,
        fmap2_w,
        diameter,
        radius,
        grid_dim=ceildiv(total, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    torch.cuda.synchronize(device=fmap1.device)
    return Python.tuple(fmap1_grad, fmap2_grad)


def corr_backward(args_obj: PythonObject) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var fmap1 = args_obj[0].contiguous()
    var fmap2 = args_obj[1].contiguous()
    var coords = args_obj[2].contiguous()
    var ii = args_obj[3].contiguous()
    var jj = args_obj[4].contiguous()
    var grad = args_obj[5].contiguous()

    var batch: Int = Int(py=coords.shape[0])
    var edges: Int = Int(py=coords.shape[1])
    var channels: Int = Int(py=fmap1.shape[2])
    var patch_h: Int = Int(py=coords.shape[3])
    var patch_w: Int = Int(py=coords.shape[4])
    var fmap1_frames: Int = Int(py=fmap1.shape[1])
    var fmap2_frames: Int = Int(py=fmap2.shape[1])
    var fmap2_h: Int = Int(py=fmap2.shape[3])
    var fmap2_w: Int = Int(py=fmap2.shape[4])
    var radius: Int = Int(py=args_obj[6])
    var diameter: Int = 2 * radius + 2
    var scatter_total: Int = batch * edges * patch_h * patch_w * diameter * diameter

    var fmap1_grad: PythonObject = torch.zeros_like(fmap1)
    var fmap2_grad: PythonObject = torch.zeros_like(fmap2)

    var fmap1_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap1),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap1_frames, channels, patch_h, patch_w)),
    )
    var fmap2_lt = LayoutTensor[DType.float32, FMAP5_LT, MutAnyOrigin](
        _torch_float32_ptr(fmap2),
        RuntimeLayout[FMAP5_LT].row_major(Index(batch, fmap2_frames, channels, fmap2_h, fmap2_w)),
    )
    var coords_lt = LayoutTensor[DType.float32, EDGE_COORDS5_LT, MutAnyOrigin](
        _torch_float32_ptr(coords),
        RuntimeLayout[EDGE_COORDS5_LT].row_major(Index(batch, edges, 2, patch_h, patch_w)),
    )
    var ii_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(ii),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var jj_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(jj),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var grad_lt = LayoutTensor[DType.float32, CORR6_LT, MutAnyOrigin](
        _torch_float32_ptr(grad),
        RuntimeLayout[CORR6_LT].row_major(Index(batch, edges, diameter - 1, diameter - 1, patch_h, patch_w)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    ctx_ptr[].enqueue_function[corr_backward_interpolated_kernel_loop_channels, corr_backward_interpolated_kernel_loop_channels](
        _torch_float32_ptr(fmap1_grad),
        _torch_float32_ptr(fmap2_grad),
        fmap1_lt,
        fmap2_lt,
        coords_lt,
        ii_lt,
        jj_lt,
        grad_lt,
        scatter_total,
        edges,
        channels,
        patch_h,
        patch_w,
        fmap1_frames,
        fmap2_frames,
        fmap2_h,
        fmap2_w,
        diameter,
        radius,
        grid_dim=ceildiv(scatter_total, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    torch.cuda.synchronize(device=fmap1.device)
    return Python.tuple(fmap1_grad, fmap2_grad)


# ── Python Module Export ──────────────────────────────────────────────────────

@export
def PyInit_dpvo_altcorr_mojo_backends() -> PythonObject:
    try:
        var m = PythonModuleBuilder("dpvo_altcorr_mojo_backends")
        m.def_function[patchify_forward]("patchify_forward")
        m.def_function[patchify_backward]("patchify_backward")
        m.def_function[corr_forward]("corr_forward")
        m.def_function[corr_backward]("corr_backward")
        m.def_function[corr_forward_raw]("corr_forward_raw")
        m.def_function[corr_backward_raw]("corr_backward_raw")
        var module = m.finalize()
        _install_cached_context(module)
        return module
    except e:
        abort(String("Failed to create dpvo_altcorr_mojo_backends module: ", e))
