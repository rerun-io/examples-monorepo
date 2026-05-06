from compiler import register
from tensor import InputTensor, OutputTensor
from std.gpu import global_idx
from std.math import ceildiv, floor
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList


# MAX CustomOpLibrary fallback kernels for DPVO altcorr.
#
# The production path is the PythonModuleBuilder native extension in
# `native/dpvo_altcorr_mojo_backends.mojo`. These custom ops stay here for smoke
# tests and as a readable reference implementation when the native `.so` is not
# present. Keep the registered names stable: Python looks them up by string.
#
# Shape legend used below:
#   net          [B, C, H, W]
#   patch coords [B, N, 2]
#   patches      [B, N, C, D, D] where D = 2 * radius + 2
#   feature maps [B, F, C, H, W]
#   edge coords  [B, E, 2, Ph, Pw]
#   corr         [B, E, S, S, Ph, Pw] where S = 2 * radius + 1


@always_inline
def _within_bounds(row: Int, col: Int, height: Int, width: Int) -> Bool:
    return row >= 0 and row < height and col >= 0 and col < width


@always_inline
def _load_f32_1d(tensor: InputTensor[dtype=DType.float32, rank=1, ...], i0: Int) -> Float32:
    return tensor.load[1](IndexList[1](i0))[0]


@always_inline
def _load_i64_1d(tensor: InputTensor[dtype=DType.int64, rank=1, ...], i0: Int) -> Int:
    return Int(tensor.load[1](IndexList[1](i0))[0])


@always_inline
def _load_f32_3d(tensor: InputTensor[dtype=DType.float32, rank=3, ...], i0: Int, i1: Int, i2: Int) -> Float32:
    return tensor.load[1](IndexList[3](i0, i1, i2))[0]


@always_inline
def _load_f32_4d(tensor: InputTensor[dtype=DType.float32, rank=4, ...], i0: Int, i1: Int, i2: Int, i3: Int) -> Float32:
    return tensor.load[1](IndexList[4](i0, i1, i2, i3))[0]


@always_inline
def _load_f32_5d(tensor: InputTensor[dtype=DType.float32, rank=5, ...], i0: Int, i1: Int, i2: Int, i3: Int, i4: Int) -> Float32:
    return tensor.load[1](IndexList[5](i0, i1, i2, i3, i4))[0]


@always_inline
def _load_f32_6d(tensor: InputTensor[dtype=DType.float32, rank=6, ...], i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int) -> Float32:
    return tensor.load[1](IndexList[6](i0, i1, i2, i3, i4, i5))[0]


@register("altcorr_smoke_scale")
struct AltCorrSmokeScale:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=1, ...],
        x: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("altcorr_smoke_scale currently requires a GPU tensor")
        var gpu_ctx = ctx.get_device_context()
        var element_count = output.dim_size[0]()
        comptime BLOCK_SIZE = 256

        @parameter
        def smoke_scale_kernel(element_count: Int):
            var tid = global_idx.x
            if tid < element_count:
                var idx = tid
                var value = _load_f32_1d(x, idx) * 2.0
                output.store[1](IndexList[1](idx), SIMD[DType.float32, 1](value))

        gpu_ctx.enqueue_function_experimental[smoke_scale_kernel](
            element_count,
            grid_dim=ceildiv(element_count, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )


@register("patchify_forward")
struct PatchifyForward[radius: Int]:
    @staticmethod
    def execute[target: StaticString](
        patches: OutputTensor[dtype=DType.float32, rank=5, ...],
        net: InputTensor[dtype=DType.float32, rank=4, ...],
        coords: InputTensor[dtype=DType.float32, rank=3, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("patchify_forward currently requires GPU tensors")
        var gpu_ctx = ctx.get_device_context()
        # One GPU thread owns one patch spatial location and loops over channels.
        var patch_locations = (
            patches.dim_size[0]()
            * patches.dim_size[1]()
            * patches.dim_size[3]()
            * patches.dim_size[4]()
        )
        var channels = patches.dim_size[2]()
        comptime PATCH_DIAMETER = 2 * Self.radius + 2
        comptime BLOCK_SIZE = 256

        @parameter
        def patchify_forward_kernel(patch_locations: Int, channels: Int):
            var tid = global_idx.x
            if tid < patch_locations:
                var linear_index = tid
                var patch_col = linear_index % PATCH_DIAMETER
                linear_index = linear_index // PATCH_DIAMETER
                var patch_row = linear_index % PATCH_DIAMETER
                linear_index = linear_index // PATCH_DIAMETER
                var patch_id = linear_index % coords.dim_size[1]()
                linear_index = linear_index // coords.dim_size[1]()
                var batch = linear_index

                var x = _load_f32_3d(coords, batch, patch_id, 0)
                var y = _load_f32_3d(coords, batch, patch_id, 1)
                var src_row = Int(floor(y)) + patch_row - Self.radius
                var src_col = Int(floor(x)) + patch_col - Self.radius
                for channel in range(channels):
                    var value: Float32 = 0.0
                    if _within_bounds(src_row, src_col, net.dim_size[2](), net.dim_size[3]()):
                        value = _load_f32_4d(net, batch, channel, src_row, src_col)
                    patches.store[1](IndexList[5](batch, patch_id, channel, patch_row, patch_col), SIMD[DType.float32, 1](value))

        gpu_ctx.enqueue_function_experimental[patchify_forward_kernel](
            patch_locations,
            channels,
            grid_dim=ceildiv(patch_locations, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )


@register("patchify_backward")
struct PatchifyBackward[radius: Int]:
    @staticmethod
    def execute[target: StaticString](
        net_grad: OutputTensor[dtype=DType.float32, rank=4, ...],
        coords: InputTensor[dtype=DType.float32, rank=3, ...],
        patch_grad: InputTensor[dtype=DType.float32, rank=5, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("patchify_backward currently requires GPU tensors")
        var gpu_ctx = ctx.get_device_context()
        # One GPU thread owns one output feature-map pixel and gathers every
        # patch sample that lands on that pixel.
        var feature_pixels = (
            net_grad.dim_size[0]()
            * net_grad.dim_size[1]()
            * net_grad.dim_size[2]()
            * net_grad.dim_size[3]()
        )
        var channels = net_grad.dim_size[1]()
        var height = net_grad.dim_size[2]()
        var width = net_grad.dim_size[3]()
        var num_patches = coords.dim_size[1]()
        comptime PATCH_DIAMETER = 2 * Self.radius + 2
        comptime BLOCK_SIZE = 256

        @parameter
        def patchify_backward_kernel(feature_pixels: Int, channels: Int, height: Int, width: Int, num_patches: Int):
            var tid = global_idx.x
            if tid < feature_pixels:
                var linear_index = tid
                var dst_col = linear_index % width
                linear_index = linear_index // width
                var dst_row = linear_index % height
                linear_index = linear_index // height
                var channel = linear_index % channels
                linear_index = linear_index // channels
                var batch = linear_index

                var acc: Float32 = 0.0
                for patch_id in range(num_patches):
                    var x = _load_f32_3d(coords, batch, patch_id, 0)
                    var y = _load_f32_3d(coords, batch, patch_id, 1)
                    var base_row = Int(floor(y)) - Self.radius
                    var base_col = Int(floor(x)) - Self.radius
                    var patch_row = dst_row - base_row
                    var patch_col = dst_col - base_col
                    if patch_row >= 0 and patch_row < PATCH_DIAMETER and patch_col >= 0 and patch_col < PATCH_DIAMETER:
                        acc += _load_f32_5d(patch_grad, batch, patch_id, channel, patch_row, patch_col)
                net_grad.store[1](IndexList[4](batch, channel, dst_row, dst_col), SIMD[DType.float32, 1](acc))

        gpu_ctx.enqueue_function_experimental[patchify_backward_kernel](
            feature_pixels,
            channels,
            height,
            width,
            num_patches,
            grid_dim=ceildiv(feature_pixels, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )


@always_inline
def _descriptor_dot_at_target_pixel(
    fmap1: InputTensor[dtype=DType.float32, rank=5, ...],
    fmap2: InputTensor[dtype=DType.float32, rank=5, ...],
    batch: Int,
    src_frame: Int,
    dst_frame: Int,
    patch_row: Int,
    patch_col: Int,
    tgt_row: Int,
    tgt_col: Int,
) -> Float32:
    # Dot product between one source patch descriptor and one candidate target
    # pixel. Out-of-image candidates contribute zero, matching the CUDA op.
    var acc: Float32 = 0.0
    if _within_bounds(tgt_row, tgt_col, fmap2.dim_size[3](), fmap2.dim_size[4]()):
        for channel in range(fmap1.dim_size[2]()):
            acc += (
                _load_f32_5d(fmap1, batch, src_frame, channel, patch_row, patch_col)
                * _load_f32_5d(fmap2, batch, dst_frame, channel, tgt_row, tgt_col)
            )
    return acc


@register("corr_forward")
struct CorrForward[radius: Int]:
    @staticmethod
    def execute[target: StaticString](
        corr: OutputTensor[dtype=DType.float32, rank=6, ...],
        fmap1: InputTensor[dtype=DType.float32, rank=5, ...],
        fmap2: InputTensor[dtype=DType.float32, rank=5, ...],
        coords: InputTensor[dtype=DType.float32, rank=5, ...],
        ii: InputTensor[dtype=DType.int64, rank=1, ...],
        jj: InputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("corr_forward currently requires GPU tensors")
        var gpu_ctx = ctx.get_device_context()
        # One GPU thread owns one output correlation value:
        # corr[batch, edge, search_x, search_y, patch_row, patch_col].
        var corr_values = (
            corr.dim_size[0]()
            * corr.dim_size[1]()
            * corr.dim_size[2]()
            * corr.dim_size[3]()
            * corr.dim_size[4]()
            * corr.dim_size[5]()
        )
        comptime SEARCH_SIZE = 2 * Self.radius + 1
        comptime BLOCK_SIZE = 256

        @parameter
        def corr_forward_kernel(corr_values: Int):
            var tid = global_idx.x
            if tid < corr_values:
                var linear_index = tid
                var patch_col = linear_index % coords.dim_size[4]()
                linear_index = linear_index // coords.dim_size[4]()
                var patch_row = linear_index % coords.dim_size[3]()
                linear_index = linear_index // coords.dim_size[3]()
                var search_y = linear_index % SEARCH_SIZE
                linear_index = linear_index // SEARCH_SIZE
                var search_x = linear_index % SEARCH_SIZE
                linear_index = linear_index // SEARCH_SIZE
                var edge = linear_index % coords.dim_size[1]()
                linear_index = linear_index // coords.dim_size[1]()
                var batch = linear_index

                var src_frame = _load_i64_1d(ii, edge)
                var dst_frame = _load_i64_1d(jj, edge)
                var x = _load_f32_5d(coords, batch, edge, 0, patch_row, patch_col)
                var y = _load_f32_5d(coords, batch, edge, 1, patch_row, patch_col)
                var fx = floor(x)
                var fy = floor(y)
                var dx = x - fx
                var dy = y - fy
                var base_col = Int(fx) + search_x - Self.radius
                var base_row = Int(fy) + search_y - Self.radius

                # Bilinear interpolation over the target-frame correlation grid.
                var v00 = _descriptor_dot_at_target_pixel(fmap1, fmap2, batch, src_frame, dst_frame, patch_row, patch_col, base_row, base_col)
                var v01 = _descriptor_dot_at_target_pixel(fmap1, fmap2, batch, src_frame, dst_frame, patch_row, patch_col, base_row, base_col + 1)
                var v10 = _descriptor_dot_at_target_pixel(fmap1, fmap2, batch, src_frame, dst_frame, patch_row, patch_col, base_row + 1, base_col)
                var v11 = _descriptor_dot_at_target_pixel(fmap1, fmap2, batch, src_frame, dst_frame, patch_row, patch_col, base_row + 1, base_col + 1)
                var value = (1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v01 + (1.0 - dx) * dy * v10 + dx * dy * v11
                corr.store[1](IndexList[6](batch, edge, search_x, search_y, patch_row, patch_col), SIMD[DType.float32, 1](value))

        gpu_ctx.enqueue_function_experimental[corr_forward_kernel](
            corr_values,
            grid_dim=ceildiv(corr_values, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )


@register("corr_backward")
struct CorrBackward[radius: Int]:
    @staticmethod
    def execute[target: StaticString](
        fmap1_grad: OutputTensor[dtype=DType.float32, rank=5, ...],
        fmap2_grad: OutputTensor[dtype=DType.float32, rank=5, ...],
        fmap1: InputTensor[dtype=DType.float32, rank=5, ...],
        fmap2: InputTensor[dtype=DType.float32, rank=5, ...],
        coords: InputTensor[dtype=DType.float32, rank=5, ...],
        ii: InputTensor[dtype=DType.int64, rank=1, ...],
        jj: InputTensor[dtype=DType.int64, rank=1, ...],
        corr_grad: InputTensor[dtype=DType.float32, rank=6, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("corr_backward currently requires GPU tensors")
        var gpu_ctx = ctx.get_device_context()
        var fmap1_grad_values = (
            fmap1_grad.dim_size[0]()
            * fmap1_grad.dim_size[1]()
            * fmap1_grad.dim_size[2]()
            * fmap1_grad.dim_size[3]()
            * fmap1_grad.dim_size[4]()
        )
        var fmap2_grad_values = (
            fmap2_grad.dim_size[0]()
            * fmap2_grad.dim_size[1]()
            * fmap2_grad.dim_size[2]()
            * fmap2_grad.dim_size[3]()
            * fmap2_grad.dim_size[4]()
        )
        var channels = fmap1.dim_size[2]()
        var edges = coords.dim_size[1]()
        var patch_h = coords.dim_size[3]()
        var patch_w = coords.dim_size[4]()
        var fmap2_h = fmap2.dim_size[3]()
        var fmap2_w = fmap2.dim_size[4]()
        comptime SEARCH_SIZE = 2 * Self.radius + 1
        comptime BLOCK_SIZE = 256

        @parameter
        def corr_backward_fmap1_kernel(
            fmap1_grad_values: Int,
            channels: Int,
            edges: Int,
            patch_h: Int,
            patch_w: Int,
            fmap2_h: Int,
            fmap2_w: Int,
            ):
            var tid = global_idx.x
            if tid < fmap1_grad_values:
                var linear_index = tid
                var patch_col = linear_index % patch_w
                linear_index = linear_index // patch_w
                var patch_row = linear_index % patch_h
                linear_index = linear_index // patch_h
                var channel = linear_index % channels
                linear_index = linear_index // channels
                var src_frame = linear_index % fmap1.dim_size[1]()
                linear_index = linear_index // fmap1.dim_size[1]()
                var batch = linear_index

                var acc: Float32 = 0.0
                for edge in range(edges):
                    if _load_i64_1d(ii, edge) == src_frame:
                        var dst_frame = _load_i64_1d(jj, edge)
                        var x = _load_f32_5d(coords, batch, edge, 0, patch_row, patch_col)
                        var y = _load_f32_5d(coords, batch, edge, 1, patch_row, patch_col)
                        var fx = floor(x)
                        var fy = floor(y)
                        var dx = x - fx
                        var dy = y - fy
                        for search_x in range(SEARCH_SIZE):
                            for search_y in range(SEARCH_SIZE):
                                var base_col = Int(fx) + search_x - Self.radius
                                var base_row = Int(fy) + search_y - Self.radius
                                var grad = _load_f32_6d(corr_grad, batch, edge, search_x, search_y, patch_row, patch_col)
                                if _within_bounds(base_row, base_col, fmap2_h, fmap2_w):
                                    acc += grad * (1.0 - dx) * (1.0 - dy) * _load_f32_5d(fmap2, batch, dst_frame, channel, base_row, base_col)
                                if _within_bounds(base_row, base_col + 1, fmap2_h, fmap2_w):
                                    acc += grad * dx * (1.0 - dy) * _load_f32_5d(fmap2, batch, dst_frame, channel, base_row, base_col + 1)
                                if _within_bounds(base_row + 1, base_col, fmap2_h, fmap2_w):
                                    acc += grad * (1.0 - dx) * dy * _load_f32_5d(fmap2, batch, dst_frame, channel, base_row + 1, base_col)
                                if _within_bounds(base_row + 1, base_col + 1, fmap2_h, fmap2_w):
                                    acc += grad * dx * dy * _load_f32_5d(fmap2, batch, dst_frame, channel, base_row + 1, base_col + 1)
                fmap1_grad.store[1](IndexList[5](batch, src_frame, channel, patch_row, patch_col), SIMD[DType.float32, 1](acc))

        @parameter
        def corr_backward_fmap2_kernel(fmap2_grad_values: Int, channels: Int, edges: Int, patch_h: Int, patch_w: Int):
            var tid = global_idx.x
            if tid < fmap2_grad_values:
                var linear_index = tid
                var dst_col = linear_index % fmap2.dim_size[4]()
                linear_index = linear_index // fmap2.dim_size[4]()
                var dst_row = linear_index % fmap2.dim_size[3]()
                linear_index = linear_index // fmap2.dim_size[3]()
                var channel = linear_index % channels
                linear_index = linear_index // channels
                var dst_frame = linear_index % fmap2.dim_size[1]()
                linear_index = linear_index // fmap2.dim_size[1]()
                var batch = linear_index

                var acc: Float32 = 0.0
                for edge in range(edges):
                    if _load_i64_1d(jj, edge) == dst_frame:
                        var src_frame = _load_i64_1d(ii, edge)
                        for patch_row in range(patch_h):
                            for patch_col in range(patch_w):
                                var x = _load_f32_5d(coords, batch, edge, 0, patch_row, patch_col)
                                var y = _load_f32_5d(coords, batch, edge, 1, patch_row, patch_col)
                                var fx = floor(x)
                                var fy = floor(y)
                                var dx = x - fx
                                var dy = y - fy
                                for search_x in range(SEARCH_SIZE):
                                    for search_y in range(SEARCH_SIZE):
                                        var base_col = Int(fx) + search_x - Self.radius
                                        var base_row = Int(fy) + search_y - Self.radius
                                        var grad = _load_f32_6d(corr_grad, batch, edge, search_x, search_y, patch_row, patch_col)
                                        var f1 = _load_f32_5d(fmap1, batch, src_frame, channel, patch_row, patch_col)
                                        if dst_row == base_row and dst_col == base_col:
                                            acc += grad * (1.0 - dx) * (1.0 - dy) * f1
                                        if dst_row == base_row and dst_col == base_col + 1:
                                            acc += grad * dx * (1.0 - dy) * f1
                                        if dst_row == base_row + 1 and dst_col == base_col:
                                            acc += grad * (1.0 - dx) * dy * f1
                                        if dst_row == base_row + 1 and dst_col == base_col + 1:
                                            acc += grad * dx * dy * f1
                fmap2_grad.store[1](IndexList[5](batch, dst_frame, channel, dst_row, dst_col), SIMD[DType.float32, 1](acc))

        gpu_ctx.enqueue_function_experimental[corr_backward_fmap1_kernel](
            fmap1_grad_values,
            channels,
            edges,
            patch_h,
            patch_w,
            fmap2_h,
            fmap2_w,
            grid_dim=ceildiv(fmap1_grad_values, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
        gpu_ctx.enqueue_function_experimental[corr_backward_fmap2_kernel](
            fmap2_grad_values,
            channels,
            edges,
            patch_h,
            patch_w,
            grid_dim=ceildiv(fmap2_grad_values, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
