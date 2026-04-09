from compiler import register
from layout import LayoutTensor
from tensor import InputTensor, OutputTensor, foreach
from std.gpu import block_dim, block_idx, thread_idx
from std.math import abs, cos, exp, sin, sqrt
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList

comptime POSE_DOF = 7
comptime POSE_STRIDE = 8
comptime THREADS = 256


@always_inline
def scalar_at[
    dtype: DType,
    rank: Int,
    simd_width: Int,
](
    tensor: InputTensor[dtype=dtype, rank=rank, ...],
    idx: IndexList[rank],
) -> Scalar[dtype]:
    return tensor.load[simd_width](idx)[0]


@always_inline
def pack_scalar[simd_width: Int](value: Float32) -> SIMD[DType.float32, simd_width]:
    return SIMD[DType.float32, simd_width](value)


@always_inline
def quat_comp_components(
    aix: Float32,
    aiy: Float32,
    aiz: Float32,
    aiw: Float32,
    bjx: Float32,
    bjy: Float32,
    bjz: Float32,
    bjw: Float32,
) -> InlineArray[Float32, 4]:
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = aiw * bjx + aix * bjw + aiy * bjz - aiz * bjy
    out[1] = aiw * bjy - aix * bjz + aiy * bjw + aiz * bjx
    out[2] = aiw * bjz + aix * bjy - aiy * bjx + aiz * bjw
    out[3] = aiw * bjw - aix * bjx - aiy * bjy - aiz * bjz
    return out^


@always_inline
def act_so3_components(
    qx: Float32,
    qy: Float32,
    qz: Float32,
    qw: Float32,
    x0: Float32,
    x1: Float32,
    x2: Float32,
) -> InlineArray[Float32, 3]:
    var uv0 = 2.0 * (qy * x2 - qz * x1)
    var uv1 = 2.0 * (qz * x0 - qx * x2)
    var uv2 = 2.0 * (qx * x1 - qy * x0)
    var out = InlineArray[Float32, 3](fill=0.0)
    out[0] = x0 + qw * uv0 + (qy * uv2 - qz * uv1)
    out[1] = x1 + qw * uv1 + (qz * uv0 - qx * uv2)
    out[2] = x2 + qw * uv2 + (qx * uv1 - qy * uv0)
    return out^


@always_inline
def exp_so3_components(
    phi0: Float32,
    phi1: Float32,
    phi2: Float32,
) -> InlineArray[Float32, 4]:
    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta_p4 = theta_sq * theta_sq
    var imag: Float32
    var real: Float32
    if theta_sq < 1e-6:
        imag = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
        real = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_p4
    else:
        var theta = sqrt(theta_sq)
        imag = sin(0.5 * theta) / theta
        real = cos(0.5 * theta)
    var out = InlineArray[Float32, 4](fill=0.0)
    out[0] = imag * phi0
    out[1] = imag * phi1
    out[2] = imag * phi2
    out[3] = real
    return out^


@always_inline
def exp_sim3_components(
    xi0: Float32,
    xi1: Float32,
    xi2: Float32,
    xi3: Float32,
    xi4: Float32,
    xi5: Float32,
    xi6: Float32,
) -> InlineArray[Float32, POSE_STRIDE]:
    var tau0 = xi0
    var tau1 = xi1
    var tau2 = xi2
    var phi0 = xi3
    var phi1 = xi4
    var phi2 = xi5
    var sigma = xi6
    var scale = exp(sigma)
    var q = exp_so3_components(phi0, phi1, phi2)

    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta = sqrt(theta_sq)
    var A: Float32
    var B: Float32
    var C: Float32
    if abs(sigma) < Float32(1e-6):
        C = 1.0
        if abs(theta) < Float32(1e-6):
            A = 0.5
            B = 1.0 / 6.0
        else:
            A = (1.0 - cos(theta)) / theta_sq
            B = (theta - sin(theta)) / (theta_sq * theta)
    else:
        C = (scale - 1.0) / sigma
        if abs(theta) < Float32(1e-6):
            var sigma_sq = sigma * sigma
            A = ((sigma - 1.0) * scale + 1.0) / sigma_sq
            B = (scale * 0.5 * sigma_sq + scale - 1.0 - sigma * scale) / (sigma_sq * sigma)
        else:
            var a = scale * sin(theta)
            var b = scale * cos(theta)
            var c = theta_sq + sigma * sigma
            A = (a * sigma + (1.0 - b) * theta) / (theta * c)
            B = (C - (((b - 1.0) * sigma + a * theta) / c)) / theta_sq

    var out = InlineArray[Float32, POSE_STRIDE](fill=0.0)
    out[0] = C * tau0
    out[1] = C * tau1
    out[2] = C * tau2

    var cx0 = phi1 * tau2 - phi2 * tau1
    var cx1 = phi2 * tau0 - phi0 * tau2
    var cx2 = phi0 * tau1 - phi1 * tau0
    out[0] += A * cx0
    out[1] += A * cx1
    out[2] += A * cx2

    var c2x0 = phi1 * cx2 - phi2 * cx1
    var c2x1 = phi2 * cx0 - phi0 * cx2
    var c2x2 = phi0 * cx1 - phi1 * cx0
    out[0] += B * c2x0
    out[1] += B * c2x1
    out[2] += B * c2x2
    out[3] = q[0]
    out[4] = q[1]
    out[5] = q[2]
    out[6] = q[3]
    out[7] = scale
    return out^


@register("pose_retr")
struct PoseRetr:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        poses_out: OutputTensor[dtype=DType.float32, rank=2, ...],
        poses_in: InputTensor[dtype=DType.float32, rank=2, ...],
        dx_in: InputTensor[dtype=DType.float32, rank=2, ...],
        num_fix_in: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        def update_pose[
            simd_width: Int,
        ](idx: IndexList[poses_out.rank]) -> SIMD[DType.float32, simd_width]:
            var row = idx[0]
            var col = idx[1]
            var in_idx = IndexList[2](row, col)
            var num_fix = Int(
                scalar_at[DType.int32, 1, simd_width](num_fix_in, IndexList[1](0))
            )
            if row < num_fix:
                return poses_in.load[simd_width](in_idx)

            var dx_row = row - num_fix
            var xi = exp_sim3_components(
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 0)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 1)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 2)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 3)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 4)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 5)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 6)),
            )

            var px = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 0))
            var py = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 1))
            var pz = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 2))
            var qx = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 3))
            var qy = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 4))
            var qz = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 5))
            var qw = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 6))
            var s = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 7))

            var rot_t = act_so3_components(xi[3], xi[4], xi[5], xi[6], px, py, pz)
            var q1 = quat_comp_components(xi[3], xi[4], xi[5], xi[6], qx, qy, qz, qw)

            if col == 0:
                return pack_scalar[simd_width](rot_t[0] * xi[7] + xi[0])
            if col == 1:
                return pack_scalar[simd_width](rot_t[1] * xi[7] + xi[1])
            if col == 2:
                return pack_scalar[simd_width](rot_t[2] * xi[7] + xi[2])
            if col == 3:
                return pack_scalar[simd_width](q1[0])
            if col == 4:
                return pack_scalar[simd_width](q1[1])
            if col == 5:
                return pack_scalar[simd_width](q1[2])
            if col == 6:
                return pack_scalar[simd_width](q1[3])
            return pack_scalar[simd_width](xi[7] * s)

        foreach[update_pose, target=target, simd_width=1](poses_out, ctx)


@register("pose_retr_launch")
struct PoseRetrLaunch:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        poses_out: OutputTensor[dtype=DType.float32, rank=2, ...],
        poses_in: InputTensor[dtype=DType.float32, rank=2, ...],
        dx_in: InputTensor[dtype=DType.float32, rank=2, ...],
        num_fix_in: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        if target == "gpu":
            var device_ctx = ctx.get_device_context()
            var out_tensor = poses_out.to_layout_tensor()
            var in_tensor = poses_in.to_layout_tensor()
            var dx_tensor = dx_in.to_layout_tensor()
            var num_fix_tensor = num_fix_in.to_layout_tensor()
            var rows = Int(poses_out.shape()[0])
            var blocks = (rows + THREADS - 1) // THREADS

            @parameter
            def pose_retr_kernel_local(
                poses_out_kernel: type_of(out_tensor),
                poses_in_kernel: type_of(in_tensor),
                dx_kernel: type_of(dx_tensor),
                num_fix_kernel: type_of(num_fix_tensor),
            ):
                var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
                var rows_local = Int(poses_out_kernel.dim[0]())
                if row >= rows_local:
                    return

                var num_fix = Int(num_fix_kernel[0])
                if row < num_fix:
                    for col in range(POSE_STRIDE):
                        poses_out_kernel[row, col] = poses_in_kernel[row, col]
                    return

                var dx_row = row - num_fix
                var xi = exp_sim3_components(
                    rebind[Float32](dx_kernel[dx_row, 0]),
                    rebind[Float32](dx_kernel[dx_row, 1]),
                    rebind[Float32](dx_kernel[dx_row, 2]),
                    rebind[Float32](dx_kernel[dx_row, 3]),
                    rebind[Float32](dx_kernel[dx_row, 4]),
                    rebind[Float32](dx_kernel[dx_row, 5]),
                    rebind[Float32](dx_kernel[dx_row, 6]),
                )

                var px = rebind[Float32](poses_in_kernel[row, 0])
                var py = rebind[Float32](poses_in_kernel[row, 1])
                var pz = rebind[Float32](poses_in_kernel[row, 2])
                var qx = rebind[Float32](poses_in_kernel[row, 3])
                var qy = rebind[Float32](poses_in_kernel[row, 4])
                var qz = rebind[Float32](poses_in_kernel[row, 5])
                var qw = rebind[Float32](poses_in_kernel[row, 6])
                var s = rebind[Float32](poses_in_kernel[row, 7])

                var rot_t = act_so3_components(xi[3], xi[4], xi[5], xi[6], px, py, pz)
                var q1 = quat_comp_components(xi[3], xi[4], xi[5], xi[6], qx, qy, qz, qw)

                poses_out_kernel[row, 0] = rot_t[0] * xi[7] + xi[0]
                poses_out_kernel[row, 1] = rot_t[1] * xi[7] + xi[1]
                poses_out_kernel[row, 2] = rot_t[2] * xi[7] + xi[2]
                poses_out_kernel[row, 3] = q1[0]
                poses_out_kernel[row, 4] = q1[1]
                poses_out_kernel[row, 5] = q1[2]
                poses_out_kernel[row, 6] = q1[3]
                poses_out_kernel[row, 7] = xi[7] * s

            device_ctx.enqueue_function[
                pose_retr_kernel_local,
                pose_retr_kernel_local,
            ](
                out_tensor,
                in_tensor,
                dx_tensor,
                num_fix_tensor,
                grid_dim=blocks,
                block_dim=THREADS,
            )
            device_ctx.synchronize()
            return

        @parameter
        @always_inline
        def update_pose_cpu[
            simd_width: Int,
        ](idx: IndexList[poses_out.rank]) -> SIMD[DType.float32, simd_width]:
            var row = idx[0]
            var col = idx[1]
            var in_idx = IndexList[2](row, col)
            var num_fix = Int(
                scalar_at[DType.int32, 1, simd_width](num_fix_in, IndexList[1](0))
            )
            if row < num_fix:
                return poses_in.load[simd_width](in_idx)

            var dx_row = row - num_fix
            var xi = exp_sim3_components(
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 0)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 1)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 2)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 3)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 4)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 5)),
                scalar_at[DType.float32, 2, simd_width](dx_in, IndexList[2](dx_row, 6)),
            )

            var px = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 0))
            var py = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 1))
            var pz = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 2))
            var qx = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 3))
            var qy = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 4))
            var qz = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 5))
            var qw = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 6))
            var s = scalar_at[DType.float32, 2, simd_width](poses_in, IndexList[2](row, 7))

            var rot_t = act_so3_components(xi[3], xi[4], xi[5], xi[6], px, py, pz)
            var q1 = quat_comp_components(xi[3], xi[4], xi[5], xi[6], qx, qy, qz, qw)

            if col == 0:
                return pack_scalar[simd_width](rot_t[0] * xi[7] + xi[0])
            if col == 1:
                return pack_scalar[simd_width](rot_t[1] * xi[7] + xi[1])
            if col == 2:
                return pack_scalar[simd_width](rot_t[2] * xi[7] + xi[2])
            if col == 3:
                return pack_scalar[simd_width](q1[0])
            if col == 4:
                return pack_scalar[simd_width](q1[1])
            if col == 5:
                return pack_scalar[simd_width](q1[2])
            if col == 6:
                return pack_scalar[simd_width](q1[3])
            return pack_scalar[simd_width](xi[7] * s)

        foreach[update_pose_cpu, target=target, simd_width=1](poses_out, ctx)


@register("pose_copy_launch")
struct PoseCopyLaunch:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        poses_out: OutputTensor[dtype=DType.float32, rank=2, ...],
        poses_in: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var total = Int(poses_out.shape()[0]) * Int(poses_out.shape()[1])
        var blocks = (total + THREADS - 1) // THREADS
        if target == "gpu":
            var device_ctx = ctx.get_device_context()
            var out_tensor = poses_out.to_layout_tensor()
            var in_tensor = poses_in.to_layout_tensor()
            @parameter
            def pose_copy_kernel_local(
                poses_out_kernel: type_of(out_tensor),
                poses_in_kernel: type_of(in_tensor),
            ):
                var linear_idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
                var rows = Int(poses_out_kernel.dim[0]())
                var cols = Int(poses_out_kernel.dim[1]())
                var total = rows * cols
                if linear_idx >= total:
                    return

                var row = linear_idx // cols
                var col = linear_idx - row * cols
                poses_out_kernel[row, col] = poses_in_kernel[row, col]

            device_ctx.enqueue_function[
                pose_copy_kernel_local,
                pose_copy_kernel_local,
            ](
                out_tensor,
                in_tensor,
                grid_dim=blocks,
                block_dim=THREADS,
            )
            device_ctx.synchronize()
            return

        @parameter
        @always_inline
        def copy_pose_cpu[
            simd_width: Int,
        ](idx: IndexList[2]) -> SIMD[DType.float32, simd_width]:
            return poses_in.load[simd_width](idx)

        foreach[copy_pose_cpu, target=target, simd_width=1](poses_out, ctx)
