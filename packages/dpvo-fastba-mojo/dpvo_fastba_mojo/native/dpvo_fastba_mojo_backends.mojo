from std.atomic import Atomic
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv, cos, max, sin, sqrt
from std.memory import alloc
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.utils.index import Index
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE


# Mojo implementation of the DPVO bundle-adjustment hot path.
#
# DPVO alternates sparse patch-flow updates with differentiable bundle
# adjustment. These kernels cover the local, dense-in-window BA path used by the
# demo gate: reproject patches, accumulate a small Schur system, retract camera
# poses on SE(3), and retract inverse-depth patch planes.
#
# The Python tensors carry a leading batch dimension of size 1. The original CUDA
# extension and this port both treat the underlying contiguous storage as:
#   poses      -> [frames, 7]
#   patches    -> [patches, 3, patch_size, patch_size]
#   intrinsics -> [4]
#   target     -> [edges, 2]
# LayoutTensor documents that view explicitly and removes most manual stride
# arithmetic inside the kernels. The layout names include rank because several
# tensors share scalar element types but have different logical indexing rules.
comptime POSES2_LT = Layout.row_major(UNKNOWN_VALUE, 7)  # [frames, tx ty tz qx qy qz qw]
comptime PATCHES4_LT = Layout.row_major(UNKNOWN_VALUE, 3, UNKNOWN_VALUE, UNKNOWN_VALUE)  # [patches, x/y/idepth, patch_h, patch_w]
comptime INTR4_LT = Layout.row_major(4)  # [fx, fy, cx, cy]
comptime COORDS4_LT = Layout.row_major(UNKNOWN_VALUE, 2, UNKNOWN_VALUE, UNKNOWN_VALUE)  # [edges, x/y, patch_h, patch_w]
comptime EDGE1_LT = Layout.row_major(UNKNOWN_VALUE)  # edge-index vectors: ii, jj, kk, ku
comptime TARGET2_LT = Layout.row_major(UNKNOWN_VALUE, 2)  # [edges, residual_x/residual_y]
comptime UPDATE2_LT = Layout.row_major(UNKNOWN_VALUE, 6)  # pose increments in se(3)
comptime UPDATE1_LT = Layout.row_major(UNKNOWN_VALUE)  # one inverse-depth update per unique patch


# ── Python/PyTorch Interop ────────────────────────────────────────────────────

def _install_cached_context(module: PythonObject) raises:
    # Python imports this shared library many times during tests. Building and
    # caching one DeviceContext per module keeps launch overhead low and gives
    # every wrapper the same GPU stream owner.
    var ctx_storage = alloc[DeviceContext](1)
    var cached_ctx = DeviceContext()
    ctx_storage.init_pointee_move(cached_ctx^)
    Python.add_object(module, "_ctx_addr", PythonObject(Int(ctx_storage)))


def _get_cached_context_ptr() raises -> UnsafePointer[DeviceContext, MutAnyOrigin]:
    # The cached DeviceContext is stored as a Python integer on the extension
    # module. Safe Pointer cannot be reconstructed from that external address, so
    # this interop boundary remains intentionally unsafe.
    var module = Python.import_module("dpvo_fastba_mojo_backends")
    var ctx_addr = Int(py=module._ctx_addr)
    return UnsafePointer[DeviceContext, MutAnyOrigin](unsafe_from_address=ctx_addr)


@always_inline
def _torch_float32_ptr(tensor: PythonObject) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    # PyTorch exposes tensor storage as `data_ptr()`. Callers immediately wrap
    # these addresses in LayoutTensor where structured indexing is possible.
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=tensor.data_ptr()))


@always_inline
def _torch_int64_ptr(tensor: PythonObject) raises -> UnsafePointer[Int64, MutAnyOrigin]:
    # Same PyTorch-owned storage boundary as _torch_float32_ptr, specialized for
    # edge index vectors.
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=tensor.data_ptr()))


# ── SE(3) Math Helpers ────────────────────────────────────────────────────────

@always_inline
def _act_so3(qx: Float32, qy: Float32, qz: Float32, qw: Float32, x0: Float32, x1: Float32, x2: Float32) -> Tuple[Float32, Float32, Float32]:
    # Rotate a 3-vector by quaternion q without materializing a 3x3 matrix.
    # This is the CUDA extension's q * x * q^-1 formula in fused scalar form.
    var uv0 = 2.0 * (qy * x2 - qz * x1)
    var uv1 = 2.0 * (qz * x0 - qx * x2)
    var uv2 = 2.0 * (qx * x1 - qy * x0)
    var y0 = x0 + qw * uv0 + (qy * uv2 - qz * uv1)
    var y1 = x1 + qw * uv1 + (qz * uv0 - qx * uv2)
    var y2 = x2 + qw * uv2 + (qx * uv1 - qy * uv0)
    return y0, y1, y2


@always_inline
def _rel_se3(
    poses: LayoutTensor[DType.float32, POSES2_LT, MutAnyOrigin],
    ix: Int,
    jx: Int,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    # Relative camera transform used by reprojection and Jacobian assembly.
    # Pose rows are [tx, ty, tz, qx, qy, qz, qw]. The sign/order here mirrors
    # DPVO's CUDA helper; changing it flips the patch projection direction.
    var ti0 = poses[ix, 0][0]
    var ti1 = poses[ix, 1][0]
    var ti2 = poses[ix, 2][0]
    var qi0 = poses[ix, 3][0]
    var qi1 = poses[ix, 4][0]
    var qi2 = poses[ix, 5][0]
    var qi3 = poses[ix, 6][0]

    var tj0 = poses[jx, 0][0]
    var tj1 = poses[jx, 1][0]
    var tj2 = poses[jx, 2][0]
    var qj0 = poses[jx, 3][0]
    var qj1 = poses[jx, 4][0]
    var qj2 = poses[jx, 5][0]
    var qj3 = poses[jx, 6][0]

    var qij0 = -qj3 * qi0 + qj0 * qi3 - qj1 * qi2 + qj2 * qi1
    var qij1 = -qj3 * qi1 + qj0 * qi2 + qj1 * qi3 - qj2 * qi0
    var qij2 = -qj3 * qi2 - qj0 * qi1 + qj1 * qi0 + qj2 * qi3
    var qij3 = qj3 * qi3 + qj0 * qi0 + qj1 * qi1 + qj2 * qi2

    var ri0, ri1, ri2 = _act_so3(qij0, qij1, qij2, qij3, ti0, ti1, ti2)

    return tj0 - ri0, tj1 - ri1, tj2 - ri2, qij0, qij1, qij2, qij3


@always_inline
def _rel_se3_raw(
    poses: UnsafePointer[Float32, MutAnyOrigin],
    ix: Int,
    jx: Int,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    # `reproject_kernel` is the only kernel that keeps pointer arithmetic for
    # read-only tensor storage. The LayoutTensor version is correct, but exceeds
    # the 1.05 speed gate for the 8192-edge p3 benchmark. This raw helper keeps
    # the speed-critical projection path equivalent to the CUDA extension.
    var ti0 = (poses + ix * 7)[]
    var ti1 = (poses + ix * 7 + 1)[]
    var ti2 = (poses + ix * 7 + 2)[]
    var qi0 = (poses + ix * 7 + 3)[]
    var qi1 = (poses + ix * 7 + 4)[]
    var qi2 = (poses + ix * 7 + 5)[]
    var qi3 = (poses + ix * 7 + 6)[]

    var tj0 = (poses + jx * 7)[]
    var tj1 = (poses + jx * 7 + 1)[]
    var tj2 = (poses + jx * 7 + 2)[]
    var qj0 = (poses + jx * 7 + 3)[]
    var qj1 = (poses + jx * 7 + 4)[]
    var qj2 = (poses + jx * 7 + 5)[]
    var qj3 = (poses + jx * 7 + 6)[]

    var qij0 = -qj3 * qi0 + qj0 * qi3 - qj1 * qi2 + qj2 * qi1
    var qij1 = -qj3 * qi1 + qj0 * qi2 + qj1 * qi3 - qj2 * qi0
    var qij2 = -qj3 * qi2 - qj0 * qi1 + qj1 * qi0 + qj2 * qi3
    var qij3 = qj3 * qi3 + qj0 * qi0 + qj1 * qi1 + qj2 * qi2

    var ri0, ri1, ri2 = _act_so3(qij0, qij1, qij2, qij3, ti0, ti1, ti2)

    return tj0 - ri0, tj1 - ri1, tj2 - ri2, qij0, qij1, qij2, qij3


@always_inline
def _act_se3(
    tx: Float32,
    ty: Float32,
    tz: Float32,
    qx: Float32,
    qy: Float32,
    qz: Float32,
    qw: Float32,
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
) -> Tuple[Float32, Float32, Float32, Float32]:
    # Apply SE(3) to an inverse-depth homogeneous point:
    #   X_j = R_ji * [x, y, 1] + inverse_depth * t_ji
    # The final component is kept as W so the BA Jacobian can differentiate
    # with respect to inverse depth.
    var y0, y1, y2 = _act_so3(qx, qy, qz, qw, x0, x1, x2)
    return y0 + x3 * tx, y1 + x3 * ty, y2 + x3 * tz, x3


@always_inline
def _adj_se3(
    tx: Float32,
    ty: Float32,
    tz: Float32,
    qx: Float32,
    qy: Float32,
    qz: Float32,
    qw: Float32,
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
    x4: Float32,
    x5: Float32,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32]:
    # Apply the SE(3) adjoint to a 6-vector. In the BA kernel this converts the
    # target-frame projection Jacobian into the source-frame pose Jacobian.
    var y0, y1, y2 = _act_so3(-qx, -qy, -qz, qw, x0, x1, x2)
    var y3, y4, y5 = _act_so3(-qx, -qy, -qz, qw, x3, x4, x5)
    var u0 = tz * x1 - ty * x2
    var u1 = tx * x2 - tz * x0
    var u2 = ty * x0 - tx * x1
    var v0, v1, v2 = _act_so3(-qx, -qy, -qz, qw, u0, u1, u2)
    return y0, y1, y2, y3 + v0, y4 + v1, y5 + v2


@always_inline
def _cross(ax: Float32, ay: Float32, az: Float32, bx: Float32, by: Float32, bz: Float32) -> Tuple[Float32, Float32, Float32]:
    # Small inline cross product used by the SE(3) exponential map.
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@always_inline
def _exp_so3(phi0: Float32, phi1: Float32, phi2: Float32) -> Tuple[Float32, Float32, Float32, Float32]:
    # Exponential map from so(3) vector to quaternion. The Taylor branch avoids
    # division by a tiny angle and matches the CUDA implementation's numerics.
    var theta_sq = phi0 * phi0 + phi1 * phi1 + phi2 * phi2
    var theta_p4 = theta_sq * theta_sq
    var theta = sqrt(theta_sq)
    var imag: Float32
    var real: Float32
    if theta_sq < 1e-8:
        imag = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_p4
        real = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_p4
    else:
        imag = sin(0.5 * theta) / theta
        real = cos(0.5 * theta)
    return imag * phi0, imag * phi1, imag * phi2, real


@always_inline
def _exp_se3(
    xi0: Float32,
    xi1: Float32,
    xi2: Float32,
    xi3: Float32,
    xi4: Float32,
    xi5: Float32,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    # Exponential map from se(3) twist to SE(3). Translation uses the standard
    # left-Jacobian correction so pose updates are applied on the manifold, not
    # as independent Euclidean translation/rotation deltas.
    var qx, qy, qz, qw = _exp_so3(xi3, xi4, xi5)
    var tx = xi0
    var ty = xi1
    var tz = xi2
    var theta_sq = xi3 * xi3 + xi4 * xi4 + xi5 * xi5
    var theta = sqrt(theta_sq)
    if theta > 1e-4:
        var a = (1.0 - cos(theta)) / theta_sq
        var c0, c1, c2 = _cross(xi3, xi4, xi5, xi0, xi1, xi2)
        tx += a * c0
        ty += a * c1
        tz += a * c2
        var b = (theta - sin(theta)) / (theta * theta_sq)
        var d0, d1, d2 = _cross(xi3, xi4, xi5, c0, c1, c2)
        tx += b * d0
        ty += b * d1
        tz += b * d2
    return tx, ty, tz, qx, qy, qz, qw


@always_inline
def _retr_se3(
    xi0: Float32,
    xi1: Float32,
    xi2: Float32,
    xi3: Float32,
    xi4: Float32,
    xi5: Float32,
    t0: Float32,
    t1: Float32,
    t2: Float32,
    q0: Float32,
    q1: Float32,
    q2: Float32,
    q3: Float32,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    # Retract a 6D tangent update onto an existing pose. DPVO applies the update
    # by left-multiplying exp(xi) with the current pose and writing the result
    # back in-place.
    var dt0, dt1, dt2, dq0, dq1, dq2, dq3 = _exp_se3(xi0, xi1, xi2, xi3, xi4, xi5)
    var out_q0 = dq3 * q0 + dq0 * q3 + dq1 * q2 - dq2 * q1
    var out_q1 = dq3 * q1 + dq1 * q3 + dq2 * q0 - dq0 * q2
    var out_q2 = dq3 * q2 + dq2 * q3 + dq0 * q1 - dq1 * q0
    var out_q3 = dq3 * q3 - dq0 * q0 - dq1 * q1 - dq2 * q2
    var rt0, rt1, rt2 = _act_so3(dq0, dq1, dq2, dq3, t0, t1, t2)
    return rt0 + dt0, rt1 + dt1, rt2 + dt2, out_q0, out_q1, out_q2, out_q3


# ── Reprojection ──────────────────────────────────────────────────────────────

def reproject_kernel(
    coords: UnsafePointer[Float32, MutAnyOrigin],
    poses: UnsafePointer[Float32, MutAnyOrigin],
    patches: UnsafePointer[Float32, MutAnyOrigin],
    intrinsics: UnsafePointer[Float32, MutAnyOrigin],
    ii: UnsafePointer[Int64, MutAnyOrigin],
    jj: UnsafePointer[Int64, MutAnyOrigin],
    kk: UnsafePointer[Int64, MutAnyOrigin],
    total: Int,
    patch_size: Int,
):
    # Generic raw reprojection path for arbitrary square patch sizes. One thread
    # projects one patch pixel for one edge and writes coords[edge, x/y, row,col].
    # This remains raw because the p3-specialized raw path is the speed-gated
    # demo path, and keeping both variants structurally similar makes them easy
    # to compare against the CUDA extension.
    var tid = global_idx.x
    if tid < total:
        var linear_index = tid
        var col = linear_index % patch_size
        linear_index = linear_index // patch_size
        var row = linear_index % patch_size
        var edge_id = linear_index // patch_size
        var ix = Int((ii + edge_id)[])
        var jx = Int((jj + edge_id)[])
        var kx = Int((kk + edge_id)[])

        var tx, ty, tz, qx, qy, qz, qw = _rel_se3_raw(poses, ix, jx)
        var fx = intrinsics[]
        var fy = (intrinsics + 1)[]
        var cx = (intrinsics + 2)[]
        var cy = (intrinsics + 3)[]

        # A patch is a tiny fronto-parallel plane: x grid, y grid, inverse depth.
        var patch_base = ((kx * 3) * patch_size + row) * patch_size + col
        var xi0 = ((patches + patch_base)[] - cx) / fx
        var xi1 = ((patches + patch_size * patch_size + patch_base)[] - cy) / fy
        var xi2: Float32 = 1.0
        var xi3 = (patches + 2 * patch_size * patch_size + patch_base)[]
        var xj0, xj1, xj2, _ = _act_se3(tx, ty, tz, qx, qy, qz, qw, xi0, xi1, xi2, xi3)
        var out_base = ((edge_id * 2) * patch_size + row) * patch_size + col
        (coords + out_base)[] = fx * (xj0 / xj2) + cx
        (coords + patch_size * patch_size + out_base)[] = fy * (xj1 / xj2) + cy


def reproject_kernel_p3_pixels_raw(
    coords: UnsafePointer[Float32, MutAnyOrigin],
    poses: UnsafePointer[Float32, MutAnyOrigin],
    patches: UnsafePointer[Float32, MutAnyOrigin],
    intrinsics: UnsafePointer[Float32, MutAnyOrigin],
    ii: UnsafePointer[Int64, MutAnyOrigin],
    jj: UnsafePointer[Int64, MutAnyOrigin],
    kk: UnsafePointer[Int64, MutAnyOrigin],
    total: Int,
):
    # Specialized DPVO hot path for 3x3 patches. Removing runtime patch_size
    # divisions and LayoutTensor indexing here was required to keep the
    # reproject benchmark within the 1.05 CUDA speed gate.
    var tid = global_idx.x
    if tid < total:
        var patch_pixel = tid % 9
        var edge_id = tid // 9
        var row = patch_pixel // 3
        var col = patch_pixel - row * 3

        var ix = Int((ii + edge_id)[])
        var jx = Int((jj + edge_id)[])
        var kx = Int((kk + edge_id)[])

        var tx, ty, tz, qx, qy, qz, qw = _rel_se3_raw(poses, ix, jx)
        var fx = intrinsics[]
        var fy = (intrinsics + 1)[]
        var cx = (intrinsics + 2)[]
        var cy = (intrinsics + 3)[]

        # Specialized [patch, 3, 3, 3] indexing for the DPVO demo/benchmark
        # path. This keeps raw pointers only where LayoutTensor missed the speed
        # gate, while removing the dynamic patch_size divisions from the generic
        # kernel above.
        var patch_base = kx * 27 + patch_pixel
        var xi0 = ((patches + patch_base)[] - cx) / fx
        var xi1 = ((patches + 9 + patch_base)[] - cy) / fy
        var xi3 = (patches + 18 + patch_base)[]
        var xj0, xj1, xj2, _ = _act_se3(tx, ty, tz, qx, qy, qz, qw, xi0, xi1, 1.0, xi3)
        var out_base = edge_id * 18 + patch_pixel
        (coords + out_base)[] = fx * (xj0 / xj2) + cx
        (coords + 9 + out_base)[] = fy * (xj1 / xj2) + cy


def reproject(
    poses_obj: PythonObject,
    patches_obj: PythonObject,
    intrinsics_obj: PythonObject,
    ii_obj: PythonObject,
    jj_obj: PythonObject,
    kk_obj: PythonObject,
) raises -> PythonObject:
    # Python-facing wrapper for reproject(). It allocates the PyTorch output and
    # launches the appropriate GPU kernel, then restores DPVO's leading batch
    # dimension in the returned view.
    var torch = Python.import_module("torch")
    var poses = poses_obj
    var rank = Int(py=patches_obj.dim())
    var patch_size = Int(py=patches_obj.shape[rank - 1])
    var patches = patches_obj
    var intrinsics = intrinsics_obj
    var ii = ii_obj
    var jj = jj_obj
    var kk = kk_obj
    var edges = Int(py=ii.shape[0])
    var total = edges * patch_size * patch_size

    var coords: PythonObject = torch.empty(
        Python.list(edges, 2, patch_size, patch_size),
        device=poses.device,
        dtype=torch.float32,
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    if patch_size == 3:
        ctx_ptr[].enqueue_function[reproject_kernel_p3_pixels_raw, reproject_kernel_p3_pixels_raw](
            _torch_float32_ptr(coords),
            _torch_float32_ptr(poses),
            _torch_float32_ptr(patches),
            _torch_float32_ptr(intrinsics),
            _torch_int64_ptr(ii),
            _torch_int64_ptr(jj),
            _torch_int64_ptr(kk),
            total,
            grid_dim=ceildiv(total, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
    else:
        ctx_ptr[].enqueue_function[reproject_kernel, reproject_kernel](
            _torch_float32_ptr(coords),
            _torch_float32_ptr(poses),
            _torch_float32_ptr(patches),
            _torch_float32_ptr(intrinsics),
            _torch_int64_ptr(ii),
            _torch_int64_ptr(jj),
            _torch_int64_ptr(kk),
            total,
            patch_size,
            grid_dim=ceildiv(total, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )
    return coords.view(Python.list(1, edges, 2, patch_size, patch_size))


# ── Dense BA Accumulation ─────────────────────────────────────────────────────

@always_inline
def _select6(i: Int, x0: Float32, x1: Float32, x2: Float32, x3: Float32, x4: Float32, x5: Float32) -> Float32:
    # Runtime 6-vector component selection. Kept for parity with earlier code
    # paths; the hot BA accumulation loop uses the compile-time version below.
    if i == 0:
        return x0
    if i == 1:
        return x1
    if i == 2:
        return x2
    if i == 3:
        return x3
    if i == 4:
        return x4
    return x5


@always_inline
def _select6c[i: Int](x0: Float32, x1: Float32, x2: Float32, x3: Float32, x4: Float32, x5: Float32) -> Float32:
    # Compile-time 6-vector component selection used inside comptime loops.
    # This lets the compiler unroll the small 6x6 Schur block accumulation.
    comptime if i == 0:
        return x0
    elif i == 1:
        return x1
    elif i == 2:
        return x2
    elif i == 3:
        return x3
    elif i == 4:
        return x4
    else:
        return x5


def ba_dense_accumulate_kernel(
    B: UnsafePointer[Float32, MutAnyOrigin],
    E: UnsafePointer[Float32, MutAnyOrigin],
    C: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    u: UnsafePointer[Float32, MutAnyOrigin],
    poses: LayoutTensor[DType.float32, POSES2_LT, MutAnyOrigin],
    patches: LayoutTensor[DType.float32, PATCHES4_LT, MutAnyOrigin],
    intrinsics: LayoutTensor[DType.float32, INTR4_LT, MutAnyOrigin],
    target: LayoutTensor[DType.float32, TARGET2_LT, MutAnyOrigin],
    weight: LayoutTensor[DType.float32, TARGET2_LT, MutAnyOrigin],
    ii: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    jj: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    kk: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    ku: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    total_rows: Int,
    active_poses: Int,
    unique_patches: Int,
    patch_size: Int,
    t0: Int,
):
    # Assemble the dense Schur-complement pieces for one Gauss-Newton BA step:
    #   B: pose-pose Hessian blocks
    #   E: pose-depth coupling
    #   C: inverse-depth diagonal
    #   v: pose RHS
    #   u: inverse-depth RHS
    #
    # Each edge contributes two residual rows. Because many edges touch the same
    # active poses and patch ids, every output is an atomic scatter.
    var tid = global_idx.x
    if tid < total_rows:
        # One thread accumulates either the x or y residual row for one edge.
        # The Schur blocks are small and shared by many edges, so the writes
        # below are atomic scatters into dense buffers.
        var residual_component = tid % 2
        var edge_id = tid // 2
        var unique_patch_id = Int(ku[edge_id])
        var ix_global = Int(ii[edge_id])
        var jx_global = Int(jj[edge_id])
        var patch_id = Int(kk[edge_id])

        var tx, ty, tz, qx, qy, qz, qw = _rel_se3(poses, ix_global, jx_global)
        var fx = intrinsics[0][0]
        var fy = intrinsics[1][0]
        var cx = intrinsics[2][0]
        var cy = intrinsics[3][0]

        var center = 1
        # The dense BA approximation uses the center pixel of each 3x3 patch to
        # form the depth Jacobian. This matches the original CUDA fastba kernel.
        var xi0 = (patches[patch_id, 0, center, center][0] - cx) / fx
        var xi1 = (patches[patch_id, 1, center, center][0] - cy) / fy
        var xi3 = patches[patch_id, 2, center, center][0]
        var X, Y, Z, W = _act_se3(tx, ty, tz, qx, qy, qz, qw, xi0, xi1, 1.0, xi3)

        var d: Float32 = 0.0
        if Z >= 0.2:
            d = 1.0 / Z
        var d2 = d * d
        var x1 = fx * (X / Z) + cx
        var y1 = fy * (Y / Z) + cy
        var rx = target[edge_id, 0][0]
        rx -= x1
        var ry = target[edge_id, 1][0]
        ry -= y1
        var err = sqrt(rx * rx + ry * ry)
        # Match CUDA's robust gating: ignore projections behind the camera,
        # far outside the image, or with a very large residual.
        var in_bounds = err < 128.0 and Z > 0.2 and x1 > -64.0 and y1 > -64.0 and x1 < 2.0 * cx + 64.0 and y1 < 2.0 * cy + 64.0
        var mask: Float32 = 0.0
        if in_bounds:
            mask = 1.0

        var ix = ix_global - t0
        var jx = jx_global - t0

        var r: Float32
        var w: Float32
        var jz: Float32
        var jj0: Float32
        var jj1: Float32
        var jj2: Float32
        var jj3: Float32
        var jj4: Float32
        var jj5: Float32
        # Projection Jacobian for either x or y residual. ji is obtained by
        # applying the SE(3) adjoint to jj below; jz is the inverse-depth column.
        if residual_component == 0:
            r = rx
            w = mask * weight[edge_id, 0][0]
            jz = fx * (tx * d - tz * (X * d2))
            jj0 = fx * W * d
            jj1 = 0.0
            jj2 = fx * -X * W * d2
            jj3 = fx * -X * Y * d2
            jj4 = fx * (1.0 + X * X * d2)
            jj5 = fx * -Y * d
        else:
            r = ry
            w = mask * weight[edge_id, 1][0]
            jz = fy * (ty * d - tz * (Y * d2))
            jj0 = 0.0
            jj1 = fy * W * d
            jj2 = fy * -Y * W * d2
            jj3 = fy * (-1.0 - Y * Y * d2)
            jj4 = fy * (X * Y * d2)
            jj5 = fy * X * d

        var ji0, ji1, ji2, ji3, ji4, ji5 = _adj_se3(tx, ty, tz, qx, qy, qz, qw, jj0, jj1, jj2, jj3, jj4, jj5)

        comptime for a in range(6):
            var ji_a = _select6c[a](ji0, ji1, ji2, ji3, ji4, ji5)
            var jj_a = _select6c[a](jj0, jj1, jj2, jj3, jj4, jj5)
            # B/E/C/v/u are dense accumulation buffers. Many edges contribute
            # to the same rows, so atomic scatter still requires raw element
            # addresses even though all read-only tensors are LayoutTensors.
            if ix >= 0:
                _ = Atomic.fetch_add(v + 6 * ix + a, -w * r * ji_a)
                _ = Atomic.fetch_add(E + (6 * ix + a) * unique_patches + unique_patch_id, -w * jz * ji_a)
            if jx >= 0:
                _ = Atomic.fetch_add(v + 6 * jx + a, w * r * jj_a)
                _ = Atomic.fetch_add(E + (6 * jx + a) * unique_patches + unique_patch_id, w * jz * jj_a)
            comptime for b in range(6):
                var ji_b = _select6c[b](ji0, ji1, ji2, ji3, ji4, ji5)
                var jj_b = _select6c[b](jj0, jj1, jj2, jj3, jj4, jj5)
                if ix >= 0:
                    _ = Atomic.fetch_add(B + (6 * ix + a) * 6 * active_poses + 6 * ix + b, w * ji_a * ji_b)
                if jx >= 0:
                    _ = Atomic.fetch_add(B + (6 * jx + a) * 6 * active_poses + 6 * jx + b, w * jj_a * jj_b)
                if ix >= 0 and jx >= 0:
                    _ = Atomic.fetch_add(B + (6 * ix + a) * 6 * active_poses + 6 * jx + b, -w * ji_a * jj_b)
                    _ = Atomic.fetch_add(B + (6 * jx + a) * 6 * active_poses + 6 * ix + b, -w * jj_a * ji_b)

        _ = Atomic.fetch_add(C + unique_patch_id, w * jz * jz)
        _ = Atomic.fetch_add(u + unique_patch_id, w * r * jz)


# ── Retraction Kernels ────────────────────────────────────────────────────────

def pose_retract_kernel(
    poses: LayoutTensor[DType.float32, POSES2_LT, MutAnyOrigin],
    update: LayoutTensor[DType.float32, UPDATE2_LT, MutAnyOrigin],
    active_poses: Int,
    t0: Int,
):
    # Apply solved pose increments to the active keyframe window. One thread
    # owns one pose row and writes [tx, ty, tz, qx, qy, qz, qw] in place.
    var tid = global_idx.x
    if tid < active_poses:
        var i = tid
        var pose = t0 + i
        var out_t0, out_t1, out_t2, out_q0, out_q1, out_q2, out_q3 = _retr_se3(
            update[i, 0][0],
            update[i, 1][0],
            update[i, 2][0],
            update[i, 3][0],
            update[i, 4][0],
            update[i, 5][0],
            poses[pose, 0][0],
            poses[pose, 1][0],
            poses[pose, 2][0],
            poses[pose, 3][0],
            poses[pose, 4][0],
            poses[pose, 5][0],
            poses[pose, 6][0],
        )
        poses[pose, 0] = out_t0
        poses[pose, 1] = out_t1
        poses[pose, 2] = out_t2
        poses[pose, 3] = out_q0
        poses[pose, 4] = out_q1
        poses[pose, 5] = out_q2
        poses[pose, 6] = out_q3


def patch_retract_kernel(
    patches: LayoutTensor[DType.float32, PATCHES4_LT, MutAnyOrigin],
    index: LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin],
    update: LayoutTensor[DType.float32, UPDATE1_LT, MutAnyOrigin],
    unique_patches: Int,
    patch_size: Int,
):
    # Apply solved inverse-depth increments to the unique patches touched by the
    # current window. One thread owns one unique patch id.
    var tid = global_idx.x
    if tid < unique_patches:
        var unique_patch_id = tid
        var patch_id = Int(index[unique_patch_id])
        # DPVO's patch depth update is one scalar per patch; every pixel in the
        # 3x3 patch shares that retracted inverse depth.
        var d = patches[patch_id, 2, 0, 0][0] + update[unique_patch_id][0]
        if d > 20.0:
            d = 1.0
        d = max(d, 1e-4)
        for row in range(patch_size):
            for col in range(patch_size):
                patches[patch_id, 2, row, col] = d


# ── Python Module Wrappers ────────────────────────────────────────────────────

def ba_dense_accumulate(args_obj: PythonObject) raises -> PythonObject:
    # PythonModuleBuilder exposes only a limited arity for def_function, so the
    # backend passes BA inputs as one Python tuple. This wrapper unpacks the
    # tuple, creates PyTorch accumulation buffers, wraps inputs in LayoutTensors,
    # and launches ba_dense_accumulate_kernel.
    var torch = Python.import_module("torch")
    var poses_obj = args_obj[0]
    var patches_obj = args_obj[1]
    var intrinsics_obj = args_obj[2]
    var target_obj = args_obj[3]
    var weight_obj = args_obj[4]
    var ii_obj = args_obj[5]
    var jj_obj = args_obj[6]
    var kk_obj = args_obj[7]
    var ku_obj = args_obj[8]
    var t0_obj = args_obj[9]
    var t1_obj = args_obj[10]
    var poses = poses_obj
    var poses_rank = Int(py=poses_obj.dim())
    var frames = Int(py=poses_obj.shape[poses_rank - 2])
    var rank = Int(py=patches_obj.dim())
    var patch_size = Int(py=patches_obj.shape[rank - 1])
    var num_patches = Int(py=patches_obj.shape[rank - 4])
    var patches = patches_obj
    var intrinsics = intrinsics_obj
    var target = target_obj
    var weight = weight_obj
    var ii = ii_obj
    var jj = jj_obj
    var kk = kk_obj
    var ku = ku_obj
    var edges = Int(py=ii.shape[0])
    var total_rows = edges * 2
    var t0 = Int(py=t0_obj)
    var t1 = Int(py=t1_obj)
    var active_poses = t1 - t0
    var unique_patches = Int(py=ku.max().item()) + 1

    # These buffers are the dense system returned to Python, where torch.linalg
    # solves the Schur complement before the retract kernels update tensors.
    var B: PythonObject = torch.zeros(Python.list(6 * active_poses, 6 * active_poses), device=poses.device, dtype=torch.float32)
    var E: PythonObject = torch.zeros(Python.list(6 * active_poses, unique_patches), device=poses.device, dtype=torch.float32)
    var C: PythonObject = torch.zeros(Python.list(unique_patches), device=poses.device, dtype=torch.float32)
    var v: PythonObject = torch.zeros(Python.list(6 * active_poses), device=poses.device, dtype=torch.float32)
    var u: PythonObject = torch.zeros(Python.list(unique_patches), device=poses.device, dtype=torch.float32)

    var poses_lt = LayoutTensor[DType.float32, POSES2_LT, MutAnyOrigin](
        _torch_float32_ptr(poses),
        RuntimeLayout[POSES2_LT].row_major(Index(frames, 7)),
    )
    var patches_lt = LayoutTensor[DType.float32, PATCHES4_LT, MutAnyOrigin](
        _torch_float32_ptr(patches),
        RuntimeLayout[PATCHES4_LT].row_major(Index(num_patches, 3, patch_size, patch_size)),
    )
    var intrinsics_lt = LayoutTensor[DType.float32, INTR4_LT, MutAnyOrigin](
        _torch_float32_ptr(intrinsics),
        RuntimeLayout[INTR4_LT].row_major(Index(4)),
    )
    var target_lt = LayoutTensor[DType.float32, TARGET2_LT, MutAnyOrigin](
        _torch_float32_ptr(target),
        RuntimeLayout[TARGET2_LT].row_major(Index(edges, 2)),
    )
    var weight_lt = LayoutTensor[DType.float32, TARGET2_LT, MutAnyOrigin](
        _torch_float32_ptr(weight),
        RuntimeLayout[TARGET2_LT].row_major(Index(edges, 2)),
    )
    var ii_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(ii),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var jj_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(jj),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var kk_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(kk),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )
    var ku_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(ku),
        RuntimeLayout[EDGE1_LT].row_major(Index(edges)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    ctx_ptr[].enqueue_function[ba_dense_accumulate_kernel, ba_dense_accumulate_kernel](
        _torch_float32_ptr(B),
        _torch_float32_ptr(E),
        _torch_float32_ptr(C),
        _torch_float32_ptr(v),
        _torch_float32_ptr(u),
        poses_lt,
        patches_lt,
        intrinsics_lt,
        target_lt,
        weight_lt,
        ii_lt,
        jj_lt,
        kk_lt,
        ku_lt,
        total_rows,
        active_poses,
        unique_patches,
        patch_size,
        t0,
        grid_dim=ceildiv(edges, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Python.tuple(B, E, C, v, u)


def pose_retract(poses_obj: PythonObject, update_obj: PythonObject, t0_obj: PythonObject, t1_obj: PythonObject) raises -> PythonObject:
    # Python wrapper for in-place pose updates. The Python side already solved
    # the 6D increments; Mojo only applies the manifold retraction on GPU.
    var poses = poses_obj
    var update = update_obj
    var poses_rank = Int(py=poses_obj.dim())
    var frames = Int(py=poses_obj.shape[poses_rank - 2])
    var t0 = Int(py=t0_obj)
    var t1 = Int(py=t1_obj)
    var active_poses = t1 - t0

    var poses_lt = LayoutTensor[DType.float32, POSES2_LT, MutAnyOrigin](
        _torch_float32_ptr(poses),
        RuntimeLayout[POSES2_LT].row_major(Index(frames, 7)),
    )
    var update_lt = LayoutTensor[DType.float32, UPDATE2_LT, MutAnyOrigin](
        _torch_float32_ptr(update),
        RuntimeLayout[UPDATE2_LT].row_major(Index(active_poses, 6)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    ctx_ptr[].enqueue_function[pose_retract_kernel, pose_retract_kernel](
        poses_lt,
        update_lt,
        active_poses,
        t0,
        grid_dim=ceildiv(active_poses, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Python.none()


def patch_retract(patches_obj: PythonObject, index_obj: PythonObject, update_obj: PythonObject) raises -> PythonObject:
    # Python wrapper for in-place inverse-depth updates. `index` maps compact
    # solver rows back to patch ids in the full patch tensor.
    var rank = Int(py=patches_obj.dim())
    var patch_size = Int(py=patches_obj.shape[rank - 1])
    var num_patches = Int(py=patches_obj.shape[rank - 4])
    var patches = patches_obj
    var index = index_obj
    var update = update_obj
    var unique_patches = Int(py=index.shape[0])

    var patches_lt = LayoutTensor[DType.float32, PATCHES4_LT, MutAnyOrigin](
        _torch_float32_ptr(patches),
        RuntimeLayout[PATCHES4_LT].row_major(Index(num_patches, 3, patch_size, patch_size)),
    )
    var index_lt = LayoutTensor[DType.int64, EDGE1_LT, MutAnyOrigin](
        _torch_int64_ptr(index),
        RuntimeLayout[EDGE1_LT].row_major(Index(unique_patches)),
    )
    var update_lt = LayoutTensor[DType.float32, UPDATE1_LT, MutAnyOrigin](
        _torch_float32_ptr(update),
        RuntimeLayout[UPDATE1_LT].row_major(Index(unique_patches)),
    )

    var ctx_ptr = _get_cached_context_ptr()
    comptime BLOCK_SIZE = 256
    ctx_ptr[].enqueue_function[patch_retract_kernel, patch_retract_kernel](
        patches_lt,
        index_lt,
        update_lt,
        unique_patches,
        patch_size,
        grid_dim=ceildiv(unique_patches, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Python.none()


# ── Python Module Export ──────────────────────────────────────────────────────

@export
def PyInit_dpvo_fastba_mojo_backends() -> PythonObject:
    # CPython extension-module entry point. The names registered here are the
    # stable API consumed by dpvo_fastba_mojo/backend.py.
    try:
        var m = PythonModuleBuilder("dpvo_fastba_mojo_backends")
        m.def_function[reproject]("reproject")
        m.def_function[ba_dense_accumulate]("ba_dense_accumulate")
        m.def_function[pose_retract]("pose_retract")
        m.def_function[patch_retract]("patch_retract")
        var module = m.finalize()
        _install_cached_context(module)
        return module
    except e:
        abort(String("Failed to create dpvo_fastba_mojo_backends module: ", e))
