"""Kernels over the flat tangent vector `[poses 6 | rigs 6 | calib K | points 3]` and its block diagonal.

A block kernel runs one thread per node block and covers every block type in one launch: the branch on the
block index picks the comptime block size. The damped block is CASPAR's `normalize` matrix,
M = B + lambda diag(B) + lambda 1e-6 I, solved exactly (dense Cholesky, unrolled at compile time).
"""

from max.gpu import global_idx

from layout import MutTileTensor

from factor_kernels import F32In, F64Ptr, I32In, L2, Vec, VecIn, damped_solve, reduce_add, tri, tri_index
from camera_models import CameraModel
from geometry import Pose, retract

comptime F32Out = MutTileTensor[DType.float32, L2, MutUntrackedOrigin]



@always_inline
def precond_block[D: Int](
    r: Vec, z: Vec, s: Vec, p: VecIn, w: VecIn, hblk: VecIn, v: Int, h: Int, lam: Float32, alpha: Float32, first: Bool
) -> Float32:
    comptime assert r.flat_rank == 1 and z.flat_rank == 1 and s.flat_rank == 1 and p.flat_rank == 1 and w.flat_rank == 1
    var rb = Array[Float32, D](fill=0.0)
    comptime for i in range(D):
        if first:
            rb[i] = r[v + i][0]
            s[v + i] = 0.0
        else:
            # The previous iteration's r and step updates, deferred to here to save a launch.
            rb[i] = r[v + i][0] - alpha * w[v + i][0]
            r[v + i] = rb[i]
            s[v + i] = s[v + i][0] + alpha * p[v + i][0]
    var zb = damped_solve[D](hblk, h, rb, lam)
    var rho: Float32 = 0.0
    comptime for i in range(D):
        z[v + i] = zb[i]
        rho += rb[i] * zb[i]
    return rho


def pcg_precond_kernel[K: Int](
    r: Vec,
    z: Vec,
    s: Vec,
    p: VecIn,
    w: VecIn,
    hblk: VecIn,
    n6: Int32,
    n_calib: Int32,
    lam: Float32,
    alpha: Float32,
    first: Int32,
    scalars: F64Ptr,
    rho_slot: Int32,
):
    """Camera blocks: (k > 0) r -= alpha w, s += alpha p; z = M^-1 r; rho += r.z. (k = 0) s = 0."""
    var b = global_idx.x
    var nn6 = Int(n6)
    var rho: Float32 = 0.0
    if b < nn6:
        rho = precond_block[6](r, z, s, p, w, hblk, 6 * b, tri(6) * b, lam, alpha, first != 0)
    elif b < nn6 + Int(n_calib):
        var c = b - nn6
        rho = precond_block[K](r, z, s, p, w, hblk, 6 * nn6 + K * c, tri(6) * nn6 + tri(K) * c, lam, alpha, first != 0)
    reduce_add(scalars, Int(rho_slot), rho)


def pcg_direction_kernel(
    r: VecIn,
    z: VecIn,
    p: Vec,
    mp: Vec,
    w: Vec,
    n_cam: Int32,
    first: Int32,
    scalars: F64Ptr,
    rho_slot: Int32,
):
    """Camera part: beta = rho_k / rho_{k-1}, p = z + beta p, Mp = r + beta Mp, w = Mp. Point part: w = 0."""
    comptime assert r.flat_rank == 1 and z.flat_rank == 1 and p.flat_rank == 1 and mp.flat_rank == 1 and w.flat_rank == 1
    var i = global_idx.x
    if i >= Int(w.dim[0]()):
        return
    if i >= Int(n_cam):
        w[i] = 0.0
        return
    if first != 0:
        p[i] = z[i][0]
        mp[i] = r[i][0]
        w[i] = r[i][0]
        return
    # The slots sit two apart: [rho_k, pw_k, rho_{k+1}, ...]. The generated BetaFromNumDenom keeps rho_k.
    var beta = Float32(scalars[unsafe_offset=Int(rho_slot)] / scalars[unsafe_offset=Int(rho_slot) - 2])
    var mpi = r[i][0] + beta * mp[i][0]
    p[i] = z[i][0] + beta * p[i][0]
    mp[i] = mpi
    w[i] = mpi


def dot_kernel(a: VecIn, b: VecIn, n: Int32, scalars: F64Ptr, slot: Int32):
    comptime assert a.flat_rank == 1 and b.flat_rank == 1
    var i = global_idx.x
    var v: Float32 = 0.0
    if i < Int(n):
        v = a[i][0] * b[i][0]
    reduce_add(scalars, Int(slot), v)


def axpy_kernel(y: Vec, x: VecIn, alpha: Float32, n: Int32):
    comptime assert y.flat_rank == 1 and x.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        y[i] = y[i][0] + alpha * x[i][0]


def point_solve_kernel(g: VecIn, dst: Vec, hblk: VecIn, point_v: Int32, point_h: Int32, n_points: Int32, lam: Float32):
    """dst_p = -M_pp^-1 g_p per free point, with the damped point block M_pp (the Schur H_pp)."""
    comptime assert g.flat_rank == 1 and dst.flat_rank == 1
    var j = global_idx.x
    if j >= Int(n_points):
        return
    var v = Int(point_v) + 3 * j
    var rhs = Array[Float32, 3](fill=0.0)
    comptime for i in range(3):
        rhs[i] = -g[v + i][0]
    var sol = damped_solve[3](hblk, Int(point_h) + tri(3) * j, rhs, lam)
    comptime for i in range(3):
        dst[v + i] = sol[i]


@always_inline
def retract_vector[D: Int](
    src: F32In, dst: F32Out, row: Int, col0: Int, v: Int, s: VecIn, g: VecIn, hblk: VecIn, h: Int, lam: Float32
) -> Float32:
    """dst[row, col0:col0+D] = src + s[v:v+D]; returns s.(g + lambda diag(B) s) over the block."""
    comptime assert src.flat_rank == 2 and dst.flat_rank == 2 and s.flat_rank == 1 and g.flat_rank == 1
    var pred: Float32 = 0.0
    comptime for i in range(D):
        var si = s[v + i][0]
        dst[row, col0 + i] = src[row, col0 + i][0] + si
        pred += si * (g[v + i][0] + lam * hblk[h + tri_index[D](i, i)][0] * si)
    return pred


@always_inline
def retract_pose(src: F32In, dst: F32Out, row: Int, v: Int, s: VecIn, g: VecIn, hblk: VecIn, h: Int, lam: Float32) -> Float32:
    comptime assert src.flat_rank == 2 and dst.flat_rank == 2 and s.flat_rank == 1 and g.flat_rank == 1
    var pose = Pose(src[row, 0][0], src[row, 1][0], src[row, 2][0], src[row, 3][0], src[row, 4][0], src[row, 5][0], src[row, 6][0])
    var out = retract(pose, s[v][0], s[v + 1][0], s[v + 2][0], s[v + 3][0], s[v + 4][0], s[v + 5][0])
    dst[row, 0] = out.qx
    dst[row, 1] = out.qy
    dst[row, 2] = out.qz
    dst[row, 3] = out.qw
    dst[row, 4] = out.tx
    dst[row, 5] = out.ty
    dst[row, 6] = out.tz
    var pred: Float32 = 0.0
    comptime for i in range(6):
        var si = s[v + i][0]
        pred += si * (g[v + i][0] + lam * hblk[h + tri_index[6](i, i)][0] * si)
    return pred


@always_inline
def copy_row[W: Int](src: F32In, dst: F32Out, row: Int):
    comptime assert src.flat_rank == 2 and dst.flat_rank == 2
    comptime for i in range(W):
        dst[row, i] = src[row, i][0]


def retract_kernel[Model: CameraModel](
    poses: F32In,
    rigs: F32In,
    intrinsics: F32In,
    points: F32In,
    new_poses: F32Out,
    new_rigs: F32Out,
    new_intrinsics: F32Out,
    new_points: F32Out,
    pose_off: I32In,
    rig_off: I32In,
    calib_off: I32In,
    point_off: I32In,
    s: VecIn,
    g: VecIn,
    hblk: VecIn,
    n6: Int32,
    n_calib: Int32,
    lam: Float32,
    scalars: F64Ptr,
    pred_slot: Int32,
):
    """One thread per node (frames, cameras, points): trial = retract(state, s); fused s.(g + lambda D s)."""
    comptime assert pose_off.flat_rank == 1 and rig_off.flat_rank == 1 and calib_off.flat_rank == 1
    comptime assert point_off.flat_rank == 1
    comptime K = Model.NUM_CALIB
    var i = global_idx.x
    var n_frames = Int(pose_off.dim[0]())
    var n_cameras = Int(rig_off.dim[0]())
    var n_points = Int(point_off.dim[0]())
    var nn6 = Int(n6)
    var calib_v = 6 * nn6
    var calib_h = tri(6) * nn6
    var point_v = calib_v + K * Int(n_calib)
    var point_h = calib_h + tri(K) * Int(n_calib)
    var pred: Float32 = 0.0
    if i < n_frames:
        var v = Int(pose_off[i][0])
        if v >= 0:
            pred = retract_pose(poses, new_poses, i, v, s, g, hblk, tri(6) * (v // 6), lam)
        else:
            copy_row[7](poses, new_poses, i)
    elif i < n_frames + n_cameras:
        var c = i - n_frames
        var v = Int(rig_off[c][0])
        if v >= 0:
            pred = retract_pose(rigs, new_rigs, c, v, s, g, hblk, tri(6) * (v // 6), lam)
        else:
            copy_row[7](rigs, new_rigs, c)
        copy_row[12](intrinsics, new_intrinsics, c)
        var vc = Int(calib_off[c][0])
        if vc >= 0:
            var h = calib_h + tri(K) * ((vc - calib_v) // K)
            comptime for k in range(K):
                comptime col = Model.calib_column(k)
                var sk = s[vc + k][0]
                new_intrinsics[c, col] = intrinsics[c, col][0] + sk
                pred += sk * (g[vc + k][0] + lam * hblk[h + tri_index[K](k, k)][0] * sk)
    elif i < n_frames + n_cameras + n_points:
        var j = i - n_frames - n_cameras
        var v = Int(point_off[j][0])
        if v >= 0:
            pred = retract_vector[3](points, new_points, j, 0, v, s, g, hblk, point_h + tri(3) * ((v - point_v) // 3), lam)
        else:
            copy_row[3](points, new_points, j)
    reduce_add(scalars, Int(pred_slot), pred)
