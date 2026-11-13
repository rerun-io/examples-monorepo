"""Inference pixel motion with per-camera intrinsics and the projective_ops near plane."""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.math import max
from std.utils.static_tuple import StaticTuple

from layout import ImmTileTensor, MutTileTensor

from ba import L2, _pose
from corr import L1, L3, L4, NUM_THREADS
from lie_math import sqrt_
from se3 import Vec4, act_se3, rel_se3
from so3 import SO3


@always_inline
def _normalize_quaternion(q: Vec4) -> Vec4:
    """lietorch-mojo's normalising SO3 constructor (Eigen's pinned FMA order), as an xyzw vector."""
    var unit = SO3[DType.float32](q[0], q[1], q[2], q[3])
    return Vec4(unit.x, unit.y, unit.z, unit.w)


@always_inline
def _intrinsics(intrinsics: ImmTileTensor[DType.float32, L2, _], frame: Int) -> Vec4:
    """``fx fy cx cy`` of one frame."""
    return Vec4(intrinsics[frame, 0][0], intrinsics[frame, 1][0], intrinsics[frame, 2][0], intrinsics[frame, 3][0])


@always_inline
def _patch_point(patches: ImmTileTensor[DType.float32, L4, _], k: Int, row: Int, col: Int, ki: Vec4) -> Vec4:
    """One pixel of patch ``k`` back-projected in its source camera: normalised ``x y 1`` and inverse depth."""
    return Vec4((patches[k, 0, row, col][0] - ki[2]) / ki[0], (patches[k, 1, row, col][0] - ki[3]) / ki[1], 1.0, patches[k, 2, row, col][0])


@always_inline
def _project(point: Vec4, calibration: Vec4) -> Tuple[Float32, Float32]:
    var inverse_depth = 1.0 / max(point[2], Float32(0.1))
    return (
        calibration[0] * (inverse_depth * point[0]) + calibration[2],
        calibration[1] * (inverse_depth * point[1]) + calibration[3],
    )


@always_inline
def project_points_element(
    n: Int,
    poses: ImmTileTensor[DType.float32, L2, _],
    patches: ImmTileTensor[DType.float32, L4, _],
    intrinsics: ImmTileTensor[DType.float32, L2, _],
    source: ImmTileTensor[DType.int64, L1, _],
    target: ImmTileTensor[DType.int64, L1, _],
    patch: ImmTileTensor[DType.int64, L1, _],
    coordinates: MutTileTensor[DType.float32, L4, _],
):
    """Project all pixels of one patch using the edge's normalized relative pose."""
    comptime assert poses.flat_rank == 2 and patches.flat_rank == 4 and intrinsics.flat_rank == 2
    comptime assert source.flat_rank == 1 and target.flat_rank == 1 and patch.flat_rank == 1 and coordinates.flat_rank == 4
    var i = Int(source[n][0])
    var j = Int(target[n][0])
    var k = Int(patch[n][0])
    var ki = _intrinsics(intrinsics, i)
    var kj = _intrinsics(intrinsics, j)
    var ti, qi = _pose(poses, i)
    var tj, qj = _pose(poses, j)
    var tij, qij = rel_se3(ti, _normalize_quaternion(qi), tj, _normalize_quaternion(qj))
    qij = _normalize_quaternion(qij)
    for row in range(Int(patches.dim[2]())):
        for col in range(Int(patches.dim[3]())):
            var point = _patch_point(patches, k, row, col, ki)
            var x, y = _project(act_se3(tij, qij, point), kj)
            coordinates[n, row, col, 0] = x
            coordinates[n, row, col, 1] = y


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def project_points_kernel(
    poses: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    patches: ImmTileTensor[DType.float32, L4, ImmUntrackedOrigin],
    intrinsics: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    source: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    target: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    patch: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    coordinates: MutTileTensor[DType.float32, L4, MutUntrackedOrigin],
):
    var n = global_idx.x
    if n < Int(source.dim[0]()):
        project_points_element(n, poses, patches, intrinsics, source, target, patch, coordinates)


@always_inline
def flow_magnitude_element(
    n: Int,
    beta: Float32,
    poses: ImmTileTensor[DType.float32, L2, _],
    patches: ImmTileTensor[DType.float32, L4, _],
    intrinsics: ImmTileTensor[DType.float32, L2, _],
    source: ImmTileTensor[DType.int64, L1, _],
    target: ImmTileTensor[DType.int64, L1, _],
    patch: ImmTileTensor[DType.int64, L1, _],
    flow: MutTileTensor[DType.float32, L3, _],
):
    """Compute both motion terms for every pixel of one edge's patch."""
    comptime assert poses.flat_rank == 2 and patches.flat_rank == 4 and intrinsics.flat_rank == 2
    comptime assert source.flat_rank == 1 and target.flat_rank == 1 and patch.flat_rank == 1 and flow.flat_rank == 3
    var i = Int(source[n][0])
    var j = Int(target[n][0])
    var k = Int(patch[n][0])
    var ki = _intrinsics(intrinsics, i)
    var kj = _intrinsics(intrinsics, j)
    var ti, qi = _pose(poses, i)
    var tj, qj = _pose(poses, j)
    qi = _normalize_quaternion(qi)
    qj = _normalize_quaternion(qj)
    var tij, qij = rel_se3(ti, qi, tj, qj)
    var tii, qii = rel_se3(ti, qi, ti, qi)
    qij = _normalize_quaternion(qij)
    qii = _normalize_quaternion(qii)
    for row in range(Int(patches.dim[2]())):
        for col in range(Int(patches.dim[3]())):
            var point = _patch_point(patches, k, row, col, ki)
            var x0, y0 = _project(act_se3(tii, qii, point), ki)
            var x1, y1 = _project(act_se3(tij, qij, point), kj)
            var x2, y2 = _project(act_se3(tij, Vec4(0.0, 0.0, 0.0, 1.0), point), kj)
            var dx1 = x1 - x0
            var dy1 = y1 - y0
            var dx2 = x2 - x0
            var dy2 = y2 - y0
            flow[n, row, col] = beta * sqrt_(dx1 * dx1 + dy1 * dy1) + (1.0 - beta) * sqrt_(dx2 * dx2 + dy2 * dy2)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def flow_magnitude_kernel(
    beta: Float32,
    poses: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    patches: ImmTileTensor[DType.float32, L4, ImmUntrackedOrigin],
    intrinsics: ImmTileTensor[DType.float32, L2, ImmUntrackedOrigin],
    source: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    target: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    patch: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    flow: MutTileTensor[DType.float32, L3, MutUntrackedOrigin],
):
    var n = global_idx.x
    if n < Int(source.dim[0]()):
        flow_magnitude_element(n, beta, poses, patches, intrinsics, source, target, patch, flow)
