"""Inference pixel motion with per-camera intrinsics and the projective_ops near plane."""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.math import max
from std.utils.static_tuple import StaticTuple

from layout import ImmTileTensor, MutTileTensor

from ba import L2, _pose
from corr import L1, L3, L4, NUM_THREADS
from lie_math import sqrt_
from se3 import Vec4, act_se3, rel_se3


@always_inline
def _normalize_quaternion(q: Vec4) -> Vec4:
    var squared_norm = q[3].fma(q[3], q[2].fma(q[2], q[1].fma(q[1], q[0] * q[0])))
    return q / sqrt_(squared_norm) if squared_norm > 0.0 else q


@always_inline
def _project(point: Vec4, calibration: Vec4) -> Tuple[Float32, Float32]:
    var inverse_depth = 1.0 / max(point[2], Float32(0.1))
    return (
        calibration[0] * (inverse_depth * point[0]) + calibration[2],
        calibration[1] * (inverse_depth * point[1]) + calibration[3],
    )


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
    var ki = Vec4(intrinsics[i, 0][0], intrinsics[i, 1][0], intrinsics[i, 2][0], intrinsics[i, 3][0])
    var kj = Vec4(intrinsics[j, 0][0], intrinsics[j, 1][0], intrinsics[j, 2][0], intrinsics[j, 3][0])
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
            var point = Vec4(
                (patches[k, 0, row, col][0] - ki[2]) / ki[0],
                (patches[k, 1, row, col][0] - ki[3]) / ki[1],
                1.0,
                patches[k, 2, row, col][0],
            )
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
