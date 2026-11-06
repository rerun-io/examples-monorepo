"""Sim3 pose retraction of MASt3R-SLAM's Gauss-Newton loop: T_new[k] = Exp(dx[k - num_fix]) * T_old[k] for k >= num_fix.

The same update as mast3r-slam's `pose_retr_kernel` (backend/mojo/gn_kernels.mojo,
a port of its CUDA `pose_retr_kernel`), written on this package's `Sim3`, so it
equals `Sim3.retr` row for row. Poses are (num_poses, 8) rows
(tx ty tz qx qy qz qw s) updated in place; `dx` holds one 7-vector
(tau, phi, sigma) per pose after the first `num_fix` (the gauge-fixed poses).

Other Mojo code builds against it with `-I <lietorch-mojo>/lietorch_mojo/mojo`:

    from pose_retr import pose_retr_kernel
    ctx.enqueue_function[pose_retr_kernel[.float32]](poses, dx, Int64(num_fix), grid_dim=..., block_dim=POSE_RETR_THREADS)

`pose_retr_row` is the per-pose function the kernel and the CPU loop share.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.math import ceildiv
from std.python import PythonObject
from std.utils.static_tuple import StaticTuple

from layout import ImmTileTensor, MutTileTensor

from launch import CPURuntime, GPURuntime, InRows, OutRows, in_rows, out_rows
from lie_linalg import Mat
from sim3 import Sim3

comptime POSE_RETR_THREADS = 256
comptime POSE_WIDTH = Sim3[.float32].N
comptime TANGENT_WIDTH = Sim3[.float32].K


@inline(.always)
def pose_retr_row[
    dtype: DType
](k: Int, poses: MutTileTensor[dtype, ...], dx: ImmTileTensor[dtype, ...], num_fix: Int):
    """Pose row `k` (>= num_fix) becomes Exp(dx[k - num_fix]) * pose[k]."""
    comptime assert poses.flat_rank == 2 and poses.static_shape[1] == POSE_WIDTH, "poses must be (n, 8)"
    var old = Mat[dtype, POSE_WIDTH, 1]()
    comptime for j in range(POSE_WIDTH):
        old[j] = poses[k, j][0]
    var xi = Mat[dtype, TANGENT_WIDTH, 1].load(dx, k - num_fix)
    Sim3[dtype].Exp(xi).mul(Sim3[dtype].from_data(old)).data().store(poses, k)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](POSE_RETR_THREADS))
def pose_retr_kernel[
    dtype: DType
](poses: OutRows[dtype, POSE_WIDTH], dx: InRows[dtype, TANGENT_WIDTH], num_fix: Int64):
    """One thread per unfixed pose; launch ceildiv(num_poses - num_fix, POSE_RETR_THREADS) blocks of POSE_RETR_THREADS."""
    var k = global_idx.x + Int(num_fix)
    if k < Int(poses.dim[0]()):
        pose_retr_row(k, poses, dx, Int(num_fix))


def pose_retr_cuda[
    dtype: DType
](mut rt: GPURuntime, stream: Int, poses_address: PythonObject, dx_address: PythonObject, num_poses: Int, num_fix: Int) raises:
    """Enqueue the retraction on `stream` (PyTorch's current stream); caller guarantees 0 <= num_fix < num_poses."""
    var num_vars = num_poses - num_fix
    var func = rt.ctx.compile_function[pose_retr_kernel[dtype]]()
    rt.torch_stream(stream).enqueue_function(
        func,
        out_rows[dtype, POSE_WIDTH](poses_address, num_poses),
        in_rows[dtype, TANGENT_WIDTH](dx_address, num_vars),
        Int64(num_fix),
        grid_dim=ceildiv(num_vars, POSE_RETR_THREADS),
        block_dim=POSE_RETR_THREADS,
    )


def pose_retr_cpu[dtype: DType](rt: CPURuntime, poses_address: PythonObject, dx_address: PythonObject, num_poses: Int, num_fix: Int) raises:
    var poses = out_rows[dtype, POSE_WIDTH](poses_address, num_poses)
    var dx = in_rows[dtype, TANGENT_WIDTH](dx_address, num_poses - num_fix)

    def row(i: Int) {imm}:
        pose_retr_row(i + num_fix, poses, dx, num_fix)

    rt.for_each_row(row, num_poses - num_fix)
