"""Python-facing wrappers of the Gauss-Newton kernels in gn_kernels.mojo.

Python drives the iterations (mast3r_slam/ops/optim.py: linearise -> dense CPU solve -> retract -> converged?). The torch
orchestration stays in Python: this module only launches the step and retraction kernels.
"""

from std.math import ceildiv, clamp
from std.python import Python, PythonObject
from layout import Idx, TileTensor, row_major

from gn_kernels import POSE_DIM, POSE_STRIDE, RAYS_THREADS, gauss_newton_rays_step_kernel, pose_retr_kernel
from python_interop import cached_context, torch_ptr

# A large edge is split over at most this many blocks (partial sums, added afterwards).
comptime RAYS_BLOCKS_PER_EDGE_MAX = 8


def pose_retr_py(poses_obj: PythonObject, dx_obj: PythonObject, num_fix_obj: PythonObject) raises -> PythonObject:
    """In place on the GPU: T_new[k] = Exp(dx[k - num_fix]) * T_old[k] for every pose after the first `num_fix`.

    poses_obj: [N_kf, 8] contiguous float32 Sim(3) poses; dx_obj: [N_kf - num_fix, 7] increments.
    """
    var num_fix = Int(py=num_fix_obj)
    var num_poses = Int(py=poses_obj.shape[0])
    if num_fix >= num_poses:
        return Python.none()
    var poses = poses_obj.contiguous().float()
    var dx = dx_obj.contiguous().float()
    cached_context()[].enqueue_function[pose_retr_kernel](
        TileTensor(torch_ptr[.float32](poses), row_major((num_poses, Idx[POSE_STRIDE]))),
        TileTensor(torch_ptr[.float32](dx), row_major((num_poses - num_fix, Idx[POSE_DIM]))),
        Int32(num_fix),
        grid_dim=ceildiv(num_poses - num_fix, 256),
        block_dim=256,
    )
    return Python.none()


def gauss_newton_rays_step_py(args: PythonObject) raises -> PythonObject:
    """One linearisation of the ray cost. Returns (hs [4, N_edges, 7, 7], gs [2, N_edges, 7]).

    args = (twc [N_kf, 8], xs [N_kf, N_pts, 3], cs [N_kf, N_pts, 1], ii [E], jj [E], idx_ii2jj [E, N_pts],
    valid_match [E, N_pts, 1] bool, q [E, N_pts, 1], sigma_ray, sigma_dist, c_thresh, q_thresh).
    """
    var torch = Python.import_module("torch")
    var twc = args[0].contiguous().float()
    var xs = args[1].contiguous().float()
    var cs = args[2].contiguous().float()
    var ii = args[3].contiguous()
    var jj = args[4].contiguous()
    var idx_ii2jj = args[5].contiguous()
    var valid_match = args[6].contiguous()
    var q = args[7].contiguous().float()

    var num_poses = Int(py=twc.shape[0])
    var num_points = Int(py=xs.shape[1])
    var num_edges = Int(py=ii.shape[0])
    var blocks_per_edge = clamp(ceildiv(num_points, 16384), 1, RAYS_BLOCKS_PER_EDGE_MAX)
    var num_partials = num_edges * blocks_per_edge
    var hs_partial = torch.zeros(Python.list(4, num_partials, POSE_DIM, POSE_DIM), device=twc.device, dtype=twc.dtype)
    var gs_partial = torch.zeros(Python.list(2, num_partials, POSE_DIM), device=twc.device, dtype=twc.dtype)

    cached_context()[].enqueue_function[gauss_newton_rays_step_kernel](
        TileTensor(torch_ptr[.float32](twc), row_major((num_poses, Idx[POSE_STRIDE]))),
        TileTensor(torch_ptr[.float32](xs), row_major((num_poses, num_points, Idx[3]))),
        TileTensor(torch_ptr[.float32](cs), row_major((num_poses, num_points))),
        TileTensor(torch_ptr[.int64](ii), row_major((num_edges,))),
        TileTensor(torch_ptr[.int64](jj), row_major((num_edges,))),
        TileTensor(torch_ptr[.int64](idx_ii2jj), row_major((num_edges, num_points))),
        TileTensor(torch_ptr[.uint8](valid_match), row_major((num_edges, num_points))),
        TileTensor(torch_ptr[.float32](q), row_major((num_edges, num_points))),
        TileTensor(torch_ptr[.float32](hs_partial), row_major((Idx[4], num_partials, Idx[POSE_DIM], Idx[POSE_DIM]))),
        TileTensor(torch_ptr[.float32](gs_partial), row_major((Idx[2], num_partials, Idx[POSE_DIM]))),
        Int32(blocks_per_edge),
        Float32(py=args[8]),
        Float32(py=args[9]),
        Float32(py=args[10]),
        Float32(py=args[11]),
        grid_dim=num_partials,
        block_dim=RAYS_THREADS,
    )
    # torch sums the partials on its own stream: hand over explicitly.
    torch.cuda.synchronize(device=twc.device)
    if blocks_per_edge == 1:
        return Python.tuple(hs_partial, gs_partial)
    var hs = hs_partial.reshape(4, num_edges, blocks_per_edge, POSE_DIM, POSE_DIM).sum(dim=2)
    var gs = gs_partial.reshape(2, num_edges, blocks_per_edge, POSE_DIM).sum(dim=2)
    return Python.tuple(hs, gs)
