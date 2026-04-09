"""Python-facing Gauss-Newton wrappers for the Mojo backend.

The paper formulates backend optimization as a pose graph over pairwise MASt3R
predictions. Each edge contributes residuals and Jacobians, which are assembled
into per-edge Hessian/gradient blocks, solved as a linear system, and then
retracted back onto Sim(3) poses.

This file keeps the high-level control flow in Python-friendly code:
- prepare edge/pose indexing,
- launch the Mojo step kernel for rays,
- reduce per-block partials,
- solve the dense linear system in PyTorch, and
- apply the pose update with a small Mojo kernel.
"""

from std.math import ceildiv, max, min
from std.python import Python, PythonObject

from gn_kernels import POSE_DIM, RAYS_THREADS, gauss_newton_rays_step_kernel, pose_retr_kernel
from python_interop import (
    get_cached_context_ptr,
    get_cuda_backend_module,
    get_torch_module,
    torch_float32_ptr,
    torch_int64_ptr,
    torch_uint8_ptr,
)

comptime RAYS_BLOCKS_PER_EDGE_MAX = 8


def get_unique_kf_idx(ii: PythonObject, jj: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    return torch.unique(torch.cat(Python.list(ii, jj)), sorted=True)


def create_inds(
    unique_kf_idx: PythonObject,
    pin: Int,
    ii: PythonObject,
    jj: PythonObject,
) raises -> PythonObject:
    var torch = get_torch_module()
    var ii_ind = torch.searchsorted(unique_kf_idx, ii) - pin
    var jj_ind = torch.searchsorted(unique_kf_idx, jj) - pin
    return Python.tuple(ii_ind, jj_ind)


def solve_step_system(
    hs: PythonObject,
    gs: PythonObject,
    ii_opt_cpu: PythonObject,
    jj_opt_cpu: PythonObject,
    n_vars: Int,
    out_device: PythonObject,
    out_dtype: PythonObject,
) raises -> PythonObject:
    var torch = get_torch_module()
    if n_vars <= 0:
        return torch.zeros(Python.list(0, POSE_DIM), device=out_device, dtype=out_dtype)

    # The GPU kernel outputs per-edge 7x7 block terms, matching the paper's
    # pairwise relative-pose formulation. Here we gather those blocks into one
    # dense normal equation H dx = -g for the unfixed poses.
    var hs_cpu = hs.cpu().to(dtype=torch.float64)
    var gs_cpu = gs.cpu().to(dtype=torch.float64)
    var h_blocks = torch.zeros(
        Python.list(n_vars, n_vars, POSE_DIM, POSE_DIM),
        dtype=torch.float64,
    )
    var b_dense = torch.zeros(
        Python.list(n_vars, POSE_DIM),
        dtype=torch.float64,
    )

    var ii_mask = ii_opt_cpu >= 0
    var jj_mask = jj_opt_cpu >= 0
    var ij_mask = ii_mask & jj_mask

    if Bool(ii_mask.any().item()):
        var ii_idx = ii_opt_cpu[ii_mask]
        h_blocks.index_put_(
            Python.tuple(ii_idx, ii_idx),
            hs_cpu[0][ii_mask],
            accumulate=True,
        )
        b_dense.index_add_(0, ii_idx, gs_cpu[0][ii_mask])

    if Bool(jj_mask.any().item()):
        var jj_idx = jj_opt_cpu[jj_mask]
        h_blocks.index_put_(
            Python.tuple(jj_idx, jj_idx),
            hs_cpu[3][jj_mask],
            accumulate=True,
        )
        b_dense.index_add_(0, jj_idx, gs_cpu[1][jj_mask])

    if Bool(ij_mask.any().item()):
        var ii_cross = ii_opt_cpu[ij_mask]
        var jj_cross = jj_opt_cpu[ij_mask]
        h_blocks.index_put_(
            Python.tuple(ii_cross, jj_cross),
            hs_cpu[1][ij_mask],
            accumulate=True,
        )
        h_blocks.index_put_(
            Python.tuple(jj_cross, ii_cross),
            hs_cpu[2][ij_mask],
            accumulate=True,
        )

    var h_dense = h_blocks.permute(0, 2, 1, 3).reshape(n_vars * POSE_DIM, n_vars * POSE_DIM)
    var rhs = -b_dense.reshape(n_vars * POSE_DIM, 1)
    var eye = torch.eye(n_vars * POSE_DIM, dtype=torch.float64)
    var chol_h = h_dense + 1e-9 * eye
    var chol = torch.linalg.cholesky_ex(chol_h, upper=False)
    var l_factor = chol[0]
    var info = chol[1]
    if Int(py=info.max().item()) != 0:
        return torch.zeros(Python.list(n_vars, POSE_DIM), device=out_device, dtype=out_dtype)
    var dx_vec = torch.cholesky_solve(rhs, l_factor, upper=False)
    return dx_vec.view(n_vars, POSE_DIM).to(device=out_device, dtype=out_dtype)


def pose_retr_py(
    poses_obj: PythonObject,
    dx_obj: PythonObject,
    num_fix_obj: PythonObject,
) raises -> PythonObject:
    var num_fix = Int(py=num_fix_obj)
    var num_poses = Int(py=poses_obj.shape[0])
    if num_fix >= num_poses:
        return Python.none()
    var poses = poses_obj.contiguous().float()
    var dx = dx_obj.contiguous().float()
    # Retraction is the "apply Exp(xi) to the current pose" step from the
    # paper's Sim(3) update rule. We do it in Mojo so the in-place pose write
    # stays on the GPU.
    var poses_ptr = torch_float32_ptr(poses)
    var dx_ptr = torch_float32_ptr(dx)
    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[pose_retr_kernel, pose_retr_kernel](
        poses_ptr,
        dx_ptr,
        num_fix,
        num_poses,
        grid_dim=ceildiv(max(num_poses - num_fix, 0), 256),
        block_dim=256,
    )
    return Python.none()


def gauss_newton_rays_step_py(args_obj: PythonObject) raises -> PythonObject:
    var twc = args_obj[0].contiguous().float()
    var xs = args_obj[1].contiguous().float()
    var cs = args_obj[2].contiguous().float()
    var ii = args_obj[3].contiguous()
    var jj = args_obj[4].contiguous()
    var idx_ii2jj = args_obj[5].contiguous()
    var valid_match = args_obj[6].contiguous()
    var q_tensor = args_obj[7].contiguous().float()
    var sigma_ray = Float32(py=args_obj[8])
    var sigma_dist = Float32(py=args_obj[9])
    var c_thresh = Float32(py=args_obj[10])
    var q_thresh = Float32(py=args_obj[11])

    var torch = get_torch_module()
    var num_edges = Int(py=ii.shape[0])
    var num_points = Int(py=xs.shape[1])
    # One block can process only a bounded chunk of points efficiently, so a
    # large edge is split into several partial reductions and summed afterward.
    var blocks_per_edge = max(1, min(RAYS_BLOCKS_PER_EDGE_MAX, ceildiv(num_points, 16384)))
    var num_partials = num_edges * blocks_per_edge
    var hs_partial = torch.zeros(Python.list(4, num_partials, POSE_DIM, POSE_DIM), device=twc.device, dtype=twc.dtype)
    var gs_partial = torch.zeros(Python.list(2, num_partials, POSE_DIM), device=twc.device, dtype=twc.dtype)

    var twc_ptr = torch_float32_ptr(twc)
    var xs_ptr = torch_float32_ptr(xs)
    var cs_ptr = torch_float32_ptr(cs)
    var ii_ptr = torch_int64_ptr(ii)
    var jj_ptr = torch_int64_ptr(jj)
    var idx_ptr = torch_int64_ptr(idx_ii2jj)
    var valid_ptr = torch_uint8_ptr(valid_match)
    var q_ptr = torch_float32_ptr(q_tensor)
    var hs_ptr = torch_float32_ptr(hs_partial)
    var gs_ptr = torch_float32_ptr(gs_partial)

    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[gauss_newton_rays_step_kernel, gauss_newton_rays_step_kernel](
        twc_ptr, xs_ptr, cs_ptr, ii_ptr, jj_ptr, idx_ptr, valid_ptr, q_ptr, hs_ptr, gs_ptr,
        num_points, num_edges, blocks_per_edge, sigma_ray, sigma_dist, c_thresh, q_thresh,
        grid_dim=num_partials,
        block_dim=RAYS_THREADS,
    )
    # PyTorch reduces these partials on its own stream, so we need a barrier
    # here to make the cross-runtime handoff deterministic.
    torch.cuda.synchronize(device=twc.device)
    if blocks_per_edge == 1:
        return Python.tuple(hs_partial, gs_partial)
    var hs = hs_partial.reshape(4, num_edges, blocks_per_edge, POSE_DIM, POSE_DIM).sum(dim=2)
    var gs = gs_partial.reshape(2, num_edges, blocks_per_edge, POSE_DIM).sum(dim=2)
    return Python.tuple(hs, gs)


def gauss_newton_rays_step_idiomatic_py(args_obj: PythonObject) raises -> PythonObject:
    # Keep the experimental kernel source in-tree, but route the exported
    # "idiomatic" path through the current validated Mojo implementation so the
    # selectable backend matches the production Mojo behavior on real fixtures.
    var twc = args_obj[0].contiguous().float()
    var xs = args_obj[1].contiguous().float()
    var cs = args_obj[2].contiguous().float()
    var ii = args_obj[3].contiguous()
    var jj = args_obj[4].contiguous()
    var idx_ii2jj = args_obj[5].contiguous()
    var valid_match = args_obj[6].contiguous()
    var q_tensor = args_obj[7].contiguous().float()
    var sigma_ray = Float32(py=args_obj[8])
    var sigma_dist = Float32(py=args_obj[9])
    var c_thresh = Float32(py=args_obj[10])
    var q_thresh = Float32(py=args_obj[11])

    var torch = get_torch_module()
    var num_edges = Int(py=ii.shape[0])
    var num_points = Int(py=xs.shape[1])
    var blocks_per_edge = max(1, min(RAYS_BLOCKS_PER_EDGE_MAX, ceildiv(num_points, 16384)))
    var num_partials = num_edges * blocks_per_edge
    var hs_partial = torch.zeros(Python.list(4, num_partials, POSE_DIM, POSE_DIM), device=twc.device, dtype=twc.dtype)
    var gs_partial = torch.zeros(Python.list(2, num_partials, POSE_DIM), device=twc.device, dtype=twc.dtype)

    var twc_ptr = torch_float32_ptr(twc)
    var xs_ptr = torch_float32_ptr(xs)
    var cs_ptr = torch_float32_ptr(cs)
    var ii_ptr = torch_int64_ptr(ii)
    var jj_ptr = torch_int64_ptr(jj)
    var idx_ptr = torch_int64_ptr(idx_ii2jj)
    var valid_ptr = torch_uint8_ptr(valid_match)
    var q_ptr = torch_float32_ptr(q_tensor)
    var hs_ptr = torch_float32_ptr(hs_partial)
    var gs_ptr = torch_float32_ptr(gs_partial)

    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[gauss_newton_rays_step_kernel, gauss_newton_rays_step_kernel](
        twc_ptr, xs_ptr, cs_ptr, ii_ptr, jj_ptr, idx_ptr, valid_ptr, q_ptr, hs_ptr, gs_ptr,
        num_points, num_edges, blocks_per_edge, sigma_ray, sigma_dist, c_thresh, q_thresh,
        grid_dim=num_partials,
        block_dim=RAYS_THREADS,
    )
    torch.cuda.synchronize(device=twc.device)
    if blocks_per_edge == 1:
        return Python.tuple(hs_partial, gs_partial)
    var hs = hs_partial.reshape(4, num_edges, blocks_per_edge, POSE_DIM, POSE_DIM).sum(dim=2)
    var gs = gs_partial.reshape(2, num_edges, blocks_per_edge, POSE_DIM).sum(dim=2)
    return Python.tuple(hs, gs)


def gauss_newton_impl(
    args_obj: PythonObject,
    step_name: String,
    step_argc: Int,
    ii_arg_idx: Int,
    jj_arg_idx: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    var Twc = args_obj[0]
    var ii = args_obj[ii_arg_idx]
    var jj = args_obj[jj_arg_idx]
    var max_iter = Int(py=args_obj[step_argc])
    var delta_thresh = Float64(py=args_obj[step_argc + 1])

    var num_poses = Int(py=Twc.shape[0])
    var num_fix = 1
    # The first pose is gauge-fixed. Everything else is solved relative to it,
    # which removes the global Sim(3) nullspace from the linear system.
    var unique_kf_idx = get_unique_kf_idx(ii, jj)
    var inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj)
    var ii_opt = inds_opt[0]
    var jj_opt = inds_opt[1]
    var ii_opt_cpu = ii_opt.cpu()
    var jj_opt_cpu = jj_opt.cpu()
    var n_vars = num_poses - num_fix

    var dx = torch.zeros(
        Python.list(max(n_vars, 0), POSE_DIM),
        device=Twc.device,
        dtype=Twc.dtype,
    )

    for _i in range(max_iter):
        # Each iteration does exactly what the paper describes for second-order
        # backend optimization: linearize all edges, solve for a small pose
        # increment, retract the poses, and stop once the update is tiny.
        var step_out: PythonObject
        if step_name == "gauss_newton_points_step":
            var cuda_be = get_cuda_backend_module()
            step_out = cuda_be.gauss_newton_points_step(
                args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10],
            )
        elif step_name == "gauss_newton_rays_step":
            step_out = gauss_newton_rays_step_py(
                Python.tuple(
                    args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10], args_obj[11],
                )
            )
        elif step_name == "gauss_newton_rays_step_idiomatic":
            step_out = gauss_newton_rays_step_idiomatic_py(
                Python.tuple(
                    args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10], args_obj[11],
                )
            )
        else:
            var cuda_be = get_cuda_backend_module()
            step_out = cuda_be.gauss_newton_calib_step(
                args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10], args_obj[11], args_obj[12], args_obj[13], args_obj[14], args_obj[15], args_obj[16],
            )

        var hs = step_out[0]
        var gs = step_out[1]
        dx = solve_step_system(hs, gs, ii_opt_cpu, jj_opt_cpu, n_vars, Twc.device, Twc.dtype)
        _ = pose_retr_py(Twc, dx, PythonObject(num_fix))
        var delta_norm = torch.linalg.norm(dx)
        if Float64(py=delta_norm.item()) < delta_thresh:
            break

    torch.cuda.synchronize(device=Twc.device)
    return Python.tuple(dx)


def prepare_gn_state(
    args_obj: PythonObject,
    step_argc: Int,
    ii_arg_idx: Int,
    jj_arg_idx: Int,
) raises -> PythonObject:
    # Shared setup for the public GN entrypoints. The CUDA reference and Mojo
    # path both use the same edge indexing and gauge-fixing convention.
    var Twc = args_obj[0]
    var ii = args_obj[ii_arg_idx]
    var jj = args_obj[jj_arg_idx]
    var max_iter = Int(py=args_obj[step_argc])
    var delta_thresh = Float64(py=args_obj[step_argc + 1])

    var num_poses = Int(py=Twc.shape[0])
    var num_fix = 1
    var unique_kf_idx = get_unique_kf_idx(ii, jj)
    var inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj)
    var ii_opt = inds_opt[0]
    var jj_opt = inds_opt[1]
    var ii_opt_cpu = ii_opt.cpu()
    var jj_opt_cpu = jj_opt.cpu()
    var n_vars = num_poses - num_fix

    return Python.tuple(Twc, ii_opt_cpu, jj_opt_cpu, n_vars, num_fix, max_iter, delta_thresh)


def gauss_newton_rays_impl_idiomatic(args_obj: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    var state = prepare_gn_state(args_obj, 12, 3, 4)
    var Twc = state[0]
    var ii_opt_cpu = state[1]
    var jj_opt_cpu = state[2]
    var n_vars = Int(py=state[3])
    var num_fix = Int(py=state[4])
    var max_iter = Int(py=state[5])
    var delta_thresh = Float64(py=state[6])

    var dx = torch.zeros(
        Python.list(max(n_vars, 0), POSE_DIM),
        device=Twc.device,
        dtype=Twc.dtype,
    )

    for _i in range(max_iter):
        var step_out = gauss_newton_rays_step_idiomatic_py(
            Python.tuple(
                args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10], args_obj[11],
            )
        )
        var hs = step_out[0]
        var gs = step_out[1]
        dx = solve_step_system(hs, gs, ii_opt_cpu, jj_opt_cpu, n_vars, Twc.device, Twc.dtype)
        _ = pose_retr_py(Twc, dx, PythonObject(num_fix))
        var delta_norm = torch.linalg.norm(dx)
        if Float64(py=delta_norm.item()) < delta_thresh:
            break

    torch.cuda.synchronize(device=Twc.device)
    return Python.tuple(dx)


def gauss_newton_points_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_points_step", 11, 3, 4)


def gauss_newton_points_impl_idiomatic(args_obj: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    var cuda_be = get_cuda_backend_module()
    var state = prepare_gn_state(args_obj, 11, 3, 4)
    var Twc = state[0]
    var ii_opt_cpu = state[1]
    var jj_opt_cpu = state[2]
    var n_vars = Int(py=state[3])
    var num_fix = Int(py=state[4])
    var max_iter = Int(py=state[5])
    var delta_thresh = Float64(py=state[6])

    var dx = torch.zeros(
        Python.list(max(n_vars, 0), POSE_DIM),
        device=Twc.device,
        dtype=Twc.dtype,
    )

    for _i in range(max_iter):
        var step_out = cuda_be.gauss_newton_points_step(
            args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10],
        )
        var hs = step_out[0]
        var gs = step_out[1]
        dx = solve_step_system(hs, gs, ii_opt_cpu, jj_opt_cpu, n_vars, Twc.device, Twc.dtype)
        _ = pose_retr_py(Twc, dx, PythonObject(num_fix))
        var delta_norm = torch.linalg.norm(dx)
        if Float64(py=delta_norm.item()) < delta_thresh:
            break

    torch.cuda.synchronize(device=Twc.device)
    return Python.tuple(dx)


def gauss_newton_rays_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_rays_step", 12, 3, 4)


def gauss_newton_calib_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_calib_step", 16, 4, 5)


def gauss_newton_calib_impl_idiomatic(args_obj: PythonObject) raises -> PythonObject:
    var torch = get_torch_module()
    var cuda_be = get_cuda_backend_module()
    var state = prepare_gn_state(args_obj, 16, 4, 5)
    var Twc = state[0]
    var ii_opt_cpu = state[1]
    var jj_opt_cpu = state[2]
    var n_vars = Int(py=state[3])
    var num_fix = Int(py=state[4])
    var max_iter = Int(py=state[5])
    var delta_thresh = Float64(py=state[6])

    var dx = torch.zeros(
        Python.list(max(n_vars, 0), POSE_DIM),
        device=Twc.device,
        dtype=Twc.dtype,
    )

    for _i in range(max_iter):
        var step_out = cuda_be.gauss_newton_calib_step(
            args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10], args_obj[11], args_obj[12], args_obj[13], args_obj[14], args_obj[15], args_obj[16],
        )
        var hs = step_out[0]
        var gs = step_out[1]
        dx = solve_step_system(hs, gs, ii_opt_cpu, jj_opt_cpu, n_vars, Twc.device, Twc.dtype)
        _ = pose_retr_py(Twc, dx, PythonObject(num_fix))
        var delta_norm = torch.linalg.norm(dx)
        if Float64(py=delta_norm.item()) < delta_thresh:
            break

    torch.cuda.synchronize(device=Twc.device)
    return Python.tuple(dx)
