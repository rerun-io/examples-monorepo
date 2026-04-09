"""Python-facing Gauss-Newton wrappers for the Mojo backend.

Overview
--------
This file is the **orchestration layer** between Python/PyTorch and the low-level
GPU kernels in `gn_kernels.mojo`. It implements the full GN iteration loop
described in §3.3 of the MASt3R-SLAM paper (arXiv 2412.12392).

The Gauss-Newton (GN) loop works as follows (each iteration):

  1. **Linearise** — launch `gauss_newton_rays_step_kernel` on the GPU to compute
     per-edge Hessian (H) and gradient (g) blocks from all point correspondences.
  2. **Reduce** — if an edge was split across multiple GPU blocks (large number
     of points), sum the partial reductions.
  3. **Assemble & solve** — `solve_step_system()` gathers edge blocks into one
     dense normal-equation system  H·dx = −g  and solves it via Cholesky
     factorisation (PyTorch, on CPU in float64 for numerical stability).
  4. **Retract** — `pose_retr_kernel` applies the solved 7D Sim(3) increment
     to each unfixed pose via the Exp map.
  5. **Check convergence** — stop if ‖dx‖ < threshold.

Gauge fixing
------------
The first pose is always fixed (not optimised). This removes the global Sim(3)
gauge freedom — without it the system would be rank-deficient.

Three step variants
-------------------
- `gauss_newton_rays` — ray-based cost, linearised in Mojo (this file).
- `gauss_newton_points` — point-based cost, delegated to the CUDA backend.
- `gauss_newton_calib` — calibration cost, delegated to the CUDA backend.

Only the rays step is fully implemented in Mojo; the other two fall through to
the existing CUDA extension so the Python pipeline can swap backends without
changing call sites.
"""

from std.math import ceildiv, max, min
from std.python import Python, PythonObject
from std.utils.index import Index
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE

from gn_kernels import (
    POSE_DIM, POSE_STRIDE, RAYS_THREADS,
    POSES_LT, DX_LT, XS_LT, CS_LT, EDGES_LT, IDX_LT,
    gauss_newton_rays_step_kernel, pose_retr_kernel,
)
from python_interop import (
    get_cached_context_ptr,
    get_cuda_backend_module,
    get_torch_module,
    torch_float32_ptr,
    torch_int64_ptr,
    torch_uint8_ptr,
)

# Maximum number of GPU blocks per edge. Large edges (many points) are split
# into up to this many partial reductions, then summed afterwards.
comptime RAYS_BLOCKS_PER_EDGE_MAX = 8


# ── Index helpers ─────────────────────────────────────────────────────────────

def get_unique_kf_idx(ii: PythonObject, jj: PythonObject) raises -> PythonObject:
    """Return the sorted, unique keyframe indices that appear in any edge.

    Used to build the dense variable ordering: only keyframes that appear in
    at least one edge are included in the normal-equation system.
    """
    var torch = get_torch_module()
    return torch.unique(torch.cat(Python.list(ii, jj)), sorted=True)


def create_inds(
    unique_kf_idx: PythonObject,
    pin: Int,
    ii: PythonObject,
    jj: PythonObject,
) raises -> PythonObject:
    """Map raw keyframe indices (ii, jj) to dense optimisation-variable indices.

    `pin` is the number of gauge-fixed poses (always 1). The result is shifted
    by −pin so that the first optimisable variable has index 0. Fixed poses
    get index −1, which is later masked out when filling the normal equations.
    """
    var torch = get_torch_module()
    var ii_ind = torch.searchsorted(unique_kf_idx, ii) - pin
    var jj_ind = torch.searchsorted(unique_kf_idx, jj) - pin
    return Python.tuple(ii_ind, jj_ind)


# ── Normal-equation assembly and solve ────────────────────────────────────────

def solve_step_system(
    hs: PythonObject,
    gs: PythonObject,
    ii_opt_cpu: PythonObject,
    jj_opt_cpu: PythonObject,
    n_vars: Int,
    out_device: PythonObject,
    out_dtype: PythonObject,
) raises -> PythonObject:
    """Assemble per-edge Hessian/gradient blocks into a dense system and solve.

    The GPU kernel produces four 7×7 Hessian sub-blocks per edge:
      hs[0] = H_ii   (pose i vs pose i)
      hs[1] = H_ij   (pose i vs pose j, off-diagonal)
      hs[2] = H_ji   (transpose of H_ij)
      hs[3] = H_jj   (pose j vs pose j)
    and two gradient vectors:
      gs[0] = g_i, gs[1] = g_j

    This function scatters those blocks into a dense (n_vars×7, n_vars×7)
    matrix and (n_vars×7, 1) vector, adds a tiny Tikhonov regulariser (1e-9),
    and solves via Cholesky decomposition in float64 for numerical stability.

    Returns dx of shape [n_vars, 7] on the original device/dtype.
    """
    var torch = get_torch_module()
    if n_vars <= 0:
        return torch.zeros(Python.list(0, POSE_DIM), device=out_device, dtype=out_dtype)

    # Move everything to CPU float64 for the Cholesky solve — the GPU kernel
    # uses float32 which isn't stable enough for the normal equations.
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

    # Masks: edges where the i/j endpoint is an optimisable variable (index >= 0).
    # Gauge-fixed poses have index −1 and are excluded from the system.
    var ii_mask = ii_opt_cpu >= 0
    var jj_mask = jj_opt_cpu >= 0
    var ij_mask = ii_mask & jj_mask  # both endpoints are optimisable

    # ── Scatter diagonal blocks (H_ii and H_jj) ──
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

    # ── Scatter off-diagonal blocks (H_ij and H_ji) ──
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

    # ── Reshape from block form [n, n, 7, 7] to dense matrix [n*7, n*7] ──
    var h_dense = h_blocks.permute(0, 2, 1, 3).reshape(n_vars * POSE_DIM, n_vars * POSE_DIM)
    var rhs = -b_dense.reshape(n_vars * POSE_DIM, 1)

    # ── Solve H·dx = −g via Cholesky ──
    # A small Tikhonov regulariser (1e-9·I) ensures positive-definiteness
    # even when some DOFs are poorly constrained.
    var eye = torch.eye(n_vars * POSE_DIM, dtype=torch.float64)
    var chol_h = h_dense + 1e-9 * eye
    var chol = torch.linalg.cholesky_ex(chol_h, upper=False)
    var l_factor = chol[0]
    var info = chol[1]
    if Int(py=info.max().item()) != 0:
        # Cholesky failed (not positive-definite) — return zero update.
        return torch.zeros(Python.list(n_vars, POSE_DIM), device=out_device, dtype=out_dtype)
    var dx_vec = torch.cholesky_solve(rhs, l_factor, upper=False)
    return dx_vec.view(n_vars, POSE_DIM).to(device=out_device, dtype=out_dtype)


# ── Pose retraction (Python entry point) ─────────────────────────────────────

def pose_retr_py(
    poses_obj: PythonObject,
    dx_obj: PythonObject,
    num_fix_obj: PythonObject,
) raises -> PythonObject:
    """Apply solved Sim(3) increments to the pose tensor (in-place on GPU).

    Called from Python after the Cholesky solve. Launches the Mojo GPU kernel
    `pose_retr_kernel` which does:  T_new[k] = Exp(dx[k]) · T_old[k]
    for every unfixed pose k.

    Args:
        poses_obj: Tensor [N_kf, 8] of Sim(3) poses (modified in place).
        dx_obj:    Tensor [N_vars, 7] of Lie-algebra increments.
        num_fix_obj: Number of gauge-fixed poses (typically 1).
    """
    var num_fix = Int(py=num_fix_obj)
    var num_poses = Int(py=poses_obj.shape[0])
    if num_fix >= num_poses:
        return Python.none()
    var poses = poses_obj.contiguous().float()
    var dx = dx_obj.contiguous().float()
    var num_vars = num_poses - num_fix
    # Wrap PyTorch tensors in LayoutTensors for structured kernel access.
    var poses_lt = LayoutTensor[DType.float32, POSES_LT, MutAnyOrigin](
        torch_float32_ptr(poses),
        RuntimeLayout[POSES_LT].row_major(Index(num_poses, POSE_STRIDE)),
    )
    var dx_lt = LayoutTensor[DType.float32, DX_LT, MutAnyOrigin](
        torch_float32_ptr(dx),
        RuntimeLayout[DX_LT].row_major(Index(num_vars, POSE_DIM)),
    )
    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[pose_retr_kernel, pose_retr_kernel](
        poses_lt,
        dx_lt,
        num_fix,
        num_poses,
        grid_dim=ceildiv(max(num_poses - num_fix, 0), 256),
        block_dim=256,
    )
    return Python.none()


# ── Ray step: Python → Mojo GPU kernel ───────────────────────────────────────

def gauss_newton_rays_step_py(args_obj: PythonObject) raises -> PythonObject:
    """One linearisation pass for the ray cost — launches the Mojo GPU kernel.

    Unpacks PyTorch tensors from the Python tuple, extracts raw GPU pointers,
    launches `gauss_newton_rays_step_kernel`, synchronises, and reduces any
    partial blocks.

    Args (packed in args_obj tuple):
        [0] twc          — Sim(3) poses [N_kf, 8]
        [1] xs           — pointmaps [N_kf, N_pts, 3]
        [2] cs           — confidences [N_kf, N_pts]
        [3] ii           — edge i-indices [N_edges]
        [4] jj           — edge j-indices [N_edges]
        [5] idx_ii2jj    — point-match mapping [N_edges, N_pts]
        [6] valid_match  — validity mask [N_edges, N_pts]
        [7] q_tensor     — match quality [N_edges, N_pts]
        [8] sigma_ray    — ray residual std dev
        [9] sigma_dist   — distance residual std dev
        [10] c_thresh    — confidence threshold
        [11] q_thresh    — match quality threshold

    Returns (hs, gs):
        hs — Hessian blocks [4, N_edges, 7, 7]
        gs — gradient blocks [2, N_edges, 7]
    """
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


# ── Full GN iteration loop ───────────────────────────────────────────────────

def gauss_newton_impl(
    args_obj: PythonObject,
    step_name: String,
    step_argc: Int,
    ii_arg_idx: Int,
    jj_arg_idx: Int,
) raises -> PythonObject:
    """Generic Gauss-Newton iteration loop shared by all three cost variants.

    Runs up to `max_iter` iterations of: linearise → solve → retract → check.
    The `step_name` selects which linearisation kernel to use:
      - "gauss_newton_rays_step"  → Mojo kernel (this backend)
      - "gauss_newton_points_step" → CUDA extension (fallback)
      - "gauss_newton_calib_step"  → CUDA extension (fallback)

    `step_argc`, `ii_arg_idx`, `jj_arg_idx` parameterise the argument layout
    so the three public entry points can share this implementation.
    """
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
    """Shared initialisation for all GN entry points (index mapping + gauge fix)."""
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


# ── Public entry points (one per cost variant) ───────────────────────────────
# These are registered in mast3r_slam_mojo_backends.mojo as the Python-callable
# functions. The argument index parameters encode where ii/jj and max_iter sit
# in each variant's argument tuple.

def gauss_newton_points_impl(args_obj: PythonObject) raises -> PythonObject:
    """GN loop with point-based cost (delegates linearisation to CUDA)."""
    return gauss_newton_impl(args_obj, "gauss_newton_points_step", 11, 3, 4)


def gauss_newton_rays_impl(args_obj: PythonObject) raises -> PythonObject:
    """GN loop with ray-based cost (linearisation runs in Mojo)."""
    return gauss_newton_impl(args_obj, "gauss_newton_rays_step", 12, 3, 4)


def gauss_newton_calib_impl(args_obj: PythonObject) raises -> PythonObject:
    """GN loop with calibration cost (delegates linearisation to CUDA)."""
    return gauss_newton_impl(args_obj, "gauss_newton_calib_step", 16, 4, 5)

