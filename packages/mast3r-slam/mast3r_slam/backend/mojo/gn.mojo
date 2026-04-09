from std.math import ceildiv, max
from std.gpu.host import DeviceContext
from std.python import Python, PythonObject

from gn_kernels import (
    POSE_DIM,
    RAYS_THREADS,
    assemble_b_dense_kernel,
    assemble_h_dense_kernel,
    gauss_newton_rays_step_kernel,
    pose_retr_kernel,
)


def get_cached_context_ptr() raises -> UnsafePointer[DeviceContext, MutAnyOrigin]:
    var module = Python.import_module("mast3r_slam_mojo_backends")
    var ctx_addr = Int(py=module._ctx_addr)
    return UnsafePointer[DeviceContext, MutAnyOrigin](
        unsafe_from_address=ctx_addr
    )


def get_torch_module() raises -> PythonObject:
    return Python.import_module("torch")


def get_cuda_backend_module() raises -> PythonObject:
    return Python.import_module("mast3r_slam._backends")


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


def solve_dense_system(
    h_dense: PythonObject,
    b_dense: PythonObject,
    n_vars: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    if n_vars <= 0:
        return torch.zeros(Python.list(0, POSE_DIM), device=h_dense.device, dtype=h_dense.dtype)

    var h64 = h_dense.to(dtype=torch.float64)
    var b64 = b_dense.to(dtype=torch.float64)
    var rhs = -b64.reshape(n_vars * POSE_DIM, 1)
    var eye = torch.eye(n_vars * POSE_DIM, device=h_dense.device, dtype=torch.float64)
    var chol_h = h64 + 1e-9 * eye
    var chol = torch.linalg.cholesky_ex(chol_h, upper=False)
    var l_factor = chol[0]
    var info = chol[1]
    if Int(py=info.max().item()) != 0:
        return torch.zeros(Python.list(n_vars, POSE_DIM), device=h_dense.device, dtype=h_dense.dtype)
    var dx_vec = torch.cholesky_solve(rhs, l_factor, upper=False)
    return dx_vec.to(dtype=h_dense.dtype).view(n_vars, POSE_DIM)


def assemble_dense_system_gpu(
    hs: PythonObject,
    gs: PythonObject,
    ii_opt: PythonObject,
    jj_opt: PythonObject,
    n_vars: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    var num_edges = Int(py=hs.shape[1])
    var dim = n_vars * POSE_DIM
    var h_dense = torch.zeros(
        Python.list(dim, dim),
        device=hs.device,
        dtype=torch.float64,
    )
    var b_dense = torch.zeros(
        Python.list(n_vars, POSE_DIM),
        device=gs.device,
        dtype=torch.float64,
    )
    if n_vars <= 0:
        return Python.tuple(h_dense, b_dense)

    var hs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=hs.data_ptr()))
    var gs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=gs.data_ptr()))
    var ii_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=ii_opt.contiguous().data_ptr()))
    var jj_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=jj_opt.contiguous().data_ptr()))
    var h_ptr = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=h_dense.data_ptr()))
    var b_ptr = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=b_dense.data_ptr()))

    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[assemble_h_dense_kernel, assemble_h_dense_kernel](
        hs_ptr, ii_ptr, jj_ptr, h_ptr, num_edges, n_vars,
        grid_dim=(ceildiv(dim, 16), ceildiv(dim, 16)),
        block_dim=(16, 16),
    )
    ctx_ptr[].enqueue_function[assemble_b_dense_kernel, assemble_b_dense_kernel](
        gs_ptr, ii_ptr, jj_ptr, b_ptr, num_edges, n_vars,
        grid_dim=(ceildiv(n_vars, 16), 1),
        block_dim=(16, POSE_DIM),
    )
    ctx_ptr[].synchronize()
    return Python.tuple(h_dense, b_dense)


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
    var poses_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=poses.data_ptr()))
    var dx_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=dx.data_ptr()))
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
    var hs = torch.zeros(Python.list(4, num_edges, POSE_DIM, POSE_DIM), device=twc.device, dtype=twc.dtype)
    var gs = torch.zeros(Python.list(2, num_edges, POSE_DIM), device=twc.device, dtype=twc.dtype)

    var twc_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=twc.data_ptr()))
    var xs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=xs.data_ptr()))
    var cs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=cs.data_ptr()))
    var ii_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=ii.data_ptr()))
    var jj_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=jj.data_ptr()))
    var idx_ptr = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=idx_ii2jj.data_ptr()))
    var valid_ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(py=valid_match.data_ptr()))
    var q_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=q_tensor.data_ptr()))
    var hs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=hs.data_ptr()))
    var gs_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=gs.data_ptr()))

    var ctx_ptr = get_cached_context_ptr()
    ctx_ptr[].enqueue_function[gauss_newton_rays_step_kernel, gauss_newton_rays_step_kernel](
        twc_ptr, xs_ptr, cs_ptr, ii_ptr, jj_ptr, idx_ptr, valid_ptr, q_ptr, hs_ptr, gs_ptr,
        num_points, num_edges, sigma_ray, sigma_dist, c_thresh, q_thresh,
        grid_dim=num_edges,
        block_dim=RAYS_THREADS,
    )
    return Python.tuple(hs, gs)


def gauss_newton_impl(
    args_obj: PythonObject,
    step_name: String,
    step_argc: Int,
) raises -> PythonObject:
    var torch = get_torch_module()
    var cuda_be = get_cuda_backend_module()
    var Twc = args_obj[0]
    var ii = args_obj[3]
    var jj = args_obj[4]
    var max_iter = Int(py=args_obj[step_argc])
    var delta_thresh = Float64(py=args_obj[step_argc + 1])

    var num_poses = Int(py=Twc.shape[0])
    var num_fix = 1
    var unique_kf_idx = get_unique_kf_idx(ii, jj)
    var inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj)
    var ii_opt = inds_opt[0]
    var jj_opt = inds_opt[1]
    var n_vars = num_poses - num_fix

    var dx = torch.zeros(
        Python.list(max(n_vars, 0), POSE_DIM),
        device=Twc.device,
        dtype=Twc.dtype,
    )

    for _i in range(max_iter):
        var step_out: PythonObject
        if step_name == "gauss_newton_points_step":
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
            step_out = cuda_be.gauss_newton_calib_step(
                args_obj[0], args_obj[1], args_obj[2], args_obj[3], args_obj[4], args_obj[5], args_obj[6], args_obj[7], args_obj[8], args_obj[9], args_obj[10], args_obj[11], args_obj[12], args_obj[13], args_obj[14], args_obj[15],
            )

        var hs = step_out[0]
        var gs = step_out[1]
        var sys = assemble_dense_system_gpu(hs, gs, ii_opt, jj_opt, n_vars)
        var h_dense = sys[0]
        var b_dense = sys[1]
        dx = solve_dense_system(h_dense, b_dense, n_vars).to(dtype=Twc.dtype)
        _ = pose_retr_py(Twc, dx, PythonObject(num_fix))
        var delta_norm = torch.linalg.norm(dx)
        if Float64(py=delta_norm.item()) < delta_thresh:
            break

    return Python.tuple(dx)


def gauss_newton_points_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_points_step", 11)


def gauss_newton_rays_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_rays_step", 12)


def gauss_newton_calib_impl(args_obj: PythonObject) raises -> PythonObject:
    return gauss_newton_impl(args_obj, "gauss_newton_calib_step", 16)
