from compiler import register
from tensor import InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList

from gn_kernels import RAYS_THREADS, gauss_newton_rays_step_kernel


@register("gauss_newton_rays_step_partial")
struct GaussNewtonRaysStepPartial:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        hs_partial: OutputTensor[dtype=DType.float32, rank=4, ...],
        gs_partial: OutputTensor[dtype=DType.float32, rank=3, ...],
        twc: InputTensor[dtype=DType.float32, rank=2, ...],
        xs: InputTensor[dtype=DType.float32, rank=3, ...],
        cs: InputTensor[dtype=DType.float32, rank=3, ...],
        ii: InputTensor[dtype=DType.int64, rank=1, ...],
        jj: InputTensor[dtype=DType.int64, rank=1, ...],
        idx_ii2jj: InputTensor[dtype=DType.int64, rank=2, ...],
        valid_match: InputTensor[dtype=DType.uint8, rank=3, ...],
        q_tensor: InputTensor[dtype=DType.float32, rank=3, ...],
        sigma_ray: InputTensor[dtype=DType.float32, rank=1, ...],
        sigma_dist: InputTensor[dtype=DType.float32, rank=1, ...],
        c_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        q_thresh: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target != "gpu":
            raise Error("gauss_newton_rays_step_partial only supports gpu target")

        var gpu_ctx = ctx.get_device_context()
        var num_edges = ii.dim_size(0)
        if num_edges <= 0:
            return

        var num_points = xs.dim_size(1)
        var num_partials = hs_partial.dim_size(1)
        var blocks_per_edge = num_partials // num_edges
        var zero_idx = IndexList[1](0)

        gpu_ctx.enqueue_function[
            gauss_newton_rays_step_kernel,
            gauss_newton_rays_step_kernel,
        ](
            twc.unsafe_ptr(),
            xs.unsafe_ptr(),
            cs.unsafe_ptr(),
            ii.unsafe_ptr(),
            jj.unsafe_ptr(),
            idx_ii2jj.unsafe_ptr(),
            valid_match.unsafe_ptr(),
            q_tensor.unsafe_ptr(),
            hs_partial.unsafe_ptr(),
            gs_partial.unsafe_ptr(),
            num_points,
            num_edges,
            blocks_per_edge,
            sigma_ray.load[1](zero_idx)[0],
            sigma_dist.load[1](zero_idx)[0],
            c_thresh.load[1](zero_idx)[0],
            q_thresh.load[1](zero_idx)[0],
            grid_dim=num_partials,
            block_dim=RAYS_THREADS,
        )
        gpu_ctx.synchronize()
