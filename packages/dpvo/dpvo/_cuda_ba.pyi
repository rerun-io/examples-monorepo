"""Type stubs for the _cuda_ba C++/CUDA extension (built at install time)."""

from torch import Tensor

def forward(
    poses: Tensor,
    patches: Tensor,
    intrinsics: Tensor,
    target: Tensor,
    weight: Tensor,
    lmbda: Tensor,
    depth_prior: Tensor,
    depth_prior_weight: Tensor,
    depth_prior_lambda: float,
    depth_prior_huber: float,
    ii: Tensor,
    jj: Tensor,
    kk: Tensor,
    M: int,
    t0: int,
    t1: int,
    iterations: int,
    eff_impl: bool,
) -> list[Tensor]: ...
def neighbors(ii: Tensor, jj: Tensor) -> list[Tensor]: ...
def reproject(poses: Tensor, patches: Tensor, intrinsics: Tensor, ii: Tensor, jj: Tensor, kk: Tensor) -> Tensor: ...
def solve_system(J_Ginv_i: Tensor, J_Ginv_j: Tensor, ii: Tensor, jj: Tensor, res: Tensor, ep: float, lm: float, freen: int) -> list[Tensor]: ...
