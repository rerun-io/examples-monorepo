from jaxtyping import Float, Int
from torch import Tensor

from mini_dpvo import _cuda_ba

neighbors = _cuda_ba.neighbors
reproject = _cuda_ba.reproject

def BA(
    poses: Float[Tensor, "..."],
    patches: Float[Tensor, "..."],
    intrinsics: Float[Tensor, "..."],
    target: Float[Tensor, "..."],
    weight: Float[Tensor, "..."],
    lmbda: float,
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    t0: int,
    t1: int,
    iterations: int = 2,
) -> None:
    return _cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations)
