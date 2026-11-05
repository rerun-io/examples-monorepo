"""Runtime guards for flag-dependent MHR forward outputs."""

from jaxtyping import Float32
from torch import Tensor


def expect_mhr_vertices(output: object) -> Float32[Tensor, "batch n_verts 3"]:
    """Validate an MHR forward result configured to return vertices only."""
    if not isinstance(output, Tensor):
        raise TypeError(f"Expected MHR vertices tensor, got {type(output).__name__}.")
    if output.ndim != 3 or output.shape[-1] != 3:
        raise ValueError(f"Expected MHR vertices with shape [B, N, 3], got {tuple(output.shape)}.")
    return output


def expect_mhr_tensor_tuple(output: object, *, min_len: int) -> tuple[Tensor, ...]:
    """Validate an MHR forward result configured to return multiple tensors."""
    if not isinstance(output, tuple):
        raise TypeError(f"Expected MHR tensor tuple, got {type(output).__name__}.")
    if len(output) < min_len:
        raise ValueError(f"Expected at least {min_len} MHR outputs, got {len(output)}.")
    if not all(isinstance(item, Tensor) for item in output):
        raise TypeError("Expected all MHR outputs to be torch.Tensor instances.")
    return output
