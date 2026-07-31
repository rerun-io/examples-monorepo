from functools import partial

import numpy as np
from jaxtyping import Float
from scipy.optimize import OptimizeResult, least_squares


def solve_optimal_focal_shift(
    uv: Float[np.ndarray, "... 2"],
    xyz: Float[np.ndarray, "... 3"],
) -> tuple[Float[np.ndarray, ""], np.floating[np.generic]]:
    """Solve ``min |focal * xy / (z + shift) - uv|`` for shift and focal.

    Args:
        uv: Float view-plane coordinates shaped ``... 2``.
        xyz: Float affine point coordinates shaped ``... 3``.

    Returns:
        Optimal 0-D float32 shift array and floating-point focal value.
    """
    uv_n2: Float[np.ndarray, "n 2"] = uv.reshape(-1, 2)
    xy_n2: Float[np.ndarray, "n 2"] = xyz[..., :2].reshape(-1, 2)
    z_n: Float[np.ndarray, "n"] = xyz[..., 2].reshape(-1)

    def residual(
        uv_values_n2: Float[np.ndarray, "n 2"],
        xy_values_n2: Float[np.ndarray, "n 2"],
        z_values_n: Float[np.ndarray, "n"],
        shift_1: Float[np.ndarray, "1"],
    ) -> Float[np.ndarray, "errors"]:
        xy_projected_n2: Float[np.ndarray, "n 2"] = xy_values_n2 / (z_values_n + shift_1)[:, None]
        focal: np.floating[np.generic] = (xy_projected_n2 * uv_values_n2).sum() / np.square(xy_projected_n2).sum()
        error: Float[np.ndarray, "errors"] = (focal * xy_projected_n2 - uv_values_n2).ravel()
        return error

    solution: OptimizeResult = least_squares(partial(residual, uv_n2, xy_n2, z_n), x0=0.0, ftol=1e-3, method="lm")
    optimal_shift: Float[np.ndarray, ""] = solution.x.squeeze().astype(np.float32)
    xy_projected_n2: Float[np.ndarray, "n 2"] = xy_n2 / (z_n + optimal_shift)[:, None]
    optimal_focal: np.floating[np.generic] = (xy_projected_n2 * uv_n2).sum() / np.square(xy_projected_n2).sum()
    return optimal_shift, optimal_focal


def solve_optimal_shift(
    uv: Float[np.ndarray, "... 2"],
    xyz: Float[np.ndarray, "... 3"],
    focal: float,
) -> Float[np.ndarray, ""]:
    """Solve ``min |focal * xy / (z + shift) - uv|`` for shift.

    Args:
        uv: Float view-plane coordinates shaped ``... 2``.
        xyz: Float affine point coordinates shaped ``... 3``.
        focal: Known focal length relative to half the image diagonal.

    Returns:
        Optimal 0-D float32 shift array.
    """
    uv_n2: Float[np.ndarray, "n 2"] = uv.reshape(-1, 2)
    xy_n2: Float[np.ndarray, "n 2"] = xyz[..., :2].reshape(-1, 2)
    z_n: Float[np.ndarray, "n"] = xyz[..., 2].reshape(-1)

    def residual(
        uv_values_n2: Float[np.ndarray, "n 2"],
        xy_values_n2: Float[np.ndarray, "n 2"],
        z_values_n: Float[np.ndarray, "n"],
        shift_1: Float[np.ndarray, "1"],
    ) -> Float[np.ndarray, "errors"]:
        xy_projected_n2: Float[np.ndarray, "n 2"] = xy_values_n2 / (z_values_n + shift_1)[:, None]
        error: Float[np.ndarray, "errors"] = (focal * xy_projected_n2 - uv_values_n2).ravel()
        return error

    solution: OptimizeResult = least_squares(partial(residual, uv_n2, xy_n2, z_n), x0=0.0, ftol=1e-3, method="lm")
    optimal_shift: Float[np.ndarray, ""] = solution.x.squeeze().astype(np.float32)
    return optimal_shift
