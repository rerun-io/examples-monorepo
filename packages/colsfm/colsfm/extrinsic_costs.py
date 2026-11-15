"""The pyceres residuals of the extrinsics-only solve, and their Jacobians.

Split out of `colsfm.extrinsic_refinement` because a residual block is its own kind of
object: it answers "what does Ceres get when it evaluates these parameters", where the
solve module answers "which blocks does the problem hold" and the driver answers "how
often is the problem rebuilt". The prior residuals are `colsfm.pose_graph`'s
`RelativePoseCost` reused as-is; what is specific to extrinsic refinement is the
vectorised reprojection block below and the robust-loss algebra it carries.

## Why one residual block can hold a whole camera

A Python `pyceres` cost function costs roughly a millisecond per call, and Galileo's
refinement carries ~30 000 observations. One block per observation would be minutes per
Ceres iteration. `RigReprojectionCost` therefore holds **all** of one camera's
observations in a single block of `2 * n_obs` residuals, and applies the robust loss
*inside* `Evaluate`.

That substitution is exact, not an approximation. Ceres wraps a lossy block through
`ceres::Corrector`, which for any loss with `rho'' <= 0` — Cauchy, Huber and SoftL1 all
qualify — drops its rank-one term and reduces to scaling. Rather than reproduce the
scaling (which would report the wrong cost to the trust region), the cost function
returns the **square-root form** `r' = sqrt(rho(s)/s) * r` per observation, so that

* `|r'|^2 = rho(s)`, i.e. the block's cost is exactly `0.5 * sum_i rho(s_i)`, and
* `J'^T r' = rho'(s) J^T r`, i.e. the gradient is exactly Ceres' corrected gradient
  (`d(s * rho(s)/s)/ds = rho'(s)` does the algebra).

Only the Gauss-Newton Hessian differs from Ceres' by the Triggs-style rank-one term, and
both approximate the same objective, so the stationary point is the same.

`RepeatedCauchyLoss` is the same trick on the other side: one relative-prior block stands
for all the rig frames in which its camera pair co-observed (see
`colsfm.extrinsic_observations`), so scaling `rho` by that multiplicity is exactly the
blob's repeated blocks.

Every residual here is whitened — reprojection by `1 / reprojection_sigma_px` — and every
loss is `CauchyLoss(1.0)`, the convention `colsfm.extrinsic_refinement` explains.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pyceres
import pycolmap
from jaxtyping import Bool, Float64
from numpy import ndarray

from colsfm.ceres_pose import batched_skew
from colsfm.extrinsic_observations import CameraObservations, PixelObservations, PointsInVehicle
from colsfm.geometry import Matrix3
from colsfm.pose_graph import quaternion_plus_jacobian_transpose, rotation_matrix_from_quat_xyzw

_SMALL_SQUARED_NORM: Final[float] = 1e-8
"""Below this squared residual the Cauchy scale is taken from its series expansion."""

_PROJECTION_STEP_SCALE: Final[float] = 1e-6
"""Relative step of the central-difference camera Jacobian, in metres per metre."""

_LOCAL_PARAMETERS: Final[int] = 6
"""Rotation (3) then translation (3): the tangent space of one `vehicle_T_cam`."""


class RepeatedCauchyLoss(pyceres.LossFunction):
    """`CauchyLoss(1.0)` summed over `multiplicity` identical residual blocks.

    A rig's extrinsics are shared across rig frames, so all the blocks the blob adds for
    one camera pair evaluate the same residual. Scaling `rho` and its derivatives by the
    count is exactly equivalent to adding the blocks, and turns Galileo's 777 Python
    evaluations per iteration into 28.
    """

    def __init__(self, multiplicity: int) -> None:
        """Set up the loss.

        Args:
            multiplicity: How many identical blocks this one stands for; at least 1.
        """
        super().__init__()
        if multiplicity < 1:
            raise ValueError(f"multiplicity must be at least 1, got {multiplicity}")
        self._multiplicity: float = float(multiplicity)

    def Evaluate(self, squared_norm: float, out: Float64[ndarray, "3"]) -> None:
        """Write `multiplicity * [rho, rho', rho'']` of `CauchyLoss(1.0)` into `out`.

        The trampoline pyceres 2.6 installs is the C++ signature `Evaluate(double, double*)`,
        not the `evaluate(double) -> array` that the concrete losses expose to Python; a
        subclass that overrides the latter never gets called and Ceres aborts with
        `<Evaluate> not implemented`.

        Args:
            squared_norm: Squared norm of the block's residual.
            out: Three-element output buffer for `rho`, `rho'` and `rho''`.
        """
        denominator: float = 1.0 + squared_norm
        out[0] = self._multiplicity * np.log1p(squared_norm)
        out[1] = self._multiplicity / denominator
        out[2] = -self._multiplicity / denominator**2


def cauchy_square_root_scale(squared_norm: Float64[ndarray, " n_obs"]) -> tuple[Float64[ndarray, " n_obs"], Float64[ndarray, " n_obs"]]:
    """Per-observation `sqrt(rho(s)/s)` of `CauchyLoss(1.0)` and the derivative of `rho(s)/s`.

    Multiplying a residual by the first return value makes its squared norm equal
    `rho(s)`, so a block that carries many observations reports exactly the cost Ceres
    would report for the same observations as separate lossy blocks.

    Args:
        squared_norm: `s = |r|^2` per observation, whitened.

    Returns:
        `(sqrt(rho(s)/s), d(rho(s)/s)/ds)`, both shaped like the input. Near `s = 0` the
        series `rho(s)/s = 1 - s/2 + s^2/3` supplies both, since the closed forms are
        `0/0` there.
    """
    squared: Float64[ndarray, " n_obs"] = np.asarray(squared_norm, dtype=np.float64)
    is_small: Bool[ndarray, " n_obs"] = squared < _SMALL_SQUARED_NORM
    safe: Float64[ndarray, " n_obs"] = np.where(is_small, 1.0, squared)
    ratio: Float64[ndarray, " n_obs"] = np.where(is_small, 1.0 - 0.5 * squared, np.log1p(safe) / safe)
    derivative: Float64[ndarray, " n_obs"] = np.where(
        is_small, -0.5 + (2.0 / 3.0) * squared, (safe / (1.0 + safe) - np.log1p(safe)) / safe**2
    )
    return np.sqrt(ratio), derivative


class RigReprojectionCost(pyceres.CostFunction):
    """Every reprojection residual of one camera, in one block, against `vehicle_T_cam`.

    Parameter blocks are `[quaternion (4, x-y-z-w), translation (3)]` holding
    `vehicle_T_cam`; the residual is `2 * n_obs` long and already carries the whitening
    and the robust loss (see the module docstring for why that is exact).

    The projected point is `p_cam = vehicle_R_cam^T (p_vehicle - vehicle_t_cam)`, so with
    the manifold's right perturbation `R <- R Exp(d)` the analytic chain rule is
    `d p_cam / d[d, dt] = [skew(p_cam), -vehicle_R_cam^T]`; only `d img / d p_cam` is
    numeric.
    """

    def __init__(self, observations: CameraObservations, reprojection_sigma_px: float) -> None:
        """Set up one camera's block.

        Args:
            observations: That camera's frozen points and measured pixels.
            reprojection_sigma_px: `reprojection_error_standard_deviation`, in pixels.

        Raises:
            ValueError: When the camera has no observations or the sigma is not positive.
        """
        super().__init__()
        num_observations: int = int(observations.points_in_vehicle.shape[0])
        if num_observations == 0:
            raise ValueError(f"camera {observations.camera_params_id} has no observations to refine against")
        if reprojection_sigma_px <= 0.0:
            raise ValueError(f"reprojection sigma must be positive, got {reprojection_sigma_px}")
        self.set_num_residuals(2 * num_observations)
        self.set_parameter_block_sizes([4, 3])
        self._camera: pycolmap.Camera = observations.camera
        self._points_in_vehicle: PointsInVehicle = observations.points_in_vehicle
        self._observed_px: PixelObservations = observations.observed_px
        self._inverse_sigma: float = 1.0 / reprojection_sigma_px
        self._num_observations: int = num_observations
        self._vehicle_R_cam: Matrix3 = np.empty((3, 3), dtype=np.float64)
        self._plus_jacobian_t: Float64[ndarray, "3 4"] = np.empty((3, 4), dtype=np.float64)

    def _projection_jacobian(self, points_in_cam: Float64[ndarray, "n_obs 3"]) -> Float64[ndarray, "n_obs 2 3"]:
        """Central-difference `d img_from_cam / d p_cam`, six batched projections.

        Args:
            points_in_cam: The observed points in camera coordinates, metres.

        Returns:
            A 2x3 Jacobian per observation, pixels per metre.
        """
        step: Float64[ndarray, " n_obs"] = _PROJECTION_STEP_SCALE * np.maximum(
            1.0, np.abs(points_in_cam).max(axis=1)
        )
        jacobian: Float64[ndarray, "n_obs 2 3"] = np.empty((self._num_observations, 2, 3), dtype=np.float64)
        for axis in range(3):
            offset: Float64[ndarray, "n_obs 3"] = np.zeros_like(points_in_cam)
            offset[:, axis] = step
            forward: PixelObservations = self._camera.img_from_cam(points_in_cam + offset)
            backward: PixelObservations = self._camera.img_from_cam(points_in_cam - offset)
            jacobian[:, :, axis] = (forward - backward) / (2.0 * step)[:, None]
        return jacobian

    def Evaluate(self, parameters: list, residuals: ndarray, jacobians: list | None) -> bool:
        """Evaluate the whitened, robustified residual and, when asked, ambient Jacobians."""
        quat_xyzw, vehicle_t_cam = parameters
        vehicle_R_cam: Matrix3 = rotation_matrix_from_quat_xyzw(quat_xyzw, self._vehicle_R_cam)
        points_in_cam: Float64[ndarray, "n_obs 3"] = (self._points_in_vehicle - vehicle_t_cam) @ vehicle_R_cam
        error: PixelObservations = (self._camera.img_from_cam(points_in_cam) - self._observed_px) * self._inverse_sigma
        squared_norm: Float64[ndarray, " n_obs"] = np.einsum("ij,ij->i", error, error)
        scale, ratio_derivative = cauchy_square_root_scale(squared_norm)
        residuals[:] = (scale[:, None] * error).ravel()

        if jacobians is None or (jacobians[0] is None and jacobians[1] is None):
            return True

        projection_jacobian: Float64[ndarray, "n_obs 2 3"] = self._projection_jacobian(points_in_cam)
        point_jacobian: Float64[ndarray, "n_obs 3 6"] = np.empty(
            (self._num_observations, 3, _LOCAL_PARAMETERS), dtype=np.float64
        )
        point_jacobian[:, :, :3] = batched_skew(points_in_cam)
        point_jacobian[:, :, 3:] = -vehicle_R_cam.T
        local_jacobian: Float64[ndarray, "n_obs 2 6"] = self._inverse_sigma * (projection_jacobian @ point_jacobian)

        # d(g r)/dp with g = sqrt(rho(s)/s): the scaling plus the rank-one term that makes
        # the returned residual an exact function of the parameters, so that the analytic
        # Jacobian survives a finite-difference check.
        error_times_jacobian: Float64[ndarray, "n_obs 6"] = np.einsum("ni,nij->nj", error, local_jacobian)
        robust: Float64[ndarray, "n_obs 2 6"] = (
            scale[:, None, None] * local_jacobian
            + (ratio_derivative / scale)[:, None, None] * error[:, :, None] * error_times_jacobian[:, None, :]
        )
        if jacobians[0] is not None:
            plus_jacobian_t: Float64[ndarray, "3 4"] = quaternion_plus_jacobian_transpose(
                quat_xyzw, self._plus_jacobian_t
            )
            jacobians[0][:] = (robust[:, :, :3] @ plus_jacobian_t).ravel()
        if jacobians[1] is not None:
            jacobians[1][:] = robust[:, :, 3:].ravel()
        return True

