from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, TypeAlias, TypedDict

import jax.numpy as npj
import numpy as np
from einops import rearrange
from jax import jit
from jaxopt import LevenbergMarquardt
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Array, Bool, Float
from numpy import ndarray

from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.ops.mano.mano_jax import ManoSimpleLayerJAX


class LossWeights(TypedDict):
    keypoint_2d: float
    depth: float
    temp: float
    pose_reg: float


@jit
def proj_3d_vectorized(
    xyz_hom: Float[Array, "n_frames n_kpts 4"],
    P: Float[Array, "n_views 3 4"],
) -> Float[Array, "n_frames n_views n_kpts 2"]:
    """
    Projects 3D points to 2D using the projection matrix for a batch of frames and views.

    xyz_hom: [n_frames, 21, 4]  -> [x, y, z, 1]
    P:       [n_views, 3, 4]    -> K [R|t]

    returns: [n_frames, n_views, n_kpts, 2]
    """
    xyz_hom: Float[Array, "n_frames 1 4 n_kpts"] = rearrange(
        xyz_hom, "n_frames n_kpts xyz_hom -> n_frames 1 xyz_hom n_kpts"
    )
    P_b: Float[Array, "1 n_views 3 4"] = rearrange(P, "n_views n m -> 1 n_views n m")
    # [1, n_views, 3, 4] @ [n_frames, 1, 4, n_kpts] -> [n_frames, n_views, 3, n_kpts]
    uv_hom: Float[Array, "n_frames n_views 3 n_kpts"] = P_b @ xyz_hom
    uv_hom: Float[Array, "n_frames n_views n_kpts 3"] = rearrange(
        uv_hom, "n_frames n_views xyz_hom n_kpts -> n_frames n_views n_kpts xyz_hom"
    )
    # Robust division to avoid Inf/NaN during early iterations (z ≈ 0)
    denom = uv_hom[..., 2:]
    eps = npj.array(1e-8, dtype=denom.dtype)
    denom_safe = npj.where(npj.abs(denom) < eps, npj.sign(denom) * eps, denom)
    uv = uv_hom[..., :2] / denom_safe
    return uv


FwdKinematics: TypeAlias = Callable[
    [Float[Array, "b 48"], Float[Array, "b 10"], Float[Array, "b 3"]],
    tuple[Float[Array, "b n_verts=778 3"], Float[Array, "b joints_and_tips=21 3"]],
]

# jaxopt residual signature
ResidualFn: TypeAlias = Callable[
    [
        Float[Array, "_"],  # flattened params (51)
        Float[Array, "b 10"],  # betas
        Float[Array, "b 3 4"],  # Pall
        Float[Array, "b n_views 21 2"],  # uv_pred
        "LossWeights",
        bool | Bool[Array, ""],  # unused (kept for interface parity)
    ],
    Float[Array, "_"],  # flat residual vector
]


def make_mv_scaled_residual(side: Literal["left", "right"]) -> tuple[ResidualFn, FwdKinematics]:
    """Factory: residual function + FK for the requested hand side."""
    mano_fwd = jit(ManoSimpleLayerJAX(side=side, mano_root=Path("data/")))

    @jit
    def mv_2d_scaled_residual(
        param_to_optimize: Float[Array, "_"],
        beta: Float[Array, "b 10"],
        Pall: Float[Array, "b 3 4"],
        uv_pred: Float[Array, "b n_views n_kpts=21 2"],
        loss_weights: LossWeights,
        is_left: bool | Bool[Array, ""],  # not used in single-hand path; kept for signature
    ) -> Float[Array, "_"]:
        batch_size: int = uv_pred.shape[0]
        params_2d: Float[Array, "b 51"] = param_to_optimize.reshape(batch_size, 51)
        so3: Float[Array, "b 48"] = params_2d[:, 0:48]
        trans: Float[Array, "b 3"] = params_2d[:, 48:51]

        # MANO forward (mm), convert to meters for projection
        mano_out: tuple[Float[Array, "b n_verts=778 3"], Float[Array, "b n_kpts=21 3"]] = mano_fwd(so3, beta, trans)
        xyz_mano_mm: Float[Array, "b n_kpts=21 3"] = mano_out[1]
        xyz_mano: Float[Array, "b n_kpts=21 3"] = xyz_mano_mm / 1000.0

        xyz_mano_hom: Float[Array, "b n_kpts=21 4"] = npj.concatenate(
            [xyz_mano, npj.ones_like(xyz_mano)[..., 0:1]], axis=-1
        )
        uv_mano: Float[Array, "b n_views n_kpts=21 2"] = proj_3d_vectorized(xyz_hom=xyz_mano_hom, P=Pall)

        res_2d: Float[Array, "b n_views n_kpts=21 2"] = uv_mano - uv_pred
        # sanitize residuals
        lambda_2d: Float[Array, ""] = npj.array(loss_weights["keypoint_2d"], dtype=res_2d.dtype)
        res_2d = npj.nan_to_num(res_2d * lambda_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------------
        # Pose L2 regularization (squared L2 of pose params).
        # For LM, we append sqrt(lambda) * residual so that
        # the objective becomes ||r_data||^2 + lambda * ||theta||^2.
        # We exclude the first 3 dims (global rotation) following MANO convention
        # and regularize only the 45 internal joint parameters, matching the
        # PyTorch reference: poses[..., 3:48].
        # TODO this still needs work, seems like good initialization of pose matters alot to avoid craziness
        # ------------------------------------------------------------------
        pose_only: Float[Array, "b 45"] = so3[:, 3:48]
        lambda_pose: Float[Array, ""] = npj.array(loss_weights["pose_reg"], dtype=pose_only.dtype)
        # If lambda is zero, this stays zero and won't affect residual size
        reg_residual: Float[Array, "b 45"] = npj.sqrt(lambda_pose) * pose_only

        # Concatenate data term and regularization term and return flattened vector
        return npj.concatenate([res_2d.reshape((batch_size, -1)), reg_residual], axis=-1).flatten()

    return mv_2d_scaled_residual, mano_fwd


@dataclass
class PoseOptimConfig:
    """Configuration for pose optimization."""

    beta: Float[ndarray, "10"]
    Pall: Float[ndarray, "n_views 3 4"]
    hand_side: Literal["left", "right"] = "left"
    loss_weights: LossWeights = field(
        default_factory=lambda: LossWeights(keypoint_2d=1.0, depth=0.0, temp=0.0, pose_reg=0.2)
    )
    n_optim_iters: int = 30


@dataclass
class OptimInput:
    uv_pred: Float[ndarray, "b n_views n_coco_kpts=133 2"]
    so3_init: Float[ndarray, "b 48"]
    trans_init: Float[ndarray, "b 3"]


@dataclass
class OptimResult:
    """Results for a single hand."""

    so3_optim: Float[ndarray, "b 48"]
    trans_optim: Float[ndarray, "b 3"]


class SingleHandOptim:
    def __init__(self, *, config: PoseOptimConfig) -> None:
        """
        Single-hand optimizer over MANO pose (axis-angle 48) + translation (3).
        Should avoid chaning the number of views, this causes retraces in jax jit which is bad
        instead use a validity mask
        """
        self.cfg: PoseOptimConfig = config

        self.beta: Float[Array, "1 10"] = npj.array(self.cfg.beta)[npj.newaxis, ...]
        self.Pall: Float[Array, "n_views 3 4"] = npj.array(self.cfg.Pall)
        self.loss_weights: LossWeights = self.cfg.loss_weights

        residual_fn, self.mano_fwd = make_mv_scaled_residual(self.cfg.hand_side)

        self.optimizer = LevenbergMarquardt(
            residual_fun=residual_fn,
            maxiter=self.cfg.n_optim_iters,
            solver="cholesky",
            jit=True,
            xtol=1e-6,
            gtol=1e-6,
        )

        # Trace JIT once
        print("Tracing JIT, can take a while...")
        start_trace_time: float = timer()
        n_views: int = self.Pall.shape[0]
        # important to always have the same number of views while optimizing to avoid retraces.
        # bad views just need to be cleared via a mask (uv_pred - uv_optim) * valid_mask

        so3_init: Float[Array, "1 48"] = npj.zeros((1, 48))
        # Sensible depth prior (meters) to aid convergence
        trans_init: Float[Array, "1 3"] = npj.array([[0.0, 0.0, 0.6]])
        init_params: Float[Array, "1 51"] = npj.concatenate([so3_init, trans_init], axis=-1)
        uv_batch_init: Float[Array, "1 n_views 21 2"] = npj.zeros((1, n_views, 21, 2))
        _ = self.optimizer.run(
            init_params.flatten(),
            beta=self.beta,
            Pall=self.Pall,
            uv_pred=uv_batch_init,
            loss_weights=self.loss_weights,
            is_left=(self.cfg.hand_side == "left"),
        )
        self.optimizer = jit(self.optimizer.run)
        print(f"Trace Done in {timer() - start_trace_time:.2f}s")

    def __call__(
        self,
        optim_input: OptimInput,
    ) -> tuple[OptimResult, LevenbergMarquardtState]:
        uv_pred_batched: Float[Array, "b=1 n_views 133 2"] = npj.array(optim_input.uv_pred)
        # filter to either left or right hand idx
        match self.cfg.hand_side:
            case "left":
                uv_pred_batched: Float[Array, "b=1 n_views 21 2"] = uv_pred_batched[..., LEFT_HAND_IDX, :]
            case "right":
                uv_pred_batched: Float[Array, "b=1 n_views 21 2"] = uv_pred_batched[..., RIGHT_HAND_IDX, :]

        # Try multiple inits to escape poor local minima in single-view scenarios
        so3_init: Float[Array, "b=1 48"] = npj.array(optim_input.so3_init)
        trans_init: Float[Array, "b=1 3"] = npj.array(optim_input.trans_init)

        init_params: Float[Array, "1 51"] = npj.concatenate([so3_init, trans_init], axis=-1)

        optim_tuple: tuple[Float[Array, "_"], LevenbergMarquardtState] = self.optimizer(
            init_params.flatten(),
            beta=self.beta,
            Pall=self.Pall,
            uv_pred=uv_pred_batched,
            loss_weights=self.loss_weights,
            is_left=self.cfg.hand_side == "left",
        )

        optimized_params: Float[Array, "b=1 51"] = optim_tuple[0].reshape(1, 51)
        state: LevenbergMarquardtState = optim_tuple[1]

        so3: Float[Array, "b=1 48"] = optimized_params[:, 0:48]
        trans: Float[Array, "b=1 3"] = optimized_params[:, 48:51]
        # Sanitize any potential NaNs/Infs from the optimizer
        so3 = npj.nan_to_num(so3, nan=0.0, posinf=0.0, neginf=0.0)
        trans = npj.nan_to_num(trans, nan=0.0, posinf=0.0, neginf=0.0)

        results = OptimResult(
            so3_optim=np.array(so3),
            trans_optim=np.array(trans),
        )
        return results, state
