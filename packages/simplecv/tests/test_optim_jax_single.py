from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float
from numpy import ndarray

from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX
from simplecv.ops.mano.mano_jax import ManoSimpleLayerJAX
from simplecv.ops.mano.optim_jax_single import (
    LossWeights,
    OptimInput,
    PoseOptimConfig,
    SingleHandOptim,
    proj_3d_vectorized,
)


def have_mano_pkls() -> bool:
    data_dir = Path("data")
    return (data_dir / "MANO_LEFT.pkl").exists() and (data_dir / "MANO_RIGHT.pkl").exists()


def make_identity_P(n_views: int) -> Float[ndarray, "n_views 3 4"]:
    P = np.zeros((n_views, 3, 4), dtype=np.float32)
    P[:, 0, 0] = 1.0
    P[:, 1, 1] = 1.0
    P[:, 2, 2] = 1.0
    # zero translation -> camera at origin looking down +Z (OpenCV)
    return P


def make_coco_uv_batch(uv_hand: Float[ndarray, "n_views 21 2"]) -> Float[ndarray, "1 n_views 133 2"]:
    uv_coco: Float[ndarray, "1 n_views 133 2"] = np.zeros((1, uv_hand.shape[0], 133, 2), dtype=np.float32)
    uv_coco[:, :, LEFT_HAND_IDX, :] = uv_hand[np.newaxis, ...]
    return uv_coco


def test_proj_3d_vectorized_identity_no_nan():
    # one frame, 3 points, 2 views, simple identity projection
    xyz = np.array(
        [
            [  # [1, 3, 3]
                [0.1, 0.2, 1.0],
                [0.0, 0.0, 2.0],
                [-0.3, 0.5, 4.0],
            ]
        ],
        dtype=np.float32,
    )

    xyz_hom = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)  # [1, 3, 4]
    P = make_identity_P(n_views=2)  # [2, 3, 4]

    uv = np.array(proj_3d_vectorized(jnp.asarray(xyz_hom), jnp.asarray(P)))
    assert uv.shape == (1, 2, 3, 2)
    # u=x/z, v=y/z
    gt = xyz[..., :2] / xyz[..., 2:3]
    assert np.allclose(uv[0, 0], gt, atol=1e-6)
    assert np.all(np.isfinite(uv)), "Projection produced NaN/Inf"


@pytest.mark.skipif(not have_mano_pkls(), reason="MANO PKLs not found under data/")
def test_residual_composition_is_finite():
    # Build a tiny synthetic scene using MANO left hand
    side = "left"
    mano = ManoSimpleLayerJAX(side=side, mano_root=Path("data"))
    beta = jnp.zeros((1, 10), dtype=jnp.float32)
    so3 = jnp.zeros((1, 48), dtype=jnp.float32)  # rest pose
    trans = jnp.array([[0.0, 0.0, 0.6]], dtype=jnp.float32)  # 60 cm in front of cam
    _, joints_mm = mano(so3, beta, trans)
    joints_m = np.array(joints_mm) / 1000.0  # [1,21,3]

    P = make_identity_P(n_views=3)
    uv = np.array(
        proj_3d_vectorized(
            jnp.asarray(np.concatenate([joints_m, np.ones_like(joints_m[..., :1])], axis=-1)), jnp.asarray(P)
        )
    ).astype(np.float32)[0]  # [3,21,2]

    # Single-hand residual will be exercised via the optimizer call below.
    opt = SingleHandOptim(
        config=PoseOptimConfig(
            beta=np.array(beta[0]),
            Pall=P,
            hand_side=side,
            loss_weights=LossWeights(keypoint_2d=1.0, depth=0.0, temp=0.0, pose_reg=0.0),
            n_optim_iters=5,
        )
    )
    # Ensure we don't crash and residual is finite through the run
    optim_input = OptimInput(
        uv_pred=make_coco_uv_batch(uv),
        so3_init=np.zeros((1, 48), dtype=np.float32),
        trans_init=np.array([[0.0, 0.0, 0.6]], dtype=np.float32),
    )
    results, state = opt(optim_input)
    assert np.isfinite(results.so3_optim).all(), "Non-finite so3_optim returned"
    assert np.isfinite(results.trans_optim).all(), "Non-finite trans_optim returned"


@pytest.mark.slow
@pytest.mark.skipif(not have_mano_pkls(), reason="MANO PKLs not found under data/")
def test_single_hand_optim_recovers_close_3d():
    side = "left"
    mano = ManoSimpleLayerJAX(side=side, mano_root=Path("data"))
    beta = jnp.zeros((1, 10), dtype=jnp.float32)
    so3 = jnp.zeros((1, 48), dtype=jnp.float32)  # easy pose
    trans = jnp.array([[0.05, -0.02, 0.7]], dtype=jnp.float32)  # not at origin
    _, joints_mm = mano(so3, beta, trans)
    joints_m = np.array(joints_mm) / 1000.0  # [1,21,3]
    gt_xyz: Float[ndarray, "21 3"] = joints_m[0]

    # 3 simple views (identity intrinsics/extrinsics)
    P = make_identity_P(n_views=3)
    uv = np.array(
        proj_3d_vectorized(
            jnp.asarray(np.concatenate([joints_m, np.ones_like(joints_m[..., :1])], axis=-1)), jnp.asarray(P)
        )
    ).astype(np.float32)[0]  # [3,21,2]

    opt = SingleHandOptim(
        config=PoseOptimConfig(
            beta=np.array(beta[0]),
            Pall=P,
            hand_side=side,
            loss_weights=LossWeights(keypoint_2d=1.0, depth=0.0, temp=0.0, pose_reg=0.0),
            n_optim_iters=20,
        )
    )
    optim_input = OptimInput(
        uv_pred=make_coco_uv_batch(uv),
        so3_init=np.zeros((1, 48), dtype=np.float32),
        trans_init=np.array([[0.0, 0.0, 0.6]], dtype=np.float32),
    )
    results, _ = opt(optim_input)
    assert np.isfinite(results.so3_optim).all(), "Non-finite so3_optim returned"
    assert np.isfinite(results.trans_optim).all(), "Non-finite trans_optim returned"
    _, joints_optim_mm = mano(jnp.asarray(results.so3_optim), beta, jnp.asarray(results.trans_optim))
    joints_optim_m: Float[ndarray, "21 3"] = np.array(joints_optim_mm)[0] / 1000.0

    # metric: mean L2 distance in meters (ignore NaNs in GT if any)
    valid = ~np.isnan(gt_xyz).any(axis=-1)
    if np.any(valid):
        mean_err = float(np.mean(np.linalg.norm(gt_xyz[valid] - joints_optim_m[valid], axis=-1)))
        # Tolerate a few mm to cm level
        assert mean_err < 0.03, f"Mean 3D error too large: {mean_err:.4f} m"
