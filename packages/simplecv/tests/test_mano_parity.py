from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from beartype.door import die_if_unbearable
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float32
from numpy import ndarray
from torch import Tensor

from simplecv.ops.mano import mano_jax, mano_np, mano_torch

# ----------------------
# Small functional parity
# ----------------------


@given(
    quat=arrays(
        dtype=np.float32,
        shape=(1, 4),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=50, deadline=None)
def test_quat2mat_parity(quat: Float32[ndarray, "b=1 4"]) -> None:
    # Validate input from Hypothesis
    die_if_unbearable(quat, Float32[ndarray, "b=1 4"])  # (1,4)

    q_t: Float32[Tensor, "b=1 4"] = torch.from_numpy(quat)
    die_if_unbearable(q_t, Float32[Tensor, "b=1 4"])  # (1,4)

    m_t: Float32[ndarray, "b=1 3 3"] = mano_torch.quat2mat(q_t).detach().cpu().numpy()
    die_if_unbearable(m_t, Float32[ndarray, "b=1 3 3"])  # (1,3,3)

    m_n: Float32[ndarray, "b=1 3 3"] = mano_np.quat2mat(quat)
    die_if_unbearable(m_n, Float32[ndarray, "b=1 3 3"])  # (1,3,3)
    np.testing.assert_allclose(m_n, m_t, rtol=1e-5, atol=1e-5)

    # JAX parity (skip extremely small-norm quats that may be numerically unstable)
    assume(float(np.linalg.norm(quat)) > 1e-12)
    m_j = mano_jax.quat2mat(jnp.asarray(quat))
    np.testing.assert_allclose(np.asarray(m_j), m_t, rtol=1e-5, atol=1e-5)


@given(
    aa=arrays(
        dtype=np.float32,
        shape=(1, 3),
        elements=st.floats(min_value=-3.14, max_value=3.14, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=50, deadline=None)
def test_batch_rodrigues_parity(aa: Float32[ndarray, "b=1 3"]) -> None:
    die_if_unbearable(aa, Float32[ndarray, "b=1 3"])  # (1,3)

    a_t: Float32[Tensor, "b=1 3"] = torch.from_numpy(aa)
    die_if_unbearable(a_t, Float32[Tensor, "b=1 3"])  # (1,3)

    r_t: Float32[ndarray, "b=1 9"] = mano_torch.batch_rodrigues(a_t).detach().cpu().numpy()
    die_if_unbearable(r_t, Float32[ndarray, "b=1 9"])  # (1,9)

    r_n: Float32[ndarray, "b=1 9"] = mano_np.batch_rodrigues(aa)
    die_if_unbearable(r_n, Float32[ndarray, "b=1 9"])  # (1,9)
    np.testing.assert_allclose(r_n, r_t, rtol=1e-5, atol=1e-5)
    r_j = mano_jax.batch_rodrigues(jnp.asarray(aa))
    np.testing.assert_allclose(np.asarray(r_j), r_t, rtol=1e-5, atol=1e-5)


@given(
    pose=arrays(
        dtype=np.float32,
        shape=(1, 48),
        elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=25, deadline=None)
def test_posemap_and_subflatid_parity(pose: Float32[ndarray, "b=1 48"]) -> None:
    die_if_unbearable(pose, Float32[ndarray, "b=1 48"])  # (1,48)

    p_t: Float32[Tensor, "b=1 48"] = torch.from_numpy(pose)
    die_if_unbearable(p_t, Float32[Tensor, "b=1 48"])  # (1,48)

    rm_t: Float32[ndarray, "b=1 144"] = mano_torch.th_posemap_axisang(p_t).detach().cpu().numpy()
    die_if_unbearable(rm_t, Float32[ndarray, "b=1 144"])  # (1,144)
    pm_t: Float32[ndarray, "b=1 144"] = mano_torch.subtract_flat_id(torch.from_numpy(rm_t)).detach().cpu().numpy()
    die_if_unbearable(pm_t, Float32[ndarray, "b=1 144"])  # (1,144)

    rm_n: Float32[ndarray, "b=1 144"] = mano_np.th_posemap_axisang(pose)
    die_if_unbearable(rm_n, Float32[ndarray, "b=1 144"])  # (1,144)
    pm_n: Float32[ndarray, "b=1 144"] = mano_np.subtract_flat_id(rm_n)
    die_if_unbearable(pm_n, Float32[ndarray, "b=1 144"])  # (1,144)

    np.testing.assert_allclose(rm_n, rm_t, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pm_n, pm_t, rtol=1e-5, atol=1e-5)
    rm_j = mano_jax.th_posemap_axisang(jnp.asarray(pose))
    pm_j = mano_jax.subtract_flat_id(rm_j)
    np.testing.assert_allclose(np.asarray(rm_j), rm_t, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(pm_j), pm_t, rtol=1e-5, atol=1e-5)


@given(
    mats=arrays(
        dtype=np.float32,
        shape=(1, 3, 4),
        elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=25, deadline=None)
def test_with_zeros_parity(mats: Float32[ndarray, "b=1 3 4"]) -> None:
    die_if_unbearable(mats, Float32[ndarray, "b=1 3 4"])  # (1,3,4)

    m_t: Float32[ndarray, "b=1 4 4"] = mano_torch.th_with_zeros(torch.from_numpy(mats)).detach().cpu().numpy()
    die_if_unbearable(m_t, Float32[ndarray, "b=1 4 4"])  # (1,4,4)

    m_n: Float32[ndarray, "b=1 4 4"] = mano_np.th_with_zeros(mats)
    die_if_unbearable(m_n, Float32[ndarray, "b=1 4 4"])  # (1,4,4)
    np.testing.assert_allclose(m_n, m_t, rtol=1e-6, atol=1e-6)
    m_j = mano_jax.th_with_zeros(jnp.asarray(mats))
    np.testing.assert_allclose(np.asarray(m_j), m_t, rtol=1e-6, atol=1e-6)


# ----------------------
# Integration on HOCAP
# ----------------------


def _find_hocap_sample() -> tuple[Float32[ndarray, "10"], Float32[ndarray, "n 2 51"]] | None:
    base = Path("data/hocap/sample")
    if not base.exists():
        return None

    # Betas YAMLs are under calibration/mano/subject_*.yaml
    # We'll read subject_8 (present in sample)
    try:
        from serde.yaml import from_yaml

        from simplecv.data.exoego.hocap import CalibratedMano
    except Exception:
        return None

    beta_yaml = base / "calibration" / "mano" / "subject_8.yaml"
    if not beta_yaml.exists():
        return None
    betas: Float32[ndarray, "10"] = from_yaml(CalibratedMano, beta_yaml.read_text()).betas.astype(np.float32)

    # Locate poses_m.npy (sample stores under poses/subject_8/<seq>/)
    poses_candidates = list((base / "poses").glob("subject_8/*/poses_m.npy"))
    if not poses_candidates:
        return None
    poses_m_arr: Float32[ndarray, "2 n 51"] = np.load(poses_candidates[0]).astype(np.float32)  # [2, N, 51]
    poses_m: Float32[ndarray, "n 2 51"] = np.transpose(poses_m_arr, (1, 0, 2))  # [N, 2, 51]
    return betas, poses_m


@pytest.mark.slow
def test_mano_np_matches_torch_on_hocap_sample() -> None:
    res = _find_hocap_sample()
    if res is None:
        pytest.skip("HoCap sample not available; skipping integration test")
    betas, poses_m = res
    die_if_unbearable(betas, Float32[ndarray, "10"])  # (10,)
    die_if_unbearable(poses_m, Float32[ndarray, "n 2 51"])  # (N,2,51)

    mano_root = Path("data")
    assert (mano_root / "MANO_RIGHT.pkl").exists() and (mano_root / "MANO_LEFT.pkl").exists()

    # Compare for both hands on a few frames
    n: int = min(3, poses_m.shape[0])
    for side, idx in [("right", 0), ("left", 1)]:
        # Torch layer
        layer_t = mano_torch.MANOLayerTorch(side=side, betas=betas, mano_root_dir=mano_root)
        poses: Float32[ndarray, "b n_poses=48"] = poses_m[:n, idx, :48]
        trans: Float32[ndarray, "b dim=3"] = poses_m[:n, idx, 48:51]
        die_if_unbearable(poses, Float32[ndarray, "b n_poses=48"])  # (b,48)
        die_if_unbearable(trans, Float32[ndarray, "b dim=3"])  # (b,3)

        vt_t_t, jt_t_t = layer_t(torch.from_numpy(poses), torch.from_numpy(trans))
        vt_t: Float32[ndarray, "b n_verts=778 dim=3"] = vt_t_t.detach().cpu().numpy()
        jt_t: Float32[ndarray, "b n_joints=21 dim=3"] = jt_t_t.detach().cpu().numpy()
        die_if_unbearable(vt_t, Float32[ndarray, "b n_verts=778 dim=3"])  # (b,778,3)
        die_if_unbearable(jt_t, Float32[ndarray, "b n_joints=21 dim=3"])  # (b,21,3)

        # NumPy layer
        layer_n = mano_np.MANOLayerNP(side=side, betas=betas, mano_root_dir=mano_root)
        vt_n, jt_n = layer_n(poses, trans)
        die_if_unbearable(vt_n, Float32[ndarray, "b n_verts=778 dim=3"])  # (b,778,3)
        die_if_unbearable(jt_n, Float32[ndarray, "b n_joints=21 dim=3"])  # (b,21,3)

        np.testing.assert_allclose(vt_n, vt_t, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jt_n, jt_t, rtol=1e-3, atol=1e-3)

        # JAX layer
        try:
            from simplecv.ops.mano.mano_jax import MANOLayerJAX
        except Exception:
            pytest.skip("JAX MANO layer not available; skipping JAX parity")
        layer_j = MANOLayerJAX(side=side, betas=betas, mano_root_dir=mano_root)
        vt_j, jt_j = layer_j(poses, trans)
        die_if_unbearable(vt_j, Float32[ndarray, "b n_verts=778 dim=3"])  # (b,778,3)
        die_if_unbearable(jt_j, Float32[ndarray, "b n_joints=21 dim=3"])  # (b,21,3)
        np.testing.assert_allclose(vt_j, vt_t, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jt_j, jt_t, rtol=1e-3, atol=1e-3)
