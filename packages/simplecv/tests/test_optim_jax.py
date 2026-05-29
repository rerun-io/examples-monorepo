from pathlib import Path

import numpy as np
import pytest
from beartype.door import die_if_unbearable
from jaxtyping import Float32
from numpy import ndarray

from simplecv.ops.mano import mano_np, mano_torch
from simplecv.ops.mano.mano_jax import MANOLayerJAX

ROOT = Path("/mnt/8tb/data/hocap/datasets")
SUBJECT = "8"
SEQUENCES = [
    "20231024_180111",
    "20231024_180651",
    "20231024_180733",
    "20231024_181413",
]

"""
Beartype + pytest-beartype enforce function annotations (args/returns) at runtime.
For locals, prefer checking at function boundaries. Avoid intentionally wrong
local annotations; rely on return annotations for runtime validation.
"""


def _load_betas() -> Float32[ndarray, "10"] | None:
    try:
        from serde.yaml import from_yaml

        from simplecv.data.exoego.hocap import CalibratedMano
    except Exception:
        return None
    yaml_path = ROOT / "calibration" / "mano" / f"subject_{SUBJECT}.yaml"
    if not yaml_path.exists():
        return None
    betas: Float32[ndarray, "10"] = from_yaml(CalibratedMano, yaml_path.read_text()).betas.astype(np.float32)
    die_if_unbearable(betas, Float32[ndarray, "10"])  # (10,)
    return betas


def _load_poses(seq: str) -> Float32[ndarray, "n_frames 2 51"] | None:
    # primary location (full dataset)
    p = ROOT / f"subject_{SUBJECT}" / seq / "poses_m.npy"
    if p.exists():
        poses_raw: Float32[ndarray, "n_hands=2 n_frames 51"] = np.load(p).astype(np.float32)
        die_if_unbearable(poses_raw, Float32[ndarray, "n_hands=2 n_frames 51"])  # (2,N,51)
        poses: Float32[ndarray, "n_frames n_hands=2 51"] = np.transpose(poses_raw, (1, 0, 2))
        die_if_unbearable(poses, Float32[ndarray, "n_frames n_hands=2 51"])  # (N,2,51)
        return poses
    # fallback like the sample layout
    p2 = ROOT / "poses" / f"subject_{SUBJECT}" / seq / "poses_m.npy"
    if p2.exists():
        poses_raw: Float32[ndarray, "n_hands=2 n_frames 51"] = np.load(p2).astype(np.float32)
        die_if_unbearable(poses_raw, Float32[ndarray, "n_hands=2 n_frames 51"])  # (2,N,51)
        poses: Float32[ndarray, "n_frames n_hands=2 51"] = np.transpose(poses_raw, (1, 0, 2))
        die_if_unbearable(poses, Float32[ndarray, "n_frames n_hands=2 51"])  # (N,2,51)
        return poses
    return None


def _have_mano_pkls(mano_root: Path) -> bool:
    return (mano_root / "MANO_RIGHT.pkl").exists() and (mano_root / "MANO_LEFT.pkl").exists()


@pytest.mark.slow
@pytest.mark.parametrize("sequence", SEQUENCES)
def test_mano_np_matches_torch_on_full_hocap(sequence: str) -> None:
    torch = pytest.importorskip("torch")

    if not ROOT.exists():
        pytest.skip(f"Hocap root {ROOT} not present")

    betas = _load_betas()
    if betas is None:
        pytest.skip("Could not load subject betas; skipping")

    poses_m = _load_poses(sequence)
    if poses_m is None:
        pytest.skip(f"Could not load poses for sequence {sequence}")

    die_if_unbearable(betas, Float32[ndarray, "10"])  # (10,)
    die_if_unbearable(poses_m, Float32[ndarray, "n_frames 2 51"])  # (N,2,51)

    mano_root = Path("data")  # use local MANO pkl files
    if not _have_mano_pkls(mano_root):
        pytest.skip("MANO PKLs not found under data/")

    n: int = min(3, poses_m.shape[0])
    for side, idx in [("right", 0), ("left", 1)]:
        # Torch
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

        # NumPy
        layer_n = mano_np.MANOLayerNP(side=side, betas=betas, mano_root_dir=mano_root)
        vt_n, jt_n = layer_n(poses, trans)
        die_if_unbearable(vt_n, Float32[ndarray, "b n_verts=778 dim=3"])  # (b,778,3)
        die_if_unbearable(jt_n, Float32[ndarray, "b n_joints=21 dim=3"])  # (b,21,3)

        np.testing.assert_allclose(vt_n, vt_t, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jt_n, jt_t, rtol=1e-3, atol=1e-3)

        # JAX
        layer_j = MANOLayerJAX(side=side, betas=betas, mano_root_dir=mano_root)
        vt_j, jt_j = layer_j(poses, trans)
        die_if_unbearable(vt_j, Float32[ndarray, "b n_verts=778 dim=3"])  # (b,778,3)
        die_if_unbearable(jt_j, Float32[ndarray, "b n_joints=21 dim=3"])  # (b,21,3)
        np.testing.assert_allclose(vt_j, vt_t, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jt_j, jt_t, rtol=1e-3, atol=1e-3)
