"""The numpy MANO layer (dataforge) and the torch one (posekit) pose a hand the same way."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float32
from numpy import ndarray

from simplecv.ops.mano import mano_np, mano_torch

MANO_ROOT: Path = Path(__file__).resolve().parents[1] / "simplecv" / "data"


@given(pose=arrays(dtype=np.float32, shape=(2, 48), elements=st.floats(min_value=-2.0, max_value=2.0, width=32)))
@settings(max_examples=25, deadline=None)
def test_pose_maps_match(pose: Float32[ndarray, "2 48"]) -> None:
    np.testing.assert_allclose(mano_np.th_posemap_axisang(pose), mano_torch.th_posemap_axisang(torch.from_numpy(pose)).numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mano_np.batch_rodrigues(pose[:, :3]), mano_torch.batch_rodrigues(torch.from_numpy(pose[:, :3])).numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("use_pca", [True, False], ids=["pca", "axis-angle"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_layers_pose_the_same_hand(side: Literal["left", "right"], use_pca: bool) -> None:
    if not (MANO_ROOT / f"MANO_{side.upper()}.pkl").is_file():
        pytest.skip(f"MANO_{side.upper()}.pkl required under {MANO_ROOT}")
    rng: np.random.Generator = np.random.default_rng(0)
    betas: Float32[ndarray, "10"] = rng.uniform(-1.5, 1.5, 10).astype(np.float32)
    poses: Float32[ndarray, "3 48"] = rng.uniform(-0.6, 0.6, (3, 48)).astype(np.float32)
    translations: Float32[ndarray, "3 3"] = rng.uniform(-0.5, 0.5, (3, 3)).astype(np.float32)

    verts_np, joints_np = mano_np.MANOLayerNP(side=side, betas=betas, mano_root_dir=MANO_ROOT, use_pca=use_pca)(poses, translations)
    layer_torch: mano_torch.MANOLayerTorch = mano_torch.MANOLayerTorch(side=side, betas=betas, mano_root_dir=MANO_ROOT, use_pca=use_pca)
    verts_torch, joints_torch = layer_torch(torch.from_numpy(poses), torch.from_numpy(translations))

    np.testing.assert_allclose(verts_np, verts_torch.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(joints_np, joints_torch.detach().numpy(), atol=1e-5)
