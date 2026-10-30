from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("soma", reason="requires specialized SOMA environment")
from soma.body import SOMALayer

from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import FitOptions, fit
from selfcap_soma.observations import coco_joint_names


@pytest.mark.integration
def test_ears_only_fit_is_independent_of_quest_head_values() -> None:
    assets = Path("/tmp/soma-fit/assets")
    if not torch.cuda.is_available() or not (assets / "SOMA_neutral.npz").exists():
        pytest.skip("requires CUDA and SOMA assets")
    with torch.no_grad():
        model = SOMALayer(
            data_root=assets,
            device="cuda",
            identity_model_type="soma",
            mode="torch",
            lod="low",
            enable_procedural_transforms=False,
            correctives_model_path=None,
        )
        model.prepare_identity(torch.zeros(1, 128, device="cuda"))
        neutral = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False).joints[0].cpu().numpy()
    points = np.full((3, 133, 3), np.nan)
    for key, name in coco_joint_names().items():
        points[:, key] = neutral[model.public_joint_names[1:].index(name)]
    data = Observations(np.array([0.0, 0.05, 0.1]), points, np.ones((3, 133)), points.copy(), np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)), 0.0)
    points += np.random.default_rng(42).normal(0.0, 0.01, points.shape)
    options = FitOptions(head_mode="ears")
    a = fit(data, assets, 2, 1, options)
    b = fit(replace(data, head_positions=np.full((3, 3), np.nan), head_rotations=np.full((3, 3, 3), np.nan)), assets, 2, 1, options)
    # GPU hand skinning uses atomic reductions; the torso/head remain much more stable.
    np.testing.assert_allclose(a.joints[:, :20], b.joints[:, :20], atol=2e-5)
    np.testing.assert_allclose(a.joints, b.joints, atol=2e-4)
    np.testing.assert_allclose(a.head_offset, 0.0)
