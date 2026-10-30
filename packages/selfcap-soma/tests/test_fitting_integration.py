from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("soma", reason="requires the selfcap-soma environment")
from soma.body import SOMALayer

from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import fit
from selfcap_soma.observations import coco_joint_names

pytestmark = pytest.mark.integration


def test_shared_shape_freeze_keeps_repeated_pose_gradients_finite():
    assets = Path("/tmp/soma-fit/assets")
    if any(not (assets / name).exists() for name in ("SOMA_neutral.npz", "SOMA_template_rig.usda")):
        pytest.skip("requires /tmp/soma-fit/assets/SOMA_neutral.npz and SOMA_template_rig.usda")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA on the host")
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
        neutral = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True)
        names = model.public_joint_names[1:]
        joints = neutral.joints[0].cpu().numpy()
    points = np.full((3, 133, 3), np.nan)
    for key, name in coco_joint_names().items():
        points[:, key] = joints[names.index(name)]
    data = Observations(
        np.array([0.0, 0.05, 0.1]),
        points,
        np.ones((3, 133)),
        points.copy(),
        np.tile(joints[names.index("LeftEye")], (3, 1)).astype(np.float64),
        np.tile(np.diag([-1.0, -1.0, 1.0]), (3, 1, 1)),
        0.0,
    )
    # Eight steps cover neutral-cache reuse, live shape updates, and frozen
    # cache reuse. Each used to expose a second-backward failure.
    result = fit(data, assets, iterations=8, shape_iterations=2)
    assert np.isfinite(result.poses).all()
    assert np.isfinite(result.joints).all()
    assert result.vertices.shape == (3, 4505, 3)
