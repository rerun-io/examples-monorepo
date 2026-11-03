from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("soma", reason="requires the selfcap-soma environment")
from soma.body import SOMALayer

from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import Calibration, calibrate_shape, fit_pose
from selfcap_soma.landmarks import EAR_VERTICES
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
    result = calibrate_shape(data, assets, iterations=8, shape_iterations=2)
    assert np.isfinite(result.poses).all()
    assert np.isfinite(result.joints).all()
    assert result.vertices.shape == (3, 4505, 3)
    np.testing.assert_array_equal(result.ear_landmarks, result.vertices[:, EAR_VERTICES].reshape(-1, 2, 3, 3).mean(axis=2))
    identity = result.identity.copy()
    identity[0, 126] = 0.1
    calibration = Calibration(identity, result.scale, result.head_offset, result.bone_scales, result.head_attachment, result.head_anchor)
    frozen = fit_pose(data, calibration, assets, iterations=4)
    np.testing.assert_array_equal(frozen.identity, identity)
    np.testing.assert_array_equal(frozen.bone_scales, result.bone_scales)
    np.testing.assert_array_equal(frozen.head_offset, result.head_offset)
    assert frozen.scale == pytest.approx(result.scale, abs=1e-6)

    from dataclasses import replace

    from selfcap_soma.fitting import FitOptions
    from selfcap_soma.upper_body import lower_body_mask

    leg_ids = np.flatnonzero(lower_body_mask(tuple(model.public_joint_names), model.public_joint_parent_ids.cpu().numpy().astype(np.int64))[1:])
    bent = result.poses.copy()
    bent[:, leg_ids] = 0.4
    upper = fit_pose(
        data,
        calibration,
        assets,
        iterations=4,
        options=FitOptions(freeze_legs=True, leg_support_weight=0.0, hip_landmark_weight=0.15, display_twists=True, body_acceleration_weight=1.0, final_learning_rate=0.001),
        initial=replace(result, poses=bent),
    )
    np.testing.assert_array_equal(upper.poses[:, leg_ids], 0.0)


def test_sparse_ear_vertices_match_full_mesh_and_ear_calibration_freezes():
    from selfcap_soma.fitting import FitOptions
    from selfcap_soma.landmarks import EAR_VERTICES, ear_vertices

    assets = Path("/tmp/soma-fit/assets")
    if not (assets / "SOMA_neutral.npz").exists() or not torch.cuda.is_available():
        pytest.skip("requires SOMA assets in /tmp/soma-fit/assets and CUDA")
    model = SOMALayer(
        data_root=assets,
        device="cuda",
        identity_model_type="soma",
        mode="torch",
        lod="low",
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    with torch.no_grad():
        model.prepare_identity(torch.zeros(1, 128, device="cuda"))
        pose = torch.zeros(3, 77, 3, device="cuda")
        pose[:, 0, 1] = torch.tensor([0.0, 0.2, -0.3], device="cuda")
        output = model.pose(pose, apply_correctives=False)
        skin = model.batched_skinning
        sparse = ear_vertices(
            skin.bind_shapes,
            model.skinning_weights,
            skin.inverse_bind_transform,
            output.transforms[:, torch.argsort(model.public_transform_joint_indices)],
        )
        actual = output.vertices[:, EAR_VERTICES].reshape(3, 2, 3, 3).mean(2)
        torch.testing.assert_close(sparse, actual, atol=2e-6, rtol=2e-5)
        names = model.public_joint_names[1:]
        points = np.full((3, 133, 3), np.nan)
        for key, name in coco_joint_names().items():
            points[:, key] = output.joints[:, names.index(name)].cpu().numpy()
        points[:, 3:5] = actual.cpu().numpy()
        head = output.joints[:, names.index("Head")].cpu().numpy().astype(np.float64)
    data = Observations(
        np.array([0.0, 0.05, 0.1]), points, np.ones((3, 133)), points.copy(), head, np.tile(np.diag([-1.0, -1.0, 1.0]), (3, 1, 1)), 0.0
    )
    result = calibrate_shape(data, assets, 8, 2, FitOptions(learn_head_rotation=True))
    assert result.head_anchor == "ears-rotation"
    assert np.isfinite(result.ear_landmarks).all()
    calibration = Calibration(result.identity, result.scale, result.head_offset, result.bone_scales, result.head_attachment, result.head_anchor)
    frozen = fit_pose(data, calibration, assets, 4)
    np.testing.assert_allclose(frozen.head_attachment, result.head_attachment)
    np.testing.assert_array_equal(frozen.head_offset, result.head_offset)
