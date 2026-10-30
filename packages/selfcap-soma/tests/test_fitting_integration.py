from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("soma", reason="requires the selfcap-soma environment")
from soma.body import SOMALayer

from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import Calibration, fit
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
    identity = result.identity.copy()
    identity[0, 126] = 0.1
    calibration = Calibration(identity, result.scale, result.head_offset, result.bone_scales)
    frozen = fit(data, assets, iterations=4, shape_iterations=1, fixed=calibration)
    np.testing.assert_array_equal(frozen.identity, identity)
    np.testing.assert_array_equal(frozen.bone_scales, result.bone_scales)
    np.testing.assert_array_equal(frozen.head_offset, result.head_offset)
    assert frozen.scale == pytest.approx(result.scale, abs=1e-6)


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
    result = fit(data, assets, 8, 2, FitOptions(use_ears=True, learn_head_offset=True, learn_head_rotation=True))
    assert result.head_anchor == "head"
    assert np.isfinite(result.ear_landmarks).all()
    calibration = Calibration(result.identity, result.scale, result.head_offset, result.bone_scales, result.head_attachment, result.head_anchor)
    frozen = fit(data, assets, 4, 1, fixed=calibration)
    np.testing.assert_allclose(frozen.head_attachment, result.head_attachment)
    np.testing.assert_array_equal(frozen.head_offset, result.head_offset)


def test_foot_surface_targets_drive_ankle_rotation():
    from selfcap_soma.fitting import FitOptions
    from selfcap_soma.landmarks import FOOT_VERTICES, foot_vertices

    assets = Path("/tmp/soma-fit/assets")
    if not assets.exists() or not torch.cuda.is_available():
        pytest.skip("requires SOMA assets and CUDA")
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
        names = tuple(model.public_joint_names[1:])
        poses = torch.zeros(3, 77, 3, device="cuda")
        rest = model.pose(poses, apply_correctives=False)
        poses[:, names.index("LeftFoot"), 0] = 0.25
        poses[:, names.index("RightFoot"), 0] = -0.2
        target = model.pose(poses, apply_correctives=False)
        feet = target.vertices[:, torch.as_tensor(FOOT_VERTICES, device="cuda")].mean(2).cpu().numpy()
        skin = model.batched_skinning
        sparse = foot_vertices(
            skin.bind_shapes,
            model.skinning_weights,
            skin.inverse_bind_transform,
            target.transforms[:, torch.argsort(model.public_transform_joint_indices)],
        )
        np.testing.assert_allclose(sparse.cpu().numpy(), feet, atol=2e-6)
        original = rest.vertices[:, torch.as_tensor(FOOT_VERTICES, device="cuda")].mean(2).cpu().numpy()
        points = np.full((3, 133, 3), np.nan)
        for key, name in coco_joint_names().items():
            points[:, key] = target.joints[:, names.index(name)].cpu().numpy()
        points[:, 17:23] = feet
    data = Observations(
        np.array([0.0, 0.05, 0.1]), points, np.ones((3, 133)), np.full_like(points, np.nan), np.full((3, 3), np.nan), np.full((3, 3, 3), np.nan), 0.0
    )
    calibration = Calibration(
        np.zeros((1, 128), np.float32), 1.0, np.zeros(3, np.float32), np.ones((1, 60), np.float32), np.eye(3, dtype=np.float32), "ears"
    )
    result = fit(data, assets, 80, 1, FitOptions(head_mode="ears", use_feet=True, foot_contact_weight=0.0), fixed=calibration)
    assert result.foot_landmarks is not None
    assert np.median(np.linalg.norm(result.foot_landmarks - feet, axis=-1)) < np.median(np.linalg.norm(original - feet, axis=-1)) * 0.6
    assert np.linalg.norm(result.poses[:, [names.index("LeftFoot"), names.index("RightFoot")]]) > 0.05
