import numpy as np
import pytest

from selfcap_soma.hand_pose import frame


def test_bone_frame_is_orthonormal_and_follows_bone():
    basis = frame(np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 3.0]))
    np.testing.assert_allclose(basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) > 0.999


@pytest.mark.integration
def test_procedural_twists_preserve_public_joint_centres():
    from pathlib import Path

    import pytest
    import torch
    from soma.body import SOMALayer

    if not torch.cuda.is_available() or not Path("/tmp/soma-fit/assets/SOMA_neutral.npz").exists():
        pytest.skip("requires CUDA and SOMA assets")
    values = []
    for enabled in (False, True):
        model = SOMALayer(
            data_root=Path("/tmp/soma-fit/assets"),
            device="cuda",
            identity_model_type="soma",
            mode="torch",
            lod="low",
            enable_procedural_transforms=enabled,
            correctives_model_path=None,
        )
        with torch.no_grad():
            model.prepare_identity(torch.zeros(1, 128, device="cuda"))
            pose = torch.zeros(1, 77, 3, device="cuda")
            pose[0, model.public_joint_names[1:].index("LeftHand"), 0] = 0.6
            values.append(model.pose(pose, apply_correctives=False).joints.cpu().numpy())
    np.testing.assert_allclose(values[0], values[1], atol=2e-6)


def test_finger_flexion_is_not_misreported_as_roll_at_euler_singularity():
    from scipy.spatial.transform import Rotation

    from selfcap_soma.hand_diagnostics import swing_twist_angles

    rotations = Rotation.from_rotvec(np.deg2rad([[0.0, 90.0, 0.0], [90.0, 0.0, 0.0], [0.0, 180.0, 0.0]])).as_matrix()
    angles = swing_twist_angles(rotations)
    np.testing.assert_allclose(angles[:, 0], [0.0, 90.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(abs(angles[:, 1]), [90.0, 0.0, 180.0], atol=1e-8)


def test_xyz_only_hand_reader_records_missing_orientation_per_hand(tmp_path):
    from selfcap_soma.hand_pose import read_orientations

    for name, width in [("left", 3), ("right", 7)]:
        header = ["ts_ns"] + [f"v{i}" for i in range(26 * width)]
        rows = []
        for frame_index in range(20):
            joint = [1.0, 2.0, 3.0] + ([0.0, 0.0, 0.0, 1.0] if width == 7 else [])
            rows.append(",".join(map(str, [frame_index * 10_000_000] + joint * 26)))
        (tmp_path / f"{name}_hand_poses.csv").write_text(",".join(header) + "\n" + "\n".join(rows))
    observed = read_orientations(tmp_path, np.array([0.02, 0.03, 0.04]), 0.0)
    assert observed.sources == ("none", "quest")
    assert np.isfinite(observed.positions).all()


def test_hand_limit_prior_penalizes_roll_and_excessive_wrist_twist():
    import torch

    from selfcap_soma.hand_pose import hand_limit_penalty

    neutral = torch.zeros((1, 2, 3))
    bases = torch.eye(3).repeat(2, 1, 1)
    limits = torch.tensor([[0.35, 2.0, 0.5, 0.35], [1.2, 1.2, 0.6, 0.8]])
    bent = neutral.clone()
    bent[0, :, 0] = 1.8
    bent.requires_grad_()
    loss = hand_limit_penalty(bent, bases, limits, torch.ones(2))
    assert loss > hand_limit_penalty(neutral, bases, limits, torch.ones(2)) + 1.0
    loss.backward()
    assert torch.isfinite(bent.grad).all()
