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
