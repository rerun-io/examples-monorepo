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
    from dataforge.quest_joint_poses import read_quest_joint_poses

    from selfcap_soma.hand_pose import read_orientations

    for name, width in [("left", 3), ("right", 7)]:
        header = ["ts_ns"] + [f"v{i}" for i in range(26 * width)]
        rows = []
        for frame_index in range(20):
            joint = [1.0, 2.0, 3.0] + ([0.0, 0.0, 0.0, 1.0] if width == 7 else [])
            rows.append(",".join(map(str, [frame_index * 10_000_000] + joint * 26)))
        (tmp_path / f"{name}_hand_poses.csv").write_text(",".join(header) + "\n" + "\n".join(rows))
    hands = {name: read_quest_joint_poses(tmp_path / f"{name}_hand_poses.csv") for name in ("left", "right")}
    observed = read_orientations(hands, np.array([0.02, 0.03, 0.04]), 0.0)
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


def test_orientation_targets_name_static_neutral_alignment() -> None:
    from selfcap_soma.hand_pose import HandOrientations, OrientationTargets, orientation_targets

    names = []
    positions = np.zeros((3, 2, 26, 3))
    neutral = []
    for side, label in enumerate(("Left", "Right")):
        names.append(label + "Hand")
        neutral.append([float(side), 0.0, 0.0])
        positions[:, side, 1] = neutral[-1]
        for finger, (digit, base) in enumerate(zip(("Thumb", "Index", "Middle", "Ring", "Pinky"), (2, 6, 11, 16, 21), strict=True)):
            count = 3 if digit == "Thumb" else 4
            for joint in range(count + 1):
                names.append(label + "Hand" + digit + (str(joint + 1) if joint < count else "End"))
                point = [side + 0.03 * (finger + 1), 0.03 * (joint + 1), 0.0]
                neutral.append(point)
                positions[:, side, base + joint] = point
    names.extend(f"unused_{i}" for i in range(77 - len(names)))
    neutral.extend([[0.0, 0.0, 0.0]] * (77 - len(neutral)))
    observed = HandOrientations(positions, np.tile(np.eye(3), (3, 2, 26, 1, 1)), np.ones((3, 2, 26), dtype=bool), ("none", "none"))
    result = orientation_targets(observed, np.arange(3, dtype=np.float64), tuple(names), np.asarray(neutral, dtype=np.float32), np.tile(np.eye(3, dtype=np.float32), (78, 1, 1)))
    assert isinstance(result, OrientationTargets)
    assert result.rotations.shape == (3, 40, 3, 3)
    np.testing.assert_allclose(result.rotations, np.broadcast_to(np.eye(3), result.rotations.shape), atol=1e-6)
    assert result.valid.all()
    np.testing.assert_array_equal(result.spread_deg, np.zeros(40))
    assert names[result.joint_indices[0]] == "LeftHand"
    assert names[result.joint_indices[-1]] == "RightHandPinky4"


def test_a_hand_absent_from_the_base_is_marked_none():
    from selfcap_soma.hand_pose import read_orientations

    observed = read_orientations({}, np.array([0.02, 0.03]), 0.0)
    assert observed.sources == ("none", "none")
    assert not observed.valid.any()
