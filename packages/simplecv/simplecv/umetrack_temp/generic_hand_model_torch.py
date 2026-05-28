from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy

NUM_LANDMARKS_PER_HAND = 21
NUM_FINGERTIPS_PER_HAND = 5
NUM_JOINTS_PER_HAND = 22
LEFT_HAND_INDEX = 0
RIGHT_HAND_INDEX = 1

NUM_DIGITS: int = 5
NUM_JOINT_FRAMES: int = 1 + 1 + 3 * 5  # root + wrist + finger frames * 5
DOF_PER_FINGER: int = 4
MAX_LANDMARK_WEIGHTS: int = 3
JOINT_LIMIT_BOUNDS: int = 2


class LANDMARK(IntEnum):
    THUMB_FINGERTIP = 0
    INDEX_FINGER_FINGERTIP = 1
    MIDDLE_FINGER_FINGERTIP = 2
    RING_FINGER_FINGERTIP = 3
    PINKY_FINGER_FINGERTIP = 4
    WRIST_JOINT = 5
    THUMB_INTERMEDIATE_FRAME = 6
    THUMB_DISTAL_FRAME = 7
    INDEX_PROXIMAL_FRAME = 8
    INDEX_INTERMEDIATE_FRAME = 9
    INDEX_DISTAL_FRAME = 10
    MIDDLE_PROXIMAL_FRAME = 11
    MIDDLE_INTERMEDIATE_FRAME = 12
    MIDDLE_DISTAL_FRAME = 13
    RING_PROXIMAL_FRAME = 14
    RING_INTERMEDIATE_FRAME = 15
    RING_DISTAL_FRAME = 16
    PINKY_PROXIMAL_FRAME = 17
    PINKY_INTERMEDIATE_FRAME = 18
    PINKY_DISTAL_FRAME = 19
    PALM_CENTER = 20


UME_HAND_CONNECTIONS = frozenset(
    [
        (5, 6),
        (6, 7),
        (7, 0),
        (5, 8),
        (8, 9),
        (9, 10),
        (10, 1),
        (5, 11),
        (11, 12),
        (12, 13),
        (13, 2),
        (5, 14),
        (14, 15),
        (15, 16),
        (16, 3),
        (5, 17),
        (17, 18),
        (18, 19),
        (19, 4),
    ]
)


@dataclass
class HandModelTorch:
    joint_rotation_axes: Float32[Tensor, "n_joints=22 3"]
    joint_rest_positions: Float32[Tensor, "n_joints=22 3"]
    joint_frame_index: Int64[Tensor, "n_joints=22"]
    joint_parent: Int64[Tensor, "n_joints=22"]
    joint_first_child: Int64[Tensor, "n_joints=22"]
    joint_next_sibling: Int64[Tensor, "n_joints=22"]
    landmark_rest_positions: Float32[Tensor, "num_landmarks 3"]
    landmark_rest_bone_weights: Float32[Tensor, "num_landmarks max_landmark_weights"]
    landmark_rest_bone_indices: Int64[Tensor, "num_landmarks max_landmark_weights"]
    hand_scale: Float32[Tensor, ""]
    mesh_vertices: Float32[Tensor, "num_mesh_vertices 3"]
    mesh_triangles: Int64[Tensor, "num_mesh_faces 3"]
    dense_bone_weights: Float32[Tensor, "num_mesh_vertices num_joint_frames"]
    joint_limits: Float32[Tensor, "n_joints=22 joint_limit_bounds"]


def hand_model_numpy_to_tensor(hand_model: HandModelNumpy) -> HandModelTorch:
    """Materialise the serde-loaded hand model using NumPy arrays only."""

    return HandModelTorch(**asdict(hand_model))


class SingleHandPose(NamedTuple):
    """
    A hand pose is composed of two fields:
    1) joint angles where # joints == # DoFs
    2) root-to-world rigid wrist transformation
    """

    joint_angles: Float32[np.ndarray, "n_joints=22"] = np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32)
    wrist_xform: Float32[np.ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    hand_confidence: float = 1.0


@dataclass
class HandPoseLabels:
    """
    Dataclass for hand pose labels for a single sequence.
    """

    camera_angles: list[float]
    """List of camera angles in degrees."""
    camera_to_world_transforms: Float32[np.ndarray, "n_frames n_cams 4 4"]
    """Camera to world transform matrix."""
    hand_model: HandModelTorch
    """Hand model."""
    joint_angles: Float32[np.ndarray, "n_frames n_hands=2 n_joints=22"]
    """Joint angles in degrees."""
    wrist_transforms: Float32[np.ndarray, "n_frames n_hands=2 4 4"]
    """Wrist transform matrix."""
    hand_confidences: Float32[np.ndarray, "n_frames n_hands=2"]
    """Hand confidence."""

    def __len__(self):
        return len(self.joint_angles)


def so3_exp_map(log_rot: Tensor, eps: float = 0.0001) -> Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]


def _so3_exp_map(log_rot: Tensor, eps: float = 0.0001) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def hat(v: Tensor) -> Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def _finger_fk(
    joint_local_xfs: Float32[Tensor, "... dof_per_finger 4 4"],
    parent_transform: Float32[Tensor, "... 4 4"],
) -> list[Float32[Tensor, "... 4 4"]]:
    """
    Computes the forward kinematics for a finger with 4 degrees of freedom (DoF),
    i.e., 4 joints, and returns 3 transformation frames.

    Args:
        joint_local_xfs (Float32[Tensor, "... dof_per_finger 4 4"]): Local joint transformations.
        parent_transform (Float32[Tensor, "... 4 4"]): Parent transformation matrix.

    Returns:
        list[Float32[Tensor, "... 4 4"]]: List of computed transformation matrices.
    """
    transform_mats = [parent_transform]
    for i in range(4):
        transform_mats.append(torch.matmul(transform_mats[-1], joint_local_xfs[:, i]))
    return transform_mats[2:]


def _joint_local_transform(
    rotation_axis: Float32[Tensor, "... 20 3"],
    rest_pose: Float32[Tensor, "... 20 3"],
    joint_angles: Float32[Tensor, "... 20"],
) -> Float32[Tensor, "... 20 4 4"]:
    """
    Computes the local transformation matrix for joints given their rotation axes,
    rest poses, and joint angles.

    Args:
        rotation_axis (Float32[Tensor, "... 20 3"]): Rotation axes of the joints.
        rest_pose (Float32[Tensor, "... 20 3"]): Rest poses of the joints.
        joint_angles (Float32[Tensor, "... 20"]): Joint angles.

    Returns:
        Float32[Tensor, "... 20 4 4"]: Computed local transformation matrix.
    """
    rotation_axis_flat = rotation_axis.reshape(-1, 3)
    rest_pose_flat = rest_pose.reshape(-1, 3)
    joint_angles_flat = joint_angles.reshape(-1)

    angle_axis = rotation_axis_flat * joint_angles_flat.unsqueeze(-1)
    local_transform = torch.eye(4, dtype=angle_axis.dtype, device=angle_axis.device)
    local_transform = local_transform.unsqueeze(dim=0).repeat(angle_axis.shape[0], 1, 1)

    rot_mat = so3_exp_map(angle_axis)
    translation = rest_pose_flat - torch.matmul(rot_mat, rest_pose_flat.unsqueeze(dim=-1)).squeeze(dim=-1)
    local_transform[:, :3, :3] = rot_mat
    local_transform[:, 0:3, 3] = torch.squeeze(translation, dim=-1)

    return local_transform.reshape(*rotation_axis.shape[0:-1], 4, 4)


def _lbs(
    trans_mats: Float32[Tensor, "... num_joint_frames 4 4"],
    skinned_points: Float32[Tensor, "... num_landmarks num_joint_frames 4"],
) -> Float32[Tensor, "... num_landmarks 4"]:
    """
    Performs linear blend skinning (LBS) on the given points using the given transformation matrices.

    Args:
        trans_mats (Float32[Tensor, "... num_joint_frames 4 4"]): Transformation matrices.
        skinned_points (Float32[Tensor, "... num_landmarks num_joint_frames 4"]): Skinned points to be transformed.

    Returns:
        Float32[Tensor, "... num_landmarks 4"]: Transformed points.
    """
    trans_mats = trans_mats.unsqueeze(dim=1)
    skinned_points = skinned_points.unsqueeze(dim=-1)
    fk_points = torch.matmul(trans_mats, skinned_points).sum(dim=2).squeeze(dim=-1)

    return fk_points


def _get_skinning_weights(
    bone_indices: Int64[Tensor, "... num_landmarks max_landmark_weights"],
    bone_weights: Float32[Tensor, "... num_landmarks max_landmark_weights"],
    n_frames: int,
) -> Float32[Tensor, "... num_landmarks num_joint_frames"]:
    """
    Computes skinning weights for the vertices given the bone indices, bone weights,
    and number of transformation frames.

    Args:
        bone_indices (Int64[Tensor, "... num_landmarks max_landmark_weights"]): Indices of bones influencing each vertex.
        bone_weights (Float32[Tensor, "... num_landmarks max_landmark_weights"]): Weights of bones for each vertex.
        n_frames (int): Number of transformation frames.

    Returns:
        Float32[Tensor, "... num_landmarks num_joint_frames"]: Computed skinning weights for each vertex.
    """
    bs = bone_indices.shape[0]
    n_lms = bone_indices.shape[1]
    # Offset all the bones linearly from 0 to (bs*n_lms*n_frames) so that we can directly
    # index into the flattened weight matrix and set the corresponding skinning weights
    flat_idx_offset = torch.arange(0, bs * n_lms, device=bone_indices.device) * n_frames
    bone_flat_idx = bone_indices.long() + flat_idx_offset.reshape(bs, n_lms, 1)
    skin_mat = torch.zeros(bs * n_lms * n_frames, device=bone_weights.device, dtype=bone_weights.dtype)
    non0_w_mask = bone_weights != 0
    non0_indices = bone_flat_idx[non0_w_mask]
    skin_mat[non0_indices] = bone_weights[non0_w_mask]
    skin_mat = skin_mat.reshape(bs, n_lms, n_frames)

    return skin_mat


def _hand_skinning_transform(
    rotation_axis: Float32[Tensor, "... n_joints=22 3"],
    rest_poses: Float32[Tensor, "... n_joints=22 3"],
    joint_angles: Float32[Tensor, "... n_joints=22"],
    wrist_transforms: Float32[Tensor, "... 4 4"],
) -> Float32[Tensor, "... num_joint_frames 4 4"]:
    """
    Computes skinning transformation matrices for a hand model given rotation axes,
    rest poses, joint angles, and wrist transformations.

    Args:
        rotation_axis (Float32[Tensor, "... n_joints=22 3"]): Rotation axes of the joints.
        rest_poses (Float32[Tensor, "... n_joints=22 3"]): Rest poses of the joints.
        joint_angles (Float32[Tensor, "... n_joints=22"]): Joint angles.
        wrist_transforms (Float32[Tensor, "... 4 4"]): Wrist transformations.

    Returns:
        Float32[Tensor, "... num_joint_frames 4 4"]: Computed skinning transformation matrices.
    """
    transform_mats = [wrist_transforms] * 2  # [root_transform, wrist_transform]
    d = DOF_PER_FINGER

    joint_local_xfs = _joint_local_transform(rotation_axis[:, 0:20], rest_poses[:, 0:20], joint_angles[:, 0:20])

    for finger_idx in range(NUM_DIGITS):
        transform_mats += _finger_fk(joint_local_xfs[:, d * finger_idx : d * finger_idx + d], wrist_transforms)
    transform_mats = torch.cat([m.unsqueeze(1) for m in transform_mats], dim=1)
    return transform_mats


def _get_skinned_vertices(
    vertices: Float32[Tensor, "... num_landmarks 3"] | Float32[Tensor, "... num_landmarks 4"],
    weights: Float32[Tensor, "... num_landmarks num_joint_frames"],
) -> Float32[Tensor, "... num_landmarks num_joint_frames 4"]:
    """
    Computes skinned vertices given the original vertices and their corresponding skinning weights.

    Args:
        vertices (Float32[Tensor, "... num_landmarks 3"] | Float32[Tensor, "... num_landmarks 4"]): Original vertices.
        weights (Float32[Tensor, "... num_landmarks num_joint_frames"]): Skinning weights for each vertex.

    Returns:
        Float32[Tensor, "... num_landmarks num_joint_frames 4"]: Skinned vertices.
    """
    if vertices.shape[2] == 3:
        n_vertices = vertices.shape[1]
        homo = torch.ones(
            vertices.shape[0],
            n_vertices,
            1,
            dtype=vertices.dtype,
            device=vertices.device,
        )
        vertices = torch.cat([vertices, homo], dim=-1)

    vertices = vertices.unsqueeze(dim=2)
    weights = weights.unsqueeze(dim=-1)
    return vertices * weights


def _skin_points(
    joint_rest_positions: Float32[Tensor, "... n_joints=22 3"],
    joint_rotation_axes: Float32[Tensor, "... n_joints=22 3"],
    skin_mat: Float32[Tensor, "... num_landmarks num_joint_frames"],
    joint_angles: Float32[Tensor, "... n_joints=22"],
    points: Float32[Tensor, "... num_landmarks 3"],
    wrist_transforms: Float32[Tensor, "... 4 4"],
) -> Float32[Tensor, "... num_landmarks 3"]:
    """
    Computes skin points for the given joint and wrist transforms.

    Args:
        joint_rest_positions (Float32[Tensor, "... n_joints=22 3"]): The rest positions of the joints.
        joint_rotation_axes (Float32[Tensor, "... n_joints=22 3"]): The rotation axes of the joints.
        skin_mat (Float32[Tensor, "... num_landmarks num_joint_frames"]): Skin matrix.
        joint_angles (Float32[Tensor, "... n_joints=22"]): The angles of the joints.
        points (Float32[Tensor, "... num_landmarks 3"]): Points to be skinned.
        wrist_transforms (Float32[Tensor, "... 4 4"]): Wrist transformations.

    Returns:
        Float32[Tensor, "... num_landmarks 3"]: The skinned vectors for the skin points.
    """
    leading_dims = joint_angles.shape[:-1]
    assert joint_rest_positions.shape[:-2] == leading_dims, (
        "Leading dimensions do not match, " + f"got {leading_dims} and {joint_rest_positions.shape[:-2]}"
    )

    # This allows querying the product of leading dimensions without making the
    # model specialized to a particular shape
    numel = torch.flatten(joint_angles, end_dim=-2).shape[0] if len(leading_dims) else 1

    batched_joint_rest_positions = joint_rest_positions.reshape(numel, -1, 3)

    skin_xfs = _hand_skinning_transform(
        rotation_axis=joint_rotation_axes.reshape(numel, -1, 3),
        rest_poses=batched_joint_rest_positions,
        joint_angles=joint_angles.reshape(numel, -1),
        wrist_transforms=wrist_transforms.reshape(numel, 4, 4),
    )

    verts = _get_skinned_vertices(points.reshape(numel, -1, 3), skin_mat)
    skinned_vecs = _lbs(skin_xfs, verts)[..., :3]
    skinned_vecs = skinned_vecs.reshape(list(leading_dims) + list(skinned_vecs.shape[-2:]))

    return skinned_vecs


def skin_landmarks(
    hand_model: HandModelTorch,
    joint_angles: Float32[Tensor, "... n_joints=22"],
    wrist_transforms: Float32[Tensor, "... 4 4"],
) -> Float32[Tensor, "... num_landmarks 3"]:
    """
    Computes the skin landmarks for a given hand model, joint angles, and wrist transforms.

    Args:
        hand_model (HandModel): A model representing a hand.
        joint_angles (Float32[Tensor, "... n_joints=22"]): The angles of the joints.
        wrist_transforms (Float32[Tensor, "... 4 4"]): Wrist transformations.

    Returns:
        Float32[Tensor, "... num_landmarks 3"]: The skinned landmarks.
    """
    leading_dims = joint_angles.shape[:-1]
    numel = torch.flatten(joint_angles, end_dim=-2).shape[0] if len(leading_dims) else 1
    max_weights = hand_model.landmark_rest_bone_indices.shape[-1]
    skin_mat = _get_skinning_weights(
        hand_model.landmark_rest_bone_indices.reshape(numel, -1, max_weights),
        hand_model.landmark_rest_bone_weights.reshape(numel, -1, max_weights),
        NUM_JOINT_FRAMES,
    )
    return _skin_points(
        hand_model.joint_rest_positions,
        hand_model.joint_rotation_axes,
        skin_mat,
        joint_angles,
        hand_model.landmark_rest_positions,
        wrist_transforms,
    )


def skin_landmarks_np(
    hand_model: HandModelTorch,
    joint_angles: Float32[np.ndarray, "... n_joints=22"],
    wrist_transforms: Float32[np.ndarray, "... 4 4"],
) -> Float32[np.ndarray, "... num_landmarks 3"]:
    """
    Computes the skin landmarks for a given hand model, joint angles, and wrist transforms.

    This is a numpy version of the skin_landmarks function.

    Args:
        hand_model (HandModel): A model representing a hand.
        joint_angles (Float32[np.ndarray, "... n_joints=22"]): The angles of the joints.
        wrist_transforms (Float32[np.ndarray, "... 4 4"]): Wrist transformations.

    Returns:
        Float32[np.ndarray, "... num_landmarks 3"]: The skinned landmarks.
    """
    landmarks = skin_landmarks(
        hand_model,
        torch.from_numpy(joint_angles).float(),
        torch.from_numpy(wrist_transforms).float(),
    )
    return landmarks.numpy()


def landmarks_from_hand_pose(
    hand_model: HandModelTorch, hand_pose: SingleHandPose, hand_idx: int
) -> Float32[np.ndarray, "num_landmarks 3"]:
    """
    Compute 3D landmarks in the world space given the hand model and hand pose.

    Args:
        hand_model (HandModelTensor): A model representing a hand.
        hand_pose (SingleHandPose): A pose representing the hand's position and orientation.
        hand_idx (int): Index of the hand. Equals 0 if left hand, 1 if right hand.

    Returns:
        Float32[np.ndarray, "num_landmarks 3"]: The 3D landmarks in the world space.
    """
    xf: Float32[np.ndarray, "4 4"] = hand_pose.wrist_xform.copy()
    # This function expects the user hand model to be a left hand.
    if hand_idx == RIGHT_HAND_INDEX:
        xf[:, 0] *= -1
    landmarks: Float32[np.ndarray, "... num_landmarks 3"] = skin_landmarks_np(hand_model, hand_pose.joint_angles, xf)
    return landmarks
