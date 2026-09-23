"""NumPy implementation of the generic hand model utilities.

This module mirrors the torch-based variant but operates entirely on NumPy
arrays so that downstream consumers can avoid a PyTorch dependency when GPU
autograd is unnecessary.
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from serde import serde

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


@serde
class HandModelNumpy:
    """Hand model parameters stored as NumPy arrays.

    Notes:
        Serde loads each field as a NumPy ndarray with the dtype/shape indicated
        by the jaxtyping annotations below.
    """

    joint_rotation_axes: Float32[ndarray, "n_joints=22 3"]
    """Unit rotation axes for each joint frame."""
    joint_rest_positions: Float32[ndarray, "n_joints=22 3"]
    """Joint rest positions expressed in the hand root frame."""
    joint_frame_index: Int64[ndarray, "n_joints=22"]
    """Mapping from joint to the frame index used during skinning."""
    joint_parent: Int64[ndarray, "n_joints=22"]
    """Parent joint indices (negative values indicate the root)."""
    joint_first_child: Int64[ndarray, "n_joints=22"]
    """Index to the first child joint for hierarchical traversal."""
    joint_next_sibling: Int64[ndarray, "n_joints=22"]
    """Index to the next sibling joint for hierarchical traversal."""
    landmark_rest_positions: Float32[ndarray, "num_landmarks 3"]
    """Rest pose landmark coordinates in the hand model frame."""
    landmark_rest_bone_weights: Float32[ndarray, "num_landmarks max_landmark_weights"]
    """Bone blend weights per landmark."""
    landmark_rest_bone_indices: Int64[ndarray, "num_landmarks max_landmark_weights"]
    """Bone indices paired with `landmark_rest_bone_weights`."""
    hand_scale: Float32[ndarray, ""]
    """Global uniform hand scale factor."""
    mesh_vertices: Float32[ndarray, "num_mesh_vertices 3"]
    """Skinned mesh vertices at rest pose."""
    mesh_triangles: Int64[ndarray, "num_mesh_faces 3"]
    """Triangle indices defining the mesh topology."""
    dense_bone_weights: Float32[ndarray, "num_mesh_vertices num_joint_frames"]
    """Blend weights used for dense mesh skinning."""
    joint_limits: Float32[ndarray, "n_joints=22 joint_limit_bounds"]
    """Lower/upper joint angle limits in radians."""


class SingleHandPose(NamedTuple):
    """
    A hand pose is composed of two fields:
    1) joint angles where # joints == # DoFs
    2) root-to-world rigid wrist transformation
    """

    joint_angles: Float32[ndarray, "n_joints=22"] = np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32)
    wrist_xform: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    hand_confidence: float = 1.0


@dataclass
class HandPoseLabels:
    """
    Dataclass for hand pose labels for a single sequence.
    """

    camera_angles: list[float]
    """List of camera angles in degrees."""
    camera_to_world_transforms: Float32[ndarray, "n_frames n_cams 4 4"]
    """Camera to world transform matrix."""
    hand_model: HandModelNumpy
    """Hand model."""
    joint_angles: Float32[ndarray, "n_frames n_hands=2 n_joints=22"]
    """Joint angles in degrees."""
    wrist_transforms: Float32[ndarray, "n_frames n_hands=2 4 4"]
    """Wrist transform matrix."""
    hand_confidences: Float32[ndarray, "n_frames n_hands=2"]
    """Hand confidence."""

    def __len__(self):
        return len(self.joint_angles)


def so3_exp_map(log_rot: Float32[ndarray, "n 3"], eps: float = 0.0001) -> Float32[ndarray, "n 3 3"]:
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


def _so3_exp_map(
    log_rot: Float32[ndarray, "n 3"], eps: float = 0.0001
) -> tuple[Float32[ndarray, "n 3 3"], Float32[ndarray, "n"], Float32[ndarray, "n 3 3"], Float32[ndarray, "n 3 3"]]:
    log_rot_arr: Float32[ndarray, "n 3"] = np.asarray(log_rot, dtype=np.float32)
    if log_rot_arr.ndim != 2 or log_rot_arr.shape[1] != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms: Float32[ndarray, "n"] = np.sum(log_rot_arr * log_rot_arr, axis=1, dtype=log_rot_arr.dtype)
    rot_angles: Float32[ndarray, "n"] = np.sqrt(np.clip(nrms, eps, None))
    rot_angles_inv: Float32[ndarray, "n"] = 1.0 / rot_angles
    fac1: Float32[ndarray, "n"] = rot_angles_inv * np.sin(rot_angles)
    fac2: Float32[ndarray, "n"] = rot_angles_inv * rot_angles_inv * (1.0 - np.cos(rot_angles))
    skews: Float32[ndarray, "n 3 3"] = hat(log_rot_arr)
    skews_square: Float32[ndarray, "n 3 3"] = np.matmul(skews, skews)

    eye: Float32[ndarray, "1 3 3"] = np.eye(3, dtype=log_rot_arr.dtype)[None]
    R: Float32[ndarray, "n 3 3"] = fac1[:, None, None] * skews + fac2[:, None, None] * skews_square + eye

    return R, rot_angles, skews, skews_square


def hat(v: Float32[ndarray, "n 3"]) -> Float32[ndarray, "n 3 3"]:
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

    v_arr: Float32[ndarray, "n 3"] = np.asarray(v, dtype=np.float32)
    if v_arr.ndim != 2 or v_arr.shape[1] != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    n: int = v_arr.shape[0]
    h: Float32[ndarray, "n 3 3"] = np.zeros((n, 3, 3), dtype=v_arr.dtype)

    x, y, z = v_arr[:, 0], v_arr[:, 1], v_arr[:, 2]

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def _finger_fk(
    joint_local_xfs: Float32[ndarray, "batch dof_per_finger 4 4"],
    parent_transform: Float32[ndarray, "batch 4 4"],
) -> list[Float32[ndarray, "batch 4 4"]]:
    """
    Computes the forward kinematics for a finger with 4 degrees of freedom (DoF),
    i.e., 4 joints, and returns 3 transformation frames.

    Args:
        joint_local_xfs (Float32[ndarray, "batch dof_per_finger 4 4"]): Local joint transformations.
        parent_transform (Float32[ndarray, "batch 4 4"]): Parent transformation matrix.

    Returns:
        list[Float32[ndarray, "batch 4 4"]]: List of computed transformation matrices.
    """

    joint_local_xfs_arr: Float32[ndarray, "batch dof_per_finger 4 4"] = np.asarray(joint_local_xfs, dtype=np.float32)
    parent_transform_arr: Float32[ndarray, "batch 4 4"] = np.asarray(parent_transform, dtype=np.float32)

    transform_mats: list[Float32[ndarray, "batch 4 4"]] = [parent_transform_arr]
    for i in range(DOF_PER_FINGER):
        transform_mats.append(np.matmul(transform_mats[-1], joint_local_xfs_arr[:, i]))
    return transform_mats[2:]


def _joint_local_transform(
    rotation_axis: Float32[ndarray, "batch 20 3"],
    rest_pose: Float32[ndarray, "batch 20 3"],
    joint_angles: Float32[ndarray, "batch 20"],
) -> Float32[ndarray, "batch 20 4 4"]:
    """
    Computes the local transformation matrix for joints given their rotation axes,
    rest poses, and joint angles.

    Args:
        rotation_axis (Float32[ndarray, "batch 20 3"]): Rotation axes of the joints.
        rest_pose (Float32[ndarray, "batch 20 3"]): Rest poses of the joints.
        joint_angles (Float32[ndarray, "batch 20"]): Joint angles.

    Returns:
        Float32[ndarray, "batch 20 4 4"]: Computed local transformation matrix.
    """
    rotation_axis_arr: Float32[ndarray, "batch 20 3"] = np.asarray(rotation_axis, dtype=np.float32)
    rest_pose_arr: Float32[ndarray, "batch 20 3"] = np.asarray(rest_pose, dtype=np.float32)
    joint_angles_arr: Float32[ndarray, "batch 20"] = np.asarray(joint_angles, dtype=np.float32)

    rotation_axis_flat: Float32[ndarray, "n_axes 3"] = rotation_axis_arr.reshape(-1, 3)
    rest_pose_flat: Float32[ndarray, "n_axes 3"] = rest_pose_arr.reshape(-1, 3)
    joint_angles_flat: Float32[ndarray, "n_axes"] = joint_angles_arr.reshape(-1)

    angle_axis: Float32[ndarray, "n_axes 3"] = rotation_axis_flat * joint_angles_flat[:, None]
    local_transform: Float32[ndarray, "n_axes 4 4"] = np.tile(
        np.eye(4, dtype=angle_axis.dtype), (angle_axis.shape[0], 1, 1)
    )

    rot_mat: Float32[ndarray, "n_axes 3 3"] = so3_exp_map(angle_axis)
    translated_rest_pose: Float32[ndarray, "n_axes 3"] = np.einsum("nij,nj->ni", rot_mat, rest_pose_flat)
    translation: Float32[ndarray, "n_axes 3"] = rest_pose_flat - translated_rest_pose
    local_transform[:, :3, :3] = rot_mat
    local_transform[:, :3, 3] = translation

    return local_transform.reshape(rotation_axis_arr.shape[0], -1, 4, 4)


def _get_skinning_weights(
    bone_indices: Int64[ndarray, "num_landmarks max_landmark_weights"],
    bone_weights: Float32[ndarray, "num_landmarks max_landmark_weights"],
    n_frames: int,
) -> Float32[ndarray, "num_landmarks num_joint_frames"]:
    """
    Computes skinning weights for the vertices given the bone indices, bone weights,
    and number of transformation frames.

    Args:
        bone_indices (Int64[ndarray, "num_landmarks max_landmark_weights"]): Indices of bones influencing each vertex.
        bone_weights (Float32[ndarray, "num_landmarks max_landmark_weights"]): Weights of bones for each vertex.
        n_frames (int): Number of transformation frames.

    Returns:
        Float32[ndarray, "num_landmarks num_joint_frames"]: Computed skinning weights for each vertex.
    """
    n_landmarks: int = bone_indices.shape[0]
    skin_mat: Float32[ndarray, "num_landmarks num_joint_frames"] = np.zeros(
        (n_landmarks, n_frames), dtype=bone_weights.dtype
    )
    landmark_indices: Int64[ndarray, "num_landmarks max_landmark_weights"] = np.broadcast_to(
        np.arange(n_landmarks, dtype=np.int64)[:, None], bone_indices.shape
    )
    non_zero_mask: Bool[ndarray, "num_landmarks max_landmark_weights"] = bone_weights != 0
    skin_mat[landmark_indices[non_zero_mask], bone_indices[non_zero_mask]] = bone_weights[non_zero_mask]
    return skin_mat


def _hand_skinning_transform(
    rotation_axis: Float32[ndarray, "batch n_joints=22 3"],
    rest_poses: Float32[ndarray, "batch n_joints=22 3"],
    joint_angles: Float32[ndarray, "batch n_joints=22"],
    wrist_transforms: Float32[ndarray, "batch 4 4"],
) -> Float32[ndarray, "batch num_joint_frames 4 4"]:
    """
    Computes skinning transformation matrices for a hand model given rotation axes,
    rest poses, joint angles, and wrist transformations.

    Args:
        rotation_axis (Float32[ndarray, "batch n_joints=22 3"]): Rotation axes of the joints.
        rest_poses (Float32[ndarray, "batch n_joints=22 3"]): Rest poses of the joints.
        joint_angles (Float32[ndarray, "batch n_joints=22"]): Joint angles.
        wrist_transforms (Float32[ndarray, "batch 4 4"]): Wrist transformations.

    Returns:
        Float32[ndarray, "batch num_joint_frames 4 4"]: Computed skinning transformation matrices.
    """

    transform_mats: list[Float32[ndarray, "batch 4 4"]] = [wrist_transforms, wrist_transforms]
    d = DOF_PER_FINGER

    joint_local_xfs: Float32[ndarray, "batch 20 4 4"] = _joint_local_transform(
        rotation_axis[:, :20], rest_poses[:, :20], joint_angles[:, :20]
    )

    for finger_idx in range(NUM_DIGITS):
        start = d * finger_idx
        end = start + d
        transform_mats.extend(_finger_fk(joint_local_xfs[:, start:end], wrist_transforms))

    return np.concatenate([m[:, None] for m in transform_mats], axis=1)


def _skin_points(
    hand_model: HandModelNumpy,
    skin_mat: Float32[ndarray, "num_points num_joint_frames"],
    points: Float32[ndarray, "num_points 3"],
    joint_angles: Float32[ndarray, "*batch n_joints=22"],
    wrist_transforms: Float32[ndarray, "*#batch 4 4"],
) -> Float32[ndarray, "*batch num_points 3"]:
    """
    Computes skin points for the given joint and wrist transforms.

    Args:
        hand_model (HandModelNumpy): Shared joint rest positions and rotation axes.
        skin_mat (Float32[ndarray, "num_points num_joint_frames"]): Skin matrix.
        joint_angles (Float32[ndarray, "*batch n_joints=22"]): The angles of the joints.
        points (Float32[ndarray, "num_points 3"]): Points to be skinned.
        wrist_transforms (Float32[ndarray, "*#batch 4 4"]): Wrist transformations.

    Returns:
        Float32[ndarray, "*batch num_points 3"]: The skinned vectors for the skin points.
    """
    leading_dims: tuple[int, ...] = tuple(joint_angles.shape[:-1])
    numel: int = math.prod(leading_dims)

    # Joint transforms require batched axes and pivots; points and weights stay shared.
    joint_rest_flat: Float32[ndarray, "batch_flat 22 3"] = np.broadcast_to(hand_model.joint_rest_positions, (*leading_dims, 22, 3)).reshape(numel, 22, 3)
    joint_axis_flat: Float32[ndarray, "batch_flat 22 3"] = np.broadcast_to(hand_model.joint_rotation_axes, (*leading_dims, 22, 3)).reshape(numel, 22, 3)
    joint_angles_flat: Float32[ndarray, "batch_flat 22"] = joint_angles.reshape(numel, 22)
    wrist_transforms_flat: Float32[ndarray, "batch_flat 4 4"] = np.broadcast_to(wrist_transforms, (*leading_dims, 4, 4)).reshape(numel, 4, 4)
    skin_xfs: Float32[ndarray, "batch_flat num_joint_frames 4 4"] = _hand_skinning_transform(
        rotation_axis=joint_axis_flat,
        rest_poses=joint_rest_flat,
        joint_angles=joint_angles_flat,
        wrist_transforms=wrist_transforms_flat,
    )
    points_homogeneous: Float32[ndarray, "num_points 4"] = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=-1
    )
    blended: Float32[ndarray, "batch_flat num_points 4 4"] = np.einsum("vf,nfij->nvij", skin_mat, skin_xfs)
    out: Float32[ndarray, "batch_flat num_points 3"] = np.einsum("nvij,vj->nvi", blended, points_homogeneous)[..., :3]
    return out.reshape(*leading_dims, points.shape[0], 3)


def skin_landmarks(
    hand_model: HandModelNumpy,
    joint_angles: Float32[ndarray, "*batch n_joints=22"],
    wrist_transforms: Float32[ndarray, "*batch 4 4"],
) -> Float32[ndarray, "*batch num_landmarks 3"]:
    """
    Computes the skin landmarks for a given hand model, joint angles, and wrist transforms.

    Use ``wrist_for_hand`` to prepare wrist transforms for either hand.

    Args:
        hand_model (HandModel): A model representing a hand.
        joint_angles (Float32[ndarray, "*batch n_joints=22"]): The angles of the joints.
        wrist_transforms (Float32[ndarray, "*batch 4 4"]): Wrist transformations.

    Returns:
        Float32[ndarray, "*batch num_landmarks 3"]: The skinned landmarks.
    """

    return _skin_points(
        hand_model,
        _get_skinning_weights(
            hand_model.landmark_rest_bone_indices,
            hand_model.landmark_rest_bone_weights,
            NUM_JOINT_FRAMES,
        ),
        hand_model.landmark_rest_positions,
        joint_angles,
        wrist_transforms,
    )


def skin_mesh(
    hand_model: HandModelNumpy,
    joint_angles: Float32[ndarray, "*batch n_joints=22"],
    wrist_transforms: Float32[ndarray, "*batch 4 4"],
) -> Float32[ndarray, "*batch num_mesh_vertices 3"]:
    """Skin a shared hand mesh over the leading pose batch dimensions.

    Coordinates retain the model's units. Use ``wrist_for_hand`` to prepare
    wrist transforms for either hand.

    Args:
        hand_model (HandModelNumpy): Left-hand rest mesh and dense blend weights.
        joint_angles (Float32[ndarray, "*batch n_joints=22"]): Joint angles in radians.
        wrist_transforms (Float32[ndarray, "*batch 4 4"]): World-from-wrist transforms.

    Returns:
        Float32[ndarray, "*batch num_mesh_vertices 3"]: Mesh vertices in the world frame.
    """
    return _skin_points(
        hand_model,
        hand_model.dense_bone_weights,
        hand_model.mesh_vertices,
        joint_angles,
        wrist_transforms,
    )


def wrist_for_hand(wrist_transforms: Float32[ndarray, "*batch 4 4"], hand_idx: int) -> Float32[ndarray, "*batch 4 4"]:
    """Copy wrist transforms and mirror right hands for the left-hand model.

    Args:
        wrist_transforms (Float32[ndarray, "*batch 4 4"]): World-from-wrist transforms.
        hand_idx (int): LEFT_HAND_INDEX or RIGHT_HAND_INDEX.

    Returns:
        Float32[ndarray, "*batch 4 4"]: Independent transforms ready for skinning.
    """
    wrist: Float32[ndarray, "*batch 4 4"] = wrist_transforms.copy()
    if hand_idx == RIGHT_HAND_INDEX:
        wrist[..., :, 0] *= -1
    return wrist


def landmarks_from_hand_pose(
    hand_model: HandModelNumpy, hand_pose: SingleHandPose, hand_idx: int
) -> Float32[ndarray, "num_landmarks 3"]:
    """
    Compute 3D landmarks in the world space given the hand model and hand pose.

    Args:
        hand_model (HandModelTensor): A model representing a hand.
        hand_pose (SingleHandPose): A pose representing the hand's position and orientation.
        hand_idx (int): Index of the hand. Equals 0 if left hand, 1 if right hand.

    Returns:
        Float32[ndarray, "num_landmarks 3"]: The 3D landmarks in the world space.
    """

    xf: Float32[ndarray, "4 4"] = wrist_for_hand(hand_pose.wrist_xform, hand_idx)
    landmarks: Float32[ndarray, "num_landmarks 3"] = skin_landmarks(
        hand_model,
        hand_pose.joint_angles,
        xf,
    )
    return landmarks
