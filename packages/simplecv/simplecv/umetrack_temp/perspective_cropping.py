import cv2
import numpy as np
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.umetrack_temp.cameras import Camera
from simplecv.umetrack_temp.generic_hand_model_numpy import (
    NUM_JOINTS_PER_HAND,
    SingleHandPose,
    landmarks_from_hand_pose,
)
from simplecv.umetrack_temp.generic_hand_model_numpy import (
    HandModelNumpy as HandModel,
)


def normalized(vector: ndarray, axis: int = -1, epsilon: float = 5.43e-20) -> ndarray:
    """
    Returns a normalized version of the input vector along the specified axis.

    Args:
        vector: A numpy array representing the vector to normalize.
        axis: An integer representing the axis along which to compute the norm. Default is -1.
        epsilon: A small value added to the norm to avoid division by zero. Default is 5.43e-20.

    Returns:
        A numpy array representing the normalized vector.
    """
    # Compute the norm of the input vector along the specified axis
    norm = np.maximum(epsilon, (vector * vector).sum(axis=axis, keepdims=True) ** 0.5)

    # Divide the input vector by its norm to obtain the normalized version
    normalized_vector = vector / norm

    return normalized_vector


def skew_matrix(vector: Float32[ndarray, "3"]) -> Float32[ndarray, "3 3"]:
    """
    Computes the skew-symmetric matrix of a 3D vector.

    Args:
        vector: A 3D numpy array representing the input vector.

    Returns:
        A 3x3 numpy array representing the skew-symmetric matrix of the input vector.
    """
    skew_symmetric_matrix = np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ],
        dtype=vector.dtype,
    )

    return skew_symmetric_matrix


def from_two_vectors(vector_a: Float32[ndarray, "3"], vector_b: Float32[ndarray, "3"]) -> Float32[ndarray, "3 3"]:
    """
    Computes a 3x3 rotation matrix that rotates vector `vector_a` towards vector `vector_b`.

    Args:
        vector_a: A 3D numpy array representing the original vector to rotate.
        vector_b: A 3D numpy array representing the target vector to rotate towards.

    Returns:
        A 3x3 numpy array representing the rotation matrix that rotates `vector_a` towards `vector_b`.
    """
    # Normalize the input vectors to have unit length
    normalized_vector_a = normalized(vector_a)
    normalized_vector_b = normalized(vector_b)

    # Compute the cross product of the normalized vectors to obtain a vector perpendicular to both
    perpendicular_vector = np.cross(normalized_vector_a, normalized_vector_b)

    # Compute the norm of the perpendicular vector and the dot product of the input vectors
    perpendicular_vector_norm = np.linalg.norm(perpendicular_vector)
    dot_product = np.dot(normalized_vector_a, normalized_vector_b)

    # Construct a skew-symmetric matrix from the perpendicular vector
    skew_symmetric_matrix = skew_matrix(perpendicular_vector)

    # Compute the rotation matrix as a sum of three terms
    dtype = normalized_vector_a.dtype
    identity_matrix = np.eye(3, dtype=dtype)
    rotation_term_1 = skew_symmetric_matrix
    rotation_term_2 = (
        np.matmul(skew_symmetric_matrix, skew_symmetric_matrix)
        * (1 - dot_product)
        / (
            max(
                perpendicular_vector_norm * perpendicular_vector_norm,
                np.array(1e-15, dtype=dtype),
            )
        )
    )
    rotation_matrix = (identity_matrix + rotation_term_1 + rotation_term_2).astype(dtype, copy=False)

    return rotation_matrix


def make_look_at_matrix(
    origin_cam_T_world: Float32[ndarray, "4 4"],
    center: Float32[ndarray, "3"],
    camera_angle: float = 0,
) -> Float32[ndarray, "4 4"]:
    """
    Computes a 4x4 transformation matrix that transforms points from world coordinates to camera coordinates.

    Args:
        origin_cam_T_world: A 4x4 numpy array representing the transformation matrix that transforms points from world coordinates to camera coordinates.
        center: A 3D numpy array representing the center of the camera in world coordinates.
        camera_angle: A float representing the angle of the camera in degrees. Default is 0.

    Returns:
        A 4x4 numpy array representing the transformation matrix that transforms points from world coordinates to camera coordinates.
    """
    origin_cam_T_world = origin_cam_T_world.astype(np.float32, copy=False)
    center = center.astype(np.float32, copy=False)

    # Compute the center of the camera in camera coordinates
    center_hom = np.concatenate((center, np.array([1.0], dtype=np.float32)))
    center_cam_hom = origin_cam_T_world @ center_hom
    center_cam = center_cam_hom[0:3] / center_cam_hom[3]

    # Compute the direction of the camera in camera coordinates
    z_dir_cam = center_cam / np.linalg.norm(center_cam)
    z_dir_cam = z_dir_cam.astype(np.float32, copy=False)

    # Compute the rotation matrix that rotates the world z-axis to the camera direction
    delta_r_local = from_two_vectors(np.array([0, 0, 1], dtype=np.float32), z_dir_cam)
    orig_world_T_cam = np.linalg.inv(origin_cam_T_world).astype(np.float32)
    new_world_T_cam = orig_world_T_cam.copy()
    new_world_T_cam[0:3, 0:3] = orig_world_T_cam[0:3, 0:3] @ delta_r_local

    # Rotate the camera around the z-axis to align with the camera angle
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix().astype(np.float32)
    new_world_T_cam[0:3, 0:3] = new_world_T_cam[0:3, 0:3] @ z_local_rot

    # Compute the transformation matrix that transforms points from world coordinates to camera coordinates
    new_cam_T_world = np.linalg.inv(new_world_T_cam).astype(np.float32)

    return new_cam_T_world


def gen_intrinsics_from_bounding_pts(
    pts_cam: Float32[ndarray, "num_pts 3"], width: int, height: int, min_focal: float = 5
) -> tuple[Float32[ndarray, "2"], Float32[ndarray, "2"]]:
    """
    Computes the camera intrinsics matrix from the input points in camera space and the target image size.

    Args:
        pts_cam: Points in camera space that must be projected inside the image.
        width: The target image width.
        height: The target image height.
        min_focal: The minimum focal length allowed. Default is 5.

    Returns:
        A tuple containing the focal length and principal point of the camera intrinsics matrix.
    """
    pts_ndc = pts_cam[..., 0:2] / pts_cam[..., 2:]
    img_size = np.array([width, height], dtype=pts_cam.dtype)
    # Need to shift one pixel before dividing by 2.
    pricipal_pts = (img_size - 1) / 2
    focal = pricipal_pts / np.absolute(pts_ndc).max()

    if np.any(pts_cam[..., 2:] < 0.0001) or np.any(focal < min_focal):
        raise ValueError("Unable to create crop camera", focal)

    return focal, pricipal_pts


def gen_crop_parameters_from_points(
    camera_orig: Camera,
    pts_world: Float32[ndarray, "... 3"],
    new_image_size: tuple[int, int],
    mirror_img_x: bool,
    camera_angle: float = 0,
    focal_multiplier: float = 0.95,
) -> PinholeParameters:
    """
    Computes a new perspective camera that ensures all input points can be projected inside the image.

    Args:
        camera_orig: original camera used for creating image. The returned camera will have the same position but different rotation and intrinsics parameters.
        pts_world: Points in world space that must be projected inside the image by the generated world to cam transform and intrinsics.
        new_image_size: The target image size.
        mirror_img_x: Whether to flip the image. A typical use case is to mirror the right-hand images so that a model needs to handle left-hand data only.
        camera_angle: How the camera is oriented physically so that we can rotate the object of interest to the 'upright' direction. Default is 0.
        focal_multiplier: The focal multiplier. When less than 1, we are zooming out a little. The effect on the image is some margin will be left at the boundary. Default is 0.95.

    Returns:
        A PinholeParameters object representing the new perspective camera.
    """
    # Implementation code here
    cam_T_world = camera_orig.cam_T_world.astype(np.float32, copy=False)
    pts_world = pts_world.astype(np.float32, copy=False)

    crop_center = (pts_world.min(axis=0) + pts_world.max(axis=0)) / 2.0
    new_cam_T_world = make_look_at_matrix(cam_T_world, crop_center, camera_angle)
    if mirror_img_x:
        mirrorx = np.eye(4, dtype=np.float32)
        mirrorx[0, 0] = -1
        new_cam_T_world = mirrorx @ new_cam_T_world

    # convert pts_world to homogenous coordinates
    pts_world_hom = np.concatenate(
        [pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    # compute camera coordinates of input points
    pts_cam_hom = pts_world_hom @ new_cam_T_world.T
    pts_cam = pts_cam_hom[:, :3] / pts_cam_hom[:, 3:]

    focal, principal_point = gen_intrinsics_from_bounding_pts(
        pts_cam,
        new_image_size[0],
        new_image_size[1],
    )
    focal = focal_multiplier * focal

    K: Float32[ndarray, "3 3"] = np.array(
        [
            [focal[0], 0.0, principal_point[0]],
            [0.0, focal[1], principal_point[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cam_r_world: Float32[ndarray, "3 3"] = new_cam_T_world[:3, :3].astype(np.float32, copy=False)
    cam_t_world: Float32[ndarray, "3"] = new_cam_T_world[:3, 3].astype(np.float32, copy=False)

    intrinsics = Intrinsics(
        camera_conventions="RDF",
        fl_x=float(K[0, 0]),
        fl_y=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        height=new_image_size[1],
        width=new_image_size[0],
    )
    extrinsics = Extrinsics(cam_R_world=cam_r_world, cam_t_world=cam_t_world)
    base_name = getattr(camera_orig.camera_parameters, "name", "crop")
    return PinholeParameters(name=f"{base_name}_crop", extrinsics=extrinsics, intrinsics=intrinsics)


def neutral_joint_angles(up: HandModel, lower_factor: float = 0.5) -> Float32[ndarray, "n_joints=22"]:
    joint_limits = up.joint_limits
    assert joint_limits is not None
    neutral_pose = joint_limits[..., 0] * lower_factor + joint_limits[..., 1] * (1 - lower_factor)
    return np.asarray(neutral_pose, dtype=np.float32)


def rank_hand_visibility_in_cameras(
    cameras: list[Camera],
    hand_model: HandModel,
    hand_pose: SingleHandPose,
    hand_idx: int,
    min_required_vis_landmarks: int,
) -> list[int]:
    """
    Ranks the given cameras based on the visibility of hand landmarks in their view.

    This function evaluates each camera's view and ranks them based on the number of hand
    landmarks they can see from the given hand pose. It only includes the cameras that can see
    a minimum required number of landmarks.

    Args:
        cameras (List[Camera]): A list of camera wrappers to rank.
        hand_model (HandModelTensor): The model of the hand whose visibility is to be evaluated.
        hand_pose (SingleHandPose): The pose of the hand in the scene.
        hand_idx (int): The index of the hand to be evaluated (0 is left 1 is right).
        min_required_vis_landmarks (int): The minimum number of landmarks a camera needs to see to be included in the ranking.

    Returns:
        List[int]: A list of indices of the cameras in the input list, sorted in decreasing order of the number of visible landmarks.
    """
    landmarks_world = landmarks_from_hand_pose(hand_model, hand_pose, hand_idx)
    n_landmarks_in_view = []
    ranked_cam_indices = []
    for cam_idx, camera in enumerate(cameras):
        h = camera.camera_parameters.intrinsics.height
        w = camera.camera_parameters.intrinsics.width
        landmarks_cam = camera.world_to_camera(landmarks_world)
        landmarks_uv = camera.camera_to_image(landmarks_cam)

        # ensures that the projected points are within the image and infront of the camera
        width_visible = (landmarks_uv[..., 0] >= 0) & (landmarks_uv[..., 0] <= w - 1)
        height_visible = (landmarks_uv[..., 1] >= 0) & (landmarks_uv[..., 1] <= h - 1)
        infront_of_cam = landmarks_cam[..., 2] > 0

        n_visible = (width_visible & height_visible & infront_of_cam).sum()
        n_landmarks_in_view.append(n_visible)

        # Only push the cameras that can see enough hand points
        if n_visible >= min_required_vis_landmarks:
            ranked_cam_indices.append(cam_idx)

    # Favor the view that sees more landmarks
    ranked_cam_indices.sort(
        reverse=True,
        key=lambda x: n_landmarks_in_view[x],
    )
    return ranked_cam_indices


def get_crop_points_from_hand_pose(
    hand_model: HandModel,
    gt_hand_pose: SingleHandPose,
    hand_idx: int,
    num_crop_points: int,
) -> Float32[ndarray, "n_crop_points 3"]:
    """
    Generates crop points for hand images based on various hand poses.

    Crop points, used to isolate the hand in an image, are the hand's landmark points. Depending on the
    requested number of crop points (21, 42, or 63), this function uses landmarks from the ground truth pose,
    neutral pose, and/or open hand pose.

    Args:
        hand_model (HandModelTensor): The hand model for pose generation.
        gt_hand_pose (SingleHandPose): The ground truth hand pose.
        hand_idx (int): Index of the hand.
        num_crop_points (int): Desired number of crop points (must be 21, 42, or 63).

    Returns:
        ndarray: Array of concatenated landmarks serving as crop points for the image.

    Raises:
        AssertionError: If num_crop_points is not one of 21, 42, or 63.
    """
    assert num_crop_points in [21, 42, 63]
    neutral_joint_angles_np: Float32[ndarray, "n_joints=22"] = neutral_joint_angles(hand_model)
    neutral_hand_pose = SingleHandPose(
        joint_angles=neutral_joint_angles_np,
        wrist_xform=gt_hand_pose.wrist_xform,
    )
    open_hand_pose = SingleHandPose(
        joint_angles=np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32),
        wrist_xform=gt_hand_pose.wrist_xform,
    )

    crop_points = []
    crop_points.append(landmarks_from_hand_pose(hand_model, gt_hand_pose, hand_idx))
    if num_crop_points > 21:
        crop_points.append(landmarks_from_hand_pose(hand_model, neutral_hand_pose, hand_idx))
    if num_crop_points > 42:
        crop_points.append(landmarks_from_hand_pose(hand_model, open_hand_pose, hand_idx))
    return np.concatenate(crop_points, axis=0)


def warp_image_between_cameras(
    src_camera: Camera,
    dst_camera: Camera,
    src_image: UInt8[ndarray, "H W channels"],
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> UInt8[ndarray, "H W channels"]:
    """
    Warps an image from the source camera to the destination camera.

    Args:
        src_camera: The source camera.
        dst_camera: The destination camera.
        src_image: The image to warp.
        interpolation: The interpolation method to use. Default is cv2.INTER_LINEAR.
        depth_check: Whether to mask out points with negative z coordinates. Default is True.

    Returns:
        The warped image.
    """
    # Compute the destination image size
    W: int = dst_camera.camera_parameters.intrinsics.width
    H: int = dst_camera.camera_parameters.intrinsics.height

    # Generate a grid of destination image points
    meshgrid_axes: tuple[Float32[ndarray, "H W"], Float32[ndarray, "H W"]] = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    px: Float32[ndarray, "H W"] = meshgrid_axes[0]
    py: Float32[ndarray, "H W"] = meshgrid_axes[1]
    num_pixels: int = H * W
    dst_img_pts: Float32[ndarray, "num_pixels 2"] = np.column_stack((px.flatten(), py.flatten())).astype(
        np.float32, copy=False
    )
    assert dst_img_pts.shape[0] == num_pixels

    # Compute the corresponding world points and camera points
    dst_cam_pts: Float32[ndarray, "num_pixels 3"] = dst_camera.image_to_camera(dst_img_pts)
    world_pts: Float32[ndarray, "num_pixels 3"] = dst_camera.camera_to_world(dst_cam_pts)
    src_cam_pts: Float32[ndarray, "num_pixels 3"] = src_camera.world_to_camera(world_pts)
    src_img_pts: Float32[ndarray, "num_pixels 2"] = src_camera.camera_to_image(src_cam_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask: Bool[ndarray, "num_pixels"] = src_cam_pts[:, 2] < 0
        src_img_pts[mask] = -1

    # Convert the image points to map coordinates
    src_img_pts_float32: Float32[ndarray, "num_pixels 2"] = src_img_pts.astype(np.float32)
    map_x: Float32[ndarray, "H W"] = src_img_pts_float32[:, 0].reshape((H, W))
    map_y: Float32[ndarray, "H W"] = src_img_pts_float32[:, 1].reshape((H, W))

    # Warp the source image to the destination image
    warped_image: UInt8[ndarray, "H W channels"] = cv2.remap(src_image, map_x, map_y, interpolation)

    return warped_image
