from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from vistadream.ops.gs.basic import Gaussian_Scene
from vistadream.ops.utils import dpt2xyz


class Traj_Base(ABC):
    def __init__(self, scene: Gaussian_Scene, nframe: int = 100) -> None:
        self.scene: Gaussian_Scene = scene
        self.nframe: int = nframe
        self.min_percentage: int | float = scene.traj_min_percentage
        self.max_percentage: int | float = scene.traj_max_percentage
        self._radius()

    def _radius(self) -> None:
        """
        Computes and sets the radius attribute based on the 3D coordinates derived from the depth map.

        This method performs the following steps:
            1. Converts the depth map (`dpt`) from the first scene frame into 3D coordinates (`xyz`)
               using the camera intrinsic matrix.
            2. Reshapes the 3D coordinates to a 2D array of shape (n_points, 3) if necessary.
            3. Calculates the minimum and maximum percentiles along each axis (x, y, z) using
               `self.min_percentage` and `self.max_percentage`.
            4. Computes the range for each axis as the difference between the corresponding percentiles.
            5. Sets `self.radius` to the mean of the ranges across all three axes.

        Returns:
            None
        """
        dpt: Float[ndarray, "H W"] = self.scene.frames[0].dpt
        intrinsic: Float[ndarray, "3 3"] = self.scene.frames[0].intrinsic
        self.xyz: Float[ndarray, "H W 3"] = dpt2xyz(dpt, intrinsic)
        if self.xyz.ndim > 2:
            self.xyz: Float[ndarray, "n_points 3"] = self.xyz.reshape(-1, 3)
        # get range
        _min: Float[ndarray, "3"] = np.percentile(self.xyz, self.min_percentage, axis=0)
        _max: Float[ndarray, "3"] = np.percentile(self.xyz, self.max_percentage, axis=0)
        _range: Float[ndarray, "3"] = _max - _min
        # set radius to mean range of three axes
        self.radius: float = float(np.mean(_range))

    def rot_by_look_at(
        self,
        camera_position: Float[ndarray, "3"],
        target_position: Float[ndarray, "3"],
        camera_up: Float[ndarray, "3"],
    ) -> Float[ndarray, "3 3"]:
        """Calculates the camera's rotation matrix to look at a target point.

        This function computes a 3x3 rotation matrix that defines the orientation
        of a camera in world space. The camera is positioned at `camera_position`,
        looks towards `target_position`, and is oriented with the given `camera_up`
        vector.

        The calculation follows the standard "look-at" logic to create an orthonormal basis
        (right, up, direction) for the camera's coordinate system. The input `camera_up`
        is inverted to align with typical image coordinate systems where the Y-axis points down.

        Args:
            camera_position: The 3D coordinates of the camera's position.
            target_position: The 3D coordinates of the point to look at.
            camera_up: The initial "up" vector for the camera, defining its roll.

        Returns:
            A 3x3 rotation matrix representing the camera's orientation in world space.
        """
        # look at direction
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        up = -camera_up  # For the image origin is left-up: y is inverse
        up /= np.linalg.norm(up)
        # calculate rotation matrix
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.column_stack([right, up, direction])
        return rotation_matrix

    def trans_by_look_at(
        self, camera_triples: list[tuple[Float[ndarray, "3"], Float[ndarray, "3"], Float[ndarray, "3"]]]
    ) -> Float[np.ndarray, "n_frames 4 4"]:
        """
        Generates a sequence of 4x4 world-to-camera transformation matrices from camera poses.

        Args:
            camera_triples (list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
                A list of tuples, each containing:
                    - pos (np.ndarray of shape (3,)): Camera position in world coordinates.
                    - target (np.ndarray of shape (3,)): Point in world space the camera is looking at.
                    - up (np.ndarray of shape (3,)): Approximate up direction for the camera.

        Returns:
            np.ndarray: Array of shape (n_frames, 4, 4), where each [i] is a 4x4 transformation matrix
            representing the world-to-camera pose for the i-th frame.

        Notes:
            - The coordinate system convention is: z (forward), x (right), y (down).
            - Each transformation matrix is constructed by computing the rotation matrix from the look-at
              parameters and embedding the camera position as the translation component.
        """
        world_T_cam: list[Float[ndarray, "1 4 4"]] = []
        for camera in camera_triples:
            pos: Float[ndarray, "3"] = camera[0]
            target: Float[ndarray, "3"] = camera[1]
            up: Float[ndarray, "3"] = camera[2]
            rotation_matrix: Float[ndarray, "3 3"] = self.rot_by_look_at(pos, target, up)
            transform_matrix: Float[ndarray, "4 4"] = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = pos
            world_T_cam.append(transform_matrix[None])
        world_T_cam: Float[ndarray, "n_frames 4 4"] = np.concatenate(world_T_cam, axis=0)
        return world_T_cam

    @abstractmethod
    def camera_target_up(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self):
        camera_triples = self.camera_target_up()
        trajs = self.trans_by_look_at(camera_triples)
        return trajs
