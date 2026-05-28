import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    PinholeParameters,
    apply_radial_tangential_distortion,
    arctan_projection,
    perspective_projection,
)


class Camera:
    """Lightweight wrapper exposing common projections for pinhole and fisheye models."""

    def __init__(self, camera_parameters: PinholeParameters | Fisheye62Parameters) -> None:
        self.camera_parameters: PinholeParameters | Fisheye62Parameters = camera_parameters
        self._refresh_extrinsics()

    def _refresh_extrinsics(self) -> None:
        extrinsics = self.camera_parameters.extrinsics
        self.cam_T_world: Float32[ndarray, "4 4"] = np.asarray(extrinsics.cam_T_world, dtype=np.float32)
        self.world_T_cam: Float32[ndarray, "4 4"] = np.asarray(extrinsics.world_T_cam, dtype=np.float32)

    def set_extrinsic(self, cam_T_world: Float32[ndarray, "4 4"]) -> None:
        cam_T_world = np.asarray(cam_T_world, dtype=np.float32)
        cam_R_world: Float32[ndarray, "3 3"] = cam_T_world[:3, :3]
        cam_t_world: Float32[ndarray, "3"] = cam_T_world[:3, 3]
        self.camera_parameters.extrinsics = Extrinsics(cam_R_world=cam_R_world, cam_t_world=cam_t_world)
        self.camera_parameters.compute_projection_matrix()
        self._refresh_extrinsics()

    def camera_to_image(self, points_3d: Float32[ndarray, "n_points 3"]) -> Float32[ndarray, "n_points 2"]:
        intrinsics: Intrinsics = self.camera_parameters.intrinsics
        if isinstance(self.camera_parameters, PinholeParameters):
            points_2d = perspective_projection(points_3d, intrinsics.k_matrix)
        elif isinstance(self.camera_parameters, Fisheye62Parameters):
            points_2d = arctan_projection(points_3d, intrinsics.k_matrix)
            # Apply the camera distortion parameters to the 2D image coordinates
            # normalize points before applying distortion
            if self.camera_parameters.distortion is not None:
                points_2d[:, 0] -= intrinsics.cx
                points_2d[:, 1] -= intrinsics.cy
                points_2d[:, 0] /= intrinsics.fl_x
                points_2d[:, 1] /= intrinsics.fl_y

                points_2d = apply_radial_tangential_distortion(self.camera_parameters.distortion, points_2d)

                # denormalize points after applying distortion
                points_2d[:, 0] *= intrinsics.fl_x
                points_2d[:, 1] *= intrinsics.fl_y
                points_2d[:, 0] += intrinsics.cx
                points_2d[:, 1] += intrinsics.cy
        else:
            raise NotImplementedError(f"Camera model {type(self.camera_parameters)} not supported.")

        return points_2d.astype(np.float32, copy=False)

    def image_to_camera(self, points_2d: Float32[ndarray, "num_points 2"]) -> Float32[ndarray, "num_points 3"]:
        assert isinstance(self.camera_parameters, PinholeParameters), "Only pinhole cameras support back-projection"
        assert self.camera_parameters.distortion is None, "Inverse distortion not implemented for crop cameras"

        K_inv: Float32[ndarray, "3 3"] = np.linalg.inv(self.camera_parameters.intrinsics.k_matrix).astype(
            np.float32, copy=False
        )
        points_2d_hom: Float32[ndarray, "num_points 3"] = np.concatenate(
            [points_2d, np.ones((points_2d.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        points_3d_hom: Float32[ndarray, "3 num_points"] = K_inv @ points_2d_hom.T
        points_3d: Float32[ndarray, "num_points 3"] = points_3d_hom[:3, :].T
        norm: Float32[ndarray, "num_points 1"] = np.linalg.norm(points_3d, axis=1, keepdims=True)
        return points_3d / norm

    def camera_to_world(self, points_3d_cam: Float32[ndarray, "num_points 3"]) -> Float32[ndarray, "num_points 3"]:
        points3d_hom: Float32[ndarray, "num_points 4"] = np.ones((points_3d_cam.shape[0], 4), dtype=np.float32)
        points3d_hom[:, :3] = points_3d_cam
        points3d_world: Float32[ndarray, "num_points 3"] = (
            self.world_T_cam @ points3d_hom.T
        ).T[:, :3].astype(np.float32, copy=False)
        return points3d_world

    def world_to_camera(self, points_3d_world: Float32[ndarray, "num_points 3"]) -> Float32[ndarray, "num_points 3"]:
        points3d_hom: Float32[ndarray, "num_points 4"] = np.ones((points_3d_world.shape[0], 4), dtype=np.float32)
        points3d_hom[:, :3] = points_3d_world
        points3d_cam: Float32[ndarray, "num_points 3"] = (
            self.cam_T_world @ points3d_hom.T
        ).T[:, :3].astype(np.float32, copy=False)
        return points3d_cam
