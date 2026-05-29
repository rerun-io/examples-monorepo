import numpy as np
from jaxtyping import Float

from simplecv.umetrack_temp.cameras import Camera


def project_points(
    points3d_world: Float[np.ndarray, "num_points 3"],
    camera: Camera,
) -> Float[np.ndarray, "num_points 2"]:
    """
    Projects 3D points onto a 2D camera plane based on the given camera parameters. Out of bounds points are set to NaN.

    Args:
        points3d (Float[np.ndarray, "num_points 3"]): A numpy array containing the 3D coordinates of the points.
        camera: Camera wrapper containing either fisheye or pinhole parameters.

    Returns:
        Float[np.ndarray, "num_points 2"]: A numpy array containing the 2D coordinates of the projected points.
    """
    points3d_cam: Float[np.ndarray, "num_points 3"] = camera.world_to_camera(points3d_world)
    points2d: Float[np.ndarray, "num_points 2"] = camera.camera_to_image(points3d_cam)
    h: int = camera.camera_parameters.intrinsics.height
    w: int = camera.camera_parameters.intrinsics.width

    # make sure points are within image bounds
    out_of_bounds = np.logical_or(points2d[:, 0] >= w, points2d[:, 1] >= h)
    out_of_bounds = np.logical_or(out_of_bounds, points2d[:, 0] < 0)
    out_of_bounds = np.logical_or(out_of_bounds, points2d[:, 1] < 0)
    # make sure points are in front of camera
    out_of_bounds = np.logical_or(out_of_bounds, points3d_cam[:, 2] < 0)

    # if out of bounds, set to nan
    points2d[out_of_bounds, :] = np.nan

    return points2d
