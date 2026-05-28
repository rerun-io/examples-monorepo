import numpy as np
from jaxtyping import Float

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters


def print_tensor_info(tensor):
    print(f"shape: {tensor.shape}")
    print(f"dtype: {tensor.dtype}")


def get_transformation_matrix(
    camera: Fisheye62Parameters | PinholeParameters,
) -> Float[np.ndarray, "4 4"]:
    """Return the camera-from-world transform for unified camera parameters."""

    return np.array(camera.extrinsics.cam_T_world, dtype=np.float32, copy=False)
