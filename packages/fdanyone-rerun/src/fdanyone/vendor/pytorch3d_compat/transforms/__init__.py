"""PyTorch3D 0.7.6 rotation conversions used by classic GVHMR inference."""

from .rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


def so3_exp_map(log_rot, eps: float = 0.0001):
    """Match the PyTorch3D 0.7.6 SO(3) exponential map used by GVHMR."""

    del eps
    if log_rot.ndim != 2 or log_rot.shape[1] != 3:
        raise ValueError("Input tensor shape has to be Nx3.")
    return axis_angle_to_matrix(log_rot)


def so3_log_map(rotation, eps: float = 0.0001, cos_bound: float = 1e-4):
    """Match the PyTorch3D 0.7.6 SO(3) logarithm used by GVHMR."""

    del eps, cos_bound
    if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
        raise ValueError("Input has to be a batch of 3x3 Tensors.")
    return matrix_to_axis_angle(rotation)


__all__ = [
    "axis_angle_to_matrix",
    "euler_angles_to_matrix",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rotation_6d_to_matrix",
    "so3_exp_map",
    "so3_log_map",
]
