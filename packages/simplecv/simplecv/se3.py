"""Rigid transforms held the way Sophus holds them: a unit quaternion and a translation.

Project Aria calibrations state every pose as a scalar-first quaternion and a
translation, and projectaria-tools composes them with Sophus 1.22. ``SE3``
repeats that arithmetic operation for operation (normalisation, the quaternion
product, the rotation of a point, Eigen's quaternion-to-matrix formula), so a
pose read and composed here has the same float64 bits as the SDK's
``SE3.to_matrix()``. A matrix round trip (quaternion to matrix, then matrix
products) is equally correct but differs from it in the last bits.

Quaternion order is **w, x, y, z** throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

SOPHUS_EPSILON: float = 1e-10
"""``Sophus::Constants<double>::epsilon()``: below its square, ``SO3::exp`` switches to a Taylor expansion."""


def _normalized(w: float, x: float, y: float, z: float) -> Float64[ndarray, "4"]:
    """``Eigen::Quaternion::normalize``: divide by the norm of the (x, y, z, w) coefficients.

    Eigen's unrolled reduction sums the squared coefficients pairwise, (x² + y²) + (z² + w²).
    """
    norm: float = math.sqrt((x * x + y * y) + (z * z + w * w))
    return np.array([w / norm, x / norm, y / norm, z / norm], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SE3:
    """A rigid transform ``a_T_b``: rotation as a unit quaternion (w, x, y, z), then translation in metres."""

    quaternion_wxyz: Float64[ndarray, "4"]
    """Unit rotation quaternion, scalar first."""
    translation: Float64[ndarray, "3"]
    """Translation of ``b``'s origin in ``a``."""

    @classmethod
    def from_quaternion(cls, quaternion_wxyz: Float64[ndarray, "4"], translation: Float64[ndarray, "3"]) -> SE3:
        """Normalise the quaternion, as ``Sophus::SE3d(Eigen::Quaterniond, t)`` does."""
        w, x, y, z = (float(value) for value in quaternion_wxyz)
        return cls(_normalized(w, x, y, z), np.array(translation, dtype=np.float64))

    @classmethod
    def rot_z(cls, angle_rad: float) -> SE3:
        """A rotation about z with no translation, as ``Sophus::SE3d::rotZ`` builds it (``SO3::exp``).

        Below Sophus's epsilon (1e-10) it takes Sophus's Taylor branch, so ``rot_z(0.0)`` is the identity.
        """
        theta_sq: float = angle_rad * angle_rad
        if theta_sq < SOPHUS_EPSILON * SOPHUS_EPSILON:
            theta_po4: float = theta_sq * theta_sq
            imag_factor: float = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4
            real_factor: float = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_po4
        else:
            theta: float = math.sqrt(theta_sq)
            imag_factor = math.sin(0.5 * theta) / theta
            real_factor = math.cos(0.5 * theta)
        return cls(np.array([real_factor, 0.0, 0.0, imag_factor * angle_rad], dtype=np.float64), np.zeros(3, dtype=np.float64))

    def __matmul__(self, other: SE3) -> SE3:
        """Compose ``a_T_b @ b_T_c`` into ``a_T_c``; the product quaternion is renormalised, as in Sophus."""
        aw, ax, ay, az = (float(value) for value in self.quaternion_wxyz)
        bw, bx, by, bz = (float(value) for value in other.quaternion_wxyz)
        rotation: Float64[ndarray, "4"] = _normalized(
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx,
        )
        return SE3(rotation, self.translation + self.rotate(other.translation))

    def rotate(self, point: Float64[ndarray, "3"]) -> Float64[ndarray, "3"]:
        """Rotate one point: ``p + 2w (v × p) + v × (2 (v × p))`` with ``v`` the vector part, as Sophus does."""
        w, x, y, z = (float(value) for value in self.quaternion_wxyz)
        px, py, pz = (float(value) for value in point)
        ux, uy, uz = y * pz - z * py, z * px - x * pz, x * py - y * px
        ux, uy, uz = ux + ux, uy + uy, uz + uz
        return np.array(
            [px + w * ux + (y * uz - z * uy), py + w * uy + (z * ux - x * uz), pz + w * uz + (x * uy - y * ux)],
            dtype=np.float64,
        )

    def rotation_matrix(self) -> Float64[ndarray, "3 3"]:
        """The rotation as a matrix, by Eigen's ``Quaternion::toRotationMatrix``."""
        w, x, y, z = (float(value) for value in self.quaternion_wxyz)
        tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
        twx, twy, twz = tx * w, ty * w, tz * w
        txx, txy, txz = tx * x, ty * x, tz * x
        tyy, tyz, tzz = ty * y, tz * y, tz * z
        return np.array(
            [
                [1.0 - (tyy + tzz), txy - twz, txz + twy],
                [txy + twz, 1.0 - (txx + tzz), tyz - twx],
                [txz - twy, tyz + twx, 1.0 - (txx + tyy)],
            ],
            dtype=np.float64,
        )

    def matrix(self) -> Float64[ndarray, "4 4"]:
        """The transform as a homogeneous 4x4."""
        result: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.rotation_matrix()
        result[:3, 3] = self.translation
        return result
