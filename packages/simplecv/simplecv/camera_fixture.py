"""Typed projection fixtures shared by reference generators and parity tests."""
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from serde import serde

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, PinholeParameters


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ProjectionFixture:
    """Independent reference projections of one calibration."""

    source: str
    """Reference implementation and version."""
    intrinsics: list[float]
    """Focal lengths and principal point: fx, fy, cx, cy."""
    distortion: list[float]
    """Coefficients in the reference model's documented order."""
    points: Float64[ndarray, "n 3"]
    """Camera-frame input points."""
    pixels: Float64[ndarray, "n 2"]
    """Reference output pixels."""



def fixture_camera(parameters: list[float], distortion: BrownConradyDistortion | KannalaBrandtDistortion) -> PinholeParameters | Fisheye62Parameters:
    """Build the identity-pose camera of a typed projection fixture."""
    fx, fy, cx, cy = parameters
    matrix: Float64[ndarray, "3 3"] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    intrinsics: Intrinsics = Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=matrix, height=480, width=640)
    extrinsics: Extrinsics = Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3))
    if isinstance(distortion, BrownConradyDistortion):
        return PinholeParameters(name="fixture", intrinsics=intrinsics, extrinsics=extrinsics, distortion=distortion)
    return Fisheye62Parameters(name="fixture", intrinsics=intrinsics, extrinsics=extrinsics, distortion=distortion)
