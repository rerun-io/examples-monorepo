"""Session-resolved nimble calibration and the two shipped lens models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from serde import coerce, serde
from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, PinholeParameters

from dataforge.datasets.assembly101_source import pose_path, read_transforms
from dataforge.records import read_json


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Lens:
    """Each camera's own nimble record, in source pixels."""

    DistortionModel: Literal["OpenCV", "OVFishEye62"]
    """Projection convention."""
    ImageSizeX: int
    """Source width."""
    ImageSizeY: int
    """Source height."""
    SerialNo: str
    """Serial with modality suffix."""
    fx: float
    """Horizontal focal length."""
    fy: float
    """Vertical focal length."""
    cx: float
    """Principal x."""
    cy: float
    """Principal y."""
    k1: float
    """First radial coefficient."""
    k2: float
    """Second radial coefficient."""
    k3: float
    """Third radial coefficient."""
    k4: float
    """Fourth radial coefficient (fisheye)."""
    k5: float
    """Fifth radial coefficient (fisheye)."""
    k6: float
    """Sixth radial coefficient (fisheye)."""
    p1: float
    """First tangential coefficient, passed to simplecv unchanged."""
    p2: float
    """Second tangential coefficient."""

    @property
    def key(self) -> str:
        return self.SerialNo.replace("_", ":")

    def matrix(self, stored: tuple[int, int]) -> Float64[ndarray, "3 3"]:
        """Scale K from source to stored pixels exactly once."""
        matrix: Float64[ndarray, "3 3"] = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        matrix[0] *= stored[0] / self.ImageSizeX
        matrix[1] *= stored[1] / self.ImageSizeY
        return matrix


@serde
@dataclass(frozen=True, slots=True)
class NimbleRecord:
    """Ignore unrelated toolkit fields at the third-party boundary."""

    Camera: Lens
    """The camera lens record."""


@dataclass(frozen=True, slots=True)
class ResolvedLens:
    """One camera's lens and calibration provenance."""

    lens: Lens
    """Shipped nimble parameters."""
    source: str
    """Matching nimble sequence name."""


def resolve_calibration(root: Path, fixed: dict[str, Float64[ndarray, "4 4"]]) -> dict[str, ResolvedLens]:
    """Match identical fixed extrinsics for exo; ego lenses are invariant by serial."""
    resolved: dict[str, ResolvedLens] = {}
    for path in sorted((root / "assemblyhands-toolkit/calib/nimble_json_calib").glob("*.json")):
        records: list[NimbleRecord] = read_json(path, list[NimbleRecord])
        candidate_path: Path = pose_path(root, "camera_extrinsics_fixed", path.stem)
        if not candidate_path.is_file():
            raise ValueError(f"Assembly101 calibration donor missing: {candidate_path}")
        candidate: dict[str, Float64[ndarray, "4 4"]] = read_transforms(candidate_path)
        matched: bool = candidate.keys() == fixed.keys() and all(
            np.allclose(candidate[key], value, rtol=0.0, atol=1e-3) for key, value in fixed.items()
        )
        for record in records:
            lens: Lens = record.Camera
            if lens.key not in resolved and (lens.DistortionModel == "OVFishEye62" or matched):
                resolved[lens.key] = ResolvedLens(lens, f"nimble:{path.stem}")
    return resolved


def camera_parameters(lens: Lens, transform: Float64[ndarray, "4 4"], stored: tuple[int, int]) -> PinholeParameters | Fisheye62Parameters:
    """Build a camera with Float64[4,4] parent-from-camera metres and stored pixels.

    Nimble distortion coefficients enter simplecv's models unchanged.
    """
    intrinsics: Intrinsics = Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=lens.matrix(stored), width=stored[0], height=stored[1])
    extrinsics: Extrinsics = Extrinsics(world_R_cam=transform[:3, :3], world_t_cam=transform[:3, 3])
    if lens.DistortionModel == "OpenCV":
        return PinholeParameters(
            lens.SerialNo, extrinsics, intrinsics, distortion=BrownConradyDistortion(lens.k1, lens.k2, lens.p1, lens.p2, lens.k3)
        )
    return Fisheye62Parameters(
        lens.SerialNo,
        extrinsics,
        intrinsics,
        distortion=KannalaBrandtDistortion(lens.k1, lens.k2, lens.k3, lens.k4, lens.k5, lens.k6, lens.p1, lens.p2),
    )
