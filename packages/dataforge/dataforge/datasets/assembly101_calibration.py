"""Session-resolved nimble calibration and the two shipped lens models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from serde import coerce, serde

from dataforge.datasets.assembly101_source import pose_path, read_record, read_transforms


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
    """First tangential coefficient; fisheye swaps the OpenCV convention."""
    p2: float
    """Second tangential coefficient."""

    p3: float = 0.0
    """Third tangential coefficient (zero in this release)."""
    p4: float = 0.0
    """Fourth tangential coefficient (zero in this release)."""

    @property
    def key(self) -> str:
        return self.SerialNo.replace("_", ":")

    @property
    def coefficients(self) -> list[float]:
        """Source order k1..k6,p1..p4, preserved as camera data."""
        return [self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.p1, self.p2, self.p3, self.p4]

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
class Calibration:
    """Resolved lenses and per-camera provenance; no cross-session borrowing."""

    lenses: dict[str, Lens]
    """Camera key to lens."""
    sources: dict[str, str]
    """Camera key to matching nimble sequence, or absent."""


def resolve_calibration(root: Path, fixed: dict[str, Float64[ndarray, "4 4"]], sequence: str | None = None) -> Calibration:
    """Match identical fixed extrinsics for exo; ego lenses are invariant by serial."""
    lenses: dict[str, Lens] = {}
    sources: dict[str, str] = {}
    for path in sorted((root / "assemblyhands-toolkit/calib/nimble_json_calib").glob("*.json")):
        records: list[NimbleRecord] = read_record(path, NimbleFile).data
        candidate_path: Path = pose_path(root, "camera_extrinsics_fixed", path.stem)
        matched: bool = False
        if fixed and candidate_path.is_file():
            candidate: dict[str, Float64[ndarray, "4 4"]] = fixed if path.stem == sequence else read_transforms(candidate_path)
            matched = candidate.keys() == fixed.keys() and all(
                np.allclose(candidate[key], value, rtol=0.0, atol=1e-3) for key, value in fixed.items()
            )
        for record in records:
            lens: Lens = record.Camera
            if lens.key not in lenses and (lens.DistortionModel == "OVFishEye62" or matched):
                lenses[lens.key] = lens
                sources[lens.key] = f"nimble:{path.stem}"
    return Calibration(lenses, sources)


def project(points: Float64[ndarray, "n 3"], lens: Lens, stored: tuple[int, int]) -> Float64[ndarray, "n 2"]:
    """Project camera-space Float64[n,3] points into stored pixels, including distortion."""
    xy: Float64[ndarray, "n 2"] = points[:, :2] / points[:, 2:]
    radius2: Float64[ndarray, "n"] = np.sum(xy * xy, axis=1)
    if lens.DistortionModel == "OVFishEye62":
        radius: Float64[ndarray, "n"] = np.sqrt(radius2)
        theta: Float64[ndarray, "n"] = np.arctan(radius)
        distorted: Float64[ndarray, "n"] = theta.copy()
        for index, coefficient in enumerate(lens.coefficients[:6], start=1):
            distorted += coefficient * theta ** (2 * index + 1)
        xy = xy * np.divide(distorted, radius, out=np.ones_like(radius), where=radius != 0)[:, None]
        radius2 = np.sum(xy * xy, axis=1)
        p1, p2 = lens.p2, lens.p1
        radial: Float64[ndarray, "n"] = np.ones_like(radius2)
    else:
        p1, p2 = lens.p1, lens.p2
        radial = 1.0 + lens.k1 * radius2 + lens.k2 * radius2**2 + lens.k3 * radius2**3
    x: Float64[ndarray, "n"] = xy[:, 0]
    y: Float64[ndarray, "n"] = xy[:, 1]
    uv: Float64[ndarray, "n 2"] = np.column_stack(
        (
            x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
            y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y,
        )
    )
    matrix: Float64[ndarray, "3 3"] = lens.matrix(stored)
    return uv * np.array([matrix[0, 0], matrix[1, 1]]) + matrix[:2, 2]


@serde
@dataclass(frozen=True, slots=True)
class NimbleFile:
    """Typed envelope for nimble's bare list."""

    data: list[NimbleRecord]
    """One record per camera."""
