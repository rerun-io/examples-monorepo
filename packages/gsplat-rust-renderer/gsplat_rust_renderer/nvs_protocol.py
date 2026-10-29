"""The Hexels NVS camera contract: the published per-scene cameras and split (HF ``jhu-ccvl/Hexels``,
``cameras/nvs/<dataset>/<scene>/contract.json``) and a JSON record reader that names malformed files.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json


@serde
@dataclass(frozen=True, slots=True)
class ContractSettings:
    """Upstream loader options recorded in the camera contract."""

    images: str
    """Image folder below the scene directory."""
    resolution: int
    """2DGS resolution divisor."""
    llffhold: int
    """Every llffhold-th name-sorted camera is held out."""


@serde
@dataclass(frozen=True, slots=True)
class ContractCamera:
    """One camera exactly as the upstream loader built it."""

    name: str
    """Image name without extension."""
    width: int
    """Target width in pixels."""
    height: int
    """Target height in pixels."""
    fov_x: float
    """Horizontal field of view in radians; the principal point is the image centre."""
    fov_y: float
    """Vertical field of view in radians."""
    world_view_transform: Float64[ndarray, "4 4"]
    """Transposed OpenCV world-to-camera matrix (2DGS row-vector convention)."""
    camera_center: Float64[ndarray, "3"]
    """Camera position in world coordinates."""

    def world_from_camera(self) -> Float64[ndarray, "4 4"]:
        """OpenCV (right/down/forward) camera-to-world pose."""
        return np.linalg.inv(self.world_view_transform.T)


@serde
@dataclass(frozen=True, slots=True)
class ContractSplit:
    """Summary of one split's inputs."""

    count: int
    """Number of cameras."""
    ordered_target_tensors_sha256: str
    """Hash of the ordered (name, float32 CHW target hash) pairs."""


@serde
@dataclass(frozen=True, slots=True)
class ContractGeometry:
    """Ordered camera records per split."""

    training: list[ContractCamera]
    """Training cameras in loader order."""
    test: list[ContractCamera]
    """Held-out cameras in loader order."""


@serde
@dataclass(frozen=True, slots=True)
class CameraContract:
    """The published per-scene camera contract (third-party document, read partially)."""

    settings: ContractSettings
    """Loader options that produced the cameras."""
    training: ContractSplit
    """Training split summary."""
    test: ContractSplit
    """Held-out split summary."""
    geometry: ContractGeometry
    """Camera records."""


def read_json[Record](cls: type[Record], path: Path) -> Record:
    """Decode a JSON record (camera contract, run record, report); malformed input names the file."""
    try:
        return from_json(cls, path.read_text())
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
