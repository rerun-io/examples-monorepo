"""Held-out cameras and ground-truth targets of a scene, from upstream's camera contract.

Upstream publishes one camera contract per NVS scene (HF `jhu-ccvl/Hexels`,
`cameras/nvs/<dataset>/<scene>/contract.json`): the 2DGS camera geometry of every training and test
view plus SHA256 digests of the exact float32 RGB targets. `cameras.json` (contract
`hexels-work/contracts/run-layout.md`) copies the test records, and `write_targets` reproduces the
targets bit-exactly (checked against the contract digest) as 16-bit PNGs for the shared scorer.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from serde import SerdeError, serde
from serde.json import from_json, to_json

from hexels_bench.png16 import write_png16


@serde
@dataclass(frozen=True, slots=True)
class ContractCamera:
    """One camera of upstream's contract `geometry` lists (2DGS conventions)."""

    name: str
    """Image stem."""
    width: int
    """Render width in pixels."""
    height: int
    """Render height in pixels."""
    fov_x: float
    """Horizontal field of view in radians."""
    fov_y: float
    """Vertical field of view in radians."""
    world_view_transform: list[list[float]]
    """4x4, row-vector convention: [x y z 1] @ M is the OpenCV camera point."""
    camera_center: list[float]
    """Camera center in world space."""


@serde
@dataclass(frozen=True, slots=True)
class ContractSettings:
    """How upstream loaded the scene."""

    images: str
    """Image subdirectory."""
    resolution: int
    """Downsampling factor."""
    llffhold: int
    """Hold-out stride."""


@serde
@dataclass(frozen=True, slots=True)
class ContractSplit:
    """Digests of one split."""

    count: int
    """Number of views."""
    ordered_target_tensors_sha256: str
    """SHA256 over [[name, tensor digest], ...] of the float32 RGB targets, in order."""


@serde
@dataclass(frozen=True, slots=True)
class ContractGeometry:
    """Camera records per split."""

    training: list[ContractCamera]
    """Training cameras in upstream order (sorted by image name)."""
    test: list[ContractCamera]
    """Held-out cameras in render order."""


@serde
@dataclass(frozen=True, slots=True)
class CameraContract:
    """Upstream camera contract (read partially; unknown fields are upstream's)."""

    settings: ContractSettings
    """Scene loading settings."""
    test: ContractSplit
    """Test split digests."""
    geometry: ContractGeometry
    """Camera geometry."""


def load_contract(path: Path) -> CameraContract:
    """Load an upstream camera contract.json."""
    try:
        return from_json(CameraContract, path.read_text())
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CamerasFile:
    """`cameras.json` written beside a run's renders (contract run-layout.md)."""

    format: str
    """Always hexels-cameras-1."""
    dataset: str
    """mipnerf360, dtu or tnt."""
    scene: str
    """Upstream scene name."""
    split: str
    """test for NVS renders; train or all for the surface cameras (`hexels-surface-run`)."""
    background: list[float]
    """RGB the upstream render composited over."""
    cameras: list[ContractCamera]
    """One record per held-out view, in render order."""


def cameras_file(contract: CameraContract, dataset: str, scene: str, background: list[float]) -> CamerasFile:
    """`cameras.json` content for a scene's held-out views."""
    return CamerasFile(format="hexels-cameras-1", dataset=dataset, scene=scene, split="test", background=background, cameras=contract.geometry.test)


def write_cameras_file(path: Path, cameras: CamerasFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(cameras) + "\n")


def load_cameras_file(path: Path) -> CamerasFile:
    try:
        return from_json(CamerasFile, path.read_text())
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def _json_sha256(value: list[list[str]]) -> str:
    text: str = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _tensor_sha256(target: Float32[ndarray, "3 h w"]) -> str:
    """Upstream's tensor digest: dtype string, shape JSON, then the raw bytes."""
    digest = hashlib.sha256()
    digest.update(target.dtype.str.encode())
    digest.update(json.dumps(list(target.shape), separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(target).tobytes())
    return digest.hexdigest()


def load_target(image_path: Path, width: int, height: int) -> UInt8[ndarray, "h w 3"]:
    """The RGB target 2DGS's loadCam builds: each of the first three bands resized on its own (PIL's
    default filter), so an RGBA image is not premultiplied; alpha is dropped."""
    with Image.open(image_path) as image:
        image.load()
        if image.mode == "RGB":
            return np.asarray(image.resize((width, height)), dtype=np.uint8)
        if image.mode != "RGBA":
            raise ValueError(f"{image_path}: expected RGB or RGBA, got {image.mode}")
        bands: tuple[Image.Image, ...] = image.split()
        resized: list[UInt8[ndarray, "h w"]] = [np.asarray(band.resize((width, height)), dtype=np.uint8) for band in bands[:3]]
    return np.stack(resized, axis=-1)


def find_image(images_dir: Path, stem: str) -> Path:
    matches: list[Path] = sorted(path for path in images_dir.iterdir() if path.stem == stem)
    if len(matches) != 1:
        raise FileNotFoundError(f"{images_dir}: expected one image named {stem}.*, found {len(matches)}")
    return matches[0]


def write_targets(contract: CameraContract, scene_dir: Path, out_dir: Path) -> list[Path]:
    """Write the exact float32 test targets as 16-bit PNGs (k/255 maps to k*257, lossless) and check
    them against the contract's ordered target digest. Returns the written paths in render order."""
    images_dir: Path = scene_dir / contract.settings.images
    digests: list[list[str]] = []
    written: list[Path] = []
    for camera in contract.geometry.test:
        rgb: UInt8[ndarray, "h w 3"] = load_target(find_image(images_dir, camera.name), camera.width, camera.height)
        chw: Float32[ndarray, "3 h w"] = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(255.0)).clip(0, 1).transpose(2, 0, 1))
        digests.append([camera.name, _tensor_sha256(chw)])
        path: Path = out_dir / f"{camera.name}.png"
        write_png16(path, rgb.astype(np.uint16) * np.uint16(257))
        written.append(path)
    if _json_sha256(digests) != contract.test.ordered_target_tensors_sha256:
        raise ValueError(f"{scene_dir}: test targets do not match the camera contract digest")
    return written
