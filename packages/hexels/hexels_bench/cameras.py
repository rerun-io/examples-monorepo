"""The protocol inputs of a scene: held-out cameras and ground-truth targets from upstream's camera contract, and
the SfM points both trainers initialize from (and cap their primitive count at).

Upstream publishes one camera contract per NVS scene (HF `jhu-ccvl/Hexels`,
`cameras/nvs/<dataset>/<scene>/contract.json`): the 2DGS camera geometry of every training and test
view plus SHA256 digests of the exact float32 RGB targets. `cameras.json` (contract
`hexels-work/contracts/run-layout.md`) copies the test records, and `write_targets` reproduces the
targets bit-exactly (checked against the contract digest) as 16-bit PNGs for the shared scorer.
"""

import hashlib
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from PIL import Image
from plyfile import PlyData
from serde import serde
from serde.json import to_json

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
    world_view_transform: Float64[ndarray, "4 4"]
    """Row-vector convention: [x y z 1] @ M is the OpenCV camera point (M = world-to-camera transposed)."""
    camera_center: Float64[ndarray, "3"]
    """Camera center in world space."""

    def world_from_camera(self) -> Float64[ndarray, "4 4"]:
        """OpenCV (right/down/forward) camera-to-world pose."""
        return np.linalg.inv(self.world_view_transform.T)


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
    training: ContractSplit
    """Training split digests."""
    test: ContractSplit
    """Test split digests."""
    geometry: ContractGeometry
    """Camera geometry."""


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


def ordered_targets_sha256(digests: list[list[str]]) -> str:
    """Upstream's digest of the ordered [[name, target digest], ...] list (hexels/benchmark_metrics.py `_json_sha256`)."""
    text: str = json.dumps(digests, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def target_sha256(rgb: UInt8[ndarray, "h w 3"]) -> str:
    """Upstream's digest of a float32 CHW target tensor (`_tensor_sha256`): dtype string, shape JSON, then the raw bytes."""
    target: Float32[ndarray, "3 h w"] = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1))
    digest = hashlib.sha256()
    digest.update(target.dtype.str.encode())
    digest.update(json.dumps(list(target.shape), separators=(",", ":")).encode())
    digest.update(target.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SfmPoints:
    """The COLMAP point cloud upstream initializes from."""

    xyz: Float32[ndarray, "n 3"]
    """Point positions."""
    rgb: UInt8[ndarray, "n 3"]
    """Point colors."""
    source: str
    """File name the points came from."""


def read_sfm_points(sparse: Path) -> SfmPoints:
    """Read the SfM cloud with 2DGS precedence: points3D.ply, then points3D.bin, then points3D.txt."""
    if (sparse / "points3D.ply").is_file():
        vertex = PlyData.read(str(sparse / "points3D.ply"))["vertex"]
        xyz: Float32[ndarray, "n 3"] = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        rgb: UInt8[ndarray, "n 3"] = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
        return SfmPoints(xyz=xyz, rgb=rgb, source="points3D.ply")
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    if (sparse / "points3D.bin").is_file():
        data: bytes = (sparse / "points3D.bin").read_bytes()
        offset: int = 8
        for _ in range(struct.unpack_from("<Q", data)[0]):
            record: tuple = struct.unpack_from("<Q3d3BdQ", data, offset)
            positions.append(record[1:4])
            colors.append(record[4:7])
            offset += struct.calcsize("<Q3d3BdQ") + 8 * record[8]
        source: str = "points3D.bin"
    elif (sparse / "points3D.txt").is_file():
        for line in (sparse / "points3D.txt").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                fields: list[str] = line.split()
                positions.append((float(fields[1]), float(fields[2]), float(fields[3])))
                colors.append((int(fields[4]), int(fields[5]), int(fields[6])))
        source = "points3D.txt"
    else:
        raise FileNotFoundError(f"No SfM points (points3D.ply/.bin/.txt) in {sparse}")
    return SfmPoints(xyz=np.array(positions, dtype=np.float32).reshape(-1, 3), rgb=np.array(colors, dtype=np.uint8).reshape(-1, 3), source=source)


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
    """Write the exact float32 test targets as 16-bit PNGs (k/255 maps to k*257, lossless) into `out_dir`.

    The PNGs are staged beside `out_dir` and replace it only once they match the contract's ordered
    target digest, so a directory of targets is always a checked one. Returns the paths in render order.

    Raises:
        ValueError: The targets do not match the digest; `out_dir` is left as it was.
    """
    images_dir: Path = scene_dir / contract.settings.images
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging: Path = Path(mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        digests: list[list[str]] = []
        for camera in contract.geometry.test:
            rgb: UInt8[ndarray, "h w 3"] = load_target(find_image(images_dir, camera.name), camera.width, camera.height)
            digests.append([camera.name, target_sha256(rgb)])
            write_png16(staging / f"{camera.name}.png", rgb.astype(np.uint16) * np.uint16(257))
        if ordered_targets_sha256(digests) != contract.test.ordered_target_tensors_sha256:
            raise ValueError(f"{scene_dir}: test targets do not match the camera contract digest")
        if out_dir.exists():
            shutil.rmtree(out_dir)  # incomplete targets of an earlier writer
        staging.rename(out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return [out_dir / f"{camera.name}.png" for camera in contract.geometry.test]
