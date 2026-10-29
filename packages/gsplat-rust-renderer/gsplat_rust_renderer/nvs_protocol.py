"""Hexels NVS protocol inputs for Brush: upstream cameras, split, target images and SfM initialization.

Upstream Hexels loads every scene through the 2DGS COLMAP reader (hbb1/2d-gaussian-splatting@335ad612):
cameras sorted by image name, every eighth held out from index 0, field of view from COLMAP fx/fy with the
principal point at the image centre, images resized with PIL to ``round(size / resolution)`` and alpha kept
out of the RGB target. The published camera contract (HF ``jhu-ccvl/Hexels``,
``cameras/nvs/<dataset>/<scene>/contract.json``) records the resulting cameras; this module turns it into a
NeRF-style directory that Brush reads with exactly those cameras and images.
"""

import hashlib
import json
import os
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal, TypeAlias

import numpy as np
import PIL
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from PIL import Image
from plyfile import PlyData
from serde import SerdeError, serde
from serde.json import from_json, to_json

Dataset: TypeAlias = Literal["mipnerf360", "dtu", "tnt"]
UPSTREAM_DATASETS: dict[Dataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tanks_and_temples"}
"""Dataset names in the upstream camera contract tree."""
DATA_DIRS: dict[Dataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tnt_nvs"}
"""Scene roots below the dataset directory (upstream configs/nvs.json dataset_roots)."""
PROTOCOL_SCENES: dict[Dataset, tuple[str, ...]] = {
    "mipnerf360": ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"),
    "dtu": ("scan24", "scan37", "scan40", "scan55", "scan63", "scan65", "scan69", "scan83", "scan97", "scan105", "scan106", "scan110", "scan114", "scan118", "scan122"),
    "tnt": ("Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck", "Train", "Francis", "Horse", "Lighthouse"),
}
"""The 34 NVS scenes of upstream configs/nvs.json."""


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


@serde
@dataclass(frozen=True, slots=True)
class BrushFrame:
    """One NeRF-style frame with per-frame fields of view; Brush centres the principal point when cx/cy are absent."""

    file_path: str
    """Image path relative to the transforms file."""
    transform_matrix: Float64[ndarray, "4 4"]
    """OpenGL camera-to-world matrix."""
    camera_angle_x: float
    """Horizontal field of view in radians."""
    camera_angle_y: float
    """Vertical field of view in radians."""


@serde
@dataclass(frozen=True, slots=True)
class BrushTransforms:
    """One split as Brush's nerfstudio loader reads it."""

    camera_angle_x: float
    """Scene-level field of view for readers that need one; Brush prefers the per-frame values."""
    ply_file_path: str
    """Initialization cloud relative to this file."""
    frames: list[BrushFrame]
    """Frames in contract order."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PreparedScene:
    """Record of one prepared scene; written last, so its presence marks completion."""

    source: Path
    """Scene directory the images and SfM points came from."""
    contract: Path
    """Camera contract that defined cameras and split."""
    images: str
    """Source image folder."""
    resolution: int
    """2DGS resolution divisor applied to the source images."""
    train_views: int
    """Training camera count."""
    test_views: int
    """Held-out camera count."""
    sfm_points: int
    """SfM points in the initialization cloud (the primitive cap)."""
    init_source: str
    """SfM file the cloud was read from."""
    train_targets_match: bool
    """Training images reproduce the contract's target-tensor hash."""
    test_targets_match: bool
    """Held-out images reproduce the contract's target-tensor hash."""
    pillow: str
    """Pillow version that decoded and resized the images."""


@dataclass(frozen=True, slots=True)
class SfmPoints:
    """The COLMAP point cloud upstream initializes from."""

    xyz: Float32[ndarray, "n 3"]
    """Point positions."""
    rgb: UInt8[ndarray, "n 3"]
    """Point colors."""
    source: str
    """File name the points came from."""


def read_json[Record](cls: type[Record], path: Path) -> Record:
    """Decode a JSON record (camera contract, run record, report); malformed input names the file."""
    try:
        return from_json(cls, path.read_text())
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def target_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """2DGS ``loadCam`` target size for an explicit resolution divisor (Python rounding, half to even)."""
    if resolution not in (1, 2, 4, 8):
        raise ValueError(f"2DGS resolution must be 1, 2, 4 or 8, got {resolution}")
    return round(width / resolution), round(height / resolution)


def load_target(path: Path, resolution: int) -> UInt8[ndarray, "h w 3"]:
    """Decode and resize like 2DGS ``loadCam``: RGBA is resized band by band and alpha is dropped, not composited."""
    with Image.open(path) as image:
        bands: tuple[Image.Image, ...] = image.split()
        size: tuple[int, int] = target_size(image.width, image.height, resolution)
        if len(bands) == 3:
            return np.asarray(image.resize(size))
        if len(bands) != 4:
            raise ValueError(f"{path}: expected an RGB or RGBA image, got mode {image.mode}")
        return np.stack([np.asarray(band.resize(size)) for band in bands[:3]], axis=-1)


def target_sha256(rgb: UInt8[ndarray, "h w 3"]) -> str:
    """Hash of upstream's float32 CHW target tensor (hexels/benchmark_metrics.py ``_tensor_sha256``)."""
    target: Float32[ndarray, "3 h w"] = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1))
    digest = hashlib.sha256()
    digest.update(target.dtype.str.encode())
    digest.update(json.dumps(target.shape, separators=(",", ":")).encode())
    digest.update(target.tobytes())
    return digest.hexdigest()


def ordered_targets_sha256(pairs: list[tuple[str, str]]) -> str:
    """Hash of the ordered (name, target hash) list (hexels/benchmark_metrics.py ``_json_sha256``)."""
    payload: bytes = json.dumps([list(pair) for pair in pairs], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def opengl_world_from_camera(world_view_transform: Float64[ndarray, "4 4"]) -> Float64[ndarray, "4 4"]:
    """Convert upstream's transposed OpenCV world-to-camera matrix to an OpenGL camera-to-world matrix."""
    world_from_camera: Float64[ndarray, "4 4"] = np.linalg.inv(world_view_transform.T)
    world_from_camera[:3, 1:3] *= -1.0
    return world_from_camera


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


def init_ply(points: SfmPoints) -> bytes:
    """Binary PLY with float colors in [0, 1], so Brush's ``rgb_to_sh`` sees rgb / 255 as its COLMAP loader does."""
    values: Float32[ndarray, "n 6"] = np.concatenate([points.xyz, points.rgb.astype(np.float32) / np.float32(255.0)], axis=1).astype("<f4")
    names: tuple[str, ...] = ("x", "y", "z", "red", "green", "blue")
    header: str = f"ply\nformat binary_little_endian 1.0\ncomment SfM points ({points.source})\nelement vertex {len(values)}\n"
    return (header + "".join(f"property float {name}\n" for name in names) + "end_header\n").encode() + values.tobytes()


def find_image(folder: Path, name: str) -> Path:
    """The single file in ``folder`` whose stem is the camera name."""
    matches: list[Path] = [path for path in folder.glob(f"{name}.*") if path.stem == name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image named {name} in {folder}, found {len(matches)}")
    return matches[0]


def prepare_scene(source: Path, contract_path: Path, output: Path) -> PreparedScene:
    """Write ``transforms_{train,test}.json``, ``{train,test}/<name>.png`` and ``init.ply`` for one scene.

    The contract fixes the cameras and the split; the split must also equal the lexical every-8th rule.
    Images are the upstream target pixels stored losslessly, so ``test/`` is also the ground truth for scoring.
    A finished ``prepare.json`` is reused as is.
    """
    record_path: Path = output / "prepare.json"
    if record_path.is_file():
        return read_json(PreparedScene, record_path)
    contract: CameraContract = read_json(CameraContract, contract_path)
    settings: ContractSettings = contract.settings
    splits: dict[str, list[ContractCamera]] = {"train": contract.geometry.training, "test": contract.geometry.test}
    names: list[str] = sorted(camera.name for cameras in splits.values() for camera in cameras)
    held_out: list[str] = names[:: settings.llffhold]
    if [camera.name for camera in splits["test"]] != held_out:
        raise ValueError(f"{contract_path}: held-out cameras are not every {settings.llffhold}th name-sorted camera from index 0")
    folder: Path = source / settings.images

    def write_target(camera: ContractCamera, images: Path) -> str:
        rgb: UInt8[ndarray, "h w 3"] = load_target(find_image(folder, camera.name), settings.resolution)
        if rgb.shape != (camera.height, camera.width, 3):
            raise ValueError(f"{folder / camera.name}: target is {rgb.shape[1]}x{rgb.shape[0]}, contract says {camera.width}x{camera.height}")
        Image.fromarray(rgb, "RGB").save(images / f"{camera.name}.png", compress_level=1)
        return target_sha256(rgb)

    points: SfmPoints = read_sfm_points(source / "sparse/0")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Path = Path(mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        matches: dict[str, bool] = {}
        for split, cameras in splits.items():
            (staging / split).mkdir()
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
                hashes: list[str] = list(pool.map(write_target, cameras, [staging / split] * len(cameras)))
            expected: str = (contract.training if split == "train" else contract.test).ordered_target_tensors_sha256
            matches[split] = ordered_targets_sha256([(camera.name, digest) for camera, digest in zip(cameras, hashes, strict=True)]) == expected
            frames: list[BrushFrame] = [
                BrushFrame(
                    file_path=f"{split}/{camera.name}.png",
                    transform_matrix=opengl_world_from_camera(camera.world_view_transform),
                    camera_angle_x=camera.fov_x,
                    camera_angle_y=camera.fov_y,
                )
                for camera in cameras
            ]
            (staging / f"transforms_{split}.json").write_text(to_json(BrushTransforms(camera_angle_x=frames[0].camera_angle_x, ply_file_path="init.ply", frames=frames)))
        (staging / "init.ply").write_bytes(init_ply(points))
        record: PreparedScene = PreparedScene(
            source=source.resolve(),
            contract=contract_path.resolve(),
            images=settings.images,
            resolution=settings.resolution,
            train_views=len(splits["train"]),
            test_views=len(splits["test"]),
            sfm_points=len(points.xyz),
            init_source=points.source,
            train_targets_match=matches["train"],
            test_targets_match=matches["test"],
            pillow=PIL.__version__,
        )
        (staging / "prepare.json").write_text(to_json(record))
        staging.rename(output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not (record.train_targets_match and record.test_targets_match):
        print(f"Warning: {output} target images differ from the contract's target-tensor hashes (decoder or resize differs)")
    return record
