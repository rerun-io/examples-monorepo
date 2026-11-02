"""Brush inputs of the Hexels NVS protocol: the upstream cameras, split, target images and SfM initialization.

Upstream Hexels loads every scene through the 2DGS COLMAP reader (hbb1/2d-gaussian-splatting@335ad612):
cameras sorted by image name, every eighth held out from index 0, field of view from COLMAP fx/fy with the
principal point at the image centre, images resized with PIL to ``round(size / resolution)`` and alpha kept
out of the RGB target. The published camera contract (`hexels_bench.cameras`) records the resulting
cameras; this module turns it into a NeRF-style directory that Brush reads with exactly those cameras and images.
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import PIL
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from PIL import Image
from serde import serde
from serde.json import to_json

from hexels_bench.cameras import (
    CameraContract,
    ContractCamera,
    ContractSettings,
    SfmPoints,
    find_image,
    load_target,
    ordered_targets_sha256,
    read_sfm_points,
    target_sha256,
)
from hexels_bench.layout import read_json


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
    """Training images reproduce the contract's target-tensor hash (preparation refuses a mismatch; reuse checks it)."""
    test_targets_match: bool
    """Held-out images reproduce the contract's target-tensor hash (preparation refuses a mismatch; reuse checks it)."""
    pillow: str
    """Pillow version that decoded and resized the images."""


def target_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """2DGS ``loadCam`` target size for an explicit resolution divisor (Python rounding, half to even)."""
    if resolution not in (1, 2, 4, 8):
        raise ValueError(f"2DGS resolution must be 1, 2, 4 or 8, got {resolution}")
    return round(width / resolution), round(height / resolution)


def opengl_world_from_camera(world_view_transform: Float64[ndarray, "4 4"]) -> Float64[ndarray, "4 4"]:
    """Convert upstream's transposed OpenCV world-to-camera matrix to an OpenGL camera-to-world matrix."""
    world_from_camera: Float64[ndarray, "4 4"] = np.linalg.inv(world_view_transform.T)
    world_from_camera[:3, 1:3] *= -1.0
    return world_from_camera


def init_ply(points: SfmPoints) -> bytes:
    """Binary PLY with float colors in [0, 1], so Brush's ``rgb_to_sh`` sees rgb / 255 as its COLMAP loader does."""
    values: Float32[ndarray, "n 6"] = np.concatenate([points.xyz, points.rgb.astype(np.float32) / np.float32(255.0)], axis=1).astype("<f4")
    names: tuple[str, ...] = ("x", "y", "z", "red", "green", "blue")
    header: str = f"ply\nformat binary_little_endian 1.0\ncomment SfM points ({points.source})\nelement vertex {len(values)}\n"
    return (header + "".join(f"property float {name}\n" for name in names) + "end_header\n").encode() + values.tobytes()


def prepare_scene(source: Path, contract_path: Path, output: Path) -> PreparedScene:
    """Write ``transforms_{train,test}.json``, ``{train,test}/<name>.png`` and ``init.ply`` for one scene.

    The contract fixes the cameras and the split; the split must also equal the lexical every-8th rule.
    Images are the upstream target pixels stored losslessly, and both splits must reproduce the contract's
    target digests, so ``test/`` is also the ground truth for scoring. A finished ``prepare.json`` is reused.

    Raises:
        ValueError: The split, a target size or a target digest differs from the contract (nothing is written),
            or a reused record says its targets did not match.
    """
    record_path: Path = output / "prepare.json"
    if record_path.is_file():
        reused: PreparedScene = read_json(PreparedScene, record_path)
        if not (reused.train_targets_match and reused.test_targets_match):
            raise ValueError(f"{output}: target images differ from the contract's target-tensor hashes; remove it and prepare again")
        return reused
    contract: CameraContract = read_json(CameraContract, contract_path)
    settings: ContractSettings = contract.settings
    splits: dict[str, list[ContractCamera]] = {"train": contract.geometry.training, "test": contract.geometry.test}
    names: list[str] = sorted(camera.name for cameras in splits.values() for camera in cameras)
    held_out: list[str] = names[:: settings.llffhold]
    if [camera.name for camera in splits["test"]] != held_out:
        raise ValueError(f"{contract_path}: held-out cameras are not every {settings.llffhold}th name-sorted camera from index 0")
    folder: Path = source / settings.images

    def write_target(camera: ContractCamera, images: Path) -> str:
        path: Path = find_image(folder, camera.name)
        with Image.open(path) as image:
            size: tuple[int, int] = target_size(image.width, image.height, settings.resolution)
        if size != (camera.width, camera.height):
            raise ValueError(f"{path}: target is {size[0]}x{size[1]}, contract says {camera.width}x{camera.height}")
        rgb: UInt8[ndarray, "h w 3"] = load_target(path, camera.width, camera.height)
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
            matches[split] = ordered_targets_sha256([[camera.name, digest] for camera, digest in zip(cameras, hashes, strict=True)]) == expected
            if not matches[split]:
                raise ValueError(f"{source}: {split} target images do not match the camera contract digest (decoder or resize differs)")
            frames: list[BrushFrame] = [
                BrushFrame(
                    file_path=f"{split}/{camera.name}.png",
                    transform_matrix=opengl_world_from_camera(camera.world_view_transform),
                    camera_angle_x=camera.fov_x,
                    camera_angle_y=camera.fov_y,
                )
                for camera in cameras
            ]
            (staging / f"transforms_{split}.json").write_text(
                to_json(BrushTransforms(camera_angle_x=frames[0].camera_angle_x, ply_file_path="init.ply", frames=frames))
            )
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
    return record
