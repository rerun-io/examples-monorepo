"""Derive Brush training inputs with the standard seeded 3DGS cube initialization."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial import KDTree
from serde import serde
from serde.json import from_json, to_json

from gsplat_rust_renderer.nerfbaselines import DATA_ROOT, download_and_extract


@serde
@dataclass(frozen=True, slots=True)
class Frame:
    """NeRF-synthetic camera frame."""

    file_path: str
    rotation: float
    transform_matrix: list[list[float]]


@serde
@dataclass(frozen=True, slots=True)
class Transforms:
    """NeRF-synthetic camera document with an optional Brush initialization."""

    camera_angle_x: float
    frames: list[Frame]
    ply_file_path: str = "points3d.ply"


@dataclass(frozen=True, slots=True)
class Config:
    """Download a scene if needed and derive its training directory."""

    scene: Literal["lego", "hotdog", "chair", "drums", "ficus", "materials", "mic", "ship"] = "lego"
    source_root: Path = DATA_ROOT / "data"
    output_root: Path = field(default_factory=lambda: Path(os.environ.get("GSPLAT_NERF_INIT_ROOT", str(DATA_ROOT.parent / "nerf-synthetic-init"))))


def initialization_ply() -> bytes:
    """Produce the shared seed-42, 100k-point cube byte for byte."""
    rng: np.random.Generator = np.random.default_rng(42)
    xyz: Float32[ndarray, "n 3"] = rng.uniform(-1.3, 1.3, (100_000, 3)).astype(np.float32)
    rgb: Float64[ndarray, "n 3"] = rng.random((100_000, 3))
    distances, _ = KDTree(xyz).query(xyz, k=4)
    scales: Float64[ndarray, "n"] = np.log(np.sqrt(np.maximum(np.mean(distances[:, 1:] ** 2, axis=1), 1e-7)))
    values: Float32[ndarray, "n 14"] = np.zeros((100_000, 14), dtype="<f4")
    values[:, :3] = xyz
    values[:, 3:6] = scales[:, None]
    values[:, 6] = np.log(0.1 / 0.9)
    values[:, 7] = 1.0
    values[:, 11:14] = (rgb - 0.5) / 0.28209479177387814
    names: tuple[str, ...] = ("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header: str = "ply\nformat binary_little_endian 1.0\ncomment numpy default_rng(42); uniform cube; three-neighbor scales\nelement vertex 100000\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    return header.encode() + values.tobytes()


def main(config: Config) -> None:
    """Keep source images unchanged; publish camera metadata and deterministic PLY."""
    source: Path = config.source_root / config.scene
    output: Path = config.output_root / config.scene
    if (output / "transforms_train.json").is_file() and (output / "points3d.ply").is_file():
        return
    if not source.exists():
        source = download_and_extract("data", config.scene, root=config.source_root.parent)
    transforms: Transforms = from_json(Transforms, (source / "transforms_train.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test", "transforms_val.json", "transforms_test.json"):
        destination: Path = output / name
        if not destination.exists():
            destination.symlink_to(os.path.relpath(source / name, output), target_is_directory=(source / name).is_dir())
    (output / "points3d.ply").write_bytes(initialization_ply())
    (output / "transforms_train.json").write_text(to_json(transforms))
    print(f"Initialized {config.scene}: {output}")
