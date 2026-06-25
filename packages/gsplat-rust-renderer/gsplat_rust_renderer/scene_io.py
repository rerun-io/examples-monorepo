"""Shared scene I/O for the gsplat CLIs: NeRF-synthetic + COLMAP camera loaders
and GT-image compositing.

Imported by both ``apis/visualize_brush_training.py`` and
``apis/log_splats_with_cameras.py`` — kept in the package (not ``tools/``) so
``beartype_this_package()`` instruments it in dev. pycolmap is not in this env;
the COLMAP binary layout is stable and these parsers avoid a heavyweight dep.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters


def load_nerf_cameras(scene_dir: Path, split: Literal["train", "val", "test"]) -> list[tuple[PinholeParameters, Path]]:
    """Read NeRF-synthetic cameras of one split as (pinhole, image path) pairs.

    The c2w matrices are OpenGL/RUB; simplecv carries the convention through to
    the Pinhole's ``camera_xyz``, so they are used unmodified.
    """
    transforms = json.loads((scene_dir / f"transforms_{split}.json").read_text())
    camera_angle_x: float = transforms["camera_angle_x"]
    cameras: list[tuple[PinholeParameters, Path]] = []
    for frame in transforms["frames"]:
        image_path: Path = (scene_dir / frame["file_path"]).with_suffix(".png")
        with Image.open(image_path) as probe:
            width, height = probe.size
        focal: float = 0.5 * width / np.tan(0.5 * camera_angle_x)
        c2w: Float64[ndarray, "4 4"] = np.array(frame["transform_matrix"], dtype=np.float64)
        camera = PinholeParameters(
            name=image_path.stem,
            extrinsics=Extrinsics(world_R_cam=c2w[:3, :3], world_t_cam=c2w[:3, 3]),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RUB",
                fl_x=focal,
                fl_y=focal,
                cx=width / 2.0,
                cy=height / 2.0,
                height=height,
                width=width,
            ),
        )
        cameras.append((camera, image_path))
    return cameras


def load_rgb_composited(image_path: Path, background: float) -> UInt8[ndarray, "h w 3"]:
    """Load an image as raw RGB with alpha composited onto a constant background.

    Raw ``rr.Image`` on purpose — ``rr.EncodedImage`` makes the viewer re-decode
    every visible image every frame (decode-cache misses).
    """
    with Image.open(image_path) as img:
        rgba: UInt8[ndarray, "h w c"] = np.asarray(img.convert("RGBA"))
    alpha: Float64[ndarray, "h w 1"] = rgba[..., 3:].astype(np.float64) / 255.0
    rgb: UInt8[ndarray, "h w 3"] = (rgba[..., :3].astype(np.float64) * alpha + background * (1.0 - alpha)).astype(np.uint8)
    return rgb


def read_colmap_cameras_bin(path: Path) -> dict[int, dict]:
    """Parse a COLMAP ``cameras.bin`` into ``{camera_id: {width, height, params}}``."""
    models: dict[int, tuple[str, int]] = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
    }
    cameras: dict[int, dict] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            _name, n_params = models[model_id]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {"width": w, "height": h, "params": params}
    return cameras


def read_colmap_images_bin(path: Path) -> list[dict]:
    """Parse a COLMAP ``images.bin`` into a list of ``{qvec, tvec, camera_id, name}``."""
    images: list[dict] = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)  # image id
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)  # skip 2D points
            images.append({"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name.decode()})
    return images


def qvec_to_rotmat(q: tuple[float, float, float, float]) -> Float64[ndarray, "3 3"]:
    """COLMAP quaternion (wxyz) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
