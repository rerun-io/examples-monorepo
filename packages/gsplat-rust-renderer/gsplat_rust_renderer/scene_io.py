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
from scipy.spatial.transform import Rotation
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


def qvec_to_rotmat(q: tuple[float, float, float, float]) -> Float64[ndarray, "3 3"]:
    """COLMAP quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = q
    xyzw: Float64[ndarray, "4"] = np.asarray([x, y, z, w], dtype=np.float64)  # COLMAP wxyz -> scipy xyzw
    return np.asarray(Rotation.from_quat(xyzw).as_matrix(), dtype=np.float64)


def read_colmap_cameras_bin(path: Path) -> dict[int, dict]:
    """Parse a COLMAP ``cameras.bin`` into ``{camera_id: {model, width, height, params}}``."""
    models: dict[int, tuple[str, int]] = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12),
    }
    cameras: dict[int, dict] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            model_name, n_params = models[model_id]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {"model": model_name, "width": w, "height": h, "params": params}
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


def colmap_sparse_dir(scene_dir: Path) -> Path | None:
    """Return the COLMAP ``sparse/0`` dir (nerfstudio nests it under ``colmap/``),
    or None if this isn't a COLMAP capture."""
    for candidate in (scene_dir / "colmap" / "sparse" / "0", scene_dir / "sparse" / "0"):
        if (candidate / "cameras.bin").exists():
            return candidate
    return None


def colmap_image_path(scene_dir: Path, name: str, subdirs: tuple[str, ...]) -> Path:
    """Resolve an image for a COLMAP view, trying ``subdirs`` in order — a
    resolution-preference ladder over the nerfstudio downscales, e.g.
    ``("images_8", "images_4", "images_2", "images")`` for a small frustum
    thumbnail or ``("images_4", "images_2", "images")`` for a medium-res GT panel.
    Falls back to the full-res ``images/`` copy when none of the laddered dirs exist."""
    for sub in subdirs:
        p: Path = scene_dir / sub / name
        if p.exists():
            return p
    return scene_dir / "images" / name


def _colmap_fx_fy_cx_cy(model: str, params: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Pull (fx, fy, cx, cy) out of a COLMAP camera's params per its model layout.
    Single-focal models store ``f, cx, cy, ...`` (one focal) — NOT ``fx, fy, cx, cy``;
    every other COLMAP model leads with ``fx, fy, cx, cy``."""
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        return params[0], params[1], params[2], params[3]
    raise ValueError(f"unsupported COLMAP camera model: {model}")


def load_colmap_cameras(
    scene_dir: Path, image_subdirs: tuple[str, ...] = ("images", "images_2", "images_4", "images_8")
) -> list[tuple[PinholeParameters, Path]]:
    """Read a COLMAP sparse model as (pinhole, image path) pairs, sorted by image
    name (the order brush registers them, so eval-split selection lines up). COLMAP
    poses are world-to-cam in RDF — used unmodified. Distortion is dropped for the
    pinhole frustum. ``image_subdirs`` is the resolution-preference ladder for the
    returned image path (full-res first by default; pass a thumbnail-first ladder
    for small frusta). Intrinsics are rescaled to whichever file is resolved.
    """
    sparse: Path | None = colmap_sparse_dir(scene_dir)
    if sparse is None:
        raise FileNotFoundError(f"{scene_dir}: no COLMAP sparse model found")
    calibrations = read_colmap_cameras_bin(sparse / "cameras.bin")
    poses = sorted(read_colmap_images_bin(sparse / "images.bin"), key=lambda im: im["name"])
    cameras: list[tuple[PinholeParameters, Path]] = []
    for im in poses:
        calib = calibrations[im["camera_id"]]
        fx, fy, cx, cy = _colmap_fx_fy_cx_cy(calib["model"], calib["params"])
        image_path: Path = colmap_image_path(scene_dir, im["name"], image_subdirs)
        with Image.open(image_path) as probe:
            width, height = probe.size
        sx: float = width / calib["width"]
        sy: float = height / calib["height"]
        camera = PinholeParameters(
            name=Path(im["name"]).stem,
            extrinsics=Extrinsics(cam_R_world=qvec_to_rotmat(im["qvec"]), cam_t_world=np.asarray(im["tvec"], dtype=np.float64)),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RDF", fl_x=fx * sx, fl_y=fy * sy, cx=cx * sx, cy=cy * sy, height=height, width=width
            ),
        )
        cameras.append((camera, image_path))
    return cameras
