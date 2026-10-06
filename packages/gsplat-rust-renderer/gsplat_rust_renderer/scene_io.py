"""NeRF camera loading and image compositing for the scene logger."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray
from PIL import Image
from serde import SerdeError, serde
from serde.json import from_json
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters


@serde
@dataclass(frozen=True, slots=True)
class NerfFrame:
    """A camera pose from a third-party NeRF document."""
    file_path: str
    """Image path relative to the scene directory."""
    transform_matrix: Float64[ndarray, "4 4"]
    """OpenGL camera-to-world matrix."""


@serde
@dataclass(frozen=True, slots=True)
class NerfTransforms:
    """Camera calibration for one NeRF split."""
    camera_angle_x: float
    """Horizontal field of view in radians."""
    frames: list[NerfFrame]
    """Ordered camera poses and image paths."""


def load_nerf_cameras(scene_dir: Path, split: Literal["train", "val", "test"]) -> list[tuple[PinholeParameters, Path]]:
    """Read NeRF-synthetic cameras of one split as (pinhole, image path) pairs.

    The c2w matrices are OpenGL/RUB; simplecv carries the convention through to
    the Pinhole's ``camera_xyz``, so they are used unmodified.
    """
    source: Path = scene_dir / f"transforms_{split}.json"
    try:
        transforms: NerfTransforms = from_json(NerfTransforms, source.read_text())
    except (SerdeError, JSONDecodeError) as error:
        raise ValueError(f"Invalid NeRF cameras in {source}: {error}") from error
    camera_angle_x: float = transforms.camera_angle_x
    cameras: list[tuple[PinholeParameters, Path]] = []
    for frame in transforms.frames:
        image_path: Path = (scene_dir / frame.file_path).with_suffix(".png")
        with Image.open(image_path) as probe:
            width, height = probe.size
        focal: float = 0.5 * width / np.tan(0.5 * camera_angle_x)
        c2w: Float64[ndarray, "4 4"] = frame.transform_matrix
        camera: PinholeParameters = PinholeParameters(
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
