"""Shared readers and codecs for rendered gauss-surf artifacts."""

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image

from gauss_surf.catalog import _single_instance
from gauss_surf.contracts import RGB_JPEG_QUALITY, CameraTag


@dataclass(frozen=True, slots=True)
class RenderCamera:
    """One native-resolution camera read from the full-grid manifest."""

    stem: str
    """Stable full-grid frame stem."""
    camera: CameraTag
    """Wide or rectified-ultrawide tag."""
    timestamp_ns: int
    """Exact duration since recording start, in nanoseconds."""
    width: int
    """Native render width in pixels."""
    height: int
    """Native render height in pixels."""
    fx: float
    """Horizontal focal length in pixels."""
    fy: float
    """Vertical focal length in pixels."""
    cx: float
    """Principal-point horizontal coordinate in pixels."""
    cy: float
    """Principal-point vertical coordinate in pixels."""
    world_from_camera_34: Float32[ndarray, "3 4"]
    """Metric OpenGL camera-to-world pose."""


def load_render_cameras(
    cameras_path: Path,
) -> list[RenderCamera]:
    """Read metric-world full-grid render cameras."""
    manifest: dict[str, Any] = json.loads(cameras_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"{cameras_path} has unsupported camera manifest schema {manifest.get('schema_version')!r}")
    if manifest.get("camera_model") != "OPENCV":
        raise ValueError(f"{cameras_path} has unsupported camera model {manifest.get('camera_model')!r}")
    raw_frames: Any = manifest.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError(f"{cameras_path} has no frame list")
    cameras: list[RenderCamera] = []
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise ValueError("camera manifest frame metadata must be an object")
        camera: str = str(raw_frame["camera"])
        if camera not in ("wide", "uw"):
            raise ValueError(f"camera manifest frame has invalid camera tag {camera!r}")
        world_from_camera_44: Float32[ndarray, "4 4"] = np.asarray(raw_frame["transform_matrix"], dtype=np.float32)
        if world_from_camera_44.shape != (4, 4):
            raise ValueError(f"camera manifest transform must have shape (4, 4), got {world_from_camera_44.shape}")
        if not np.all(np.isfinite(world_from_camera_44)):
            raise ValueError("camera manifest transform contains nonfinite values")
        width: int = int(raw_frame["w"])
        height: int = int(raw_frame["h"])
        intrinsics_4: tuple[float, float, float, float] = (
            float(raw_frame["fl_x"]),
            float(raw_frame["fl_y"]),
            float(raw_frame["cx"]),
            float(raw_frame["cy"]),
        )
        if width <= 0 or height <= 0:
            raise ValueError(f"camera manifest resolution must be positive, got {(width, height)}")
        if not np.all(np.isfinite(intrinsics_4)) or intrinsics_4[0] <= 0.0 or intrinsics_4[1] <= 0.0:
            raise ValueError(f"camera manifest intrinsics are invalid: {intrinsics_4}")
        cameras.append(
            RenderCamera(
                stem=str(raw_frame["stem"]),
                camera=camera,  # pyrefly: ignore  # bad-argument-type — validated above
                timestamp_ns=int(raw_frame["timestamp_ns"]),
                width=width,
                height=height,
                fx=intrinsics_4[0],
                fy=intrinsics_4[1],
                cx=intrinsics_4[2],
                cy=intrinsics_4[3],
                world_from_camera_34=world_from_camera_44[:3].copy(),
            )
        )
    stems: list[str] = [camera.stem for camera in cameras]
    if len(stems) != len(set(stems)):
        raise ValueError("camera manifest frame stems are not unique")
    camera_timestamps: list[tuple[CameraTag, int]] = [(camera.camera, camera.timestamp_ns) for camera in cameras]
    if len(camera_timestamps) != len(set(camera_timestamps)):
        raise ValueError("camera manifest timestamps are not unique within each camera")
    raw_counts: Any = manifest.get("counts")
    if not isinstance(raw_counts, dict):
        raise ValueError("camera manifest has no count mapping")
    actual_counts: dict[str, int] = {
        "wide": sum(camera.camera == "wide" for camera in cameras),
        "uw": sum(camera.camera == "uw" for camera in cameras),
        "total": len(cameras),
    }
    declared_counts: dict[str, int] = {key: int(raw_counts.get(key, -1)) for key in actual_counts}
    if declared_counts != actual_counts:
        raise ValueError(f"camera manifest counts {declared_counts} do not match frames {actual_counts}")
    return cameras


def blob_bytes(value: Any, component_name: str) -> bytes:
    """Unwrap one encoded Rerun blob instance from an Arrow cell."""
    blob_value: Any = _single_instance(value, component_name)
    return blob_value if isinstance(blob_value, bytes) else bytes(blob_value)


def encode_rgb_png(rgb_hw3: UInt8[ndarray, "h w 3"]) -> bytes:
    """Encode one uint8 RGB image as a lossless PNG."""
    if rgb_hw3.ndim != 3 or rgb_hw3.shape[-1] != 3 or rgb_hw3.dtype != np.uint8:
        raise ValueError("RGB image must be uint8 with shape (H, W, 3)")
    output: BytesIO = BytesIO()
    Image.fromarray(rgb_hw3, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def encode_rgb_jpeg(rgb_hw3: UInt8[ndarray, "h w 3"], *, quality: int = RGB_JPEG_QUALITY) -> bytes:
    """Encode one uint8 RGB image as a JPEG at the requested quality."""
    if rgb_hw3.ndim != 3 or rgb_hw3.shape[-1] != 3 or rgb_hw3.dtype != np.uint8:
        raise ValueError("RGB image must be uint8 with shape (H, W, 3)")
    output: BytesIO = BytesIO()
    Image.fromarray(rgb_hw3, mode="RGB").save(output, format="JPEG", quality=quality)
    return output.getvalue()


def decode_rgb_image(image_bytes: bytes) -> UInt8[ndarray, "h w 3"]:
    """Decode stored image bytes to an owning uint8 RGB array."""
    with Image.open(BytesIO(image_bytes)) as image:
        rgb_hw3: UInt8[ndarray, "h w 3"] = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return rgb_hw3


def decode_splat_depth_png(png_bytes: bytes) -> Float32[ndarray, "h w"]:
    """Decode a uint16 millimetre PNG into float32 metres.

    Args:
        png_bytes: Encoded 16-bit grayscale PNG bytes.

    Returns:
        Float32 metre depth shaped ``h w`` with a preserved zero sentinel.
    """
    with Image.open(BytesIO(png_bytes)) as image:
        depth_mm_hw: ndarray = np.asarray(image, dtype=np.uint16)
    if depth_mm_hw.ndim != 2:
        raise ValueError(f"depth PNG must be grayscale, got {depth_mm_hw.shape}")
    return (depth_mm_hw.astype(np.float32) / 1000.0).astype(np.float32, copy=False)
