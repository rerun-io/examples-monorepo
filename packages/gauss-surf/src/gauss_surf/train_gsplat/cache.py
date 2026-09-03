"""GPU-resident training cache populated directly from one catalog segment."""

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from arkitscenes_download.ingest.paths import (
    DEPTH_ULTRAWIDE_RECT,
    NORMALS_MOGE,
    NORMALS_ULTRAWIDE_RECT,
    RGB_ULTRAWIDE_RECT,
    VIDEO_WIDE,
)
from einops import rearrange
from jaxtyping import Float32
from PIL import Image

from gauss_surf.catalog import SegmentReader, TimedeltaNs, table_timestamps
from gauss_surf.contracts import (
    PROMPTDA_DEPTH_BLOB_COLUMN,
    ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
    WIDE_CHOSEN_SHARPNESS_COLUMN,
    WIDE_FPS,
    CameraTag,
)
from gauss_surf.render_io import RenderCamera, blob_bytes, decode_rgb_image
from gauss_surf.train_gsplat.core import decode_normal_uint8, holdout_hash

WIDE_NORMAL_COLUMN: str = f"/{NORMALS_MOGE}:EncodedImage:blob"
UW_RGB_COLUMN: str = f"/{RGB_ULTRAWIDE_RECT}:EncodedImage:blob"
UW_DEPTH_COLUMN: str = f"/{DEPTH_ULTRAWIDE_RECT}:EncodedDepthImage:blob"
UW_NORMAL_COLUMN: str = f"/{NORMALS_ULTRAWIDE_RECT}:EncodedImage:blob"


@dataclass(frozen=True, slots=True)
class RasterCamera:
    """Minimal camera values consumed by the gsplat renderer."""

    width: int
    """Native camera width in pixels."""
    height: int
    """Native camera height in pixels."""
    viewmat_44: torch.Tensor
    """OpenCV world-to-camera transform."""
    K_33: torch.Tensor
    """Native camera intrinsic matrix."""


@dataclass(frozen=True, slots=True)
class TrainingCamera(RasterCamera):
    """One metric-world training camera and its cache location."""

    stem: str
    """Stable chosen-frame stem."""
    camera: CameraTag
    """Wide or rectified-ultrawide tag."""
    timestamp_ns: int
    """Exact duration since recording start, in nanoseconds."""
    cache_index: int
    """Index within the corresponding resident camera-group tensors."""
    holdout: bool
    """Whether this camera belongs to the wide holdout split."""


@dataclass(slots=True)
class GpuTrainingCache:
    """Quantized source signals and metric-world cameras resident on one GPU."""

    wide_rgb_nhw3: torch.Tensor
    wide_depth_nhw: torch.Tensor
    wide_normal_nhw3: torch.Tensor
    uw_rgb_nhw3: torch.Tensor
    uw_depth_nhw: torch.Tensor
    uw_normal_nhw3: torch.Tensor
    cameras: tuple[TrainingCamera, ...]
    train_indices: torch.Tensor
    holdout_indices: tuple[int, ...]
    holdout_sha256: str
    scene_scale: float

    def sample(self, training_index: int, downscale: int) -> tuple[TrainingCamera, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dequantize one randomly selected resident frame on the GPU."""
        camera_index: int = int(self.train_indices[training_index].item())
        camera: TrainingCamera = self.cameras[camera_index]
        if camera.camera == "wide":
            rgb_hw3: torch.Tensor = self.wide_rgb_nhw3[camera.cache_index]
            depth_hw: torch.Tensor = self.wide_depth_nhw[camera.cache_index]
            normal_hw3: torch.Tensor = self.wide_normal_nhw3[camera.cache_index]
        else:
            rgb_hw3 = self.uw_rgb_nhw3[camera.cache_index]
            depth_hw = self.uw_depth_nhw[camera.cache_index]
            normal_hw3 = self.uw_normal_nhw3[camera.cache_index]
        rgb_13hw: torch.Tensor = rgb_hw3.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
        depth_11hw: torch.Tensor = depth_hw.unsqueeze(0).unsqueeze(0).to(torch.float32) / 1_000.0
        normal_13hw: torch.Tensor = decode_normal_uint8(normal_hw3).permute(2, 0, 1).unsqueeze(0)
        if downscale > 1:
            output_size: tuple[int, int] = (camera.height // downscale, camera.width // downscale)
            rgb_13hw = torch.nn.functional.interpolate(rgb_13hw, output_size, mode="bilinear", align_corners=False, antialias=True)
            depth_11hw = torch.nn.functional.interpolate(depth_11hw, output_size, mode="bilinear", align_corners=False)
            normal_13hw = torch.nn.functional.interpolate(normal_13hw, output_size, mode="bilinear", align_corners=False)
        elif normal_13hw.shape[-2:] != rgb_13hw.shape[-2:]:
            normal_13hw = torch.nn.functional.interpolate(normal_13hw, rgb_13hw.shape[-2:], mode="bilinear", align_corners=False)
        return camera, rgb_13hw.squeeze(0).permute(1, 2, 0), depth_11hw.squeeze(0).permute(1, 2, 0), normal_13hw.squeeze(0)


@dataclass(frozen=True, slots=True)
class CacheParity:
    """Pixel/component counts from a full cache-to-bundle comparison."""

    compared_values: dict[str, int]
    mismatched_values: dict[str, int]

    @property
    def mismatch_count(self) -> int:
        """Return mismatches summed across all six modalities."""
        return sum(self.mismatched_values.values())


def _decode_png_uint16(encoded: bytes) -> np.ndarray:
    """Decode one lossless uint16 depth PNG without dequantizing it."""
    with Image.open(BytesIO(encoded)) as image:
        depth: np.ndarray = np.asarray(image, dtype=np.uint16).copy()
    if depth.ndim != 2:
        raise ValueError(f"depth PNG must be two-dimensional, got {depth.shape}")
    return depth


def _metric_camera(
    raw_frame: dict[str, Any],
    *,
    cache_index: int,
) -> TrainingCamera:
    """Convert a metric-world OpenGL camera to an OpenCV view matrix."""
    world_from_camera_44: np.ndarray = np.asarray(raw_frame["transform_matrix"], dtype=np.float32)
    world_from_opencv_44: np.ndarray = world_from_camera_44.copy()
    world_from_opencv_44[:, 1:3] *= -1.0
    viewmat_44: np.ndarray = np.linalg.inv(world_from_opencv_44).astype(np.float32)
    K_33: np.ndarray = np.asarray(
        [
            [raw_frame["fl_x"], 0.0, raw_frame["cx"]],
            [0.0, raw_frame["fl_y"], raw_frame["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return TrainingCamera(
        stem=Path(str(raw_frame["file_path"])).stem,
        camera=str(raw_frame["camera"]),  # pyrefly: ignore  # validated by bundle schema
        timestamp_ns=int(raw_frame["timestamp_ns"]),
        width=int(raw_frame["w"]),
        height=int(raw_frame["h"]),
        viewmat_44=torch.from_numpy(viewmat_44),
        K_33=torch.from_numpy(K_33),
        cache_index=cache_index,
        holdout=bool(raw_frame["holdout"]),
    )


def raster_camera_from_render_camera(camera: RenderCamera) -> RasterCamera:
    """Adapt one metric full-grid camera to the minimal renderer contract."""
    world_from_camera_44: np.ndarray = np.eye(4, dtype=np.float32)
    world_from_camera_44[:3] = camera.world_from_camera_34
    world_from_opencv_44: np.ndarray = world_from_camera_44.copy()
    world_from_opencv_44[:, 1:3] *= -1.0
    K_33: np.ndarray = np.asarray(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return RasterCamera(
        width=camera.width,
        height=camera.height,
        viewmat_44=torch.from_numpy(np.linalg.inv(world_from_opencv_44).astype(np.float32)),
        K_33=torch.from_numpy(K_33),
    )


def scene_scale_from_camera_poses(world_from_camera_n44: Float32[torch.Tensor, "n 4 4"]) -> float:
    """Return the inverse of the applied automatic pose scale.

    This reproduces the historical pose-centering rule in float32: translate
    by the mean camera origin, rotate the mean camera-up direction onto +Z,
    then take the largest absolute translation coordinate.

    Args:
        world_from_camera_n44: Float32 camera-to-world matrices with shape
            ``(n, 4, 4)``.

    Returns:
        Positive scene scale in world units.
    """
    if world_from_camera_n44.ndim != 3 or world_from_camera_n44.shape[1:] != (4, 4) or len(world_from_camera_n44) == 0:
        raise ValueError(f"camera poses must have non-empty shape (n, 4, 4), got {tuple(world_from_camera_n44.shape)}")
    if not bool(torch.isfinite(world_from_camera_n44).all().item()):
        raise ValueError("camera poses must be finite")
    origins_n3: Float32[torch.Tensor, "n 3"] = world_from_camera_n44[:, :3, 3]
    translation_3: Float32[torch.Tensor, "3"] = origins_n3.mean(dim=0)
    up_3: Float32[torch.Tensor, "3"] = world_from_camera_n44[:, :3, 1].mean(dim=0)
    up_norm: torch.Tensor = torch.linalg.vector_norm(up_3)
    if not bool(torch.isfinite(up_norm).item()) or float(up_norm.item()) <= 0.0:
        raise ValueError("mean camera-up direction must be finite and nonzero")
    up_3 = up_3 / up_norm
    target_up_3: Float32[torch.Tensor, "3"] = torch.tensor(
        (0.0, 0.0, 1.0), dtype=world_from_camera_n44.dtype, device=world_from_camera_n44.device
    )
    up_3 = up_3 / torch.linalg.vector_norm(up_3)
    target_up_3 = target_up_3 / torch.linalg.vector_norm(target_up_3)
    rotation_axis_3: Float32[torch.Tensor, "3"] = torch.linalg.cross(up_3, target_up_3)
    if float(torch.abs(rotation_axis_3).sum().item()) < 1e-6:
        fallback_axis_3: Float32[torch.Tensor, "3"] = torch.tensor(
            (1.0, 0.0, 0.0) if abs(float(up_3[0].item())) < 1e-6 else (0.0, 1.0, 0.0),
            dtype=world_from_camera_n44.dtype,
            device=world_from_camera_n44.device,
        )
        rotation_axis_3 = torch.linalg.cross(up_3, fallback_axis_3)
    rotation_axis_3 = rotation_axis_3 / torch.linalg.vector_norm(rotation_axis_3)
    axis_x: torch.Tensor = rotation_axis_3[0]
    axis_y: torch.Tensor = rotation_axis_3[1]
    axis_z: torch.Tensor = rotation_axis_3[2]
    zero: torch.Tensor = torch.zeros((), dtype=world_from_camera_n44.dtype, device=world_from_camera_n44.device)
    skew_33: Float32[torch.Tensor, "3 3"] = torch.stack(
        (
            torch.stack((zero, -axis_z, axis_y)),
            torch.stack((axis_z, zero, -axis_x)),
            torch.stack((-axis_y, axis_x, zero)),
        )
    )
    angle: torch.Tensor = torch.acos(torch.clip(torch.dot(up_3, target_up_3), -1.0, 1.0))
    identity_33: Float32[torch.Tensor, "3 3"] = torch.eye(
        3, dtype=world_from_camera_n44.dtype, device=world_from_camera_n44.device
    )
    rotation_33: Float32[torch.Tensor, "3 3"] = (
        identity_33 + torch.sin(angle) * skew_33 + (1.0 - torch.cos(angle)) * (skew_33 @ skew_33)
    )
    transform_34: Float32[torch.Tensor, "3 4"] = torch.cat(
        (rotation_33, rotation_33 @ -translation_3[:, None]), dim=-1
    )
    oriented_poses_n34: Float32[torch.Tensor, "n 3 4"] = transform_34 @ world_from_camera_n44
    scene_scale: float = float(torch.abs(oriented_poses_n34[:, :3, 3]).max().item())
    if not np.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError(f"camera scene scale must be positive and finite, got {scene_scale}")
    return scene_scale


def load_training_cameras(
    bundle_dir: Path,
) -> tuple[tuple[TrainingCamera, ...], float]:
    """Read metric bundle cameras and reproduce their automatic scene scale."""
    transforms: dict[str, Any] = json.loads((bundle_dir / "transforms.json").read_text(encoding="utf-8"))
    raw_frames: list[dict[str, Any]] = list(transforms["frames"])
    if len(raw_frames) == 0:
        raise ValueError("training camera manifest is empty")
    world_from_camera_n44: Float32[torch.Tensor, "n 4 4"] = torch.tensor(
        [raw_frame["transform_matrix"] for raw_frame in raw_frames], dtype=torch.float32
    )
    scene_scale: float = scene_scale_from_camera_poses(world_from_camera_n44)
    camera_counts: dict[str, int] = {"wide": 0, "uw": 0}
    cameras: list[TrainingCamera] = []
    for raw_frame in raw_frames:
        camera_tag: str = str(raw_frame["camera"])
        cache_index: int = camera_counts[camera_tag]
        cameras.append(_metric_camera(raw_frame, cache_index=cache_index))
        camera_counts[camera_tag] += 1
    return tuple(cameras), scene_scale


def load_gpu_cache(
    reader: SegmentReader,
    bundle_dir: Path,
    device: torch.device,
) -> GpuTrainingCache:
    """Fill timeline-ordered quantized tensors from catalog source layers."""
    cameras, scene_scale = load_training_cameras(bundle_dir)
    wide_cameras: tuple[TrainingCamera, ...] = tuple(camera for camera in cameras if camera.camera == "wide")
    uw_cameras: tuple[TrainingCamera, ...] = tuple(camera for camera in cameras if camera.camera == "uw")

    wide_table: pa.Table = reader.chosen_table(
        WIDE_CHOSEN_SHARPNESS_COLUMN,
        (PROMPTDA_DEPTH_BLOB_COLUMN, WIDE_NORMAL_COLUMN),
    )
    uw_table: pa.Table = reader.chosen_table(
        ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
        (UW_RGB_COLUMN, UW_DEPTH_COLUMN, UW_NORMAL_COLUMN),
    )
    if wide_table.num_rows != len(wide_cameras) or uw_table.num_rows != len(uw_cameras):
        raise RuntimeError(
            f"catalog/bundle counts differ: wide {wide_table.num_rows}/{len(wide_cameras)}, uw {uw_table.num_rows}/{len(uw_cameras)}"
        )
    wide_timestamps: TimedeltaNs = table_timestamps(wide_table)
    uw_timestamps: TimedeltaNs = table_timestamps(uw_table)
    expected_wide_ns: np.ndarray = np.asarray([camera.timestamp_ns for camera in wide_cameras], dtype=np.int64)
    expected_uw_ns: np.ndarray = np.asarray([camera.timestamp_ns for camera in uw_cameras], dtype=np.int64)
    if not np.array_equal(wide_timestamps.astype(np.int64), expected_wide_ns) or not np.array_equal(uw_timestamps.astype(np.int64), expected_uw_ns):
        raise RuntimeError("catalog and Part 10 bundle timelines differ")

    wide_rows: list[dict[str, Any]] = wide_table.to_pylist()
    uw_rows: list[dict[str, Any]] = uw_table.to_pylist()
    wide_resolutions: set[tuple[int, int]] = {(camera.height, camera.width) for camera in wide_cameras}
    uw_resolutions: set[tuple[int, int]] = {(camera.height, camera.width) for camera in uw_cameras}
    if len(wide_resolutions) != 1 or len(uw_resolutions) != 1:
        raise RuntimeError(
            f"training cameras must have one resolution per group, got wide={sorted(wide_resolutions)} uw={sorted(uw_resolutions)}"
        )
    wide_hw: tuple[int, int] = next(iter(wide_resolutions))
    uw_hw: tuple[int, int] = next(iter(uw_resolutions))
    first_wide_normal_hw3: np.ndarray = decode_rgb_image(blob_bytes(wide_rows[0][WIDE_NORMAL_COLUMN], WIDE_NORMAL_COLUMN))
    wide_normal_hw3: tuple[int, int, int] = first_wide_normal_hw3.shape
    if wide_normal_hw3[-1:] != (3,):
        raise RuntimeError(f"wide normal layer must contain RGB images, got {wide_normal_hw3}")
    wide_depth_nhw: torch.Tensor = torch.empty((len(wide_rows), *wide_hw), dtype=torch.uint16, device=device)
    wide_normal_nhw3: torch.Tensor = torch.empty((len(wide_rows), *wide_normal_hw3), dtype=torch.uint8, device=device)
    wide_rgb_nhw3: torch.Tensor = torch.empty((len(wide_rows), *wide_hw, 3), dtype=torch.uint8, device=device)
    decoded_wide = reader.decode_frames(VIDEO_WIDE, wide_timestamps, fps=WIDE_FPS, device=device)
    for index, (row, frame_chw) in enumerate(zip(wide_rows, decoded_wide, strict=True)):
        decoded_rgb_hw3: torch.Tensor = rearrange(frame_chw, "c h w -> h w c")
        depth_hw: np.ndarray = _decode_png_uint16(blob_bytes(row[PROMPTDA_DEPTH_BLOB_COLUMN], PROMPTDA_DEPTH_BLOB_COLUMN))
        normal_hw3: np.ndarray = first_wide_normal_hw3 if index == 0 else decode_rgb_image(blob_bytes(row[WIDE_NORMAL_COLUMN], WIDE_NORMAL_COLUMN))
        if tuple(decoded_rgb_hw3.shape) != (*wide_hw, 3) or depth_hw.shape != wide_hw or normal_hw3.shape != wide_normal_hw3:
            raise RuntimeError(
                f"wide cache signal shapes differ from contracts: rgb={tuple(decoded_rgb_hw3.shape)} depth={depth_hw.shape} "
                f"normal={normal_hw3.shape}; expected rgb/depth={wide_hw} normal={wide_normal_hw3[:2]}"
            )
        wide_rgb_nhw3[index].copy_(decoded_rgb_hw3)
        wide_depth_nhw[index].copy_(torch.from_numpy(depth_hw))
        wide_normal_nhw3[index].copy_(torch.from_numpy(normal_hw3))
        if (index + 1) % 25 == 0 or index + 1 == len(wide_rows):
            print(f"cached {index + 1}/{len(wide_rows)} wide frames", flush=True)

    uw_rgb_nhw3: torch.Tensor = torch.empty((len(uw_rows), *uw_hw, 3), dtype=torch.uint8, device=device)
    uw_depth_nhw: torch.Tensor = torch.empty((len(uw_rows), *uw_hw), dtype=torch.uint16, device=device)
    uw_normal_nhw3: torch.Tensor = torch.empty((len(uw_rows), *uw_hw, 3), dtype=torch.uint8, device=device)
    for index, row in enumerate(uw_rows):
        rgb_hw3 = decode_rgb_image(blob_bytes(row[UW_RGB_COLUMN], UW_RGB_COLUMN))
        depth_hw = _decode_png_uint16(blob_bytes(row[UW_DEPTH_COLUMN], UW_DEPTH_COLUMN))
        normal_hw3 = decode_rgb_image(blob_bytes(row[UW_NORMAL_COLUMN], UW_NORMAL_COLUMN))
        if rgb_hw3.shape != (*uw_hw, 3) or depth_hw.shape != uw_hw or normal_hw3.shape != (*uw_hw, 3):
            raise RuntimeError(
                f"ultrawide cache signal shapes differ from camera resolution {uw_hw}: "
                f"rgb={rgb_hw3.shape} depth={depth_hw.shape} normal={normal_hw3.shape}"
            )
        uw_rgb_nhw3[index].copy_(torch.from_numpy(rgb_hw3))
        uw_depth_nhw[index].copy_(torch.from_numpy(depth_hw))
        uw_normal_nhw3[index].copy_(torch.from_numpy(normal_hw3))
        if (index + 1) % 50 == 0 or index + 1 == len(uw_rows):
            print(f"cached {index + 1}/{len(uw_rows)} ultrawide frames", flush=True)

    holdout_indices: tuple[int, ...] = tuple(index for index, camera in enumerate(cameras) if camera.holdout)
    train_indices: torch.Tensor = torch.tensor(
        [index for index, camera in enumerate(cameras) if not camera.holdout], dtype=torch.int64, device="cpu"
    )
    return GpuTrainingCache(
        wide_rgb_nhw3=wide_rgb_nhw3,
        wide_depth_nhw=wide_depth_nhw,
        wide_normal_nhw3=wide_normal_nhw3,
        uw_rgb_nhw3=uw_rgb_nhw3,
        uw_depth_nhw=uw_depth_nhw,
        uw_normal_nhw3=uw_normal_nhw3,
        cameras=cameras,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        holdout_sha256=holdout_hash(tuple(cameras[index].stem for index in holdout_indices)),
        scene_scale=scene_scale,
    )


def compare_cache_to_bundle(cache: GpuTrainingCache, bundle_dir: Path) -> CacheParity:
    """Compare all six cached modalities with the Part 10 bundle pixel-by-pixel."""
    compared: dict[str, int] = {name: 0 for name in ("wide_rgb", "wide_depth", "wide_normal", "uw_rgb", "uw_depth", "uw_normal")}
    mismatched: dict[str, int] = dict.fromkeys(compared, 0)
    for camera in cache.cameras:
        if camera.camera == "wide":
            image_relative: Path = Path("images") / f"{camera.stem}.png"
            rgb_actual: np.ndarray = cache.wide_rgb_nhw3[camera.cache_index].cpu().numpy()
            depth_actual: np.ndarray = cache.wide_depth_nhw[camera.cache_index].cpu().numpy()
            normal_actual: np.ndarray = cache.wide_normal_nhw3[camera.cache_index].cpu().numpy()
            prefix: str = "wide"
        else:
            image_relative = Path("images_uw") / f"{camera.stem}.jpg"
            rgb_actual = cache.uw_rgb_nhw3[camera.cache_index].cpu().numpy()
            depth_actual = cache.uw_depth_nhw[camera.cache_index].cpu().numpy()
            normal_actual = cache.uw_normal_nhw3[camera.cache_index].cpu().numpy()
            prefix = "uw"
        with Image.open(bundle_dir / image_relative) as image:
            rgb_expected: np.ndarray = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with np.load(bundle_dir / "depth" / f"{camera.stem}.npz") as archive:
            depth_expected: np.ndarray = np.rint(archive["depth"] * 1_000.0).astype(np.uint16)
        with np.load(bundle_dir / "normals" / f"{camera.stem}.npz") as archive:
            normal_float: np.ndarray = archive["normal"]
            normal_expected: np.ndarray = np.rint((normal_float + 1.0) / 2.0 * 255.0).astype(np.uint8)
        for suffix, actual, expected in (
            ("rgb", rgb_actual, rgb_expected),
            ("depth", depth_actual, depth_expected),
            ("normal", normal_actual, normal_expected),
        ):
            name: str = f"{prefix}_{suffix}"
            if actual.shape != expected.shape:
                raise RuntimeError(f"cache parity shape mismatch for {camera.stem} {name}: {actual.shape} != {expected.shape}")
            compared[name] += actual.size
            mismatched[name] += int(np.count_nonzero(actual != expected))
    return CacheParity(compared_values=compared, mismatched_values=mismatched)
