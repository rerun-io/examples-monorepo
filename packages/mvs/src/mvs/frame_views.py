"""What each view of the live-meshing layout (:mod:`mvs.blueprint`) shows for one depth frame.

Ported from the original SimpleRecon live-meshing demo by Pablo Vela (repo pablovela5620/simplerecon-inference,
``live_meshing.py``, ``config.LogConfig`` and ``utils/geometry_utils.NormalGenerator`` at d5bc1a0): the green current
frustum, the source frusta and their images, the depth normals and the depth of the cost volume's best plane (``lowest_cost``).
"""

from dataclasses import dataclass

import depth_png
import kornia
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as functional
from einops import rearrange, reduce
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from torch import Tensor

from mvs.blueprint import CURRENT_CAM_COLOR, CURRENT_CAM_PLANE_M, SOURCE_CAM_PLANE_M, EntityPaths
from mvs.depth_engine import IMAGENET_MEAN, IMAGENET_STD, DepthInputs

DEPTH_WH: tuple[int, int] = (256, 192)
"""The depth raster, where the source frusta and images are logged (the original logs them at the depth size)."""


@dataclass(frozen=True)
class LogConfig:
    """Which groups of the layout are logged, and how the pictures are stored (the original's ``LogConfig``)."""

    camera: bool = True
    """Relay the segment's video under the current pinhole (Current view, and the image in the green frustum)."""
    source_cams: bool = True
    """Log each source camera frustum and image (the filmstrip)."""
    normals: bool = True
    """Log the normals of the predicted depth."""
    lowest_cost: bool = True
    """Log the depth of the cost volume's best plane per pixel (the original's lowest cost)."""
    timings: bool = True
    """Log the per-stage timings, ms."""
    jpeg_quality: int | None = 75
    """JPEG-compress the source and normal images at this quality; ``--log.jpeg-quality None`` logs raw uint8 (the
    original's default), about 7x the recording size."""

    def __post_init__(self) -> None:
        if self.jpeg_quality is not None and not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in 1..100, got {self.jpeg_quality}")


def log_current_cam_style(paths: EntityPaths) -> None:
    """Tint the current camera's frustum green; a static partial update composes with the per-frame pinhole."""
    rr.log(paths.current_pinhole, rr.Pinhole.from_fields(image_plane_distance=CURRENT_CAM_PLANE_M, color=CURRENT_CAM_COLOR), static=True)


def rgb_at_depth_raster(images_n3hw: Float32[Tensor, "n 3 h=384 w=512"]) -> UInt8[ndarray, "n h=192 w=256 3"]:
    """Model input images, un-normalised and averaged 2x2 down to the depth raster, as 8-bit RGB."""
    mean_311: Float32[Tensor, "3 1 1"] = rearrange(images_n3hw.new_tensor(IMAGENET_MEAN), "c -> c 1 1")
    std_311: Float32[Tensor, "3 1 1"] = rearrange(images_n3hw.new_tensor(IMAGENET_STD), "c -> c 1 1")
    rgb_n3hw: Float32[Tensor, "n 3 h w"] = images_n3hw * std_311 + mean_311
    rgb_nhw3: Float32[Tensor, "n h2 w2 3"] = reduce(rgb_n3hw, "n c (h 2) (w 2) -> n h w c", "mean")
    return (rgb_nhw3.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()


def source_cameras(inputs: DepthInputs) -> tuple[Float32[ndarray, "m 4 4"], Float32[ndarray, "m 3 3"]]:
    """The source views' world-from-camera poses and their intrinsics at the depth raster (twice the s1 matching scale)."""
    world_T_cam_m44: Float32[Tensor, "m 4 4"] = torch.linalg.inv(inputs.src_cam_T_world_bm44[0])
    K_depth_m33: Float32[Tensor, "m 3 3"] = inputs.src_K_bm44[0, :, :3, :3] * inputs.src_K_bm44.new_tensor([2.0, 2.0, 1.0])[:, None]
    return world_T_cam_m44.cpu().numpy(), K_depth_m33.cpu().numpy()


def log_image(entity_path: str, image_hw3: UInt8[ndarray, "h w 3"], jpeg_quality: int | None) -> None:
    """Log 8-bit RGB, JPEG-compressed when ``jpeg_quality`` is set."""
    image: rr.Image = rr.Image(image_hw3)
    rr.log(entity_path, image.compress(jpeg_quality=jpeg_quality) if jpeg_quality is not None else image)


def log_sources(paths: EntityPaths, inputs: DepthInputs, jpeg_quality: int | None) -> None:
    """Log the source views of one depth query: their frusta (0.09 m image plane) and images at the depth raster."""
    world_T_cam_m44, K_m33 = source_cameras(inputs)
    images_mhw3: UInt8[ndarray, "m h w 3"] = rgb_at_depth_raster(inputs.src_image_bm3hw[0])
    index: int
    for index in range(len(world_T_cam_m44)):
        rr.log(paths.source_cam(index), rr.Transform3D(mat3x3=world_T_cam_m44[index, :3, :3], translation=world_T_cam_m44[index, :3, 3]))
        rr.log(
            paths.source_pinhole(index),
            rr.Pinhole(
                image_from_camera=K_m33[index],
                width=DEPTH_WH[0],
                height=DEPTH_WH[1],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=SOURCE_CAM_PLANE_M,
            ),
        )
        log_image(paths.source_rgb(index), images_mhw3[index], jpeg_quality)


def depth_normals(
    depth_b1hw: Float32[Tensor, "b 1 h w"], invK_b33: Float32[Tensor, "b 3 3"], kernel_size: int = 5, std: float = 2.0
) -> Float32[Tensor, "b 3 h w"]:
    """Camera-space unit normals of a depth image: Gaussian-smoothed depth, back-projected at pixel centres, and the
    cross product of its two spatial gradients (the original's ``NormalGenerator``)."""
    height: int = depth_b1hw.shape[-2]
    width: int = depth_b1hw.shape[-1]
    smooth_b1hw: Float32[Tensor, "b 1 h w"] = kornia.filters.gaussian_blur2d(depth_b1hw, (kernel_size, kernel_size), (std, std))
    xx_hw, yy_hw = torch.meshgrid(torch.arange(width, device=depth_b1hw.device), torch.arange(height, device=depth_b1hw.device), indexing="xy")
    pix_3n: Float32[Tensor, "3 n"] = torch.stack((xx_hw + 0.5, yy_hw + 0.5, torch.ones_like(xx_hw, dtype=torch.float32))).flatten(1).float()
    rays_b3n: Float32[Tensor, "b 3 n"] = torch.matmul(invK_b33, pix_3n)
    points_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(smooth_b1hw.flatten(start_dim=2) * rays_b3n, "b i (h w) -> b i h w", h=height, w=width)
    gradients_b32hw: Float32[Tensor, "b 3 2 h w"] = kornia.filters.spatial_gradient(points_b3hw)
    normals_b3hw: Float32[Tensor, "b 3 h w"] = torch.cross(gradients_b32hw[:, :, 0], gradients_b32hw[:, :, 1], dim=1)
    return functional.normalize(normals_b3hw, dim=1)


def normals_rgb(normals_3hw: Float32[Tensor, "3 h w"]) -> UInt8[ndarray, "h w 3"]:
    """Unit normals mapped from [-1, 1] to 8-bit RGB, as the original logs them."""
    rgb_hw3: Float32[ndarray, "h w 3"] = rearrange(0.5 * (1.0 + normals_3hw), "c h w -> h w c").cpu().numpy()
    return np.asarray(np.clip(rgb_hw3, 0.0, 1.0) * 255.0, dtype=np.uint8)


def log_normals(paths: EntityPaths, depth_b1hw: Float32[Tensor, "1 1 h w"], K_depth_33: Float32[ndarray, "3 3"], jpeg_quality: int | None) -> None:
    """Log the normals of the predicted depth under the depth pinhole."""
    invK_133: Float32[Tensor, "1 3 3"] = torch.linalg.inv(torch.from_numpy(K_depth_33).to(depth_b1hw.device))[None]
    log_image(paths.normal, normals_rgb(depth_normals(depth_b1hw, invK_133)[0]), jpeg_quality)


def log_lowest_cost(paths: EntityPaths, lowest_cost_bhw: Float32[Tensor, "1 h w"]) -> None:
    """Log the depth of the cost volume's best plane at its matching size, as a millimetre 16-bit PNG like the depth."""
    lowest_cost_mm_hw: UInt16[ndarray, "h w"] = (
        (lowest_cost_bhw[0].float() * 1000.0).round().clamp(0.0, 65535.0).to(torch.int32).cpu().numpy().astype(np.uint16)
    )
    rr.log(paths.lowest_cost, rr.EncodedDepthImage(blob=depth_png.encode(lowest_cost_mm_hw), media_type=rr.components.MediaType.PNG, meter=1000.0))
