"""What each view of the live-meshing layout (:mod:`mvs.blueprint`) shows for one depth frame.

Ported from the original SimpleRecon live-meshing demo by Pablo Vela (repo pablovela5620/simplerecon-inference,
``live_meshing.py`` and ``utils/geometry_utils.NormalGenerator`` at d5bc1a0): the green current
frustum, the source frusta and their pictures, the depth normals and the depth of the cost volume's best plane
(``lowest_cost``). Unlike the original, which logs each source image's pixels, a source view is a
``VideoFrameReference`` into the segment's relayed VideoStream: the recording holds every video frame once. The
references resolve on the video's timeline (``video_time``), where the blueprint opens; on ``frame`` they show no frame.
"""

from dataclasses import dataclass

import depth_png
import kornia
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as functional
from einops import rearrange
from jaxtyping import Bool, Float32, Int64, UInt8, UInt16
from numpy import ndarray
from torch import Tensor

from mvs.blueprint import CURRENT_CAM_COLOR, CURRENT_CAM_PLANE_M, SOURCE_CAM_PLANE_M, EntityPaths
from mvs.depth_engine import DepthInputs

S1_WH: tuple[int, int] = (128, 96)
"""The model's s1 matching raster, where the source intrinsics come in (``depth_engine.s1_intrinsics``)."""
NORMALS_JPEG_QUALITY: int = 75
"""JPEG quality of the normals picture: it is display only, so lossy is fine."""


@dataclass(frozen=True)
class LogConfig:
    """What the layout logs beyond its fixed views (the original ``config.LogConfig``'s ``source_cams``)."""

    source_cams: bool = True
    """Log each source camera's frustum and its frame of the relayed video (the filmstrip)."""


def log_current_cam_style(paths: EntityPaths) -> None:
    """Tint the current camera's frustum green; a static partial update composes with the per-frame pinhole."""
    rr.log(paths.current_pinhole, rr.Pinhole.from_fields(image_plane_distance=CURRENT_CAM_PLANE_M, color=CURRENT_CAM_COLOR), static=True)


def source_cameras(inputs: DepthInputs, video_wh: tuple[int, int]) -> tuple[Float32[ndarray, "m 4 4"], Float32[ndarray, "m 3 3"]]:
    """The source views' world-from-camera poses and their intrinsics at the video size (the s1 ones scaled back up)."""
    world_T_cam_m44: Float32[Tensor, "m 4 4"] = torch.linalg.inv(inputs.src_cam_T_world_bm44[0])
    scale_31: Float32[Tensor, "3 1"] = inputs.src_K_bm44.new_tensor([video_wh[0] / S1_WH[0], video_wh[1] / S1_WH[1], 1.0])[:, None]
    K_video_m33: Float32[Tensor, "m 3 3"] = inputs.src_K_bm44[0, :, :3, :3] * scale_31
    return world_T_cam_m44.cpu().numpy(), K_video_m33.cpu().numpy()


def video_offsets_ns(grid_times_ns: Int64[ndarray, " m"], video_times_ns: Int64[ndarray, " n"], stream_start_ns: int) -> Int64[ndarray, " m"]:
    """The video sample each grid time decoded to (the latest at or before it, as ``SegmentNvdecDecoder.decode_at``
    picks it), as nanoseconds from ``stream_start_ns``, the relayed stream's first sample: a ``VideoFrameReference``
    into a VideoStream counts its timestamp from the stream's first sample on the active timeline (re_view_spatial 0.42
    ``visualizers/video/video_frame_reference.rs``, ``first_stream_timestamp.decode_timestamp() + timestamp``)."""
    sample_m: Int64[ndarray, " m"] = np.searchsorted(video_times_ns, grid_times_ns, side="right") - 1
    sample_times_m_ns: Int64[ndarray, " m"] = video_times_ns[np.maximum(sample_m, 0)]
    early_m: Bool[ndarray, " m"] = (sample_m < 0) | (sample_times_m_ns < stream_start_ns)
    if early_m.any():
        raise ValueError(f"grid times {grid_times_ns[early_m].tolist()} decode to samples before the relayed stream's start {stream_start_ns}")
    return sample_times_m_ns - stream_start_ns


def log_sources(paths: EntityPaths, inputs: DepthInputs, video_wh: tuple[int, int], video_offsets_m_ns: Int64[ndarray, " m"]) -> None:
    """Log the source views of one depth query: their frusta (0.09 m image plane) at the video size, and under each
    pinhole a ``VideoFrameReference`` to its frame of the relayed video; no pixels are logged."""
    world_T_cam_m44, K_m33 = source_cameras(inputs, video_wh)
    index: int
    for index in range(len(world_T_cam_m44)):
        rr.log(paths.source_cam(index), rr.Transform3D(mat3x3=world_T_cam_m44[index, :3, :3], translation=world_T_cam_m44[index, :3, 3]))
        rr.log(
            paths.source_pinhole(index),
            rr.Pinhole(
                image_from_camera=K_m33[index],
                width=video_wh[0],
                height=video_wh[1],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=SOURCE_CAM_PLANE_M,
            ),
        )
        rr.log(paths.source_rgb(index), rr.VideoFrameReference(nanoseconds=int(video_offsets_m_ns[index]), video_reference=f"/{paths.video}"))


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


def log_normals(paths: EntityPaths, depth_b1hw: Float32[Tensor, "1 1 h w"], K_depth_33: Float32[ndarray, "3 3"]) -> None:
    """Log the normals of the predicted depth under the depth pinhole, as a JPEG."""
    invK_133: Float32[Tensor, "1 3 3"] = torch.linalg.inv(torch.from_numpy(K_depth_33).to(depth_b1hw.device))[None]
    rr.log(paths.normal, rr.Image(normals_rgb(depth_normals(depth_b1hw, invK_133)[0])).compress(jpeg_quality=NORMALS_JPEG_QUALITY))


def log_lowest_cost(paths: EntityPaths, lowest_cost_bhw: Float32[Tensor, "1 h w"]) -> None:
    """Log the depth of the cost volume's best plane at its matching size, as a millimetre 16-bit PNG like the depth."""
    lowest_cost_mm_hw: UInt16[ndarray, "h w"] = (
        (lowest_cost_bhw[0] * 1000.0).round().clamp(0.0, 65535.0).to(torch.int32).cpu().numpy().astype(np.uint16)
    )
    rr.log(paths.lowest_cost, rr.EncodedDepthImage(blob=depth_png.encode(lowest_cost_mm_hw), media_type=rr.components.MediaType.PNG, meter=1000.0))
