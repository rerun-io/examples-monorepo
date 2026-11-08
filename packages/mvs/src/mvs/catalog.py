"""The ARKitScenes catalog input the mvs apps share, and how they log what every frame carries.

Entity paths mirror the arkitscenes-download rig schema (``ingest/paths.py``); the source video is relayed once, from
the raw AV1 samples the NVDEC decoder already fetched, never re-encoded per frame.
"""

from dataclasses import dataclass

import numpy as np
import rerun as rr
from einops import rearrange
from jaxtyping import Float32
from numpy import ndarray
from simplecv.rerun_dataloader import SegmentNvdecDecoder

from mvs.depth_engine import DepthOutputs

CLOUD_CATALOG_URL: str = "rerun://api.latest.cloud.rerun.io:443"
VIDEO_WIDE: str = "world/rig_00/cam_00/pinhole/video"
"""The wide camera's AV1 VideoStream in the catalog segments."""
TIMELINE: str = "video_time"
NATIVE_FPS: float = 60.0
"""Catalog segments store the wide camera as 60 fps AV1."""

CAMERA_PATH: str = "world/rig_00/cam_00"
PINHOLE_PATH: str = f"{CAMERA_PATH}/pinhole"
VIDEO_PATH: str = f"{PINHOLE_PATH}/video"
PINHOLE_LOWRES_PATH: str = f"{CAMERA_PATH}/pinhole_lowres"
"""The viewer back-projects a DepthImage with its parent pinhole's intrinsics, so the 256x192 depth needs its own
depth-resolution pinhole beside the native one."""
DEPTH_HW: tuple[int, int] = (192, 256)
"""The depth model's output raster."""


@dataclass(frozen=True)
class CatalogDataConfig:
    """ARKitScenes Rerun-catalog input configuration."""

    catalog_url: str = CLOUD_CATALOG_URL
    """Catalog server URL."""
    dataset_name: str = "arkitscenes"
    """Catalog dataset entry name."""
    segments: tuple[str, ...] = ("42899799",)
    """Segment ids to process sequentially; empty runs every registered segment."""


def log_camera(world_T_cam_44: Float32[ndarray, "4 4"], K_native_33: Float32[ndarray, "3 3"], native_wh: tuple[int, int]) -> None:
    """The frame's camera pose and native-resolution Pinhole, at the current time point."""
    rr.log(CAMERA_PATH, rr.Transform3D(mat3x3=world_T_cam_44[:3, :3], translation=world_T_cam_44[:3, 3]))
    rr.log(
        PINHOLE_PATH,
        rr.Pinhole(image_from_camera=K_native_33, width=native_wh[0], height=native_wh[1], camera_xyz=rr.ViewCoordinates.RDF),
    )


def log_source_video(video_decoder: SegmentNvdecDecoder) -> None:
    """Relay the segment's raw AV1 samples once as a VideoStream (the source catalog's own schema)."""
    assert video_decoder.samples is not None and video_decoder.keyframes is not None
    rr.log(VIDEO_PATH, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
    rr.send_columns(
        VIDEO_PATH,
        indexes=[rr.TimeColumn(TIMELINE, duration=video_decoder.times)],
        columns=rr.VideoStream.columns(sample=video_decoder.samples, is_keyframe=video_decoder.keyframes),
    )


def depth_at_lowres(
    depth_outputs: DepthOutputs, K_native_33: Float32[ndarray, "3 3"], native_wh: tuple[int, int]
) -> tuple[Float32[ndarray, "h=192 w=256"], Float32[ndarray, "3 3"]]:
    """The predicted metric depth and the intrinsics of its 256x192 raster; logs that raster's Pinhole."""
    depth_hw: Float32[ndarray, "h=192 w=256"] = rearrange(depth_outputs.depth_b1hw, "1 1 h w -> h w").cpu().numpy()
    K_depth_33: Float32[ndarray, "3 3"] = K_native_33.copy()
    K_depth_33[0] *= DEPTH_HW[1] / native_wh[0]
    K_depth_33[1] *= DEPTH_HW[0] / native_wh[1]
    rr.log(
        PINHOLE_LOWRES_PATH,
        rr.Pinhole(image_from_camera=K_depth_33, width=DEPTH_HW[1], height=DEPTH_HW[0], camera_xyz=rr.ViewCoordinates.RDF),
    )
    return np.ascontiguousarray(depth_hw, dtype=np.float32), K_depth_33
