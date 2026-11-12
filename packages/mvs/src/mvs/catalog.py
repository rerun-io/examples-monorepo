"""The ARKitScenes catalog input the mvs apps share, and how they log what every frame carries.

Entity paths mirror the arkitscenes-download rig schema (``ingest/paths.py``); the source video is relayed once, from
the raw AV1 samples the NVDEC decoder already fetched, never re-encoded per frame.
"""

from dataclasses import dataclass

import depth_png
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Float32, Int64, UInt16
from numpy import ndarray
from simplecv.rerun_dataloader import SegmentNvdecDecoder, relay_video_packets

from mvs.blueprint import TIMELINE, EntityPaths
from mvs.depth_engine import DepthOutputs

CLOUD_CATALOG_URL: str = "rerun://api.latest.cloud.rerun.io:443"
FRAME_TIMELINE: str = "frame"
"""The segment's frame index, beside :data:`TIMELINE`."""
NATIVE_FPS: float = 60.0
"""Catalog segments store the wide camera as 60 fps AV1."""

PATHS: EntityPaths = EntityPaths()
"""Entity paths the apps log to (the original live-meshing demo's, with the catalog's camera)."""
CAMERA_PATH: str = PATHS.current_cam
PINHOLE_PATH: str = PATHS.current_pinhole
VIDEO_WIDE: str = PATHS.video
"""The wide camera's AV1 VideoStream in the catalog segments; the apps relay it at the same path."""
PINHOLE_LOWRES_PATH: str = PATHS.depth_pinhole
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
    """Segment ids to process sequentially; empty runs every registered segment (live_mesh takes exactly one)."""


def log_camera(world_T_cam_44: Float32[ndarray, "4 4"], K_native_33: Float32[ndarray, "3 3"], native_wh: tuple[int, int]) -> None:
    """The frame's camera pose and native-resolution Pinhole, at the current time point."""
    rr.log(CAMERA_PATH, rr.Transform3D(mat3x3=world_T_cam_44[:3, :3], translation=world_T_cam_44[:3, 3]))
    rr.log(
        PINHOLE_PATH,
        rr.Pinhole(image_from_camera=K_native_33, width=native_wh[0], height=native_wh[1], camera_xyz=rr.ViewCoordinates.RDF),
    )


def log_source_video(video_decoder: SegmentNvdecDecoder) -> int:
    """Relay the segment's raw AV1 samples once, whole, as a VideoStream (the source catalog's own schema), and return
    the time of its first relayed sample, ns: the relay opens at the first keyframe, which a ``VideoFrameReference``
    into the stream counts from."""
    assert video_decoder.samples is not None and video_decoder.keyframes is not None
    times_ns: Int64[ndarray, " n"] = video_decoder.times.astype("timedelta64[ns]").astype(np.int64)
    relay_video_packets(
        VIDEO_WIDE, TIMELINE, video_decoder.times, video_decoder.samples, video_decoder.keyframes, rr.VideoCodec.AV1, int(times_ns[0]), int(times_ns[-1])
    )
    # relay_video_packets opens at the last keyframe at or before the window start; the window starts at sample 0.
    return int(times_ns[int(np.flatnonzero(np.asarray(video_decoder.keyframes, dtype=bool))[0])])


def log_video_then_blueprint(video_decoder: SegmentNvdecDecoder, blueprint: rrb.Blueprint) -> int:
    """Relay the whole video, then send the blueprint, whose time panel plays ``video_time`` from 0 (a blueprint cannot
    seek, so it must come after the video and before the first depth frame); returns the stream's first sample time."""
    stream_start_ns: int = log_source_video(video_decoder)
    rr.send_blueprint(blueprint)
    return stream_start_ns


def log_depth(depth_hw: Float32[ndarray, "h=192 w=256"], K_depth_33: Float32[ndarray, "3 3"]) -> None:
    """The predicted metric depth as a millimetre PNG under its own 256x192 Pinhole, at the current time point."""
    # The viewer back-projects the depth with its parent's intrinsics: the 256x192 raster's own Pinhole.
    rr.log(
        PINHOLE_LOWRES_PATH,
        rr.Pinhole(image_from_camera=K_depth_33, width=DEPTH_HW[1], height=DEPTH_HW[0], camera_xyz=rr.ViewCoordinates.RDF),
    )
    depth_mm_hw: UInt16[ndarray, "h=192 w=256"] = np.rint(depth_hw * 1000.0).astype(np.uint16)
    rr.log(
        PATHS.depth,
        rr.EncodedDepthImage(blob=depth_png.encode(depth_mm_hw), media_type=rr.components.MediaType.PNG, meter=1000.0),
    )


def depth_at_lowres(
    depth_outputs: DepthOutputs, K_native_33: Float32[ndarray, "3 3"], native_wh: tuple[int, int]
) -> tuple[Float32[ndarray, "h=192 w=256"], Float32[ndarray, "3 3"]]:
    """The predicted metric depth and the intrinsics of its 256x192 raster."""
    depth_hw: Float32[ndarray, "h=192 w=256"] = rearrange(depth_outputs.depth_b1hw, "1 1 h w -> h w").cpu().numpy()
    K_depth_33: Float32[ndarray, "3 3"] = K_native_33.copy()
    K_depth_33[0] *= DEPTH_HW[1] / native_wh[0]
    K_depth_33[1] *= DEPTH_HW[0] / native_wh[1]
    return np.ascontiguousarray(depth_hw, dtype=np.float32), K_depth_33
