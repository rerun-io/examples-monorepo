from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Int, UInt8
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm

TIMELINE: str = "video_time"
PREVIEW_SECONDS: float = 1.0


def build_blueprint(video_names: list[str]) -> rrb.Blueprint:
    """Build a compact video-stream vs decoded-image comparison blueprint."""
    rows: list[rrb.Horizontal] = []
    for video_name in video_names:
        video_root: str = f"/videos/{video_name}"
        row: rrb.Horizontal = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{video_root}/video", name=f"{video_name} video"),
            rrb.Spatial2DView(origin=f"{video_root}/decoded", name=f"{video_name} decoded"),
            column_shares=[1, 1],
        )
        rows.append(row)

    preview_start_time: rr.datatypes.TimeInt = rr.datatypes.TimeInt(seconds=0.0)
    preview_end_time: rr.datatypes.TimeInt = rr.datatypes.TimeInt(seconds=PREVIEW_SECONDS)
    preview_time_selection: rrb.components.AbsoluteTimeRange = rrb.components.AbsoluteTimeRange(
        min=preview_start_time,
        max=preview_end_time,
    )
    time_panel: rrb.TimePanel = rrb.TimePanel(
        timeline=TIMELINE,
        play_state="playing",
        loop_mode="selection",
        time_selection=preview_time_selection,
    )
    return rrb.Blueprint(rrb.Tabs(contents=rows), time_panel, collapse_panels=True)


def log_torchcodec_decoded_chunks(
    *,
    chunked_videos: Iterable[list[UInt8[Tensor, "b 3 h w"]]],
    video_names: list[str],
    frame_timestamps_by_video: list[Int[ndarray, "num_frames"]],
    total_frames: int,
    chunk_size: int,
) -> None:
    """Log TorchCodec RGB chunks as decoded-image streams next to native videos."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    total_chunks: int = (total_frames + chunk_size - 1) // chunk_size
    for chunk_idx, videos in enumerate(
        tqdm(
            chunked_videos,
            total=total_chunks,
            desc="TorchCodec images",
            unit="chunk",
        )
    ):
        start_frame_idx: int = chunk_idx * chunk_size
        for video_idx, video in enumerate(
            tqdm(
                videos,
                desc="Videos in chunk",
                leave=False,
            )
        ):
            video_name: str = video_names[video_idx]
            frame_timestamps_ns: Int[ndarray, "num_frames"] = frame_timestamps_by_video[video_idx]
            bgr_nchw: UInt8[Tensor, "b 3 h w"] = video.detach().flip(dims=(1,))
            bgr_nhwc_tensor: UInt8[Tensor, "b h w 3"] = rearrange(bgr_nchw, "b c h w -> b h w c")
            bgr_nhwc: UInt8[ndarray, "b h w 3"] = np.ascontiguousarray(bgr_nhwc_tensor.cpu().numpy(), dtype=np.uint8)
            for local_frame_idx in range(video.shape[0]):
                frame_idx: int = start_frame_idx + local_frame_idx
                if frame_idx >= len(frame_timestamps_ns):
                    continue
                bgr_hwc: UInt8[ndarray, "h w 3"] = bgr_nhwc[local_frame_idx]
                rr.set_time(TIMELINE, duration=float(frame_timestamps_ns[frame_idx]) * 1e-9)
                rr.set_time("frame", sequence=frame_idx)
                rr.log(
                    f"/videos/{video_name}/decoded",
                    rr.Image(bgr_hwc, color_model=rr.ColorModel.BGR).compress(jpeg_quality=80),
                )
