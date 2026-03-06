# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import UInt8
from simplecv.video_io import VideoReader

try:
    import imageio
    import matplotlib.cm as cm
except ImportError:
    imageio = None
    cm = None


def ensure_even(value):
    return value if value % 2 == 0 else value + 1


def read_video_frames(
    video_path: Path, process_length: int, target_fps: int = -1, max_res: int = -1
) -> tuple[UInt8[np.ndarray, "T H W 3"], int | float]:
    video_reader = VideoReader(video_path)
    original_fps = video_reader.fps
    original_height = video_reader.height
    original_width = video_reader.width

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale: float = max_res / max(original_height, original_width)
        height: int = round(original_height * scale)
        width: int = round(original_width * scale)

    fps: int | float = original_fps if target_fps < 0 else target_fps
    stride: int = max(round(original_fps / fps), 1)

    frames = []
    for frame_count, bgr in enumerate(video_reader):
        if process_length > 0 and frame_count >= process_length:
            break
        if frame_count % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if max_res > 0 and max(original_height, original_width) > max_res:
                rgb = cv2.resize(rgb, (width, height))
            frames.append(rgb)
    frames = np.stack(frames, axis=0)

    return frames, fps
