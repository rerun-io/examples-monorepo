# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from einops import rearrange
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.dc_utils import read_video_frames
from monopriors.models.relative_depth.base_relative_depth import RelativeDepthPrediction
from monopriors.models.relative_depth.video_depth_anything import (
    DepthChunkIntermediate,
    FinalDepthResult,
    VideoDepthAnythingPredictor,
)


@dataclass
class VDAConfig:
    rr_config: RerunTyroConfig
    video_path: Path = Path("data/examples/video/davis_rollercoaster.mp4")
    max_len: int = 100
    target_fps: int = -1
    max_res: int = 1280
    input_size: int = 518
    encoder: Literal["vits", "vitl"] = "vits"
    iterate: bool = False


def vda_inference(config: VDAConfig) -> None:
    video_depth_anything = VideoDepthAnythingPredictor(device="cuda", encoder=config.encoder)
    read_output: tuple[UInt8[ndarray, "T H W 3"], float] = read_video_frames(
        config.video_path, config.max_len, config.target_fps, config.max_res
    )
    frames: UInt8[ndarray, "T H W 3"] = read_output[0]

    if not config.iterate:
        depths: list[RelativeDepthPrediction] = video_depth_anything(frames, K_33=None)
        for i, depth_pred in enumerate(depths):
            rr.set_time("frame", sequence=i)
            rr.log("depth", rr.DepthImage(depth_pred.disparity, colormap=rr.components.Colormap.Magma))
    else:
        # Use the iterator-based inference function, larger memory footprint since logging all intermediate results.
        frame_idx = 0
        for result in video_depth_anything.infer_video_depth_iter(frames, K_33=None):
            match result:
                case DepthChunkIntermediate(chunk_index=ci, raw_depth_list=raw_depth_list, progress=progress):
                    # Intermediate result: update visualization with raw (unaligned) depth for this chunk.
                    for depth_pred in raw_depth_list:
                        rr.set_time("frame", sequence=frame_idx)
                        rr.log("depth", rr.DepthImage(depth_pred, colormap=rr.components.Colormap.Magma))
                        frame_idx += 1
                case FinalDepthResult(aligned_depth=aligned_depth):
                    # clear out all previously logged depth images
                    for i in range(frame_idx):
                        rr.set_time("frame", sequence=i)
                        rr.log("depth", rr.Clear(recursive=True))
                    # Final result: update visualization with the final aligned depths.
                    print("Final aligned depth result received.")
                    # Example: iterate over the final aligned depth predictions and log them.
                    for i, depth_pred in enumerate(aligned_depth):
                        rr.set_time("frame", sequence=i)
                        rr.log("depth", rr.DepthImage(depth_pred.depth, rr.components.Colormap.Turbo))
                case _:
                    # Fallback in case of an unexpected type.
                    print("Unexpected result type:", result)
