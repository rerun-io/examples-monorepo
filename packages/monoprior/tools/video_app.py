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
import os
from pathlib import Path

import gradio as gr
import numpy as np

from monopriors.dc_utils import read_video_frames, save_video
from monopriors.relative_depth_models.base_relative_depth import RelativeDepthPrediction
from monopriors.relative_depth_models.video_depth_anything import VideoDepthAnythingPredictor

examples = [
    ["assets/example_videos/davis_rollercoaster.mp4", -1, -1, 1280],
]

video_depth_anything = VideoDepthAnythingPredictor(device="cuda", encoder="vits")


def infer_video_depth(
    input_video: str,
    max_len: int = -1,
    target_fps: int = -1,
    max_res: int = 1280,
    output_dir: str = "./outputs",
    input_size: int = 518,
):
    frames, target_fps = read_video_frames(Path(input_video), max_len, target_fps, max_res)
    depths_pred_list: list[RelativeDepthPrediction] = video_depth_anything(frames, K_33=None)
    disparitys = np.stack([depth_pred.disparity for depth_pred in depths_pred_list], axis=0)

    video_name = os.path.basename(input_video)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_video_path = os.path.join(output_dir, os.path.splitext(video_name)[0] + "_src.mp4")
    depth_vis_path = os.path.join(output_dir, os.path.splitext(video_name)[0] + "_vis.mp4")
    save_video(frames, processed_video_path, fps=target_fps)
    save_video(disparitys, depth_vis_path, fps=target_fps, is_depths=True)

    return [processed_video_path, depth_vis_path]


with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_video = gr.Video(label="Input Video")

        with gr.Column(scale=2), gr.Row(equal_height=True):
            processed_video = gr.Video(
                label="Preprocessed video",
                interactive=False,
                autoplay=True,
                loop=True,
                show_share_button=True,
                scale=5,
            )
            depth_vis_video = gr.Video(
                label="Generated Depth Video",
                interactive=False,
                autoplay=True,
                loop=True,
                show_share_button=True,
                scale=5,
            )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1), gr.Row(equal_height=False):
            with gr.Accordion("Advanced Settings", open=False):
                max_len = gr.Slider(
                    label="max process length",
                    minimum=-1,
                    maximum=1000,
                    value=100,
                    step=1,
                )
                target_fps = gr.Slider(
                    label="target FPS",
                    minimum=-1,
                    maximum=30,
                    value=15,
                    step=1,
                )
                max_res = gr.Slider(
                    label="max side resolution",
                    minimum=480,
                    maximum=1920,
                    value=1280,
                    step=1,
                )
            generate_btn = gr.Button("Generate")
        with gr.Column(scale=2):
            pass

    gr.Examples(
        examples=examples,
        inputs=[input_video, max_len, target_fps, max_res],
        outputs=[processed_video, depth_vis_video],
        fn=infer_video_depth,
        cache_examples="lazy",
    )

    generate_btn.click(
        fn=infer_video_depth,
        inputs=[input_video, max_len, target_fps, max_res],
        outputs=[processed_video, depth_vis_video],
    )


demo.queue()
demo.launch()
