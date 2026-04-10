"""Gradio web interface for the Mini-DPVO visual odometry demo.

Provides a streaming Rerun viewer that visualizes camera trajectories and
point clouds in real time as DPVO processes an uploaded video. Camera
intrinsics are auto-estimated via DUSt3R (no calibration file required).

The module exposes a single ``dpvo_block`` :class:`gr.Blocks` instance that
can be launched directly or composed into a larger Gradio application.
"""

try:
    import spaces  # type: ignore # noqa: F401

    IN_SPACES: bool = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

from collections.abc import Generator
from multiprocessing import Process, Queue
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import gradio as gr
import mmcv
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gradio_rerun import Rerun
from jaxtyping import Float32, Float64, UInt8
from mini_dust3r.model import AsymmetricCroCo3DStereo
from tqdm import tqdm

from mini_dpvo.api.inference import (
    calculate_num_frames,
    calib_from_dust3r,
    create_reader,
    log_trajectory,
)
from mini_dpvo.config import cfg as base_cfg
from mini_dpvo.dpvo import DPVO

# Heavy resources (model weights, device selection) are loaded once and
# guarded by ``gr.NO_RELOAD`` so they survive Gradio's hot-reload cycles.
if gr.NO_RELOAD:
    NETWORK_PATH: str = "checkpoints/dpvo.pth"
    DEVICE: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    MODEL: AsymmetricCroCo3DStereo = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(DEVICE)


@rr.thread_local_stream("mini_dpvo")
@torch.no_grad()
def run_dpvo(
    video_file_path: str,
    jpg_quality: str,
    stride: int = 1,
    skip: int = 0,
    config_type: Literal["accurate", "fast"] = "accurate",
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> Generator[tuple[bytes, float], None, None]:
    """Run DPVO on a video and stream Rerun binary data to the viewer.

    This is a Gradio streaming callback: it ``yield`` s
    ``(rerun_bytes, elapsed_seconds)`` after every initialized SLAM step so
    the Rerun viewer updates incrementally.

    The function auto-estimates camera intrinsics with DUSt3R, reads frames
    via a background process, and feeds them through the DPVO SLAM system.

    Args:
        video_file_path: Local path to the uploaded video file.
        jpg_quality: JPEG quality for image logging (passed through to
            :func:`log_trajectory`).
        stride: Keep every *stride*-th frame.
        skip: Number of leading frames to discard.
        config_type: Which DPVO config to load -- ``"accurate"`` merges
            ``config/default.yaml``, ``"fast"`` merges ``config/fast.yaml``.
        progress: Gradio progress tracker for the calibration step.

    Yields:
        A tuple of ``(rerun_binary_bytes, elapsed_time_seconds)`` after each
        SLAM iteration once the system is initialized.
    """
    # Create a binary stream to incrementally send data to the Rerun viewer
    stream: rr.BinaryStream = rr.binary_stream()
    parent_log_path: Path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    blueprint: rrb.Blueprint = rrb.Blueprint(
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)

    if config_type == "accurate":
        base_cfg.merge_from_file("config/default.yaml")
    elif config_type == "fast":
        base_cfg.merge_from_file("config/fast.yaml")
    else:
        raise ValueError("Invalid config type")
    base_cfg.BUFFER_SIZE = 2048

    slam: DPVO | None = None
    start_time: float = timer()
    queue: Queue = Queue(maxsize=8)

    reader: Process = create_reader(video_file_path, None, stride, skip, queue)
    reader.start()

    # get the first frame
    progress(progress=0.1, desc="Estimating Camera Intrinsics")
    _, bgr_hw3, _ = queue.get()
    K_33_pred: Float64[np.ndarray, "3 3"] = calib_from_dust3r(bgr_hw3, MODEL, DEVICE)
    intri_np: Float64[np.ndarray, "4"] = np.array(
        [K_33_pred[0, 0], K_33_pred[1, 1], K_33_pred[0, 2], K_33_pred[1, 2]]
    )

    num_frames: int = calculate_num_frames(video_file_path, stride, skip)
    path_list: list[list[float]] = []

    with tqdm(total=num_frames, desc="Processing Frames") as pbar:
        while True:
            timestep: int
            bgr_hw3: UInt8[np.ndarray, "h w 3"]
            intri_np: Float64[np.ndarray, "4"]
            (timestep, bgr_hw3, _) = queue.get()
            # queue will have a (-1, image, intrinsics) tuple when the reader is done
            if timestep < 0:
                break

            rr.set_time("timestep", sequence=timestep)

            bgr_3hw: UInt8[torch.Tensor, "h w 3"] = (
                torch.from_numpy(bgr_hw3).permute(2, 0, 1).cuda()
            )
            intri_torch: Float64[torch.Tensor, "4"] = torch.from_numpy(intri_np).cuda()

            if slam is None:
                h: int
                w: int
                _, h, w = bgr_3hw.shape
                slam = DPVO(base_cfg, NETWORK_PATH, ht=h, wd=w)

            slam(timestep, bgr_3hw, intri_torch)
            pbar.update(1)

            if slam.is_initialized:
                poses: Float32[torch.Tensor, "buffer_size 7"] = slam.poses_
                points: Float32[torch.Tensor, "buffer_size*num_patches 3"] = (
                    slam.points_
                )
                colors: UInt8[torch.Tensor, "buffer_size num_patches 3"] = slam.colors_
                path_list = log_trajectory(
                    parent_log_path,
                    poses,
                    points,
                    colors,
                    intri_np,
                    bgr_hw3,
                    path_list,
                    jpg_quality,
                )
                yield stream.read(), timer() - start_time


# Wrap the callback with HuggingFace Spaces GPU decorator when running
# on Zero GPU infrastructure.
if IN_SPACES:
    run_dpvo = spaces.GPU(run_dpvo)


def on_file_upload(video_file_path: str) -> str:
    """Extract and format basic video metadata for the Gradio info panel.

    Called when a user uploads a new video file. Returns a Markdown string
    showing frame count, FPS, and duration.

    Args:
        video_file_path: Local path to the uploaded video file.

    Returns:
        A Markdown-formatted string with video statistics.
    """
    video_reader: mmcv.VideoReader = mmcv.VideoReader(video_file_path)
    video_info: str = f"""
    **Video Info:**
    - Number of Frames: {video_reader.frame_cnt}
    - FPS: {round(video_reader.fps)}
    - Duration: {(video_reader.frame_cnt/video_reader.fps):.2f} seconds
    """
    return video_info


# ---------------------------------------------------------------------------
# Gradio Blocks UI layout
# ---------------------------------------------------------------------------
# ``dpvo_block`` is the top-level Blocks instance. It can be launched
# standalone via ``dpvo_block.launch()`` or mounted inside a larger app.
with gr.Blocks(
    css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
    title="Mini-DPVO Demo",
) as dpvo_block:
    gr.HTML('<h2 style="text-align: center;">Mini-DPVO Demo</h2>')
    gr.HTML(
        '<p style="text-align: center;">Unofficial DPVO demo using the mini-dpvo. Learn more about mini-dpvo <a href="https://github.com/pablovela5620/mini-dpvo">here</a>.</p>'
    )
    with gr.Column():
        with gr.Row():
            video_input = gr.File(
                height=100,
                file_count="single",
                file_types=[".mp4", ".mov", ".MOV", ".webm"],
                label="Video File",
            )
            with gr.Column():
                video_info = gr.Markdown(
                    value="""
                **Video Info:**
                """
                )
                time_taken = gr.Number(
                    label="Time Taken (s)", precision=2, interactive=False
                )
        with gr.Accordion(label="Advanced", open=False), gr.Row():
            jpg_quality = gr.Radio(
                label="JPEG Quality %: Lower quality means faster streaming",
                choices=[10, 50, 90],
                value=90,
                type="value",
            )
            stride = gr.Slider(
                label="Stride: How many frames to sample between each prediction",
                minimum=1,
                maximum=5,
                step=1,
                value=5,
            )
            skip = gr.Number(
                label="Skip: How many frames to skip at the beginning",
                value=0,
                precision=0,
            )
            config_type = gr.Dropdown(
                label="Config Type: Choose between accurate and fast",
                value="fast",
                choices=["accurate", "fast"],
                max_choices=1,
            )
        with gr.Row():
            start_btn = gr.Button("Run")
            stop_btn = gr.Button("Stop")
        rr_viewer = Rerun(height=600, streaming=True)

        # Pre-populate the examples gallery from bundled video directories
        base_example_params: list[int | str] = [50, 4, 0, "fast"]
        example_dpvo_dir: Path = Path("data/movies")
        example_iphone_dir: Path = Path("data/iphone")
        example_video_paths: list[Path] = sorted(example_iphone_dir.glob("*.MOV")) + sorted(
            example_dpvo_dir.glob("*.MOV")
        )
        example_video_paths_str: list[str] = [str(path) for path in example_video_paths]

        gr.Examples(
            examples=[[path, *base_example_params] for path in example_video_paths_str],
            inputs=[video_input, jpg_quality, stride, skip, config_type],
            cache_examples=False,
        )

        click_event = start_btn.click(
            fn=run_dpvo,
            inputs=[video_input, jpg_quality, stride, skip, config_type],
            outputs=[rr_viewer, time_taken],
            api_visibility="private",
        )

        stop_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            cancels=[click_event],
        )

        video_input.change(
            fn=on_file_upload,
            inputs=[video_input],
            outputs=[video_info],
            api_visibility="private",
        )
