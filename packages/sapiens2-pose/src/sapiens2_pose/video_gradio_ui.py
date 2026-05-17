"""Sapiens2 video pose Gradio UI."""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import cast

import gradio as gr
import rerun as rr
import spaces
import torch
from gradio.data_classes import FileData
from gradio_rerun import Rerun

from .api.metadata import KeypointSchemaName
from .api.runtime import DEFAULT_BBOX_THR, DEFAULT_MODEL_SIZE, DEFAULT_NMS_THR, POSE_MODELS, ModelSize
from .api.sam3_tracking import DEFAULT_SAM3_MEMORY_RETENTION_FRAMES, DEFAULT_SAM3_MIN_MASK_AREA_PX
from .api.video import PoseBackend, SapiensVideoPoseConfig, TrackingBackend, run_video_pose_pipeline

DEFAULT_SIZE: ModelSize = DEFAULT_MODEL_SIZE
DEFAULT_SCHEMA: KeypointSchemaName = "coco133"
DEFAULT_TRACKING_BACKEND: TrackingBackend = "sam3_tracking"
DEFAULT_TENSORRT_ENGINE_ENV_VAR: str = "SAPIENS2_POSE_TENSORRT_ENGINE_PATH"
DEFAULT_TENSORRT_ENGINE_FILENAME: str = "sapiens2_0_4b_pose_static_b1_bf16_current_static_graph.trt"


def _server_port() -> int:
    env_port: str | None = os.environ.get("GRADIO_SERVER_PORT")
    if env_port is not None:
        return int(env_port)

    for port in range(7861, 8060):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("Cannot find empty localhost port in range 7861-8059.")


def _video_input_to_path(video: str | FileData | None) -> Path:
    if video is None:
        raise gr.Error("No video provided.")
    if isinstance(video, FileData):
        return Path(video.path)
    return Path(video)


def _remux_mov_for_rerun(video_path: Path, output_dir: Path) -> Path:
    suffix: str = video_path.suffix.lower()
    if suffix not in {".mov", ".qt"}:
        return video_path

    remuxed_path: Path = output_dir / f"{video_path.stem}_rerun.mp4"
    command: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        str(remuxed_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr_tail: str = exc.stderr[-2000:] if exc.stderr else ""
        raise gr.Error(f"Could not remux MOV input for Rerun video logging. ffmpeg stderr: {stderr_tail}") from exc
    return remuxed_path


def _default_tensorrt_engine_path() -> str:
    """Return the app's default TensorRT engine path."""
    explicit_engine_path: str | None = os.environ.get(DEFAULT_TENSORRT_ENGINE_ENV_VAR)
    if explicit_engine_path is not None:
        return explicit_engine_path

    xdg_cache_home: str | None = os.environ.get("XDG_CACHE_HOME")
    cache_root: Path = Path(xdg_cache_home).expanduser() if xdg_cache_home is not None else Path.home() / ".cache"
    engine_path: Path = cache_root / "sapiens2-pose" / "tensorrt" / DEFAULT_TENSORRT_ENGINE_FILENAME
    return str(engine_path)


def _format_video_inference_status(*, status: str, backend_label: str, elapsed_seconds: float) -> str:
    """Append whole-video inference timing to a completion status."""
    return f"{status} {backend_label} inference took {elapsed_seconds:.2f}s for the whole video."


@rr.thread_local_stream("sapiens2_pose_video")
def predict_video_ui(
    video_path: str | FileData | None,
    size: str,
    keypoint_schema: str,
    tracking_backend: str,
    bbox_thr: float,
    nms_thr: float,
    kpt_thr: float,
    use_tensorrt: bool,
    tensorrt_engine_path: str | None,
    sam3_min_mask_area_px: float | int | None,
    sam3_memory_retention_frames: float | int | None,
    max_frames: float | int | None,
    frame_stride: float | int | None,
) -> Generator[tuple[bytes | None, str], None, None]:
    """Run video pose inference and stream Rerun viewer bytes with status."""
    stream: rr.BinaryStream = rr.binary_stream()
    recording: rr.RecordingStream | None = rr.get_thread_local_data_recording()
    temp_dir: Path = Path(tempfile.mkdtemp(prefix="sapiens2_pose_video_"))
    input_path: Path = _video_input_to_path(video_path)
    rerun_video_path: Path = _remux_mov_for_rerun(input_path, temp_dir)

    model_size: ModelSize = cast(ModelSize, size)
    schema_name: KeypointSchemaName = cast(KeypointSchemaName, keypoint_schema)
    tracking_backend_name: TrackingBackend = cast(TrackingBackend, tracking_backend)
    pose_backend: PoseBackend = "tensorrt" if use_tensorrt else "pytorch"
    backend_label: str = "TensorRT" if use_tensorrt else "PyTorch"
    resolved_tensorrt_engine_path: Path | None = None
    if use_tensorrt:
        if model_size != "0.4B":
            raise gr.Error("TensorRT video mode currently expects a 0.4B pose engine.")
        engine_path_text: str = str(tensorrt_engine_path or _default_tensorrt_engine_path()).strip()
        if engine_path_text == "":
            raise gr.Error(f"TensorRT video mode requires an engine path or {DEFAULT_TENSORRT_ENGINE_ENV_VAR}.")
        resolved_tensorrt_engine_path = Path(engine_path_text).expanduser()
        if not resolved_tensorrt_engine_path.exists():
            raise gr.Error(f"TensorRT engine does not exist: {resolved_tensorrt_engine_path}")
    max_frames_int: int | None = None
    if max_frames is not None and int(max_frames) > 0:
        max_frames_int = int(max_frames)
    frame_stride_int: int = max(1, int(frame_stride or 1))
    sam3_min_mask_area_px_int: int = max(1, int(sam3_min_mask_area_px or DEFAULT_SAM3_MIN_MASK_AREA_PX))
    sam3_memory_retention_frames_int: int = max(
        1,
        int(sam3_memory_retention_frames or DEFAULT_SAM3_MEMORY_RETENTION_FRAMES),
    )

    rrd_path: Path = temp_dir / "sapiens2_pose_video.rrd"
    config: SapiensVideoPoseConfig = SapiensVideoPoseConfig(
        video_path=rerun_video_path,
        rrd_path=rrd_path,
        model_size=model_size,
        keypoint_schema=schema_name,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        max_frames=max_frames_int,
        frame_stride=frame_stride_int,
        kpt_thr=kpt_thr,
        tracking_backend=tracking_backend_name,
        pose_backend=pose_backend,
        tensorrt_engine_path=resolved_tensorrt_engine_path,
        sam3_min_mask_area_px=sam3_min_mask_area_px_int,
        sam3_memory_retention_frames=sam3_memory_retention_frames_int,
    )
    source_note: str = " Remuxed MOV to MP4 for Rerun." if rerun_video_path != input_path else ""
    started_at: float = time.perf_counter()
    for status in run_video_pose_pipeline(config, recording=recording, save_rrd=False):
        status_with_note: str = f"{status}{source_note}" if source_note and status.startswith(("Loaded", "Complete")) else status
        if status.startswith("Complete"):
            elapsed_seconds: float = time.perf_counter() - started_at
            status_with_note = _format_video_inference_status(
                status=status_with_note,
                backend_label=backend_label,
                elapsed_seconds=elapsed_seconds,
            )
        yield stream.read(), status_with_note


predict_video_ui = spaces.GPU(duration=120)(predict_video_ui)


def _reset_video_outputs() -> tuple[None, str]:
    return None, "Starting Sapiens2 video pose estimation..."


CUSTOM_CSS = """
:root, body, .gradio-container, button, input, select, textarea,
.gradio-container *:not(code):not(pre) {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
}
#title { text-align: center; font-size: 36px; font-weight: 700; margin: 24px 0 4px; }
#subtitle { text-align: center; font-size: 12px; color: #64748b; margin: 0 0 24px; text-transform: uppercase; }
"""


with gr.Blocks(title="Sapiens2 Video Pose") as demo:
    gr.HTML('<div id="title">Sapiens2: Video Pose</div><div id="subtitle">Streaming Video Pose</div>')

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(
                label="Input Video",
                file_types=[".mp4", ".mov", ".webm"],
                type="filepath",
            )
            run = gr.Button("Run Video Pose", variant="primary")

            with gr.Accordion("Config", open=True):
                kpt_thr = gr.Slider(
                    0.0,
                    1.0,
                    value=0.3,
                    step=0.05,
                    label="Keypoint Threshold",
                )
                size = gr.Radio(
                    choices=list(POSE_MODELS.keys()),
                    value=DEFAULT_SIZE,
                    label="Model",
                )
                keypoint_schema = gr.Radio(
                    choices=["coco133", "sapiens308"],
                    value=DEFAULT_SCHEMA,
                    label="Keypoint Schema",
                )
                use_tensorrt = gr.Checkbox(
                    value=False,
                    label="Use TensorRT Backend",
                )
                tensorrt_engine_path = gr.Textbox(
                    value=_default_tensorrt_engine_path(),
                    label="TensorRT Engine Path",
                    placeholder=f"Set {DEFAULT_TENSORRT_ENGINE_ENV_VAR} or paste a .trt path",
                )
                tracking_backend = gr.Radio(
                    choices=["sam3_tracking", "detr_per_frame"],
                    value=DEFAULT_TRACKING_BACKEND,
                    label="Tracking Backend",
                )
                bbox_thr = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_BBOX_THR,
                    step=0.05,
                    label="DETR Box Threshold",
                )
                nms_thr = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_NMS_THR,
                    step=0.05,
                    label="DETR NMS IoU",
                )
                sam3_min_mask_area_px = gr.Number(
                    value=DEFAULT_SAM3_MIN_MASK_AREA_PX,
                    precision=0,
                    label="SAM3 Min Mask Area Px",
                )
                sam3_memory_retention_frames = gr.Number(
                    value=DEFAULT_SAM3_MEMORY_RETENTION_FRAMES,
                    precision=0,
                    label="SAM3 Memory Retention Frames",
                )
                max_frames = gr.Number(
                    value=0,
                    precision=0,
                    label="Max Frames",
                )
                frame_stride = gr.Number(
                    value=1,
                    precision=0,
                    label="Frame Stride",
                )

            status_text = gr.Textbox(
                label="Status",
                value="Select a video to begin.",
                interactive=False,
            )
        with gr.Column(scale=4):
            viewer = Rerun(
                label="Video Pose",
                streaming=True,
                panel_states={
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=800,
            )

    run.click(
        fn=_reset_video_outputs,
        inputs=None,
        outputs=[viewer, status_text],
        api_visibility="private",
    ).then(
        fn=predict_video_ui,
        inputs=[
            video_input,
            size,
            keypoint_schema,
            tracking_backend,
            bbox_thr,
            nms_thr,
            kpt_thr,
            use_tensorrt,
            tensorrt_engine_path,
            sam3_min_mask_area_px,
            sam3_memory_retention_frames,
            max_frames,
            frame_stride,
        ],
        outputs=[viewer, status_text],
    )


def launch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    demo.launch(
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        server_port=_server_port(),
        show_error=True,
    )


if __name__ == "__main__":
    launch()
