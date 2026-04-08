"""Gradio UI for MASt3R-SLAM.

Provides an interactive web interface for running the MASt3R-SLAM pipeline.
The left panel holds video upload, a run button, and config; the right panel
streams results into an embedded Rerun viewer.

The actual tracking loop lives in :func:`mast3r_slam.api.inference.run_slam_pipeline`.
This module only handles Gradio-specific concerns: file validation, recording
setup, binary streaming, and nerfstudio output zipping.
"""

import contextlib
import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path

import beartype
import gradio as gr
import rerun as rr
import torch
import torch.multiprocessing as mp
from gradio_rerun import Rerun

from mast3r_slam.api.inference import SlamPipelineHandle, run_slam_pipeline
from mast3r_slam.frame import Mode, SharedStates
from mast3r_slam.mast3r_utils import load_mast3r

# Global reference to the active states so the stop button can signal termination.
# The SlamBackend context manager handles all cleanup.
active_states: SharedStates | None = None

# Initialize multiprocessing start method (spawn required for CUDA).
# If already set (e.g. by a parent process), reuse the existing setting.
with contextlib.suppress(RuntimeError):
    mp.set_start_method("spawn")

DEVICE: str = "cuda:0"
model = load_mast3r(device=DEVICE)
model.share_memory()


def stop_streaming():
    """Signal the backend to terminate.  Actual cleanup is handled by the
    ``SlamBackend`` context manager in the generator function."""
    global active_states
    if active_states is not None:
        try:
            active_states.set_mode(Mode.TERMINATED)
        except beartype.roar.BeartypeException:
            raise
        except (OSError, BrokenPipeError, EOFError):
            pass
    return None


def _mov_to_mp4(video_path: Path) -> Path:
    """Convert a MOV file to MP4 using ffmpeg."""
    mp4_path: Path = video_path.with_suffix(".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-c:v", "copy", "-an", "-y", str(mp4_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Failed to convert MOV to MP4: {e.stderr.decode()}") from e
    return mp4_path


@rr.thread_local_stream("rerun_example_streaming_blur")
def streaming_mast3r_slam_fn(
    video_file: str,
    subsample: int,
    progress=gr.Progress(),  # noqa: B008
) -> Generator[tuple[bytes | None, str | None, str], None, None]:
    """Gradio streaming callback that runs the SLAM pipeline.

    Uses ``@rr.thread_local_stream`` to create a per-call recording.
    All ``rr.log()`` calls from the nested ``run_slam_pipeline()``
    generator target this recording, and ``stream.read()`` yields
    the accumulated bytes to the embedded Rerun viewer.

    Args:
        video_file: Path to the uploaded video file.
        subsample: Frame subsample rate.
        progress: Gradio progress callback.

    Yields:
        Tuple of (Rerun binary stream bytes, zip file path or None, status message).
    """
    global active_states

    stream = rr.binary_stream()
    recording: rr.RecordingStream = rr.get_thread_local_data_recording()

    try:
        video_path: Path = Path(video_file)
    except beartype.roar.BeartypeCallHintParamViolation:
        raise gr.Error(  # noqa: B904
            "Did you make sure the zipfile finished uploading?. Try to hit run again.",
            duration=20,
        )
    except Exception as e:
        raise gr.Error(  # noqa: B904
            f"Error: {e}\n Did you wait for zip file to upload?", duration=20
        )

    # Convert MOV to MP4 if needed.
    if video_path.suffix.lower() == ".mov":
        video_path = _mov_to_mp4(video_path)

    # Enable TF32 for faster matmuls on Ampere+ GPUs; disable autograd
    # since the entire pipeline runs in inference mode.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    progress(0.05, desc="Loading pipeline")
    handle: SlamPipelineHandle = SlamPipelineHandle()

    # Nerfstudio export happens inside the generator (before
    # SlamBackend.__exit__) because SharedKeyframes needs the mp.Manager.
    ns_output_dir: Path = video_path.parent / "nerfstudio-output"

    for msg in run_slam_pipeline(
        model=model,
        dataset_path=str(video_path),
        config_path="config/base.yaml",
        device=DEVICE,
        subsample=subsample,
        ns_save_path=ns_output_dir,
        handle=handle,
        recording=recording,
    ):
        active_states = handle.states
        yield stream.read(), None, msg

    # Post-loop: either stopped early or completed.
    if handle.stopped_early:
        yield stream.read(), None, "MASt3R-SLAM stopped"
        active_states = None
        return

    # Nerfstudio export already happened inside the generator.
    # Only the zip step needs to happen here.
    zip_output_path: Path = video_path.parent / "nerfstudio-output.zip"

    try:
        if ns_output_dir.exists():
            rr.log("world/logs", rr.TextLog(f"Zipping nerfstudio output to {zip_output_path}", level="INFO"))
            shutil.make_archive(
                base_name=str(zip_output_path.with_suffix("")),
                format="zip",
                root_dir=video_path.parent,
                base_dir="nerfstudio-output",
            )
    except Exception as e:
        raise gr.Error(f"Failed to zip nerfstudio output: {e}") from e

    assert zip_output_path.exists(), f"Zip file {zip_output_path} does not exist"

    rr.log("world/logs", rr.TextLog("Finished processing", level="INFO"))
    yield stream.read(), str(zip_output_path), "MASt3R-SLAM complete"

    active_states = None


def _switch_to_outputs():
    return gr.update(selected="outputs")


def _switch_to_inputs():
    return gr.update(selected="inputs")


def _reset_run_outputs():
    return None, "Starting MASt3R-SLAM..."


with gr.Blocks() as mast3r_slam_block:
    with gr.Row():
        with gr.Column(scale=1):
            tabs = gr.Tabs(selected="inputs")
            with tabs:
                with gr.TabItem("Inputs", id="inputs"):
                    gr.Markdown(
                        "Upload a video, preview it in the Rerun viewer, then run the full "
                        "MASt3R-SLAM pipeline."
                    )
                    video_file = gr.File(
                        label="Input Video",
                        file_types=[".mp4", ".mov", ".MOV"],
                    )
                    run_slam_btn = gr.Button("Run MASt3R-SLAM", variant="primary")

                    with gr.Accordion("Config", open=False):
                        subsample_slider = gr.Slider(
                            label="Frame Subsample",
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=4,
                        )
                        gr.Markdown(
                            "Base config file: `config/base.yaml`\n\n"
                            "Advanced calibration and backend settings remain in the YAML "
                            "for now so the UI does not expose partially applied controls."
                        )

                with gr.TabItem("Outputs", id="outputs"):
                    status_text = gr.Textbox(
                        label="Status",
                        value="Upload a video to begin.",
                        interactive=False,
                    )
                    stop_slam_btn = gr.Button("Stop MASt3R-SLAM")
                    output_zip_file = gr.File(
                        label="Download Nerfstudio Output",
                        file_count="single",
                    )

                examples = gr.Examples(
                    examples=[
                    ["data/normal-apt-tour.mp4"],
                ],
                inputs=[video_file],
                cache_examples=False,
            )

        with gr.Column(scale=5):
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=800,
            )

    if hasattr(examples, "load_input_event"):
        examples.load_input_event.then(
            fn=_switch_to_inputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        )

    run_event = run_slam_btn.click(
        fn=_switch_to_outputs,
        inputs=None,
        outputs=[tabs],
        api_visibility="private",
    ).then(
        fn=_reset_run_outputs,
        inputs=None,
        outputs=[output_zip_file, status_text],
        api_visibility="private",
    ).then(
        fn=streaming_mast3r_slam_fn,
        inputs=[
            video_file,
            subsample_slider,
        ],
        outputs=[viewer, output_zip_file, status_text],
    )

    stop_slam_btn.click(
        fn=stop_streaming,
        inputs=[],
        outputs=[],
        cancels=[run_event],
    ).then(
        fn=lambda: "MASt3R-SLAM stopped",
        inputs=None,
        outputs=[status_text],
        api_visibility="private",
    )
