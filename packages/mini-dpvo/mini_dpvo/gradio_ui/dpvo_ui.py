"""Gradio web interface for the Mini-DPVO visual odometry demo.

Provides a streaming Rerun viewer that visualizes camera trajectories and
point clouds in real time as DPVO processes an uploaded video.  Camera
intrinsics are auto-estimated via DUSt3R (no calibration file required).

The module follows the **monoprior sync_config pattern**: a module-level
``_CONFIG`` singleton is kept in sync with UI widgets via ``_sync_config``,
and the streaming callback delegates to the shared
:func:`~mini_dpvo.api.inference.run_dpvo_pipeline` generator so CLI and
Gradio always run the same code path.

Rerun streaming follows the **mast3r-slam pattern**:
``@rr.thread_local_stream`` + ``rr.get_thread_local_data_recording()``
with the recording passed through to the pipeline.
"""

from collections.abc import Generator
from pathlib import Path

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
import spaces  # type: ignore
import torch
from gradio_rerun import Rerun
from mini_dust3r.model import AsymmetricCroCo3DStereo

from mini_dpvo.api.inference import (
    DPVOPipelineHandle,
    run_dpvo_pipeline,
)
from mini_dpvo.config import DPVOConfig

# ── Heavy resources (loaded once, survive hot-reload) ───────────────────
if gr.NO_RELOAD:
    NETWORK_PATH: str = "checkpoints/dpvo.pth"
    DEVICE: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    DUST3R_MODEL: AsymmetricCroCo3DStereo = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(DEVICE)

# ── Module-level config singleton ───────────────────────────────────────
_CONFIG: DPVOConfig = DPVOConfig.fast()
"""Module-level config, kept in sync with the UI widgets via ``_sync_config``."""


def _sync_config(preset: str) -> None:
    """Sync UI widget values into the module-level config singleton.

    Args:
        preset: ``"accurate"`` or ``"fast"``.
    """
    global _CONFIG
    _CONFIG = DPVOConfig.accurate() if preset == "accurate" else DPVOConfig.fast()


# ── Streaming callback (mast3r-slam pattern) ────────────────────────────


@rr.thread_local_stream("mini_dpvo")
@torch.no_grad()
def dpvo_streaming_fn(
    video_file_path: str,
    stride: int,
    skip: int,
    jpg_quality: int,
) -> Generator[tuple[bytes | None, str], None, None]:
    """Run DPVO pipeline and stream Rerun bytes to the viewer.

    Uses ``@rr.thread_local_stream`` to create a per-call recording
    (mast3r-slam pattern).  All ``rr.log()`` calls from the nested
    ``run_dpvo_pipeline()`` generator target this recording, and
    ``stream.read()`` yields the accumulated bytes to the embedded
    Rerun viewer.

    Args:
        video_file_path: Local path to the uploaded video file.
        stride: Keep every *stride*-th frame.
        skip: Number of leading frames to discard.
        jpg_quality: JPEG quality for image logging (unused by pipeline
            directly — reserved for future per-frame quality control).

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    stream: rr.BinaryStream = rr.binary_stream()
    recording: rr.RecordingStream | None = rr.get_thread_local_data_recording()

    blueprint: rrb.Blueprint = rrb.Blueprint(collapse_panels=True)
    rr.send_blueprint(blueprint)

    handle: DPVOPipelineHandle = DPVOPipelineHandle()

    for msg in run_dpvo_pipeline(
        dpvo_config=_CONFIG,
        network_path=NETWORK_PATH,
        imagedir=video_file_path,
        calib=None,
        stride=stride,
        skip=skip,
        dust3r_model=DUST3R_MODEL,
        handle=handle,
        recording=recording,
    ):
        yield stream.read(), msg

    elapsed: str = f"{handle.elapsed_time:.2f}s" if handle.prediction else "N/A"
    yield stream.read(), f"Complete ({elapsed})"


dpvo_streaming_fn = spaces.GPU(dpvo_streaming_fn)


# ── UI helpers ──────────────────────────────────────────────────────────


def _switch_to_outputs():
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def _switch_to_inputs():
    """Switch the Gradio Tabs component to the Inputs tab."""
    return gr.update(selected="inputs")


# ── Factory function ────────────────────────────────────────────────────


def main() -> gr.Blocks:
    """Build and return the Mini-DPVO Gradio app.

    Layout:
        - **Left column** (scale=1): Tabs with Inputs (file upload, run
          button, config accordion) and Outputs (status + stop button);
          example sets below.
        - **Right column** (scale=5): Embedded Rerun viewer.

    Click chain::

        click -> _switch_to_outputs -> _sync_config -> dpvo_streaming_fn

    Returns:
        The assembled ``gr.Blocks`` instance ready for ``.queue().launch()``.
    """
    rr_viewer = Rerun(
        streaming=True,
        panel_states={
            "time": "collapsed",
            "blueprint": "collapsed",
            "selection": "collapsed",
        },
        height=800,
    )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        video_input = gr.File(
                            file_count="single",
                            file_types=[".mp4", ".mov", ".MOV", ".webm"],
                            label="Input Video",
                        )
                        run_btn = gr.Button("Run DPVO", variant="primary")

                        with gr.Accordion("Config", open=False):
                            preset_dropdown = gr.Dropdown(
                                label="Preset",
                                choices=["accurate", "fast"],
                                value="fast",
                            )
                            stride_slider = gr.Slider(
                                label="Frame Stride",
                                minimum=1,
                                maximum=5,
                                step=1,
                                value=4,
                            )
                            skip_number = gr.Number(
                                label="Skip Frames",
                                value=0,
                                precision=0,
                            )
                            jpg_quality_radio = gr.Radio(
                                label="JPEG Quality %",
                                choices=[10, 50, 90],
                                value=90,
                                type="value",
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(
                            label="Status",
                            value="Upload a video to begin.",
                            interactive=False,
                        )
                        stop_btn = gr.Button("Stop")

                # Pre-populate the examples gallery from bundled video directories
                example_dpvo_dir: Path = Path("data/movies")
                example_iphone_dir: Path = Path("data/iphone")
                example_video_paths: list[Path] = sorted(example_iphone_dir.glob("*.MOV")) + sorted(
                    example_dpvo_dir.glob("*.MOV")
                )
                example_video_paths_str: list[str] = [str(path) for path in example_video_paths]

                examples = gr.Examples(
                    examples=[[path] for path in example_video_paths_str],
                    inputs=[video_input],
                    cache_examples=False,
                )

            with gr.Column(scale=5):
                rr_viewer.render()

        # Switch to Inputs tab when examples populate the input
        if hasattr(examples, "load_input_event"):
            examples.load_input_event.then(
                fn=_switch_to_inputs,
                inputs=None,
                outputs=[tabs],
                api_visibility="private",
            )

        # Click chain: switch tab -> sync config -> run pipeline
        run_event = run_btn.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(
            fn=_sync_config,
            inputs=[preset_dropdown],
        ).then(
            fn=dpvo_streaming_fn,
            inputs=[video_input, stride_slider, skip_number, jpg_quality_radio],
            outputs=[rr_viewer, status_text],
        )

        stop_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            cancels=[run_event],
        )

    return demo
