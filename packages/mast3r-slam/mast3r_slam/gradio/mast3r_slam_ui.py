"""Gradio UI for MASt3R-SLAM.

Provides an interactive web interface for running the MASt3R-SLAM pipeline.
The left panel holds video upload, a run button, and config; the right panel
streams results into an embedded Rerun viewer.

The app loads the model with :func:`mast3r_slam.api.inference.load_slam_model` and runs
:func:`mast3r_slam.api.inference.run_slam_pipeline`, the same two functions the CLI calls.
This module only handles Gradio-specific concerns: file validation, recording
setup, binary streaming, and nerfstudio output zipping.

The SLAM hyper-parameters are a module-level ``_CONFIG`` (a :class:`~mast3r_slam.config.SlamConfig`) that
``_sync_config`` rebuilds from the Config widgets when Run is clicked, before the run starts.
"""

import dataclasses
import shutil
import tempfile
import threading
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TypeAlias

import beartype
import gradio as gr
import rerun as rr
from gradio_rerun import Rerun
from mast3r.model import AsymmetricMASt3R

from mast3r_slam.api.inference import APPLICATION_ID, SlamSettings, load_slam_dataset, load_slam_model, run_slam_pipeline
from mast3r_slam.config import SLAM_PRESETS, SlamConfig, SlamPresetName
from mast3r_slam.mast3r_utils import prepare_trt_engines
from mast3r_slam.rerun_log_utils import LOG_PATH

EXAMPLE_VIDEOS: list[str] = ["data/livingroom-tour.mp4", "data/normal-apt-tour.mp4"]

engine_status: str = "Preparing TensorRT engines for the example videos..."
"""Shown above the Run button while :func:`_prepare_example_engines` runs."""
engines_ready: threading.Event = threading.Event()

VIDEO_PRESETS: list[SlamPresetName] = [name for name, (_, cfg) in SLAM_PRESETS.items() if not cfg.use_calib]
"""The presets an uploaded video can run: a video carries no calibration."""
DEFAULT_PRESET: SlamPresetName = "fast"

_CONFIG: SlamConfig = SLAM_PRESETS[DEFAULT_PRESET][1]
"""Module-level SLAM config, kept in sync with the Config widgets via ``_sync_config``."""


def _sync_config(preset: SlamPresetName, subsample: int | float, single_thread: bool) -> None:
    """Rebuild ``_CONFIG`` from the Config widgets: the preset, with the widgets' values over it.

    Args:
        preset: Name of a preset in ``SLAM_PRESETS``.
        subsample: Keep every n-th frame.
        single_thread: Run frontend and backend in lockstep.
    """
    global _CONFIG
    _CONFIG = dataclasses.replace(SLAM_PRESETS[preset][1], single_thread=single_thread).with_subsample(int(subsample))


def _preset_values(preset: SlamPresetName) -> tuple[int, bool]:
    """The Config widgets' values for ``preset``: subsample, single_thread."""
    cfg: SlamConfig = SLAM_PRESETS[preset][1]
    return cfg.dataset.subsample, cfg.single_thread


def _prepare_example_engines(model: AsymmetricMASt3R) -> None:
    """Build (first time on a machine) or load the TensorRT engines for the example videos' frame sizes.

    Runs on a thread at app start, so the first Run on an example does not wait for them. A Run that starts earlier
    waits for this thread (``engines_ready``): an engine's CUDA graph capture must not overlap a run's threads.
    """
    global engine_status
    try:
        sizes: list[tuple[int, int]] = sorted({load_slam_dataset(SlamSettings(dataset=path)).get_img_shape()[0] for path in EXAMPLE_VIDEOS})
        for n, (h, w) in enumerate(sizes, start=1):
            engine_status = f"Preparing TensorRT engines for {w}x{h} frames ({n}/{len(sizes)}); a first build on this machine takes a few minutes..."
            prepare_trt_engines(model, (h, w))
        engine_status = f"TensorRT engines ready for {', '.join(f'{w}x{h}' for h, w in sizes)} frames."
    except beartype.roar.BeartypeException:
        raise
    except Exception as exc:
        engine_status = f"Preparing TensorRT engines failed ({exc}); runs build them on first use."
        raise
    finally:
        engines_ready.set()


def _engine_status_tick() -> tuple[str, gr.Timer]:
    """The engine status line; stops its timer once the engines are ready."""
    return engine_status, gr.Timer(active=not engines_ready.is_set())


StreamingFn: TypeAlias = Callable[..., Generator[tuple[bytes | None, str | None, str], None, None]]


def _make_streaming_fn(model: AsymmetricMASt3R) -> StreamingFn:
    """The Run button's streaming callback, bound to the app's model."""

    @rr.thread_local_stream(APPLICATION_ID)
    def streaming_mast3r_slam_fn(
        video_file: str | None,
        progress=gr.Progress(),  # noqa: B008
    ) -> Generator[tuple[bytes | None, str | None, str], None, None]:
        """Gradio streaming callback that runs the SLAM pipeline with ``_CONFIG``.

        Uses ``@rr.thread_local_stream`` to create a per-call recording.
        All ``rr.log()`` calls from the nested ``run_slam_pipeline()``
        generator target this recording, and ``stream.read()`` yields
        the accumulated bytes to the embedded Rerun viewer.

        Args:
            video_file: Path to the uploaded video file.
            progress: Gradio progress callback.

        Yields:
            Tuple of (Rerun binary stream bytes, zip file path or None, status message).
        """
        if video_file is None:
            raise gr.Error("Upload a video first.")
        stream = rr.binary_stream()
        recording: rr.RecordingStream | None = rr.get_thread_local_data_recording()

        progress(0.05, desc="Loading pipeline" if engines_ready.is_set() else "Waiting for the TensorRT engines")
        engines_ready.wait()

        # A fresh directory per run, so a download never mixes in another run's frames.
        run_dir: Path = Path(tempfile.mkdtemp(prefix="mast3r-slam-"))
        ns_output_dir: Path = run_dir / "nerfstudio-output"
        settings: SlamSettings = SlamSettings(dataset=video_file, slam=_CONFIG, ns_save_path=ns_output_dir)
        for msg in run_slam_pipeline(settings, model, recording=recording):
            yield stream.read(), None, msg

        # The generator exported the keyframes while the shared buffers were alive; zip them for the download.
        zip_output_path: Path = run_dir / "nerfstudio-output.zip"
        rr.log(LOG_PATH, rr.TextLog(f"Zipping nerfstudio output to {zip_output_path}", level="INFO"))
        shutil.make_archive(
            base_name=str(zip_output_path.with_suffix("")),
            format="zip",
            root_dir=run_dir,
            base_dir="nerfstudio-output",
        )

        rr.log(LOG_PATH, rr.TextLog("Finished processing", level="INFO"))
        yield stream.read(), str(zip_output_path), "MASt3R-SLAM complete"

    return streaming_mast3r_slam_fn


def _switch_to_outputs():
    return gr.update(selected="outputs")


def _switch_to_inputs():
    return gr.update(selected="inputs")


def _reset_run_outputs():
    return None, "Starting MASt3R-SLAM..."


def build_app() -> gr.Blocks:
    """Load the model, start preparing the example videos' TensorRT engines, and build the app."""
    defaults: SlamSettings = SlamSettings()
    model: AsymmetricMASt3R = load_slam_model(defaults)
    threading.Thread(target=_prepare_example_engines, args=(model,), name="prepare-trt-engines", daemon=True).start()

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
                        engine_status_md = gr.Markdown(engine_status)
                        engine_timer = gr.Timer(1.0)
                        run_slam_btn = gr.Button("Run MASt3R-SLAM", variant="primary")

                        with gr.Accordion("Config", open=False):
                            preset_dropdown = gr.Dropdown(
                                label="SLAM Preset",
                                choices=[(f"{name}: {SLAM_PRESETS[name][0]}", name) for name in VIDEO_PRESETS],
                                value=DEFAULT_PRESET,
                                info="Sets the fields below; the calibrated presets need a calibrated dataset (CLI).",
                            )
                            subsample_slider = gr.Slider(
                                label="Frame Subsample",
                                minimum=1,
                                maximum=8,
                                step=1,
                                precision=0,
                                value=_CONFIG.dataset.subsample,
                            )
                            single_thread_checkbox = gr.Checkbox(
                                label="Single-threaded",
                                value=_CONFIG.single_thread,
                                info="Frontend and backend in lockstep: deterministic, slower.",
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
                        examples=[[path] for path in EXAMPLE_VIDEOS],
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

        preset_dropdown.change(
            fn=_preset_values,
            inputs=[preset_dropdown],
            outputs=[subsample_slider, single_thread_checkbox],
            api_visibility="private",
        )
        engine_timer.tick(fn=_engine_status_tick, inputs=None, outputs=[engine_status_md, engine_timer], api_visibility="private")

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
            fn=_sync_config,
            inputs=[preset_dropdown, subsample_slider, single_thread_checkbox],
            outputs=None,
            api_visibility="private",
        ).then(
            fn=_make_streaming_fn(model),
            inputs=[video_file],
            outputs=[viewer, output_zip_file, status_text],
        )

        # Cancelling the run closes the pipeline generator, whose SlamBackend.__exit__ shuts the backend down.
        stop_slam_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            cancels=[run_event],
        ).then(
            fn=lambda: "MASt3R-SLAM stopped",
            inputs=None,
            outputs=[status_text],
            api_visibility="private",
        )

    return mast3r_slam_block
