"""Gradio UI for executing the full exo/ego pipeline directly from RRD inputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import rerun as rr
from gradio.data_classes import FileData
from gradio_rerun import Rerun
from monopriors.apis.multiview_calibration import MultiViewCalibratorConfig
from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
from simplecv.rerun_log_utils import RerunTyroConfig

from mv_api.api.full_exoego_pipeline import RRDPipelineConfig, run_full_exoego_pipeline

EXAMPLE_RRD_PATH: Path = Path("/mnt/8tb/data/exoego-self-collected/gus/statisOrangePNP_av1.rrd")
INPUT_TAB_ID: str = "rrd_input_tab"
OUTPUT_TAB_ID: str = "rrd_output_tab"


def _resolve_rrd_path(rrd_input: FileData | str | None) -> Path:
    """Normalize a Gradio file input to a filesystem path."""
    if rrd_input is None:
        raise ValueError("No RRD file available. Upload or pick an example first.")

    if isinstance(rrd_input, FileData):
        rrd_path_str: str | None = rrd_input.path
        if rrd_path_str is None:
            raise ValueError("Uploaded RRD input is missing a backing path on disk.")
        rrd_path: Path = Path(rrd_path_str)
        return rrd_path

    rrd_path = Path(rrd_input)
    return rrd_path


def cleanup_temp_rrds(pending_cleanup: list[str]) -> None:
    """Remove temporary RRD artefacts created during UI interactions."""
    for temp_path_str in pending_cleanup:
        temp_path: Path = Path(temp_path_str)
        if temp_path.exists():
            os.unlink(temp_path)


@rr.thread_local_stream("rerun_full_pipeline_rrd")
def run_rrd_pipeline(
    rrd_input: FileData | str | None,
    max_frames: int | float | None,
    pending_cleanup: list[str],
    progress: gr.Progress | None = None,
) -> tuple[str, str]:
    """Execute the full pipeline against an uploaded RRD and return a temporary RRD for the viewer."""
    if progress is None:
        progress = gr.Progress()

    progress(0.0, desc="Preparing pipeline inputs")
    rrd_path: Path = _resolve_rrd_path(rrd_input)

    with tempfile.NamedTemporaryFile(prefix="full_pipeline_", suffix=".rrd", delete=False) as temp_file:
        pending_cleanup.append(temp_file.name)
        rr_config: RerunTyroConfig = RerunTyroConfig(save=Path(temp_file.name))

        dataset_cfg: RRDExoEgoConfig = RRDExoEgoConfig(rrd_path=rrd_path)
        max_frames_sanitized: int | None
        if max_frames is None:
            max_frames_sanitized = None
        else:
            candidate_frames: int = int(max_frames)
            max_frames_sanitized = candidate_frames if candidate_frames > 0 else None
        pipeline_config: RRDPipelineConfig = RRDPipelineConfig(
            rr_config=rr_config,
            calib_confg=MultiViewCalibratorConfig(segment_people=False),
            dataset=dataset_cfg,
            max_frames=max_frames_sanitized,
        )

        progress(0.2, desc="Running full exo/ego pipeline")
        run_full_exoego_pipeline(config=pipeline_config)

    progress(1.0, desc="Pipeline complete")
    return temp_file.name, temp_file.name


def build_full_pipeline_rrd_block() -> None:
    """Create the Gradio layout for the RRD-based full pipeline experience."""
    pending_cleanup: gr.State = gr.State([], time_to_live=10, delete_callback=cleanup_temp_rrds)

    with gr.Row():  # noqa: SIM117 - nested layout contexts keep column scopes inside the row
        with gr.Column(scale=1), gr.Tabs(selected=INPUT_TAB_ID) as io_tabs:
            with gr.TabItem("Input RRD", id=INPUT_TAB_ID):
                rrd_file: gr.File = gr.File(file_types=[".rrd"], label="Upload or select RRD file")
                max_frames: gr.Number = gr.Number(
                    label="Max frames (optional, leave blank for all)",
                    value=None,
                    precision=0,
                )
                run_button: gr.Button = gr.Button("Run Full Pipeline")
                example_rows: list[list[str]] = []
                if EXAMPLE_RRD_PATH.exists():
                    example_rows.append([str(EXAMPLE_RRD_PATH)])
                if example_rows:
                    gr.Examples(
                        examples=example_rows,
                        inputs=[rrd_file],
                        cache_examples=False,
                    )
            with gr.TabItem("Output Viewer", id=OUTPUT_TAB_ID):
                output_rrd_file: gr.File = gr.File(label="Generated RRD output will appear here", interactive=False)

        with gr.Column(scale=5):
            viewer: Rerun = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=800,
            )

        def _select_output_tab() -> gr.Tabs:
            return gr.Tabs(selected=OUTPUT_TAB_ID)

    run_button.click(
        _select_output_tab,
        inputs=None,
        outputs=[io_tabs],
    ).then(
        run_rrd_pipeline,
        inputs=[rrd_file, max_frames, pending_cleanup],
        outputs=[viewer, output_rrd_file],
    )


__all__ = ["build_full_pipeline_rrd_block", "run_rrd_pipeline", "cleanup_temp_rrds", "EXAMPLE_RRD_PATH"]
