"""Stereo depth Gradio app: a rectified left/right pair + calibration in, an exoego:v2 rig in the Rerun viewer out.

Layout follows ``posekit.track_ui`` (inputs + status on the left, Radio-driven Input/Config/Outputs panels, a
streaming Rerun viewer on the right). One predictor per model size is kept warm for the process lifetime.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import gradio as gr
import numpy as np
import rerun as rr
import tyro
from gradio_rerun import Rerun
from jaxtyping import Float32, UInt8
from numpy import ndarray

from monopriors.apis.stereo_depth import MiddleburyCalibration, read_middlebury_calib
from monopriors.models.stereo_depth import LiteAnyStereoPredictor, StereoDepthPrediction
from monopriors.models.stereo_depth.liteanystereo import LAS2ModelSize
from monopriors.rr_logging_utils import create_stereo_depth_blueprint, log_stereo_pred

APP_ID: str = "monoprior_stereo_depth"
TabName: TypeAlias = Literal["Input", "Config", "Outputs"]
EXAMPLE_SCENE: Path = Path("data/examples/stereo/eth3d/two_view_training/playground_1l")
"""ETH3D two-view sample fetched by the ``_monoprior-download-stereo`` task (im0/im1 + Middlebury calib.txt)."""

_PREDICTORS: dict[LAS2ModelSize, LiteAnyStereoPredictor] = {}


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Server settings for the stereo depth app."""

    host: str = "127.0.0.1"
    """Interface to bind. The embedded viewer needs a secure context, so expose it to the tailnet with
    ``tailscale serve --bg --https=<port> http://127.0.0.1:<port>`` rather than binding ``0.0.0.0`` (plain http breaks the viewer)."""
    port: int = 7871
    """Port to serve on."""
    root_path: str = ""
    """Root path when mounted under a reverse-proxy subpath (empty when served at ``/``)."""


def _predictor(model_size: LAS2ModelSize) -> LiteAnyStereoPredictor:
    if model_size not in _PREDICTORS:
        _PREDICTORS[model_size] = LiteAnyStereoPredictor(device="cuda", model_size=model_size)
    return _PREDICTORS[model_size]


def show_control_tab(selected: TabName) -> tuple[gr.Column, gr.Column, gr.Column]:
    return gr.Column(visible=selected == "Input"), gr.Column(visible=selected == "Config"), gr.Column(visible=selected == "Outputs")


def run_stereo(
    left_rgb: UInt8[ndarray, "h w 3"] | None,
    right_rgb: UInt8[ndarray, "h w 3"] | None,
    model_size: LAS2ModelSize,
    fx: float | int,
    cx: float | int,
    cy: float | int,
    baseline_mm: float | int,
    max_depth_m: float | int,
    remove_flying_pixels: bool,
    depth_edge_threshold: float | int,
) -> Iterator[tuple[bytes | None, str, TabName]]:
    """Predict, log the rig into a fresh recording, and stream it to the viewer (Gradio hands ints for whole-number widgets)."""
    if left_rgb is None or right_rgb is None:
        raise gr.Error("Upload a left and a right image.")
    if left_rgb.shape != right_rgb.shape:
        raise gr.Error(f"Left {left_rgb.shape[:2]} and right {right_rgb.shape[:2]} images must have the same size (rectified pair).")
    K_33: Float32[ndarray, "3 3"] = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    predictor: LiteAnyStereoPredictor = _predictor(model_size)
    start: float = time.perf_counter()
    stereo_pred: StereoDepthPrediction = predictor(left_rgb, right_rgb, K_33=K_33, baseline_m=float(baseline_mm) / 1000.0)
    elapsed_ms: float = (time.perf_counter() - start) * 1000.0

    rec: rr.RecordingStream = rr.RecordingStream(application_id=APP_ID)
    stream: rr.BinaryStream = rec.binary_stream()
    with rec:
        rr.send_blueprint(create_stereo_depth_blueprint(Path("world")))
        log_stereo_pred(
            Path("world"),
            stereo_pred,
            left_rgb,
            right_rgb,
            max_depth_m=float(max_depth_m),
            remove_flying_pixels=remove_flying_pixels,
            depth_edge_threshold=float(depth_edge_threshold),
        )
    valid_hw = stereo_pred.disparity > 0.0
    status: str = (
        f"LAS2-{model_size.upper()} · {left_rgb.shape[1]}×{left_rgb.shape[0]} · {elapsed_ms:.0f} ms end-to-end · "
        f"disparity {stereo_pred.disparity[valid_hw].min():.1f}–{stereo_pred.disparity[valid_hw].max():.1f} px"
    )
    yield stream.read(), status, "Outputs"


def build_demo() -> gr.Blocks:
    example_calibration: MiddleburyCalibration | None = read_middlebury_calib(EXAMPLE_SCENE / "calib.txt") if EXAMPLE_SCENE.is_dir() else None
    with gr.Blocks(title="monoprior: Stereo Depth") as demo:
        gr.Markdown("**Stereo depth** — a rectified left/right pair + pinhole calibration → LiteAnyStereo V2 disparity, metric depth, and the backprojected cloud as an exoego:v2 rig.")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    left_in: gr.Image = gr.Image(label="Left", type="numpy", image_mode="RGB", height=200)
                    right_in: gr.Image = gr.Image(label="Right", type="numpy", image_mode="RGB", height=200)
                status: gr.Markdown = gr.Markdown("Upload a rectified pair (or pick the example) and press Run.")
                # Three native gr.Tab components trigger a Svelte effect loop in Gradio 6.13 (see posekit.track_ui).
                tabs_radio: gr.Radio = gr.Radio(choices=list(get_args(TabName)), value="Input", show_label=False, container=False)
                with gr.Column() as input_panel:
                    with gr.Row():
                        fx_in: gr.Number = gr.Number(label="fx (px)", value=float(example_calibration.K_33[0, 0]) if example_calibration else 500.0)
                        baseline_in: gr.Number = gr.Number(label="baseline (mm)", value=example_calibration.baseline_m * 1000.0 if example_calibration else 100.0)
                    with gr.Row():
                        cx_in: gr.Number = gr.Number(label="cx (px)", value=float(example_calibration.K_33[0, 2]) if example_calibration else 0.0)
                        cy_in: gr.Number = gr.Number(label="cy (px)", value=float(example_calibration.K_33[1, 2]) if example_calibration else 0.0)
                    if example_calibration is not None:
                        gr.Examples(examples=[[str(EXAMPLE_SCENE / "im0.png"), str(EXAMPLE_SCENE / "im1.png")]], inputs=[left_in, right_in], label="ETH3D playground_1l")
                with gr.Column(visible=False) as config_panel:
                    model_dd: gr.Dropdown = gr.Dropdown(label="Model", choices=list(get_args(LAS2ModelSize)), value="m")
                    max_depth_slider: gr.Slider = gr.Slider(label="Max depth (m)", minimum=1.0, maximum=100.0, value=20.0, step=1.0)
                    flying_box: gr.Checkbox = gr.Checkbox(label="Remove flying pixels", value=True)
                    edge_slider: gr.Slider = gr.Slider(label="Depth edge threshold (m/px)", minimum=0.05, maximum=5.0, value=0.5, step=0.05)
                with gr.Column(visible=False) as outputs_panel:
                    gr.Markdown("Left cloud = `rig_00/cam_00/pinhole/depth`; orbit the 3D view, hover the depth image for metres.")
                run_btn: gr.Button = gr.Button("Run", variant="primary")
            with gr.Column(scale=3):
                viewer: Rerun = Rerun(label="Stereo rig", streaming=True, panel_states={"time": "hidden", "blueprint": "hidden", "selection": "hidden"}, height=800)

        tabs_radio.change(fn=show_control_tab, inputs=[tabs_radio], outputs=[input_panel, config_panel, outputs_panel], queue=False, show_progress="hidden", api_visibility="private")
        run_btn.click(
            fn=run_stereo,
            inputs=[left_in, right_in, model_dd, fx_in, cx_in, cy_in, baseline_in, max_depth_slider, flying_box, edge_slider],
            outputs=[viewer, status, tabs_radio],
        )
    return demo


def launch(config: AppConfig) -> None:
    """Launch the stereo depth Gradio app."""
    build_demo().launch(server_name=config.host, server_port=config.port, root_path=config.root_path, allowed_paths=[str(EXAMPLE_SCENE)], show_error=True)


if __name__ == "__main__":
    launch(tyro.cli(AppConfig))
