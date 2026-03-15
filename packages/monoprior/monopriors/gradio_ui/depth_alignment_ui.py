"""Gradio UI for depth alignment.

Provides an interactive web interface for aligning a target depth map to a
reference depth map's coordinate frame. Accepts ``.npy`` files containing
float32 depth arrays. The left panel holds file inputs, a run button, and
a config accordion; the right panel streams results into an embedded Rerun
viewer showing before/after comparison.
"""

import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Final

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray

from monopriors.apis.depth_alignment import DepthAlignmentConfig, DepthAlignmentResult, run_depth_alignment

PARENT_LOG_PATH: Final[Path] = Path("world")
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "examples" / "depth_alignment"

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

_CONFIG: DepthAlignmentConfig = DepthAlignmentConfig()
"""Module-level config, kept in sync with UI widgets."""


def _sync_config(edge_threshold: float, scale_only: bool) -> None:
    """Sync UI widget values into the module-level config.

    Args:
        edge_threshold: Threshold for depth edge masking.
        scale_only: Whether to use scale-only alignment.
    """
    global _CONFIG
    _CONFIG = DepthAlignmentConfig(
        edge_threshold=edge_threshold,
        scale_only=scale_only,
    )


def _get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    """Create a Rerun recording stream for a given session."""
    return rr.RecordingStream(application_id="depth_alignment", recording_id=recording_id)


def _load_depth(filepath: str) -> Float32[ndarray, "H W"]:
    """Load a depth map from a .npy file.

    Args:
        filepath: Path to .npy file containing a float32 depth array.

    Returns:
        Float32 depth array in meters.
    """
    depth: Float32[ndarray, "H W"] = np.load(filepath).astype(np.float32)
    return depth


def depth_alignment_fn(
    recording_id: uuid.UUID,
    ref_file: str | None,
    tgt_file: str | None,
    conf_file: str | None,
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that runs depth alignment.

    Args:
        recording_id: Session-scoped recording identifier.
        ref_file: Path to reference depth .npy file.
        tgt_file: Path to target depth .npy file.
        conf_file: Optional path to confidence .npy file.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    if ref_file is None:
        raise gr.Error("Please provide a reference depth .npy file.")
    if tgt_file is None:
        raise gr.Error("Please provide a target depth .npy file.")

    recording: rr.RecordingStream = _get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    with recording:
        ref_depth: Float32[ndarray, "H W"] = _load_depth(ref_file)
        tgt_depth: Float32[ndarray, "H W"] = _load_depth(tgt_file)

        # Load confidence if provided
        confidence_mask: Bool[ndarray, "H W"] | None = None
        if conf_file is not None:
            raw_conf: ndarray = np.load(conf_file)
            confidence_mask = (raw_conf > 0).astype(bool)

        # Run alignment
        result: DepthAlignmentResult = run_depth_alignment(
            reference_depth=ref_depth,
            target_depth=tgt_depth,
            confidence_mask=confidence_mask,
            config=_CONFIG,
        )

        # Build blueprint
        blueprint: rrb.Blueprint = rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(origin=f"{PARENT_LOG_PATH}/reference_depth", name="Reference"),
                rrb.Spatial2DView(origin=f"{PARENT_LOG_PATH}/target_depth", name="Target"),
                rrb.Spatial2DView(origin=f"{PARENT_LOG_PATH}/aligned_depth", name="Aligned"),
                rrb.Spatial2DView(origin=f"{PARENT_LOG_PATH}/confidence", name="Confidence"),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log(f"{PARENT_LOG_PATH}", rr.ViewCoordinates.RDF, static=True)

        # Log depth maps
        rr.log(f"{PARENT_LOG_PATH}/reference_depth", rr.DepthImage(ref_depth, meter=1), static=True)
        rr.log(f"{PARENT_LOG_PATH}/target_depth", rr.DepthImage(tgt_depth, meter=1), static=True)
        rr.log(f"{PARENT_LOG_PATH}/aligned_depth", rr.DepthImage(result.aligned_depth, meter=1), static=True)

        # Log confidence
        if confidence_mask is not None:
            conf_vis: UInt8[ndarray, "H W"] = confidence_mask.astype(np.uint8) * 255
        else:
            conf_vis = (ref_depth > 0).astype(np.uint8) * 255
        rr.log(f"{PARENT_LOG_PATH}/confidence", rr.Image(conf_vis, color_model=rr.ColorModel.L), static=True)

    yield stream.read(), f"Alignment complete (scale={result.scale:.4f}, shift={result.shift:.4f})"


def _switch_to_outputs():
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def _switch_to_inputs():
    """Switch the Gradio Tabs component to the Inputs tab."""
    return gr.update(selected="inputs")


def main() -> gr.Blocks:
    """Build and return the depth alignment Gradio app.

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
        recording_id = gr.State(uuid.uuid4())

        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        ref_file = gr.File(
                            label="Reference Depth (.npy)",
                            file_types=[".npy"],
                        )
                        tgt_file = gr.File(
                            label="Target Depth (.npy)",
                            file_types=[".npy"],
                        )
                        conf_file = gr.File(
                            label="Confidence (.npy, optional)",
                            file_types=[".npy"],
                        )
                        run_btn = gr.Button("Align Depths")

                        with gr.Accordion("Config", open=False):
                            edge_threshold_slider = gr.Slider(
                                label="Edge Threshold",
                                minimum=0.001,
                                maximum=0.1,
                                step=0.005,
                                value=_CONFIG.edge_threshold,
                            )
                            scale_only_checkbox = gr.Checkbox(
                                label="Scale Only (no shift)",
                                value=_CONFIG.scale_only,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(label="Status", interactive=False)

                gr.Examples(
                    examples=[[
                        str(EXAMPLE_DATA_DIR / "reference_depth.npy"),
                        str(EXAMPLE_DATA_DIR / "target_depth.npy"),
                        str(EXAMPLE_DATA_DIR / "confidence.npy"),
                    ]],
                    inputs=[ref_file, tgt_file, conf_file],
                    cache_examples=False,
                )

            with gr.Column(scale=5):
                rr_viewer.render()

        # Switch to Inputs tab when examples populate the input
        ref_file.change(fn=_switch_to_inputs, inputs=None, outputs=[tabs], api_visibility="private")

        # Click chain: UI transition → fresh session → sync config → run alignment
        run_btn.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(
            fn=lambda: uuid.uuid4(),
            inputs=None,
            outputs=[recording_id],
            api_visibility="private",
        ).then(  # Sync config widgets
            _sync_config,
            inputs=[edge_threshold_slider, scale_only_checkbox],
        ).then(  # Run depth alignment and stream results to the Rerun viewer
            depth_alignment_fn,
            inputs=[recording_id, ref_file, tgt_file, conf_file],
            outputs=[rr_viewer, status_text],
        )

    return demo
