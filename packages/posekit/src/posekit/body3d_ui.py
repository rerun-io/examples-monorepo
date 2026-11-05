"""Gradio app: upload an image, reconstruct every person with SAM 3D Body, inspect them in an embedded Rerun viewer.

Moved from ``packages/sam3d-body`` (``sam3d-body-app``). One shared pipeline serves the queue (SAM3 person masks,
MoGe-2 camera and depth, SAM 3D Body); each request streams its recording to the viewer and can export GLB meshes.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Final

import cv2
import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray

from posekit.apis.body3d import EXAMPLES, Body3dFrame, Body3dPipeline, log_body3d_frame
from posekit.body_rerun import create_view, export_meshes_to_glb, set_annotation_context
from posekit.models.sam3_segmenter import Sam3SegmenterConfig
from posekit.models.sam3d_body import Sam3dBodyConfig

READY: Final[str] = "✅ Ready"
MAX_SIDE: Final[int] = 1024
"""Uploads are resized so their longer side is this many pixels."""
EXAMPLE_ROWS: Final[list[tuple[str, bool, bool, bool]]] = [
    ("Planche.jpg", True, False, True),
    ("Amir-Khan-Lamont-Peterson_2689582.jpg", False, False, True),
    ("BNAAHPYGMYSE26U6C6T7VA6544.jpg", False, True, True),
    ("yoga-example.jpg", True, True, False),
]
"""Example image with its depth, GLB-export and GLB-centring switches."""

_PIPELINE: Body3dPipeline | None = None


def pipeline() -> Body3dPipeline:
    """The shared SAM3 + MoGe-2 + SAM 3D Body pipeline, loaded on first use."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Body3dPipeline(Sam3dBodyConfig(), Sam3SegmenterConfig())
    return _PIPELINE


@rr.thread_local_stream("posekit_body3d_ui")
def predict(
    rgb: UInt8[ndarray, "h w 3"], log_depth: bool, export_glb: bool, center_glb: bool, pending_cleanup: list[str] | None = None
) -> tuple[str, str, list[str]]:
    """Reconstruct the people in one upload and return the recording, a status line and any GLB files."""
    scale: float = MAX_SIDE / max(rgb.shape[0], rgb.shape[1])
    rgb = cv2.resize(rgb, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    with tempfile.NamedTemporaryFile(prefix="body3d_", suffix=".rrd", delete=False) as temp:
        rrd_path: str = temp.name
    if pending_cleanup is not None:
        pending_cleanup.append(rrd_path)
    rr.save(path=rrd_path, default_blueprint=rrb.Blueprint(create_view(log_depth), collapse_panels=True))
    set_annotation_context()
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    frame: Body3dFrame = pipeline()(rgb)
    rr.set_time("image_sequence", sequence=0)
    log_body3d_frame(frame, rgb, log_depth=log_depth)
    glb_files: list[str] = []
    if export_glb and frame.people.num_people:
        glb_dir: Path = Path(tempfile.mkdtemp(prefix="body3d_glb_"))
        glb_files = [str(path) for path in export_meshes_to_glb(frame.people, glb_dir, center_mesh=center_glb)]
        if pending_cleanup is not None:
            pending_cleanup.extend([*glb_files, str(glb_dir)])
    return rrd_path, f"{READY} ({frame.people.num_people} people)", glb_files


def cleanup(pending_cleanup: list[str]) -> None:
    """Delete a session's recordings and meshes."""
    for name in pending_cleanup:
        path = Path(name)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def build() -> gr.Blocks:
    """The app layout: inputs and outputs tabs beside a streaming Rerun viewer."""
    viewer = Rerun(streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height=800)
    with gr.Blocks() as demo, gr.Tab("SAM 3D Body"):
        pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup)
        gr.Markdown(
            "# SAM 3D Body with Rerun\n"
            "Meta's SAM 3D Body (DINOv3) on SAM3 person masks, through posekit. "
            'With "Log metric depth", the 3D panel has tabs for the full, background-only and people-only MoGe-2 depth clouds.'
        )
        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        image = gr.Image(interactive=True, label="Image", type="numpy", image_mode="RGB")
                        depth = gr.Checkbox(label="Log metric depth", value=False)
                        with gr.Row():
                            export = gr.Checkbox(label="Export GLB meshes", value=False)
                            center = gr.Checkbox(label="Center GLB at origin", value=True)
                        run = gr.Button("Predict Pose")
                    with gr.TabItem("Outputs", id="outputs"):
                        status = gr.Text(READY, label="Status")
                        meshes = gr.Files(label="GLB meshes", file_count="multiple")
                if EXAMPLES.is_dir():
                    gr.Examples(
                        examples=[[str(EXAMPLES / name), *switches] for name, *switches in EXAMPLE_ROWS],
                        inputs=[image, depth, export, center],
                        outputs=[viewer, status, meshes],
                        fn=predict,
                        run_on_click=True,
                        cache_examples=False,
                        examples_per_page=2,
                    )
            with gr.Column(scale=5):
                viewer.render()
        run.click(fn=lambda: gr.update(selected="outputs"), inputs=None, outputs=[tabs]).then(
            predict, inputs=[image, depth, export, center, pending_cleanup], outputs=[viewer, status, meshes]
        )
    return demo


def launch(server_name: str = "127.0.0.1", server_port: int = 7871) -> None:
    """Load the pipeline and serve the app."""
    pipeline()
    demo = build()
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(server_name=server_name, server_port=server_port, ssr_mode=False, allowed_paths=[str(EXAMPLES)])
