"""Gradio UI for running PromptDA on Polycam captures.

The UI mirrors the upstream app while using the monorepo's shared dependencies
and the same API helpers as the CLI pipeline.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import beartype
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import UInt16
from monopriors.models.depth_completion.base_completion_depth import (
    CompletionDepthPrediction,
)
from monopriors.models.depth_completion.prompt_da import PromptDAPredictor
from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from tqdm import tqdm

from rerun_prompt_da.apis.prompt_da_polycam import (
    create_blueprint,
    filter_depth,
    log_polycam_data,
)


@dataclass
class InputComponents:
    """Typed container for the Gradio input widgets."""

    polycam_zip_path: gr.File
    max_depth_range_meter: gr.Number
    depth_fusion_resolution: gr.Slider

    def to_list(self) -> list:
        """Return component instances in declaration order for Gradio wiring."""

        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class InputValues:
    """Parsed runtime values coming back from the Gradio form."""

    polycam_zip_path: str
    max_depth_range_meter: int | float
    depth_fusion_resolution: float

    def __post_init__(self) -> None:
        """Normalize numeric widget values to the types the pipeline expects."""

        self.max_depth_range_meter = float(self.max_depth_range_meter)


@rr.thread_local_stream("rerun_prompt_da")
def stream_polycam_da(
    *input_params,
    progress=gr.Progress(),  # noqa: B008
):
    """Run PromptDA inference and stream incremental Rerun bytes to the viewer."""

    try:
        parameters = InputValues(*input_params)
    except beartype.roar.BeartypeCallHintParamViolation as e:
        raise gr.Error(
            "Did you make sure the zipfile finished uploading?. Try to hit run again.",
            duration=20,
        ) from e
    except Exception as e:
        raise gr.Error(
            f"Error: {e}\n Did you wait for zip file to upload?",
            duration=20,
        ) from e

    # The Gradio Rerun component consumes a binary stream yielded by the callback.
    stream: rr.BinaryStream = rr.binary_stream()

    polycam_zip_path = Path(parameters.polycam_zip_path)
    parent_log_path: Path = Path("world")
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    blueprint: rrb.Blueprint = create_blueprint(parent_log_path=parent_log_path)
    rr.send_blueprint(blueprint)
    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=polycam_zip_path)

    pred_fuser = Open3DFuser(
        fusion_resolution=parameters.depth_fusion_resolution,
        max_fusion_depth=parameters.max_depth_range_meter,
    )

    progress(progress=0.1, desc="Loading PromptDA model")
    model = PromptDAPredictor(device="cuda", model_type="large", max_size=1008)
    polycam_data: PolycamData
    for frame_idx, polycam_data in enumerate(tqdm(polycam_dataset, desc="Inferring", total=len(polycam_dataset))):
        rr.set_time("frame_idx", sequence=frame_idx)

        depth_pred: CompletionDepthPrediction = model(
            rgb=polycam_data.rgb_hw3,
            prompt_depth=polycam_data.original_depth_hw,
        )

        pred_filtered_depth_mm: UInt16[np.ndarray, "h w"] = filter_depth(
            depth_mm=depth_pred.depth_mm,
            confidence=polycam_data.confidence_hw,
            confidence_threshold=DepthConfidenceLevel.MEDIUM,
            max_depth_meter=parameters.max_depth_range_meter,
        )

        pred_fuser.fuse_frames(
            depth_hw=pred_filtered_depth_mm,
            K_33=polycam_data.pinhole_params.intrinsics.k_matrix,
            cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
            rgb_hw3=polycam_data.rgb_hw3,
        )

        log_polycam_data(
            parent_path=parent_log_path,
            polycam_data=polycam_data,
            depth_pred=depth_pred.depth_mm,
            rescale_factor=1,
        )

        pred_mesh = pred_fuser.get_mesh()
        pred_mesh.compute_vertex_normals()

        rr.log(
            f"{parent_log_path}/pred_mesh",
            rr.Mesh3D(
                vertex_positions=pred_mesh.vertices,
                triangle_indices=pred_mesh.triangles,
                vertex_normals=pred_mesh.vertex_normals,
                vertex_colors=pred_mesh.vertex_colors,
            ),
        )

        yield stream.read()


# Matches the upstream UI text so users know how to create their own captures.
instructions = """
    # Instructions to Generate Zip
    - iPhone 12 Pro or later Pro models, iPad 2020 Pro or later Pro models.
    - Free iOS App: [Polycam](https://poly.cam/get-the-app).
    # Steps:
    - Follow instructions provided in the [nerfstudio guide](https://docs.nerf.studio/quickstart/custom_dataset.html#polycam-capture) to generate a zip file.
"""
viewer = Rerun(
    streaming=True,
    panel_states={
        "time": "collapsed",
        "blueprint": "hidden",
        "selection": "hidden",
    },
)

with gr.Blocks() as prompt_da_block:
    # Export a reusable block so the standalone app can embed it directly.
    with gr.Row():
        with gr.Column():
            polycam_zip_path = gr.File(
                label="Polycam Zip Path",
                file_count="single",
                file_types=[".zip"],
                height=100,
            )
            with gr.Accordion("Instructions to Generate Zip", open=False):
                gr.Markdown(instructions)

        with gr.Column():
            with gr.Row():
                prompt_da_btn = gr.Button("Run PromptDA")
                stop_prompt_da_btn = gr.Button("Stop PromptDA")
            with gr.Accordion("Advanced Settings", open=False):
                max_depth_range_meter = gr.Number(label="Max Depth Range (m)", value=4.0, precision=2)
                depth_fusion_resolution = gr.Slider(
                    minimum=0.01,
                    maximum=0.08,
                    value=0.04,
                    step=0.01,
                    label="Depth Fusion Resolution",
                )
    with gr.Row():
        viewer.render()

    input_params = InputComponents(
        polycam_zip_path=polycam_zip_path,
        max_depth_range_meter=max_depth_range_meter,
        depth_fusion_resolution=depth_fusion_resolution,
    )
    gr.Examples(
        examples=[
            ["data/6G-room-example.zip", 4.0, 0.02],
        ],
        inputs=input_params.to_list(),
        cache_examples=False,
    )
    # The callback yields binary stream chunks that the viewer consumes live.
    prompt_da_event = prompt_da_btn.click(
        stream_polycam_da,
        inputs=input_params.to_list(),
        outputs=[viewer],
    )
    stop_prompt_da_btn.click(fn=None, inputs=[], outputs=[], cancels=[prompt_da_event])
