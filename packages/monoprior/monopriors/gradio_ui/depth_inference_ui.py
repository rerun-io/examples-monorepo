import gc
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from beartype.roar import BeartypeException
from gradio_rerun import Rerun
from jaxtyping import UInt8

from monopriors.models.relative_depth import (
    BaseRelativePredictor,
    BaseRelativePredictorConfig,
    MoGeV1Config,
    RelativeDepthPrediction,
    relative_predictor_defaults,
)
from monopriors.rr_logging_utils import log_relative_pred

try:
    import spaces  # type: ignore
except ImportError:
    spaces = None
    print("Not running on Zero")

model_load_status: str = "Models loaded and ready to use!"
if gr.NO_RELOAD:
    DEPTH_PREDICTOR: BaseRelativePredictor = MoGeV1Config().setup(device="cuda")


def predict_depth(rgb_hw3: UInt8[np.ndarray, "h w 3"]) -> RelativeDepthPrediction:
    relative_pred: RelativeDepthPrediction = DEPTH_PREDICTOR.__call__(rgb=rgb_hw3, K_33=None)
    return relative_pred


if spaces is not None:
    predict_depth = spaces.GPU(predict_depth)


_LOAD_MODEL_PROGRESS = gr.Progress()


def load_model(
    model: str,
    progress=_LOAD_MODEL_PROGRESS,
) -> str:
    print(model)
    global DEPTH_PREDICTOR
    # delete the previous models and clear gpu memory
    if "DEPTH_PREDICTOR" in globals():
        del DEPTH_PREDICTOR

    gc.collect()
    torch.cuda.empty_cache()

    progress(0, desc="Loading Model please wait...")

    try:
        predictor_config: BaseRelativePredictorConfig = relative_predictor_defaults[model]
    except KeyError:
        raise gr.Error(f"{model} is not a relative depth predictor.") from None
    DEPTH_PREDICTOR = predictor_config.setup(device="cuda")

    return model_load_status


@rr.thread_local_stream("depth_inference")
def relative_depth_from_img(
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    remove_flying_pixels: bool,
    depth_map_threshold: float,
    pending_cleanup: list[str],
) -> str:
    try:
        parent_log_path = Path("world")
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    contents=[
                        "$origin/**",
                        "- $origin/camera/pinhole/depth",
                        "- /world/camera/pinhole/depth",
                        "- /world/camera/pinhole/image",
                    ],
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/image"),
                    rrb.Spatial2DView(origin=f"{parent_log_path}/camera/pinhole/depth"),
                    rrb.Spatial2DView(origin=f"{parent_log_path}/camera/disparity"),
                ),
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint=blueprint)

        # resize the image to have a max dim of 1024
        max_dim: int = 1024
        height, width, _ = rgb_hw3.shape
        current_dim = max(height, width)
        if current_dim > max_dim:
            scale_factor = max_dim / current_dim
            new_h: int = int(rgb_hw3.shape[0] * scale_factor)
            new_w: int = int(rgb_hw3.shape[1] * scale_factor)
            rgb_hw3 = cv2.resize(rgb_hw3, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        relative_pred: RelativeDepthPrediction = predict_depth(rgb_hw3)
        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        log_relative_pred(
            parent_log_path,
            relative_pred,
            rgb_hw3,
            remove_flying_pixels=remove_flying_pixels,
            depth_edge_threshold=depth_map_threshold,
        )

        # We eventually want to clean up the RRD file after it's sent to the viewer, so tracking
        # any pending files to be cleaned up when the state is deleted.
        with tempfile.NamedTemporaryFile(prefix="depth_inf_", suffix=".rrd", delete=False) as temp:
            pending_cleanup.append(temp.name)
            rr.save(temp.name)
            return temp.name
    except NameError as e:
        raise gr.Error(f"Please wait Model is being loaded: {e}") from e
    except BeartypeException:
        raise
    except Exception as e:
        raise gr.Error(f"Error predicting depth: {e}") from e


def cleanup_rrds(pending_cleanup: list[str]) -> None:
    for f in pending_cleanup:
        os.unlink(f)


with gr.Blocks() as depth_inference_block:
    pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup_rrds)
    with gr.Row():
        input_image = gr.Image(
            label="Input Image",
            image_mode="RGB",
            type="numpy",
            height=300,
        )
        with gr.Column():
            with gr.Row():
                remove_flying_pixels = gr.Checkbox(
                    label="Remove Flying Pixels",
                    value=True,
                    interactive=True,
                )
                depth_map_threshold = gr.Slider(
                    label="⬇️ number == more pruning ⬆️ less pruning",
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=0.1,
                )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(relative_predictor_defaults),
                    label="Model",
                    value="moge-v1",
                    interactive=True,
                )
            with gr.Row():
                model_type = gr.Radio(
                    choices=["Metric (TODO)", "Relative"],
                    value="Relative",
                )
                model_status = gr.Textbox(
                    label="Model Status",
                    value=model_load_status,
                    interactive=False,
                )
    with gr.Row():
        run_btn = gr.Button("Run Depth Inference")
        load_model_btn = gr.Button("Load Model")
    with gr.Row():
        rr_viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "collapsed",
                "selection": "collapsed",
            },
        )

    # get all jpegs in examples path
    examples_paths = Path("examples").glob("*.jpeg")
    # set the examples to be the sorted list of input parameterss (path, remove_flying_pixels, depth_map_threshold)
    examples_list = sorted([[str(path)] for path in examples_paths])
    examples = gr.Examples(
        examples=examples_list,
        inputs=[
            input_image,
            remove_flying_pixels,
            depth_map_threshold,
            pending_cleanup,
        ],
        outputs=[rr_viewer],
        fn=relative_depth_from_img,
        cache_examples=False,
    )

    run_btn.click(
        relative_depth_from_img,
        inputs=[
            input_image,
            remove_flying_pixels,
            depth_map_threshold,
            pending_cleanup,
        ],
        outputs=[rr_viewer],
    )

    load_model_btn.click(
        load_model,
        inputs=[model_dropdown],
        outputs=[model_status],
    )
