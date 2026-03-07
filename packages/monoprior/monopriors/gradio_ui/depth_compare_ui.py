import gc
from pathlib import Path
from typing import Literal, get_args

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import torch
from gradio_rerun import Rerun
from jaxtyping import Float32, UInt8
from tqdm import tqdm

from monopriors.depth_utils import estimate_intrinsics
from monopriors.models.metric_depth import (
    METRIC_PREDICTORS,
    MetricDepthPrediction,
    get_metric_predictor,
)
from monopriors.models.relative_depth import (
    RELATIVE_PREDICTORS,
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.rr_logging_utils import (
    create_compare_depth_blueprint,
    log_metric_pred,
    log_relative_pred,
)

try:
    import spaces  # type: ignore

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

model_load_status: str = "Models loaded and ready to use!"
DEVICE: Literal["cuda"] | Literal["cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_TO_SKIP: list[str] = []


def predict_depth(
    predictor, model_type: Literal["Relative", "Metric"], rgb: UInt8[np.ndarray, "h w 3"]
) -> RelativeDepthPrediction | MetricDepthPrediction:
    if model_type == "Relative":
        relative_pred: RelativeDepthPrediction = predictor(device=DEVICE).__call__(rgb, None)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        return relative_pred
    elif model_type == "Metric":
        K_33: Float32[np.ndarray, "3 3"] = estimate_intrinsics(H=rgb.shape[0], W=rgb.shape[1])
        metric_pred: MetricDepthPrediction = predictor(device=DEVICE).__call__(rgb, K_33=K_33)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        return metric_pred


if IN_SPACES:
    predict_depth = spaces.GPU(predict_depth)
    # remove any model that fails on zerogpu spaces
    MODELS_TO_SKIP.extend(["Metric3DRelativePredictor", "Metric3DPredictor"])


@rr.thread_local_stream("depth")
def on_submit(
    rgb: UInt8[np.ndarray, "h w 3"] | None,
    remove_flying_pixels: bool,
    depth_map_threshold: float,
    model_type: Literal["Metric", "Relative"],
    model_1_name: RELATIVE_PREDICTORS | METRIC_PREDICTORS,
    model_2_name: RELATIVE_PREDICTORS | METRIC_PREDICTORS,
    progress=gr.Progress(track_tqdm=True),
) -> bytes:
    stream: rr.BinaryStream = rr.binary_stream()
    model_names = [model_1_name, model_2_name]
    blueprint = create_compare_depth_blueprint(model_names)
    rr.send_blueprint(blueprint)

    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    # resize the image to have a max dim of 1024
    max_dim: int = 1024
    height, width, _ = rgb.shape
    current_dim = max(height, width)
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        new_h: int = int(rgb.shape[0] * scale_factor)
        new_w: int = int(rgb.shape[1] * scale_factor)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    for model_name in tqdm(model_names, desc="Loading Model and Predicting Depth"):
        # get the name of the model
        parent_log_path = Path(f"{model_name}")
        if model_type == "Metric":
            predictor = get_metric_predictor(model_name)
            metric_pred: MetricDepthPrediction = predict_depth(predictor, model_type, rgb)
            log_metric_pred(
                parent_log_path=parent_log_path,
                metric_pred=metric_pred,
                rgb_hw3=rgb,
                remove_flying_pixels=remove_flying_pixels,
                depth_edge_threshold=depth_map_threshold,
            )
        elif model_type == "Relative":
            predictor = get_relative_predictor(model_name)
            relative_pred: RelativeDepthPrediction = predict_depth(predictor, model_type, rgb)

            log_relative_pred(
                parent_log_path=parent_log_path,
                relative_pred=relative_pred,
                rgb_hw3=rgb,
                remove_flying_pixels=remove_flying_pixels,
                depth_edge_threshold=depth_map_threshold,
            )

    return stream.read()


with gr.Blocks() as relative_compare_block:
    with gr.Row():
        input_image = gr.Image(
            label="Input Image",
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
                    minimum=0.01,
                    maximum=0.95,
                    step=0.01,
                    value=0.05,
                )
            with gr.Row():
                model_1_dropdown = gr.Dropdown(
                    choices=list(get_args(RELATIVE_PREDICTORS)),
                    label="Model1",
                    value="DepthAnythingV2Predictor",
                )
                model_2_dropdown = gr.Dropdown(
                    choices=list(get_args(RELATIVE_PREDICTORS)),
                    label="Model2",
                    value="MogeV1Predictor",
                )
            with gr.Row():
                model_type = gr.Radio(
                    choices=["Metric", "Relative"],
                    value="Relative",
                )
                model_status = gr.Textbox(
                    label="Model Status",
                    value=model_load_status,
                    interactive=False,
                )

    with gr.Row():
        submit = gr.Button(value="Compare Depth")
    rr_viewer = Rerun(streaming=False, height=800)

    submit.click(
        on_submit,
        inputs=[input_image, remove_flying_pixels, depth_map_threshold, model_type, model_1_dropdown, model_2_dropdown],
        outputs=[rr_viewer],
    )

    def change_dropdown(model_type: Literal["Metric", "Relative"]) -> tuple[gr.Dropdown, gr.Dropdown]:
        choices = list(get_args(METRIC_PREDICTORS)) if model_type == "Metric" else list(get_args(RELATIVE_PREDICTORS))
        model_1_dropdown = gr.Dropdown(
            choices=choices,
            label="Model1",
            value="UniDepthMetricPredictor" if model_type == "Metric" else "DepthAnythingV2Predictor",
        )
        model_2_dropdown = gr.Dropdown(
            choices=choices,
            label="Model2",
            value="Metric3DPredictor" if model_type == "Metric" else "UniDepthRelativePredictor",
        )
        return model_1_dropdown, model_2_dropdown

    model_type.input(
        fn=change_dropdown,
        inputs=model_type,
        outputs=[model_1_dropdown, model_2_dropdown],
    )

    # get all jpegs in examples path
    examples_paths = Path("examples").glob("*.jpeg")
    # set the examples to be the sorted list of input parameterss (path, remove_flying_pixels, depth_map_threshold)
    examples_list = sorted([[str(path)] for path in examples_paths])
    examples = gr.Examples(
        examples=examples_list,
        inputs=[input_image, remove_flying_pixels, depth_map_threshold, model_type, model_1_dropdown, model_2_dropdown],
        outputs=[rr_viewer],
        fn=on_submit,
        cache_examples=False,
    )
