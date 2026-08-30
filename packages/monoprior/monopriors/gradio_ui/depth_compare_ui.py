import gc
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, cast

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
    BaseMetricPredictor,
    BaseMetricPredictorConfig,
    MetricDepthPrediction,
    metric_predictor_defaults,
)
from monopriors.models.relative_depth import (
    BaseRelativePredictor,
    BaseRelativePredictorConfig,
    RelativeDepthPrediction,
    relative_predictor_defaults,
)
from monopriors.rr_logging_utils import (
    create_compare_depth_blueprint,
    log_metric_pred,
    log_relative_pred,
)

try:
    import spaces  # type: ignore
except ImportError:
    spaces = None
    print("Not running on Zero")

model_load_status: str = "Models loaded and ready to use!"
DEVICE: Literal["cuda"] | Literal["cpu"] = "cuda" if torch.cuda.is_available() else "cpu"


def predict_depth(
    config: BaseMetricPredictorConfig | BaseRelativePredictorConfig,
    rgb: UInt8[np.ndarray, "h w 3"],
    K_33: Float32[np.ndarray, "3 3"] | None,
) -> RelativeDepthPrediction | MetricDepthPrediction:
    predictor: BaseMetricPredictor | BaseRelativePredictor = config.setup(device=DEVICE)
    prediction: RelativeDepthPrediction | MetricDepthPrediction = predictor(rgb, K_33)
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    return prediction


if spaces is not None:
    predict_depth = spaces.GPU(predict_depth)


_ON_SUBMIT_PROGRESS = gr.Progress(track_tqdm=True)


@rr.thread_local_stream("depth")
def on_submit(
    rgb: UInt8[np.ndarray, "h w 3"] | None,
    remove_flying_pixels: bool,
    depth_map_threshold: float,
    model_type: Literal["Metric", "Relative"],
    model_1_name: str,
    model_2_name: str,
    progress=_ON_SUBMIT_PROGRESS,
) -> bytes:
    stream: rr.BinaryStream = rr.binary_stream()
    if rgb is None:
        raise gr.Error("Please provide an input image.")
    display_labels: list[str] = [model_1_name, model_2_name]

    # resize the image to have a max dim of 1024
    max_dim: int = 1024
    height, width, _ = rgb.shape
    current_dim = max(height, width)
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        new_h: int = int(rgb.shape[0] * scale_factor)
        new_w: int = int(rgb.shape[1] * scale_factor)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    defaults: Mapping[str, BaseMetricPredictorConfig | BaseRelativePredictorConfig]
    K_33: Float32[np.ndarray, "3 3"] | None
    log_fn: Callable[..., None]
    if model_type == "Metric":
        defaults = metric_predictor_defaults
        K_33 = estimate_intrinsics(H=rgb.shape[0], W=rgb.shape[1])
        log_fn = cast(Callable[..., None], log_metric_pred)
    else:
        defaults = relative_predictor_defaults
        K_33 = None
        log_fn = cast(Callable[..., None], log_relative_pred)

    try:
        predictor_configs: list[BaseMetricPredictorConfig | BaseRelativePredictorConfig] = [defaults[name] for name in display_labels]
    except KeyError as error:
        raise gr.Error(f"{error.args[0]} is not a {model_type.lower()} depth predictor.") from None

    blueprint = create_compare_depth_blueprint(display_labels)
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    config_and_label: tuple[BaseMetricPredictorConfig | BaseRelativePredictorConfig, str]
    for config_and_label in tqdm(zip(predictor_configs, display_labels, strict=True), desc="Loading Model and Predicting Depth"):
        predictor_config: BaseMetricPredictorConfig | BaseRelativePredictorConfig = config_and_label[0]
        display_label: str = config_and_label[1]
        prediction: RelativeDepthPrediction | MetricDepthPrediction = predict_depth(predictor_config, rgb, K_33)
        log_fn(
            Path(display_label),
            prediction,
            rgb,
            remove_flying_pixels=remove_flying_pixels,
            depth_edge_threshold=depth_map_threshold,
        )

    return stream.read() or b""


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
                    choices=list(relative_predictor_defaults),
                    label="Model1",
                    value="depth-anything-v2",
                )
                model_2_dropdown = gr.Dropdown(
                    choices=list(relative_predictor_defaults),
                    label="Model2",
                    value="moge-v1",
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
        choices = list(metric_predictor_defaults) if model_type == "Metric" else list(relative_predictor_defaults)
        model_1_dropdown = gr.Dropdown(
            choices=choices,
            label="Model1",
            value="unidepth-metric" if model_type == "Metric" else "depth-anything-v2",
        )
        model_2_dropdown = gr.Dropdown(
            choices=choices,
            label="Model2",
            value="moge-v2-metric" if model_type == "Metric" else "unidepth-relative",
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
