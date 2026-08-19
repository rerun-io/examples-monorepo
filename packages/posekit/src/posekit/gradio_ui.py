"""Unified single-image 2D pose inference in Gradio and Rerun."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gradio_rerun import Rerun
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from torch import Tensor

from posekit.models import (
    RtmPoseConfig,
    SapiensPoseConfig,
    SegmentationPrompts,
    TopDownDenseLandmarks2d,
    TopDownPose2d,
    VitPoseConfig,
)
from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig
from posekit.models.sapiens import SapiensModelSize
from posekit.predictions import BoxDetections, DenseLandmarks2d, Keypoints2d
from posekit.rerun_logging import log_approximate_skeletons, log_dense_predictions, log_topdown_predictions
from posekit.runtimes import (
    BackendConfig,
    OnnxBackendConfig,
    TensorRtBackendConfig,
    TorchBackendConfig,
)

if TYPE_CHECKING:
    from mamma.landmarks.skeleton import SkeletonRegressor

BackendName: TypeAlias = Literal["torch", "onnx", "tensorrt"]

# The pose-app environment is CUDA-only; model setup reports host availability when a request loads a network.
DEVICE: str = "cuda"
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


NETWORKS: dict[str, list[BackendName]] = {
    "rtmpose-m-coco17": ["onnx", "tensorrt"],
    "rtmpose-x-coco17": ["onnx", "tensorrt"],
    "rtmw-x-coco133": ["onnx", "tensorrt"],
    "vitpose": ["torch"],
    "sapiens2-0.4b": ["torch", "onnx", "tensorrt"],
    "sapiens2-1b": ["torch", "onnx", "tensorrt"],
    "mamma-dense512": ["torch", "tensorrt"],
}
# Backend lists are ordered least-to-most preferred, so defaults are derived once.
DEFAULT_BACKENDS: dict[str, BackendName] = {network_name: backends[-1] for network_name, backends in NETWORKS.items()}
BACKEND_CONFIGS: dict[BackendName, Callable[[], BackendConfig]] = {
    "torch": TorchBackendConfig,
    "onnx": OnnxBackendConfig,
    "tensorrt": TensorRtBackendConfig,
}


def _build_estimator(network_name: str, backend_name: BackendName) -> TopDownPose2d | TopDownDenseLandmarks2d:
    """Construct one network on one backend; NETWORKS already gated the pair."""
    match network_name:
        case "rtmpose-m-coco17" | "rtmpose-x-coco17" | "rtmw-x-coco133":
            accelerated: OnnxBackendConfig | TensorRtBackendConfig = (
                OnnxBackendConfig() if backend_name == "onnx" else TensorRtBackendConfig()
            )
            return RtmPoseConfig(variant=network_name, backend=accelerated).setup()
        case "vitpose":
            return VitPoseConfig(device=DEVICE).setup()
        case "sapiens2-0.4b" | "sapiens2-1b":
            model_size: SapiensModelSize = "0.4B" if network_name == "sapiens2-0.4b" else "1B"
            backend: BackendConfig = BACKEND_CONFIGS[backend_name]()
            return SapiensPoseConfig(model_size=model_size, backend=backend).setup()
        case "mamma-dense512":
            try:
                from mamma.landmarks.posekit_role import MammaNetLandmarksConfig
            except ImportError as error:
                raise gr.Error("mamma is not installed in this environment") from error
            mamma_backend: TorchBackendConfig | TensorRtBackendConfig = (
                TorchBackendConfig() if backend_name == "torch" else TensorRtBackendConfig()
            )
            return MammaNetLandmarksConfig(device=DEVICE, backend=mamma_backend).setup()
        case _:
            raise gr.Error(f"No builder for {network_name!r}")


_estimator_cache: dict[tuple[str, str], TopDownPose2d | TopDownDenseLandmarks2d] = {}


@functools.cache
def _get_segmenter() -> Sam3Segmenter:
    """Load SAM3 once and reuse it across requests."""
    return Sam3SegmenterConfig(device=DEVICE).setup()


def _get_estimator(network_name: str, backend_name: str) -> TopDownPose2d | TopDownDenseLandmarks2d:
    """Load and cache the selected network/backend pair; at most two stay resident on the GPU."""
    allowed: list[BackendName] = NETWORKS[network_name]
    if backend_name not in allowed:
        raise gr.Error(f"Backend {backend_name!r} is not available for {network_name}.")

    cache_key: tuple[str, str] = (network_name, backend_name)
    if cache_key in _estimator_cache:
        _estimator_cache[cache_key] = _estimator_cache.pop(cache_key)
        return _estimator_cache[cache_key]
    if len(_estimator_cache) >= 2:
        evicted: TopDownPose2d | TopDownDenseLandmarks2d = _estimator_cache.pop(next(iter(_estimator_cache)))
        # The local reference would keep the evicted weights alive through empty_cache().
        del evicted
        torch.cuda.empty_cache()
    _estimator_cache[cache_key] = _build_estimator(network_name, cast(BackendName, backend_name))
    return _estimator_cache[cache_key]


@rr.thread_local_stream("posekit_2d_pose_playground")
def predict_ui(
    image: Image.Image | None,
    prompt: str | None,
    network_name: str,
    backend_name: str,
    keypoint_threshold: float,
    show_skeleton: bool,
) -> Iterator[tuple[bytes | None, dict[str, object], str]]:
    """Run SAM3 and the selected pose estimator, then stream one recording."""
    if image is None:
        raise gr.Error("No image provided.")
    if prompt is None or not prompt.strip():
        raise gr.Error("No prompt provided.")
    stream: rr.BinaryStream = rr.binary_stream()

    image_rgb: UInt8[ndarray, "h w 3"] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    frames_rgb: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(image_rgb).unsqueeze(0).to(device=DEVICE)
    prompts: SegmentationPrompts = SegmentationPrompts(
        frame_indices=torch.zeros((1,), dtype=torch.long, device=frames_rgb.device),
        text=prompt.strip(),
    )
    detections: BoxDetections = _get_segmenter()(frames_rgb, prompts)
    estimator: TopDownPose2d | TopDownDenseLandmarks2d = _get_estimator(network_name, backend_name)
    boxes_xyxy: Float32[ndarray, "n 4"] = detections.xyxy_numpy()

    rr.send_blueprint(rrb.Blueprint(rrb.Spatial2DView(name="2D Pose", origin="image"), collapse_panels=True))
    rr.set_time("iteration", sequence=0)
    rr.log("image", rr.Image(image_rgb, color_model=rr.ColorModel.RGB))
    instances: list[dict[str, object]] = []
    if isinstance(estimator, TopDownDenseLandmarks2d):
        landmarks: DenseLandmarks2d = estimator(frames_rgb, detections)
        landmarks_xy: Float32[ndarray, "n p 2"] = landmarks.xy_numpy()
        visibility: Float32[ndarray, "n p"] = landmarks.visibility_numpy()
        contact: Float32[ndarray, "n p"] = landmarks.contact_numpy()
        floor_contact: Float32[ndarray, "n p"] = landmarks.floor_contact_numpy()
        log_dense_predictions(boxes_xyxy, landmarks_xy, visibility, keypoint_threshold)
        if show_skeleton:
            try:
                from mamma.landmarks.skeleton import joints_from_landmarks, load_skeleton_regressor, skeleton_strips
            except ImportError as error:
                raise gr.Error("mamma is not installed in this environment") from error
            regressor: SkeletonRegressor = load_skeleton_regressor()
            joints_xy: Float32[ndarray, "n j 2"] = joints_from_landmarks(landmarks_xy, regressor)
            strips_by_person: list[Float32[ndarray, "b 2 2"]] = [
                skeleton_strips(joints_xy[person_idx], regressor) for person_idx in range(boxes_xyxy.shape[0])
            ]
            log_approximate_skeletons(joints_xy, strips_by_person)
        for person_idx in range(boxes_xyxy.shape[0]):
            instances.append(
                {
                    "bbox": boxes_xyxy[person_idx].tolist(),
                    "keypoints": landmarks_xy[person_idx].tolist(),
                    "visibility": visibility[person_idx].tolist(),
                    "contact": contact[person_idx].tolist(),
                    "floor_contact": floor_contact[person_idx].tolist(),
                }
            )
    else:
        keypoints: Keypoints2d = estimator(frames_rgb, detections)
        keypoints_xy: Float32[ndarray, "n k 2"] = keypoints.xy_numpy()
        scores: Float32[ndarray, "n k"] = keypoints.scores_numpy()
        log_topdown_predictions(boxes_xyxy, keypoints_xy, scores, keypoints.skeleton, keypoint_threshold)
        for person_idx in range(boxes_xyxy.shape[0]):
            instances.append(
                {
                    "bbox": boxes_xyxy[person_idx].tolist(),
                    "keypoints": keypoints_xy[person_idx].tolist(),
                    "scores": scores[person_idx].tolist(),
                }
            )

    stream_bytes: bytes | None = stream.read()
    yield stream_bytes, {"instances": instances}, f"Complete: {len(instances)} person(s) — {network_name} ({backend_name})"


def _update_backend_choices(network_name: str) -> dict[str, object]:
    """Update the backend dropdown when the selected network changes."""
    choices: list[BackendName] | None = NETWORKS.get(network_name)
    if choices is None:
        raise gr.Error(f"Unknown network: {network_name}")
    return gr.update(choices=choices, value=DEFAULT_BACKENDS[network_name])


def _example_images() -> list[str]:
    """Find MAMMA examples, falling back to the Sapiens2 assets."""
    candidate_dirs: list[Path] = []
    for package_name in ("mamma", "sapiens2_pose"):
        try:
            candidate_dirs.append(Path(str(resources.files(package_name))) / "assets" / "images")
        except ImportError:
            continue
    for images_dir in candidate_dirs:
        if images_dir.is_dir():
            return sorted(str(path) for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return []


EXAMPLES: list[str] = _example_images()

CUSTOM_CSS: str = """
:root, body, .gradio-container, button, input, select, textarea,
.gradio-container *:not(code):not(pre) {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

#title { text-align: center; font-size: 44px; font-weight: 700;
         letter-spacing: -0.01em; margin: 28px 0 4px;
         background: linear-gradient(90deg, #1d4ed8 0%, #6d28d9 50%, #be185d 100%);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         background-clip: text; }
#subtitle { text-align: center; font-size: 12px; color: #64748b;
            letter-spacing: 0.18em; margin: 0 0 14px; text-transform: uppercase;
            font-weight: 500; }
#badges { display: flex; justify-content: center; flex-wrap: wrap;
          gap: 8px; margin: 0 0 32px; }
.pill { display: inline-flex; align-items: center; gap: 6px;
        padding: 7px 14px; border-radius: 999px;
        background: #f1f5f9; color: #0f172a !important;
        font-size: 13px; font-weight: 500; letter-spacing: 0.01em;
        text-decoration: none !important; border: 1px solid #e2e8f0;
        transition: background 150ms ease, transform 150ms ease, border-color 150ms ease; }
.pill:hover { background: #0f172a; color: #f8fafc !important;
              border-color: #0f172a; transform: translateY(-1px); }
.pill svg { width: 14px; height: 14px; }
"""

HEADER_HTML: str = """
<div id="title">posekit: 2D Pose Playground</div>
<div id="subtitle">One detector &middot; Seven networks &middot; Three runtimes</div>
<div id="badges">
  <a class="pill" href="https://github.com/rerun-io/examples-monorepo/tree/main/packages/posekit" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.3.1 2 1.3 2 1.3 1.1 1.9 3 1.4 3.7 1 .1-.8.4-1.4.8-1.7-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.3-3.2-.1-.4-.6-1.6.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.6.2 2.8.1 3.2.8.8 1.3 1.9 1.3 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .3"/></svg>
    Code
  </a>
</div>
"""

DEFAULT_NETWORK: str = "rtmw-x-coco133"

with gr.Blocks(title="posekit: 2D Pose Playground") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            tabs: gr.Tabs = gr.Tabs(selected="inputs")
            with tabs:
                with gr.TabItem("Inputs", id="inputs"):
                    inp: gr.Image = gr.Image(label="Input Image", type="pil", image_mode="RGB", height=360)
                    run: gr.Button = gr.Button("Run Pose", variant="primary")

                    with gr.Accordion("Config", open=False):
                        prompt_text: gr.Textbox = gr.Textbox(label="Prompt", value="person")
                        network_dropdown: gr.Dropdown = gr.Dropdown(
                            choices=list(NETWORKS),
                            value=DEFAULT_NETWORK,
                            label="Network",
                        )
                        backend_dropdown: gr.Dropdown = gr.Dropdown(
                            choices=NETWORKS[DEFAULT_NETWORK],
                            value=DEFAULT_BACKENDS[DEFAULT_NETWORK],
                            label="Backend",
                        )
                        keypoint_thr: gr.Slider = gr.Slider(
                            0.0,
                            1.0,
                            value=0.3,
                            step=0.05,
                            label="Keypoint/Visibility Threshold",
                        )
                        show_skeleton: gr.Checkbox = gr.Checkbox(value=True, label="Show Approximate Skeleton")

                    examples: gr.helpers.Examples = gr.Examples(
                        examples=EXAMPLES,
                        inputs=[inp],
                        examples_per_page=4,
                        cache_examples=False,
                    )

                with gr.TabItem("Outputs", id="outputs"):
                    status_text: gr.Textbox = gr.Textbox(
                        label="Status",
                        value="Select an image to begin.",
                        interactive=False,
                    )
                    out_json: gr.JSON = gr.JSON(label="Predictions", value={"instances": []})

        with gr.Column(scale=5):
            viewer: Rerun = Rerun(
                label="2D Pose",
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=800,
            )

    network_dropdown.change(
        fn=_update_backend_choices,
        inputs=[network_dropdown],
        outputs=[backend_dropdown],
        api_visibility="private",
    )

    examples.load_input_event.then(
        fn=lambda: gr.update(selected="inputs"),
        inputs=None,
        outputs=[tabs],
        api_visibility="private",
    )

    run.click(
        fn=lambda: gr.update(selected="outputs"),
        inputs=None,
        outputs=[tabs],
        api_visibility="private",
    ).then(
        fn=lambda: (None, {"instances": []}, "Starting pose estimation..."),
        inputs=None,
        outputs=[viewer, out_json, status_text],
        api_visibility="private",
    ).then(
        fn=predict_ui,
        inputs=[inp, prompt_text, network_dropdown, backend_dropdown, keypoint_thr, show_skeleton],
        outputs=[viewer, out_json, status_text],
    )


# The viewer stream (crypto.randomUUID) and the image "Paste from clipboard"
# button (navigator.clipboard.read) need a secure context. Serve over HTTPS —
# on the tailnet: `tailscale serve --bg --https=7860 http://127.0.0.1:7860`.
def launch() -> None:
    """Launch the unified pose Gradio app."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    demo.launch(
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        # The example images live in sibling packages (mamma/sapiens2-pose
        # assets), outside this app's cwd — gradio blocks them otherwise.
        allowed_paths=sorted({str(Path(example).parent) for example in EXAMPLES}),
    )


if __name__ == "__main__":
    launch()
