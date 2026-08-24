"""Sapiens2 pose-estimation Gradio UI using the lightweight runtime."""

import os
import socket
import uuid
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gradio_rerun import Rerun
from huggingface_hub import hf_hub_download
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from simplecv.rerun_custom_types import Points2DWithConfidence, confidence_scores_to_rgb
from transformers import DetrForObjectDetection, DetrImageProcessor

from sapiens2_pose.api.pose_artifact import PosePredictionArtifact
from sapiens2_pose.api.tensorrt_pose import default_tensorrt_engine_path
from sapiens2_pose.sapiens_lite.pose import estimate_pose, init_pose_model, nms, parse_pose_metainfo

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
CONFIGS_DIR = ASSETS_DIR / "configs"

POSE_MODELS = {
    "0.4B": {
        "repo": "facebook/sapiens2-pose-0.4b",
        "filename": "sapiens2_0.4b_pose.safetensors",
    },
    "0.8B": {
        "repo": "facebook/sapiens2-pose-0.8b",
        "filename": "sapiens2_0.8b_pose.safetensors",
    },
    "1B": {
        "repo": "facebook/sapiens2-pose-1b",
        "filename": "sapiens2_1b_pose.safetensors",
    },
    "5B": {
        "repo": "facebook/sapiens2-pose-5b",
        "filename": "sapiens2_5b_pose.safetensors",
    },
}
DEFAULT_SIZE = "1B"

DETECTOR_MODEL_ID = "facebook/detr-resnet-50"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BBOX_THR = 0.3
NMS_THR = 0.3

TENSORRT_MODEL_SIZE = "0.4B"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _server_port() -> int:
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port is not None:
        return int(env_port)

    for port in range(7860, 8060):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("Cannot find empty localhost port in range 7860-8059.")


_pose_model_cache: dict[str, Any] = {}
_detector_cache: dict[str, Any] = {}
_trt_runner_cache: dict[str, Any] = {}
_metainfo_cache: dict[str, Any] | None = None
_skeleton_cache: dict[str, Any] | None = None


def _get_trt_runner(engine_path: str) -> Any:
    cache_key: str = str(Path(engine_path).expanduser().resolve())
    if cache_key not in _trt_runner_cache:
        if not Path(cache_key).is_file():
            raise gr.Error(f"TensorRT engine not found at {cache_key}. Build it with the sapiens2-pose-build-trt task.")
        from sapiens2_pose.api.tensorrt_pose import TensorRtPoseHeatmapRunner

        _trt_runner_cache[cache_key] = TensorRtPoseHeatmapRunner(Path(cache_key))
        if len(_trt_runner_cache) > 1:
            oldest_key: str = next(iter(_trt_runner_cache))
            evicted: Any = _trt_runner_cache.pop(oldest_key)
            # TensorRT allocates outside Torch's caching allocator; del drops the last reference.
            del evicted
    return _trt_runner_cache[cache_key]


def _get_metainfo() -> dict[str, Any]:
    global _metainfo_cache
    if _metainfo_cache is None:
        meta_path = CONFIGS_DIR / "_base_" / "keypoints308.py"
        _metainfo_cache = parse_pose_metainfo(dict(from_file=str(meta_path)))
    return _metainfo_cache


def _get_sapiens_skeleton() -> dict[str, Any]:
    global _skeleton_cache
    if _skeleton_cache is None:
        meta = _get_metainfo()
        ids = list(range(int(meta["num_keypoints"])))
        _skeleton_cache = {
            "ids": ids,
            "id2name": {
                int(idx): str(meta["keypoint_id2name"][idx])
                for idx in ids
            },
            "links": [
                (int(a), int(b))
                for a, b in meta["skeleton_links"]
            ],
            "keypoint_colors": np.asarray(meta["keypoint_colors"], dtype=np.uint8),
        }
    return _skeleton_cache


def _get_detector() -> tuple[Any, Any]:
    if "model" not in _detector_cache:
        proc = DetrImageProcessor.from_pretrained(DETECTOR_MODEL_ID)
        model = DetrForObjectDetection.from_pretrained(DETECTOR_MODEL_ID).eval().to(DEVICE)
        _detector_cache["proc"] = proc
        _detector_cache["model"] = model
    return _detector_cache["proc"], _detector_cache["model"]


def _get_pose_model(size: str) -> Any:
    if size not in _pose_model_cache:
        spec = POSE_MODELS[size]
        ckpt = hf_hub_download(repo_id=spec["repo"], filename=spec["filename"])
        model: Any = init_pose_model(size, ckpt, device=DEVICE)
        model.pose_metainfo = _get_metainfo()
        _pose_model_cache[size] = model
    return _pose_model_cache[size]


if _env_flag("SAPIENS_PRELOAD_MODELS", default=False):
    print("[startup] pre-loading detector + all pose sizes ...")
    _get_detector()
    for _size in POSE_MODELS:
        _get_pose_model(_size)
    print("[startup] ready.")
else:
    print("[startup] lazy model loading enabled.")


def _detect_persons(image_rgb: np.ndarray) -> np.ndarray:
    proc, model = _get_detector()
    pil_img = Image.fromarray(image_rgb)
    inputs = proc(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]], device=DEVICE)
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=BBOX_THR
    )[0]
    person_mask = results["labels"] == 1
    boxes = results["boxes"][person_mask].cpu().numpy()
    scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)
    bboxes = np.concatenate([boxes, scores], axis=1)
    bboxes = bboxes[nms(bboxes, NMS_THR), :4]
    if len(bboxes) == 0:
        h, w = image_rgb.shape[:2]
        bboxes = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
    return bboxes


def _ensure_uuid(recording_id: uuid.UUID | str | None) -> uuid.UUID:
    if recording_id is None:
        return uuid.uuid4()
    if isinstance(recording_id, uuid.UUID):
        return recording_id
    return uuid.UUID(str(recording_id))


def _get_recording(recording_id: uuid.UUID | str | None) -> rr.RecordingStream:
    return rr.RecordingStream(
        application_id="sapiens2_pose",
        recording_id=_ensure_uuid(recording_id),
    )


def _log_annotation_context() -> None:
    skeleton = _get_sapiens_skeleton()
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Sapiens2 308", color=(0, 255, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=idx, label=name)
                        for idx, name in skeleton["id2name"].items()
                    ],
                    keypoint_connections=skeleton["links"],
                )
            ]
        ),
        static=True,
    )


def _log_pose_recording(
    image_rgb: np.ndarray,
    bboxes: np.ndarray,
    keypoints: Float32[ndarray, "n k 2"],
    scores: Float32[ndarray, "n k"],
    kpt_thr: float,
) -> None:
    skeleton = _get_sapiens_skeleton()
    keypoint_ids = skeleton["ids"]

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Spatial2DView(name="Pose", origin="image"),
            collapse_panels=True,
        )
    )
    rr.set_time("iteration", sequence=0)
    _log_annotation_context()
    rr.log("image", rr.Image(image_rgb, color_model=rr.ColorModel.RGB))

    for idx, (bbox, kpts, scr) in enumerate(zip(bboxes, keypoints, scores, strict=False)):
        rr.log(
            f"image/person_{idx}/bbox",
            rr.Boxes2D(
                array=bbox.reshape(1, 4),
                array_format=rr.Box2DFormat.XYXY,
                class_ids=0,
                labels=f"person_{idx}",
                colors=(0, 255, 0),
            ),
        )

        kpts_arr = kpts.copy()
        scores_arr = scr.reshape(-1)
        kpts_arr[scores_arr < kpt_thr] = np.nan
        confidence_rgb: UInt8[ndarray, "k 3"] = confidence_scores_to_rgb(scores_arr.astype(np.float32)[None, :, None])[0]
        rr.log(
            f"image/person_{idx}/keypoints",
            Points2DWithConfidence(
                positions=kpts_arr,
                confidences=scores_arr,
                class_ids=0,
                keypoint_ids=keypoint_ids,
                colors=confidence_rgb,
            ),
        )


def _predict_impl(
    image: Image.Image,
    size: str,
    kpt_thr: float,
    use_tensorrt: bool,
    recording_id: uuid.UUID | str | None,
):
    if image is None:
        raise gr.Error("No image provided.")

    image_pil = image.convert("RGB")
    image_rgb = np.array(image_pil)

    bboxes = _detect_persons(image_rgb)
    if use_tensorrt:
        if size != TENSORRT_MODEL_SIZE:
            raise gr.Error(f"The TensorRT engine is a static {TENSORRT_MODEL_SIZE} build; select the {TENSORRT_MODEL_SIZE} model.")
        engine_path: str = default_tensorrt_engine_path()
        from sapiens2_pose.api.tensorrt_pose import estimate_sapiens_pose_tensorrt

        artifact: PosePredictionArtifact = estimate_sapiens_pose_tensorrt(
            image_rgb,
            np.asarray(bboxes, dtype=np.float32),
            engine_path=Path(engine_path),
            model_size=TENSORRT_MODEL_SIZE,
            heatmap_runner=_get_trt_runner(engine_path),
        )
        keypoints: Float32[ndarray, "n k 2"] = artifact.keypoints
        scores: Float32[ndarray, "n k"] = artifact.scores
    else:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        model = _get_pose_model(size)
        pose_result: tuple[list[np.ndarray], list[np.ndarray]] = estimate_pose(image_bgr, bboxes, model)
        keypoints = np.stack(pose_result[0], axis=0).astype(np.float32, copy=False)
        scores = np.stack(pose_result[1], axis=0).astype(np.float32, copy=False)

    instances = [
        {
            "bbox": bbox.reshape(-1)[:4].tolist(),
            "keypoints": kpts.tolist(),
            "keypoint_scores": s.reshape(-1).tolist(),
        }
        for bbox, kpts, s in zip(bboxes, keypoints, scores, strict=False)
    ]
    payload = {"instances": instances}

    recording = _get_recording(recording_id)
    stream = recording.binary_stream()
    with recording:
        _log_pose_recording(image_rgb, bboxes, keypoints, scores, kpt_thr)

    yield stream.read(), payload


def predict_ui(
    image: Image.Image,
    size: str,
    kpt_thr: float,
    use_tensorrt: bool,
    recording_id: uuid.UUID | str | None,
):
    backend_label = "TensorRT" if use_tensorrt else "PyTorch"
    for stream, payload in _predict_impl(image, size, kpt_thr, use_tensorrt, recording_id):
        count = len(payload["instances"])
        suffix = "" if count == 1 else "s"
        yield stream, payload, f"Complete: {count} person{suffix} detected with Sapiens2 {size} ({backend_label})."


def _switch_to_outputs():
    return gr.update(selected="outputs")


def _switch_to_inputs():
    return gr.update(selected="inputs")


def _reset_run_outputs():
    return None, {"instances": []}, "Starting Sapiens2 pose estimation..."


EXAMPLES = sorted(
    str(ASSETS_DIR / "images" / n)
    for n in os.listdir(ASSETS_DIR / "images")
    if n.lower().endswith((".jpg", ".jpeg", ".png"))
)

CUSTOM_CSS = """
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

HEADER_HTML = """
<div id="title">Sapiens2: Pose</div>
<div id="subtitle">ICLR 2026</div>
<div id="badges">
  <a class="pill" href="https://github.com/facebookresearch/sapiens2" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.3.1 2 1.3 2 1.3 1.1 1.9 3 1.4 3.7 1 .1-.8.4-1.4.8-1.7-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.3-3.2-.1-.4-.6-1.6.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.6.2 2.8.1 3.2.8.8 1.3 1.9 1.3 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .3"/></svg>
    Code
  </a>
  <a class="pill" href="https://huggingface.co/facebook/sapiens2" target="_blank" rel="noopener">
    🤗 Models
  </a>
  <a class="pill" href="https://arxiv.org/pdf/2604.21681" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="15" y2="17"/></svg>
    Paper
  </a>
  <a class="pill" href="https://rawalkhirodkar.github.io/sapiens2" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
    Project
  </a>
</div>
"""

# The viewer stream (crypto.randomUUID) and the image "Paste from clipboard"
# button (navigator.clipboard.read) need a secure context. Serve over HTTPS —
# on the tailnet: `tailscale serve --bg --https=7860 http://127.0.0.1:7860`.
with gr.Blocks(title="Sapiens2 Pose") as demo:
    gr.HTML(HEADER_HTML)
    recording_id = gr.State(str(uuid.uuid4()))

    with gr.Row():
        with gr.Column(scale=1):
            tabs = gr.Tabs(selected="inputs")
            with tabs:
                with gr.TabItem("Inputs", id="inputs"):
                    inp = gr.Image(label="Input Image", type="pil", height=360)
                    run = gr.Button("Run Sapiens2 Pose", variant="primary")

                    with gr.Accordion("Config", open=False):
                        thr = gr.Slider(
                            0.0,
                            1.0,
                            value=0.3,
                            step=0.05,
                            label="Keypoint Threshold",
                        )
                        size = gr.Radio(
                            choices=list(POSE_MODELS.keys()),
                            value=DEFAULT_SIZE,
                            label="Model",
                        )
                        use_trt = gr.Checkbox(
                            value=False,
                            label=f"Use TensorRT Backend ({TENSORRT_MODEL_SIZE} only)",
                        )

                    examples = gr.Examples(
                        examples=EXAMPLES,
                        inputs=[inp],
                        examples_per_page=8,
                        cache_examples=False,
                    )

                with gr.TabItem("Outputs", id="outputs"):
                    status_text = gr.Textbox(
                        label="Status",
                        value="Select an image to begin.",
                        interactive=False,
                    )
                    out_json = gr.JSON(label="Keypoints", value={"instances": []})

        with gr.Column(scale=5):
            viewer = Rerun(
                label="Pose",
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

    run_event = run.click(
        fn=_switch_to_outputs,
        inputs=None,
        outputs=[tabs],
        api_visibility="private",
    ).then(
        fn=_reset_run_outputs,
        inputs=None,
        outputs=[viewer, out_json, status_text],
        api_visibility="private",
    ).then(
        fn=predict_ui,
        inputs=[inp, size, thr, use_trt, recording_id],
        outputs=[viewer, out_json, status_text],
    )

    run_event.then(
        lambda: str(uuid.uuid4()),
        inputs=None,
        outputs=recording_id,
        api_visibility="private",
    )


def launch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    demo.launch(
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        server_port=_server_port(),
    )


if __name__ == "__main__":
    launch()
