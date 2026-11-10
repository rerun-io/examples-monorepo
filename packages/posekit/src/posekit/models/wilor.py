"""WiLoR as posekit hand roles: YOLO hand detector + top-down WiLoR (ViT + RefineNet + MANO).

Two backends keep the two paths of the former ``wilor-nano`` package:

- ``torch``: the reference. Ultralytics YOLO per frame, OpenCV crops with an
  anti-aliasing blur, the fp16 PyTorch network per frame.
- ``tensorrt``: static-batch engines built through trtkit from ONNX exported
  once into the trtkit cache. Letterbox on the GPU, Ultralytics NMS, GPU crops
  (no blur), one crop batch across all frames.

WiLoR predicts in a virtual pinhole camera: focal ``5000 / 256 * max(w, h)``,
principal point at the image centre. ``xyz_camera`` and ``vertices`` are in
that camera; a supplied ``CalibratedCamera`` is not used. That focal comes from
WiLoR's near-orthographic training crops and puts the hands metres out;
``WilorHands.in_pinhole`` keeps the wrist-relative hands and solves their
translation in a real pinhole instead.
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from huggingface_hub import snapshot_download
from jaxtyping import Bool, Float, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor
from trtkit import TensorRtBackendConfig, TensorRtRuntime, TensorRuntime, TorchBackendConfig
from trtkit.artifacts import build_once
from trtkit.trt_builder import TrtBuildConfig, ensure_engine

from posekit.artifacts import OnnxTransform
from posekit.cameras import CalibratedCamera
from posekit.models.base import HandDetector, TopDownHandPose3d
from posekit.models.wilor_net.geometry import cam_crop_to_full, gen_trans_from_patch_cv, generate_image_patch_cv2, perspective_projection
from posekit.models.wilor_net.patches import generate_rgb_image_patches_torch
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d, HandSide, Keypoints3d, validate_frames_rgb
from posekit.skeletons import HAND_21, KeypointSkeleton

if TYPE_CHECKING:
    WilorBackendConfig = TorchBackendConfig | TensorRtBackendConfig
else:
    WilorBackendConfig = tyro.extras.subcommand_type_from_defaults(
        {"torch": TorchBackendConfig(autocast="fp16"), "tensorrt": TensorRtBackendConfig()}, prefix_names=False
    )

WILOR_REVISION: str = "2e05e331c35255af64fe5ef555362bd8454a5ce5"
"""Pinned commit of pablovela5620/wilor-nano (detector.pt, wilor_final.ckpt, MANO, mean params)."""
WILOR_OUTPUT_NAMES: tuple[str, ...] = ("global_orient", "hand_pose", "betas", "pred_cam", "pred_keypoints_3d", "pred_vertices")
"""Engine outputs of the post-crop network, in export order."""
FOCAL_LENGTH: float = 5000.0
"""WiLoR's training focal length for a 256 px crop."""
CROP_SIZE: int = 256
"""Network crop side in pixels."""
DETECTOR_SHAPE: tuple[int, int] = (512, 416)
"""Static TensorRT detector input as ``(height, width)``."""
SIDES: tuple[HandSide, HandSide] = ("left", "right")
"""YOLO class id -> anatomical side."""


def weights_dir(repo_id: str, revision: str, override: Path | None) -> Path:
    """Directory holding ``pretrained_models/`` at the pinned revision (or ``override``)."""
    if override is not None:
        return override
    return Path(snapshot_download(repo_id=repo_id, revision=revision))


def _export_static(model: torch.nn.Module, example: Tensor, onnx_path: Path, input_name: str, output_names: list[str], opset: int) -> None:
    """The export recipe of wilor-nano's tensorrt_conversion: dynamo, static batch, model dtype unchanged."""
    partial: Path = onnx_path.with_name(f".{onnx_path.stem}.partial.onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (example,),
            partial,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=output_names,
            dynamic_axes=None,
            dynamo=True,
        )
    sidecar: Path = partial.with_suffix(".onnx.data")
    if sidecar.exists():
        # The graph names its external weights by file name; keep it beside the published graph.
        import onnx

        graph = onnx.load(partial, load_external_data=True)
        onnx.save(graph, onnx_path, save_as_external_data=True, location=f"{onnx_path.name}.data")
        partial.unlink()
        sidecar.unlink()
    else:
        partial.replace(onnx_path)


class _DetectorExport(torch.nn.Module):
    """Raw YOLO head output ``(b, 4 + classes, anchors)``, without NMS."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: torch.nn.Module = model

    def forward(self, images: Float[Tensor, "b 3 512 416"]) -> Float[Tensor, "b fields anchors"]:
        output: Any = self.model(images)
        return output[0] if isinstance(output, (tuple, list)) else output


class _NetworkExport(torch.nn.Module):
    """Post-crop WiLoR network with its outputs as a fixed tuple."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: torch.nn.Module = model

    def forward(self, crops: Float[Tensor, "b 256 256 3"]) -> tuple[Tensor, ...]:
        output: dict[str, Tensor] = self.model(crops)
        return tuple(output[name] for name in WILOR_OUTPUT_NAMES)


@dataclass(frozen=True, slots=True)
class WilorHandDetectorConfig:
    """WiLoR's YOLO hand detector (classes: left, right)."""

    backend: WilorBackendConfig = field(default_factory=lambda: TorchBackendConfig(autocast="fp16"))
    """``torch`` runs Ultralytics per frame; ``tensorrt`` a static-batch engine with GPU letterbox."""
    score_threshold: float = 0.3
    """Minimum hand confidence."""
    nms_iou: float = 0.7
    """Non-maximum suppression IoU threshold."""
    max_detections: int = 300
    """Maximum boxes kept per frame."""
    static_batch_size: int = 110
    """Frames per TensorRT engine call (baked into the exported graph)."""
    weights: Path | None = None
    """Local directory holding ``pretrained_models/detector.pt``; None downloads the pinned revision."""
    hf_repo_id: str = "pablovela5620/wilor-nano"
    """Hugging Face repo holding the weights."""
    hf_revision: str = WILOR_REVISION
    """Pinned commit of that repo."""
    onnx_transform: tyro.conf.Suppress[OnnxTransform | None] = None
    """Optional graph transform applied to the exported ONNX before the engine build."""

    def setup(self) -> "WilorHandDetector":
        """Load the weights, build/load the backend, and return a ready detector."""
        return WilorHandDetector(self)


class WilorHandDetector(HandDetector):
    """YOLO hand boxes with their left/right class as anatomical side."""

    def __init__(self, config: WilorHandDetectorConfig) -> None:
        from ultralytics import YOLO

        self.config: WilorHandDetectorConfig = config
        detector_path: Path = weights_dir(config.hf_repo_id, config.hf_revision, config.weights) / "pretrained_models" / "detector.pt"
        self.yolo: Any = YOLO(detector_path)
        self.yolo.to(torch.device("cuda"))
        self.runtime: TensorRuntime | None = None
        if isinstance(config.backend, TensorRtBackendConfig):
            batch: int = config.static_batch_size
            onnx_path: Path = config.backend.cache_dir / f"wilor-detector-b{batch}-512x416-{config.hf_revision[:8]}.onnx"
            assert isinstance(self.yolo.model, torch.nn.Module)
            model: torch.nn.Module = _DetectorExport(self.yolo.model).to(device="cuda", dtype=torch.float32).eval()
            build_once(
                onnx_path,
                lambda: _export_static(model, torch.zeros((batch, 3, *DETECTOR_SHAPE), device="cuda"), onnx_path, "images", ["output0"], 18),
            )
            if config.onnx_transform is not None:
                onnx_path = config.onnx_transform(onnx_path)
            engine: Path = ensure_engine(onnx_path, TrtBuildConfig(max_batch_size=batch, opt_batch_size=batch), cache_dir=config.backend.cache_dir)
            self.runtime = TensorRtRuntime(engine, use_cuda_graph=config.backend.use_cuda_graph)
            self.yolo = None

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> HandDetections:
        """Detect hands in every frame; boxes stay in detector order within a frame."""
        validate_frames_rgb(frames_rgb)
        rows: list[Float32[Tensor, "m 6"]] = []
        indices: list[Tensor] = []
        if self.runtime is None:
            for index, frame in enumerate(frames_rgb):
                result: Any = self.yolo(frame.cpu().numpy(), conf=self.config.score_threshold, verbose=False)[0]
                data: Tensor = result.boxes.data.float()
                rows.append(data)
                indices.append(torch.full((len(data),), index, dtype=torch.int64, device=data.device))
        else:
            from ultralytics.utils import nms, ops

            image_hw: tuple[int, int] = (int(frames_rgb.shape[1]), int(frames_rgb.shape[2]))
            for start in range(0, len(frames_rgb), self.config.static_batch_size):
                chunk: UInt8[Tensor, "c h w 3"] = frames_rgb[start : start + self.config.static_batch_size]
                raw: Tensor = self.runtime({"images": _letterbox_bgr(chunk)})["output0"]
                for offset, prediction in enumerate(
                    nms.non_max_suppression(
                        raw.float(), self.config.score_threshold, self.config.nms_iou, None, False, max_det=self.config.max_detections, nc=2
                    )
                ):
                    # Box, score, class; the head's hand keypoints after them are unused, as before.
                    boxes: Tensor = prediction[:, :6].clone()
                    if len(boxes):
                        scaled: Tensor | ndarray = ops.scale_boxes(DETECTOR_SHAPE, boxes[:, :4], image_hw)
                        assert isinstance(scaled, Tensor)
                        boxes[:, :4] = scaled
                    rows.append(boxes)
                    indices.append(torch.full((len(boxes),), start + offset, dtype=torch.int64, device=boxes.device))
        data_all: Float32[Tensor, "n 6"] = torch.cat(rows).to(frames_rgb.device)
        if not len(data_all):
            return HandDetections.empty(frames_rgb.device)
        return HandDetections(
            BoxDetections(data_all[:, :4].contiguous(), data_all[:, 4].contiguous(), torch.cat(indices).to(frames_rgb.device)),
            tuple(SIDES[int(label)] for label in data_all[:, 5].tolist()),
        )


def _letterbox_bgr(frames_rgb: UInt8[Tensor, "b h w 3"]) -> Float32[Tensor, "b 3 512 416"]:
    """Resize and centre-pad RGB frames into the detector's BGR ``[0, 1]`` input, YOLO fill ``114 / 255``."""
    height: int = int(frames_rgb.shape[1])
    width: int = int(frames_rgb.shape[2])
    ratio: float = min(DETECTOR_SHAPE[0] / height, DETECTOR_SHAPE[1] / width)
    resized_w: int = round(width * ratio)
    resized_h: int = round(height * ratio)
    pad_w: int = DETECTOR_SHAPE[1] - resized_w
    pad_h: int = DETECTOR_SHAPE[0] - resized_h
    frames_bgr: Float32[Tensor, "b 3 h w"] = frames_rgb.flip((-1,)).permute(0, 3, 1, 2).float() / 255.0
    resized: Float32[Tensor, "b 3 rh rw"] = F.interpolate(frames_bgr, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    # Bilinear kernels can overshoot 0..1 by an ULP on some platforms (GB10/aarch64); the contract is a closed [0, 1].
    padding: tuple[int, int, int, int] = (round(pad_w / 2 - 0.1), round(pad_w / 2 + 0.1), round(pad_h / 2 - 0.1), round(pad_h / 2 + 0.1))
    return F.pad(resized.clamp_(0.0, 1.0), padding, value=114.0 / 255.0).contiguous()


@dataclass(frozen=True, slots=True, kw_only=True)
class WilorHands(HandPoses3d):
    """MANO hands from WiLoR, in its virtual full-image camera (``in_pinhole`` moves them into another pinhole)."""

    global_orient: Float32[Tensor, "n 1 3"]
    """Wrist orientation (axis-angle), left hands mirrored back from the right-hand frame."""
    hand_pose: Float32[Tensor, "n 15 3"]
    """MANO joint rotations (axis-angle)."""
    betas: Float32[Tensor, "n 10"]
    """MANO shape coefficients."""
    cam_t_full: Float32[Tensor, "n 3"]
    """Translation of the wrist-relative MANO output into the virtual camera, metres."""
    focal_length: Float32[Tensor, "n"]
    """Virtual camera focal length in pixels; principal point at the image centre."""

    def in_pinhole(self, intrinsics: Float64[ndarray, "3 3"]) -> "WilorHands":
        """The same hands in the pinhole ``intrinsics``: each hand keeps its wrist-relative joints and mesh
        and gets the translation (``fit_translation``) that reprojects its joints onto WiLoR's own 2D joints.

        ``keypoints.xy`` stays as predicted; ``cam_t_full`` and ``focal_length`` describe the new camera.
        """
        if self.num_hands == 0 or self.keypoints.xyz_camera is None or self.vertices is None:
            return self
        camera: Float32[Tensor, "3 3"] = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.keypoints.xy.device)
        wrist_relative: Float32[Tensor, "n 21 3"] = self.keypoints.xyz_camera - self.cam_t_full[:, None]
        translation: Float32[Tensor, "n 3"] = fit_translation(wrist_relative, self.keypoints.xy, camera)
        return replace(
            self,
            keypoints=replace(self.keypoints, xyz_camera=wrist_relative + translation[:, None]),
            vertices=self.vertices - self.cam_t_full[:, None] + translation[:, None],
            cam_t_full=translation,
            focal_length=torch.full_like(self.focal_length, float(intrinsics[0, 0])),
        )


def fit_translation(points: Float32[Tensor, "n k 3"], uv: Float32[Tensor, "n k 2"], intrinsics: Float32[Tensor, "3 3"]) -> Float32[Tensor, "n 3"]:
    """Translation ``t`` per hand such that the pinhole ``intrinsics`` projects ``points + t`` onto ``uv``.

    Each joint gives two equations linear in ``t`` (``fx tx - (u - cx) tz = (u - cx) Z - fx X``, the same in y);
    one batched least-squares solve answers all hands. The equations weight each joint by its depth, which
    is nearly constant across one hand, so this is the reprojection least squares in practice.
    """
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    du: Float32[Tensor, "n k"] = uv[..., 0] - cx
    dv: Float32[Tensor, "n k"] = uv[..., 1] - cy
    zeros: Float32[Tensor, "n k"] = torch.zeros_like(du)
    rows: Float32[Tensor, "n rows 3"] = torch.cat(
        [torch.stack([fx.expand_as(du), zeros, -du], dim=-1), torch.stack([zeros, fy.expand_as(dv), -dv], dim=-1)], dim=1
    )
    targets: Float32[Tensor, "n rows 1"] = torch.cat([du * points[..., 2] - fx * points[..., 0], dv * points[..., 2] - fy * points[..., 1]], dim=1)[..., None]
    return torch.linalg.lstsq(rows, targets).solution[..., 0]


@dataclass(frozen=True, slots=True)
class WilorHandPoseConfig:
    """WiLoR top-down 3D hands (ViT-H backbone, RefineNet, MANO)."""

    backend: WilorBackendConfig = field(default_factory=lambda: TorchBackendConfig(autocast="fp16"))
    """``torch`` (fp16 or fp32) reference, or a static-batch fp16 ``tensorrt`` engine."""
    rescale_factor: float = 2.5
    """Crop side as a multiple of the longer box side."""
    static_batch_size: int = 224
    """Crops per TensorRT engine call (baked into the exported graph)."""
    weights: Path | None = None
    """Local directory holding ``pretrained_models/``; None downloads the pinned revision."""
    hf_repo_id: str = "pablovela5620/wilor-nano"
    """Hugging Face repo holding the weights."""
    hf_revision: str = WILOR_REVISION
    """Pinned commit of that repo."""
    onnx_transform: tyro.conf.Suppress[OnnxTransform | None] = None
    """Optional graph transform applied to the exported ONNX before the engine build."""

    def setup(self) -> "WilorHandPose3d":
        """Load the weights, build/load the backend, and return a ready model."""
        return WilorHandPose3d(self)


def reference_hand_patch(
    image: UInt8[ndarray, "h w 3"], center_x: np.float32, center_y: np.float32, box_size: np.float32, flip: bool
) -> UInt8[ndarray, "256 256 3"] | Float64[ndarray, "256 256 3"]:
    """The torch path's crop: Gaussian blur before downsampling (as the WiLoR demo does), then an OpenCV warp.

    Only the source window the warp samples is blurred: the box, its bilinear neighbours and the
    Gaussian's reach (scipy ``truncate=4``), so every sampled pixel equals the full-frame blur. The
    window goes into a lazily zeroed full-size canvas, so the warp runs with the reference's exact
    transform; a shifted origin rounds OpenCV's fixed-point source coordinates differently.

    Returns:
        The crop; uint8 without a blur, float64 after one (skimage's float type for uint8).
    """
    from skimage.filters import gaussian

    # float32 like the reference, so sigma matches bit for bit.
    downsampling: np.float32 = box_size * 1.0 / CROP_SIZE / 2.0
    if downsampling <= 1.1:
        patch, _ = generate_image_patch_cv2(
            image, center_x, center_y, box_size, box_size, CROP_SIZE, CROP_SIZE, flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT
        )
        return patch
    sigma: np.float32 = (downsampling - 1) / 2
    margin: int = int(4.0 * float(sigma) + 0.5) + 2
    height, width = image.shape[:2]
    half: float = float(box_size) / 2.0
    x0: int = max(math.floor(center_x - half) - margin, 0)
    y0: int = max(math.floor(center_y - half) - margin, 0)
    x1: int = min(math.ceil(center_x + half) + margin + 1, width)
    y1: int = min(math.ceil(center_y + half) + margin + 1, height)
    window: Float64[ndarray, "wh ww 3"] = gaussian(image[y0:y1, x0:x1], sigma=sigma, channel_axis=2, preserve_range=True)
    canvas: Float64[ndarray, "h w 3"] = np.zeros((height, width, 3), dtype=np.float64)
    if flip:
        canvas[y0:y1, width - x1 : width - x0] = window[:, ::-1]
    else:
        canvas[y0:y1, x0:x1] = window
    # generate_image_patch_cv2's transform and warp, with the flip already applied to the canvas.
    trans: Float64[ndarray, "2 3"] = gen_trans_from_patch_cv(
        width - center_x - 1 if flip else center_x, center_y, box_size, box_size, CROP_SIZE, CROP_SIZE, 1.0, 0
    )
    return cv2.warpAffine(canvas, trans, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def load_wilor_network(directory: Path, dtype: torch.dtype) -> torch.nn.Module:
    """The WiLoR network (``wilor_net.network.WiLor``) with its MANO layer, on CUDA in eval mode."""
    from posekit.models.wilor_net.network import WiLor

    models: Path = directory / "pretrained_models"
    network = WiLor(
        mano_root_dir=models / "mano_clean", mano_mean_path=models / "mano_mean_params.npz", focal_length=FOCAL_LENGTH, image_size=CROP_SIZE
    )
    network.load_state_dict(torch.load(models / "wilor_final.ckpt")["state_dict"], strict=False)
    return network.eval().to("cuda", dtype=dtype)


class WilorHandPose3d(TopDownHandPose3d):
    """Hand boxes + sides -> MANO joints and meshes (left hands run mirrored through the right-hand network)."""

    skeleton: KeypointSkeleton = HAND_21

    def __init__(self, config: WilorHandPoseConfig) -> None:
        self.config: WilorHandPoseConfig = config
        tensorrt: bool = isinstance(config.backend, TensorRtBackendConfig)
        dtype: torch.dtype = torch.float32 if isinstance(config.backend, TorchBackendConfig) and config.backend.autocast == "fp32" else torch.float16
        network: torch.nn.Module = load_wilor_network(weights_dir(config.hf_repo_id, config.hf_revision, config.weights), dtype)
        mano_faces: Tensor = network.get_buffer("mano.th_faces")
        right: Int64[Tensor, "f 3"] = mano_faces.to(device="cuda", dtype=torch.int64)
        self.faces: dict[HandSide, Tensor] = {"right": right, "left": right[:, [0, 2, 1]].contiguous()}
        self.dtype: torch.dtype = dtype
        self.network: torch.nn.Module | None = network
        self.runtime: TensorRuntime | None = None
        if tensorrt:
            backend = cast(TensorRtBackendConfig, config.backend)
            batch: int = config.static_batch_size
            onnx_path: Path = backend.cache_dir / f"wilor-postcrop-b{batch}-fp16-{config.hf_revision[:8]}.onnx"
            export = _NetworkExport(network).to(device="cuda", dtype=torch.float16).eval()
            build_once(
                onnx_path,
                lambda: _export_static(
                    export, torch.zeros((batch, CROP_SIZE, CROP_SIZE, 3), dtype=torch.float16, device="cuda"), onnx_path, "img_patches",
                    list(WILOR_OUTPUT_NAMES), 17,
                ),
            )
            if config.onnx_transform is not None:
                onnx_path = config.onnx_transform(onnx_path)
            engine: Path = ensure_engine(
                onnx_path, TrtBuildConfig(max_batch_size=batch, opt_batch_size=batch, allow_tf32=False), cache_dir=backend.cache_dir
            )
            self.runtime = TensorRtRuntime(engine, use_cuda_graph=backend.use_cuda_graph)
            self.network = None
            del export, network
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(
        self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: HandDetections, camera: CalibratedCamera | None = None
    ) -> WilorHands:
        """Reconstruct every detected hand; rows follow the detections.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Hand boxes and sides referencing ``frames_rgb`` by index.
            camera: Ignored; WiLoR predicts in its own virtual pinhole camera.

        Returns:
            One hand per detection.
        """
        validate_frames_rgb(frames_rgb)
        boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
        frames: Int64[ndarray, "n"] = detections.boxes.frame_indices.cpu().numpy().astype(np.int64)
        rights: Bool[ndarray, "n"] = np.asarray([side == "right" for side in detections.sides], dtype=bool)
        centers: Float32[ndarray, "n 2"] = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        box_sizes: Float32[ndarray, "n"] = (self.config.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2])).max(axis=1)
        outputs: dict[str, Float32[ndarray, "..."]] = (
            self._run_torch(frames_rgb, frames, centers, box_sizes, rights) if self.runtime is None else self._run_tensorrt(frames_rgb, frames, centers, box_sizes, rights)
        )
        return self._to_hands(outputs, detections, centers, box_sizes, rights, (int(frames_rgb.shape[2]), int(frames_rgb.shape[1])))

    @torch.inference_mode()
    def hands_from_crops(self, crops: Float[Tensor, "n 256 256 3"], detections: HandDetections, image_wh: tuple[int, int]) -> WilorHands:
        """Reconstruct hands from crops the caller made (RGB, 0..255, left hands already mirrored).

        Callers with their own crop geometry (mv-api's keypoint refiner) share the network and the
        camera lift; the crop must follow WiLoR's convention around ``detections``' boxes.

        Args:
            crops: One NHWC crop per detection row, on CUDA.
            detections: The boxes and sides the crops were taken from.
            image_wh: Source image size as ``(width, height)``.

        Returns:
            One hand per detection.
        """
        boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
        rights: Bool[ndarray, "n"] = np.asarray([side == "right" for side in detections.sides], dtype=bool)
        centers: Float32[ndarray, "n 2"] = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        box_sizes: Float32[ndarray, "n"] = (self.config.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2])).max(axis=1)
        if self.network is not None:
            output: dict[str, Tensor] = self.network(crops.to(device="cuda", dtype=self.dtype))
            outputs: dict[str, Float32[ndarray, "..."]] = {
                name: output[name].cpu().float().numpy() for name in WILOR_OUTPUT_NAMES
            }
        else:
            outputs = self._run_engine(crops.to(dtype=torch.float16).contiguous())
        return self._to_hands(outputs, detections, centers, box_sizes, rights, image_wh)

    def _run_torch(
        self,
        frames_rgb: UInt8[Tensor, "b h w 3"],
        frames: Int64[ndarray, "n"],
        centers: Float32[ndarray, "n 2"],
        box_sizes: Float32[ndarray, "n"],
        rights: Bool[ndarray, "n"],
    ) -> dict[str, Float32[ndarray, "..."]]:
        """OpenCV crops with an anti-aliasing blur, one network call per frame (the reference path)."""
        assert self.network is not None
        chunks: dict[str, list[ndarray]] = {name: [] for name in WILOR_OUTPUT_NAMES}
        for frame in np.unique(frames):
            rows: Int64[ndarray, "m"] = np.flatnonzero(frames == frame)
            image: UInt8[ndarray, "h w 3"] = frames_rgb[int(frame)].cpu().numpy()
            patches: list[ndarray] = [
                reference_hand_patch(image, centers[row, 0], centers[row, 1], box_sizes[row], bool(not rights[row])) for row in rows
            ]
            output: dict[str, Tensor] = self.network(torch.from_numpy(np.stack(patches)).to(device="cuda", dtype=self.dtype))
            for name in WILOR_OUTPUT_NAMES:
                chunks[name].append(output[name].cpu().float().numpy())
        return {name: np.concatenate(values) if values else np.empty((0,), np.float32) for name, values in chunks.items()}

    def _run_tensorrt(
        self,
        frames_rgb: UInt8[Tensor, "b h w 3"],
        frames: Int64[ndarray, "n"],
        centers: Float32[ndarray, "n 2"],
        box_sizes: Float32[ndarray, "n"],
        rights: Bool[ndarray, "n"],
    ) -> dict[str, Float32[ndarray, "..."]]:
        """GPU crops for every hand at once, then static-batch engine calls."""
        assert self.runtime is not None
        patches: Float[Tensor, "n 256 256 3"] = generate_rgb_image_patches_torch(
            frames_rgb,
            frame_indices=torch.from_numpy(frames).cuda(),
            centers_xy=torch.from_numpy(centers).cuda(),
            box_sizes=torch.from_numpy(box_sizes).cuda(),
            flips=torch.from_numpy(~rights).cuda(),
            patch_size=CROP_SIZE,
        )
        return self._run_engine(patches.round().clamp(0, 255).to(dtype=torch.float16).contiguous())

    def _run_engine(self, crops: Float[Tensor, "n 256 256 3"]) -> dict[str, Float32[ndarray, "..."]]:
        """Static-batch engine calls over fp16 crops."""
        assert self.runtime is not None
        chunks: dict[str, list[ndarray]] = {name: [] for name in WILOR_OUTPUT_NAMES}
        for start in range(0, len(crops), self.config.static_batch_size):
            output: dict[str, Tensor] = self.runtime({"img_patches": crops[start : start + self.config.static_batch_size]})
            for name in WILOR_OUTPUT_NAMES:
                chunks[name].append(output[name].cpu().float().numpy())
        return {name: np.concatenate(values) if values else np.empty((0,), np.float32) for name, values in chunks.items()}

    def _to_hands(
        self,
        outputs: dict[str, Float32[ndarray, "..."]],
        detections: HandDetections,
        centers: Float32[ndarray, "n 2"],
        box_sizes: Float32[ndarray, "n"],
        rights: Bool[ndarray, "n"],
        image_wh: tuple[int, int],
    ) -> WilorHands:
        """Mirror left hands back, lift the crop camera to the full image, and project the joints."""
        count: int = len(rights)
        device: torch.device = detections.boxes.xyxy.device
        if count == 0:
            empty = torch.empty((0,), dtype=torch.float32, device=device)
            return WilorHands(
                keypoints=Keypoints3d.empty(HAND_21, device, root_indices=(0,), with_xyz_camera=True),
                sides=(),
                vertices=empty.reshape(0, 778, 3),
                faces=self.faces,
                global_orient=empty.reshape(0, 1, 3),
                hand_pose=empty.reshape(0, 15, 3),
                betas=empty.reshape(0, 10),
                cam_t_full=empty.reshape(0, 3),
                focal_length=empty,
            )
        left: Bool[ndarray, "n"] = ~rights
        pred_cam: Float32[ndarray, "n 3"] = outputs["pred_cam"].copy()
        pred_cam[:, 1] = np.where(rights, 1.0, -1.0).astype(np.float32) * pred_cam[:, 1]
        keypoints_3d: Float32[ndarray, "n 21 3"] = outputs["pred_keypoints_3d"].copy()
        vertices: Float32[ndarray, "n 778 3"] = outputs["pred_vertices"].copy()
        global_orient: Float32[ndarray, "n 1 3"] = outputs["global_orient"].copy()
        hand_pose: Float32[ndarray, "n 15 3"] = outputs["hand_pose"].copy()
        keypoints_3d[left, :, 0] *= -1.0
        vertices[left, :, 0] *= -1.0
        global_orient[left, :, 1:3] *= -1.0
        hand_pose[left, :, 1:3] *= -1.0
        image_sizes: Float32[ndarray, "n 2"] = np.tile(np.asarray(image_wh, dtype=np.float32), (count, 1))
        focal: Float32[ndarray, "n"] = np.full((count,), FOCAL_LENGTH / CROP_SIZE * max(image_wh), dtype=np.float32)
        cam_t: Float32[ndarray, "n 3"] = cam_crop_to_full(pred_cam, centers, box_sizes, image_sizes, focal).astype(np.float32)
        xy: Float32[ndarray, "n 21 2"] = perspective_projection(
            keypoints_3d, translation=cam_t, focal_length=np.stack([focal, focal], axis=-1), camera_center=image_sizes / 2
        ).astype(np.float32)
        xyz: Float32[Tensor, "n 21 3"] = torch.from_numpy(keypoints_3d + cam_t[:, None]).to(device)
        return WilorHands(
            keypoints=Keypoints3d(
                torch.from_numpy(xy).to(device),
                xyz[:, :, 2] - xyz[:, :1, 2],
                detections.boxes.scores[:, None].expand(-1, 21).contiguous(),
                detections.boxes.frame_indices,
                HAND_21,
                (0,),
                xyz,
            ),
            sides=detections.sides,
            vertices=torch.from_numpy(vertices + cam_t[:, None]).to(device),
            faces=self.faces,
            global_orient=torch.from_numpy(global_orient).to(device),
            hand_pose=torch.from_numpy(hand_pose).to(device),
            betas=torch.from_numpy(outputs["betas"].astype(np.float32)).to(device),
            cam_t_full=torch.from_numpy(cam_t).to(device),
            focal_length=torch.from_numpy(focal).to(device),
        )
