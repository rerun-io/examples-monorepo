"""Iterable PyTorch Sapiens COCO-133 pose pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray
from sapiens2_pose.api.metadata import project_instances_to_schema
from sapiens2_pose.api.runtime import (
    DEFAULT_MODEL_SIZE,
    DeviceChoice,
    ModelSize,
    PoseEstimation,
    estimate_frame_pose_from_bboxes,
)

from sapiens_coco133_pose.api.artifact import Coco133PoseFrame, write_video_pose_rrd

TrackerMode = Literal["lightweight", "balanced", "performance", "wholebody"]


@dataclass(frozen=True, slots=True)
class _RtmlibAssets:
    """RTMLib ONNX deploy-zip URLs and input sizes for one detector+pose preset."""

    det: str
    """Download URL for the YOLOX detector ONNX deploy zip."""
    det_input_size: tuple[int, int]
    """Input width-height pair expected by the detector."""
    pose: str
    """Download URL for the RTMPose ONNX deploy zip."""
    pose_input_size: tuple[int, int]
    """Input width-height pair expected by RTMPose."""


RTMLIB_ASSETS: dict[TrackerMode, _RtmlibAssets] = {
    "performance": _RtmlibAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
        pose_input_size=(288, 384),
    ),
    "lightweight": _RtmlibAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        det_input_size=(416, 416),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        pose_input_size=(192, 256),
    ),
    "balanced": _RtmlibAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        pose_input_size=(192, 256),
    ),
    "wholebody": _RtmlibAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        pose_input_size=(192, 256),
    ),
}
PoseBackend = Literal["sapiens", "rtmlib"]
RtmlibBackend = Literal["onnxruntime", "opencv"]
RtmlibDevice = Literal["cpu", "cuda"]
DEFAULT_ITERABLE_RRD_PATH = Path("/tmp/sapiens_coco133/iterable_golden.rrd")
DetectPersonsFn = Callable[[UInt8[ndarray, "h w 3"]], Float32[ndarray, "n 4"]]


@runtime_checkable
class NativePoseEstimatorFn(Protocol):
    """Callable interface for native Sapiens pose estimation from precomputed boxes."""

    def __call__(
        self,
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
        *,
        model_size: ModelSize = DEFAULT_MODEL_SIZE,
        device: DeviceChoice = "cuda",
    ) -> PoseEstimation:
        """Estimate native Sapiens keypoints for one RGB image and its person boxes."""


@runtime_checkable
class RtmlibPoseEstimatorFn(Protocol):
    """Callable interface for RTMLib COCO-133 pose estimation from precomputed boxes."""

    def __call__(
        self,
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
    ) -> tuple[Float32[ndarray, "n 133 2"], Float32[ndarray, "n 133"]]:
        """Estimate COCO-133 keypoints for one RGB image and its person boxes."""


@dataclass(frozen=True, slots=True)
class RtmlibYoloxDetectorConfig:
    """Configuration for the RTMLib YOLOX person detector used by MV-API."""

    mode: TrackerMode = "wholebody"
    """MV-API detector preset name."""
    backend: RtmlibBackend = "onnxruntime"
    """RTMLib inference backend."""
    device: RtmlibDevice = "cuda"
    """Device used by the detector backend."""
    nms_thr: float = 0.45
    """YOLOX non-maximum suppression threshold."""
    score_thr: float = 0.7
    """Minimum YOLOX person confidence score."""


@dataclass(frozen=True, slots=True)
class RtmlibPoseEstimatorConfig:
    """Configuration for MV-API's RTMLib COCO-133 pose estimator."""

    mode: TrackerMode = "wholebody"
    """MV-API pose preset name."""
    backend: RtmlibBackend = "onnxruntime"
    """RTMLib inference backend."""
    device: RtmlibDevice = "cuda"
    """Device used by the pose backend."""


@dataclass(frozen=True, slots=True)
class IterableVideoPoseConfig:
    """Configuration for the non-batched golden-artifact video path."""

    video_path: Path
    """Input video path."""
    rrd_path: Path = DEFAULT_ITERABLE_RRD_PATH
    """Output RRD path for the golden artifact."""
    max_frames: int | None = None
    """Maximum number of frames to process. If None, process the full video."""
    model_size: ModelSize = DEFAULT_MODEL_SIZE
    """Sapiens pose model size used by the native Sapiens backend."""
    detector_mode: TrackerMode = "wholebody"
    """MV-API detector preset used for person boxes."""
    detector_backend: RtmlibBackend = "onnxruntime"
    """RTMLib backend used by the detector."""
    detector_device: RtmlibDevice = "cuda"
    """Device used by RTMLib detector inference."""
    rtmlib_pose_mode: TrackerMode = "wholebody"
    """MV-API pose preset used when ``pose_backend`` is ``"rtmlib"``."""
    rtmlib_pose_backend: RtmlibBackend = "onnxruntime"
    """RTMLib backend used when ``pose_backend`` is ``"rtmlib"``."""
    rtmlib_pose_device: RtmlibDevice = "cuda"
    """Device used when ``pose_backend`` is ``"rtmlib"``."""
    pose_device: DeviceChoice = "cuda"
    """Device used by the native Sapiens pose estimator."""
    pose_backend: PoseBackend = "sapiens"
    """Pose backend run after person detection."""
    keypoint_threshold: float = 0.3
    """Confidence threshold used only for Rerun keypoint visibility."""
    log_images: bool = True
    """Whether to log RGB images into the output RRD."""
    show_progress: bool = True
    """Whether to show tqdm progress over video frames."""


class IterableVideoPoseSummary(NamedTuple):
    """Summary returned after running the iterable video pose path."""

    rrd_path: Path
    """Written RRD path."""
    frame_count: int
    """Number of processed video frames."""
    person_count: int
    """Number of detected person instances."""


class RtmlibYoloxPersonDetector:
    """Callable RGB person detector backed by MV-API's RTMLib YOLOX assets."""

    def __init__(self, config: RtmlibYoloxDetectorConfig) -> None:
        """Initialize the RTMLib YOLOX detector.

        Args:
            config: RTMLib detector backend, device, preset, and thresholds.
        """
        from rtmlib import YOLOX

        assets = RTMLIB_ASSETS[config.mode]
        self._model = YOLOX(
            assets.det,
            model_input_size=assets.det_input_size,
            nms_thr=config.nms_thr,
            score_thr=config.score_thr,
            backend=config.backend,
            device=config.device,
        )

    def __call__(self, image_rgb: UInt8[ndarray, "h w 3"]) -> Float32[ndarray, "n 4"]:
        """Detect person boxes in one RGB image.

        Args:
            image_rgb: RGB image with shape ``(height, width, 3)``.

        Returns:
            Person bounding boxes in image-space ``xyxy`` order.
        """
        det_output = self._model(np.ascontiguousarray(image_rgb[..., ::-1], dtype=np.uint8))
        raw = np.asarray(det_output[0] if isinstance(det_output, tuple) else det_output, dtype=np.float32)
        return (
            np.empty((0, 4), dtype=np.float32)
            if raw.size == 0
            else np.ascontiguousarray(raw.reshape(-1, raw.shape[-1])[:, :4], dtype=np.float32)
        )


class RtmlibCoco133PoseEstimator:
    """Callable RGB pose estimator backed by MV-API's RTMLib wholebody assets."""

    def __init__(self, config: RtmlibPoseEstimatorConfig) -> None:
        """Initialize the RTMLib COCO-133 pose estimator.

        Args:
            config: RTMLib pose backend, device, and preset.
        """
        from rtmlib import RTMPose

        assets = RTMLIB_ASSETS[config.mode]
        self._model = RTMPose(
            assets.pose,
            model_input_size=assets.pose_input_size,
            to_openpose=False,
            backend=config.backend,
            device=config.device,
        )

    def __call__(
        self, image_rgb: UInt8[ndarray, "h w 3"], bboxes: Float32[ndarray, "n 4"]
    ) -> tuple[Float32[ndarray, "n 133 2"], Float32[ndarray, "n 133"]]:
        """Estimate COCO-133 keypoints for detected people.

        Args:
            image_rgb: RGB image with shape ``(height, width, 3)``.
            bboxes: Person bounding boxes in image-space ``xyxy`` order.

        Returns:
            A tuple containing image-space keypoints with shape ``(n, 133, 2)`` and confidence scores with shape
            ``(n, 133)``.
        """
        bboxes_np: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if bboxes_np.shape[0] == 0:
            return np.empty((0, 133, 2), dtype=np.float32), np.empty((0, 133), dtype=np.float32)
        keypoints, scores = self._model(
            np.ascontiguousarray(image_rgb[..., ::-1], dtype=np.uint8), bboxes=bboxes_np.tolist()
        )
        return np.asarray(keypoints, dtype=np.float32).reshape(-1, 133, 2), np.asarray(
            scores, dtype=np.float32
        ).reshape(-1, 133)


@cache
def _cached_rtmlib_detector(config: RtmlibYoloxDetectorConfig) -> RtmlibYoloxPersonDetector:
    """Return a cached default RTMLib detector for direct frame-level calls."""
    return RtmlibYoloxPersonDetector(config)


@cache
def _cached_rtmlib_pose_estimator(config: RtmlibPoseEstimatorConfig) -> RtmlibCoco133PoseEstimator:
    """Return a cached default RTMLib pose estimator for direct frame-level calls."""
    return RtmlibCoco133PoseEstimator(config)


def estimate_frame_pose_coco133(
    image_rgb: UInt8[ndarray, "h w 3"],
    *,
    frame_index: int,
    timestamp_ns: int,
    detector: DetectPersonsFn | None = None,
    native_pose_estimator: NativePoseEstimatorFn | None = None,
    rtmlib_pose_estimator: RtmlibPoseEstimatorFn | None = None,
    pose_backend: PoseBackend = "sapiens",
    rtmlib_pose_config: RtmlibPoseEstimatorConfig | None = None,
    model_size: ModelSize = DEFAULT_MODEL_SIZE,
    device: DeviceChoice = "cuda",
) -> Coco133PoseFrame:
    """Run person detection and one pose backend for one RGB frame.

    Args:
        image_rgb: RGB image with shape ``(height, width, 3)``.
        frame_index: Zero-based source video frame index.
        timestamp_ns: Frame timestamp in nanoseconds.
        detector: Optional person detector override used by tests and alternate integrations.
        native_pose_estimator: Optional native Sapiens pose estimator override.
        rtmlib_pose_estimator: Optional RTMLib pose estimator override.
        pose_backend: Pose backend to run after person detection.
        rtmlib_pose_config: RTMLib pose config used when ``pose_backend`` is ``"rtmlib"``.
        model_size: Sapiens model size used when ``pose_backend`` is ``"sapiens"``.
        device: Sapiens inference device used when ``pose_backend`` is ``"sapiens"``.

    Returns:
        COCO-133 pose predictions for the frame.
    """
    effective_detector: DetectPersonsFn = detector or _cached_rtmlib_detector(RtmlibYoloxDetectorConfig())
    bboxes = np.asarray(
        effective_detector(image_rgb), dtype=np.float32
    ).reshape(-1, 4)
    if bboxes.shape[0] == 0:
        return Coco133PoseFrame(
            frame_index,
            timestamp_ns,
            image_rgb,
            bboxes,
            np.empty((0, 133, 2), np.float32),
            np.empty((0, 133), np.float32),
        )
    if pose_backend == "rtmlib":
        effective_rtmlib_pose_estimator: RtmlibPoseEstimatorFn = (
            rtmlib_pose_estimator
            or _cached_rtmlib_pose_estimator(rtmlib_pose_config or RtmlibPoseEstimatorConfig())
        )
        keypoints, scores = effective_rtmlib_pose_estimator(image_rgb, bboxes)
        return Coco133PoseFrame(
            frame_index,
            timestamp_ns,
            image_rgb,
            bboxes,
            np.asarray(keypoints, dtype=np.float32).reshape(-1, 133, 2),
            np.asarray(scores, dtype=np.float32).reshape(-1, 133),
        )
    effective_native_pose_estimator: NativePoseEstimatorFn = native_pose_estimator or estimate_frame_pose_from_bboxes
    native = effective_native_pose_estimator(image_rgb, bboxes, model_size=model_size, device=device)
    keypoints, scores = project_instances_to_schema(native.keypoints, native.scores, "coco133")
    return Coco133PoseFrame(
        frame_index,
        timestamp_ns,
        image_rgb,
        np.asarray(native.bboxes, dtype=np.float32).reshape(-1, 4),
        np.stack(keypoints).astype(np.float32, copy=False),
        np.stack(scores).astype(np.float32, copy=False),
    )


def iter_video_pose_coco133(
    config: IterableVideoPoseConfig,
    *,
    detector: DetectPersonsFn | None = None,
    native_pose_estimator: NativePoseEstimatorFn | None = None,
    rtmlib_pose_estimator: RtmlibPoseEstimatorFn | None = None,
) -> Iterator[Coco133PoseFrame]:
    """Yield non-batched COCO-133 pose predictions for a video.

    Args:
        config: Video path, output controls, detector settings, and pose backend settings.
        detector: Optional person detector override.
        native_pose_estimator: Optional native Sapiens pose estimator override.
        rtmlib_pose_estimator: Optional RTMLib pose estimator override.

    Yields:
        Per-frame COCO-133 pose predictions.

    Raises:
        ValueError: If valid frame count or FPS metadata cannot be read from the video.
    """
    from torchcodec.decoders import VideoDecoder

    effective_detector = detector or RtmlibYoloxPersonDetector(
        RtmlibYoloxDetectorConfig(config.detector_mode, config.detector_backend, config.detector_device)
    )
    effective_rtmlib_pose_estimator: RtmlibPoseEstimatorFn | None = rtmlib_pose_estimator
    rtmlib_pose_config = RtmlibPoseEstimatorConfig(
        config.rtmlib_pose_mode,
        config.rtmlib_pose_backend,
        config.rtmlib_pose_device,
    )
    if effective_rtmlib_pose_estimator is None and config.pose_backend == "rtmlib":
        effective_rtmlib_pose_estimator = RtmlibCoco133PoseEstimator(
            rtmlib_pose_config
        )
    decoder = VideoDecoder(config.video_path, dimension_order="NHWC", device="cpu")
    frame_count_raw, fps_raw = decoder.metadata.num_frames, decoder.metadata.average_fps
    if frame_count_raw is None or fps_raw is None or fps_raw <= 0.0 or frame_count_raw < 0:
        raise ValueError(f"Could not read valid video metadata from {config.video_path}.")
    frame_count: int = frame_count_raw if config.max_frames is None else min(frame_count_raw, config.max_frames)
    frame_indices: Iterable[int] = range(frame_count)
    if config.show_progress:
        from tqdm.auto import tqdm

        frame_indices = tqdm(frame_indices, total=frame_count, desc=f"{config.pose_backend} coco133 iterable frames")
    for frame_index in frame_indices:
        image_rgb = np.asarray(
            decoder.get_frames_in_range(frame_index, frame_index + 1).data[0].cpu().numpy(), dtype=np.uint8
        )
        frame = estimate_frame_pose_coco133(
            image_rgb,
            frame_index=frame_index,
            timestamp_ns=round(frame_index / fps_raw * 1_000_000_000),
            detector=effective_detector,
            native_pose_estimator=native_pose_estimator,
            rtmlib_pose_estimator=effective_rtmlib_pose_estimator,
            pose_backend=config.pose_backend,
            rtmlib_pose_config=rtmlib_pose_config,
            model_size=config.model_size,
            device=config.pose_device,
        )
        yield (
            frame
            if config.log_images
            else Coco133PoseFrame(
                frame.frame_index, frame.timestamp_ns, None, frame.bboxes, frame.keypoints, frame.scores
            )
        )


def run_iterable_video_pose_coco133(
    config: IterableVideoPoseConfig,
    *,
    detector: DetectPersonsFn | None = None,
    native_pose_estimator: NativePoseEstimatorFn | None = None,
    rtmlib_pose_estimator: RtmlibPoseEstimatorFn | None = None,
) -> IterableVideoPoseSummary:
    """Run the non-batched path and write a Rerun artifact.

    Args:
        config: Iterable video pose configuration.
        detector: Optional person detector override.
        native_pose_estimator: Optional native Sapiens pose estimator override.
        rtmlib_pose_estimator: Optional RTMLib pose estimator override.

    Returns:
        Output RRD path and processed frame/person counts.
    """
    frames = list(
        iter_video_pose_coco133(
            config,
            detector=detector,
            native_pose_estimator=native_pose_estimator,
            rtmlib_pose_estimator=rtmlib_pose_estimator,
        )
    )
    write_video_pose_rrd(
        frames,
        rrd_path=config.rrd_path,
        keypoint_threshold=config.keypoint_threshold,
        video_path=config.video_path if config.log_images else None,
    )
    return IterableVideoPoseSummary(config.rrd_path, len(frames), sum(int(frame.bboxes.shape[0]) for frame in frames))
