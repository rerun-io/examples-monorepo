"""Batched GPU/TensorRT Sapiens COCO-133 video pipeline."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, SupportsFloat, SupportsInt, runtime_checkable

import numpy as np
import torch
import tyro
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from sapiens2_pose.api.coco133_gpu import (
    RTMLIB_POSE_INPUT_SIZE,
    RtmlibPoseInputBatch,
    SapiensPoseInputBatch,
    TensorRtRtmlibPoseRunner,
    TensorRtSapiensPoseRunner,
    TensorRtYoloxDetectorRunner,
    YoloxDetectorOutput,
    decode_rtmlib_simcc_to_coco133_torch,
    decode_sapiens_heatmaps_to_coco133_torch,
    postprocess_yolox_outputs_torch,
    prepare_rtmlib_pose_inputs_torch,
    prepare_sapiens_pose_inputs_torch,
    preprocess_yolox_detector_torch,
)
from sapiens2_pose.api.runtime import DEFAULT_MODEL_SIZE, ModelSize
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

from sapiens_coco133_pose.api.artifact import Coco133PoseFrame, log_video_pose_frames

TORCH_UINT8 = torch.__dict__["uint8"]
PoseBackend = Literal["sapiens", "rtmlib"]
DetectorRunner = Callable[[Float[Tensor, "batch 3 det_h det_w"]], YoloxDetectorOutput]
PoseRunner = Callable[[Float[Tensor, "batch 3 h w"]], tuple[Tensor, ...]]

DEFAULT_TRT_DIR = Path(__file__).resolve().parents[3] / "pretrained_models" / "tensorrt"
DEFAULT_DETECTOR_ENGINE_PATH = DEFAULT_TRT_DIR / "yolox_humanart_head_b8_fp16.trt"
DEFAULT_SAPIENS_POSE_ENGINE_PATH = DEFAULT_TRT_DIR / "sapiens2_0.4b_b8_fp16.trt"
DEFAULT_RTMLIB_POSE_ENGINE_PATH = DEFAULT_TRT_DIR / "rtmw_256x192_b32_fp16.trt"
DETECTOR_DEFAULT_BATCH_SIZE = 8
POSE_DEFAULT_BATCH_SIZE = 8
RTMLIB_POSE_DEFAULT_BATCH_SIZE = 32
DETECTOR_INPUT_SIZE = (640, 640)
DETECTOR_DEFAULT_SCORE_THR = 0.7
DETECTOR_DEFAULT_NMS_THR = 0.45

DETECTOR_INPUT_NAME = "input"
DETECTOR_OUTPUT_NAME = "1436"
DETECTOR_SCORE_OUTPUT_NAME = "1438"
SAPIENS_POSE_INPUT_NAME = "inputs"
SAPIENS_POSE_OUTPUT_NAME = "heatmaps"
RTMLIB_POSE_INPUT_NAME = "input"
RTMLIB_POSE_SIMCC_X_OUTPUT_NAME = "simcc_x"
RTMLIB_POSE_SIMCC_Y_OUTPUT_NAME = "simcc_y"

__all__ = (
    "AnnotatedPoseTensorRtConfig",
    "BatchedTensorRtCoco133Pipeline",
    "BatchedTensorRtPoseConfig",
    "BatchedTensorRtVideoConfig",
    "BatchedTensorRtVideoFramePredictor",
    "BatchedTensorRtVideoSummary",
    "RtmlibTensorRtPoseConfig",
    "SapiensTensorRtPoseConfig",
    "TensorRtDetectorConfig",
    "TensorRtRuntimeConfig",
    "create_batched_tensorrt_video_frame_predictor",
    "run_batched_tensorrt_video_coco133",
)


@runtime_checkable
class VideoDecoderMetadata(Protocol):
    """Metadata exposed by TorchCodec's CUDA video decoder."""

    @property
    def num_frames(self) -> SupportsInt | None:
        """Number of frames in the decoded video."""

    @property
    def average_fps(self) -> SupportsFloat | None:
        """Average video frame rate."""


@runtime_checkable
class VideoFrameBatch(Protocol):
    """Frame batch returned by TorchCodec video range reads."""

    @property
    def data(self) -> UInt8[Tensor, "batch h w 3"]:
        """CUDA uint8 RGB frames in NHWC order."""


@runtime_checkable
class CudaVideoDecoder(Protocol):
    """Protocol for the TorchCodec CUDA decoder used by the video path."""

    @property
    def metadata(self) -> VideoDecoderMetadata:
        """Video metadata read by TorchCodec."""

    def get_frames_in_range(self, start: int, stop: int) -> VideoFrameBatch:
        """Return decoded CUDA frames in ``[start, stop)``.

        Args:
            start: Inclusive start frame index.
            stop: Exclusive stop frame index.

        Returns:
            CUDA RGB frame batch in NHWC order.
        """


@dataclass(frozen=True, slots=True)
class TensorRtDetectorConfig:
    """TensorRT YOLOX HumanArt detector configuration."""

    engine_path: Path = DEFAULT_DETECTOR_ENGINE_PATH
    """Path to the YOLOX TensorRT engine."""

    @property
    def static_batch_size(self) -> int:
        """Static batch size baked into the shipped detector engine."""
        return DETECTOR_DEFAULT_BATCH_SIZE

    @property
    def input_size(self) -> tuple[int, int]:
        """Detector input size as ``(height, width)`` for the shipped engine."""
        return DETECTOR_INPUT_SIZE

    @property
    def input_name(self) -> str:
        """Fixed detector input binding name for the exported engine."""
        return DETECTOR_INPUT_NAME

    @property
    def output_name(self) -> str:
        """Fixed decoded-box output binding name for the exported engine."""
        return DETECTOR_OUTPUT_NAME

    @property
    def score_output_name(self) -> str:
        """Fixed decoded-score output binding name for the exported engine."""
        return DETECTOR_SCORE_OUTPUT_NAME


@dataclass(frozen=True, slots=True)
class SapiensTensorRtPoseConfig:
    """TensorRT Sapiens2 COCO-133 pose backend configuration."""

    engine_path: Path = DEFAULT_SAPIENS_POSE_ENGINE_PATH
    """Path to the Sapiens2 TensorRT pose engine."""

    @property
    def static_batch_size(self) -> int:
        """Static batch size baked into the shipped Sapiens pose engine."""
        return POSE_DEFAULT_BATCH_SIZE

    @property
    def model_size(self) -> ModelSize:
        """Sapiens model size used to decode the shipped engine heatmaps."""
        return DEFAULT_MODEL_SIZE

    @property
    def backend(self) -> Literal["sapiens"]:
        """Pose backend identifier."""
        return "sapiens"

    @property
    def input_name(self) -> str:
        """Fixed Sapiens pose input binding name."""
        return SAPIENS_POSE_INPUT_NAME

    @property
    def output_name(self) -> str:
        """Fixed Sapiens heatmap output binding name."""
        return SAPIENS_POSE_OUTPUT_NAME


@dataclass(frozen=True, slots=True)
class RtmlibTensorRtPoseConfig:
    """TensorRT RTMLib RTMW COCO-133 pose backend configuration."""

    engine_path: Path = DEFAULT_RTMLIB_POSE_ENGINE_PATH
    """Path to the RTMW TensorRT pose engine."""

    @property
    def static_batch_size(self) -> int:
        """Static batch size baked into the shipped RTMW pose engine."""
        return RTMLIB_POSE_DEFAULT_BATCH_SIZE

    @property
    def input_size(self) -> tuple[int, int]:
        """RTMW pose crop size as ``(width, height)`` for the shipped engine."""
        return RTMLIB_POSE_INPUT_SIZE

    @property
    def backend(self) -> Literal["rtmlib"]:
        """Pose backend identifier."""
        return "rtmlib"

    @property
    def input_name(self) -> str:
        """Fixed RTMW pose input binding name."""
        return RTMLIB_POSE_INPUT_NAME

    @property
    def simcc_x_output_name(self) -> str:
        """Fixed RTMW SimCC-x output binding name."""
        return RTMLIB_POSE_SIMCC_X_OUTPUT_NAME

    @property
    def simcc_y_output_name(self) -> str:
        """Fixed RTMW SimCC-y output binding name."""
        return RTMLIB_POSE_SIMCC_Y_OUTPUT_NAME


if TYPE_CHECKING:
    PoseTensorRtConfig = SapiensTensorRtPoseConfig | RtmlibTensorRtPoseConfig
else:
    pose_tensor_rt_defaults = {
        "sapiens": SapiensTensorRtPoseConfig(),
        "rtmlib": RtmlibTensorRtPoseConfig(),
    }
    PoseTensorRtConfig = tyro.extras.subcommand_type_from_defaults(pose_tensor_rt_defaults, prefix_names=False)

AnnotatedPoseTensorRtConfig = tyro.conf.OmitSubcommandPrefixes[PoseTensorRtConfig]


@dataclass(frozen=True, slots=True)
class TensorRtRuntimeConfig:
    """Runtime batching and detector postprocessing configuration."""

    detector_batch_size: int = DETECTOR_DEFAULT_BATCH_SIZE
    """Detector frames processed per TensorRT call."""
    pose_batch_size: int | None = None
    """Pose crops processed per TensorRT call. If unset, uses the selected pose engine static batch."""
    detector_score_thr: float = DETECTOR_DEFAULT_SCORE_THR
    """YOLOX person score threshold."""
    detector_nms_thr: float = DETECTOR_DEFAULT_NMS_THR
    """YOLOX person NMS threshold."""

    def effective_pose_batch_size(self, pose: PoseTensorRtConfig) -> int:
        """Return the runtime pose batch after applying the backend default.

        Args:
            pose: Selected TensorRT pose backend config.

        Returns:
            Explicit runtime pose batch size or the selected pose engine static batch.
        """
        return pose.static_batch_size if self.pose_batch_size is None else self.pose_batch_size

    def validate(self, *, detector_static_batch_size: int, pose_static_batch_size: int) -> None:
        """Validate runtime batches against static TensorRT engine batches.

        Args:
            detector_static_batch_size: Static batch size baked into the detector engine.
            pose_static_batch_size: Static batch size baked into the selected pose engine.

        Raises:
            ValueError: If runtime batch sizes are non-positive or exceed the static engine batches.
        """
        pose_batch_size: int = pose_static_batch_size if self.pose_batch_size is None else self.pose_batch_size
        if (
            min(self.detector_batch_size, pose_batch_size, detector_static_batch_size, pose_static_batch_size) <= 0
            or self.detector_batch_size > detector_static_batch_size
            or pose_batch_size > pose_static_batch_size
        ):
            raise ValueError("Runtime batch sizes must be positive and cannot exceed static TensorRT batches.")


@dataclass(frozen=True, slots=True)
class BatchedTensorRtPoseConfig:
    """TensorRT detector plus pose backend configuration."""

    detector: TensorRtDetectorConfig = field(default_factory=TensorRtDetectorConfig)
    """Detector engine and preprocessing configuration."""
    pose: AnnotatedPoseTensorRtConfig = field(default_factory=SapiensTensorRtPoseConfig)
    """Pose backend configuration selected by the ``sapiens`` or ``rtmlib`` subcommand."""
    execution_device: Literal["cuda"] = field(default="cuda", init=False)
    """TensorRT execution device type."""
    dtype: torch.dtype = field(default=torch.float16, init=False)
    """TensorRT pose input dtype."""

    def __post_init__(self) -> None:
        if (
            self.detector.static_batch_size <= 0
            or self.pose.static_batch_size <= 0
            or self.execution_device != "cuda"
            or self.dtype != torch.float16
        ):
            raise ValueError("The batched TensorRT path requires positive static CUDA FP16 batches.")

    @property
    def device(self) -> torch.device:
        """Backward-compatible alias for the TensorRT execution device."""
        return torch.device(self.execution_device)


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchedTensorRtVideoConfig:
    """Configuration for running batched TensorRT pose over a video."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun logging configuration."""
    video_path: Path
    """Input video path."""
    detector: TensorRtDetectorConfig = field(default_factory=TensorRtDetectorConfig)
    """Detector engine and preprocessing configuration."""
    pose: AnnotatedPoseTensorRtConfig = field(default_factory=SapiensTensorRtPoseConfig)
    """Pose backend configuration selected by the ``sapiens`` or ``rtmlib`` subcommand."""
    runtime: TensorRtRuntimeConfig = field(default_factory=TensorRtRuntimeConfig)
    """Runtime batch sizes and detector postprocessing thresholds."""
    max_frames: int | None = None
    """Maximum number of frames to process. If None, process the full video."""
    keypoint_threshold: float = 0.3
    """Confidence threshold used only for Rerun keypoint visibility."""
    include_images: bool = False
    """Whether returned frame artifacts should include CPU RGB images."""
    log_video_media: bool = True
    """Whether to log the source video media once into Rerun."""
    show_progress: bool = True
    """Whether to show tqdm progress over TensorRT batches."""

    def __post_init__(self) -> None:
        self.runtime.validate(
            detector_static_batch_size=self.detector.static_batch_size,
            pose_static_batch_size=self.pose.static_batch_size,
        )

    def to_pipeline_config(self) -> BatchedTensorRtPoseConfig:
        """Return the detector plus pose config used to initialize TensorRT runners.

        Returns:
            TensorRT detector and pose configuration without video/logging options.
        """
        return BatchedTensorRtPoseConfig(detector=self.detector, pose=self.pose)


class BatchedTensorRtVideoSummary(NamedTuple):
    """Summary returned after running batched TensorRT video pose."""

    output_path: Path | None
    """RRD output path when the Rerun config saved one."""
    frame_count: int
    """Number of processed frames."""
    person_count: int
    """Number of detected person instances."""


@dataclass(frozen=True, slots=True)
class BatchedTensorRtVideoFramePredictor:
    """Initialized CUDA decoder and TensorRT pipeline for video-frame inference."""

    config: BatchedTensorRtVideoConfig
    """Video inference configuration."""
    decoder: CudaVideoDecoder
    """CUDA video decoder."""
    fps: float
    """Video frame rate."""
    frame_count: int
    """Number of frames that will be processed."""
    pipeline: "BatchedTensorRtCoco133Pipeline"
    """Initialized batched TensorRT detector plus pose pipeline."""

    def predict_frames(self) -> list[Coco133PoseFrame]:
        """Run batched TensorRT inference over the configured video.

        Returns:
            Frame-major COCO-133 pose predictions for the configured frame range.
        """
        pose_batch_size: int = self.config.runtime.effective_pose_batch_size(self.config.pose)
        batch_stride: int = max(self.config.runtime.detector_batch_size, pose_batch_size)
        starts: Iterable[int] = range(0, self.frame_count, batch_stride)
        if self.config.show_progress:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc=f"{self.config.pose.backend} coco133 TRT batches")
        frames: list[Coco133PoseFrame] = []
        for start in starts:
            stop: int = min(start + batch_stride, self.frame_count)
            frames_rgb: UInt8[Tensor, "batch h w 3"] = self.decoder.get_frames_in_range(start, stop).data.contiguous()
            if frames_rgb.dtype != TORCH_UINT8:
                raise TypeError(f"Expected TorchCodec uint8 frames, got {frames_rgb.dtype}.")
            timestamps_ns: Int[ndarray, "batch"] = (
                np.arange(start, stop, dtype=np.float64) / self.fps * 1_000_000_000
            ).astype(np.int64)
            frames.extend(
                self.pipeline.predict_batch_rgb_tensor(
                    frames_rgb,
                    frame_start_index=start,
                    timestamps_ns=timestamps_ns,
                    detector_batch_size=self.config.runtime.detector_batch_size,
                    pose_batch_size=pose_batch_size,
                    score_thr=self.config.runtime.detector_score_thr,
                    nms_thr=self.config.runtime.detector_nms_thr,
                    include_images=self.config.include_images,
                )
            )
        return frames


class BatchedTensorRtCoco133Pipeline:
    """Batched CUDA/TensorRT person detection plus switchable COCO-133 pose."""

    def __init__(
        self,
        config: BatchedTensorRtPoseConfig,
        *,
        detector_runner_factory: Callable[..., DetectorRunner] | None = None,
        pose_runner_factory: Callable[..., PoseRunner] | None = None,
    ) -> None:
        """Initialize detector and pose TensorRT runners.

        Args:
            config: TensorRT detector and pose backend configuration.
            detector_runner_factory: Optional detector runner factory used by tests.
            pose_runner_factory: Optional pose runner factory used by tests.
        """
        self.config: BatchedTensorRtPoseConfig = config
        detector_config: TensorRtDetectorConfig = config.detector
        detector_factory: Callable[..., DetectorRunner] = detector_runner_factory or TensorRtYoloxDetectorRunner
        self.detector: DetectorRunner = detector_factory(
            detector_config.engine_path,
            input_name=detector_config.input_name,
            output_name=detector_config.output_name,
            score_output_name=detector_config.score_output_name,
        )
        pose_factory: Callable[..., PoseRunner]
        if isinstance(config.pose, RtmlibTensorRtPoseConfig):
            pose_config: RtmlibTensorRtPoseConfig = config.pose
            pose_factory = pose_runner_factory or TensorRtRtmlibPoseRunner
            self.pose: PoseRunner = pose_factory(
                pose_config.engine_path,
                input_name=pose_config.input_name,
                simcc_x_output_name=pose_config.simcc_x_output_name,
                simcc_y_output_name=pose_config.simcc_y_output_name,
            )
        else:
            sapiens_pose_config: SapiensTensorRtPoseConfig = config.pose
            pose_factory = pose_runner_factory or TensorRtSapiensPoseRunner
            self.pose = pose_factory(
                sapiens_pose_config.engine_path,
                input_name=sapiens_pose_config.input_name,
                output_name=sapiens_pose_config.output_name,
            )

    @torch.no_grad()
    def predict_batch_rgb_tensor(
        self,
        frames_rgb: UInt8[Tensor, "batch h w 3"],
        *,
        frame_start_index: int = 0,
        timestamps_ns: Int[ndarray, "batch"] | None = None,
        detector_batch_size: int | None = None,
        pose_batch_size: int | None = None,
        score_thr: float = DETECTOR_DEFAULT_SCORE_THR,
        nms_thr: float = DETECTOR_DEFAULT_NMS_THR,
        include_images: bool = False,
    ) -> list[Coco133PoseFrame]:
        """Run batched person detection and pose estimation on CUDA RGB frames.

        Args:
            frames_rgb: CUDA uint8 RGB frames with shape ``(batch, height, width, 3)``.
            frame_start_index: Global frame index for ``frames_rgb[0]``.
            timestamps_ns: Optional per-frame timestamps in nanoseconds.
            detector_batch_size: Optional detector runtime batch size.
            pose_batch_size: Optional pose runtime batch size.
            score_thr: YOLOX person score threshold.
            nms_thr: YOLOX person NMS threshold.
            include_images: Whether output frame records should include CPU RGB images.

        Returns:
            Frame-major COCO-133 pose predictions.

        Raises:
            ValueError: If inputs are not CUDA uint8 RGB frames or runtime batches exceed static engine batches.
        """
        if frames_rgb.device.type != self.config.execution_device or frames_rgb.dtype != TORCH_UINT8:
            raise ValueError("Batched TensorRT pipeline expects CUDA uint8 RGB frames.")
        det_bs: int = detector_batch_size or self.config.detector.static_batch_size
        pose_bs: int = pose_batch_size or self.config.pose.static_batch_size
        num_frames: int = int(frames_rgb.shape[0])
        if (
            det_bs <= 0
            or pose_bs <= 0
            or det_bs > self.config.detector.static_batch_size
            or pose_bs > self.config.pose.static_batch_size
        ):
            raise ValueError("Runtime batch sizes must be positive and cannot exceed static batches.")

        per_frame_boxes: list[Float[Tensor, "n 4"]] = []
        for start in range(0, num_frames, det_bs):
            stop: int = min(start + det_bs, num_frames)
            inputs, ratios = preprocess_yolox_detector_torch(
                frames_rgb[start:stop],
                model_input_size=self.config.detector.input_size,
            )
            per_frame_boxes.extend(
                postprocess_yolox_outputs_torch(
                    self.detector(inputs),
                    resize_ratios=ratios,
                    model_input_size=self.config.detector.input_size,
                    score_thr=score_thr,
                    nms_thr=nms_thr,
                )
            )

        person_counts: list[int] = []
        frame_rows: list[Int[Tensor, "n"]] = []
        bbox_rows: list[Float[Tensor, "n 4"]] = []
        for idx, boxes in enumerate(per_frame_boxes):
            count: int = int(boxes.shape[0])
            person_counts.append(count)
            if count:
                frame_rows.append(torch.full((count,), idx, dtype=torch.long, device=frames_rgb.device))
                bbox_rows.append(boxes.to(device=frames_rgb.device, dtype=torch.float32))
        if not bbox_rows:
            return [
                _empty_pose_frame(
                    frames_rgb,
                    local_frame_idx=idx,
                    global_frame_idx=frame_start_index + idx,
                    timestamps_ns=timestamps_ns,
                    include_images=include_images,
                )
                for idx in range(num_frames)
            ]

        frame_indices: Int[Tensor, "n_persons"] = torch.cat(frame_rows).long()
        bboxes: Float[Tensor, "n_persons 4"] = torch.cat(bbox_rows).float()
        if isinstance(self.config.pose, RtmlibTensorRtPoseConfig):
            keypoints, scores = self._predict_rtmlib_pose(
                frames_rgb,
                frame_indices=frame_indices,
                bboxes=bboxes,
                pose_batch_size=pose_bs,
            )
        else:
            keypoints, scores = self._predict_sapiens_pose(
                frames_rgb,
                frame_indices=frame_indices,
                bboxes=bboxes,
                pose_batch_size=pose_bs,
                pose_config=self.config.pose,
            )
        return _ordered_pose_rows_to_frames(
            frames_rgb,
            frame_start_index=frame_start_index,
            timestamps_ns=timestamps_ns,
            person_counts=person_counts,
            bboxes=bboxes.detach().cpu().numpy().astype(np.float32, copy=False),
            keypoints=keypoints,
            scores=scores,
            include_images=include_images,
        )

    def _predict_rtmlib_pose(
        self,
        frames_rgb: UInt8[Tensor, "batch h w 3"],
        *,
        frame_indices: Int[Tensor, "n_persons"],
        bboxes: Float[Tensor, "n_persons 4"],
        pose_batch_size: int,
    ) -> tuple[Float32[ndarray, "n_persons 133 2"], Float32[ndarray, "n_persons 133"]]:
        """Run RTMLib TensorRT pose inference for detected people.

        Args:
            frames_rgb: CUDA uint8 RGB frames with shape ``(batch, height, width, 3)``.
            frame_indices: Source frame index for each detected person.
            bboxes: Person boxes in image-space ``xyxy`` order.
            pose_batch_size: Runtime pose batch size.

        Returns:
            Image-space COCO-133 keypoints and confidence scores.

        Raises:
            TypeError: If the pipeline was not configured with the RTMLib backend.
        """
        if not isinstance(self.config.pose, RtmlibTensorRtPoseConfig):
            raise TypeError("RTMLib pose prediction requires an RTMLib pose config.")
        pose_config: RtmlibTensorRtPoseConfig = self.config.pose
        rtmlib_pose_batch: RtmlibPoseInputBatch = prepare_rtmlib_pose_inputs_torch(
            frames_rgb,
            frame_indices=frame_indices,
            bboxes_xyxy=bboxes,
            input_size=pose_config.input_size,
            dtype=self.config.dtype,
        )
        simcc_x_rows: list[Tensor] = []
        simcc_y_rows: list[Tensor] = []
        for start in range(0, int(rtmlib_pose_batch.inputs.shape[0]), pose_batch_size):
            stop: int = min(start + pose_batch_size, int(rtmlib_pose_batch.inputs.shape[0]))
            simcc_outputs: tuple[Tensor, ...] = self.pose(rtmlib_pose_batch.inputs[start:stop])
            simcc_x_rows.append(simcc_outputs[0])
            simcc_y_rows.append(simcc_outputs[1])
        return decode_rtmlib_simcc_to_coco133_torch(
            torch.cat(simcc_x_rows, dim=0),
            torch.cat(simcc_y_rows, dim=0),
            rtmlib_pose_batch,
        )

    def _predict_sapiens_pose(
        self,
        frames_rgb: UInt8[Tensor, "batch h w 3"],
        *,
        frame_indices: Int[Tensor, "n_persons"],
        bboxes: Float[Tensor, "n_persons 4"],
        pose_batch_size: int,
        pose_config: SapiensTensorRtPoseConfig,
    ) -> tuple[Float32[ndarray, "n_persons 133 2"], Float32[ndarray, "n_persons 133"]]:
        """Run Sapiens TensorRT pose inference for detected people.

        Args:
            frames_rgb: CUDA uint8 RGB frames with shape ``(batch, height, width, 3)``.
            frame_indices: Source frame index for each detected person.
            bboxes: Person boxes in image-space ``xyxy`` order.
            pose_batch_size: Runtime pose batch size.
            pose_config: Sapiens TensorRT pose backend configuration.

        Returns:
            Image-space COCO-133 keypoints and confidence scores.
        """
        pose_batch: SapiensPoseInputBatch = prepare_sapiens_pose_inputs_torch(
            frames_rgb,
            frame_indices=frame_indices,
            bboxes_xyxy=bboxes,
            dtype=self.config.dtype,
        )
        heatmaps: Tensor = torch.cat(
            [
                self.pose(pose_batch.inputs[start : min(start + pose_batch_size, int(pose_batch.inputs.shape[0]))])[0]
                for start in range(0, int(pose_batch.inputs.shape[0]), pose_batch_size)
            ],
            dim=0,
        )
        return decode_sapiens_heatmaps_to_coco133_torch(heatmaps, pose_batch, model_size=pose_config.model_size)


def create_batched_tensorrt_video_frame_predictor(
    config: BatchedTensorRtVideoConfig,
) -> BatchedTensorRtVideoFramePredictor:
    """Initialize CUDA decoder and TensorRT engines outside timed inference loops.

    Args:
        config: Batched TensorRT video configuration.

    Returns:
        Initialized predictor with CUDA decoder, video metadata, and TensorRT pipeline.

    Raises:
        ValueError: If valid frame count or FPS metadata cannot be read from the video.
    """
    from torchcodec.decoders import VideoDecoder

    decoder: CudaVideoDecoder = VideoDecoder(config.video_path, dimension_order="NHWC", device="cuda")
    frame_count_raw: SupportsInt | None = decoder.metadata.num_frames
    fps_raw: SupportsFloat | None = decoder.metadata.average_fps
    if frame_count_raw is None or fps_raw is None or float(fps_raw) <= 0.0 or int(frame_count_raw) < 0:
        raise ValueError(f"Could not read valid video metadata from {config.video_path}.")
    return BatchedTensorRtVideoFramePredictor(
        config,
        decoder,
        float(fps_raw),
        int(frame_count_raw) if config.max_frames is None else min(int(frame_count_raw), int(config.max_frames)),
        BatchedTensorRtCoco133Pipeline(config.to_pipeline_config()),
    )


def run_batched_tensorrt_video_coco133(config: BatchedTensorRtVideoConfig) -> BatchedTensorRtVideoSummary:
    """Run the optimized CUDA/TensorRT video path and log through ``RerunTyroConfig``.

    Args:
        config: Batched TensorRT video configuration.

    Returns:
        Output RRD path, processed frame count, and detected person count.
    """
    frames: list[Coco133PoseFrame] = create_batched_tensorrt_video_frame_predictor(config).predict_frames()
    log_video_pose_frames(
        frames,
        keypoint_threshold=config.keypoint_threshold,
        recording=config.rr_config.rec_stream,
        video_path=config.video_path if config.log_video_media else None,
    )
    return BatchedTensorRtVideoSummary(
        config.rr_config.save,
        len(frames),
        sum(int(frame.bboxes.shape[0]) for frame in frames),
    )


def _ordered_pose_rows_to_frames(
    frames_rgb: UInt8[Tensor, "batch h w 3"],
    *,
    frame_start_index: int,
    timestamps_ns: Int[ndarray, "batch"] | None,
    person_counts: list[int],
    bboxes: Float32[ndarray, "n_persons 4"],
    keypoints: Float32[ndarray, "n_persons 133 2"],
    scores: Float32[ndarray, "n_persons 133"],
    include_images: bool,
) -> list[Coco133PoseFrame]:
    frames: list[Coco133PoseFrame] = []
    row_start: int = 0
    for local_idx, count in enumerate(person_counts):
        row_stop: int = row_start + int(count)
        image: UInt8[ndarray, "h w 3"] | None = (
            frames_rgb[local_idx].detach().cpu().numpy().astype(np.uint8, copy=False) if include_images else None
        )
        frames.append(
            Coco133PoseFrame(
                frame_start_index + local_idx,
                0 if timestamps_ns is None else int(timestamps_ns[local_idx]),
                image,
                bboxes[row_start:row_stop].astype(np.float32, copy=False).reshape(-1, 4),
                keypoints[row_start:row_stop].astype(np.float32, copy=False).reshape(-1, 133, 2),
                scores[row_start:row_stop].astype(np.float32, copy=False).reshape(-1, 133),
            )
        )
        row_start = row_stop
    return frames


def _empty_pose_frame(
    frames_rgb: UInt8[Tensor, "batch h w 3"],
    *,
    local_frame_idx: int,
    global_frame_idx: int,
    timestamps_ns: Int[ndarray, "batch"] | None,
    include_images: bool,
) -> Coco133PoseFrame:
    image: UInt8[ndarray, "h w 3"] | None = (
        frames_rgb[local_frame_idx].detach().cpu().numpy().astype(np.uint8, copy=False) if include_images else None
    )
    return Coco133PoseFrame(
        global_frame_idx,
        0 if timestamps_ns is None else int(timestamps_ns[local_frame_idx]),
        image,
        np.empty((0, 4), np.float32),
        np.empty((0, 133, 2), np.float32),
        np.empty((0, 133), np.float32),
    )
