"""Optimized TensorRT-only WiLor video inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from tqdm.auto import tqdm

from wilor_nano.api.tensorrt_conversion import (
    DEFAULT_DETECTOR_ENGINE_PATH,
    DEFAULT_FULL_WILOR_ENGINE_PATH,
)
from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline_trt import (
    Detection,
    WiLorHandPose3dEstimationPipeline,
    WilorPipelineConfig,
    WilorPreds,
)
from wilor_nano.runtime import get_torch_device, get_torch_dtype

Side = Literal["left", "right"]
SideRows = tuple[list[int], list[list[float]], list[Float[ndarray, "21 2"]], list[Float[ndarray, "21 3"]]]
TORCH_UINT8: torch.dtype = torch.__dict__["uint8"]
DEFAULT_WILOR_ENGINE_PATH: Path = DEFAULT_FULL_WILOR_ENGINE_PATH


@dataclass(slots=True)
class BatchedTensorRtVideoConfig:
    """Clean preset for the accepted batched TensorRT WiLor video path."""

    rr_config: RerunTyroConfig  # noqa
    """Rerun recording configuration."""
    video_path: Path = Path("assets/video.mp4")
    """Video to run through the optimized pipeline."""
    max_frames: int | None = None
    """Optional maximum number of frames to process."""
    detector_engine_path: Path = DEFAULT_DETECTOR_ENGINE_PATH
    """Static B110 raw-detector TensorRT engine path."""
    wilor_engine_path: Path = DEFAULT_WILOR_ENGINE_PATH
    """Static B224 full-WiLor TensorRT engine path."""
    detector_batch_size: int = 110
    """Number of video frames per detector chunk."""
    detector_static_batch_size: int = 110
    """Static batch size baked into the detector TensorRT engine."""
    wilor_batch_size: int = 224
    """Number of hand crops per WiLor TensorRT chunk."""
    wilor_static_batch_size: int = 224
    """Static batch size baked into the full-WiLor TensorRT engine."""
    log_video_media: bool = True
    """Whether to log the source video media into Rerun before detections."""

    def __post_init__(self) -> None:
        """Validate runtime batch sizes against the static engine batch sizes.

        Raises:
            ValueError: If any batch size is non-positive or if a runtime batch
                size exceeds the corresponding static TensorRT engine batch.
        """
        if (
            min(
                self.detector_batch_size,
                self.detector_static_batch_size,
                self.wilor_batch_size,
                self.wilor_static_batch_size,
            )
            <= 0
        ):
            raise ValueError("Batch sizes must be positive.")
        if self.detector_batch_size > self.detector_static_batch_size:
            raise ValueError("detector_batch_size cannot exceed detector_static_batch_size.")
        if self.wilor_batch_size > self.wilor_static_batch_size:
            raise ValueError("wilor_batch_size cannot exceed wilor_static_batch_size.")

    def to_pipeline_config(self) -> WilorPipelineConfig:
        """Convert CLI config into the lower-level TRT pipeline config.

        Returns:
            ``WilorPipelineConfig`` with expanded engine paths and CUDA dtype.

        Raises:
            ValueError: If the current runtime device is not CUDA.
        """
        device = get_torch_device()
        if device.type != "cuda":
            raise ValueError("The TensorRT WiLor video path requires CUDA.")
        return WilorPipelineConfig(
            detector_engine_path=self.detector_engine_path.expanduser(),
            wilor_engine_path=self.wilor_engine_path.expanduser(),
            detector_static_batch_size=self.detector_static_batch_size,
            wilor_static_batch_size=self.wilor_static_batch_size,
            device=device,
            dtype=get_torch_dtype(device),
        )


def set_annotation_context() -> None:
    """Log the static Mediapipe hand annotation context to Rerun."""
    rr.log(
        "/",
        rr.AnnotationContext([
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=0, label="Hand", color=(0, 0, 255)),
                keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()],
                keypoint_connections=MEDIAPIPE_LINKS,
            )
        ]),
        static=True,
    )


def _video_timestamps_ns(video_path: Path, *, log_video_media: bool) -> Int[ndarray, "num_frames"]:
    """Resolve video frame timestamps, optionally logging encoded video media.

    Args:
        video_path: Path to the input video.
        log_video_media: Whether to log the source video stream to Rerun. When
            false, TorchCodec metadata is used only to synthesize timestamps.

    Returns:
        Nanosecond timestamps as ``Int[ndarray, "num_frames"]``.

    Raises:
        ValueError: If TorchCodec reports invalid FPS or frame-count metadata.
    """
    if log_video_media:
        return log_video(video_path, video_log_path=Path("video"), timeline="video_time")
    decoder: Any = VideoDecoder(video_path, dimension_order="NHWC", device="cpu")
    fps: float = float(decoder.metadata.average_fps)
    frame_count: int = int(decoder.metadata.num_frames)
    if fps <= 0.0 or frame_count < 0:
        raise ValueError(f"Could not read valid video metadata from {video_path}.")
    return (np.arange(frame_count, dtype=np.float64) / fps * 1_000_000_000).astype(np.int64)


def _group_outputs(
    frame_outputs: list[list[Detection]], frame_timestamps_ns: Int[ndarray, "num_frames"]
) -> dict[Side, SideRows]:
    """Group frame-major detections into left/right column rows.

    Args:
        frame_outputs: Frame-major detections returned by the TRT pipeline.
        frame_timestamps_ns: Timestamp for each processed frame.

    Returns:
        Mapping from side name to rows for timestamps, boxes, 2D keypoints, and
        3D keypoints.
    """
    grouped: dict[Side, SideRows] = {"left": ([], [], [], []), "right": ([], [], [], [])}
    for frame_idx, outputs in enumerate(frame_outputs):
        for output in outputs:
            times, boxes, points2d, points3d = grouped["right" if output["is_right"] == 1 else "left"]
            preds: WilorPreds = output["wilor_preds"]
            times.append(int(frame_timestamps_ns[frame_idx]))
            boxes.append(output["hand_bbox"])
            points2d.append(preds["pred_keypoints_2d"].reshape(21, 2))
            points3d.append(preds["pred_keypoints_3d"].reshape(21, 3))
    return grouped


def _log_static_fields(side: Side) -> None:
    """Log static Rerun archetype fields for one hand side.

    Args:
        side: Hand side namespace to initialize.
    """
    rr.log(
        f"{side}_xyz",
        rr.Points3D.from_fields(class_ids=0, keypoint_ids=MEDIAPIPE_IDS, show_labels=False, colors=(0, 255, 0)),
        static=True,
    )
    rr.log(
        f"video/{side}_keypoints",
        rr.Points2D.from_fields(class_ids=0, keypoint_ids=MEDIAPIPE_IDS, show_labels=False, colors=(0, 255, 0)),
        static=True,
    )
    rr.log(f"video/{side}_box", rr.Boxes2D.from_fields(show_labels=True), static=True)


def log_video_detections_columnar(
    frame_outputs: list[list[Detection]], frame_timestamps_ns: Int[ndarray, "num_frames"]
) -> None:
    """Log video detections with Rerun's columnar APIs.

    Args:
        frame_outputs: Frame-major detections with attached WiLor predictions.
        frame_timestamps_ns: Nanosecond timestamps for the processed frames.
    """
    for side, (times, boxes, points2d_rows, points3d_rows) in _group_outputs(
        frame_outputs, frame_timestamps_ns
    ).items():
        _log_static_fields(side)
        if len(times) == 0:
            continue
        timestamps_ns: Int[ndarray, "num_rows"] = np.asarray(times, dtype=np.int64)
        box_array: Float[ndarray, "num_rows 4"] = np.asarray(boxes, dtype=np.float32)
        centers: Float[ndarray, "num_rows 2"] = (box_array[:, :2] + box_array[:, 2:4]) / 2.0
        half_sizes: Float[ndarray, "num_rows 2"] = (box_array[:, 2:4] - box_array[:, :2]) / 2.0
        lengths: list[int] = [len(MEDIAPIPE_IDS)] * len(timestamps_ns)
        points2d: Float[ndarray, "num_rows 21 2"] = np.stack(points2d_rows).astype(np.float32)
        points3d: Float[ndarray, "num_rows 21 3"] = np.stack(points3d_rows).astype(np.float32)
        point_count: int = int(points2d.shape[0] * points2d.shape[1])
        indexes = [rr.TimeColumn("video_time", duration=1e-9 * timestamps_ns)]
        rr.send_columns(
            f"video/{side}_box",
            indexes=indexes,
            columns=rr.Boxes2D.columns(centers=centers, half_sizes=half_sizes).partition(
                lengths=[1] * len(timestamps_ns)
            ),
        )
        rr.send_columns(
            f"video/{side}_keypoints",
            indexes=indexes,
            columns=rr.Points2D.columns(positions=points2d.reshape(point_count, 2)).partition(lengths=lengths),
        )
        rr.send_columns(
            f"{side}_xyz",
            indexes=indexes,
            columns=rr.Points3D.columns(positions=points3d.reshape(point_count, 3)).partition(lengths=lengths),
        )


def run_batched_tensorrt_video(config: BatchedTensorRtVideoConfig) -> None:
    """Run the optimized video-only TensorRT path and log the result with Rerun.

    Args:
        config: CLI/runtime configuration. The nested ``rr_config`` is kept so
            Tyro constructs ``RerunTyroConfig`` and initializes Rerun sinks before
            inference starts.
    """

    set_annotation_context()
    timestamps_ns: Int[ndarray, "num_frames"] = _video_timestamps_ns(
        config.video_path, log_video_media=config.log_video_media
    )
    frame_count: int = len(timestamps_ns) if config.max_frames is None else min(config.max_frames, len(timestamps_ns))
    decoder: Any = VideoDecoder(config.video_path, dimension_order="NHWC", device="cuda")
    pipeline = WiLorHandPose3dEstimationPipeline(config.to_pipeline_config())
    outputs: list[list[Detection]] = []
    for start in tqdm(range(0, frame_count, config.detector_batch_size), desc="video frame batches"):
        stop: int = min(start + config.detector_batch_size, frame_count)
        frames: UInt8[Tensor, "batch h w 3"] = decoder.get_frames_in_range(start, stop).data.contiguous()
        if frames.dtype != TORCH_UINT8:
            raise TypeError(f"Expected TorchCodec uint8 frames, got {frames.dtype}.")
        outputs.extend(
            pipeline.predict_batch_rgb_tensor(
                frames, detector_batch_size=config.detector_batch_size, wilor_batch_size=config.wilor_batch_size
            )
        )
    log_video_detections_columnar(outputs, timestamps_ns[: len(outputs)])
