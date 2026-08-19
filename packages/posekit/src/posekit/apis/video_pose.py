"""Video 2D pose demo: CUDA decode -> detector -> top-down pose -> Rerun.

Frames stay on GPU from TorchCodec's CUDA decoder through detection, crop
generation, network inference, and keypoint decode; only final keypoints/boxes
cross to CPU for Rerun logging.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from torch import Tensor
from tqdm.auto import tqdm

from posekit.models import AnnotatedDetectorConfig, AnnotatedPose2dConfig, Pose2dPipeline, RtmPoseConfig, YoloxDetectorConfig
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.rerun_logging import log_person_points2d, log_skeleton_annotation_context


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoPoseConfig:
    """Configuration for the GPU video pose demo."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun logging configuration."""
    video_path: Path
    """Input video path."""
    detector: AnnotatedDetectorConfig = field(default_factory=YoloxDetectorConfig)
    """Person detector selected by subcommand (e.g. ``yolox``)."""
    pose: AnnotatedPose2dConfig = field(default_factory=RtmPoseConfig)
    """Pose estimator selected by subcommand (e.g. ``rtmpose`` or ``sapiens``)."""
    frame_batch_size: int = 32
    """Frames decoded and processed per pipeline step."""
    max_frames: int | None = None
    """Maximum number of frames to process; ``None`` runs the full video."""
    keypoint_threshold: float = 0.3
    """Minimum keypoint confidence for Rerun visibility."""
    max_people: int = 8
    """Number of stable per-person Rerun entity slots."""

    def setup_pipeline(self) -> Pose2dPipeline:
        """Instantiate the configured detector and pose stages."""
        return Pose2dPipeline(self.detector.setup(), self.pose.setup())


def log_frame_predictions(
    *,
    frame_indices: Int[ndarray, "chunk"],
    timestamps_ns: Int[ndarray, "chunk"],
    detections: BoxDetections,
    keypoints: Keypoints2d,
    keypoint_threshold: float,
    max_people: int,
    live_slots: set[int],
) -> None:
    """Log one processed chunk's boxes and keypoints onto the video timeline.

    Args:
        frame_indices: Global frame indices of the chunk.
        timestamps_ns: Per-frame video timestamps in nanoseconds.
        detections: GPU detections for the chunk.
        keypoints: GPU keypoints for the chunk.
        keypoint_threshold: Minimum keypoint confidence for visibility.
        max_people: Number of stable per-person entity slots.
        live_slots: Slots currently holding data, updated in place. A slot is
            cleared only on the populated -> empty transition so unused person
            entities are never created in the recording.
    """
    person_frames: Int[ndarray, "n"] = detections.frame_indices.cpu().numpy()
    boxes: Float32[ndarray, "n 4"] = detections.xyxy_numpy()
    xy: Float32[ndarray, "n k 2"] = keypoints.xy_numpy()
    scores: Float32[ndarray, "n k"] = keypoints.scores_numpy()
    for local_idx, (frame_idx, timestamp_ns) in enumerate(zip(frame_indices, timestamps_ns, strict=True)):
        rr.set_time("frame", sequence=int(frame_idx))
        rr.set_time("video_time", duration=1e-9 * float(timestamp_ns))
        rows: Int[ndarray, "m"] = np.flatnonzero(person_frames == local_idx)
        for slot in range(max_people):
            bbox_path: str = f"video/person_{slot}/bbox"
            keypoints_path: str = f"video/person_{slot}/keypoints"
            if slot >= int(rows.shape[0]):
                if slot in live_slots:
                    rr.log(bbox_path, rr.Clear(recursive=False))
                    rr.log(keypoints_path, rr.Clear(recursive=False))
                    live_slots.discard(slot)
                continue
            live_slots.add(slot)
            row: int = int(rows[slot])
            rr.log(bbox_path, rr.Boxes2D(array=boxes[row][None], array_format=rr.Box2DFormat.XYXY))
            log_person_points2d(
                keypoints_path,
                xy[row],
                scores[row],
                keypoint_threshold,
                keypoint_ids=np.arange(xy.shape[1], dtype=np.uint16),
                class_ids=0,
            )


def main(config: VideoPoseConfig) -> None:
    """Run the GPU video pose demo and log results to Rerun.

    Args:
        config: Demo configuration.

    Raises:
        ValueError: If the video metadata cannot be read.
    """
    from torchcodec.decoders import VideoDecoder

    pipeline: Pose2dPipeline = config.setup_pipeline()
    decoder = VideoDecoder(config.video_path, dimension_order="NHWC", device="cuda")
    num_frames_raw = decoder.metadata.num_frames
    fps_raw = decoder.metadata.average_fps
    if num_frames_raw is None or fps_raw is None or float(fps_raw) <= 0.0:
        raise ValueError(f"Could not read valid video metadata from {config.video_path}.")
    num_frames: int = int(num_frames_raw) if config.max_frames is None else min(int(num_frames_raw), config.max_frames)
    fps: float = float(fps_raw)
    video_timestamps_ns: Int[ndarray, "num_video_frames"] = log_video(config.video_path, Path("video"), timeline="video_time")
    log_skeleton_annotation_context(pipeline.pose.skeleton)
    total_people: int = 0
    live_slots: set[int] = set()
    for start in tqdm(range(0, num_frames, config.frame_batch_size), desc="posekit video pose"):
        stop: int = min(start + config.frame_batch_size, num_frames)
        frames_rgb: UInt8[Tensor, "chunk h w 3"] = decoder.get_frames_in_range(start, stop).data.contiguous()
        detections, keypoints = pipeline(frames_rgb)
        total_people += detections.num_detections
        chunk_indices: Int[ndarray, "chunk"] = np.arange(start, stop, dtype=np.int64)
        timestamps_ns: Int[ndarray, "chunk"] = (
            video_timestamps_ns[chunk_indices]
            if int(video_timestamps_ns.shape[0]) >= stop
            else (chunk_indices / fps * 1e9).astype(np.int64)
        )
        log_frame_predictions(
            frame_indices=chunk_indices,
            timestamps_ns=timestamps_ns,
            detections=detections,
            keypoints=keypoints,
            keypoint_threshold=config.keypoint_threshold,
            max_people=config.max_people,
            live_slots=live_slots,
        )
    torch.cuda.synchronize()
    print(f"[posekit] processed {num_frames} frames, {total_people} person instances")
