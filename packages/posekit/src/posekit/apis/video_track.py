"""Streaming video demo composing every Phase 2 role: detect -> track -> pose -> re-ID.

One detector pass bootstraps the tracks, then the stateful video segmenter
propagates masks frame by frame; its output (boxes + masks + track ids) feeds
the top-down pose estimator directly, and the CLIP identity encoder scores
each track's appearance against its bootstrap embedding — the mamma pipeline
shape (docs/design.md §1) expressed purely through posekit roles.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from torch import Tensor
from tqdm.auto import tqdm

from posekit.models import AnnotatedDetectorConfig, AnnotatedPose2dConfig, RtmPoseConfig, SegmentationPrompts, YoloxDetectorConfig
from posekit.models.clip_identity import ClipIdentityConfig
from posekit.models.sam2_video import Sam2VideoSegmenterConfig
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.rerun_logging import log_skeleton_annotation_context


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoTrackConfig:
    """Configuration for the streaming detect/track/pose/re-ID demo."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun logging configuration."""
    video_path: Path
    """Input video path."""
    detector: AnnotatedDetectorConfig = field(default_factory=YoloxDetectorConfig)
    """Person detector used to bootstrap tracks on the first frame."""
    pose: AnnotatedPose2dConfig = field(default_factory=RtmPoseConfig)
    """Top-down pose estimator fed by the tracker's boxes."""
    segmenter: Sam2VideoSegmenterConfig = field(default_factory=Sam2VideoSegmenterConfig)
    """Streaming video segmenter propagating instance masks."""
    identity: ClipIdentityConfig = field(default_factory=ClipIdentityConfig)
    """Appearance encoder scoring track consistency."""
    max_frames: int | None = None
    """Maximum number of frames to process; ``None`` runs the full video."""
    max_tracks: int = 4
    """Maximum number of people tracked from the bootstrap detections."""
    keypoint_threshold: float = 0.3
    """Minimum keypoint confidence for Rerun visibility."""


def _log_tracked_frame(
    *,
    frame_idx: int,
    timestamp_ns: int,
    tracks: BoxDetections,
    keypoints: Keypoints2d,
    pose_rows: Int[ndarray, "p"],
    identity_cosine: dict[int, float],
    keypoint_threshold: float,
    live_tracks: set[int],
    frame_hw: tuple[int, int],
) -> None:
    """Log one timestep's masks, boxes, keypoints, and identity scores."""
    rr.set_time("frame", sequence=frame_idx)
    rr.set_time("video_time", duration=1e-9 * float(timestamp_ns))
    assert tracks.masks is not None and tracks.track_ids is not None
    label_map: ndarray = np.zeros(frame_hw, dtype=np.uint8)
    boxes: Float32[ndarray, "n 4"] = tracks.xyxy_numpy()
    track_ids: Int[ndarray, "n"] = tracks.track_ids.cpu().numpy()
    masks: Bool[ndarray, "n h w"] = tracks.masks.cpu().numpy()
    xy: Float32[ndarray, "p k 2"] = keypoints.xy_numpy()
    scores: Float32[ndarray, "p k"] = keypoints.scores_numpy()
    row_to_pose: dict[int, int] = {int(row): pose_idx for pose_idx, row in enumerate(pose_rows)}
    seen: set[int] = set()
    for row in range(tracks.num_detections):
        track_id: int = int(track_ids[row])
        label_map[masks[row]] = track_id + 1
        has_box: bool = bool((boxes[row][2:] > boxes[row][:2]).all())
        if not has_box:
            continue
        seen.add(track_id)
        live_tracks.add(track_id)
        rr.log(f"video/track_{track_id}/bbox", rr.Boxes2D(array=boxes[row][None], array_format=rr.Box2DFormat.XYXY, class_ids=track_id + 1))
        if row in row_to_pose:
            pose_idx: int = row_to_pose[row]
            visible_xy: Float32[ndarray, "k 2"] = xy[pose_idx].copy()
            visible_xy[scores[pose_idx] < keypoint_threshold] = np.nan
            rr.log(
                f"video/track_{track_id}/keypoints",
                rr.Points2D(positions=visible_xy, keypoint_ids=np.arange(visible_xy.shape[0], dtype=np.uint16), class_ids=0),
            )
        if track_id in identity_cosine:
            rr.log(f"identity/track_{track_id}", rr.Scalars(identity_cosine[track_id]))
    for track_id in live_tracks - seen:
        rr.log(f"video/track_{track_id}/bbox", rr.Clear(recursive=False))
        rr.log(f"video/track_{track_id}/keypoints", rr.Clear(recursive=False))
    live_tracks.intersection_update(seen)
    rr.log("video/segmentation", rr.SegmentationImage(label_map))


def main(config: VideoTrackConfig) -> None:
    """Run the streaming detect/track/pose/re-ID demo and log to Rerun.

    Args:
        config: Demo configuration.

    Raises:
        ValueError: If the video metadata cannot be read or no person is found.
    """
    from torchcodec.decoders import VideoDecoder

    detector = config.detector.setup()
    pose = config.pose.setup()
    segmenter = config.segmenter.setup()
    identity = config.identity.setup()
    decoder = VideoDecoder(config.video_path, dimension_order="NHWC", device="cuda")
    num_frames_raw = decoder.metadata.num_frames
    fps_raw = decoder.metadata.average_fps
    if num_frames_raw is None or fps_raw is None or float(fps_raw) <= 0.0:
        raise ValueError(f"Could not read valid video metadata from {config.video_path}.")
    num_frames: int = int(num_frames_raw) if config.max_frames is None else min(int(num_frames_raw), config.max_frames)
    video_timestamps_ns: Int[ndarray, "num_video_frames"] = log_video(config.video_path, Path("video"), timeline="video_time")
    log_skeleton_annotation_context(pose.skeleton)
    rr.log(
        "video/segmentation",
        rr.AnnotationContext(
            [rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0))]
            + [rr.AnnotationInfo(id=tid + 1, label=f"track_{tid}") for tid in range(config.max_tracks)]
        ),
        static=True,
    )

    # Decode in chunks (amortizes NVDEC call overhead) but step the stateful
    # segmenter one frame at a time from slices of the chunk.
    decode_chunk_size: int = 32
    frames_chunk: UInt8[Tensor, "c h w 3"] = decoder.get_frames_in_range(0, min(decode_chunk_size, num_frames)).data.contiguous()
    chunk_start: int = 0
    frame_hw: tuple[int, int] = (int(frames_chunk.shape[1]), int(frames_chunk.shape[2]))
    bootstrap: BoxDetections = detector(frames_chunk[0:1])
    if bootstrap.num_detections == 0:
        raise ValueError("No person detected on the bootstrap frame.")
    order: Tensor = torch.argsort(bootstrap.scores, descending=True)[: config.max_tracks]
    seed_boxes: Float32[Tensor, "t 4"] = bootstrap.xyxy[order]
    seed_ids: Tensor = torch.arange(int(seed_boxes.shape[0]), dtype=torch.long, device=seed_boxes.device)
    prompts = SegmentationPrompts(frame_indices=torch.zeros_like(seed_ids), boxes_xyxy=seed_boxes, track_ids=seed_ids)
    reference_embeds: dict[int, Float32[Tensor, "d"]] = {}
    live_tracks: set[int] = set()

    for frame_idx in tqdm(range(num_frames), desc="posekit video track"):
        if frame_idx >= chunk_start + int(frames_chunk.shape[0]):
            chunk_start = frame_idx
            frames_chunk = decoder.get_frames_in_range(chunk_start, min(chunk_start + decode_chunk_size, num_frames)).data.contiguous()
        frame: UInt8[Tensor, "1 h w 3"] = frames_chunk[frame_idx - chunk_start : frame_idx - chunk_start + 1]
        tracks: BoxDetections = segmenter.step(frame, prompts=prompts if frame_idx == 0 else None)
        valid: Tensor = (tracks.xyxy[:, 2:] > tracks.xyxy[:, :2]).all(dim=1)
        pose_rows_t: Tensor = torch.where(valid)[0]
        pose_dets = BoxDetections(
            xyxy=tracks.xyxy[pose_rows_t], scores=tracks.scores[pose_rows_t], frame_indices=tracks.frame_indices[pose_rows_t]
        )
        keypoints: Keypoints2d = pose(frame, pose_dets)
        embeds: Float32[Tensor, "p d"] = identity(frame, pose_dets)
        identity_cosine: dict[int, float] = {}
        assert tracks.track_ids is not None
        for pose_idx, row in enumerate(pose_rows_t.tolist()):
            track_id: int = int(tracks.track_ids[row])
            if track_id not in reference_embeds:
                reference_embeds[track_id] = embeds[pose_idx]
            identity_cosine[track_id] = float(F.cosine_similarity(embeds[pose_idx], reference_embeds[track_id], dim=0))
        timestamp_ns: int = (
            int(video_timestamps_ns[frame_idx]) if int(video_timestamps_ns.shape[0]) > frame_idx else int(frame_idx / float(fps_raw) * 1e9)
        )
        _log_tracked_frame(
            frame_idx=frame_idx,
            timestamp_ns=timestamp_ns,
            tracks=tracks,
            keypoints=keypoints,
            pose_rows=pose_rows_t.cpu().numpy(),
            identity_cosine=identity_cosine,
            keypoint_threshold=config.keypoint_threshold,
            live_tracks=live_tracks,
            frame_hw=frame_hw,
        )
    torch.cuda.synchronize()
    print(f"[posekit] tracked {num_frames} frames, {len(reference_embeds)} identities")
