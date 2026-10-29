"""Catalog-only GPU phone and OAK hands with served frame-selection rules."""

from collections.abc import Callable

import numpy as np
import torch
from dataforge.schema import CAPTURE_TIMELINE, video_path
from einops import rearrange
from jaxtyping import Float16, Float32, Float64, Int64, UInt8
from numpy import ndarray
from posekit.predictions import BoxDetections, Keypoints2d
from rerun.catalog import DatasetEntry
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor

from selfcap_curate.catalog_io import Segment, read_phone_videos
from selfcap_curate.records import HAND_LATENCY_MS, HandTrack, OakDetections, PhoneDetections, SelfcapLayout, Side, TimeSync, oak_quest_offset_ms


def pick_frames(quest: dict[Side, HandTrack], num_frames: int, fps: float) -> Int64[ndarray, "n"]:
    """Select two 250-frame GOPs, twelve peaks each, spacing ≥4 and margin 2 s.

    The zero-based n/fps grid deliberately matches scratch plan(), including
    NumPy tie ordering. The margin guards cut boundaries; spacing avoids repeats.
    """
    if num_frames < 1 or fps <= 0.0:
        raise ValueError("frame count and fps must be positive")
    times: Float64[ndarray, "f"] = np.arange(num_frames, dtype=np.float64) / fps
    motion: Float64[ndarray, "f"] = np.zeros(num_frames)
    for hand in quest.values():
        if len(hand.times_s) > 1:
            motion = np.maximum(motion, np.interp(times, hand.times_s, np.where(hand.valid, hand.speed, 0.0), left=0.0, right=0.0))
    motion[(times < 2.0) | (times > times[-1] - 2.0)] = 0.0
    scores: list[tuple[float, int]] = [(float(np.sort(motion[k : k + 250])[-12:].sum()), k) for k in range(0, num_frames, 250)]
    picks: list[int] = []
    for _, start in sorted(scores, reverse=True)[:2]:
        indices: Int64[ndarray, "g"] = np.arange(start, min(start + 250, num_frames))
        chosen: list[int] = []
        for index in indices[np.argsort(-motion[indices])]:
            if motion[index] <= 0.0:
                break
            if all(abs(index - other) >= 4 for other in chosen):
                chosen.append(int(index))
            if len(chosen) == 12:
                break
        picks.extend(chosen)
    return np.asarray(sorted(picks), dtype=np.int64)


@torch.inference_mode()
def detect_frames(
    frames: UInt8[Tensor, "n h w 3"],
    times_s: Float64[ndarray, "n"],
    detector: Callable[[Tensor], BoxDetections],
    pose: Callable[[Tensor, BoxDetections], Keypoints2d],
) -> PhoneDetections:
    """Batch eight frames; only the highest-scoring person feeds Sapiens2.

    Missing people stay NaN/zero. Preserve frozen float32 xy and float16 scores
    before widening into the typed stream, preserving threshold decisions.
    """
    xy: Float32[ndarray, "n 133 2"] = np.full((len(frames), 133, 2), np.nan, dtype=np.float32)
    scores: Float16[ndarray, "n 133"] = np.zeros((len(frames), 133), dtype=np.float16)
    for start in range(0, len(frames), 8):
        batch: UInt8[Tensor, "b h w 3"] = frames[start : start + 8]
        detected: BoxDetections = detector(batch)
        fi: Int64[ndarray, "d"] = detected.frame_indices.cpu().numpy()
        confidence: Float32[ndarray, "d"] = detected.scores.cpu().numpy()
        rows: list[int] = []
        selected: list[int] = []
        for frame in range(len(batch)):
            candidates: Int64[ndarray, "c"] = np.flatnonzero(fi == frame)
            if len(candidates):
                rows.append(int(candidates[np.argmax(confidence[candidates])]))
                selected.append(frame)
        if not rows:
            continue
        boxes: BoxDetections = BoxDetections(
            detected.xyxy[rows], torch.ones(len(rows), device=batch.device), torch.tensor(selected, dtype=torch.long, device=batch.device)
        )
        keypoints: Keypoints2d = pose(batch, boxes)
        target: Int64[ndarray, "p"] = start + keypoints.frame_indices.cpu().numpy()
        xy[target], scores[target] = keypoints.xy_numpy(), keypoints.scores_numpy()
    return PhoneDetections(times_s, xy.astype(np.float64), scores.astype(np.float64))


def pose_models() -> tuple[Callable[[Tensor], BoxDetections], Callable[[Tensor, BoxDetections], Keypoints2d]]:
    """Build v2's YOLOX 0.3 and Sapiens2 0.4B TensorRT batch-eight models."""
    from posekit.models.sapiens import SapiensPoseConfig
    from posekit.models.yolox import YoloxDetectorConfig
    from trtkit.backends import TensorRtBackendConfig

    backend: TensorRtBackendConfig = TensorRtBackendConfig(max_batch_size=8, opt_batch_size=8)
    return YoloxDetectorConfig(score_thr=0.3, backend=backend).setup(), SapiensPoseConfig(model_size="0.4B", backend=backend).setup()


def decode_frames(video: CatalogVideo, indices: Int64[ndarray, "n"], fps: float) -> UInt8[Tensor, "n h w 3"]:
    """CUDA-decode catalog packets without changing the authoritative sample stamps."""
    from torchcodec.decoders import VideoDecoder

    if not len(indices):
        return torch.empty((0, 0, 0, 3), dtype=torch.uint8)
    data: bytes = wrap_mp4(video.samples, video.keyframes, round(fps), codec=video.codec)
    decoder = VideoDecoder(data, device="cuda", seek_mode="approximate")
    frames = decoder.get_frames_at(indices.tolist())
    return rearrange(frames.data, "n c h w -> n h w c").contiguous()


def detect_episode(
    segment: Segment,
    quest: dict[Side, HandTrack],
    detector: Callable[[Tensor], BoxDetections],
    pose: Callable[[Tensor, BoxDetections], Keypoints2d],
) -> dict[str, PhoneDetections]:
    """Decode selected catalog packets with CUDA; sample stamps are authoritative."""
    videos: dict[str, CatalogVideo] = read_phone_videos(segment)
    if not videos:
        raise ValueError(f"{segment.id}: no iPhone videos")
    first: CatalogVideo = next(iter(videos.values()))
    fps: float = 1e9 / float(np.median(np.diff(first.t_ns)))
    indices: Int64[ndarray, "n"] = pick_frames(quest, len(first.samples), fps)
    result: dict[str, PhoneDetections] = {}
    for phone, video in videos.items():
        # Muxing rebases decode PTS only. Detection times retain presented PTS
        # from capture_time, matching torchcodec's raw-file __t after migration.
        images: UInt8[Tensor, "n h w 3"] = decode_frames(video, indices, fps)
        result[phone] = detect_frames(images, video.t_ns[indices].astype(np.float64) * 1e-9, detector, pose)
    return result


def pick_oak_frames(quest: dict[Side, HandTrack], num_frames: int, fps: float, sync: TimeSync) -> Int64[ndarray, "n"]:
    """Pick ≤32 moving, tracked frames, 0.5 s apart and 1 s inside each cut.

    Preserve scratch's zero-based selection grid; add pose offset and subtract
    48 ms hand latency before matching motion to aligned RGB. Speed caps at
    1.5 m/s guard tracking jumps; ≤0.02 m/s and cut margins carry no evidence.
    """
    qp: float | None = oak_quest_offset_ms(sync)
    if qp is None:
        return np.empty(0, dtype=np.int64)
    if num_frames < 1 or fps <= 0.0:
        raise ValueError("OAK hands require quest_pose, positive frame count and fps")
    pts: Float64[ndarray, "f"] = np.arange(num_frames, dtype=np.float64) / fps
    aligned: Float64[ndarray, "f"] = sync.oak_rgb_model.video_time_s(pts)
    motion: Float64[ndarray, "f"] = np.zeros(num_frames)
    for hand in quest.values():
        if len(hand.times_s) > 1:
            speed = np.where(hand.valid, np.minimum(hand.speed, 1.5), 0.0)
            motion = np.maximum(motion, np.interp(aligned, hand.times_s - HAND_LATENCY_MS / 1000.0 + qp / 1000.0, speed, left=0.0, right=0.0))
    motion[(pts < 1.0) | (pts > pts[-1] - 1.0)] = 0.0
    chosen: list[int] = []
    gap: int = round(0.5 * fps)
    for index in np.argsort(-motion):
        if motion[index] <= 0.02 or len(chosen) == 32:
            break
        if all(abs(int(index) - other) >= gap for other in chosen):
            chosen.append(int(index))
    return np.asarray(sorted(chosen), dtype=np.int64)


def full_frame_boxes(frames: UInt8[Tensor, "n h w 3"]) -> BoxDetections:
    """Use the entire RGB frame: head-mounted hands may have no visible body."""
    n, height, width, _ = frames.shape
    return BoxDetections(
        torch.tensor([[0.0, 0.0, float(width), float(height)]] * n, device=frames.device),
        torch.ones(n, device=frames.device),
        torch.arange(n, dtype=torch.long, device=frames.device),
    )


def detect_oak_hands_gpu(
    dataset: DatasetEntry, sync: TimeSync, quest: dict[Side, HandTrack], layout: SelfcapLayout, pose: Callable[[Tensor, BoxDetections], Keypoints2d]
) -> OakDetections:
    """Read only one catalog base, decode picked RGB frames on CUDA, batch eight."""
    segment: str = sync.episode_key.segment
    video = read_catalog_videos(dataset, segment, [video_path(layout.oak_rig, 0)], CAPTURE_TIMELINE)[0]
    fps: float = 1e9 / float(np.median(np.diff(video.t_ns)))
    indices = pick_oak_frames(quest, len(video.samples), fps, sync)
    detected = detect_frames(decode_frames(video, indices, fps), video.t_ns[indices].astype(np.float64) * 1e-9, full_frame_boxes, pose)
    return OakDetections(sync.oak_rgb_model.video_time_s(detected.times_s), detected.xy, detected.scores)
