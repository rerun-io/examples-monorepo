"""Sample visible slow Quest fingers, then independently detect and reconstruct EgoHand.

Use seeded sampling of 64 frames so calibration samples are reproducible.
"""

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from dataforge.datasets.selfcap_raw import Intrinsic
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from posekit.cameras import CalibratedCamera
from posekit.models.egohand import (
    EgoHandDetector,
    EgoHandPredictor,
    EgoHandReconstruction,
    selfcap_egohand_configs,
)
from posekit.models.egohand_fast import EgoHandConfigs, FastEgoHandPredictor, fast_egohand_configs
from posekit.predictions import SOMA_HAND21, BoxDetections, HandDetections, HandSide
from scipy.spatial.transform import Rotation
from simplecv.catalog_video import CatalogVideo
from torch import Tensor

from selfcap_curate.factory import intrinsic_matrix
from selfcap_curate.hands_gpu import decode_frames
from selfcap_curate.keypoints import EgoCamera, project_ego
from selfcap_curate.oak_mount import OakLens
from selfcap_curate.quest_fit import hand_neighbors
from selfcap_curate.records import HAND_BASE, HAND_LATENCY_MS, MAPPED, CatalogHands, HandTrack, OakDetections, SessionEpisode, oak_quest_offset_ms

EGOHAND_FRAME_COUNT: int = 64
"""Frames sampled per episode for EgoHand calibration."""


@dataclass(frozen=True, slots=True)
class OakFrameSelection:
    indices: Int64[ndarray, "n"]
    """Sorted source-video frame indices."""
    hands: Bool[ndarray, "n 2"]
    """Eligible anatomical left/right hands on the selected frames."""
    eligible_count: int
    """Frames with at least one eligible hand before sampling."""


def select_egohand_frames(
    times: Float64[ndarray, "f"], pixels: Float32[ndarray, "f 133 2"] | None, tracked: Bool[ndarray, "f 2"],
    size: tuple[int, int], *, seed: int = 20261030,
) -> OakFrameSelection:
    """Uniform sampling without replacement after visibility and 300 px/s Quest-speed gates."""
    if len(times) < 2:
        return OakFrameSelection(np.empty(0, np.int64), np.empty((0, 2), bool), 0)
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("video times must increase")
    eligible: Bool[ndarray, "f 2"] = tracked.copy()
    # A free mount has no projection yet; sample tracked hands and let image detection find them.
    if pixels is not None:
        speed: Float32[ndarray, "f 133"] = np.linalg.norm(np.gradient(pixels, times, axis=0), axis=-1)
        width, height = size
        for side, base in enumerate(HAND_BASE.values()):
            indices: Int64[ndarray, "j"] = np.asarray(MAPPED) + base
            points: Float32[ndarray, "f j 2"] = pixels[:, indices]
            inside: Bool[ndarray, "f"] = np.isfinite(points).all((1, 2)) & (points[:, :, 0] >= 0.0).all(1) & (points[:, :, 0] < width).all(1)
            inside &= (points[:, :, 1] >= 0.0).all(1) & (points[:, :, 1] < height).all(1)
            eligible[:, side] &= inside & (speed[:, indices] <= 300.0).all(1)
    candidates: Int64[ndarray, "k"] = np.flatnonzero(eligible.any(1))
    chosen: Int64[ndarray, "n"] = np.sort(np.random.default_rng(seed).choice(candidates, min(EGOHAND_FRAME_COUNT, len(candidates)), replace=False))
    return OakFrameSelection(chosen, eligible[chosen], len(candidates))


def infer_egohand_frames(
    frames: UInt8[Tensor, "n h w 3"], times: Float64[ndarray, "n"], hands: Bool[ndarray, "n 2"], camera: CalibratedCamera,
    detector: Callable[[Tensor], HandDetections], predictor: Callable[[Tensor, HandDetections, CalibratedCamera], EgoHandReconstruction],
) -> OakDetections:
    """Detect each frame from scratch; copy its box score to its reconstructed COCO-21 hand."""
    xy: Float64[ndarray, "n 133 2"] = np.full((len(frames), 133, 2), np.nan)
    scores: Float64[ndarray, "n 133"] = np.zeros((len(frames), 133))
    for start in range(0, len(frames), 8):
        batch: UInt8[Tensor, "b h w 3"] = frames[start : start + 8]
        boxes: list[Tensor] = []
        confidence: list[Tensor] = []
        frame_ids: list[Tensor] = []
        sides: list[HandSide] = []
        for frame in range(len(batch)):
            detected: HandDetections = detector(batch[frame : frame + 1])
            keep: list[int] = [i for i, side in enumerate(detected.sides) if hands[start + frame, 0 if side == "left" else 1]]
            if keep:
                boxes.append(detected.boxes.xyxy[keep])
                confidence.append(detected.boxes.scores[keep])
                frame_ids.append(torch.full((len(keep),), frame, dtype=torch.int64, device=batch.device))
                sides.extend(detected.sides[i] for i in keep)
        if not boxes:
            continue
        detections: HandDetections = HandDetections(BoxDetections(torch.cat(boxes), torch.cat(confidence), torch.cat(frame_ids)), tuple(sides))
        result: EgoHandReconstruction = predictor(batch, detections, camera)
        hand_xy: Float32[ndarray, "m 21 2"] = result.keypoints.xy[:, list(SOMA_HAND21)].cpu().numpy()
        hand_frames: list[int] = result.keypoints.frame_indices.tolist()
        hand_scores: list[float] = result.keypoints.scores[:, 0].tolist()
        for row, side in enumerate(result.sides):
            frame = start + hand_frames[row]
            base = HAND_BASE[side]
            xy[frame, base : base + 21] = hand_xy[row]
            scores[frame, base : base + 21] = hand_scores[row]
    return OakDetections(times, xy, scores)


def egohand_configs(assets: Path | None, *, fast: bool = False) -> EgoHandConfigs:
    """The reference or fast EgoHand stack; ``assets`` None takes the pinned downloads and the trtkit cache."""
    return fast_egohand_configs(assets) if fast else selfcap_egohand_configs(assets)


@cache
def egohand_models(configs: EgoHandConfigs) -> tuple[EgoHandDetector, EgoHandPredictor | FastEgoHandPredictor]:
    """Retain engines for an explicitly long-lived batch worker."""
    return configs[0].setup(), configs[1].setup()


@torch.inference_mode()
def detect_oak_egohand(
    episode: SessionEpisode,
    video: CatalogVideo,
    intrinsic: Intrinsic,
    rotation: Float64[ndarray, "3 3"] | None,
    models: EgoHandConfigs,
    *,
    camera: EgoCamera | None = None,
    retain_models: bool = False,
) -> OakDetections:
    """Run only 64 seeded samples; release this episode's GPU models before returning."""
    qp: float | None = oak_quest_offset_ms(episode.sync)
    if qp is None:
        raise ValueError("EgoHand sampling requires the Quest/OAK clock")
    times: Float64[ndarray, "f"] = episode.sync.oak_rgb_model.video_time_s(video.t_ns.astype(np.float64) * 1e-9)
    head: CatalogHands = episode.hands
    if camera is None and rotation is not None:
        poses: Float64[ndarray, "p 4 4"] = np.tile(np.eye(4), (len(head.head_times_s), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(head.head_quaternions).as_matrix() @ rotation
        poses[:, :3, 3] = head.head_positions
        camera = EgoCamera(
            episode.layout.oak_rig,
            0,
            intrinsic_matrix(intrinsic),
            np.asarray(intrinsic.distortion.coefficients, dtype=np.float64),
            (intrinsic.capture_resolution.width, intrinsic.capture_resolution.height),
            np.eye(4),
            head.head_times_s + qp / 1000.0,
            poses,
        )
    xyz: Float32[ndarray, "f 133 3"] = np.full((len(times), 133, 3), np.nan, np.float32)
    tracked: Bool[ndarray, "f 2"] = np.zeros((len(times), 2), bool)
    for side_index, (side, base) in enumerate(HAND_BASE.items()):
        hand: HandTrack = episode.quest[side]
        indices, alpha, valid = hand_neighbors(hand, times - qp / 1000.0 + HAND_LATENCY_MS / 1000.0)
        xyz[:, base : base + 21] = (1.0 - alpha[:, None, None]) * hand.joints[indices - 1] + alpha[:, None, None] * hand.joints[indices]
        tracked[:, side_index] = valid
    size: tuple[int, int] = camera.size if camera is not None else (intrinsic.capture_resolution.width, intrinsic.capture_resolution.height)
    pixels: Float32[ndarray, "f 133 2"] | None = project_ego(camera, times, xyz) if camera is not None and rotation is not None else None
    selection: OakFrameSelection = select_egohand_frames(times, pixels, tracked, size)
    lens: CalibratedCamera = (
        CalibratedCamera(camera.K, camera.distortion, *size)
        if camera is not None
        else CalibratedCamera(intrinsic_matrix(intrinsic), np.asarray(intrinsic.distortion.coefficients, dtype=np.float64), *size)
    )
    print(f"EGOHAND_SAMPLE {episode.sync.key}: {len(selection.indices)}/{selection.eligible_count} eligible frames", flush=True)
    if not len(selection.indices):
        return OakDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133)))
    frames: UInt8[Tensor, "n h w 3"] = decode_frames(video, selection.indices, 1e9 / float(np.median(np.diff(video.t_ns))))
    started: float = time.perf_counter()
    detector: EgoHandDetector | None = None
    predictor: EgoHandPredictor | FastEgoHandPredictor | None = None
    try:
        detector, predictor = egohand_models(models)
        result: OakDetections = infer_egohand_frames(frames, times[selection.indices], selection.hands, lens, detector, predictor)
        torch.cuda.synchronize()
        print(f"EGOHAND_CALIB_SECONDS {time.perf_counter() - started:.3f}", flush=True)
        return result
    finally:
        del frames, detector, predictor
        if not retain_models:
            egohand_models.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"EGOHAND_RELEASED_MIB {torch.cuda.memory_allocated() / 2**20:.1f}", flush=True)


@torch.inference_mode()
def detect_oak_views(
    episode: SessionEpisode, videos: dict[str, CatalogVideo], lenses: dict[str, OakLens], models: EgoHandConfigs, *, retain_models: bool = False,
) -> dict[str, OakDetections]:
    """EgoHand on the same 64 instants in every OAK camera, sampled on OAK left; videos are read on ``video_time``.

    Left and right share a hardware clock; each camera's row ``f`` is its frame nearest left frame ``f``, so the
    stereo pair triangulates row by row.
    """
    qp: float | None = oak_quest_offset_ms(episode.sync)
    if qp is None:
        raise ValueError("EgoHand sampling requires the Quest/OAK clock")
    left_times: Float64[ndarray, "f"] = videos["left"].t_ns.astype(np.float64) * 1e-9
    tracked: Bool[ndarray, "f 2"] = np.stack(
        [hand_neighbors(episode.quest[side], left_times - qp / 1000.0 + HAND_LATENCY_MS / 1000.0)[2] for side in HAND_BASE], axis=1
    )
    selection: OakFrameSelection = select_egohand_frames(left_times, None, tracked, lenses["left"].size)
    print(f"EGOHAND_SAMPLE {episode.sync.key}: {len(selection.indices)}/{selection.eligible_count} eligible frames x {len(videos)} cameras", flush=True)
    if not len(selection.indices):
        return {camera: OakDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133))) for camera in videos}
    started: float = time.perf_counter()
    detector: EgoHandDetector | None = None
    predictor: EgoHandPredictor | FastEgoHandPredictor | None = None
    try:
        detector, predictor = egohand_models(models)
        result: dict[str, OakDetections] = {}
        for camera, video in videos.items():
            times: Float64[ndarray, "f"] = video.t_ns.astype(np.float64) * 1e-9
            indices: Int64[ndarray, "n"] = np.clip(np.searchsorted(times, left_times[selection.indices]), 0, len(times) - 1)
            frames: UInt8[Tensor, "n h w 3"] = decode_frames(video, indices, 1e9 / float(np.median(np.diff(video.t_ns))))
            lens = CalibratedCamera(lenses[camera].intrinsic, np.pad(lenses[camera].distortion, (0, 14 - len(lenses[camera].distortion))), *lenses[camera].size)
            result[camera] = infer_egohand_frames(frames, times[indices], selection.hands, lens, detector, predictor)
            del frames
        torch.cuda.synchronize()
        print(f"EGOHAND_CALIB_SECONDS {time.perf_counter() - started:.3f}", flush=True)
        return result
    finally:
        del detector, predictor
        if not retain_models:
            egohand_models.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"EGOHAND_RELEASED_MIB {torch.cuda.memory_allocated() / 2**20:.1f}", flush=True)
