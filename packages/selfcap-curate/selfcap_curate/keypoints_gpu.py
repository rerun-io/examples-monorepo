"""Catalog-only YOLOX + RTMW-x COCO-133 producer for CPU triangulation."""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import onnx
import torch
from jaxtyping import Bool, Float16, Float32, Float64, Int64, UInt8
from numpy import ndarray
from onnx import numpy_helper
from posekit.models.rtmpose import RtmPose2d, RtmPoseConfig
from posekit.models.yolox import YoloxDetector, YoloxDetectorConfig
from posekit.predictions import BoxDetections, Keypoints2d
from simplecv.catalog_video import CatalogVideo
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor
from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder
from trtkit.backends import TensorRtBackendConfig

from selfcap_curate.catalog_io import Segment, read_phone_videos
from selfcap_curate.keypoints import Keypoints2D, ViewDetections
from selfcap_curate.records import TimeSync

TRACK_MIN_RATE_HZ: float = 10.0
"""Plans at or above this rate take each frame's person boxes from the previous frame's pose (detection-by-tracking, as in
handtrack and mv-api): on cc808233 ep001 YOLOX ran on 263 of 6,476 phone frames, 3D within 7 mm of detect-every-frame."""
TRACK_MIN_KEYPOINTS: int = 8
"""A track survives while its pose has at least this many keypoints above 0.3."""
TRACK_CHUNKS: int = 8
"""Independent time chunks tracked in lockstep per phone, so one RTMW call carries ~phones x chunks crops, not ~phones."""


def pose_boxes(
    xy: Float32[Tensor, "n 133 2"], scores: Float32[Tensor, "n 133"], width: int, height: int
) -> tuple[Float32[Tensor, "n 4"], Float32[Tensor, "n"], Bool[Tensor, "n"]]:
    """rtmlib pose_to_bbox on confident keypoints, x1.25 about the centre and clipped; alive with TRACK_MIN_KEYPOINTS of them."""
    confident: Bool[Tensor, "n 133"] = scores > 0.3
    lo = torch.where(confident[..., None], xy, torch.inf).amin(1)
    hi = torch.where(confident[..., None], xy, -torch.inf).amax(1)
    centre, half = (lo + hi) / 2, (hi - lo) / 2 * 1.25
    upper = torch.tensor([width - 1, height - 1, width - 1, height - 1], dtype=xy.dtype, device=xy.device)
    boxes = torch.minimum(torch.cat([centre - half, centre + half], 1).clamp(min=0), upper)
    box_scores = torch.where(confident, scores, 0.0).sum(1) / confident.sum(1).clamp(min=1)
    return boxes, box_scores, confident.sum(1) >= TRACK_MIN_KEYPOINTS


def top_slots(frame_indices: Int64[Tensor, "d"], scores: Float32[Tensor, "d"]) -> Int64[Tensor, "d"]:
    """Rank of each detection within its frame by descending score, earlier detections first on ties."""
    order = torch.arange(len(scores), device=scores.device)
    ahead = (scores[None, :] > scores[:, None]) | ((scores[None, :] == scores[:, None]) & (order[None, :] < order[:, None]))
    return ((frame_indices[:, None] == frame_indices[None, :]) & ahead).sum(1)


def nearest_indices(pts: Float64[ndarray, "f"], want: Float64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    """Nearest source sample; ties go left, including irregular presentation PTS."""
    if not len(pts) or not np.isfinite(want).all() or not np.all(np.diff(pts) > 0.0):
        raise ValueError("presentation times must be nonempty and strictly increasing")
    right: Int64[ndarray, "n"] = np.clip(np.searchsorted(pts, want), 0, len(pts) - 1)
    left: Int64[ndarray, "n"] = np.maximum(right - 1, 0)
    return np.where(abs(pts[left] - want) <= abs(pts[right] - want), left, right)


@dataclass(frozen=True, slots=True)
class FramePlan:
    times: Float64[ndarray, "n"]  # Aligned sampling grid in seconds.
    indices: dict[str, Int64[ndarray, "n"]]  # Nearest original frame index per synced phone.
    videos: dict[str, CatalogVideo]  # Packets corresponding to the selected original frame indices.


def plan_frames(aligned: dict[str, CatalogVideo], rate: float) -> FramePlan:
    """Anchor a half-open common grid at cut zero; use original sample clocks."""
    if len(aligned) < 2 or not np.isfinite(rate) or rate <= 0.0 or any(len(v.samples) < 2 for v in aligned.values()):
        raise ValueError("need two synced videos, two frames and a positive finite rate")
    lo: float = max(0.0, max(float(v.t_ns[0]) * 1e-9 for v in aligned.values()))
    hi: float = min(float(v.t_ns[-1] + np.median(np.diff(v.t_ns))) * 1e-9 for v in aligned.values())
    times: Float64[ndarray, "n"] = np.arange(np.ceil(lo * rate), np.ceil(hi * rate), dtype=np.float64) / rate
    times = times[times < hi]
    if not len(times):
        raise ValueError("empty synced overlap")
    return FramePlan(times, {p: nearest_indices(v.t_ns.astype(np.float64) * 1e-9, times) for p, v in aligned.items()}, aligned)


def decode_wanted(decoder: VideoDecoder, indices: Int64[ndarray, "n"], batch: int, chunk: int = 64) -> Iterator[FrameBatch]:
    """Decode in order from the first to the last wanted frame and yield the wanted frames, repeats included, ``batch`` at a time.

    One pass costs no more than picking: AV1 references keep about 85% of a phone's frames needed even for a
    2 Hz plan, and a plan at the native rate wants every frame.
    """
    if np.any(np.diff(indices) < 0):
        raise ValueError("wanted frames must be sorted")
    if len(indices) and (indices[0] < 0 or indices[-1] >= len(decoder)):
        raise ValueError("wanted frames must lie inside the video")
    stop: int = int(indices[-1]) + 1 if len(indices) else 0
    held: FrameBatch | None = None
    for start in range(int(indices[0]) if len(indices) else 0, stop, chunk):
        frames: FrameBatch = decoder.get_frames_in_range(start, min(start + chunk, stop))
        picked: FrameBatch = frames[torch.from_numpy(indices[(indices >= start) & (indices < start + chunk)] - start)]
        held = (
            picked
            if held is None
            else FrameBatch(
                torch.cat([held.data, picked.data]),
                torch.cat([held.pts_seconds, picked.pts_seconds]),
                torch.cat([held.duration_seconds, picked.duration_seconds]),
            )
        )
        while len(held) >= batch:
            yield held[:batch]
            held = held[batch:]
    if held is not None and len(held):
        yield held


def plan_episode(segment: Segment, rate: float, sync: TimeSync) -> FramePlan:
    """Read named segment streams and prepare original-frame selections."""
    videos: dict[str, CatalogVideo] = read_phone_videos(segment, timeline="video_time")
    offsets: dict[str, float | None] = sync.offsets_ms or {}
    videos = {p: v for p, v in videos.items() if offsets.get("exo/" + p) is not None}
    return plan_frames(videos, rate)


@torch.inference_mode()
def detect_keypoints(
    segment: Segment,
    detector: Callable[[Tensor], BoxDetections],
    pose: Callable[[Tensor, BoxDetections], Keypoints2d],
    rate: float,
    *,
    sync: TimeSync,
) -> Keypoints2D:
    """Infer three people per phone on the aligned grid; dense plans track, sparse plans detect every sample in batches of eight."""
    plan: FramePlan = plan_episode(segment, rate, sync)
    if rate >= TRACK_MIN_RATE_HZ:
        return Keypoints2D(plan.times, detect_tracked(plan, detector, pose, refresh=round(rate)), rate)
    views: dict[str, ViewDetections] = {}
    for phone, video in plan.videos.items():
        fps: int = round(1e9 / float(np.median(np.diff(video.t_ns))))
        indices: Int64[ndarray, "n"] = plan.indices[phone]
        wanted: Float64[ndarray, "n"] = indices.astype(np.float64) / fps
        decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, fps, video.codec), device="cuda", dimension_order="NHWC", seek_mode="exact")
        xy: Float32[ndarray, "n 3 133 2"] = np.full((len(plan.times), 3, 133, 2), np.nan, np.float32)
        scores: Float16[ndarray, "n 3 133"] = np.zeros((len(plan.times), 3, 133), np.float16)
        boxes: Float32[ndarray, "n 3 4"] = np.full((len(plan.times), 3, 4), np.nan, np.float32)
        confidence: Float32[ndarray, "n 3"] = np.zeros((len(plan.times), 3), np.float32)
        start: int = 0
        for frames in decode_wanted(decoder, indices, batch=8):
            if not np.allclose(frames.pts_seconds.cpu().numpy(), wanted[start : start + len(frames)], rtol=0.0, atol=1e-9):
                raise ValueError("decoder changed selected source timestamps")
            detected: BoxDetections = detector(frames.data)
            fi: Int64[ndarray, "d"] = detected.frame_indices.cpu().numpy()
            sc: Float32[ndarray, "d"] = detected.scores.cpu().numpy()
            rows: list[int] = []
            for frame in range(len(frames)):
                candidates: Int64[ndarray, "d"] = np.flatnonzero(fi == frame)
                rows.extend(candidates[np.argsort(-sc[candidates], kind="stable")[:3]].tolist())
            if len(rows):
                selected: BoxDetections = BoxDetections(detected.xyxy[rows], detected.scores[rows], detected.frame_indices[rows])
                keypoints: Keypoints2d = pose(frames.data, selected)
                kfi: Int64[ndarray, "d"] = keypoints.frame_indices.cpu().numpy()
                if not np.array_equal(kfi, fi[rows]):
                    raise ValueError("pose changed detection instance order")
                slots = np.concatenate([np.arange(count) for count in np.bincount(kfi, minlength=len(frames))])
                xy[start + kfi, slots], scores[start + kfi, slots] = keypoints.xy.cpu().numpy(), keypoints.scores.cpu().numpy()
                boxes[start + kfi, slots], confidence[start + kfi, slots] = selected.xyxy.cpu().numpy(), selected.scores.cpu().numpy()
            start += len(frames)
        if start != len(plan.times):
            raise ValueError(f"{phone}: decoded {start} of {len(plan.times)} planned frames")
        del decoder
        views[phone] = ViewDetections(boxes.astype(np.float64), confidence.astype(np.float64), xy.astype(np.float64), scores.astype(np.float64))
    return Keypoints2D(plan.times, views, rate)


def detect_tracked(
    plan: FramePlan, detector: Callable[[Tensor], BoxDetections], pose: Callable[[Tensor, BoxDetections], Keypoints2d], *, refresh: int
) -> dict[str, ViewDetections]:
    """Detection-by-tracking in TRACK_CHUNKS time chunks per phone, all chunks of all phones advancing one frame per step.

    Within a chunk, boxes come from the previous pose and YOLOX runs only on a lost track or every ``refresh`` frames;
    each chunk starts with a detection. Track state and results stay on the GPU until one copy per phone at the end.
    """
    phones: list[str] = list(plan.videos)
    n, device = len(plan.times), torch.device("cuda")
    bounds: Int64[ndarray, "k"] = np.unique(np.linspace(0, n, TRACK_CHUNKS + 1).round().astype(np.int64))
    owner_phone, first_tick, length, sources, wanted = [], [], [], [], []
    for p_index, (phone, video) in enumerate(plan.videos.items()):
        fps: int = round(1e9 / float(np.median(np.diff(video.t_ns))))
        data = wrap_mp4(video.samples, video.keyframes, fps, video.codec)
        for begin, stop in zip(bounds[:-1], bounds[1:], strict=True):
            decoder = VideoDecoder(data, device="cuda", dimension_order="NHWC", seek_mode="exact")
            owner_phone.append(p_index)
            first_tick.append(int(begin))
            length.append(int(stop - begin))
            sources.append(decode_wanted(decoder, plan.indices[phone][begin:stop], batch=8, chunk=8))  # ~100 MB of frames per stream
            wanted.append(plan.indices[phone][begin:stop].astype(np.float64) / fps)
    streams, phone_of = len(sources), torch.as_tensor(owner_phone, device=device)
    start_of = torch.as_tensor(first_tick, device=device)
    xy = torch.full((len(phones), n, 3, 133, 2), float("nan"), device=device)
    scores = torch.zeros((len(phones), n, 3, 133), device=device)
    boxes = torch.full((len(phones), n, 3, 4), float("nan"), device=device)
    box_scores = torch.zeros((len(phones), n, 3), device=device)
    slot_boxes, slot_scores = torch.zeros((streams, 3, 4), device=device), torch.zeros((streams, 3), device=device)
    alive = torch.zeros((streams, 3), dtype=torch.bool, device=device)
    lost = torch.zeros(streams, dtype=torch.bool, device=device)
    last_detect: Int64[ndarray, "s"] = np.full(streams, -(2**40))
    held: list[FrameBatch | None] = [None] * streams
    for step in range(max(length)):
        active: Int64[ndarray, "a"] = np.flatnonzero(step < np.asarray(length))
        for stream in active:
            if step % 8 == 0:
                held[stream] = next(sources[stream])
                if not np.allclose(held[stream].pts_seconds.cpu().numpy(), wanted[stream][step : step + len(held[stream])], rtol=0.0, atol=1e-9):
                    raise ValueError(f"{phones[owner_phone[stream]]}: decoder changed selected source timestamps")
        frames: UInt8[Tensor, "a h w 3"] = torch.stack([held[stream].data[step % 8] for stream in active])
        height, width = int(frames.shape[1]), int(frames.shape[2])
        act = torch.as_tensor(active, device=device)
        need: Int64[ndarray, "m"] = np.flatnonzero((~alive[act].any(1) | lost[act]).cpu().numpy() | (step - last_detect[active] >= refresh))
        if len(need):
            need_t = torch.as_tensor(need, device=device)
            found: BoxDetections = detector(frames[need_t])
            rank = top_slots(found.frame_indices, found.scores)
            keep = rank < 3
            owner, slot = act[need_t[found.frame_indices[keep]]], rank[keep]
            alive[act[need_t]], lost[act[need_t]] = False, False
            slot_boxes[owner, slot], slot_scores[owner, slot], alive[owner, slot] = found.xyxy[keep], found.scores[keep], True
            last_detect[active[need]] = step
        live = alive[act].nonzero()
        if len(live):
            row, slot = live[:, 0], live[:, 1]
            owner = act[row]
            keypoints: Keypoints2d = pose(frames, BoxDetections(slot_boxes[owner, slot], slot_scores[owner, slot], row))
            phone_index, tick = phone_of[owner], start_of[owner] + step
            xy[phone_index, tick, slot], scores[phone_index, tick, slot] = keypoints.xy.float(), keypoints.scores.float()
            boxes[phone_index, tick, slot], box_scores[phone_index, tick, slot] = slot_boxes[owner, slot], slot_scores[owner, slot]
            next_boxes, next_scores, ok = pose_boxes(keypoints.xy.float(), keypoints.scores.float(), width, height)
            slot_boxes[owner, slot], slot_scores[owner, slot], alive[owner, slot] = next_boxes, next_scores, ok
            lost = torch.zeros(streams, dtype=torch.int64, device=device).index_add_(0, owner, (~ok).long()) > 0
    return {
        phone: ViewDetections(
            boxes[i].double().cpu().numpy(), box_scores[i].double().cpu().numpy(), xy[i].double().cpu().numpy(), scores[i].double().cpu().numpy()
        )
        for i, phone in enumerate(phones)
    }


def dynamic_yolox(source: Path) -> Path:
    """Make only the five reviewed YOLOX-m batch reshape constants dynamic."""
    directory = Path(os.environ.get("POSEKIT_ONNX_CACHE", "~/.cache/selfcap-curate/onnx")).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    dest: Path = directory / (source.stem + "_dynamic_batch_v1.onnx")
    if dest.exists():
        return dest
    model = onnx.load(source)
    constants = {node.output[0]: next(a.t for a in node.attribute if a.name == "value") for node in model.graph.node if node.op_type == "Constant"}
    constants.update({value.name: value for value in model.graph.initializer})
    changed: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        proto = constants[node.input[1]]
        shape = numpy_helper.to_array(proto).copy()
        if shape.ndim != 1 or shape[0] not in (0, 1):
            raise ValueError(f"Unreviewed YOLOX reshape: {node.name}")
        shape[0] = 0
        proto.CopyFrom(numpy_helper.from_array(shape.astype(np.int64), name=proto.name))
        changed.add(node.input[1])
    if len(changed) != 5:
        raise ValueError("Expected five YOLOX batch constants")
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].ClearField("dim_value")
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dest))
    return dest


@cache
def keypoint_models() -> tuple[YoloxDetector, RtmPose2d]:
    """Use installed posekit/trtkit cache settings, YOLOX .3 and RTMW-x; built once per process."""
    detector = YoloxDetectorConfig(
        backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=8), score_thr=0.3, onnx_transform=dynamic_yolox
    ).setup()
    # Tracking sends up to phones x TRACK_CHUNKS crops per call; the 1-8-32 profile takes them in one engine call.
    pose = RtmPoseConfig(variant="rtmw-x-coco133", backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=8)).setup()
    return detector, pose
