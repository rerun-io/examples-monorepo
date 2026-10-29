"""Catalog-only YOLOX + RTMW-x COCO-133 producer for CPU triangulation."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import torch
from jaxtyping import Float16, Float32, Float64, Int64
from numpy import ndarray
from onnx import numpy_helper
from posekit.models.rtmpose import RtmPose2d, RtmPoseConfig
from posekit.models.yolox import YoloxDetector, YoloxDetectorConfig
from posekit.predictions import BoxDetections, Keypoints2d
from simplecv.catalog_video import CatalogVideo
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from trtkit.backends import TensorRtBackendConfig

from selfcap_curate.catalog_io import Segment, read_phone_videos
from selfcap_curate.keypoints import Keypoints2D, ViewDetections
from selfcap_curate.keypoints_av1 import filter_video
from selfcap_curate.records import TimeSync


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
    """Infer three people in CUDA batches of eight, widening float16 confidence on the aligned grid."""
    plan: FramePlan = plan_episode(segment, rate, sync)
    views: dict[str, ViewDetections] = {}
    with TemporaryDirectory(prefix="selfcap-kp2d-") as work:
        for phone, video in plan.videos.items():
            fps: int = round(1e9 / float(np.median(np.diff(video.t_ns))))
            indices: Int64[ndarray, "n"] = plan.indices[phone]
            source, filtered = Path(work) / "source.mp4", Path(work) / "filtered.mp4"
            source.write_bytes(wrap_mp4(video.samples, video.keyframes, fps, video.codec))
            wanted: Float64[ndarray, "n"] = indices.astype(np.float64) / fps
            path: Path = source
            if video.codec == "av1":
                indices = filter_video(source, filtered, wanted)
                path = filtered
            decoder = VideoDecoder(str(path), device="cuda", dimension_order="NHWC", seek_mode="exact")
            xy: Float32[ndarray, "n 3 133 2"] = np.full((len(plan.times), 3, 133, 2), np.nan, np.float32)
            scores: Float16[ndarray, "n 3 133"] = np.zeros((len(plan.times), 3, 133), np.float16)
            boxes: Float32[ndarray, "n 3 4"] = np.full((len(plan.times), 3, 4), np.nan, np.float32)
            confidence: Float32[ndarray, "n 3"] = np.zeros((len(plan.times), 3), np.float32)
            for start in range(0, len(plan.times), 8):
                frames = decoder.get_frames_at(indices[start : start + 8].tolist())
                if not np.allclose(frames.pts_seconds.cpu().numpy(), wanted[start : start + 8], rtol=0.0, atol=1e-9):
                    raise ValueError("decoder changed selected source timestamps")
                detected: BoxDetections = detector(frames.data)
                fi: Int64[ndarray, "d"] = detected.frame_indices.cpu().numpy()
                sc: Float32[ndarray, "d"] = detected.scores.cpu().numpy()
                rows: list[int] = []
                for frame in range(len(frames.data)):
                    candidates: Int64[ndarray, "d"] = np.flatnonzero(fi == frame)
                    rows.extend(candidates[np.argsort(-sc[candidates], kind="stable")[:3]].tolist())
                if not len(rows):
                    continue
                selected: BoxDetections = BoxDetections(detected.xyxy[rows], detected.scores[rows], detected.frame_indices[rows])
                keypoints: Keypoints2d = pose(frames.data, selected)
                kfi: Int64[ndarray, "d"] = keypoints.frame_indices.cpu().numpy()
                if not np.array_equal(kfi, fi[rows]):
                    raise ValueError("pose changed detection instance order")
                slots = np.concatenate([np.arange(count) for count in np.bincount(kfi, minlength=len(frames.data))])
                xy[start + kfi, slots], scores[start + kfi, slots] = keypoints.xy.cpu().numpy(), keypoints.scores.cpu().numpy()
                boxes[start + kfi, slots], confidence[start + kfi, slots] = selected.xyxy.cpu().numpy(), selected.scores.cpu().numpy()
            del decoder
            views[phone] = ViewDetections(boxes.astype(np.float64), confidence.astype(np.float64), xy.astype(np.float64), scores.astype(np.float64))
    return Keypoints2D(plan.times, views, rate)


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


def keypoint_models() -> tuple[YoloxDetector, RtmPose2d]:
    """Use installed posekit/trtkit cache settings, YOLOX .3 and RTMW-x."""
    backend = TensorRtBackendConfig(max_batch_size=8, opt_batch_size=8)
    detector = YoloxDetectorConfig(backend=backend, score_thr=0.3, onnx_transform=dynamic_yolox).setup()
    pose = RtmPoseConfig(variant="rtmw-x-coco133", backend=backend).setup()
    return detector, pose
