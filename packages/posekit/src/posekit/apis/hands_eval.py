"""Ground-truth evaluation and batch-one speed of the posekit hand stacks on five catalog datasets.

The frames are fixed in the code (``posekit.eval.hand_gt.EVAL_SUBSETS``: three segments per
dataset, 100 frames each) and read straight from the catalog. Three protocols:

- ``gt-boxes`` (the EgoHand paper's HOT3D protocol, arXiv 2610.30293, Sec. 6.1): every visible
  ground-truth hand gets a box from its projected joints (1.2x the joint bounding box) and its true side.
- ``detector``: each model's own hand detector; a ground-truth hand counts as found when a
  same-side box overlaps its joint box with IoU >= 0.3 (recall), and only found hands are scored.
- ``end-to-end``: the stack as deployed, detector and full reconstruction at batch one on every
  frame. One-to-one IoU >= 0.3 matches give recall, mean IoU and side accuracy; the side-correct
  matches pool per-joint 2D, absolute 3D and wrist-relative 3D errors (median, p95). With
  ``--timing``, timed passes follow the scoring pass. These nine metrics are the accuracy bands:
  every fast EgoHand stack must stay within 5 % of ``egohand`` (``--baseline`` or the same run) on every dataset.

Metrics of the first two (``posekit.eval.hand_metrics``): MPJPE, PA-MPJPE, RTE (mm), 2D EPE (px),
mean and p90 over hands, on 20 joints (dataforge derives joint 1). Results go to ``results.json``;
``hands_eval.rrd`` holds each frame with ground truth beside every prediction and the per-frame
errors as scalars.
"""

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient
from serde import serde, to_dict
from serde.json import from_json
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor
from tqdm.auto import tqdm

from posekit.eval.hand_gt import EVAL_SUBSETS, MIN_JOINTS, SIDES, GtDataset, HandGtSegment, HandGtSubset, read_hand_gt
from posekit.eval.hand_metrics import HandErrors, Summary, box_iou, hand_errors, summarize
from posekit.models.base import HandDetector, TopDownHandPose3d
from posekit.models.detnet import DetNetHandDetectorConfig
from posekit.models.egohand import EgoHandAssetPaths, EgoHandConfig, EgoHandDetectorConfig, asset_paths
from posekit.models.egohand_fast import fast_egohand_configs
from posekit.models.wilor import WilorHandDetectorConfig, WilorHandPoseConfig
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d
from posekit.runtimes import TensorRtBackendConfig
from posekit.skeletons import HAND_21

ModelName: TypeAlias = Literal["wilor", "egohand", "egohand-fast", "egohand-fast-bf16", "detnet", "detnet-legacy"]
Protocol: TypeAlias = Literal["gt-boxes", "detector", "end-to-end"]
METRICS: tuple[str, ...] = ("mpjpe_mm", "pa_mpjpe_mm", "rte_mm", "epe_2d_px")
BANDED_METRICS: dict[str, bool] = {
    "recall": True,
    "mean_iou": True,
    "side_accuracy": True,
    "epe_2d_median_px": False,
    "epe_2d_p95_px": False,
    "absolute_3d_median_mm": False,
    "absolute_3d_p95_mm": False,
    "relative_3d_median_mm": False,
    "relative_3d_p95_mm": False,
}
"""The nine end-to-end accuracy bands per dataset, each mapped to whether higher is better."""


def within_accuracy_band(reference: float | None, candidate: float | None, *, higher_is_better: bool) -> bool:
    """At most five percent relative loss; missing or non-finite evidence fails."""
    if reference is None or candidate is None or not math.isfinite(reference) or not math.isfinite(candidate):
        return False
    return candidate >= 0.95 * reference if higher_is_better else candidate <= 1.05 * reference


def error_quantiles(errors: Sequence[float]) -> tuple[float | None, float | None]:
    """Median and p95 of scored joint errors, with no silent filtering."""
    if not errors:
        return None, None
    if not all(math.isfinite(x) for x in errors):
        raise ValueError("Scored errors must be finite.")
    return float(np.median(errors)), float(np.quantile(errors, 0.95))


@dataclass(frozen=True, slots=True)
class EvalModel:
    """How the viewer draws and the metrics score one evaluated model."""

    left_class: int
    """Annotation class of the model's left hand; its right hand is the next id."""
    metric_root: bool
    """The model predicts a metric camera-space wrist, so root translation error is scored; otherwise it is NaN."""
    banded: bool = False
    """End-to-end accuracy must stay within ``BANDED_METRICS``' 5 % of the ``egohand`` reference (the fast stacks)."""


EVAL_MODELS: dict[ModelName, EvalModel] = {
    "wilor": EvalModel(2, False), "egohand": EvalModel(4, True), "egohand-fast": EvalModel(6, True, banded=True),
    "egohand-fast-bf16": EvalModel(6, True, banded=True), "detnet": EvalModel(8, False), "detnet-legacy": EvalModel(10, False),
}
DETNET_TURNS: dict[str, dict[GtDataset, int]] = {
    "detnet": {"dataforge-hot3d-quest3": 0, "dataforge-umetrack": 3},
    "detnet-legacy": {"dataforge-hot3d-quest3": 1, "dataforge-umetrack": 0},
}
"""Detector-only DetNet-F entries (no pose stage, detector protocol only): clockwise quarter turns of each subset camera's image AS
RECORDED before the net (``detnet_turns`` subtracts what the conversion already turned); a dataset not listed gets none. ``detnet`` = trained upright (handtrack ``stream.upright``: UmeTrack's front camera is mounted a
quarter turn); ``detnet-legacy`` = the deployed DetNet-F's input (cameras as stored, Quest 3 portrait frames turned once)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HandsEvalConfig:
    """Fixed-frame ground-truth evaluation (and optionally speed) of the hand stacks."""

    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog serving the ``EVAL_SUBSETS`` datasets (the standing one is ``rerun+http://pablo-dl-server:51235``)."""
    output_dir: Path = Path("/tmp/posekit-hands-eval")
    """Where results.json and hands_eval.rrd go."""
    models: tuple[ModelName, ...] = ("wilor", "egohand")
    """Stacks to evaluate: ``egohand`` is the reference, the ``egohand-fast`` names are ``fast_egohand_configs``."""
    protocols: tuple[Protocol, ...] = ("gt-boxes", "detector")
    """Protocols to run."""
    datasets: tuple[GtDataset, ...] = tuple(subset.dataset for subset in EVAL_SUBSETS)
    """Datasets to run, from ``EVAL_SUBSETS``."""
    assets: Path | None = None
    """Local EgoHand asset tree; None uses the pinned Hugging Face downloads."""
    max_frames: int = 100
    """Frames per segment, spread evenly over those with a visible hand."""
    min_joints: int = 15
    """A hand is scored when at least this many of its 20 scored joints are visible (wrist included)."""
    box_scale: float = 1.2
    """GT-box protocol: box side as a multiple of the visible joints' bounding box."""
    match_iou: float = 0.3
    """Detector protocol: minimum IoU between a same-side detection and the ground-truth joint box."""
    detnet_onnx: Path | None = None
    """``detnet``: handtrack's exported ``detnet_full.onnx`` of a model trained upright."""
    detnet_legacy_onnx: Path | None = None
    """``detnet-legacy``: ``detnet_full.onnx`` of a model trained on the legacy input (the deployed DetNet-F)."""
    log_rrd: bool = True
    """Write hands_eval.rrd beside results.json."""
    timing: bool = False
    """``end-to-end``: time ``repeats`` batch-one passes after the scoring pass (run on an otherwise idle GPU)."""
    repeats: int = 2
    """Timed passes per segment."""
    baseline: Path | None = None
    """A ``results.json`` whose ``egohand`` end-to-end rows are the band reference; None uses ``egohand`` from this run."""

    def __post_init__(self) -> None:
        if (self.timing or self.baseline is not None) and "end-to-end" not in self.protocols:
            raise ValueError("--timing and --baseline measure the end-to-end protocol; add it to --protocols.")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Mean and p90 over hands; ``None`` when no hand was scored."""

    mean: float | None
    p90: float | None
    count: int


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EndToEndCell:
    """One model x dataset under the end-to-end protocol: the nine banded metrics and, with timing, latency."""

    model: ModelName
    dataset: GtDataset
    frames: int
    """Evaluated frames; timed repeats do not add to it."""
    hands_per_frame: float
    """Mean reconstructed hands per frame."""
    gt_hands: int
    """Visible ground-truth hands."""
    matched_hands: int
    """Ground-truth hands matched one-to-one at IoU >= 0.3 with the correct side and a reconstruction."""
    recall: float | None
    """Matched hands / visible ground-truth hands."""
    mean_iou: float | None
    """Best detector IoU per ground-truth hand, zero for misses."""
    side_accuracy: float | None
    """Correct sides among one-to-one IoU matches of either side."""
    epe_2d_median_px: float | None
    epe_2d_p95_px: float | None
    absolute_3d_median_mm: float | None
    """Camera-space joint error."""
    absolute_3d_p95_mm: float | None
    relative_3d_median_mm: float | None
    """Wrist-relative joint error."""
    relative_3d_p95_mm: float | None
    mean_ms: float | None
    """Mean detector + reconstruction latency per frame over the timed passes; None without ``--timing``."""
    median_ms: float | None
    p95_ms: float | None


@dataclass(slots=True)
class EndToEndScore:
    """End-to-end accuracy of one dataset, accumulated frame by frame."""

    frames: int = 0
    hands: int = 0
    gt_hands: int = 0
    matched_hands: int = 0
    side_correct: int = 0
    any_side_matches: int = 0
    ious: list[float] = field(default_factory=list)
    pixels: list[float] = field(default_factory=list)
    absolute_mm: list[float] = field(default_factory=list)
    relative_mm: list[float] = field(default_factory=list)

    def add_frame(self, segment: HandGtSegment, index: int, detections: HandDetections, hands: HandPoses3d, match_iou: float) -> None:
        """Match visible ground-truth hands one-to-one by IoU and pool the side-correct joint errors."""
        assert hands.keypoints.xyz_camera is not None
        boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
        pred_uv: Float32[ndarray, "m 21 2"] = hands.keypoints.xy.cpu().numpy()
        pred_xyz: Float32[ndarray, "m 21 3"] = hands.keypoints.xyz_camera.cpu().numpy()
        self.frames += 1
        self.hands += len(hands.sides)
        valid: Bool[ndarray, "2 21"] = segment.valid[index]
        gt_sides: list[int] = [s for s in range(2) if valid[s, 0] and valid[s].sum() >= MIN_JOINTS]
        self.gt_hands += len(gt_sides)
        gt_boxes: dict[int, Float32[ndarray, "4"]] = {s: joint_box(segment.uv[index, s], valid[s], 1.0) for s in gt_sides}
        pairs: list[tuple[float, int, int]] = sorted(((box_iou(box, gt_boxes[s]), s, j) for s in gt_sides for j, box in enumerate(boxes)), reverse=True)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        matches: list[int] = []
        for iou, s, j in pairs:
            if iou < match_iou or s in used_gt or j in used_pred:
                continue
            used_gt.add(s)
            used_pred.add(j)
            self.any_side_matches += 1
            if detections.sides[j] == SIDES[s]:
                self.side_correct += 1
                matches.append(s)
        self.ious.extend(max((iou for iou, t, _ in pairs if t == s), default=0.0) for s in gt_sides)
        for s in matches:
            if SIDES[s] not in hands.sides:
                continue
            row: int = hands.sides.index(SIDES[s])
            self.matched_hands += 1
            mask: Bool[ndarray, "21"] = valid[s]
            gt: Float32[ndarray, "21 3"] = segment.xyz_camera[index, s]
            self.pixels.extend(np.linalg.norm(pred_uv[row] - segment.uv[index, s], axis=-1)[mask].tolist())
            self.absolute_mm.extend((1000.0 * np.linalg.norm(pred_xyz[row] - gt, axis=-1)[mask]).tolist())
            self.relative_mm.extend((1000.0 * np.linalg.norm((pred_xyz[row] - pred_xyz[row, 0]) - (gt - gt[0]), axis=-1)[mask]).tolist())

    def cell(self, model: ModelName, dataset: GtDataset, latencies: list[float]) -> EndToEndCell:
        """The dataset's row."""
        epe_2d_median, epe_2d_p95 = error_quantiles(self.pixels)
        absolute_median, absolute_p95 = error_quantiles(self.absolute_mm)
        relative_median, relative_p95 = error_quantiles(self.relative_mm)
        return EndToEndCell(
            model=model,
            dataset=dataset,
            frames=self.frames,
            hands_per_frame=self.hands / self.frames,
            gt_hands=self.gt_hands,
            matched_hands=self.matched_hands,
            recall=self.matched_hands / self.gt_hands if self.gt_hands else None,
            mean_iou=float(np.mean(self.ious)) if self.ious else None,
            side_accuracy=self.side_correct / self.any_side_matches if self.any_side_matches else None,
            epe_2d_median_px=epe_2d_median,
            epe_2d_p95_px=epe_2d_p95,
            absolute_3d_median_mm=absolute_median,
            absolute_3d_p95_mm=absolute_p95,
            relative_3d_median_mm=relative_median,
            relative_3d_p95_mm=relative_p95,
            mean_ms=float(np.mean(latencies)) if latencies else None,
            median_ms=float(np.median(latencies)) if latencies else None,
            p95_ms=float(np.quantile(latencies, 0.95)) if latencies else None,
        )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EvalCell:
    """One model x dataset x protocol."""

    model: ModelName
    dataset: GtDataset
    protocol: Protocol
    gt_hands: int
    """Visible ground-truth hands in the subset."""
    scored_hands: int
    """Hands with a prediction (all for gt-boxes unless the model dropped one; found hands for detector)."""
    recall: float | None
    """Detector protocol: found with the correct side / visible ground-truth hands."""
    recall_any_side: float | None
    """Detector protocol: a box of either side overlaps the hand / visible ground-truth hands."""
    metrics: dict[str, MetricSummary]
    """Per metric (``mpjpe_mm``, ``pa_mpjpe_mm``, ``rte_mm``, ``epe_2d_px``); RTE is ``None`` for WiLoR (virtual camera)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class HandsEvalReport:
    """The whole evaluation; ``results.json``."""

    gpu: str
    """CUDA device name."""
    subsets: tuple[HandGtSubset, ...]
    max_frames: int
    min_joints: int
    box_scale: float
    match_iou: float
    cells: tuple[EvalCell, ...]
    """``gt-boxes`` and ``detector`` cells."""
    end_to_end: tuple[EndToEndCell, ...]
    """``end-to-end`` cells."""


def model_stack(name: ModelName, assets: Path | None) -> tuple[HandDetector, TopDownHandPose3d]:
    """The evaluated stacks: WiLoR on TensorRT, the unquantized EgoHand reference (the CLI's ``egohand``), and ``fast_egohand_configs``."""
    if name == "wilor":
        return WilorHandDetectorConfig(backend=TensorRtBackendConfig()).setup(), WilorHandPoseConfig(backend=TensorRtBackendConfig()).setup()
    if name == "egohand":
        paths: EgoHandAssetPaths = asset_paths(assets)
        return EgoHandDetectorConfig(paths.detector).setup(), EgoHandConfig(paths.weights, paths.soma).setup()
    detector_config, pose_config = fast_egohand_configs(assets, backbone_precision="bf16" if name == "egohand-fast-bf16" else "nvfp4")
    return detector_config.setup(), pose_config.setup()


def detnet_turns(name: ModelName, dataset: GtDataset, stored: int) -> int:
    """A DetNet-F entry's turns for a subset camera whose conversion already applied ``stored`` (``HandGtSegment.upright_quarter_turns_cw``),
    so the net sees the same image before and after an upright re-convert."""
    return (DETNET_TURNS[name].get(dataset, 0) - stored) % 4


def detnet_detector(name: ModelName, dataset: GtDataset, segments: Sequence[HandGtSegment], config: HandsEvalConfig) -> HandDetector:
    """A DetNet-F entry's detector for one subset camera (its turn count depends on the camera and on how it was converted)."""
    weights: Path | None = config.detnet_onnx if name == "detnet" else config.detnet_legacy_onnx
    if weights is None:
        raise ValueError(f"{name} needs --{name}-onnx")
    stored: set[int] = {segment.upright_quarter_turns_cw for segment in segments}
    if len(stored) != 1:
        raise ValueError(f"{dataset}: segments were converted with different upright turns {sorted(stored)}; re-convert all of them")
    return DetNetHandDetectorConfig(weights=weights, quarter_turns=detnet_turns(name, dataset, stored.pop()), backend=TensorRtBackendConfig()).setup()


def decode_samples(segment: HandGtSegment) -> UInt8[Tensor, "f h w 3"]:
    """CUDA-decode the evaluation samples of a segment's catalog video."""
    from torchcodec.decoders import VideoDecoder

    fps: float = 1e9 / float(np.median(np.diff(segment.video.t_ns)))
    decoder = VideoDecoder(wrap_mp4(segment.video.samples, segment.video.keyframes, round(fps), codec=segment.video.codec), device="cuda", seek_mode="approximate")
    return rearrange(decoder.get_frames_at(segment.sample_indices.tolist()).data, "n c h w -> n h w c").contiguous()


def joint_box(uv: Float32[ndarray, "21 2"], valid: Bool[ndarray, "21"], scale: float) -> Float32[ndarray, "4"]:
    """Square-agnostic box around the visible joints, scaled about its centre."""
    low: Float32[ndarray, "2"] = uv[valid].min(0)
    high: Float32[ndarray, "2"] = uv[valid].max(0)
    center: Float32[ndarray, "2"] = (low + high) / 2
    half: Float32[ndarray, "2"] = (high - low) / 2 * scale
    return np.concatenate([center - half, center + half]).astype(np.float32)


@dataclass(slots=True)
class ScoredHand:
    """One scored ground-truth hand of one frame."""

    frame: int
    side_index: int
    errors: HandErrors
    pred_xyz: Float32[ndarray, "21 3"]
    pred_uv: Float32[ndarray, "21 2"]


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def evaluate_segment(
    segment: HandGtSegment,
    frames: UInt8[Tensor, "f h w 3"],
    detector: HandDetector,
    pose: TopDownHandPose3d | None,
    protocol: Protocol,
    config: HandsEvalConfig,
    metric_root: bool,
) -> tuple[int, int, list[ScoredHand], HandDetections]:
    """Score one model on one segment under one protocol.

    Returns:
        Visible ground-truth hands, those overlapped by a box of either side, the scored hands,
        and the detector's raw boxes (the GT boxes for the gt-boxes protocol).
    """
    scored_mask: Bool[ndarray, "f 2"] = (segment.valid.sum(-1) >= config.min_joints) & segment.valid[:, :, 0]
    gt_boxes: dict[tuple[int, int], Float32[ndarray, "4"]] = {
        (int(f), int(s)): joint_box(segment.uv[f, s], segment.valid[f, s], 1.0) for f, s in zip(*np.nonzero(scored_mask), strict=True)
    }
    keys: list[tuple[int, int]] = []
    any_side: int = 0
    boxes: list[Float32[ndarray, "4"]] = []
    scores: list[float] = []
    found: HandDetections | None = None
    if protocol == "gt-boxes":
        for (frame, side_index), _ in gt_boxes.items():
            keys.append((frame, side_index))
            boxes.append(joint_box(segment.uv[frame, side_index], segment.valid[frame, side_index], config.box_scale))
            scores.append(1.0)
    else:
        found = detector(frames)
        found_boxes: Float32[ndarray, "n 4"] = found.boxes.xyxy_numpy()
        found_frames: Int64[ndarray, "n"] = found.boxes.frame_indices.cpu().numpy()
        found_scores: Float32[ndarray, "n"] = found.boxes.scores.cpu().numpy().astype(np.float32)
        for (frame, side_index), gt_box in gt_boxes.items():
            candidates: list[int] = [int(i) for i in np.flatnonzero(found_frames == frame) if found.sides[i] == SIDES[side_index]]
            any_side += any(box_iou(found_boxes[i], gt_box) >= config.match_iou for i in np.flatnonzero(found_frames == frame))
            ious: list[float] = [box_iou(found_boxes[i], gt_box) for i in candidates]
            if ious and max(ious) >= config.match_iou:
                best: int = candidates[int(np.argmax(ious))]
                keys.append((frame, side_index))
                boxes.append(found_boxes[best])
                scores.append(float(found_scores[best]))
    device: torch.device = frames.device
    detections = HandDetections(
        BoxDetections(
            torch.from_numpy(np.stack(boxes) if boxes else np.zeros((0, 4), np.float32)).to(device),
            torch.tensor(scores, dtype=torch.float32, device=device),
            torch.tensor([frame for frame, _ in keys], dtype=torch.int64, device=device),
        ),
        tuple(SIDES[side_index] for _, side_index in keys),
    )
    scored: list[ScoredHand] = []
    if pose is None:  # a detector-only model: each found hand counts for recall and has no pose error
        nan_errors: HandErrors = HandErrors(float("nan"), float("nan"), float("nan"), float("nan"))
        scored = [ScoredHand(frame, side_index, nan_errors, np.full((21, 3), np.nan, np.float32), np.full((21, 2), np.nan, np.float32)) for frame, side_index in keys]
    elif detections.num_detections:
        hands: HandPoses3d = pose(frames, detections, segment.camera).as_hand21()
        assert hands.keypoints.xyz_camera is not None and hands.keypoints.skeleton == HAND_21
        hand_frames: list[int] = hands.keypoints.frame_indices.tolist()
        xyz: Float32[ndarray, "m 21 3"] = hands.keypoints.xyz_camera.cpu().numpy()
        uv: Float32[ndarray, "m 21 2"] = hands.keypoints.xy.cpu().numpy()
        for row, (frame, side) in enumerate(zip(hand_frames, hands.sides, strict=True)):
            side_index: int = SIDES.index(side)
            errors: HandErrors = hand_errors(xyz[row], segment.xyz_camera[frame, side_index], uv[row], segment.uv[frame, side_index], segment.valid[frame, side_index])
            if not metric_root:
                errors = HandErrors(errors.mpjpe_mm, errors.pa_mpjpe_mm, float("nan"), errors.epe_2d_px)
            scored.append(ScoredHand(frame, side_index, errors, xyz[row], uv[row]))
    return len(gt_boxes), any_side if protocol == "detector" else len(gt_boxes), scored, found if found is not None else detections


def log_segment_frames(
    model: ModelName, protocol: Protocol, dataset: GtDataset, segment: HandGtSegment, scored: list[ScoredHand], boxes: HandDetections, frame_offset: int
) -> None:
    """Predictions next to ground truth on the ``eval_frame`` timeline, and per-frame mean errors as scalars."""
    root: str = f"{dataset}/{segment.segment_id}"
    for frame in range(len(segment.sample_indices)):
        rr.set_time("eval_frame", sequence=frame_offset + frame)
        rows: list[ScoredHand] = [hand for hand in scored if hand.frame == frame]
        in_frame: Int64[ndarray, "k"] = np.flatnonzero(boxes.boxes.frame_indices.cpu().numpy() == frame)
        rr.log(
            f"{root}/camera/image/{model}/{protocol}/boxes",
            rr.Boxes2D(
                array=boxes.boxes.xyxy_numpy()[in_frame].reshape(-1, 4), array_format=rr.Box2DFormat.XYXY,
                class_ids=[EVAL_MODELS[model].left_class + SIDES.index(boxes.sides[i]) for i in in_frame], show_labels=False,
            ),
        )
        for side_index, side in enumerate(SIDES):
            path: str = f"{model}/{protocol}/{side}"
            hand: ScoredHand | None = next((h for h in rows if h.side_index == side_index), None)
            if hand is None:
                rr.log(f"{root}/camera/image/{path}", rr.Clear(recursive=False))
                rr.log(f"{root}/camera/{path}", rr.Clear(recursive=False))
                continue
            rr.log(f"{root}/camera/image/{path}", rr.Points2D(hand.pred_uv, keypoint_ids=np.arange(21, dtype=np.uint16), class_ids=EVAL_MODELS[model].left_class + side_index, radii=-2.0, show_labels=False))
            # Shown wrist-on-wrist with the ground truth, the shape MPJPE scores (WiLoR's root is in a virtual camera).
            shown: Float32[ndarray, "21 3"] = (hand.pred_xyz - hand.pred_xyz[0] + segment.xyz_camera[frame, side_index, 0]).astype(np.float32)
            rr.log(f"{root}/camera/{path}", rr.Points3D(shown, keypoint_ids=np.arange(21, dtype=np.uint16), class_ids=EVAL_MODELS[model].left_class + side_index, radii=0.003, show_labels=False))
        for metric in METRICS:
            values: list[float] = [getattr(hand.errors, metric) for hand in rows if np.isfinite(getattr(hand.errors, metric))]
            if values:
                rr.log(f"metrics/{model}/{protocol}/{dataset}/{metric}", rr.Scalars(float(np.mean(values))))


def log_segment_ground_truth(dataset: GtDataset, segment: HandGtSegment, frames: UInt8[Tensor, "f h w 3"], frame_offset: int) -> None:
    """The evaluated frames, their camera and both ground-truth hands (2D and 3D)."""
    root: str = f"{dataset}/{segment.segment_id}"
    camera = segment.camera
    rr.log(f"{root}/camera/image", rr.Pinhole(image_from_camera=camera.intrinsics.astype(np.float32), resolution=[camera.width, camera.height], image_plane_distance=0.05), static=True)
    host: UInt8[ndarray, "f h w 3"] = frames.cpu().numpy()
    for frame in range(len(segment.sample_indices)):
        rr.set_time("eval_frame", sequence=frame_offset + frame)
        rr.set_time("video_time", duration=1e-9 * float(segment.video_time_ns[frame]))
        rr.log(f"{root}/camera/image", rr.Image(host[frame]).compress(jpeg_quality=85))
        for side_index, side in enumerate(SIDES):
            valid: Bool[ndarray, "21"] = segment.valid[frame, side_index]
            uv: Float32[ndarray, "21 2"] = np.where(valid[:, None], segment.uv[frame, side_index], np.nan).astype(np.float32)
            xyz: Float32[ndarray, "21 3"] = np.where(valid[:, None], segment.xyz_camera[frame, side_index], np.nan).astype(np.float32)
            rr.log(f"{root}/camera/image/gt/{side}", rr.Points2D(uv, keypoint_ids=np.arange(21, dtype=np.uint16), class_ids=0, radii=-2.0, show_labels=False))
            rr.log(f"{root}/camera/gt/{side}", rr.Points3D(xyz, keypoint_ids=np.arange(21, dtype=np.uint16), class_ids=0, radii=0.003, show_labels=False))
    rr.set_time("eval_frame", sequence=frame_offset + len(segment.sample_indices))
    rr.log(root, rr.Clear(recursive=True))


def log_context() -> None:
    """GT white; WiLoR left/right blue/orange; EgoHand left/right green/magenta; every class carries the HAND_21 links."""
    classes = [
        (0, "ground truth", (240, 240, 240)),
        (2, "wilor left", (66, 135, 245)), (3, "wilor right", (245, 135, 66)),
        (4, "egohand left", (60, 200, 90)), (5, "egohand right", (220, 60, 200)),
        (6, "fast egohand left", (90, 220, 240)), (7, "fast egohand right", (255, 220, 80)),
        (8, "detnet left", (150, 110, 255)), (9, "detnet right", (255, 90, 90)),
        (10, "detnet-legacy left", (120, 120, 160)), (11, "detnet-legacy right", (160, 120, 120)),
    ]
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(info=rr.AnnotationInfo(id=class_id, label=label, color=color), keypoint_connections=list(HAND_21.links))
                for class_id, label, color in classes
            ]
        ),
        static=True,
    )


def run_end_to_end(
    segment: HandGtSegment, frames: UInt8[Tensor, "f h w 3"], detector: HandDetector, pose: TopDownHandPose3d, score: EndToEndScore, latencies: list[float], config: HandsEvalConfig
) -> None:
    """Score every frame at batch one, then time ``repeats`` synchronized detector + reconstruction passes (frames already decoded)."""
    for index, frame in enumerate(frames):
        detections: HandDetections = detector(frame[None])
        score.add_frame(segment, index, detections, pose(frame[None], detections, segment.camera).as_hand21(), config.match_iou)
    for _ in range(config.repeats if config.timing else 0):
        for frame in frames:
            torch.cuda.synchronize()
            start: float = time.perf_counter()
            pose(frame[None], detector(frame[None]), segment.camera)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)


def failed_bands(reference: list[EndToEndCell], candidate: list[EndToEndCell]) -> list[str]:
    """Every candidate dataset metric outside its 5 % band; a dataset missing from the reference fails."""
    references: dict[GtDataset, EndToEndCell] = {row.dataset: row for row in reference}
    failed: list[str] = []
    for row in candidate:
        if row.dataset not in references:
            failed.append(f"{row.dataset}: no reference")
            continue
        failed += [
            f"{row.dataset}: {name}"
            for name, higher_is_better in BANDED_METRICS.items()
            if not within_accuracy_band(getattr(references[row.dataset], name), getattr(row, name), higher_is_better=higher_is_better)
        ]
    return failed


def main(config: HandsEvalConfig) -> HandsEvalReport:
    """Run the fixed-frame evaluation, write ``results.json`` (and ``hands_eval.rrd``) and check the end-to-end bands.

    Returns:
        The report that was written.

    Raises:
        RuntimeError: If a fast EgoHand stack's end-to-end band fails against the ``egohand`` reference.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.log_rrd:
        rr.init("posekit-hands-eval", spawn=False)
        rr.save(config.output_dir / "hands_eval.rrd")
        log_context()
    client = CatalogClient(config.catalog_url)
    subsets: tuple[HandGtSubset, ...] = tuple(subset for subset in EVAL_SUBSETS if subset.dataset in config.datasets)
    gt: dict[GtDataset, list[HandGtSegment]] = {
        subset.dataset: read_hand_gt(client.get_dataset(subset.dataset), subset, max_frames=config.max_frames) for subset in subsets
    }
    offsets: dict[str, int] = {}
    offset: int = 0
    for segment in (segment for segments in gt.values() for segment in segments):
        offsets[segment.segment_id] = offset
        offset += len(segment.sample_indices) + 1
    cells: list[EvalCell] = []
    end_to_end: list[EndToEndCell] = []
    for model_index, model in enumerate(config.models):
        detector_only: bool = model in DETNET_TURNS
        # A detector-only model is scored on the detector protocol alone, with a detector per subset camera.
        protocols: tuple[Protocol, ...] = ("detector",) if detector_only else config.protocols
        detector, pose = (None, None) if detector_only else model_stack(model, config.assets)
        for dataset, segments in gt.items():
            if detector_only:
                detector = detnet_detector(model, dataset, segments, config)
            assert detector is not None
            gt_hands: dict[Protocol, int] = dict.fromkeys(protocols, 0)
            overlapped: dict[Protocol, int] = dict.fromkeys(protocols, 0)
            scored: dict[Protocol, list[ScoredHand]] = {protocol: [] for protocol in protocols}
            score: EndToEndScore = EndToEndScore()
            latencies: list[float] = []
            for segment in tqdm(segments, desc=f"{model} {dataset}"):
                frames: UInt8[Tensor, "f h w 3"] = decode_samples(segment)
                if config.log_rrd and model_index == 0:
                    log_segment_ground_truth(dataset, segment, frames, offsets[segment.segment_id])
                with torch.inference_mode():
                    for protocol in protocols:
                        if protocol == "end-to-end":
                            assert pose is not None
                            run_end_to_end(segment, frames, detector, pose, score, latencies, config)
                            continue
                        count, any_side, hands, boxes = evaluate_segment(segment, frames, detector, pose, protocol, config, metric_root=EVAL_MODELS[model].metric_root)
                        gt_hands[protocol] += count
                        overlapped[protocol] += any_side
                        scored[protocol].extend(hands)
                        if config.log_rrd:
                            log_segment_frames(model, protocol, dataset, segment, hands, boxes, offsets[segment.segment_id])
                del frames
                torch.cuda.empty_cache()
            for protocol in protocols:
                if protocol == "end-to-end":
                    end_to_end.append(score.cell(model, dataset, latencies))
                    print(f"[hands-eval] {model:17s} {dataset:29s} end-to-end", to_dict(end_to_end[-1]), flush=True)
                    continue
                metrics: dict[str, MetricSummary] = {}
                for metric in METRICS:
                    values: Float64[ndarray, "n"] = np.asarray([getattr(hand.errors, metric) for hand in scored[protocol]], np.float64)
                    summary: Summary = summarize(values[np.isfinite(values)])
                    metrics[metric] = MetricSummary(summary.mean, summary.p90, summary.count)
                found: int = len(scored[protocol])
                total: int = gt_hands[protocol]
                cells.append(
                    EvalCell(
                        model, dataset, protocol, total, found,
                        found / total if protocol == "detector" and total else None,
                        overlapped[protocol] / total if protocol == "detector" and total else None,
                        metrics,
                    )
                )
                print(f"[hands-eval] {model:17s} {dataset:29s} {protocol:9s} gt={total} scored={found} "
                      + " ".join(f"{m}={_rounded(metrics[m].mean)}" for m in METRICS), flush=True)
        del detector, pose
        torch.cuda.empty_cache()
    report = HandsEvalReport(
        torch.cuda.get_device_name(), subsets, config.max_frames, config.min_joints, config.box_scale, config.match_iou, tuple(cells), tuple(end_to_end)
    )
    (config.output_dir / "results.json").write_text(json.dumps(to_dict(report), indent=2) + "\n")
    if config.log_rrd:
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        *[rrb.Spatial2DView(origin=f"{subset.dataset}/{subset.segments[0]}/camera/image", name=subset.dataset) for subset in subsets],
                        rrb.Spatial3DView(origin=f"{subsets[0].dataset}/{subsets[0].segments[0]}/camera", name="3D: GT white, WiLoR blue/orange, EgoHand green/magenta (wrist on GT wrist)"),
                    ),
                    rrb.TimeSeriesView(origin="metrics", name="per-frame errors"),
                    row_shares=[3, 1],
                ),
                collapse_panels=True,
            )
        )
        recording: rr.RecordingStream | None = rr.get_global_data_recording()
        assert recording is not None
        recording.flush()
    if "end-to-end" in config.protocols:
        references: tuple[EndToEndCell, ...] = from_json(HandsEvalReport, config.baseline.read_text()).end_to_end if config.baseline else report.end_to_end
        reference: list[EndToEndCell] = [row for row in references if row.model == "egohand"]
        if config.baseline is not None and not reference:
            raise ValueError(f"{config.baseline} has no egohand end-to-end rows")
        failures: list[str] = []
        for model in config.models:
            if not reference or not EVAL_MODELS[model].banded:
                continue
            rows: list[EndToEndCell] = [row for row in report.end_to_end if row.model == model]
            failed: list[str] = failed_bands(reference, rows)
            print(f"[hands-eval] BANDS {model}: {len(BANDED_METRICS) * len(rows) - len(failed)}/{len(BANDED_METRICS) * len(rows)} within 5 % of egohand", flush=True)
            failures += [f"{model} {failure}" for failure in failed]
        if failures:
            raise RuntimeError("Accuracy bands failed: " + "; ".join(failures))
    return report
