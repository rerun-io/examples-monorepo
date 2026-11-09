"""Ground-truth evaluation of the posekit hand models on HOT3D (Quest 3) and UmeTrack.

Reproduces the EgoHand paper's HOT3D protocol (arXiv 2610.30293, Sec. 6.1, Table 2) as
closely as dataforge's layers allow, on a fixed subset that runs in minutes:

- ``gt-boxes`` (the paper's protocol): every visible ground-truth hand gets a box from its
  projected joints (1.2x the joint bounding box) and its true side.
- ``detector``: each model's own hand detector; a ground-truth hand counts as found when a
  same-side box overlaps its joint box with IoU >= 0.3 (recall), and only found hands are scored.

Metrics (``posekit.eval.hand_metrics``): MPJPE, PA-MPJPE, RTE (mm), 2D EPE (px), mean and
p90 over hands, on 20 joints (dataforge derives joint 1). Results go to ``results.json``;
``hands_eval.rrd`` holds each frame with ground truth beside every prediction and the
per-frame errors as scalars.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient
from serde import serde, to_dict
from torch import Tensor
from tqdm.auto import tqdm

from posekit.eval.hand_gt import SIDES, GtDataset, HandGtSegment, decode_catalog_frames, read_hand_gt
from posekit.eval.hand_metrics import HandErrors, Summary, box_iou, hand_errors, summarize
from posekit.models.base import HandDetector, TopDownHandPose3d
from posekit.models.egohand import EgoHandConfig
from posekit.models.egohand_fast import EGOHAND_STACKS, EgoHandFastConfig, EgoHandStack, EgoHandStackName
from posekit.models.wilor import WilorHandDetectorConfig, WilorHandPoseConfig
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d
from posekit.runtimes import TensorRtBackendConfig
from posekit.skeletons import HAND_21

ModelName: TypeAlias = Literal["wilor"] | EgoHandStackName
Protocol: TypeAlias = Literal["gt-boxes", "detector"]
METRICS: tuple[str, ...] = ("mpjpe_mm", "pa_mpjpe_mm", "rte_mm", "epe_2d_px")


@dataclass(frozen=True, slots=True)
class EvalModel:
    """How the viewer draws and the metrics score one evaluated model."""

    left_class: int
    """Annotation class of the model's left hand; its right hand is the next id."""
    metric_root: bool
    """The model predicts a metric camera-space wrist, so root translation error is scored; otherwise it is NaN."""


def eval_model(name: ModelName) -> EvalModel:
    """WiLoR blue/orange in a virtual camera; reference EgoHand stacks green/magenta, fast ones cyan/yellow, both metric."""
    if name == "wilor":
        return EvalModel(2, False)
    return EvalModel(4 if isinstance(EGOHAND_STACKS[name](None).pose, EgoHandConfig) else 6, True)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EvalSubset:
    """One dataset's fixed evaluation subset."""

    dataset: GtDataset
    """Catalog dataset name."""
    camera: str
    """Evaluated camera entity (``cam_MM``)."""
    segments: tuple[str, ...]
    """Segment ids, one per participant/user."""


SUBSETS: tuple[EvalSubset, ...] = (
    EvalSubset(
        "dataforge-hot3d-quest3",
        "cam_00",
        ("hot3d-quest3__P0010_086ff32a", "hot3d-quest3__P0017_1a07e9e7", "hot3d-quest3__P0021_03d537b2"),
    ),
    EvalSubset(
        "dataforge-umetrack",
        "cam_01",
        (
            "umetrack__real__hand_hand__testing__user_05__recording_00",
            "umetrack__real__hand_hand__testing__user_09__recording_00",
            "umetrack__real__separate_hand__testing__user_19__recording_00",
        ),
    ),
)
"""HOT3D: SLAM-left camera of three participants (train split: the test split has no labels).
UmeTrack: front camera of three real test recordings, two users with both hands, one separate-hand."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HandsEvalConfig:
    """Fixed-subset ground-truth evaluation of WiLoR and EgoHand."""

    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog serving dataforge-hot3d-quest3 and dataforge-umetrack (base, hand_pose, projections)."""
    output_dir: Path = Path("/tmp/posekit-hands-eval")
    """Where results.json and hands_eval.rrd go."""
    models: tuple[ModelName, ...] = ("wilor", "egohand-selfcap")
    """Models to evaluate: ``wilor`` or ``EGOHAND_STACKS`` names."""
    protocols: tuple[Protocol, ...] = ("gt-boxes", "detector")
    """Box sources to evaluate."""
    stride: int = 10
    """Keep every ``stride``-th labelled frame (HOT3D and UmeTrack run at 30 fps: one frame per 1/3 s)."""
    max_frames: int = 40
    """Frames per segment."""
    min_joints: int = 15
    """A hand is scored when at least this many of its 20 scored joints are visible (wrist included)."""
    box_scale: float = 1.2
    """GT-box protocol: box side as a multiple of the visible joints' bounding box."""
    match_iou: float = 0.3
    """Detector protocol: minimum IoU between a same-side detection and the ground-truth joint box."""
    log_rrd: bool = True
    """Write hands_eval.rrd beside results.json."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Mean and p90 over hands; ``None`` when no hand was scored."""

    mean: float | None
    p90: float | None
    count: int


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

    subsets: tuple[EvalSubset, ...]
    stride: int
    max_frames: int
    min_joints: int
    box_scale: float
    match_iou: float
    cells: tuple[EvalCell, ...]


def model_stack(name: ModelName) -> tuple[HandDetector, TopDownHandPose3d]:
    """WiLoR on TensorRT, or the named EgoHand stack.

    Reference stacks run on their lens-aware CPU crops: the reference GPU crop sampler
    handles rational/pinhole lenses only, and HOT3D and UmeTrack are fisheye. The fast
    role's CUDA cropper handles every lens.
    """
    if name == "wilor":
        return WilorHandDetectorConfig(backend=TensorRtBackendConfig()).setup(), WilorHandPoseConfig(backend=TensorRtBackendConfig()).setup()
    stack: EgoHandStack[EgoHandConfig] | EgoHandStack[EgoHandFastConfig] = EGOHAND_STACKS[name](None)
    pose: EgoHandConfig | EgoHandFastConfig = replace(stack.pose, gpu_crops=False) if isinstance(stack.pose, EgoHandConfig) else stack.pose
    return stack.detector.setup(), pose.setup()


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
    pose: TopDownHandPose3d,
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
    if detections.num_detections:
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
                class_ids=[eval_model(model).left_class + SIDES.index(boxes.sides[i]) for i in in_frame], show_labels=False,
            ),
        )
        for side_index, side in enumerate(SIDES):
            path: str = f"{model}/{protocol}/{side}"
            hand: ScoredHand | None = next((h for h in rows if h.side_index == side_index), None)
            if hand is None:
                rr.log(f"{root}/camera/image/{path}", rr.Clear(recursive=False))
                rr.log(f"{root}/camera/{path}", rr.Clear(recursive=False))
                continue
            rr.log(f"{root}/camera/image/{path}", rr.Points2D(hand.pred_uv, keypoint_ids=np.arange(21, dtype=np.uint16), class_ids=eval_model(model).left_class + side_index, radii=-2.0, show_labels=False))
            # Shown wrist-on-wrist with the ground truth, the shape MPJPE scores (WiLoR's root is in a virtual camera).
            shown: Float32[ndarray, "21 3"] = (hand.pred_xyz - hand.pred_xyz[0] + segment.xyz_camera[frame, side_index, 0]).astype(np.float32)
            rr.log(f"{root}/camera/{path}", rr.Points3D(shown, keypoint_ids=np.arange(21, dtype=np.uint16), class_ids=eval_model(model).left_class + side_index, radii=0.003, show_labels=False))
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


def main(config: HandsEvalConfig) -> HandsEvalReport:
    """Run the fixed-subset evaluation and write ``results.json`` (and ``hands_eval.rrd``).

    Returns:
        The report that was written.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.log_rrd:
        rr.init("posekit-hands-eval", spawn=False)
        rr.save(config.output_dir / "hands_eval.rrd")
        log_context()
    client = CatalogClient(config.catalog_url)
    data: dict[GtDataset, list[tuple[HandGtSegment, UInt8[Tensor, "f h w 3"]]]] = {}
    offsets: dict[str, int] = {}
    offset: int = 0
    for subset in SUBSETS:
        segments: list[HandGtSegment] = read_hand_gt(
            client.get_dataset(subset.dataset), subset.dataset, subset.segments, subset.camera, stride=config.stride, max_frames=config.max_frames
        )
        data[subset.dataset] = [(segment, decode_catalog_frames(segment.video, segment.sample_indices.tolist())) for segment in segments]
        for segment, frames in data[subset.dataset]:
            offsets[segment.segment_id] = offset
            if config.log_rrd:
                log_segment_ground_truth(subset.dataset, segment, frames, offset)
            offset += len(segment.sample_indices) + 1
    cells: list[EvalCell] = []
    for model in config.models:
        detector, pose = model_stack(model)
        for dataset, segments_frames in data.items():
            for protocol in config.protocols:
                gt_hands: int = 0
                overlapped: int = 0
                scored: list[ScoredHand] = []
                for segment, frames in tqdm(segments_frames, desc=f"{model} {dataset} {protocol}"):
                    with torch.inference_mode():
                        count, any_side, hands, boxes = evaluate_segment(segment, frames, detector, pose, protocol, config, metric_root=eval_model(model).metric_root)
                    gt_hands += count
                    overlapped += any_side
                    scored.extend(hands)
                    if config.log_rrd:
                        log_segment_frames(model, protocol, dataset, segment, hands, boxes, offsets[segment.segment_id])
                metrics: dict[str, MetricSummary] = {}
                for metric in METRICS:
                    values: Float64[ndarray, "n"] = np.asarray([getattr(hand.errors, metric) for hand in scored], np.float64)
                    summary: Summary = summarize(values[np.isfinite(values)])
                    metrics[metric] = MetricSummary(summary.mean, summary.p90, summary.count)
                cells.append(
                    EvalCell(
                        model, dataset, protocol, gt_hands, len(scored),
                        len(scored) / gt_hands if protocol == "detector" and gt_hands else None,
                        overlapped / gt_hands if protocol == "detector" and gt_hands else None,
                        metrics,
                    )
                )
                print(f"[hands-eval] {model:8s} {dataset:24s} {protocol:9s} gt={gt_hands} scored={len(scored)} "
                      + " ".join(f"{m}={_rounded(metrics[m].mean)}" for m in METRICS))
        del detector, pose
        torch.cuda.empty_cache()
    report = HandsEvalReport(SUBSETS, config.stride, config.max_frames, config.min_joints, config.box_scale, config.match_iou, tuple(cells))
    (config.output_dir / "results.json").write_text(json.dumps(to_dict(report), indent=2) + "\n")
    if config.log_rrd:
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        *[
                            rrb.Spatial2DView(origin=f"{subset.dataset}/{segment}/camera/image", name=f"{segment[-24:]}")
                            for subset in SUBSETS
                            for segment in subset.segments[:1]
                        ],
                        rrb.Spatial3DView(origin=f"{SUBSETS[0].dataset}/{SUBSETS[0].segments[0]}/camera", name="3D: GT white, WiLoR blue/orange, EgoHand green/magenta (wrist on GT wrist)"),
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
    return report
