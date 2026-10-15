"""Causal multi-view person tracking: bootstrap once, then forward only.

YOLO12x + CLIP + epipolar identity run at bootstrap and on a sparse re-detect cadence; every tick the SAM2 masks advance
in one batch over the views. Identity state (CLIP banks, lost / held counters, the per-view SAM2 memories) lives here; the
models come in through the constructor (posekit ``PersonDetector`` / ``IdentityEncoder`` roles and ``Sam2Masks``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from dagkit.node import Emit, Fenced, PortType, RunContext, Tick
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from posekit.models import IdentityEncoder, PersonDetector
from posekit.predictions import BoxDetections
from serde import serde
from simplecv.gpu import consume, fence_after
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras
from torch import Tensor
from torchvision.ops import box_iou

from mamma.ops.identity import CLIP_WEIGHT, EPIPOLAR_WEIGHT, FeatureBank, assign_hungarian, epipolar_score, fundamental_matrix, resolve_epipolar_px
from mamma.types import MultiViewTracks

BOOTSTRAP_MIN_SCORE: float = 0.25
"""Minimum combined CLIP + epipolar score to accept a cross-view match."""
BANK_UPDATE_MIN_SCORE: float = 0.35
"""Minimum CLIP score to append a re-detect crop to the feature bank."""
LOST_TICKS_BEFORE_REPROMPT: int = 5
"""Consecutive empty-mask ticks before a view is re-prompted."""
ANCHOR_CONFLICT_IOU: float = 0.65
"""Re-detect prompts of two subjects whose boxes overlap this much are de-conflicted (never prompt SAM2 twice on one person)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrackParams:
    """Tracker knobs (profile node ``track``); the model fields are read by the App when it builds the models."""

    sam2_config: str = "configs/efficienttam/efficienttam_ti_512x512.yaml"
    """Hydra config inside the vendored sam2 package (EfficientTAM-ti: 21.8 ms per 4-view tick vs 52.6 for hiera-small)."""
    sam2_checkpoint: str = "data/weights/efficienttam/efficienttam_ti.pt"
    yolo_checkpoint: str = "data/weights/yolo/yolo12x.pt"
    memory_window_size: int = 7
    """Non-conditional SAM2 memories kept per object."""
    expected_subjects: int | None = None
    """People to track; None = as many as the bootstrap view detects."""
    redetect_interval: int = 120
    """Ticks between routine re-detect passes while tracking one subject (a lost track re-detects at once)."""
    redetect_interval_multi: int = 15
    """Routine cadence whenever more than one subject is tracked: epipolar re-anchoring fixes contact identity swaps."""
    reprompt_alive_on_routine: bool = True
    """Routine passes also re-prompt confidently matched live tracks (fixes slow mask drift)."""
    transient_hold: bool = True
    """Hold the previous mask for a tick when a mask collapses to a small fraction of its recent area (a flip onto a distractor)."""
    transient_area_frac: float = 0.45
    """Area fraction of the rolling median below which a non-empty mask counts as collapsed."""
    track_stride: int = 1
    """Advance the masks every Nth tick; skipped ticks repeat the previous tracks."""


@runtime_checkable
class MaskTracker(Protocol):
    """What the node needs from ``models.sam2.Sam2Masks`` (tests inject a fake)."""

    def new_states(self, views: int, video_hw: tuple[int, int]) -> list[Any]: ...
    def __call__(
        self, states: list[Any], frame_index: int, frames: UInt8[Tensor, "v 3 h w"], prompts: list[dict[int, Float32[ndarray, "4"]]]
    ) -> list[dict[int, tuple[Bool[Tensor, "h w"], Float32[Tensor, " 1"]]]]: ...


@dataclass(slots=True)
class Track:
    """One person's track in one view at one tick."""

    mask: Bool[Tensor, "h w"]
    box_xyxy: Float32[ndarray, "4"] | None
    """None when the mask is empty."""
    score: float


class TrackNode:
    """Track every person in every view.

    Masks and boxes per (view, person) at engine size; a person keeps one id across views and time (CLIP + epipolar
    identity at bootstrap and re-detects, SAM2 masks every tick).
    """

    inputs: Mapping[str, PortType] = {"frames": Fenced[MultiViewFrames], "rig": RigCameras}
    outputs: Mapping[str, PortType] = {"tracks": Fenced[MultiViewTracks]}

    def __init__(self, params: TrackParams, detector: PersonDetector, encoder: IdentityEncoder, masks: MaskTracker) -> None:
        self.params, self.detector, self.encoder, self.masks = params, detector, encoder, masks

    def reset(self, ctx: RunContext) -> None:
        self.bank = FeatureBank()
        self.states: list[Any] = []
        self.lost: list[dict[int, int]] = []
        self.held: list[dict[int, int]] = []
        self.areas: list[dict[int, list[float]]] = []
        """Rolling healthy mask areas per view and person (collapsed or empty ticks are not appended)."""
        self.last: list[dict[int, Track]] | None = None
        self.ticks_seen: int = 0

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, Fenced) and isinstance(tick.value.value, MultiViewFrames)
        frames: MultiViewFrames = consume(tick.value)
        rig = tick.asof["rig"]
        assert isinstance(rig, RigCameras), "track needs the rig (a STATIC AsOf value)"
        views: int = len(frames.views)
        height, width = int(frames.rgb.shape[2]), int(frames.rgb.shape[3])
        if not self.states:
            self.rig: RigCameras = rig.select(frames.views).scaled_to(height, width)
            self.sigma_px, self.max_dist_px = resolve_epipolar_px(width, height)
            self.states = self.masks.new_states(views, (height, width))
            self.lost, self.held, self.areas = [{} for _ in range(views)], [{} for _ in range(views)], [{} for _ in range(views)]
        if self.last is not None:
            self.ticks_seen += 1  # forward ticks since the bootstrap (track_stride counts them)
        if self.last is None:
            tracks: list[dict[int, Track]] = self._bootstrap(frames)
        elif self.ticks_seen % self.params.track_stride != 0:
            tracks = self.last
        else:
            lost: bool = any(ticks >= LOST_TICKS_BEFORE_REPROMPT for view in self.lost for ticks in view.values())
            interval: int = self.params.redetect_interval_multi if len(self.bank.obj_ids) > 1 else self.params.redetect_interval
            routine: bool = interval > 0 and frames.frame_index % interval == 0
            prompts: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in range(views)]
            if lost or routine:
                prompts = self._redetect(frames, reprompt_alive=routine and self.params.reprompt_alive_on_routine)
            tracks = self._advance(frames, prompts)
        if self.bank.obj_ids:  # bootstrapped: the next tick tracks forward
            self.last = tracks
        rows: list[tuple[int, int]] = sorted((v, person) for v, view in enumerate(tracks) for person in view)
        picked: list[Track] = [tracks[v][person] for v, person in rows]
        stored_wh: Int64[ndarray, "v 2"] = rig.select(frames.views).resolution_wh
        value = MultiViewTracks(
            frames.frame_index, frames.views, np.asarray([r[0] for r in rows], dtype=np.int64), np.asarray([r[1] for r in rows], dtype=np.int64),
            torch.stack([t.mask for t in picked]) if picked else torch.zeros((0, height, width), dtype=torch.bool, device=frames.rgb.device),
            np.asarray([t.box_xyxy if t.box_xyxy is not None else [np.nan] * 4 for t in picked], dtype=np.float32).reshape(-1, 4),
            np.asarray([t.score for t in picked], dtype=np.float32), (stored_wh / np.array([width, height])).astype(np.float32),
        )  # fmt: skip
        out.put("tracks", tick.t_ns, Fenced(value, fence_after(value.masks)))

    def flush(self, out: Emit) -> None:
        pass

    def _fundamental(self, source: int, target: int) -> Float64[ndarray, "3 3"] | None:
        return fundamental_matrix(self.rig.intrinsics[source], self.rig.cam_T_world[source], self.rig.intrinsics[target], self.rig.cam_T_world[target])

    def _detect(self, frames_hwc: UInt8[Tensor, "v h w 3"]) -> tuple[list[BoxDetections], list[Float32[Tensor, "m 512"]]]:
        """Per view: the person boxes (frame-local, best first) and their CLIP features."""
        found: BoxDetections = self.detector(frames_hwc)
        per_view: list[BoxDetections] = []
        for v in range(frames_hwc.shape[0]):
            rows: Tensor = (found.frame_indices == v).nonzero().reshape(-1)
            per_view.append(BoxDetections(found.xyxy[rows], found.scores[rows], found.frame_indices[rows]))
        features: list[Float32[Tensor, "m 512"]] = [self.encoder(frames_hwc, dets) for dets in per_view]
        return per_view, features

    def _bootstrap(self, frames: MultiViewFrames) -> list[dict[int, Track]]:
        """Pick the view that sees the expected people best, prompt them there, and match them into the other views."""
        frames_hwc: UInt8[Tensor, "v h w 3"] = frames.rgb.permute(0, 2, 3, 1).contiguous()
        detections, features = self._detect(frames_hwc)
        expected: int | None = self.params.expected_subjects

        def quality(d: BoxDetections) -> float:
            score: float = d.num_detections * float(d.scores.mean()) if d.num_detections else 0.0
            return score * 1.5 if expected is not None and d.num_detections == expected else score

        ref: int = int(np.argmax([quality(d) for d in detections]))
        if detections[ref].num_detections == 0:
            return [{} for _ in frames.views]
        ref_boxes: Float32[ndarray, "m 4"] = detections[ref].xyxy_numpy()
        subjects: int = min(len(ref_boxes), expected) if expected is not None else len(ref_boxes)
        order: list[int] = [int(i) for i in np.argsort(-detections[ref].scores.cpu().numpy())[:subjects]]
        for obj, row in enumerate(order):
            self.bank.append(obj, features[ref][row])
        prompts: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in frames.views]
        prompts[ref] = {obj: ref_boxes[row] for obj, row in enumerate(order)}
        ref_centers: list[Float32[ndarray, "3"]] = [_center(ref_boxes[row]) for row in order]
        for v, dets in enumerate(detections):
            if v == ref or dets.num_detections == 0:
                continue
            boxes: Float32[ndarray, "m 4"] = dets.xyxy_numpy()
            clip: Float32[ndarray, "k m"] = self.bank.similarity(features[v])
            combined: Float32[ndarray, "k m"] = self._with_epipolar(clip, self._fundamental(ref, v), ref_centers, boxes)
            for row, col in assign_hungarian(combined, min_score=BOOTSTRAP_MIN_SCORE).items():
                prompts[v][row] = boxes[col]
                self.bank.append(row, features[v][col])
        return self._advance(frames, prompts)

    def _with_epipolar(
        self, clip: Float32[ndarray, "k m"], f_matrix: Float64[ndarray, "3 3"] | None, reference: Sequence[Float32[ndarray, "3"] | None], boxes: Float32[ndarray, "m 4"]
    ) -> Float32[ndarray, "k m"]:
        """CLIP similarity blended with each detection's distance to every subject's epipolar line from the reference view."""
        if f_matrix is None:
            return clip.copy()
        epipolar: Float32[ndarray, "k m"] = np.zeros_like(clip)
        for row, point in enumerate(reference):
            for col, box in enumerate(boxes):
                if point is not None:
                    epipolar[row, col] = epipolar_score(f_matrix, point, _center(box), self.sigma_px, self.max_dist_px)
        return (CLIP_WEIGHT * clip + EPIPOLAR_WEIGHT * epipolar).astype(np.float32)

    def _reference(self) -> tuple[int | None, dict[int, Float32[ndarray, "3"]]]:
        """With two or more subjects: the view where all are most clearly co-visible (largest smallest mask) and their mask centroids there."""
        objs: list[int] = self.bank.obj_ids
        if self.last is None or len(objs) < 2:
            return None, {}
        best, best_area = None, -1.0
        for v, tracks in enumerate(self.last):
            if all(obj in tracks for obj in objs) and (area := min(float(tracks[obj].mask.sum().item()) for obj in objs)) > best_area:
                best, best_area = v, area
        if best is None or best_area <= 0.0:
            return None, {}
        points: dict[int, Float32[ndarray, "3"]] = {}
        for obj in objs:
            ys, xs = torch.nonzero(self.last[best][obj].mask, as_tuple=True)
            points[obj] = np.array([float(xs.float().mean()), float(ys.float().mean()), 1.0], dtype=np.float32)
        return best, points

    def _redetect(self, frames: MultiViewFrames, reprompt_alive: bool) -> list[dict[int, Float32[ndarray, "4"]]]:
        """Sparse YOLO + CLIP + epipolar pass: refresh the banks; re-prompt lost tracks (and, on routine passes, confident live ones).

        Identity is tied across views geometrically (epipolar from the reference view's mask centroids, weight 0.65 over CLIP's
        0.35): CLIP alone flips two people in contact, whose crops look alike.
        """
        frames_hwc: UInt8[Tensor, "v h w 3"] = frames.rgb.permute(0, 2, 3, 1).contiguous()
        detections, features = self._detect(frames_hwc)
        prompts: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in frames.views]
        objs: list[int] = self.bank.obj_ids
        ref, ref_points = self._reference()
        for v, dets in enumerate(detections):
            if dets.num_detections == 0:
                continue
            lost: list[int] = [obj for obj, ticks in self.lost[v].items() if ticks >= LOST_TICKS_BEFORE_REPROMPT]
            boxes: Float32[ndarray, "m 4"] = dets.xyxy_numpy()
            clip: Float32[ndarray, "k m"] = self.bank.similarity(features[v])
            f_matrix: Float64[ndarray, "3 3"] | None = self._fundamental(ref, v) if ref is not None and v != ref else None
            combined: Float32[ndarray, "k m"] = self._with_epipolar(clip, f_matrix, [ref_points.get(obj) for obj in objs], boxes)
            matches: list[tuple[int, int]] = list(assign_hungarian(combined, min_score=BOOTSTRAP_MIN_SCORE).items())
            overlap: Float32[ndarray, "m m"] = box_iou(torch.from_numpy(boxes), torch.from_numpy(boxes)).numpy()
            dropped: set[int] = set()  # never prompt two subjects onto overlapping boxes: drop the weaker match (it coasts on memory)
            for a, (ra, ca) in enumerate(matches):
                for rb, cb in matches[a + 1 :]:
                    if overlap[ca, cb] >= ANCHOR_CONFLICT_IOU:
                        dropped.add(ra if combined[ra, ca] < combined[rb, cb] else rb)
            for row, col in matches:
                if row in dropped:
                    continue
                confident: bool = bool(clip[row, col] >= BANK_UPDATE_MIN_SCORE)
                if confident:
                    self.bank.append(objs[row], features[v][col])
                if objs[row] in lost or (reprompt_alive and confident):  # a weak box would drag a live mask off the person
                    prompts[v][objs[row]] = boxes[col]
        return prompts

    def _advance(self, frames: MultiViewFrames, prompts: list[dict[int, Float32[ndarray, "4"]]]) -> list[dict[int, Track]]:
        """One SAM2 step over every view, then the per-track bookkeeping (one device-to-host copy for all tracks)."""
        masks = self.masks(self.states, frames.frame_index, frames.rgb, prompts)
        entries: list[tuple[int, int, Bool[Tensor, "h w"]]] = [(v, obj, mask) for v, view in enumerate(masks) for obj, (mask, _) in view.items()]
        tracks: list[dict[int, Track]] = [{} for _ in frames.views]
        if not entries:
            return tracks
        stacked: Bool[Tensor, "k h w"] = torch.stack([mask for _, _, mask in entries])
        rows: Tensor = stacked.any(dim=2).int()
        cols: Tensor = stacked.any(dim=1).int()
        height, width = stacked.shape[1], stacked.shape[2]
        bounds: Tensor = torch.stack([rows.sum(dim=1), cols.argmax(dim=1), rows.argmax(dim=1), (width - 1) - cols.flip(1).argmax(dim=1), (height - 1) - rows.flip(1).argmax(dim=1)], dim=1)
        ious: Tensor = torch.cat([masks[v][obj][1] for v, obj, _ in entries])
        stats: Float32[ndarray, "k 7"] = torch.cat([bounds.float(), ious[:, None], stacked.flatten(1).sum(dim=1, keepdim=True).float()], dim=1).cpu().numpy()
        for (v, obj, mask), row in zip(entries, stats, strict=True):
            tracks[v][obj] = self._resolve(v, obj, mask, row)
        return tracks

    def _resolve(self, v: int, obj: int, mask: Bool[Tensor, "h w"], stats: Float32[ndarray, "7"]) -> Track:
        """Lost / held counters and the transient-collapse hold. ``stats`` = [rows hit, x_min, y_min, x_max, y_max, IoU, area px].

        The collapse test uses the pixel area: an under-segmented mask keeps its vertical extent while losing area.
        """
        nonempty, area_px, score = float(stats[0]) > 0.0, float(stats[6]), float(stats[5])
        history: list[float] = self.areas[v].setdefault(obj, [])
        previous: Track | None = self.last[v].get(obj) if self.last is not None else None
        collapsed: bool = (
            self.params.transient_hold and nonempty and bool(history) and area_px < self.params.transient_area_frac * float(np.median(history))
            and self.held[v].get(obj, 0) < LOST_TICKS_BEFORE_REPROMPT and previous is not None and previous.box_xyxy is not None
        )  # fmt: skip
        if collapsed:
            assert previous is not None
            self.held[v][obj] = self.held[v].get(obj, 0) + 1
            self.lost[v][obj] = self.lost[v].get(obj, 0) + 1  # arms the re-prompt; the collapsed area stays out of the history
            return Track(previous.mask, previous.box_xyxy, score)
        self.held[v][obj] = 0
        if not nonempty:
            self.lost[v][obj] = self.lost[v].get(obj, 0) + 1
            return Track(mask, None, score)
        self.lost[v][obj] = 0
        history.append(area_px)
        if len(history) > 7:
            history.pop(0)
        return Track(mask, stats[1:5].astype(np.float32), score)


def _center(box: Float32[ndarray, "4"]) -> Float32[ndarray, "3"]:
    """Homogeneous box centre ``[cx, cy, 1]`` (an epipolar reference point)."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, 1.0], dtype=np.float32)
