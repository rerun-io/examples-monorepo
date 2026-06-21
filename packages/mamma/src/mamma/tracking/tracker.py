"""Causal multiview person tracking: bootstrap once, then forward-only.

Per the realtime handoff design: YOLO + CLIP + epipolar identity run only at
bootstrap and on a sparse re-detect cadence; the dense per-tick work batches
``encode_image`` across all cameras, then runs the cheap memory/decoder via a
single B=n_cams ``batched_propagate`` (per-camera ``forward_embeddings`` on
prompt/re-detect ticks), with a ``SAM2ForgetfulObjectMemoryBank`` per camera so
memory stays bounded.

One predictor (one set of weights) serves all cameras; per-camera state lives
in ``SAM2GenericVideoPredictorState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray

from mamma.calibration.npz_contract import CameraCalibration
from mamma.engine.types import CameraTracks, TrackedObject
from mamma.tracking.detection import PersonDetections, PersonDetector, bbox_iou_xyxy
from mamma.tracking.identity import (
    ClipEncoder,
    FeatureBank,
    assign_hungarian,
    epipolar_score,
    fundamental_matrix,
    resolve_epipolar_px,
)

BOOTSTRAP_MIN_SCORE: float = 0.25
"""Minimum combined CLIP+epipolar score to accept a cross-camera bootstrap match."""
BANK_UPDATE_MIN_SCORE: float = 0.35
"""Minimum CLIP score to append a re-detect crop to the feature bank."""
LOST_TICKS_BEFORE_REPROMPT: int = 5
"""Consecutive empty-mask ticks before a camera is eligible for re-prompting."""
ANCHOR_CONFLICT_IOU: float = 0.65
"""Re-detect prompts for two different subjects whose boxes overlap by >= this
are de-conflicted (suppress the lower-scoring one) — mirrors the original's
``anchor_conflict_iou`` so SAM2 is never prompted with two near-identical boxes
for two people during contact (prevents per-object memory poisoning)."""


@dataclass(slots=True)
class TrackerConfig:
    """Streaming tracker configuration."""

    sam2_config: str = "configs/efficienttam/efficienttam_ti_512x512.yaml"
    """Hydra config name inside the vendored sam2 package. EfficientTAM-ti@512
    is the default (21.8ms/tick for 4 cams batched vs 52.6 for hiera-small);
    the golden gate passes with it (see implementation-notes)."""
    sam2_checkpoint: Path = Path("data/weights/efficienttam/efficienttam_ti.pt")
    """SAM2-family checkpoint path (EfficientTAM via build_sam2_generic*)."""
    yolo_checkpoint: Path = Path("data/weights/yolo/yolo12x.pt")
    """YOLO person detector checkpoint."""
    expected_subjects: int | None = None
    """Number of people to track; ``None`` infers from bootstrap detections."""
    redetect_interval: int = 120
    """Ticks between routine YOLO+CLIP re-detect passes (bank refresh) while
    tracking a SINGLE subject. A re-detect also fires immediately whenever any
    track has been lost for ``LOST_TICKS_BEFORE_REPROMPT`` ticks, so this can
    stay slow (~1s/pass). Presets lower this (~45) to anchor more often."""
    redetect_interval_multi: int = 15
    """Routine re-detect cadence used AUTOMATICALLY whenever >1 subject is being
    tracked at runtime (decided from the live track count, not a configured
    count — so it works on arbitrary video). Matches the original's
    ``min_anchor_frame_gap``: dense geometric (epipolar) re-anchoring corrects
    contact-moment identity swaps before triangulation mixes the subjects."""
    reprompt_alive_on_routine: bool = True
    """On routine re-detect passes, re-prompt confidently-matched tracks even
    when they are not lost — anchor refresh, parity with the original DAG's
    ~22 anchors/camera. Corrects slow mask drift (e.g. a head/shoulder mask
    sliding off over the clip) that never triggers the empty-mask lost path."""
    transient_hold: bool = True
    """Hold the previous mask for a tick when a non-empty mask's area collapses
    suddenly (a single-tick drop well below the rolling-median area) — bridges
    1-2 tick identity flips onto distractors (e.g. a mask jumping onto a cone)
    that the empty-mask lost path cannot catch, and arms a re-prompt."""
    transient_area_frac: float = 0.45
    """Single-tick pixel area as a fraction of the rolling-median healthy area
    below which a non-empty mask is treated as a CATASTROPHIC collapse and the
    previous mask held. Tuned to catch identity flips onto small distractors
    (a mask jumping onto a cone collapses to <0.3x area — the previous on-person
    mask beats the garbage) WITHOUT firing on moderate fast-motion
    undersegmentation (~0.57x, e.g. f252 jump apex): there the previous mask is
    spatially stale and holding it would HURT (the person moved a body-width in
    one tick), so the native undersegmented mask is kept."""
    memory_window_size: int = 7
    """Sliding window of non-conditional SAM2 memories kept per object."""
    track_stride: int = 1
    """Run the mask tracker every Nth tick; skipped ticks reuse the previous
    masks. Default 1 (Pablo, 2026-06-10): stride 4 starves EfficientTAM's
    memory during fast motion and masks transiently collapse to the head
    (displayed-mask IoU vs per-frame baseline bottoms at 0.17); stride 1 is
    pixel-identical (0.999 IoU) at ~15.0 s vs 11.76 s wall for the 12.1 s clip.
    Set 4 to prioritize the realtime gate over mask cosmetics."""
    device: str = "cuda"
    """Compute device."""


class MultiViewTracker:
    """Bootstrap-then-forward-only person tracker across synchronized cameras."""

    def __init__(self, cameras: list[CameraCalibration], config: TrackerConfig) -> None:
        from sam2.build_sam import build_sam2_generic_video_predictor
        from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
        from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

        self.config: TrackerConfig = config
        self.cameras: list[CameraCalibration] = cameras
        self.device: str = config.device
        self._video_hw: tuple[int, int] = (cameras[0].height, cameras[0].width)

        self.predictor = build_sam2_generic_video_predictor(
            config.sam2_config, str(config.sam2_checkpoint), device=config.device
        )
        self._states = [
            SAM2GenericVideoPredictorState.create(
                video_hw=self._video_hw,
                memory_bank=SAM2ForgetfulObjectMemoryBank(
                    memory_temporal_stride=1,
                    memory_window_size=config.memory_window_size,
                    storage_device=torch.device(config.device),
                ),
            )
            for _ in cameras
        ]
        self.detector: PersonDetector = PersonDetector(config.yolo_checkpoint, device=config.device)
        self.encoder: ClipEncoder = ClipEncoder(device=config.device)
        self.bank: FeatureBank = FeatureBank()
        self.bootstrapped: bool = False
        self._lost_ticks: list[dict[int, int]] = [{} for _ in cameras]
        self._area_hist: list[dict[int, list[float]]] = [{} for _ in cameras]
        """Rolling window of recent healthy mask areas per camera/object (for
        transient-collapse detection); collapsed/empty ticks are not appended."""
        self._held_ticks: list[dict[int, int]] = [{} for _ in cameras]
        """Consecutive ticks the previous mask has been held for an object."""
        self._last_tracks: list[CameraTracks] | None = None
        self._ticks_seen: int = 0
        # Per camera-pair geometry for epipolar identity transfer.
        self._sigma_px: float
        self._max_dist_px: float
        self._sigma_px, self._max_dist_px = resolve_epipolar_px(cameras[0].width, cameras[0].height)

    def _fundamental(self, source_idx: int, target_idx: int) -> Float64[ndarray, "3 3"] | None:
        src: CameraCalibration = self.cameras[source_idx]
        tgt: CameraCalibration = self.cameras[target_idx]
        return fundamental_matrix(src.k_matrix, src.world_to_cam, tgt.k_matrix, tgt.world_to_cam)

    # ── Bootstrap ───────────────────────────────────────────────────────────

    def _try_bootstrap(self, frame_idx: int, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> list[CameraTracks]:
        detections: list[PersonDetections] = [self.detector.detect(f) for f in frames]
        expected: int | None = self.config.expected_subjects

        def cam_quality(d: PersonDetections) -> float:
            if len(d) == 0:
                return 0.0
            quality: float = len(d) * float(d.scores.mean())
            if expected is not None and len(d) == expected:
                quality *= 1.5
            return quality

        ref_idx: int = int(np.argmax([cam_quality(d) for d in detections]))
        ref: PersonDetections = detections[ref_idx]
        if len(ref) == 0:
            return [{} for _ in self.cameras]

        n_subjects: int = min(len(ref), expected) if expected is not None else len(ref)
        ref_order: list[int] = [int(i) for i in np.argsort(-ref.scores)[:n_subjects]]
        obj_ids: list[int] = list(range(n_subjects))

        ref_feats: Float32[torch.Tensor, "n 512"] = self.encoder.encode([ref.crops[i] for i in ref_order])
        for obj_id, row in zip(obj_ids, range(len(ref_order)), strict=True):
            self.bank.append(obj_id, ref_feats[row])
        ref_centers: Float32[ndarray, "n 3"] = ref.centers_xy1[ref_order]

        prompts_per_cam: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in self.cameras]
        for obj_id, det_row in zip(obj_ids, ref_order, strict=True):
            prompts_per_cam[ref_idx][obj_id] = ref.boxes_xyxy[det_row]

        for cam_idx, dets in enumerate(detections):
            if cam_idx == ref_idx or len(dets) == 0:
                continue
            det_feats: Float32[torch.Tensor, "m 512"] = self.encoder.encode(dets.crops)
            s_clip: Float32[ndarray, "k m"] = self.bank.similarity(det_feats)
            f_matrix: Float64[ndarray, "3 3"] | None = self._fundamental(ref_idx, cam_idx)
            combined: Float32[ndarray, "k m"] = s_clip.copy()
            if f_matrix is not None:
                s_epi: Float32[ndarray, "k m"] = np.zeros_like(s_clip)
                for row in range(n_subjects):
                    for col, center in enumerate(dets.centers_xy1):
                        s_epi[row, col] = epipolar_score(
                            f_matrix, ref_centers[row], center, self._sigma_px, self._max_dist_px
                        )
                combined = (0.35 * s_clip + 0.65 * s_epi).astype(np.float32)
            matches: dict[int, int] = assign_hungarian(combined, min_score=BOOTSTRAP_MIN_SCORE)
            for row, col in matches.items():
                obj_id = obj_ids[row]
                prompts_per_cam[cam_idx][obj_id] = dets.boxes_xyxy[col]
                self.bank.append(obj_id, det_feats[col])

        self.bootstrapped = True
        return self._forward_all(frame_idx, frames, prompts_per_cam)

    # ── Steady state ───────────────────────────────────────────────────────

    def step(self, frame_idx: int, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> list[CameraTracks]:
        """Process one synchronized tick; returns per-camera tracks."""
        if not self.bootstrapped:
            tracks = self._try_bootstrap(frame_idx, frames)
            if self.bootstrapped:
                self._last_tracks = tracks
            return tracks

        self._ticks_seen += 1
        if self.config.track_stride > 1 and self._ticks_seen % self.config.track_stride != 0 and self._last_tracks is not None:
            return self._last_tracks

        prompts_per_cam: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in self.cameras]
        any_lost: bool = any(
            ticks >= LOST_TICKS_BEFORE_REPROMPT for cam in self._lost_ticks for ticks in cam.values()
        )
        # Adaptive cadence: re-anchor faster when tracking multiple subjects
        # (decided from the LIVE track count, so unknown-count / arbitrary video
        # auto-adapts — no configured subject count needed).
        interval: int = self.config.redetect_interval_multi if len(self.bank.obj_ids) > 1 else self.config.redetect_interval
        routine: bool = interval > 0 and frame_idx % interval == 0
        if any_lost or routine:
            prompts_per_cam = self._redetect(frames, reprompt_alive=routine and self.config.reprompt_alive_on_routine)
        tracks = self._forward_all(frame_idx, frames, prompts_per_cam)
        self._last_tracks = tracks
        return tracks

    def _reference_points(self) -> tuple[int | None, dict[int, Float32[ndarray, "3"]]]:
        """Pick a reference camera + each subject's epipolar reference point.

        Mirrors the bootstrap's cross-camera geometric tie for the steady-state
        re-detect: the reference point is each subject's CURRENT SAM2 mask
        centroid in the camera where all subjects are most clearly co-visible
        (largest min mask area). Single-subject (or no prior tracks) -> no tie.
        """
        obj_ids: list[int] = self.bank.obj_ids
        if self._last_tracks is None or len(obj_ids) < 2:
            return None, {}
        best_idx: int | None = None
        best_min_area: float = -1.0
        for cam_idx, tracks in enumerate(self._last_tracks):
            if not all(oid in tracks for oid in obj_ids):
                continue
            min_area: float = min(float(tracks[oid].mask.sum().item()) for oid in obj_ids)
            if min_area > best_min_area:
                best_min_area, best_idx = min_area, cam_idx
        if best_idx is None or best_min_area <= 0.0:
            return None, {}
        ref_points: dict[int, Float32[ndarray, "3"]] = {}
        for oid in obj_ids:
            ys, xs = torch.nonzero(self._last_tracks[best_idx][oid].mask, as_tuple=True)
            ref_points[oid] = np.array([float(xs.float().mean()), float(ys.float().mean()), 1.0], dtype=np.float32)
        return best_idx, ref_points

    def _redetect(
        self, frames: list[UInt8[torch.Tensor, "3 h w"]], reprompt_alive: bool = False
    ) -> list[dict[int, Float32[ndarray, "4"]]]:
        """Sparse YOLO + CLIP + epipolar pass: refresh feature banks, re-prompt.

        Always re-prompts lost tracks. When ``reprompt_alive`` (a routine anchor
        pass), also re-prompts confidently-matched live tracks. Identity is tied
        across cameras geometrically (epipolar from the reference camera's mask
        centroids, weight 0.65 > CLIP 0.35) — CLIP alone flips obj0<->obj1 in one
        camera during contact since the two crops look near-identical, which our
        earlier per-camera-independent CLIP-only re-detect could not catch.
        """
        prompts_per_cam: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in self.cameras]
        obj_ids: list[int] = self.bank.obj_ids
        ref_idx: int | None
        ref_points: dict[int, Float32[ndarray, "3"]]
        ref_idx, ref_points = self._reference_points()
        for cam_idx, frame in enumerate(frames):
            lost: list[int] = [
                obj_id for obj_id, ticks in self._lost_ticks[cam_idx].items() if ticks >= LOST_TICKS_BEFORE_REPROMPT
            ]
            dets: PersonDetections = self.detector.detect(frame)
            if len(dets) == 0:
                continue
            det_feats: Float32[torch.Tensor, "m 512"] = self.encoder.encode(dets.crops)
            s_clip: Float32[ndarray, "k m"] = self.bank.similarity(det_feats)
            combined: Float32[ndarray, "k m"] = s_clip.copy()
            # Cross-camera geometric tie: score detections in this camera against
            # each subject's epipolar line from the reference camera's centroid.
            f_matrix: Float64[ndarray, "3 3"] | None = (
                self._fundamental(ref_idx, cam_idx) if ref_idx is not None and cam_idx != ref_idx else None
            )
            if f_matrix is not None:
                s_epi: Float32[ndarray, "k m"] = np.zeros_like(s_clip)
                for row, obj_id in enumerate(obj_ids):
                    ref_pt: Float32[ndarray, "3"] | None = ref_points.get(obj_id)
                    if ref_pt is None:
                        continue
                    for col, center in enumerate(dets.centers_xy1):
                        s_epi[row, col] = epipolar_score(f_matrix, ref_pt, center, self._sigma_px, self._max_dist_px)
                combined = (0.35 * s_clip + 0.65 * s_epi).astype(np.float32)
            matches: dict[int, int] = assign_hungarian(combined, min_score=BOOTSTRAP_MIN_SCORE)
            # Conflict-IoU guard: never prompt two subjects onto overlapping boxes
            # (drop the lower combined-score one; it coasts on prior SAM2 memory).
            matched: list[tuple[int, int]] = list(matches.items())
            dropped: set[int] = set()
            for a in range(len(matched)):
                for b in range(a + 1, len(matched)):
                    ra, ca = matched[a]
                    rb, cb = matched[b]
                    if bbox_iou_xyxy(dets.boxes_xyxy[ca], dets.boxes_xyxy[cb]) >= ANCHOR_CONFLICT_IOU:
                        dropped.add(ra if combined[ra, ca] < combined[rb, cb] else rb)
            for row, col in matches.items():
                if row in dropped:
                    continue
                obj_id: int = obj_ids[row]
                confident: bool = bool(s_clip[row, col] >= BANK_UPDATE_MIN_SCORE)
                if confident:
                    self.bank.append(obj_id, det_feats[col])
                # Lost tracks always re-prompt; live tracks only on a confident
                # match during a routine anchor pass (a weak box would drag the
                # mask off the person).
                if obj_id in lost or (reprompt_alive and confident):
                    prompts_per_cam[cam_idx][obj_id] = dets.boxes_xyxy[col]
        return prompts_per_cam

    def _resolve_track(
        self,
        cam_idx: int,
        obj_id: int,
        mask: Bool[torch.Tensor, "h w"],
        stat_row: Float32[ndarray, "7"],
    ) -> TrackedObject:
        """Finalize one object's track: bookkeeping + transient-collapse hold.

        ``stat_row`` is ``[bbox_h_rows, x_min, y_min, x_max, y_max, pred_iou,
        area_px]`` from the GPU stats (already host-copied). Runs no CUDA sync.
        The collapse test uses the true pixel area (index 6), not the bbox row
        extent (index 0) — an undersegmented mask keeps its vertical extent
        while losing area, so only the pixel count catches it. On a sudden
        single-tick area collapse it returns the previous tick's mask/bbox and
        arms a re-prompt; otherwise it updates the rolling-area history and the
        lost-tick counter exactly as before.
        """
        nonempty: bool = float(stat_row[0]) > 0.0
        area_px: float = float(stat_row[6])
        score: float = float(stat_row[5])
        hist: list[float] = self._area_hist[cam_idx].setdefault(obj_id, [])
        median_area: float = float(np.median(hist)) if hist else 0.0
        prev: TrackedObject | None = (
            self._last_tracks[cam_idx].get(obj_id) if self._last_tracks is not None else None
        )
        collapsed: bool = (
            self.config.transient_hold
            and nonempty
            and median_area > 0.0
            and area_px < self.config.transient_area_frac * median_area
            and self._held_ticks[cam_idx].get(obj_id, 0) < LOST_TICKS_BEFORE_REPROMPT
            and prev is not None
            and prev.bbox_xyxy is not None
        )
        if collapsed:
            assert prev is not None and prev.bbox_xyxy is not None
            # Hold the previous good mask; arm the lost-track re-prompt without
            # appending the collapsed area to the healthy-area history.
            self._held_ticks[cam_idx][obj_id] = self._held_ticks[cam_idx].get(obj_id, 0) + 1
            self._lost_ticks[cam_idx][obj_id] = self._lost_ticks[cam_idx].get(obj_id, 0) + 1
            return TrackedObject(obj_id=obj_id, mask=prev.mask, bbox_xyxy=prev.bbox_xyxy, score=score)
        self._held_ticks[cam_idx][obj_id] = 0
        bbox: Float32[ndarray, "4"] | None = None
        if nonempty:
            bbox = stat_row[1:5].astype(np.float32)
            self._lost_ticks[cam_idx][obj_id] = 0
            hist.append(area_px)
            if len(hist) > 7:
                hist.pop(0)
        else:
            self._lost_ticks[cam_idx][obj_id] = self._lost_ticks[cam_idx].get(obj_id, 0) + 1
        return TrackedObject(obj_id=obj_id, mask=mask, bbox_xyxy=bbox, score=score)

    def _tracks_from_batched(self, batched, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> list[CameraTracks]:
        """Convert a B=n_cams SAM2Result into per-camera tracks (sync-free)."""
        results: list[CameraTracks] = []
        h: int = frames[0].shape[1]
        w: int = frames[0].shape[2]
        masks: Bool[torch.Tensor, "c h w"] = batched.masks_logits[:, 0] > 0.0
        rows: torch.Tensor = masks.any(dim=2).int()
        cols: torch.Tensor = masks.any(dim=1).int()
        stats: torch.Tensor = torch.stack(
            [
                rows.sum(dim=1),
                cols.argmax(dim=1),
                rows.argmax(dim=1),
                (w - 1) - cols.flip(1).argmax(dim=1),
                (h - 1) - rows.flip(1).argmax(dim=1),
            ],
            dim=1,
        ).float()
        area_px: torch.Tensor = masks.reshape(masks.shape[0], -1).sum(dim=1, keepdim=True).float()
        stats = torch.cat([stats, batched.ious[:, 0:1].float(), area_px], dim=1)
        all_stats: Float32[ndarray, "c 7"] = stats.cpu().numpy()
        for cam_idx in range(len(frames)):
            results.append({0: self._resolve_track(cam_idx, 0, masks[cam_idx], all_stats[cam_idx])})
        return results

    def _forward_all(
        self,
        frame_idx: int,
        frames: list[UInt8[torch.Tensor, "3 h w"]],
        prompts_per_cam: list[dict[int, Float32[ndarray, "4"]]],
    ) -> list[CameraTracks]:
        from sam2.modeling.sam2_prompt import SAM2Prompt

        # Batch the (dominant) image-encoder cost across cameras, then run the
        # cheap per-camera memory/decoder via forward_embeddings. One autocast/
        # inference_mode context for the whole tick (entering per camera costs
        # measurable CPU at this rate).
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            batch: UInt8[torch.Tensor, "c 3 h w"] = torch.stack(frames, dim=0)
            # The fork annotates encode_image as Tensor pairs, but it returns
            # multi-level lists (one tensor per FPN level) at runtime.
            encoded = cast("tuple[list[torch.Tensor], list[torch.Tensor]]", self.predictor.encode_image(batch))
            embeddings: list[torch.Tensor] = encoded[0]
            pos_embeddings: list[torch.Tensor] = encoded[1]

        # Steady-state fast path: one B=n_cams propagation (memory attention,
        # mask decode, memory encode batched across cameras) instead of the
        # fork's per-camera python loop. Falls back below on prompts/misalign.
        if not any(prompts_per_cam):
            from mamma.tracking.batched_forward import batched_propagate

            with torch.inference_mode(), autocast:
                batched = batched_propagate(
                    self.predictor, self._states, frame_idx, embeddings, pos_embeddings, self._video_hw
                )
            if batched is not None:
                return self._tracks_from_batched(batched, frames)

        results: list[CameraTracks] = []
        for cam_idx in range(len(frames)):
            prompts: list[SAM2Prompt] = [
                SAM2Prompt(obj_id=obj_id, boxes=torch.as_tensor(box, device=self.device).reshape(1, 4))
                for obj_id, box in prompts_per_cam[cam_idx].items()
            ]
            state = self._states[cam_idx]
            if not prompts and not state.memory_bank.known_obj_ids:
                results.append({})
                continue
            cam_embeddings = [level[cam_idx : cam_idx + 1] for level in embeddings]
            cam_pos = [level[cam_idx : cam_idx + 1] for level in pos_embeddings]
            with torch.inference_mode(), autocast:
                # Single-mask output: with multimask the predicted-IoU argmax
                # intermittently selects the whole-scene candidate during
                # propagation (observed as full-frame masks on crossing_arms).
                raw: dict = self.predictor.forward_embeddings(
                    state, frame_idx, cam_embeddings, cam_pos, prompts=prompts, multimask_output=False
                )
            # Stay sync-free inside the loop: gather per-object stats on GPU and
            # do ONE device->host copy per camera at the end (each .item()/
            # .tolist() in the loop would stall the pipeline ~1-2 ms).
            entries: list[tuple[int, Bool[torch.Tensor, "h w"]]] = []
            stats_rows: list[torch.Tensor] = []
            h: int = frames[cam_idx].shape[1]
            w: int = frames[cam_idx].shape[2]
            for obj_id, result in raw.items():
                # multimask_output=False -> exactly one mask candidate.
                mask: Bool[torch.Tensor, "h w"] = result.masks_logits[0, 0] > 0.0
                rows: torch.Tensor = mask.any(dim=1).int()
                cols: torch.Tensor = mask.any(dim=0).int()
                stats: torch.Tensor = torch.stack(
                    [
                        rows.sum(),
                        cols.argmax(),  # x_min (0 when empty)
                        rows.argmax(),  # y_min
                        (w - 1) - cols.flip(0).argmax(),  # x_max
                        (h - 1) - rows.flip(0).argmax(),  # y_max
                    ]
                ).float()
                stats = torch.cat([stats, result.ious[0, 0:1].float(), mask.sum().reshape(1).float()])
                entries.append((obj_id, mask))
                stats_rows.append(stats)
            tracks: CameraTracks = {}
            if entries:
                all_stats: Float32[ndarray, "k 7"] = torch.stack(stats_rows).cpu().numpy()
                for (obj_id, mask), row in zip(entries, all_stats, strict=True):
                    tracks[obj_id] = self._resolve_track(cam_idx, obj_id, mask, row)
            results.append(tracks)
        return results
