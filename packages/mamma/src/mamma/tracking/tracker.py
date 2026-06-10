"""Causal multiview person tracking: bootstrap once, then forward-only.

Per the realtime handoff design: YOLO + CLIP + epipolar identity run only at
bootstrap and on a sparse re-detect cadence; the dense per-tick work is the
streaming SAM2 fork (``SAM2GenericVideoPredictor.forward`` one frame at a time)
with a ``SAM2ForgetfulObjectMemoryBank`` per camera so memory stays bounded.

One predictor (one set of weights) serves all cameras; per-camera state lives
in ``SAM2GenericVideoPredictorState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray

from mamma.calibration.npz_contract import CameraCalibration
from mamma.engine.types import CameraTracks, TrackedObject
from mamma.tracking.detection import PersonDetections, PersonDetector
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
    """Ticks between routine YOLO+CLIP re-detect passes (bank refresh). A
    re-detect also fires immediately whenever any track has been lost for
    ``LOST_TICKS_BEFORE_REPROMPT`` ticks, so this can stay slow (~1s/pass)."""
    memory_window_size: int = 7
    """Sliding window of non-conditional SAM2 memories kept per object."""
    track_stride: int = 3
    """Run the mask tracker every Nth tick; skipped ticks reuse the previous
    masks (person silhouettes move a few px/frame at 30 fps — landmark crops
    tolerate a one-frame-old mask; verified by the golden gate)."""
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
        routine: bool = self.config.redetect_interval > 0 and frame_idx % self.config.redetect_interval == 0
        if any_lost or routine:
            prompts_per_cam = self._redetect(frames)
        tracks = self._forward_all(frame_idx, frames, prompts_per_cam)
        self._last_tracks = tracks
        return tracks

    def _redetect(self, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> list[dict[int, Float32[ndarray, "4"]]]:
        """Sparse YOLO+CLIP pass: refresh feature banks, re-prompt lost tracks."""
        prompts_per_cam: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in self.cameras]
        obj_ids: list[int] = self.bank.obj_ids
        for cam_idx, frame in enumerate(frames):
            lost: list[int] = [
                obj_id for obj_id, ticks in self._lost_ticks[cam_idx].items() if ticks >= LOST_TICKS_BEFORE_REPROMPT
            ]
            dets: PersonDetections = self.detector.detect(frame)
            if len(dets) == 0:
                continue
            det_feats: Float32[torch.Tensor, "m 512"] = self.encoder.encode(dets.crops)
            s_clip: Float32[ndarray, "k m"] = self.bank.similarity(det_feats)
            matches: dict[int, int] = assign_hungarian(s_clip, min_score=BOOTSTRAP_MIN_SCORE)
            for row, col in matches.items():
                obj_id: int = obj_ids[row]
                if s_clip[row, col] >= BANK_UPDATE_MIN_SCORE:
                    self.bank.append(obj_id, det_feats[col])
                if obj_id in lost:
                    prompts_per_cam[cam_idx][obj_id] = dets.boxes_xyxy[col]
        return prompts_per_cam

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
            embeddings, pos_embeddings = self.predictor.encode_image(batch)

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
                stats = torch.cat([stats, result.ious[0, 0:1].float()])
                entries.append((obj_id, mask))
                stats_rows.append(stats)
            tracks: CameraTracks = {}
            if entries:
                all_stats: Float32[ndarray, "k 6"] = torch.stack(stats_rows).cpu().numpy()
                for (obj_id, mask), row in zip(entries, all_stats, strict=True):
                    bbox: Float32[ndarray, "4"] | None = None
                    if row[0] > 0:
                        bbox = row[1:5].astype(np.float32)
                        self._lost_ticks[cam_idx][obj_id] = 0
                    else:
                        self._lost_ticks[cam_idx][obj_id] = self._lost_ticks[cam_idx].get(obj_id, 0) + 1
                    tracks[obj_id] = TrackedObject(obj_id=obj_id, mask=mask, bbox_xyxy=bbox, score=float(row[5]))
            results.append(tracks)
        return results
