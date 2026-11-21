"""Causal multiview person tracking: bootstrap once, then forward-only.

Per the realtime handoff design: YOLO + CLIP + epipolar identity run only at
bootstrap and on a sparse re-detect cadence; a steady tick (no prompts) is one
``forward_batched`` call of the sam2-streaming predictor (every camera and object
batched per module, CUDA graphs per input signature); prompt/re-detect ticks batch
``encode_image`` across cameras and run per-camera ``forward_embeddings``. Each
camera has a ``SAM2ForgetfulObjectMemoryBank``, so memory stays bounded.

One predictor (one set of weights) serves all cameras; per-camera state lives
in ``SAM2GenericVideoPredictorState``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from posekit.models import ClipIdentityConfig, DetectorConfig, RtDetrDetectorConfig
from sam2.cuda_graphs import GraphCache
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.modeling.sam2_result import SAM2Result
from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

from mamma.calibration.npz_contract import CameraCalibration
from mamma.engine.types import CameraTracks, TrackedObject
from mamma.tracking.detection import PersonDetections, bbox_iou_xyxy
from mamma.tracking.identity import (
    CLIP_WEIGHT,
    EPIPOLAR_WEIGHT,
    FeatureBank,
    assign_hungarian,
    epipolar_score,
    fundamental_matrix,
    order_bootstrap_candidates,
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
CAPTURE_TICKS_PER_PROMPT: int = 12
"""Propagated ticks after each prompt of the synthetic stream that captures the
steady tick's CUDA graphs at construction (the memory window fills in ~6)."""


def _disable_dead_hole_filling(predictor: object) -> None:
    """Turn off SAM2 mask hole/sprinkle filling when its CUDA extension is absent.

    ``build_sam2_generic_video_predictor`` sets ``max_hole_area=8``, but the
    filling needs the compiled ``sam2._C`` connected-components kernel. Without
    it, ``postprocess_masks`` launches the threshold ops, raises inside a
    try/except, emits a warning, and returns the input masks unchanged — every
    tick. Zeroing the areas skips that dead work and the exception and gives
    byte-identical masks. With ``_C`` present, the filling stays enabled.
    """
    import importlib.util

    if importlib.util.find_spec("sam2._C") is not None:
        return
    transforms = predictor._transforms  # type: ignore[attr-defined]
    transforms.max_hole_area = 0.0
    transforms.max_sprinkle_area = 0.0


@dataclass(slots=True)
class TrackerConfig:
    """Streaming tracker configuration."""

    sam2_config: str = "configs/efficienttam/efficienttam_ti_512x512.yaml"
    """Hydra config name inside the vendored sam2 package. EfficientTAM-ti@512
    is the default (21.8ms/tick for 4 cams batched vs 52.6 for hiera-small);
    the golden gate passes with it (see implementation-notes)."""
    sam2_checkpoint: Path = Path("data/weights/efficienttam/efficienttam_ti.pt")
    """SAM2-family checkpoint path (EfficientTAM via build_sam2_generic*)."""
    detector: DetectorConfig = field(default_factory=RtDetrDetectorConfig)
    """posekit person detector for the bootstrap and re-detects (``posekit.models.DetectorConfig``: ``rtdetr``,
    ``yolox``, ``yolo`` with ``--tracker.detector.weights`` for YOLOv12-X). RT-DETRv2-R50 by default (Pablo,
    2026-11-21): same golden/dynamic as YOLOv12-X, best on the bystander case (MammaEval CatchBall), re-detect
    ~20 ms for 4 cameras vs ~45 ms. The detector config carries its own device."""
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
    cuda_graph: bool = True
    """Replay the steady tick's modules (image encoder, memory attention per
    memory layout, mask decoder, memory encoder) from CUDA graphs instead of
    ~1200 eager launches; outputs identical to the eager path. Graphs are
    captured at construction only (one object per camera); prompt and
    re-detect ticks, and signatures not captured then, stay eager."""
    cuda_graph_max_conditional: int = 4
    """Prompted (conditional) memories each object attends to, the most recent (SAM 2 ``max_cond_frames_in_attn``;
    the bank stores only those), and graphs are captured at construction for 1..N of them: one per prompt tick
    (bootstrap + each re-prompt), so a long run keeps replaying graphs after any number of re-prompts."""
    fused_rope: bool = True
    """Rotary position encoding in one Triton kernel per q/k (memory attention
    12.3 -> 8.1 ms GPU per batched tick, identical output) instead of five
    fp32/complex passes. Process-wide switch for the vendored sam2 fork;
    ignored when Triton is missing."""
    compile_image_encoder: bool = False
    """torch.compile (inductor, static shapes) the SAM2 image encoder: fuses its
    window-partition copies, casts and norms (8.4 -> 5.1 ms GPU per B=4
    encode) at ~10 s first-build compile. Not bit-identical: inductor skips
    intermediate bf16 rounding (feature mean |diff| ~1.5% of mean |x|)."""
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
        _disable_dead_hole_filling(self.predictor)
        # Attend to the cuda_graph_max_conditional most recent prompted memories (the bank stores only those): each
        # routine re-prompt adds one, and unbounded the steady step's signature grew past the captured graphs after
        # ~4 re-prompts (~16 s single subject, ~2 s with several), every later tick eager and slower.
        self.predictor.max_cond_frames_in_attn = config.cuda_graph_max_conditional
        if config.compile_image_encoder:
            encoder = self.predictor.image_encoder
            encoder.forward = torch.compile(encoder.forward, dynamic=False)
        if config.fused_rope:
            from sam2.modeling import fused_rope

            fused_rope.ENABLED = fused_rope.triton is not None
        self._states = [
            SAM2GenericVideoPredictorState.create(
                video_hw=self._video_hw,
                memory_bank=SAM2ForgetfulObjectMemoryBank(
                    memory_temporal_stride=1,
                    memory_window_size=config.memory_window_size,
                    storage_device=torch.device(config.device),
                    max_conditional_memories=config.cuda_graph_max_conditional,
                ),
            )
            for _ in cameras
        ]
        self._graphs: GraphCache | None = None
        if config.cuda_graph:
            # Captured now, before the pipeline starts its worker threads: a capture cannot overlap the fit
            # worker, CUDA rejects that thread's torch.cuda.synchronize() while any stream captures (in every
            # capture mode) and the error invalidates the capture too. Later signatures stay eager.
            self._graphs = GraphCache()
            self._capture_steady_graphs()
            self._graphs.capture = False
        self.detector = config.detector.setup()
        self.encoder = ClipIdentityConfig(device=config.device).setup()
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

    def _capture_steady_graphs(self) -> None:
        """Run a synthetic stream through ``forward_batched`` so the steady tick's graphs are captured.

        Copies of the empty memory banks see a prompt on blank frames, then
        ``CAPTURE_TICKS_PER_PROMPT`` propagated ticks, ``cuda_graph_max_conditional``
        times, at the tracker's ``track_stride``: the banks' own selection rules
        (window, pruning, pointers) decide which memory layouts a single-subject
        stream visits. A layout seen twice is captured; the ramp after each prompt,
        seen once, is left to the eager path.
        """

        h, w = self._video_hw
        states = [SAM2GenericVideoPredictorState.create(self._video_hw, copy.deepcopy(s.memory_bank)) for s in self._states]
        frames: UInt8[torch.Tensor, "c 3 h w"] = torch.zeros((len(states), 3, h, w), dtype=torch.uint8, device=self.device)
        box: torch.Tensor = torch.tensor([[0.0, 0.0, w / 2, h / 2]], device=self.device)
        frame_idx: int = 0
        for _ in range(self.config.cuda_graph_max_conditional):
            for tick in range(CAPTURE_TICKS_PER_PROMPT + 1):
                prompts = [[SAM2Prompt(obj_id=0, boxes=box)] if tick == 0 else [] for _ in states]
                self._forward_batched(states, frame_idx, frames, prompts)
                frame_idx += self.config.track_stride
        torch.cuda.synchronize()

    def _forward_batched(
        self,
        states: list[SAM2GenericVideoPredictorState],
        frame_idx: int,
        frames: UInt8[torch.Tensor, "c 3 h w"],
        prompts: list[list[SAM2Prompt]],
    ) -> list[dict[int, SAM2Result]]:
        """One ``forward_batched`` step over every camera, with every mask hypothesis (tracks read hypothesis 0); a
        steady step is one CUDA graph replay where captured."""
        # No autocast cast cache with graphs: a graph must not read a cached cast that the region's exit frees.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=self._graphs is None):
            return self.predictor.forward_batched(states, frame_idx, frames, prompts, [False] * len(states), graphs=self._graphs, best_only=False, one_graph=True)

    def _fundamental(self, source_idx: int, target_idx: int) -> Float64[ndarray, "3 3"] | None:
        src: CameraCalibration = self.cameras[source_idx]
        tgt: CameraCalibration = self.cameras[target_idx]
        return fundamental_matrix(src.k_matrix, src.world_to_cam, tgt.k_matrix, tgt.world_to_cam)

    # ── Bootstrap ───────────────────────────────────────────────────────────

    def _detect(self, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> list[PersonDetections]:
        """Person boxes and CLIP features in every camera: one detector call and one CLIP call for all cameras."""
        frames_rgb: UInt8[torch.Tensor, "c h w 3"] = torch.stack(frames).permute(0, 2, 3, 1).contiguous()
        boxes = self.detector(frames_rgb)
        features: Float32[torch.Tensor, "n d"] = self.encoder(frames_rgb, boxes).cpu()
        xyxy: Float32[ndarray, "n 4"] = boxes.xyxy_numpy()
        scores: Float32[ndarray, "n"] = boxes.scores.cpu().numpy()
        cams: ndarray = boxes.frame_indices.cpu().numpy()
        detections: list[PersonDetections] = []
        for cam_idx in range(len(frames)):
            rows: ndarray = np.flatnonzero(cams == cam_idx)
            rows = rows[np.argsort(-scores[rows], kind="stable")]
            detections.append(PersonDetections(boxes_xyxy=xyxy[rows], scores=scores[rows], features=features[torch.as_tensor(rows, dtype=torch.long)]))
        return detections

    def _try_bootstrap(self, frame_idx: int, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> list[CameraTracks]:
        detections: list[PersonDetections] = self._detect(frames)
        expected: int | None = self.config.expected_subjects

        def cam_quality(d: PersonDetections) -> float:
            if len(d) == 0:
                return 0.0
            quality: float = len(d) * float(d.scores.mean())
            if expected is not None and len(d) == expected:
                quality *= 1.5
            return quality

        ref_idx: int = int(np.argmax([cam_quality(d) for d in detections]))
        if expected is not None and max(len(d) for d in detections) > expected:
            # Bystanders in view (operators, a ball thrower on MammaEval): take
            # the camera that sees the most people so every candidate can be
            # ranked by where it stands, instead of trusting one camera's
            # most confident box.
            ref_idx = int(np.argmax([len(d) * float(d.scores.mean()) if len(d) else 0.0 for d in detections]))
        ref: PersonDetections = detections[ref_idx]
        if len(ref) == 0:
            return [{} for _ in self.cameras]

        n_subjects: int = min(len(ref), expected) if expected is not None else len(ref)
        ref_order: list[int] = [int(i) for i in np.argsort(-ref.scores)[:n_subjects]]
        if expected is not None and len(ref) > expected:
            # The capture subject is the person every camera sees, inside the camera ring.
            ref_order = order_bootstrap_candidates(
                ref_idx,
                ref.scores,
                [d.centers_xy1 for d in detections],
                [cam.k_matrix for cam in self.cameras],
                [cam.world_to_cam for cam in self.cameras],
                self._sigma_px,
                self._max_dist_px,
                BOOTSTRAP_MIN_SCORE,
            )[:n_subjects]
        obj_ids: list[int] = list(range(n_subjects))

        ref_feats: Float32[torch.Tensor, "n d"] = ref.features[ref_order]
        for obj_id, row in zip(obj_ids, range(len(ref_order)), strict=True):
            self.bank.append(obj_id, ref_feats[row])
        ref_centers: Float32[ndarray, "n 3"] = ref.centers_xy1[ref_order]

        prompts_per_cam: list[dict[int, Float32[ndarray, "4"]]] = [{} for _ in self.cameras]
        for obj_id, det_row in zip(obj_ids, ref_order, strict=True):
            prompts_per_cam[ref_idx][obj_id] = ref.boxes_xyxy[det_row]

        for cam_idx, dets in enumerate(detections):
            if cam_idx == ref_idx or len(dets) == 0:
                continue
            det_feats: Float32[torch.Tensor, "m d"] = dets.features
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
                combined = (CLIP_WEIGHT * s_clip + EPIPOLAR_WEIGHT * s_epi).astype(np.float32)
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
        for cam_idx, dets in enumerate(self._detect(frames)):
            lost: list[int] = [
                obj_id for obj_id, ticks in self._lost_ticks[cam_idx].items() if ticks >= LOST_TICKS_BEFORE_REPROMPT
            ]
            if len(dets) == 0:
                continue
            det_feats: Float32[torch.Tensor, "m d"] = dets.features
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
                combined = (CLIP_WEIGHT * s_clip + EPIPOLAR_WEIGHT * s_epi).astype(np.float32)
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

    def _tracks_from_results(self, per_cam: list[dict[int, SAM2Result]]) -> list[CameraTracks]:
        """Per-camera tracks from ``forward_batched`` results: hypothesis 0 of each object, stats on the GPU, one host copy."""
        rows: list[tuple[int, int]] = [(cam_idx, obj_id) for cam_idx, results in enumerate(per_cam) for obj_id in results]
        tracks: list[CameraTracks] = [{} for _ in per_cam]
        if not rows:
            return tracks
        masks, stats = first_hypothesis_stats(
            torch.cat([per_cam[cam_idx][obj_id].masks_logits[:, :1] for cam_idx, obj_id in rows]),
            torch.cat([per_cam[cam_idx][obj_id].ious[:, :1] for cam_idx, obj_id in rows]),
        )
        all_stats: Float32[ndarray, "n 7"] = stats.cpu().numpy()
        for row, (cam_idx, obj_id) in enumerate(rows):
            tracks[cam_idx][obj_id] = self._resolve_track(cam_idx, obj_id, masks[row], all_stats[row])
        return tracks

    def _forward_all(
        self,
        frame_idx: int,
        frames: list[UInt8[torch.Tensor, "3 h w"]],
        prompts_per_cam: list[dict[int, Float32[ndarray, "4"]]],
    ) -> list[CameraTracks]:
        # Every tick, prompted or steady, is one batched pass over all cameras and objects (memory attention, mask
        # decode, memory encode batched per memory layout; CUDA graphs where captured). Prompted objects decode a single
        # mask (multimask off: with multimask the predicted-IoU argmax intermittently picks the whole-scene candidate);
        # every object's track reads hypothesis 0.
        prompts: list[list[SAM2Prompt]] = [
            [SAM2Prompt(obj_id=obj_id, boxes=torch.as_tensor(box, device=self.device).reshape(1, 4)) for obj_id, box in cam_prompts.items()]
            for cam_prompts in prompts_per_cam
        ]
        return self._tracks_from_results(self._forward_batched(self._states, frame_idx, torch.stack(frames, dim=0), prompts))


def first_hypothesis_stats(masks_logits: torch.Tensor, ious: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary mask of hypothesis 0 and its per-row stats, all on GPU.

    Returns ``(masks (n, h, w) bool, stats (n, 7) float32)`` with stats rows
    ``[bbox_h_rows, x_min, y_min, x_max, y_max, pred_iou, area_px]``.
    """
    masks: torch.Tensor = masks_logits[:, 0] > 0.0
    h: int = masks.shape[1]
    w: int = masks.shape[2]
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
    stats = torch.cat([stats, ious[:, 0:1].float(), area_px], dim=1)
    return masks, stats
