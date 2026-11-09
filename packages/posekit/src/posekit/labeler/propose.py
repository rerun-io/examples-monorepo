"""EgoHand hand-box proposals on the 5 fps grid of every camera, written as the ``hand_proposals`` catalog layer.

Per camera, the RT-DETR hand detector runs on every grid frame in GPU batches. Hand identity (left/right) is then
assigned **per frame** with upstream's ``resolve_hand_identities`` (two hands: image geometry; one hand: the
detector class), which keeps at most one left and one right box per frame. Upstream's sequence-level track
continuity is not used: at 5 fps the IoU tracks swap hands. Checked on fe6f s13/s15 against the hands layer's
confident KeyNet sides, the whole-sequence resolution agreed on 341/551 and 967/2413 cam_00 boxes, per-frame
resolution on 542/551 and 2347/2413.

The layer holds one ``Boxes2D`` per camera and grid time at ``paths.proposal_path(cam, "egohand")`` (an empty one
where nothing was found, so the viewer never shows a stale box), plus the scores as ``posekit.HandProposal:score``.
"""

import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from jaxtyping import Bool, Float32, Int, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from rerun.chunk import RrdReader
from torch import Tensor

from posekit.labeler.frames import TIMELINE, GridBatch
from posekit.labeler.paths import class_id, proposal_path
from posekit.labeler.proposals import SCORE_ARCHETYPE, SCORE_DECIMALS, ProposalsByTime, clipped_box
from posekit.labeler.schema import PROPOSAL_LAYER, Box, CameraInfo, HandSide, Proposal
from posekit.models.egohand import EgoHandDetector, _resize_on_gpu, _resize_with_pil, resolve_hand_identities
from posekit.predictions import BoxDetections, HandDetections, validate_frames_rgb
from posekit.runtimes import run_chunked

SIDE_COLORS: dict[HandSide, tuple[int, int, int]] = {"left": (66, 165, 245), "right": (255, 152, 0)}
"""Proposal box colours: left blue, right orange."""


@dataclass(frozen=True, slots=True)
class CameraTiming:
    """Throughput of one camera's proposal pass."""

    frames: int
    """Grid frames decoded and detected."""
    decode_s: float
    """Seconds spent waiting for decoded batches."""
    detect_s: float
    """Seconds spent in the detector (synchronised)."""


def raw_detections(detector: EgoHandDetector, frames_rgb: UInt8[Tensor, "b h w 3"]) -> HandDetections:
    """Every candidate box above the detector threshold, both classes, before identity assignment.

    This is ``EgoHandDetector.__call__`` without its final sequence-level ``resolve_hand_identities`` (see the
    module docstring). It also decodes the class id inline: ``__call__`` imports
    ``egohand_geometry`` for that, which pulls ``warp``, absent from the aarch64 ``posekit-dev`` env.

    Args:
        detector: The set-up EgoHand detector.
        frames_rgb: RGB frames on the detector's device.
    """
    validate_frames_rgb(frames_rgb)
    height: int = int(frames_rgb.shape[1])
    width: int = int(frames_rgb.shape[2])
    boxes: list[Tensor] = []
    scores: list[Tensor] = []
    indices: list[Tensor] = []
    sides: list[HandSide] = []
    for start in range(0, len(frames_rgb), detector.runtime.spec.max_batch_size):
        stop: int = min(start + detector.runtime.spec.max_batch_size, len(frames_rgb))
        chunk: UInt8[Tensor, "c h w 3"] = frames_rgb[start:stop]
        pixels: Float32[Tensor, "b 3 640 640"] = _resize_on_gpu(chunk) if detector.config.gpu_preprocessing else _resize_with_pil(chunk)
        sizes: Int64[Tensor, "b 2"] = torch.empty((stop - start, 2), dtype=torch.int64, device=frames_rgb.device)
        sizes[:, 0] = width
        sizes[:, 1] = height
        outputs: dict[str, Tensor] = run_chunked(detector.runtime, {"images": pixels.contiguous(), "orig_target_sizes": sizes})
        labels: Int[Tensor, "b q"] = outputs["labels"]
        keep: Bool[Tensor, "b 2 q"] = torch.stack([labels == 1, labels == 2], dim=1) & (outputs["scores"] > detector.config.threshold)[:, None]
        hits: tuple[Tensor, ...] = torch.where(keep)
        boxes.append(outputs["boxes"][hits[0], hits[2]].float())
        scores.append(outputs["scores"][hits[0], hits[2]].float())
        indices.append(hits[0] + start)
        # Class 1 is the left hand, class 2 the right (egohand_geometry.detector_hand_side).
        sides.extend("left" if label == 0 else "right" for label in hits[1].tolist())
    if not sides:
        return HandDetections.empty(frames_rgb.device)
    return HandDetections(
        BoxDetections(torch.cat(boxes), torch.cat(scores), torch.cat(indices).to(device=frames_rgb.device, dtype=torch.int64)), tuple(sides)
    )


def concat_detections(parts: Sequence[HandDetections], offsets: Sequence[int], device: torch.device | str) -> HandDetections:
    """Join per-batch detections into one sequence, shifting each batch's frame indices by its offset."""
    kept: list[tuple[HandDetections, int]] = [(part, offset) for part, offset in zip(parts, offsets, strict=True) if part.num_detections]
    if not kept:
        return HandDetections.empty(device)
    return HandDetections(
        BoxDetections(
            torch.cat([part.boxes.xyxy for part, _ in kept]),
            torch.cat([part.boxes.scores for part, _ in kept]),
            torch.cat([part.boxes.frame_indices + offset for part, offset in kept]),
        ),
        tuple(side for part, _ in kept for side in part.sides),
    )


def resolve_each_frame(raw: HandDetections) -> HandDetections:
    """Upstream identity assignment run on every frame alone: at most one left and one right box per frame."""
    frames: Int64[ndarray, "n"] = raw.boxes.frame_indices.cpu().numpy().astype(np.int64)
    parts: list[HandDetections] = []
    offsets: list[int] = []
    for frame in np.unique(frames).tolist():
        rows: list[int] = np.flatnonzero(frames == frame).tolist()
        index: Int64[Tensor, "k"] = torch.tensor(rows, dtype=torch.int64, device=raw.boxes.xyxy.device)
        single: HandDetections = HandDetections(
            BoxDetections(raw.boxes.xyxy[index], raw.boxes.scores[index], torch.zeros_like(raw.boxes.frame_indices[index])), tuple(raw.sides[row] for row in rows)
        )
        parts.append(resolve_hand_identities(single, 1))
        offsets.append(int(frame))
    return concat_detections(parts, offsets, raw.boxes.xyxy.device)


def detect_camera(detector: EgoHandDetector, batches: Iterable[GridBatch]) -> tuple[Int64[ndarray, "g"], HandDetections, CameraTiming]:
    """Detect one camera's grid frames batch by batch, then assign hand identities frame by frame.

    Returns:
        The grid times walked, the resolved detections (frame index = position in those times), and the timing.
    """
    times: list[Int64[ndarray, "b"]] = []
    parts: list[HandDetections] = []
    offsets: list[int] = []
    count: int = 0
    decode_s: float = 0.0
    detect_s: float = 0.0
    device: torch.device | str = "cuda"
    started: float = time.perf_counter()
    for batch in batches:
        decode_s += time.perf_counter() - started
        started = time.perf_counter()
        device = batch.frames_rgb.device
        parts.append(raw_detections(detector, batch.frames_rgb))
        torch.cuda.synchronize(device)
        detect_s += time.perf_counter() - started
        offsets.append(count)
        times.append(batch.times_ns)
        count += len(batch.times_ns)
        started = time.perf_counter()
    grid: Int64[ndarray, "g"] = np.concatenate(times).astype(np.int64) if times else np.zeros(0, np.int64)
    resolved: HandDetections = resolve_each_frame(concat_detections(parts, offsets, device))
    return grid, resolved, CameraTiming(count, decode_s, detect_s)


def detections_to_proposals(detections: HandDetections, times_ns: Int64[ndarray, "g"], cam: CameraInfo) -> ProposalsByTime:
    """One camera's resolved detections as ``egohand`` proposals per grid time (every time present, maybe empty)."""
    result: ProposalsByTime = {int(t): [] for t in times_ns.tolist()}
    xyxy: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
    scores: list[float] = detections.boxes.scores.cpu().numpy().astype(np.float64).tolist()
    frames: list[int] = detections.boxes.frame_indices.cpu().numpy().tolist()
    for box_xyxy, score, frame, side in zip(xyxy.tolist(), scores, frames, detections.sides, strict=True):
        box: Box | None = clipped_box(box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3], cam)
        if box is not None:
            result[int(times_ns[frame])].append(Proposal(cam=cam.cam, side=side, box=box, score=round(min(max(score, 0.0), 1.0), SCORE_DECIMALS), model="egohand"))
    return result


def proposal_label(proposal: Proposal) -> str:
    """Viewer label of a proposal box, e.g. ``L 0.93``."""
    return f"{'L' if proposal.side == 'left' else 'R'} {proposal.score:.2f}"


def write_proposal_layer(
    path: Path, *, application_id: str, segment_id: str, cams: Sequence[CameraInfo], proposals: ProposalsByTime, properties: dict[str, str | float | int]
) -> None:
    """Write the ``hand_proposals`` layer rrd atomically (temp file, chmod 0644, rename).

    Args:
        path: Final rrd path; its directory is created.
        application_id: The base layer's application id (layers stack by application and recording id).
        segment_id: Catalog segment id, used as the recording id.
        cams: Cameras of the segment.
        proposals: ``egohand`` proposals per grid time (all cameras); every key is logged for every camera.
        properties: Provenance written as the ``hand_proposals`` recording property group.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with rr.RecordingStream(application_id, recording_id=segment_id, send_properties=False) as recording:
            recording.save(temp, write_footer=True)
            provenance: rr.AnyValues = rr.AnyValues()
            for key, value in properties.items():
                provenance = provenance.with_component_from_data(key, value)
            recording.send_property(PROPOSAL_LAYER, provenance)
            for cam in cams:
                entity: str = proposal_path(cam.cam, "egohand")
                for t_ns in sorted(proposals):
                    mine: list[Proposal] = [proposal for proposal in proposals[t_ns] if proposal.cam == cam.cam and proposal.model == "egohand"]
                    recording.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
                    recording.log(
                        entity,
                        rr.Boxes2D(
                            array=np.asarray([[p.box.x0, p.box.y0, p.box.x1, p.box.y1] for p in mine], dtype=np.float32).reshape(-1, 4),
                            array_format=rr.Box2DFormat.XYXY,
                            class_ids=np.asarray([class_id(p.side) for p in mine], dtype=np.uint16),
                            labels=[proposal_label(p) for p in mine],
                            colors=np.asarray([SIDE_COLORS[p.side] for p in mine], dtype=np.uint8).reshape(-1, 3),
                        ),
                        rr.DynamicArchetype(SCORE_ARCHETYPE, components={"score": np.asarray([p.score for p in mine], dtype=np.float32)}),
                    )
        temp.chmod(0o644)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def base_application_id(dataset: DatasetEntry, segment_id: str) -> str:
    """Application id of the segment's ``base`` layer, read from the rrd header the catalog points at.

    Raises:
        ValueError: If the segment has no file-backed base layer holding exactly one recording.
    """
    table: pa.Table = dataset.segment_table().to_arrow_table()
    for row in table.select(["rerun_segment_id", "rerun_layer_names", "rerun_storage_urls"]).to_pylist():
        if row["rerun_segment_id"] != segment_id:
            continue
        for layer, url in zip(row["rerun_layer_names"], row["rerun_storage_urls"], strict=True):
            if layer == "base" and url.startswith("file://"):
                recordings = RrdReader(Path(url.removeprefix("file://"))).recordings()
                if len(recordings) == 1:
                    return recordings[0].application_id
    raise ValueError(f"{segment_id}: no file-backed base layer with one recording on {dataset.name}")


def register_layer(dataset: DatasetEntry, path: Path) -> None:
    """Register (or replace) the segment's ``hand_proposals`` layer."""
    dataset.register([path.as_uri()], layer_name=PROPOSAL_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
