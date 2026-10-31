"""Property-free Sapiens2 detection layers shared by catalog stages."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
from dataforge.logging_toolkit import capture_column, time_column
from dataforge.schema import coco133_uv_path
from dataforge.writing import APPLICATION_ID, atomic_write
from jaxtyping import Float64
from numpy import ndarray
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS

from selfcap_curate.records import OakDetections, PhoneDetections, SelfcapLayout, TimeSync

COCO133_CLASS: rr.ClassDescription = rr.ClassDescription(
    info=rr.AnnotationInfo(id=0, label="COCO-133"),
    keypoint_annotations=[rr.AnnotationInfo(id=i, label=name) for i, name in COCO_133_ID2NAME.items()],
    keypoint_connections=COCO_133_LINKS,
)
"""Class 0 of every COCO-133 prediction layer this package writes."""


def read_oak_observations(path: Path, segment: str) -> OakDetections:
    """Read dense EgoHand COCO-133 pixels and repeated hand-box scores on video_time.

    The Parquet stream has segment_id, int64 video_time nanoseconds, xy (133, 2),
    and scores (133). Missing joints carry NaN pixels and zero confidence.
    """
    table: pa.Table = pq.read_table(path)
    if set(table.column_names) != {"segment_id", "video_time", "xy", "scores"}:
        raise ValueError(f"{path}: expected segment_id, video_time, xy and scores")
    identities: set[str] = set(table["segment_id"].to_pylist())
    if identities != {segment}:
        raise ValueError(f"{path}: expected {segment}, found {sorted(identities)}")
    if not pa.types.is_int64(table["video_time"].type):
        raise ValueError(f"{path}: video_time must be int64 nanoseconds")
    times: Float64[ndarray, "n"] = table["video_time"].to_numpy().astype(np.float64) * 1e-9
    if len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError(f"{path}: expected at least two strictly increasing timestamps")
    xy: Float64[ndarray, "n j d"] = np.asarray(table["xy"].to_pylist(), dtype=np.float64)
    scores: Float64[ndarray, "n j"] = np.asarray(table["scores"].to_pylist(), dtype=np.float64)
    if xy.shape != (len(times), 133, 2) or scores.shape != (len(times), 133):
        raise ValueError(f"{path}: expected COCO-133 xy and scores")
    if np.isinf(xy).any() or not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{path}: invalid pixels or confidence outside [0, 1]")
    return OakDetections(times, xy, scores)


def write_oak_egohand_layer(path: Path, segment: str, oak_rig: int, detections: OakDetections) -> None:
    """Persist newly inferred calibration samples as an independent, property-free layer."""
    entity: str = coco133_uv_path(oak_rig, 0, pred="egohand_calib")
    count: int = len(detections.video_time)
    context = rr.ClassDescription(info=rr.AnnotationInfo(id=0, label="EgoHand calibration", color=(255, 145, 35)),
                                  keypoint_annotations=COCO133_CLASS.keypoint_annotations, keypoint_connections=COCO_133_LINKS)

    def chunks() -> Iterator[Chunk]:
        yield Chunk.from_columns(entity.rsplit("/", 1)[0], indexes=[], columns=rr.AnnotationContext.columns(context=[context]))
        if count:
            yield Chunk.from_columns(
                entity, indexes=[time_column(np.rint(detections.video_time * 1e9).astype(np.int64))],
                columns=[
                    *rr.Points2D.columns(positions=detections.xy.astype(np.float32).reshape(-1, 2),
                                        keypoint_ids=np.tile(np.arange(133, dtype=np.uint16), count),
                                        class_ids=np.zeros(count * 133, dtype=np.uint16)).partition([133] * count),
                    *rr.AnyValues.columns(confidence=detections.scores.tolist()),
                ],
            )

    with atomic_write(path) as target:
        LazyChunkStream.from_iter(chunks()).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
            target, application_id=APPLICATION_ID, recording_id=segment,
        )


def write_detections_layer(path: Path, sync: TimeSync, layout: SelfcapLayout, detections: dict[str, PhoneDetections]) -> None:
    """Keep every joint, including missing/low-confidence cells, on presented PTS.

    Integer offset rounding is the phone rule in patch_base. Positions originate
    as float32; confidence retains the exact widened float16 model values.
    """
    if set(detections) - set(layout.phones):
        raise ValueError("detection phones absent from segment layout")

    def chunks() -> Iterator[Chunk]:
        for rig, phone in enumerate(layout.phones):
            if phone not in detections:
                continue
            hand = detections[phone]
            entity = coco133_uv_path(rig, 0, pred="sapiens2")
            yield Chunk.from_columns(entity.rsplit("/", 1)[0], indexes=[], columns=rr.AnnotationContext.columns(context=[COCO133_CLASS]))
            offset_ns = round(((sync.offsets_ms or {}).get(f"exo/{phone}") or 0.0) * 1e6)
            stamps = np.rint(hand.times_s * 1e9).astype(np.int64)
            count = len(stamps)
            columns = rr.Points2D.columns(
                positions=hand.xy.astype(np.float32).reshape(-1, 2),
                keypoint_ids=np.tile(np.arange(133, dtype=np.uint16), count),
                class_ids=np.zeros(count * 133, dtype=np.uint16),
            ).partition([133] * count)
            yield Chunk.from_columns(
                entity,
                indexes=[capture_column(stamps), time_column(stamps + offset_ns)],
                columns=[*columns, *rr.AnyValues.columns(confidence=hand.scores.tolist())],
            )

    with atomic_write(path) as target:
        LazyChunkStream.from_iter(chunks()).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
            target, application_id=APPLICATION_ID, recording_id=sync.episode_key.segment
        )
