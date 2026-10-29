"""Property-free Sapiens2 detection layers shared by catalog stages."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
from dataforge.logging_toolkit import capture_column, time_column
from dataforge.schema import coco133_uv_path
from dataforge.writing import APPLICATION_ID, atomic_write
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS

from selfcap_curate.records import PhoneDetections, SelfcapLayout, TimeSync

COCO133_CLASS: rr.ClassDescription = rr.ClassDescription(
    info=rr.AnnotationInfo(id=0, label="COCO-133"),
    keypoint_annotations=[rr.AnnotationInfo(id=i, label=name) for i, name in COCO_133_ID2NAME.items()],
    keypoint_connections=COCO_133_LINKS,
)
"""Class 0 of every COCO-133 prediction layer this package writes."""


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
