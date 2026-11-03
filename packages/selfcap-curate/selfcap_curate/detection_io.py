"""Property-free Sapiens2 detection layers shared by catalog stages."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
from dataforge.hands import shown_keypoint_columns, sparse_keypoint_columns
from dataforge.logging_toolkit import capture_column, time_column
from dataforge.schema import SHOWN_MIN_CONFIDENCE, coco133_uv_path, coco133_uv_shown_path
from dataforge.selfcap_schema import oak_egohand_path
from dataforge.writing import APPLICATION_ID, atomic_write
from jaxtyping import Float32
from numpy import ndarray
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS

from selfcap_curate.display_style import SKELETON_LINK_RGB
from selfcap_curate.records import OAK_CAMERAS, OakDetections, PhoneDetections, SelfcapLayout, TimeSync

COCO133_CLASS: rr.ClassDescription = rr.ClassDescription(
    info=rr.AnnotationInfo(id=0, label="COCO-133", color=SKELETON_LINK_RGB),
    keypoint_annotations=[rr.AnnotationInfo(id=i, label=name) for i, name in COCO_133_ID2NAME.items()],
    keypoint_connections=COCO_133_LINKS,
)
"""Class 0 of every COCO-133 prediction layer this package writes."""


def write_oak_egohand_layer(path: Path, segment: str, oak_rig: int, views: dict[str, OakDetections]) -> None:
    """Persist newly inferred calibration samples, per OAK camera (``rgb``/``left``/``right``), as a property-free layer."""
    context = rr.ClassDescription(info=rr.AnnotationInfo(id=0, label="EgoHand calibration", color=(255, 145, 35)),
                                  keypoint_annotations=COCO133_CLASS.keypoint_annotations, keypoint_connections=COCO_133_LINKS)

    def chunks() -> Iterator[Chunk]:
        for camera, detections in views.items():
            entity: str = oak_egohand_path(oak_rig, OAK_CAMERAS.index(camera))
            count: int = len(detections.video_time)
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
            scores: Float32[ndarray, "f 133"] = hand.scores.astype(np.float32)
            xy: Float32[ndarray, "f 133 2"] = hand.xy.astype(np.float32)
            copy = shown_keypoint_columns(xy, scores, scores >= SHOWN_MIN_CONFIDENCE)
            for path, columns in (
                (entity, sparse_keypoint_columns(xy, scores, np.ones(scores.shape, dtype=bool))),
                (coco133_uv_shown_path(rig, 0, pred="sapiens2"), copy.columns),
            ):
                yield Chunk.from_columns(path, indexes=[], columns=copy.style)
                yield Chunk.from_columns(path, indexes=[capture_column(stamps), time_column(stamps + offset_ns)], columns=columns)


    with atomic_write(path) as target:
        LazyChunkStream.from_iter(chunks()).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
            target, application_id=APPLICATION_ID, recording_id=sync.episode_key.segment
        )
