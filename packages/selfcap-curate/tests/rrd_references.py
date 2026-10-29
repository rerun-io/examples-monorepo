"""Typed, viewer-readable RRD fixtures for frozen detector streams."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dataforge.writing import atomic_write
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile, RrdReader
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS

from selfcap_curate.keypoints import Keypoints2D
from selfcap_curate.records import OakDetections


@dataclass(frozen=True, slots=True)
class OakReference:
    indices: Int64[ndarray, "n"]
    fps: float
    times_s: Float64[ndarray, "n"]
    detections: OakDetections
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class KeypointReference:
    detections: Keypoints2D
    pts_s: dict[str, Float64[ndarray, "n"]]
    video_times_s: dict[str, Float64[ndarray, "n"]]
    schema_version: int


def annotation(parent: str, persons: int) -> Chunk:
    return Chunk.from_columns(
        parent,
        indexes=[],
        columns=rr.AnnotationContext.columns(
            context=[
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=person, label="COCO-133"),
                    keypoint_annotations=[rr.AnnotationInfo(id=i, label=name) for i, name in COCO_133_ID2NAME.items()],
                    keypoint_connections=COCO_133_LINKS,
                )
                for person in range(persons)
            ]
        ),
    )


def save_reference(path: Path, segment: str, chunks: list[Chunk]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with atomic_write(path) as target:
        LazyChunkStream.from_iter(chunks).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
            target, application_id="dataforge", recording_id=segment
        )


def write_oak_reference(path: Path, segment: str, camera: str, reference: OakReference) -> None:
    """Store exact seconds alongside the viewer's rounded duration timeline."""
    parent = f"{camera}/pinhole/pred/sapiens2"
    hand = reference.detections
    n = len(reference.indices)
    chunks = [
        Chunk.from_columns(
            "/reference",
            indexes=[],
            columns=rr.AnyValues.columns(camera=[camera], fps=[reference.fps], width=[reference.width], height=[reference.height]),
        ),
        annotation(parent, 1),
    ]
    if n:
        chunks.append(
            Chunk.from_columns(
                f"{parent}/coco133_uv",
                indexes=[rr.TimeColumn("video_time", duration=np.rint(hand.video_time * 1e9).astype("timedelta64[ns]"))],
                columns=[
                    *rr.Points2D.columns(
                        positions=hand.xy.astype(np.float32).reshape(-1, 2),
                        keypoint_ids=np.tile(np.arange(133, dtype=np.uint16), n),
                        class_ids=np.zeros(n * 133, np.uint16),
                    ).partition([133] * n),
                    *rr.AnyValues.columns(
                        idx=reference.indices, time_s=reference.times_s, video_time_s=hand.video_time, confidence=hand.scores.tolist()
                    ),
                ],
            )
        )
    save_reference(path, segment, chunks)


def write_keypoint_reference(path: Path, segment: str, cameras: dict[str, str], reference: KeypointReference) -> None:
    """Preserve padded persons and original corners beside drawable boxes."""
    detections = reference.detections
    chunks = [
        Chunk.from_columns(
            "/reference",
            indexes=[],
            columns=rr.AnyValues.columns(
                phones=[list(detections.views)], cameras=[list(cameras.values())], rate=[detections.rate], schema_version=[reference.schema_version]
            ),
        )
    ]
    timeline = rr.TimeColumn("video_time", duration=np.rint(detections.times * 1e9).astype("timedelta64[ns]"))
    chunks.append(Chunk.from_columns("/reference/grid", indexes=[timeline], columns=rr.AnyValues.columns(time_s=detections.times)))
    for phone, view in detections.views.items():
        n, persons = view.boxes.shape[:2]
        parent = f"{cameras[phone]}/pinhole/pred/rtmw_persons"
        chunks.append(annotation(parent, persons))
        chunks.append(
            Chunk.from_columns(
                f"{parent}/coco133_uv",
                indexes=[timeline],
                columns=[
                    *rr.Points2D.columns(
                        positions=view.xy.astype(np.float32).reshape(-1, 2),
                        keypoint_ids=np.tile(np.arange(133, dtype=np.uint16), n * persons),
                        class_ids=np.tile(np.repeat(np.arange(persons, dtype=np.uint16), 133), n),
                    ).partition([persons * 133] * n),
                    *rr.AnyValues.columns(
                        confidence=view.scores.reshape(n, -1).tolist(), time_s=reference.video_times_s[phone], pts_s=reference.pts_s[phone]
                    ),
                ],
            )
        )
        drawable = np.isfinite(view.boxes).all(2)
        corners = view.boxes[drawable]
        chunks.append(
            Chunk.from_columns(
                f"{parent}/boxes",
                indexes=[timeline],
                columns=[
                    *rr.Boxes2D.columns(
                        centers=(corners[:, :2] + corners[:, 2:]) / 2.0,
                        half_sizes=(corners[:, 2:] - corners[:, :2]) / 2.0,
                    ).partition(drawable.sum(1)),
                    *rr.AnyValues.columns(xyxy=view.boxes.reshape(n, -1).tolist(), confidence=view.box_scores.tolist()),
                ],
            )
        )
    save_reference(path, segment, chunks)


def reference_tables(path: Path) -> dict[str, pa.Table]:
    """Materialize small reference streams without a catalog or viewer."""
    columns = {}
    types: dict[tuple[str, str], pa.DataType] = {}
    for chunk in RrdReader(path).stream():
        entity = str(chunk.entity_path)
        if entity != "/reference" and not entity.endswith(("/grid", "/coco133_uv", "/boxes")):
            continue
        batch = chunk.to_record_batch()
        stamps = [0] * batch.num_rows if chunk.is_static else batch.column("video_time").cast(pa.int64()).to_pylist()
        for name in batch.schema.names:
            if name.startswith("rerun.") or name == "video_time":
                continue
            types[entity, name] = batch.schema.field(name).type
            values = columns.setdefault(entity, {}).setdefault(name, {})
            values.update(zip(stamps, batch.column(name).to_pylist(), strict=True))
    return {
        entity: pa.table({name: pa.array([value[stamp] for stamp in sorted(value)], type=types[entity, name]) for name, value in fields.items()})
        for entity, fields in columns.items()
    }
