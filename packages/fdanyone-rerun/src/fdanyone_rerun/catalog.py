"""Turn a finished 4DAnyone run recording into a catalog segment in the shared ``exoego:v2`` rig layout.

The run recording (``rerun_log``) keeps each generated view at ``world/cameras/<NN>`` on the ``time`` / ``frame``
timelines. Rig readers (``simplecv.catalog_rig_layout``, ``simplecv.catalog_calibration``) and the MAMMA datasets
expect another layout, so this module moves the chunks, packets untouched:

- ``/world/rig_<NN>/cam_00``: static ``Transform3D`` ``cam_T_world`` (``ChildFromParent``) plus camera metadata;
- ``/world/rig_<NN>/cam_00/pinhole``: static ``Pinhole`` (RDF, portrait 704x1280);
- ``/world/rig_<NN>/cam_00/pinhole/video``: the view's H.264 ``VideoStream`` samples as generated;
- ``/world/source`` (temporal ``Transform3D``, ``world_T_source``), ``/world/source/pinhole`` and
  ``/world/source/pinhole/video``: the source camera fitted to the body and the canonical clip;
- ``/world/pred/fdanyone/people/person_00/body/mesh`` (SMPL-X mesh per frame), ``.../mhr70_xyz`` and
  ``.../mhr70_skeleton``: the conditioning body.

Every temporal chunk lands on ``frame_index`` (sequence) and ``video_time`` (duration), as MAMMA's do. The previews
and the progress log of the run are not part of the segment.
"""

import math
import re
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Float64
from numpy import ndarray

DATASET: str = "fdanyone-views"
"""Catalog dataset of 4DAnyone runs; one segment per run."""
SCHEMA: str = "exoego:v2"
"""The rig layout version the segment follows (``simplecv.catalog_rig_layout``)."""
FRAME_INDEX: str = "frame_index"
VIDEO_TIME: str = "video_time"
PERSON: str = "/world/pred/fdanyone/people/person_00"
"""The one generated person (4DAnyone renders a single subject)."""
RUN_TIME: str = "time"
RUN_FRAME: str = "frame"

_RUN_CAMERA: re.Pattern[str] = re.compile(r"^/world/cameras/(\d+)$")
_RUN_CAMERA_VIDEO: re.Pattern[str] = re.compile(r"^/world/cameras/(\d+)/video$")
_MOVED: dict[str, str] = {
    "/world": "/world",
    "/world/source/video": "/world/source/pinhole/video",
    "/world/body/mesh": f"{PERSON}/body/mesh",
    "/world/body/keypoints": f"{PERSON}/mhr70_xyz",
    "/world/body/skeleton": f"{PERSON}/mhr70_skeleton",
}


def segment_id(clip: str, num_views: int, seed: int) -> str:
    """Segment id of one run, e.g. ``pexels-2785536__v18-seed42``."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", clip):
        raise ValueError(f"clip name {clip!r}: use lower-case letters, digits and dashes")
    return f"{clip}__v{num_views:02d}-seed{seed}"


def camera_entity(view: int) -> str:
    """Camera node of one generated view: one rig per view, like MAMMA's one-camera rigs."""
    return f"/world/rig_{view:02d}/cam_00"


def catalog_entity(run_entity: str, *, is_pinhole: bool) -> str | None:
    """Where a run recording's entity goes in the segment; None for what the segment drops (previews, log).

    ``is_pinhole``: the chunk holds the ``Pinhole`` of a camera node (run recordings put it on the node itself;
    the rig layout puts it on the node's ``pinhole`` child).
    """
    video: re.Match[str] | None = _RUN_CAMERA_VIDEO.fullmatch(run_entity)
    if video is not None:
        return f"{camera_entity(int(video.group(1)))}/pinhole/video"
    camera: re.Match[str] | None = _RUN_CAMERA.fullmatch(run_entity)
    if camera is not None:
        node: str = camera_entity(int(camera.group(1)))
        return f"{node}/pinhole" if is_pinhole else node
    if run_entity == "/world/source":
        return "/world/source/pinhole" if is_pinhole else "/world/source"
    return _MOVED.get(run_entity)


@dataclass(frozen=True, slots=True)
class ViewPose:
    """Where a generated view looks from, read off its pose."""

    yaw_deg: float
    """Around the world Y axis, degrees in [0, 360); 0 looks along -Z (the view facing the person)."""
    pitch_deg: float
    """Degrees above the horizon the camera looks down from; positive is above the subject."""


def view_pose(world_T_cam: Float64[ndarray, "4 4"]) -> ViewPose:
    """Yaw and pitch of a camera from its optical axis (OpenCV RDF: +Z forward)."""
    forward: Float64[ndarray, "3"] = world_T_cam[:3, 2]
    yaw: float = math.degrees(math.atan2(-float(forward[0]), -float(forward[2]))) % 360.0
    pitch: float = math.degrees(math.asin(max(-1.0, min(1.0, -float(forward[1])))))
    return ViewPose(yaw_deg=round(yaw, 3) % 360.0, pitch_deg=round(pitch, 3))


def cam_T_world(world_T_cam: Float64[ndarray, "4 4"]) -> Float64[ndarray, "4 4"]:
    """Invert a rigid camera pose."""
    rotation: Float64[ndarray, "3 3"] = world_T_cam[:3, :3].T
    inverse: Float64[ndarray, "4 4"] = np.eye(4)
    inverse[:3, :3] = rotation
    inverse[:3, 3] = -rotation @ world_T_cam[:3, 3]
    return inverse


def retime(chunk: rrc.Chunk, entity: str, fps: float) -> rrc.Chunk:
    """Move a chunk to ``entity`` and onto ``frame_index`` / ``video_time``, its data columns untouched.

    A static chunk only moves. A temporal one keeps its ``time`` values as ``video_time`` and gets
    ``frame_index = round(time * fps)``; its old ``frame`` column (when present) is dropped, as it is the same clock.
    """
    if chunk.is_static:
        return chunk.with_entity_path(entity)
    batch: pa.RecordBatch = chunk.to_record_batch()
    time_ns: ndarray = batch.column(RUN_TIME).cast(pa.int64()).to_numpy()
    frames: ndarray = np.rint(time_ns.astype(np.float64) * fps / 1e9).astype(np.int64)
    sorted_meta: dict[bytes, bytes] = {b"rerun:kind": b"index", b"rerun:is_sorted": b"true"}
    fields: list[pa.Field] = []
    columns: list[pa.Array] = []
    for field, column in zip(batch.schema, batch.columns, strict=True):
        if field.name == RUN_FRAME:
            continue
        if field.name == RUN_TIME:
            fields.append(pa.field(FRAME_INDEX, pa.int64(), metadata={**sorted_meta, b"rerun:index_name": FRAME_INDEX.encode()}))
            columns.append(pa.array(frames, type=pa.int64()))
            fields.append(field.with_name(VIDEO_TIME).with_metadata({**sorted_meta, b"rerun:index_name": VIDEO_TIME.encode()}))
            columns.append(column)
            continue
        fields.append(field)
        columns.append(column)
    metadata: dict[bytes, bytes] = dict(batch.schema.metadata or {})
    metadata[b"rerun:entity_path"] = entity.encode()
    metadata.pop(b"rerun:id", None)
    moved: pa.RecordBatch = pa.RecordBatch.from_arrays(columns, schema=pa.schema(fields, metadata=metadata))
    (result,) = rrc.Chunk.from_record_batch(moved, entity_path=entity)
    return result


def column_major(matrix: list[float]) -> Float64[ndarray, "3 3"]:
    """A Rerun ``Mat3x3`` (stored column-major) as a row-major array."""
    return np.asarray(matrix, dtype=np.float64).reshape(3, 3).T


@dataclass(frozen=True, slots=True)
class RunCamera:
    """One generated view as the run recording states it."""

    view: int
    """View index (``world/cameras/<NN>``)."""
    world_T_cam: Float64[ndarray, "4 4"]
    """Pose, OpenCV RDF axes, canonical world (Y up)."""
    intrinsics: Float64[ndarray, "3 3"]
    """Pixel intrinsics."""
    width: int
    """Video width, pixels."""
    height: int
    """Video height, pixels."""


def camera_chunks(camera: RunCamera, *, codec: str) -> list[rrc.Chunk]:
    """The rig and camera node chunks a rig reader needs: ``cam_T_world`` and the writer metadata MAMMA carries."""
    node: str = camera_entity(camera.view)
    rig: str = node.rsplit("/", 1)[0]
    inverse: Float64[ndarray, "4 4"] = cam_T_world(camera.world_T_cam)
    pose: ViewPose = view_pose(camera.world_T_cam)
    name: str = f"view_{camera.view:02d}"
    return [
        rrc.Chunk.from_columns(
            node,
            indexes=[],
            columns=rr.Transform3D.columns(translation=[inverse[:3, 3]], mat3x3=[inverse[:3, :3]], relation=rr.TransformRelation.ChildFromParent),
        ),
        rrc.Chunk.from_columns(
            node,
            indexes=[],
            columns=rr.AnyValues.columns(
                name=[name], kind=["rgb"], camera_model=["pinhole"], video_codec=[codec], yaw_deg=[pose.yaw_deg], pitch_deg=[pose.pitch_deg]
            ),
        ),
        rrc.Chunk.from_columns(rig, indexes=[], columns=rr.AnyValues.columns(kind=["exo"], name=[name], num_cameras=[1], schema_version=[SCHEMA])),
    ]


@dataclass(frozen=True, slots=True)
class SegmentProperties:
    """What the segment table shows for one run (``property:<group>:<field>`` columns)."""

    num_cameras: int
    num_frames: int
    fps: float
    width: int
    height: int
    seed: int
    clip: str
    source_video: str
    yaws_deg: tuple[float, ...]
    pitches_deg: tuple[float, ...]

    def chunks(self) -> list[rrc.Chunk]:
        """The property chunks, in MAMMA's groups (``capture``, ``episode``) plus ``generation``."""
        return [
            rrc.Chunk.from_property(
                "capture",
                rr.AnyValues(
                    num_cameras=self.num_cameras,
                    num_frames=self.num_frames,
                    fps=self.fps,
                    schema=SCHEMA,
                    resolution=f"{self.width}x{self.height}",
                    video_source="4DAnyone-Turbo generated H.264 (packets as generated, no re-encode)",
                ),
            ),
            rrc.Chunk.from_property("episode", rr.AnyValues(num_people=1, body_source="gvhmr", collection=self.clip, sequence=self.source_video)),
            rrc.Chunk.from_property(
                "generation",
                rr.AnyValues(
                    generator="4danyone",
                    model="4DAnyone-Turbo (FP8 W8A8, 4 steps)",
                    seed=self.seed,
                    num_views=self.num_cameras,
                    view_yaws_deg=list(self.yaws_deg),
                    view_pitches_deg=list(self.pitches_deg),
                    layer_pitches_deg=sorted({round(pitch) for pitch in self.pitches_deg}),
                ),
            ),
        ]


VIDEO_CODECS: dict[int, str] = {0x61766331: "h264", 0x61763031: "av1", 0x68657631: "h265"}
"""``VideoStream:codec`` (a FourCC) to the name MAMMA's ``video_codec`` metadata uses."""
_DROPPED: list[str] = ["/log/**", "/previews/**", "/__properties/**"]


def _static_values(store: rrc.LazyStore, entity: str) -> dict[str, list[object]]:
    """Every static component of one entity, by column name (first row)."""
    values: dict[str, list[object]] = {}
    for chunk in store.stream().filter(content=entity, is_static=True).to_chunks():
        row: dict[str, object] = chunk.to_record_batch().to_pylist()[0]
        values.update({name: value for name, value in row.items() if ":" in name and isinstance(value, list)})
    return values


def read_run_cameras(store: rrc.LazyStore) -> list[RunCamera]:
    """The generated views of a run recording, in view order."""
    views: list[int] = sorted(
        {int(match.group(1)) for entity in store.schema().entity_paths() if (match := _RUN_CAMERA.fullmatch(str(entity))) is not None}
    )
    if views != list(range(len(views))) or not views:
        raise ValueError(f"run recording has views {views}; expected world/cameras/00..NN")
    cameras: list[RunCamera] = []
    for view in views:
        values: dict[str, list[object]] = _static_values(store, f"/world/cameras/{view:02d}")
        world_T_cam: Float64[ndarray, "4 4"] = np.eye(4)
        world_T_cam[:3, :3] = column_major(values["Transform3D:mat3x3"][0])  # type: ignore[arg-type]
        world_T_cam[:3, 3] = np.asarray(values["Transform3D:translation"][0], dtype=np.float64)
        width, height = (int(side) for side in values["Pinhole:resolution"][0])  # type: ignore[union-attr]
        cameras.append(
            RunCamera(
                view=view,
                world_T_cam=world_T_cam,
                intrinsics=column_major(values["Pinhole:image_from_camera"][0]),  # type: ignore[arg-type]
                width=width,
                height=height,
            )
        )
    return cameras


@dataclass(frozen=True, slots=True)
class VideoClock:
    """The generated videos' clock, read off the samples of view 00."""

    num_frames: int
    fps: float
    codec: str


def read_video_clock(store: rrc.LazyStore) -> VideoClock:
    """Frame count, rate and codec of the generated videos (all views share one clock)."""
    entity: str = "/world/cameras/00/video"
    time_ns: list[int] = []
    for chunk in store.stream().filter(content=entity, components=["VideoStream:sample"], is_static=False).to_chunks():
        time_ns.extend(chunk.to_record_batch().column(RUN_TIME).cast(pa.int64()).to_pylist())
    steps: ndarray = np.diff(np.sort(np.asarray(time_ns, dtype=np.int64)))
    if len(steps) == 0 or steps.min() != steps.max():
        raise ValueError(f"{entity}: expected evenly spaced samples, got steps {sorted(set(steps.tolist()))[:5]}")
    codec: int = int(_static_values(store, entity)["VideoStream:codec"][0])  # type: ignore[arg-type]
    return VideoClock(num_frames=len(time_ns), fps=round(1e9 / float(steps[0]), 6), codec=VIDEO_CODECS.get(codec, f"fourcc:{codec:#x}"))


def segment_stream(store: rrc.LazyStore, cameras: list[RunCamera], clock: VideoClock, properties: SegmentProperties) -> rrc.LazyChunkStream:
    """The run recording's chunks moved into the rig layout, plus the camera nodes and the properties."""

    def move(chunk: rrc.Chunk) -> list[rrc.Chunk]:
        is_pinhole: bool = any(name.startswith("Pinhole:") for name in chunk.to_record_batch().schema.names)
        if _RUN_CAMERA.fullmatch(chunk.entity_path) is not None and not is_pinhole:
            return []  # world_T_cam: replaced by camera_chunks' cam_T_world
        target: str | None = catalog_entity(chunk.entity_path, is_pinhole=is_pinhole)
        return [] if target is None else [retime(chunk, target, clock.fps)]

    moved: rrc.LazyChunkStream = store.stream().drop(content=_DROPPED).flat_map(move)
    added: list[rrc.Chunk] = [chunk for camera in cameras for chunk in camera_chunks(camera, codec=clock.codec)]
    added.append(
        rrc.Chunk.from_columns(PERSON, indexes=[], columns=rr.AnyValues.columns(model=["smplx"], source=["fdanyone"], subject=["person_00"]))
    )
    added.extend(properties.chunks())
    return rrc.LazyChunkStream.merge(moved, rrc.LazyChunkStream.from_iter(added))
