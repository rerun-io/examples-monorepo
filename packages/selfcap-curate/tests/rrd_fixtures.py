"""Shared synthetic RRD and timestamp adapters; no reference assertions."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from dataforge import schema
from dataforge.logging_toolkit import capture_column, log_video_stream
from dataforge.schema import CAPTURE_TIMELINE
from dataforge.writing import recording_to
from jaxtyping import Int64
from numpy import ndarray
from rerun.chunk import RrdReader
from rerun.server import Server
from serde.json import to_json

from selfcap_curate.records import TimeSync, read_sync

FIXTURES = Path(__file__).parent / "fixtures"
VIDEO_FIXTURES = Path(__file__).parents[2] / "dataforge/tests/fixtures"


def synthetic_served(tmp_path: Path) -> tuple[Path, Path, TimeSync, dict[str, int]]:
    """Build real AV1 sample chunks plus raw hand CSVs; no downloaded inputs."""
    sync = read_sync(FIXTURES / "2025-11-24__027d8215__ep002.sync.json")
    raw = tmp_path / "raw"
    cameras: list[tuple[str, str]] = [(f"ego/{name}.mp4", schema.video_path(4, i)) for i, name in enumerate(("rgb", "left", "right"))]
    cameras += [(f"quest/{name}.mp4", schema.video_path(5, i)) for i, name in enumerate(("left", "right"))]
    for i, device in enumerate(sorted(name.removeprefix("exo/") for name in sync.offsets_ms if name.startswith("exo/"))):
        cameras.append((f"exo/{device}/{device}.mp4", schema.video_path(i, 0)))
    starts: dict[str, int] = {}
    for i, (relative, entity) in enumerate(cameras):
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        late: bool = i % 2 == 0
        shutil.copyfile(VIDEO_FIXTURES / ("av1_48f_192x160_start40ms.mp4" if late else "av1_48f_192x160.mp4"), path)
        if relative.startswith("exo/"):
            path.with_name("calibration.json").touch()
        starts[entity] = 40_039_062 if late else 0
    for folder in (raw / "ego", raw / "quest"):
        (folder / "calibration.json").write_text("{}")
    for side, stamp in (("left", 17), ("right", 27)):
        (raw / "quest" / f"{side}_hand_poses.csv").write_text(
            "ts_ns," + ",".join(f"j{i}" for i in range(78)) + "\n" + str(stamp) + "," + ",".join(["1"] * 78) + "\n"
        )
    served = tmp_path / "served.rrd"
    with recording_to(served, recording_id=f"selfcap__{sync.key}", default_blueprint=rrb.Blueprint(rrb.Spatial3DView(origin="/world"))) as recording:
        recording.send_property("time_sync", rr.AnyValues(record_json=to_json(sync), method="keep-this-field"))
        recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        recording.send_columns(
            "/world/untouched",
            indexes=[
                capture_column(np.array([17, 27], dtype=np.int64)),
                rr.TimeColumn("video_time", duration=np.array([117, 127], dtype="timedelta64[ns]")),
            ],
            columns=rr.Scalars.columns(scalars=[1.0, 2.0]),
        )
        for relative, entity in cameras:
            log_video_stream(recording, raw / relative, entity, capture_time=True)
    return served, raw, sync, starts


@dataclass(frozen=True, slots=True)
class EntityTimes:
    capture_ns: Int64[ndarray, "n"]
    """Capture stamps, ordered by capture time including sparse marker rows."""
    video_ns: Int64[ndarray, "n"]
    """Aligned stamps in corresponding capture order."""
    samples: int
    """Video sample count, or temporal rows for a non-video entity."""


def entity_times(path: Path) -> dict[str, EntityTimes]:
    """Read timestamp columns without decoding any video."""
    rows: dict[str, list[tuple[int, int]]] = {}
    counts: dict[str, int] = {}
    for chunk in RrdReader(path).stream():
        if chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        entity = str(chunk.entity_path)
        capture = batch.column(CAPTURE_TIMELINE).cast(pa.int64()).to_pylist()
        video = batch.column("video_time").cast(pa.int64()).to_pylist()
        rows.setdefault(entity, []).extend(zip(capture, video, strict=True))
        if "VideoStream:sample" in batch.schema.names or "/pinhole/video" not in entity:
            counts[entity] = counts.get(entity, 0) + batch.num_rows
    result: dict[str, EntityTimes] = {}
    for entity, values in rows.items():
        array = np.asarray(values, dtype=np.int64)
        order = np.argsort(array[:, 0], kind="stable")
        result[entity] = EntityTimes(array[order, 0], array[order, 1], counts[entity])
    return result


def private_server(datasets):
    """Construct only a test-scoped loopback server on an allocated port."""
    return Server(host="127.0.0.1", port=0, datasets=datasets)


def synthetic_base(path, *, segment, phones=("A", "B", "C"), oak=True, quest=True, video=None, sync=None, calibrated=False):
    """One builder for segment-local phone/optional wearable metadata and packets."""
    with recording_to(path, recording_id=segment) as recording:
        if sync is not None:
            recording.send_property("time_sync", rr.AnyValues(record_json=to_json(sync)))
        rigs = [(i, "exo", phone, 1) for i, phone in enumerate(phones)]
        if oak:
            rigs.append((len(phones), "ego", "oak", 3))
        if quest:
            rigs.append((len(phones) + 1, "quest", "quest", 2))
        for rig, kind, name, cameras in rigs:
            recording.log(schema.rig_path(rig), rr.AnyValues(name=name, kind=kind), static=True)
            for camera in range(cameras):
                entity = schema.video_path(rig, camera)
                if video is None:
                    recording.log(entity, rr.VideoStream(codec=rr.VideoCodec.AV1, sample=[1]), static=True)
                else:
                    log_video_stream(recording, video, entity, capture_time=True)
                if calibrated and kind == "exo":
                    recording.log(
                        schema.cam_path(rig, camera),
                        rr.AnyValues(name=name, calib_status="ok", calib_phone=name),
                        rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)),
                        static=True,
                    )
                    recording.log(schema.pinhole_path(rig, camera), rr.Pinhole(focal_length=100.0, width=192, height=160), static=True)
    return path
