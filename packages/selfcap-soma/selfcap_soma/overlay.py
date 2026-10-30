"""Write a layer aligned with SelfCap's original recording and camera tree."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader

from selfcap_soma.camera_geometry import project, project_ego
from selfcap_soma.evaluation_io import first, read_views


@dataclass(frozen=True, slots=True)
class Config:
    source: Path = Path("/tmp/soma-fit/segment-final/fit.rrd")
    """Fitted geometry recording; no input videos are copied."""
    reference: Path = Path("/tmp/selfcap-dev/c4/out/keypoints/selfcap__2025-12-03__d31bf1da__ep001.rrd")
    """Original layer defining the application and recording IDs."""
    output: Path = Path("/tmp/soma-fit/layer/selfcap__2025-12-03__d31bf1da__ep001.rrd")
    """Local layer file; registration is a separate explicit action."""


def main(config: Config) -> None:
    source = RrdReader(config.source)
    identity = RrdReader(config.reference).recordings()[0]
    segment = "selfcap__2025-12-03__d31bf1da__ep001"
    if identity.recording_id != segment:
        raise ValueError("reference recording must be the requested SelfCap segment")
    rows = []
    annotations = []
    for chunk in source.stream().filter(content="/world/fit/joints").to_chunks():
        table = chunk.to_record_batch()
        if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
            times = table.column("video_time").cast("int64").to_numpy().astype(np.float64) / 1e9
            points = np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32)
            rows.append((times, points))
        if "AnnotationContext:context" in table.schema.names:
            annotations.append(chunk)
    times = np.concatenate([row[0] for row in rows])
    points = np.concatenate([row[1] for row in rows])
    order = np.argsort(times)
    times, points = times[order], points[order]
    dataset = CatalogClient("rerun+http://dgx-spark:51235").get_dataset("selfcap-post-dev")
    phones, rgb = read_views(dataset, segment, times)
    paths = [f"/world/rig_04/cam_{cam:02d}" for cam in (1, 2)]
    table = dataset.filter_segments(segment).filter_contents([*paths, *(f"{path}/pinhole" for path in paths)]).reader(index=None).to_arrow_table()
    egos = [rgb]
    for cam, path in enumerate(paths, start=1):
        transform = np.eye(4)
        transform[:3, :3] = first(table, f"{path}:Transform3D:mat3x3").reshape(3, 3).T
        transform[:3, 3] = first(table, f"{path}:Transform3D:translation")
        size = first(table, f"{path}/pinhole:Pinhole:resolution")
        egos.append(
            replace(
                rgb,
                cam=cam,
                K=first(table, f"{path}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T,
                distortion=first(table, f"{path}/pinhole:simplecv.components.DistortionCoefficients"),
                size=(int(size[0]), int(size[1])),
                rig_T_cam=transform,
            )
        )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(identity.application_id, recording_id=identity.recording_id)
    recording.save(config.output)
    for old, new in (("/world/fit/joints", "/world/pred/soma_x/joints"), ("/world/fit/mesh", "/world/pred/soma_x/mesh")):
        recording.send_chunks(source.stream().filter(content=old).map(lambda chunk, new=new: chunk.with_entity_path(new)))
    projections = []
    for view in phones:
        projected = project(view.camera, points.reshape(-1, 3).astype(np.float64)).reshape(len(times), 77, 3)
        pixels = np.where(projected[:, :, 2:] > 0.001, projected[:, :, :2], np.nan).astype(np.float32)
        projections.append((f"{view.camera.entity}/pinhole", pixels))
    projections.extend((f"/world/rig_04/cam_{ego.cam:02d}/pinhole", project_ego(ego, times, points)) for ego in egos)
    for pinhole, pixels in projections:
        path = f"{pinhole}/pred/soma_x/joints_uv"
        for chunk in annotations:
            recording.send_chunks(chunk.with_entity_path(path))
        recording.log(path, rr.Points2D.from_fields(keypoint_ids=np.arange(77), class_ids=0, colors=[0, 255, 255], radii=2.5), static=True)
        recording.send_columns(
            path,
            indexes=[rr.TimeColumn("video_time", duration=times)],
            columns=rr.Points2D.columns(positions=pixels.reshape(-1, 2)).partition([77] * len(times)),
        )
        print(path, "finite joints:", int(np.isfinite(pixels).all(-1).sum()))
    recording.flush()
    recording.disconnect()
    print(config.output)
