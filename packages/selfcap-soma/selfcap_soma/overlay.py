"""Write a layer aligned with SelfCap's original recording and camera tree."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader

from selfcap_soma.camera_geometry import project, project_ego
from selfcap_soma.catalog import ExoSource
from selfcap_soma.evaluation_io import first, nearest_indices, read_egohand_rgb, read_views
from selfcap_soma.observations import overlay_points
from selfcap_soma.overlay_style import observed_links, observed_points


@dataclass(frozen=True, slots=True)
class Config:
    exo_source: ExoSource = "sapiens2"
    """Per-phone image evidence used to classify observed versus inferred joints."""
    source: Path = Path("/tmp/soma-fit/full-segment/fit.rrd")
    """Fitted geometry recording; no input videos are copied."""
    reference: Path = Path("/tmp/selfcap-dev/c4/out/keypoints/selfcap__2025-12-03__d31bf1da__ep001.rrd")
    """Original layer defining the application and recording IDs."""
    output: Path = Path("/tmp/soma-fit/layer/selfcap__2025-12-03__d31bf1da__ep001.rrd")
    """Local layer file; registration is a separate explicit action."""


def log_joint_overlay(
    recording: rr.RecordingStream, path: str, times: Float64[ndarray, "n"], pixels: Float32[ndarray, "n 133 2"], seen: Bool[ndarray, "n 133"]
) -> None:
    """Keep points and individually styled links on the existing joint entity."""
    body_edges = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    hand_edges = [(base, base + 1 + finger * 4) for base in (91, 112) for finger in range(5)] + [
        (base + 1 + finger * 4 + joint, base + 2 + finger * 4 + joint) for base in (91, 112) for finger in range(5) for joint in range(3)
    ]
    foot_edges = [(15, 19), (19, 17), (17, 18), (18, 19), (16, 22), (22, 20), (20, 21), (21, 22)]
    edges = np.asarray(body_edges + hand_edges + foot_edges, dtype=np.int64)
    recording.log(
        path,
        rr.AnnotationContext(rr.ClassDescription(info=rr.AnnotationInfo(id=0))),
        static=True,
    )
    recording.log(path, rr.Points2D.from_fields(keypoint_ids=np.arange(133), class_ids=0), static=True)
    recording.send_columns(
        path,
        indexes=[rr.TimeColumn("video_time", duration=times)],
        columns=rr.Points2D.columns(
            positions=pixels.reshape(-1, 2),
            colors=np.where(seen[..., None], np.array([0, 255, 255, 255]), np.array([140, 140, 140, 110])).astype(np.uint8).reshape(-1, 4),
            radii=np.where(seen, 2.5, 1.1).astype(np.float32).ravel(),
        ).partition([133] * len(times)),
    )
    supported = observed_links(seen, edges)
    recording.send_columns(
        path,
        indexes=[rr.TimeColumn("video_time", duration=times)],
        columns=rr.LineStrips2D.columns(
            strips=pixels[:, edges].reshape(-1, 2, 2),
            colors=np.where(supported[..., None], np.array([0, 255, 255, 255]), np.array([130, 130, 130, 90])).astype(np.uint8).reshape(-1, 4),
            radii=np.where(supported, 1.2, 0.55).astype(np.float32).ravel(),
        ).partition([len(edges)] * len(times)),
    )


def main(config: Config) -> None:
    source = RrdReader(config.source)
    identity = RrdReader(config.reference).recordings()[0]
    segment = "selfcap__2025-12-03__d31bf1da__ep001"
    if identity.recording_id != segment:
        raise ValueError("reference recording must be the requested SelfCap segment")
    rows = []
    for chunk in source.stream().filter(content="/world/fit/joints").to_chunks():
        table = chunk.to_record_batch()
        if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
            times = table.column("video_time").cast("int64").to_numpy().astype(np.float64) / 1e9
            points = np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32)
            rows.append((times, points))
    times = np.concatenate([row[0] for row in rows])
    points = np.concatenate([row[1] for row in rows])
    order = np.argsort(times)
    times, points = times[order], points[order]
    name_chunks = source.stream().filter(content="/parameters/joint_names").to_chunks()
    if not name_chunks:
        raise ValueError("fit recording lacks joint names; rerun soma-fit before exporting overlays")
    names = tuple(name_chunks[0].to_record_batch().column("TextDocument:text")[0].as_py()[0].splitlines())
    points = overlay_points(points, names)
    ear_rows = []
    for chunk in source.stream().filter(content="/world/fit/ears").to_chunks():
        table = chunk.to_record_batch()
        if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
            ear_rows.append(
                (table.column("video_time").cast("int64").to_numpy(), np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32))
            )
    if ear_rows:
        ear_times = np.concatenate([row[0] for row in ear_rows])
        ears = np.concatenate([row[1] for row in ear_rows])[np.argsort(ear_times)]
        if len(ears) != len(times):
            raise ValueError("ear and joint streams differ in length")
        points[:, 3:5] = ears
    foot_rows = []
    quest_rows = []
    for entity, destination in (("/world/fit/feet", foot_rows), ("/world/inputs/quest", quest_rows)):
        for chunk in source.stream().filter(content=entity).to_chunks():
            table = chunk.to_record_batch()
            if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
                destination.append(
                    (
                        table.column("video_time").cast("int64").to_numpy(),
                        np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32),
                    )
                )
    if foot_rows:
        feet = np.concatenate([row[1] for row in foot_rows])[np.argsort(np.concatenate([row[0] for row in foot_rows]))]
        if len(feet) != len(times):
            raise ValueError("foot and joint streams differ in length")
        points[:, 17:23] = feet
    quest = np.full_like(points, np.nan)
    if quest_rows:
        quest = np.concatenate([row[1] for row in quest_rows])[np.argsort(np.concatenate([row[0] for row in quest_rows]))]
        if len(quest) != len(times):
            raise ValueError("Quest and joint streams differ in length")
    dataset = CatalogClient("rerun+http://dgx-spark:51235").get_dataset("selfcap-post-dev")
    phones, rgb = read_views(dataset, segment, times, config.exo_source)
    rgb_evidence = np.full((len(times), 133, 2), np.nan)
    for track in read_egohand_rgb(dataset, segment, times):
        if not len(track.times):
            continue
        indices = nearest_indices(track.times, times)
        valid = abs(track.times[indices] - times) <= 0.05
        base = 91 if track.name.endswith("/left") else 112
        rgb_evidence[valid, base : base + 21] = track.points[indices[valid]]
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
    for old, new in (
        ("/world/fit/joints", "/world/pred/soma_x/joints"),
        ("/world/fit/mesh", "/world/pred/soma_x/mesh"),
        ("/world/fit/ears", "/world/pred/soma_x/ears"),
    ):
        recording.send_chunks(source.stream().filter(content=old).map(lambda chunk, new=new: chunk.with_entity_path(new)))
    projections = []
    for view in phones:
        projected = project(view.camera, points.reshape(-1, 3).astype(np.float64)).reshape(len(times), 133, 3)
        pixels = np.where(projected[:, :, 2:] > 0.001, projected[:, :, :2], np.nan).astype(np.float32)
        seen = observed_points(pixels.astype(np.float64), view.pixels, view.scores)
        projections.append((f"{view.camera.entity}/pinhole", pixels, seen))
    for ego in egos:
        pixels = project_ego(ego, times, points)
        evidence = rgb_evidence if ego.cam == 0 else project_ego(ego, times, quest).astype(np.float64)
        # Mono images have no EgoHand detector: require RGB image support as
        # well as a nearby valid Quest projection, rather than claiming that
        # tracking alone establishes visibility through an occluder.
        if ego.cam != 0:
            evidence[~np.isfinite(rgb_evidence).all(-1)] = np.nan
        seen = observed_points(pixels.astype(np.float64), evidence, np.isfinite(evidence).all(-1).astype(np.float64))
        projections.append((f"/world/rig_04/cam_{ego.cam:02d}/pinhole", pixels, seen))
    for pinhole, pixels, seen in projections:
        path = f"{pinhole}/pred/soma_x/joints_uv"
        log_joint_overlay(recording, path, times, pixels, seen)
        print(path, "observed:", int(seen.sum()), "finite joints:", int(np.isfinite(pixels).all(-1).sum()))
    recording.flush()
    recording.disconnect()
    print(config.output)
