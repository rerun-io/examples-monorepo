"""The run recording: one ``.rrd`` per training run, registered as a segment of the local dataset ``open4dgs-runs``.

Layout (timeline ``frame_index`` = the source segment's frame index; ``step`` = optimizer step):

- ``/world/cameras/{train|holdout}/rig_NN/cam_00``: static Transform3D (world_T_cam) + ``pinhole`` Pinhole;
- ``/world/init_cloud``: the init cloud (``open4dgs.init_cloud``);
- ``/world/people/person_NN/mesh``: the SMPL-X mesh at its frames;
- ``/eval/{holdout|train}/rig_NN_cam_00/{gt,render,error}``: H.264 VideoStreams (error = |raw render - gt * mask|,
  the ``crop_raw`` comparison, times 4, clipped);
- ``/eval/{holdout|train}/rig_NN_cam_00/metrics/<name>``: per-frame Scalars (``open4dgs.evaluate.METRIC_NAMES``);
- ``/eval/holdout/rig_NN_cam_00/floor/{black|mean}/<name>``: the same metrics of the floor renders;
- ``/eval/orbit/render``: the 360-degree orbit video;
- ``/train/<tag>``: loss curves on ``step``;
- recording properties ``run`` (configuration) and ``summary`` (mean metrics, speed).
"""

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer
from simplecv.video_encoder import VideoCodecChoice, VideoEncoder
from torch import Tensor

from open4dgs.catalog_clip import FRAME_TIMELINE, MultiViewClip
from open4dgs.evaluate import METRIC_NAMES, CameraRenders, FrameMetrics

BeartypeException: tuple[type[Exception], ...] = (sys.modules["beartype.roar"].BeartypeException,) if "beartype.roar" in sys.modules else ()
RUNS_DATASET: str = "open4dgs-runs"
"""Local catalog dataset of training runs (one segment per run)."""
ERROR_GAIN: float = 4.0
"""Amplification of the absolute-error video."""


def camera_entity(name: str, held_out: bool) -> str:
    """``/world/cameras/holdout/rig_01/cam_00`` or ``/world/cameras/train/...``."""
    return f"/world/cameras/{'holdout' if held_out else 'train'}/{name}"


def log_cameras(recording: rr.RecordingStream, clip: MultiViewClip, holdout_names: Sequence[str]) -> None:
    """Every clip camera, static, under ``train`` or ``holdout``."""
    for cam, name in enumerate(clip.camera_names):
        entity: str = camera_entity(name, name in holdout_names)
        pose: np.ndarray = clip.world_T_cam[cam].double().numpy()
        recording.log(entity, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=True)
        width, height = clip.sizes[cam]
        recording.log(
            f"{entity}/pinhole",
            rr.Pinhole(
                image_from_camera=clip.K[cam].double().numpy(),
                width=width,
                height=height,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
            ),
            static=True,
        )


def log_people(recording: rr.RecordingStream, clip: MultiViewClip) -> None:
    """The SMPL-X meshes at the frames that hold one."""
    if clip.people is None:
        return
    people = clip.people
    faces: np.ndarray = people.faces.numpy().astype(np.uint32)
    for person, name in enumerate(people.names):
        entity: str = f"/world/people/{name}/mesh"
        recording.log(entity, rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[0.6, 0.6, 0.65, 0.5]), static=True)
        for f in torch.nonzero(people.present[:, person]).reshape(-1).tolist():
            recording.set_time(FRAME_TIMELINE, sequence=int(clip.frame_indices[f]))
            recording.log(entity, rr.Mesh3D.from_fields(vertex_positions=people.vertices[f, person].numpy()))


def log_video(recording: rr.RecordingStream, entity: str, frames: Tensor, frame_indices: Sequence[int], fps: float) -> None:
    """UInt8 (frames, h, w, 3) as an H.264 ``VideoStream`` on ``frame_index`` (odd sizes cropped by one pixel)."""
    height, width = (frames.shape[1] // 2) * 2, (frames.shape[2] // 2) * 2
    encoder: VideoEncoder = VideoEncoder(codec=VideoCodecChoice.H264, fps=fps)
    recording.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

    def send(packets: list[tuple[int, bytes]]) -> None:
        for pts, data in packets:
            recording.set_time(FRAME_TIMELINE, sequence=int(frame_indices[pts]))
            recording.log(entity, rr.VideoStream.from_fields(sample=np.frombuffer(data, dtype=np.uint8)))

    for frame in frames:
        send(encoder.encode_frame(np.ascontiguousarray(frame[:height, :width].numpy())))
    send(encoder.flush())


def log_camera_eval(recording: rr.RecordingStream, clip: MultiViewClip, masks: Sequence[Tensor], renders: CameraRenders, held_out: bool) -> None:
    """Ground truth, render and error videos plus per-frame metric scalars of one camera."""
    cam: int = clip.camera_names.index(renders.camera)
    base: str = f"/eval/{'holdout' if held_out else 'train'}/{renders.camera.replace('/', '_')}"
    frame_indices: list[int] = clip.frame_indices.tolist()
    gt: Tensor = clip.frames[cam]
    mask: Tensor = masks[cam].float()[..., None]
    error: Tensor = ((renders.render.float() - gt.float() * mask).abs() * ERROR_GAIN).clamp(0, 255).to(torch.uint8)
    log_video(recording, f"{base}/gt", gt, frame_indices, clip.fps)
    log_video(recording, f"{base}/render", renders.render, frame_indices, clip.fps)
    log_video(recording, f"{base}/error", error, frame_indices, clip.fps)
    log_metric_rows(recording, f"{base}/metrics", renders.metrics)
    for kind, rows in renders.baselines.items():
        log_metric_rows(recording, f"{base}/floor/{kind}", rows)


def log_metric_rows(recording: rr.RecordingStream, base: str, rows: Sequence[FrameMetrics], names: Sequence[str] = METRIC_NAMES) -> None:
    """Per-frame Scalars ``<base>/<name>`` on ``frame_index``."""
    for name in names:
        recording.send_columns(
            f"{base}/{name}",
            indexes=[rr.TimeColumn(FRAME_TIMELINE, sequence=[row.frame_index for row in rows])],
            columns=rr.Scalars.columns(scalars=np.asarray([getattr(row, name) for row in rows], dtype=np.float64)),
        )


def log_scalars(recording: rr.RecordingStream, scalars: Mapping[str, list[tuple[int, float]]]) -> None:
    """Trainer scalars (loss curves) on the ``step`` timeline."""
    for tag, rows in scalars.items():
        if not rows:
            continue
        recording.send_columns(
            f"/train/{tag}",
            indexes=[rr.TimeColumn("step", sequence=[step for step, _ in rows])],
            columns=rr.Scalars.columns(scalars=np.asarray([value for _, value in rows], dtype=np.float64)),
        )


def register_run(catalog_url: str, rrd_path: Path, dataset: str = RUNS_DATASET) -> None:
    """Register ``rrd_path`` as a segment of ``dataset`` (get-or-create with one retry; replaces a same-id segment)."""
    client: CatalogClient = CatalogClient(catalog_url)
    for attempt in range(2):
        try:
            client.create_dataset(dataset, exist_ok=True)
            break
        except BeartypeException:
            raise
        except Exception:
            if attempt == 1:  # the first failure may be another worker creating the same dataset
                raise
    client.get_dataset(dataset).register([rrd_path.resolve().as_uri()], on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
