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

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from rerun.catalog import CatalogClient
from serde import from_dict, serde
from simplecv.catalog_layers import ensure_dataset, register_layer
from simplecv.video_encoder import VideoCodecChoice, VideoEncoder
from torch import Tensor

from open4dgs.catalog_clip import FRAME_TIMELINE, MultiViewClip
from open4dgs.evaluate import METRIC_NAMES, CameraRenders, FrameMetrics
from open4dgs.masks import MATTE_LAYER
from open4dgs.prior import PriorConfig
from open4dgs.properties import Cells, PropertyValue, floats, one
from open4dgs.trigs_adapter import TrainConfig

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
    """Register ``rrd_path`` as a segment of ``dataset`` (``ensure_dataset``: get-or-create with one retry; replaces a same-id segment)."""
    register_layer(ensure_dataset(CatalogClient(catalog_url), dataset), [rrd_path], None)


def _flat_config(prefix: str, config: TrainConfig | PriorConfig) -> dict[str, PropertyValue]:
    """``<prefix>_<field>`` per set field; tuples comma-joined (the run schema on open4dgs-runs)."""
    return {
        f"{prefix}_{k}": ",".join(str(x) for x in v) if isinstance(v, tuple) else v
        for k, v in dataclasses.asdict(config).items()
        if v is not None
    }


def _config_from_cells[C: (TrainConfig, PriorConfig)](kind: type[C], prefix: str, cells: Cells) -> C:
    """``kind`` from the ``<prefix>_<field>`` cells that are present (older runs lack some: those keep the default)."""
    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(kind):
        value: object = one(cells, f"{prefix}_{f.name}")
        if value is None:
            continue
        kwargs[f.name] = floats(str(value)) if isinstance(f.default, tuple) else value
    return from_dict(kind, kwargs)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunProperties:
    """The run recording's ``run`` property (``open4dgs-train``): what was trained on, how, and with which settings.

    Runs on open4dgs-runs from before a field existed read with its default.
    """

    dataset: str
    segment_id: str
    catalog_url: str
    frame_start: int
    frame_end: int
    num_frames: int
    cameras: tuple[str, ...]
    train_cameras: tuple[str, ...]
    holdout_cameras: tuple[str, ...]
    holdout_seed: int = 0
    max_side: int | None = None
    roi_pad: float | None = None
    """The people ROI's pad; None = full frames."""
    mask_dataset: str | None = None
    """Dataset of the ``people_matte`` layer the masks came from; None = mesh silhouettes."""
    init_method: str = "body_surface"
    """``<method>`` (body init), ``layer:<layer>:<method>`` or ``rrd:<method>``: a label, not read back."""
    init_rrd: str = ""
    selfcap_roi: bool = False
    selfcap_time_start_s: float | None = None
    view_coordinates: str = ""
    samples_per_person: int = 0
    keyframe_step: int = 3
    prior_mode: str = "off"
    prior_source: str = "off"
    prior_config: PriorConfig | None = None
    """Set when the prior was on."""
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    train_time_offset_clamp_ms_resolved: float | None = None
    rs_signs_per_camera: tuple[float, ...] = ()
    selfcap_capture_times: bool = False
    holdout_fit_steps: int = 0
    holdout_protocol: str = ""

    def to_properties(self) -> dict[str, PropertyValue]:
        out: dict[str, PropertyValue] = {
            "dataset": self.dataset,
            "segment_id": self.segment_id,
            "catalog_url": self.catalog_url,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "num_frames": self.num_frames,
            "cameras": list(self.cameras),
            "train_cameras": list(self.train_cameras),
            "holdout_cameras": list(self.holdout_cameras),
            "holdout_seed": self.holdout_seed,
            "max_side": self.max_side or 0,
            "roi": "full" if self.roi_pad is None else f"people:{self.roi_pad}",
            "init_method": self.init_method,
            "init_rrd": self.init_rrd,
            "selfcap_roi": self.selfcap_roi,
            "selfcap_time_start_s": -1.0 if self.selfcap_time_start_s is None else self.selfcap_time_start_s,
            "view_coordinates": self.view_coordinates,
            "samples_per_person": self.samples_per_person,
            "keyframe_step": self.keyframe_step,
            "prior": self.prior_config is not None,  # a bool, as in every run already on open4dgs-runs
            "prior_mode": self.prior_mode,
            "prior_source": self.prior_source,
            **(_flat_config("prior", self.prior_config) if self.prior_config is not None else {}),
            "mask_source": "mesh" if self.mask_dataset is None else f"layer:{self.mask_dataset}/{MATTE_LAYER}",
            **_flat_config("train", self.train),
            "rs_signs_per_camera": ",".join(f"{v:g}" for v in self.rs_signs_per_camera),
            "selfcap_capture_times": self.selfcap_capture_times,
            "holdout_fit_steps": self.holdout_fit_steps,
            "holdout_protocol": self.holdout_protocol,
        }
        if self.train_time_offset_clamp_ms_resolved is not None:
            out["train_time_offset_clamp_ms_resolved"] = self.train_time_offset_clamp_ms_resolved
        return out

    @classmethod
    def from_cells(cls, cells: Cells) -> "RunProperties":
        """The run property of a recording (``rrd_cells(reader, "run")``) or a catalog row (``row_cells(row, "run")``)."""
        roi: str = str(one(cells, "roi", "full"))
        mask_source: str = str(one(cells, "mask_source", "mesh"))
        start_s = one(cells, "selfcap_time_start_s", -1.0)
        clamp = one(cells, "train_time_offset_clamp_ms_resolved")
        values: dict[str, object] = {
            "dataset": one(cells, "dataset"),
            "segment_id": one(cells, "segment_id"),
            "catalog_url": one(cells, "catalog_url"),
            "frame_start": one(cells, "frame_start"),
            "frame_end": one(cells, "frame_end"),
            "num_frames": one(cells, "num_frames"),
            "cameras": tuple(str(c) for c in cells.get("cameras", ())),
            "train_cameras": tuple(str(c) for c in cells.get("train_cameras", ())),
            "holdout_cameras": tuple(str(c) for c in cells.get("holdout_cameras", ())),
            "holdout_seed": one(cells, "holdout_seed", 0),
            "max_side": one(cells, "max_side", 0) or None,
            "roi_pad": None if roi == "full" else float(roi.split(":")[1]),
            # "layer:<dataset>/masks" (older runs) or "layer:<dataset>/people_matte"
            "mask_dataset": None if mask_source == "mesh" else mask_source.removeprefix("layer:").rsplit("/", 1)[0],
            "init_method": one(cells, "init_method", "body_surface"),
            "init_rrd": one(cells, "init_rrd", ""),
            "selfcap_roi": one(cells, "selfcap_roi", False),
            "selfcap_time_start_s": None if start_s == -1.0 else start_s,
            "view_coordinates": one(cells, "view_coordinates", ""),
            "samples_per_person": one(cells, "samples_per_person", 0),
            "keyframe_step": one(cells, "keyframe_step", 3),
            "prior_mode": one(cells, "prior_mode", "on" if one(cells, "prior", False) else "off"),
            "prior_source": one(cells, "prior_source", "off"),
            "train_time_offset_clamp_ms_resolved": clamp,
            "rs_signs_per_camera": floats(str(one(cells, "rs_signs_per_camera", ""))),
            "selfcap_capture_times": one(cells, "selfcap_capture_times", False),
            "holdout_fit_steps": one(cells, "holdout_fit_steps", 0),
            "holdout_protocol": one(cells, "holdout_protocol", ""),
        }
        record: RunProperties = from_dict(cls, values)
        prior: PriorConfig | None = _config_from_cells(PriorConfig, "prior", cells) if one(cells, "prior", False) else None
        return dataclasses.replace(record, prior_config=prior, train=_config_from_cells(TrainConfig, "train", cells))
