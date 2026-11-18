"""Side-by-side DetNet-F boxes on RoboCap clips: a ``detnet_compare`` layer per segment, and a six-camera blueprint.

Each camera's whole video is decoded once on NVDEC (simplecv's segment decoder) and turned to luma; every model sees it in its
own input convention (the legacy letterbox, or the upright turns of ``catalog.UPRIGHT_TURNS``), as ``detector_eval`` scores it.
A positive slot's circle is mapped back to stored pixels and logged as the square around it at
``<camera>/pinhole/detnet/<name>`` on ``frame_index`` and ``video_time``; a frame with no positive slot logs an empty set, so
no box outlives its frame. The blueprint shows each camera's ``VideoStream`` with every model's boxes as their own entity.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from handtrack.data.catalog import CATALOG_URL
from handtrack.data.luma import grey_from_rgb
from handtrack.data.rerun_source import NOMINAL_FPS, SourceSpec, _frame_times, camera_letterbox, source_spec
from handtrack.eval.boxes import PRESENCE_THRESHOLD
from handtrack.geometry.letterbox import Letterbox
from handtrack.models.detnet import DetNetF, DetNetOutput, decode_detections
from handtrack.train.checkpoint import load_weights

COLOURS: tuple[tuple[int, int, int], ...] = ((255, 80, 80), (80, 160, 255), (255, 200, 0), (200, 120, 255), (0, 220, 220))
"""Box colour by the model's position in ``names`` (red, blue, yellow, ...)."""
SIDES: str = "LR"
InputConvention: TypeAlias = Literal["legacy", "upright", "cap"]
"""How a model sees the frames: ``legacy`` letterbox (frames as stored), ``upright`` (``catalog.UPRIGHT_TURNS``), or ``cap``
(``CAP_TURNS``, what robocap-live's ``mix1-upright`` feeds the detector on the cap)."""
CAP_TURNS: tuple[int, ...] = (0, 0, 2, 2, 0, 0)
"""robocap-live's mix1-upright input: RoboCap cameras 2-3 turned 180 degrees (their stored frames are already upright)."""


@dataclass(frozen=True, slots=True)
class DetNetCompareConfig:
    """Write (and optionally register) a ``detnet_compare`` layer of DetNet-F boxes for each of a dataset's segments."""

    segments: tuple[str, ...]
    """Segment ids of ``dataset``."""
    names: tuple[str, ...]
    """One label per model; it names the entity (``detnet/<name>``) and fixes the colour by position."""
    weights: tuple[Path, ...]
    """Model-only state_dicts (``*.weights.pt`` with ``.sha256`` sidecars), one per name."""
    inputs: tuple[InputConvention, ...]
    """Per model, the input convention it is scored in (each model in the one it was trained or deployed with)."""
    out_dir: Path
    """Receives ``<segment>__<names>.rrd``; a new model set writes a new file (a registered file is never overwritten)."""
    dataset: str = "robocap-hands"
    register: bool = False
    """Register each file as the segment's ``layer`` (replacing an earlier one)."""
    blueprint: bool = False
    """Also save ``<out_dir>/<dataset>__<names>.rbl`` and, with ``register``, make it the dataset's default blueprint."""
    layer: str = "detnet_compare"
    catalog_url: str = CATALOG_URL
    device: str = "cuda:0"
    frames_per_call: int = 64
    """Frames decoded per NVDEC call (64 x 1080p RGB = 400 MB before the luma reduces it)."""

    def __post_init__(self) -> None:
        if not self.names or not (len(self.names) == len(self.weights) == len(self.inputs)) or len(set(self.names)) != len(self.names):
            raise ValueError("need one unique name, weights file and input convention per model")


@dataclass(frozen=True, slots=True)
class CameraBoxes:
    """One model's detections on a camera's frames, in stored pixels (slot 0 = left hand)."""

    centers_px: Float32[ndarray, "n 2 2"]
    half_px: Float32[ndarray, "n 2"]
    """Half the side of the square around each circle (its radius)."""
    probability: Float32[ndarray, "n 2"]


@dataclass(frozen=True, slots=True)
class BoxRows:
    """Positive slots as flat Boxes2D components, with the number of boxes on each frame."""

    lengths: Int64[ndarray, "n"]
    centers_px: Float32[ndarray, "k 2"]
    half_sizes_px: Float32[ndarray, "k 2"]
    labels: list[str]
    class_ids: Int64[ndarray, "k"]
    """0 = left, 1 = right."""


def stored_boxes(output: DetNetOutput, letterbox: Letterbox) -> CameraBoxes:
    """Map net-frame circles back through the camera's letterbox (turn, scale, pad) to stored pixels."""
    detections = decode_detections(output, PRESENCE_THRESHOLD)
    centers: Float32[Tensor, "n 2 2"] = letterbox.from_net(detections.circle[..., :2].float())
    radii: Float32[Tensor, "n 2"] = detections.circle[..., 2].float() / letterbox.scale
    return CameraBoxes(centers.cpu().numpy(), radii.cpu().numpy(), detections.probability.float().cpu().numpy())


def box_rows(boxes: CameraBoxes, name: str) -> BoxRows:
    """Keep the slots whose presence probability exceeds ``PRESENCE_THRESHOLD``, labelled ``<name> <side> <probability>``."""
    positive: np.ndarray = boxes.probability > PRESENCE_THRESHOLD
    frame, side = np.nonzero(positive)
    half: Float32[ndarray, "k"] = boxes.half_px[frame, side]
    return BoxRows(lengths=positive.sum(axis=1).astype(np.int64), centers_px=boxes.centers_px[frame, side],
                   half_sizes_px=np.stack((half, half), axis=-1), class_ids=side.astype(np.int64),
                   labels=[f"{name} {SIDES[s]} {p:.2f}" for s, p in zip(side.tolist(), boxes.probability[frame, side].tolist(), strict=True)])


def compare_blueprint(cameras: tuple[str, ...]):
    """Three columns (front cams 0-1, down cams 2-3, side cams 4-5), each camera's video with every model's boxes and the teacher."""
    import rerun.blueprint as rrb

    def view(camera: str):
        return rrb.Spatial2DView(origin=f"{camera}/pinhole", name=camera.rsplit("/", 1)[-1],
                                 contents=["+ $origin/video", "+ $origin/detnet/**", "+ $origin/hand_teacher"])

    columns = [rrb.Vertical(view(cameras[c]), view(cameras[c + 1])) for c in range(0, len(cameras), 2)]
    return rrb.Blueprint(rrb.Horizontal(*columns), rrb.TimePanel(timeline="frame_index", state="collapsed"), collapse_panels=True)


@torch.no_grad()
def write_segment(config: DetNetCompareConfig, dataset, segment: str, models: list[DetNetF], path: Path) -> None:
    """Decode each camera of one segment once and log every model's boxes into ``path``."""
    import rerun as rr
    from simplecv.rerun_dataloader import open_segment_decoder

    device: torch.device = torch.device(config.device)
    specs: list[SourceSpec] = [source_spec(config.dataset, 0, convention != "legacy") for convention in config.inputs]
    specs = [replace(spec, quarter_turns=CAP_TURNS) if convention == "cap" else spec for spec, convention in zip(specs, config.inputs, strict=True)]
    partial: Path = path.with_suffix(".partial.rrd")
    rec = rr.RecordingStream(config.dataset, recording_id=segment)
    rec.save(partial)
    for c, camera in enumerate(specs[0].cameras):
        frames, times = _frame_times(dataset, segment, camera)
        decoded_index, _, _, decoder = open_segment_decoder(dataset, segment, f"{camera}/pinhole/video".lstrip("/"), "frame_index", device, NOMINAL_FPS)
        if not np.array_equal(np.asarray(decoded_index, dtype=np.int64), frames):
            raise ValueError(f"{segment} {camera}: decoder samples and frame_index rows disagree")
        found: list[list[CameraBoxes]] = [[] for _ in models]
        for first in range(0, len(frames), config.frames_per_call):
            rgb: UInt8[Tensor, "b 3 h w"] = decoder.get_frames_at(list(range(first, min(first + config.frames_per_call, len(frames))))).data
            grey: UInt8[Tensor, "b h w"] = grey_from_rgb(rgb)
            for m, model in enumerate(models):
                letterbox: Letterbox = camera_letterbox(specs[m], c, int(grey.shape[-1]), int(grey.shape[-2]))
                net: Float32[Tensor, "b 1 480 640"] = letterbox.apply(grey)[:, None].float() / 255.0
                found[m].append(stored_boxes(model(net), letterbox))
        indexes = [rr.TimeColumn("frame_index", sequence=frames), rr.TimeColumn("video_time", duration=times.astype("timedelta64[ns]"))]
        counts: list[int] = []
        for m, name in enumerate(config.names):
            boxes = CameraBoxes(*(np.concatenate([getattr(b, f) for b in found[m]]) for f in ("centers_px", "half_px", "probability")))
            rows: BoxRows = box_rows(boxes, name)
            columns = rr.Boxes2D.columns(centers=rows.centers_px, half_sizes=rows.half_sizes_px, labels=rows.labels, class_ids=rows.class_ids,
                                         colors=np.tile(np.asarray(COLOURS[m % len(COLOURS)], dtype=np.uint8), (len(rows.labels), 1)))
            rec.send_columns(f"{camera}/pinhole/detnet/{name}", indexes=indexes, columns=columns.partition(rows.lengths))
            counts.append(len(rows.labels))
        print(f"  {segment} {camera}: {len(frames)} frames, boxes per model {dict(zip(config.names, counts, strict=True))}", flush=True)
    rec.flush()
    rec.disconnect()
    partial.chmod(0o644)
    partial.replace(path)


def main(config: DetNetCompareConfig) -> list[Path]:
    """Write one layer file per segment (and the blueprint); register them when asked. Returns the layer files."""
    import rerun as rr
    from rerun.catalog import OnDuplicateSegmentLayer

    device: torch.device = torch.device(config.device)
    models: list[DetNetF] = []
    for name, weights in zip(config.names, config.weights, strict=True):
        model: DetNetF = DetNetF()
        print(f"{name}: {weights} sha256 {load_weights(model, weights)}", flush=True)
        models.append(model.to(device).eval())
    entry = rr.catalog.CatalogClient(config.catalog_url).get_dataset(config.dataset)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    tag: str = "-".join(config.names)
    paths: list[Path] = []
    for segment in config.segments:
        path: Path = config.out_dir / f"{segment}__{tag}.rrd"
        write_segment(config, entry, segment, models, path)
        paths.append(path)
        if config.register:
            entry.register([path.resolve().as_uri()], layer_name=config.layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
            print(f"  registered {segment} layer {config.layer}", flush=True)
    if config.blueprint:
        blueprint_path: Path = config.out_dir / f"{config.dataset}__{tag}.rbl"
        compare_blueprint(source_spec(config.dataset, 0).cameras).save(config.dataset, str(blueprint_path))
        blueprint_path.chmod(0o644)
        if config.register:
            entry.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)
    return paths
