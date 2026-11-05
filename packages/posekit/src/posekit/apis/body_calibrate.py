"""Measure SAM body activation ranges on phone frames read from the Rerun catalog."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rerun.catalog import CatalogClient
from simplecv.catalog_rig_layout import catalog_components, parse_rig_layout, read_rig_kinds
from simplecv.catalog_video import read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from posekit.apis.body_accuracy import load_accuracy_report, read_local_frames
from posekit.apis.body_bench import person_boxes_per_frame
from posekit.models.sam3d_body_quantization import (
    BodyCalibrationSplit,
    CalibrationFrame,
    CalibrationSample,
    frame_sha256,
    measure_body_calibration,
    write_body_calibration,
)
from posekit.predictions import BoxDetections


@dataclass(frozen=True, slots=True)
class CatalogCalibrationSource:
    """An additional capture, read through its catalog dataset."""

    dataset: str
    """Catalog dataset containing the capture."""
    segment: str
    """Exact segment identity."""
    frames_per_camera: int = 4
    """Interior frames from every camera, including egocentric views."""
    timeline: str = "video_time"
    """Timeline carrying the encoded video samples."""


@dataclass(frozen=True, slots=True)
class Config:
    """A reproducible phone calibration split with at least 100 held-out frames."""

    output: Path
    """Destination for measured ranges and the exact split."""
    segment: str
    """Explicit source segment in the catalog."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog containing the source segment."""
    dataset: str = "dataforge-selfcap"
    """Source dataset name."""
    calibration_per_camera: int = 8
    """Frames per phone reserved for activation observation."""
    evaluation_frames: int = 100
    """Total distinct phone frames reserved for accuracy measurement."""
    additional_sources: tuple[CatalogCalibrationSource, ...] = ()
    """Other captures and datasets to broaden activation observations."""
    inputs: tuple[Path, ...] = ()
    """Additional generic images or videos, separate from evaluation assets."""
    frames_per_video: int = 8
    """Frames sampled from each additional local video."""
    evaluation_reports: tuple[Path, ...] = ()
    """Existing accuracy reports whose input identities must remain held out."""
    evaluation_inputs: tuple[Path, ...] = ()
    """Further evaluation images/videos to reserve by decoded pixel hash."""
    evaluation_frames_per_video: int = 20
    """Frames to reserve from each evaluation video."""


@torch.inference_mode()
def main(config: Config) -> None:
    """Discover phones, read their packets together, and calibrate on disjoint samples."""
    dataset = CatalogClient(config.catalog_url).get_dataset(name=config.dataset)
    components = catalog_components(dataset.filter_segments(config.segment).schema())
    layout = parse_rig_layout(components)
    kinds = read_rig_kinds(dataset, config.segment, layout.rigs)
    cameras = [camera for camera in parse_rig_layout(components, kinds).cameras if camera.rig_kind == "exo"]
    if not cameras or config.evaluation_frames < 100:
        raise ValueError("Calibration requires phone cameras and at least 100 held-out evaluation frames.")
    videos = read_catalog_videos(dataset, config.segment, [camera.video_entity for camera in cameras], "video_time")
    selected: list[tuple[str, int, Tensor]] = []
    evaluation: list[CalibrationFrame] = []
    for camera_index, (camera, video) in enumerate(zip(cameras, videos, strict=True)):
        fps: float = 1e9 / float(np.median(np.diff(video.t_ns)))
        decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, round(fps), codec=video.codec), device="cuda", dimension_order="NHWC", seek_mode="approximate")
        count: int = config.evaluation_frames // len(cameras) + int(camera_index < config.evaluation_frames % len(cameras))
        total: int = count + config.calibration_per_camera
        if len(decoder) < total + 2:
            raise ValueError(f"{camera.video_entity} has only {len(decoder)} frames; cannot form the requested split.")
        indices: list[int] = np.linspace(0, len(decoder) - 1, total + 2, dtype=np.int64)[1:-1].tolist()
        calibration_positions: set[int] = set(np.linspace(0, total - 1, config.calibration_per_camera, dtype=np.int64).tolist())
        source: str = f"{config.segment}{camera.video_entity}"
        frames = decoder.get_frames_at(indices).data.contiguous()
        for position, index in enumerate(indices):
            frame = frames[position].clone()
            if position in calibration_positions:
                selected.append((source, index, frame))
            else:
                digest: str = frame_sha256(frame)
                evaluation.append(CalibrationFrame(source, index, digest))
        print(f"[body-calibrate] {source}: {len(decoder)} frames; {len(calibration_positions)} calibration, {count} evaluation", flush=True)
    for extra_source in config.additional_sources:
        extra_dataset = CatalogClient(config.catalog_url).get_dataset(name=extra_source.dataset)
        extra_cameras = parse_rig_layout(catalog_components(extra_dataset.filter_segments(extra_source.segment).schema())).cameras
        entities = [camera.video_entity for camera in extra_cameras]
        if not entities or extra_source.frames_per_camera < 1:
            raise ValueError(f"No calibration cameras or invalid frame count for {extra_source.segment}")
        extra_videos = read_catalog_videos(extra_dataset, extra_source.segment, entities, extra_source.timeline)
        for entity, video in zip(entities, extra_videos, strict=True):
            fps = 1e9 / float(np.median(np.diff(video.t_ns)))
            decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, round(fps), codec=video.codec), device="cuda", dimension_order="NHWC", seek_mode="approximate")
            indices = np.unique(np.linspace(0, len(decoder) - 1, extra_source.frames_per_camera + 2, dtype=np.int64)[1:-1]).tolist()
            frames = decoder.get_frames_at(indices).data.contiguous()
            selected.extend((f"{extra_source.segment}{entity}", index, frame.clone()) for index, frame in zip(indices, frames, strict=True))
            print(f"[body-calibrate] additional {extra_source.dataset}/{extra_source.segment}{entity}: {len(indices)} frames", flush=True)
    selected.extend((row.source, row.index, frame) for row, frame in read_local_frames(config.inputs, config.frames_per_video))
    for path in config.evaluation_reports:
        evaluation.extend(load_accuracy_report(path).frames)
    evaluation.extend(row for row, _ in read_local_frames(config.evaluation_inputs, config.evaluation_frames_per_video))
    evaluation = list({(row.source, row.index, row.sha256): row for row in evaluation}.values())
    BodyCalibrationSplit([
        CalibrationFrame(source, index, frame_sha256(frame))
        for source, index, frame in selected
    ], evaluation)
    boxes: list[BoxDetections] = person_boxes_per_frame([frame for _, _, frame in selected])
    samples: list[CalibrationSample] = [
        CalibrationSample(source, index, frame, frame_boxes) for (source, index, frame), frame_boxes in zip(selected, boxes, strict=True)
    ]
    calibration = measure_body_calibration(samples, evaluation)
    write_body_calibration(config.output, calibration)
    print(f"[body-calibrate] saved {calibration.body_crops} body and {2 * calibration.body_crops} hand crops to {config.output}", flush=True)
