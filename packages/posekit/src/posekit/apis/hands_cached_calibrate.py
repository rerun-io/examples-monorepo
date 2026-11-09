"""Measure detector and pose activation ranges on disjoint cached catalog recordings."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor
from trtkit.trt_builder import onnx_content_hash

from posekit.apis.hands_eval import decode_samples
from posekit.eval.cached_hand_gt import load_cached_hand_gt
from posekit.eval.hand_gt import HandGtSegment
from posekit.models.calibration_rrd import CalibrationColumn, CalibrationRecording, range_tables, read_calibration_rrd, write_calibration_rrd
from posekit.models.dinov3_calibration import DinoActivationObserver
from posekit.models.egohand import (
    EgoHandAssetPaths,
    EgoHandConfig,
    EgoHandDetector,
    EgoHandDetectorConfig,
    EgoHandPredictor,
    _resize_with_pil,
    asset_paths,
)
from posekit.models.egohand_detector_quantization import DETECTOR_CALIBRATION_MODEL, detector_activation_model
from posekit.models.egohand_quantization import EGOHAND_CALIBRATION_MODEL, EgoHandCalibration, write_calibration
from posekit.predictions import HandDetections


@dataclass(frozen=True, slots=True)
class Config:
    """Offline calibration from the fixed cache, with no catalog traffic."""

    cache: Path = Path.home() / "egohand-3060/cache"
    """Native camera-scoped source recordings."""
    assets: Path = Path.home() / "egohand-3060/assets"
    """Pinned local model assets."""
    output: Path = Path.home() / "egohand-3060/calibration"
    """Activation RRDs and temporary calibration graph."""
    max_frames: int = 40
    """Visible-GT frames from each of the five disjoint calibration recordings."""


@torch.inference_mode()
def main(config: Config) -> None:
    """Hash held-out frames, reject overlaps, then observe the released default path."""
    config.output.mkdir(parents=True, exist_ok=True)
    paths: EgoHandAssetPaths = asset_paths(config.assets)
    detector_config: EgoHandDetectorConfig = EgoHandDetectorConfig(weights=paths.detector)
    weights: Path = detector_config.weights_path()
    original: onnx.ModelProto = onnx.load(weights)
    graph, names = detector_activation_model(original)
    graph_path: Path = config.output / "detector-observer.onnx"
    onnx.save(graph, graph_path)
    options: ort.SessionOptions = ort.SessionOptions()
    options.log_severity_level = 3
    session: ort.InferenceSession = ort.InferenceSession(str(graph_path), sess_options=options, providers=["CUDAExecutionProvider"])
    print("Calibration providers:", session.get_providers(), flush=True)
    forbidden: set[str] = set()
    evaluation: list[Path] = sorted(config.cache.glob("*/eval/*.rrd"))
    calibration: list[Path] = sorted(config.cache.glob("*/calibration/*.rrd"))
    if {p.stem for p in evaluation} & {p.stem for p in calibration}:
        raise ValueError("Calibration and evaluation segments overlap.")
    for path in evaluation:
        frames: UInt8[Tensor, "f h w 3"] = decode_samples(load_cached_hand_gt(path, max_frames=100))
        forbidden.update(hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest() for frame in frames)
        del frames
        print("Excluded", path.stem, flush=True)
    detector: EgoHandDetector = detector_config.setup()
    predictor: EgoHandPredictor = EgoHandConfig(weights=paths.weights, soma_assets=paths.soma).setup()
    sources: list[str] = []
    indices: list[int] = []
    hashes: list[str] = []
    maxima: Float64[ndarray, "m"] = np.zeros(len(names), dtype=np.float64)
    crops: int = 0
    with DinoActivationObserver(predictor.predictor.model.backbone) as observer:
        for path in calibration:
            segment: HandGtSegment = load_cached_hand_gt(path, max_frames=config.max_frames)
            frames = decode_samples(segment)
            for frame_index, frame in zip(segment.sample_indices, frames, strict=True):
                digest: str = hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest()
                if digest in forbidden:
                    raise ValueError(f"{path}:{frame_index} is a decoded evaluation frame.")
                batch: UInt8[Tensor, "1 h w 3"] = frame[None]
                resized: Float32[ndarray, "1 3 640 640"] = _resize_with_pil(batch).cpu().numpy()
                outputs: list[Float32[ndarray, ""]] = cast(list[Float32[ndarray, ""]], session.run(
                    [f"calibration_{i}" for i in range(len(names))],
                    {"images": resized, "orig_target_sizes": np.array([[frame.shape[1], frame.shape[0]]], dtype=np.int64)},
                )
                )
                maxima = np.maximum(maxima, np.asarray(outputs).reshape(-1))
                detections: HandDetections = detector(batch)
                predictor(batch, detections, segment.camera)
                crops += detections.num_detections
                sources.append(f"{path.parent.parent.name}/{path.stem}")
                indices.append(int(frame_index))
                hashes.append(digest)
                print("CALIBRATE", sources[-1], indices[-1], "hands", detections.num_detections, flush=True)
            del frames
        linear, attention = observer.ranges()
    # A composite recording has its own ordinal clock; retain native source identities beside it.
    measured: EgoHandCalibration = EgoHandCalibration(
        "egohand-catalog-calibration-v1", list(range(len(hashes))), [], hashes, crops, linear, attention
    )
    pose_path: Path = config.output / "pose.rrd"
    write_calibration(pose_path, measured)
    record: CalibrationRecording = read_calibration_rrd(pose_path, EGOHAND_CALIBRATION_MODEL)
    source_table: dict[str, CalibrationColumn] = {"source": sources, "index": indices, "sha256": hashes}
    write_calibration_rrd(pose_path, CalibrationRecording(record.properties, {**record.tables, "/native_sources": source_table}))
    detector_ranges: dict[str, float] = {name: float(value) for name, value in zip(names, maxima, strict=True)}
    write_calibration_rrd(
        config.output / "detector.rrd",
        CalibrationRecording(
            {
                "model": DETECTOR_CALIBRATION_MODEL,
                "source_sha256": onnx_content_hash(weights, cache_dir=paths.engines),
                "frames": len(hashes),
                "excluded_eval_frames": len(forbidden),
            },
            {**range_tables(detector_ranges, {}), "/split/calibration": source_table},
        ),
    )
    print("CALIBRATION_READY", len(hashes), "frames", crops, "crops", flush=True)
