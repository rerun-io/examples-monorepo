"""Stage B: per-view 2D whole-body keypoints on the exo videos.

For every exo camera, frames stream from the catalog (NVDEC), YOLOX keeps the
single best person box per frame (Assembly101 has one subject), and Sapiens2
produces COCO-133 keypoints with confidences. Raw results plus the crop
rectangles (needed by the AssemblyHands-X margin rule downstream) go to one NPZ
per camera; a subsampled ``Points2DWithConfidence`` overlay is registered as
the ``exocalib_kp2d`` layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32, Int64
from numpy import ndarray
from simplecv.rerun_custom_types import Points2DWithConfidence

from exo_calib.catalog_io import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    TIMELINE,
    ExoVideoStreams,
    connect_dataset,
    new_layer_recording,
    only_segment_id,
    open_exo_streams,
    register_layer,
)

PoseBackendName = Literal["tensorrt", "onnxruntime", "torch"]


@dataclass
class Keypoints2dConfig:
    """Config for the Stage B 2D keypoint sweep."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset holding the registered segment."""
    segment_id: str | None = None
    """Segment to process; ``None`` uses the dataset's single segment."""
    window_s: float = 30.0
    """Processed duration from the start of the segment."""
    frame_stride: int = 1
    """Keep every ``frame_stride``-th sample inside the window."""
    batch_size: int = 32
    """Frames decoded and inferred per chunk."""
    model_size: Literal["0.4B", "0.8B", "1B"] = "1B"
    """Sapiens2 checkpoint size."""
    pose_backend: PoseBackendName = "tensorrt"
    """Backend for the Sapiens2 pose network."""
    detection_score_thr: float = 0.5
    """Minimum YOLOX person score."""
    output_dir: Path = Path("data/outputs")
    """Directory for ``<segment>/kp2d/<cam>.npz`` and the layer RRD."""
    layer_name: str = "exocalib_kp2d"
    """Catalog layer name for the keypoint overlays."""
    application_id: str = "exocalib"
    """Application id of generated layer recordings."""
    register: bool = True
    """Register the layer RRD into the catalog after writing it."""
    log_stride: int = 5
    """Log every ``log_stride``-th processed frame into the layer."""


@dataclass(slots=True)
class CameraKeypoints:
    """Raw Stage B output for one camera."""

    sample_indices: Int64[ndarray, "t"]
    """Decoder sample indices of the processed frames."""
    times_ns: Int64[ndarray, "t"]
    """Timeline timestamps (nanoseconds) of the processed frames."""
    kp_xy: Float32[ndarray, "t 133 2"]
    """Image-space keypoints; NaN where no detection."""
    conf: Float32[ndarray, "t 133"]
    """Raw Sapiens2 keypoint confidences; 0 where no detection."""
    bbox_xyxy: Float32[ndarray, "t 4"]
    """Best person box per frame; NaN where no detection."""
    crop_origin_xy: Float32[ndarray, "t 2"]
    """Image-space origin of the model crop rectangle."""
    crop_size_wh: Float32[ndarray, "t 2"]
    """Image-space size of the model crop rectangle."""

    def save(self, npz_path: Path) -> None:
        """Write the arrays to ``npz_path``."""
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            sample_indices=self.sample_indices,
            times_ns=self.times_ns,
            kp_xy=self.kp_xy,
            conf=self.conf,
            bbox_xyxy=self.bbox_xyxy,
            crop_origin_xy=self.crop_origin_xy,
            crop_size_wh=self.crop_size_wh,
        )

    @classmethod
    def load(cls, npz_path: Path) -> "CameraKeypoints":
        """Read arrays written by :meth:`save`."""
        data = np.load(npz_path)
        return cls(**{key: data[key] for key in data.files})


def _best_box_per_frame(detections: Any, num_frames: int) -> tuple[Float32[ndarray, "t 4"], Int64[ndarray, " n_det"]]:
    """Pick the highest-scoring box per frame; return boxes (NaN when absent) and kept row indices."""
    xyxy: Float32[ndarray, "n 4"] = detections.xyxy.cpu().numpy()
    scores: Float32[ndarray, "n"] = detections.scores.cpu().numpy()
    frame_indices: Int64[ndarray, "n"] = detections.frame_indices.cpu().numpy()
    best_xyxy: Float32[ndarray, "t 4"] = np.full((num_frames, 4), np.nan, dtype=np.float32)
    kept_rows: list[int] = []
    for frame in range(num_frames):
        rows: Int64[ndarray, " m"] = np.nonzero(frame_indices == frame)[0]
        if rows.size == 0:
            continue
        best_row: int = int(rows[np.argmax(scores[rows])])
        best_xyxy[frame] = xyxy[best_row]
        kept_rows.append(best_row)
    return best_xyxy, np.asarray(kept_rows, dtype=np.int64)


def run_camera_sweep(
    streams: ExoVideoStreams,
    cam_idx: int,
    detector: Any,
    pose: Any,
    config: Keypoints2dConfig,
) -> CameraKeypoints:
    """Run detection + pose over one camera's frame window.

    Args:
        streams: Open NVDEC decoders over the segment.
        cam_idx: Camera index into ``streams``.
        detector: posekit person detector (``YoloxDetector``).
        pose: posekit top-down pose model (``SapiensPose2d``).
        config: Stage configuration.

    Returns:
        Raw keypoints, confidences, boxes, and crop rectangles for the window.
    """
    from posekit.ops.crops import bbox_xyxy_to_center_scale
    from posekit.predictions import BoxDetections

    times = streams.times_ns[cam_idx]
    dt_ns: float = float(np.median(np.diff(times)).astype("timedelta64[ns]").astype(np.int64))
    n_window: int = min(int(round(config.window_s * 1e9 / dt_ns)), len(times))
    sample_indices: Int64[ndarray, "t"] = np.arange(0, n_window, config.frame_stride, dtype=np.int64)
    num_frames: int = len(sample_indices)

    kp_xy: Float32[ndarray, "t 133 2"] = np.full((num_frames, 133, 2), np.nan, dtype=np.float32)
    conf: Float32[ndarray, "t 133"] = np.zeros((num_frames, 133), dtype=np.float32)
    bbox_xyxy: Float32[ndarray, "t 4"] = np.full((num_frames, 4), np.nan, dtype=np.float32)

    decoder = streams.decoders[cam_idx]
    for start in range(0, num_frames, config.batch_size):
        chunk: Int64[ndarray, " b"] = sample_indices[start : start + config.batch_size]
        frames_nchw: torch.Tensor = decoder.get_frames_at(chunk.tolist()).data
        frames_nhwc: torch.Tensor = frames_nchw.permute(0, 2, 3, 1).contiguous()
        detections = detector(frames_nhwc)
        best_xyxy, kept_rows = _best_box_per_frame(detections, len(chunk))
        bbox_xyxy[start : start + len(chunk)] = best_xyxy
        if kept_rows.size == 0:
            continue
        kept = BoxDetections(
            xyxy=detections.xyxy[kept_rows],
            scores=detections.scores[kept_rows],
            frame_indices=detections.frame_indices[kept_rows],
        )
        keypoints = pose(frames_nhwc, kept)
        kp_frame_indices: Int64[ndarray, " n"] = keypoints.frame_indices.cpu().numpy()
        kp_xy[start + kp_frame_indices] = keypoints.xy.cpu().numpy()
        conf[start + kp_frame_indices] = keypoints.scores.cpu().numpy()

    # Crop rectangles exactly as posekit prepares them (center/scale convention).
    valid_box: ndarray = np.isfinite(bbox_xyxy).all(axis=1)
    crop_origin_xy: Float32[ndarray, "t 2"] = np.full((num_frames, 2), np.nan, dtype=np.float32)
    crop_size_wh: Float32[ndarray, "t 2"] = np.full((num_frames, 2), np.nan, dtype=np.float32)
    if valid_box.any():
        crop_w, crop_h = pose.crop_spec.input_size
        centers, scales = bbox_xyxy_to_center_scale(
            torch.as_tensor(bbox_xyxy[valid_box]), aspect_wh=float(crop_w) / float(crop_h), padding=pose.crop_spec.padding
        )
        centers_np: Float32[ndarray, "m 2"] = centers.cpu().numpy()
        scales_np: Float32[ndarray, "m 2"] = scales.cpu().numpy()
        crop_origin_xy[valid_box] = centers_np - scales_np / 2.0
        crop_size_wh[valid_box] = scales_np

    return CameraKeypoints(
        sample_indices=sample_indices,
        times_ns=times[sample_indices].astype("timedelta64[ns]").astype(np.int64),
        kp_xy=kp_xy,
        conf=conf,
        bbox_xyxy=bbox_xyxy,
        crop_origin_xy=crop_origin_xy,
        crop_size_wh=crop_size_wh,
    )


def log_keypoints_layer(
    per_camera: dict[str, CameraKeypoints],
    recording: rr.RecordingStream,
    log_stride: int,
) -> None:
    """Log subsampled keypoint overlays under each camera's pinhole entity."""
    for name, cam in per_camera.items():
        entity: str = f"/world/{name}/pinhole/exocalib_kp2d"
        for t in range(0, len(cam.sample_indices), log_stride):
            if not np.isfinite(cam.kp_xy[t]).any():
                continue
            recording.set_time(TIMELINE, duration=1e-9 * float(cam.times_ns[t]))
            recording.log(entity, Points2DWithConfidence(positions=cam.kp_xy[t].astype(np.float64), confidences=cam.conf[t].astype(np.float64), radii=2.0))


def main(config: Keypoints2dConfig) -> None:
    """Run Stage B over all exo cameras and (optionally) register the layer."""
    from posekit.models.sapiens import SapiensPoseConfig
    from posekit.models.yolox import YoloxDetectorConfig
    from posekit.runtimes import TensorRtBackendConfig, TorchBackendConfig
    from trtkit.backends import OnnxBackendConfig

    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    streams: ExoVideoStreams = open_exo_streams(dataset, segment_id)

    detector = YoloxDetectorConfig(score_thr=config.detection_score_thr).setup()
    backend = {
        "tensorrt": TensorRtBackendConfig(max_batch_size=config.batch_size),
        "onnxruntime": OnnxBackendConfig(max_batch_size=config.batch_size),
        "torch": TorchBackendConfig(max_batch_size=config.batch_size),
    }[config.pose_backend]
    pose = SapiensPoseConfig(model_size=config.model_size, backend=backend).setup()

    per_camera: dict[str, CameraKeypoints] = {}
    for cam_idx, name in enumerate(streams.names):
        cam: CameraKeypoints = run_camera_sweep(streams, cam_idx, detector, pose, config)
        per_camera[name] = cam
        npz_path: Path = config.output_dir / segment_id / "kp2d" / f"{name.replace('/', '_')}.npz"
        cam.save(npz_path)
        detected: int = int(np.isfinite(cam.bbox_xyxy).all(axis=1).sum())
        print(f"{name}: {detected}/{len(cam.sample_indices)} frames with a person, mean conf {cam.conf[cam.conf > 0].mean():.3f}")

    rrd_path: Path = config.output_dir / segment_id / f"{config.layer_name}.rrd"
    recording: rr.RecordingStream = new_layer_recording(config.application_id, segment_id, rrd_path)
    log_keypoints_layer(per_camera, recording, config.log_stride)
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.register:
        register_layer(dataset, rrd_path, config.layer_name)
        print(f"registered layer {config.layer_name}")
