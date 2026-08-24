"""Run COCO-133 multiview pose tracking on a calibrated MAMMA capture.

The adapter keeps the camera order shared by ``meta/*.npz`` and ``videos/*.mp4``.
Frames stay on CUDA from TorchCodec/NVDEC through PoseKit inference; only the
small pose arrays cross to NumPy for triangulation, metrics, and persistence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as torch_functional
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.skeleton.coco133_layers import Coco133AnnotationLayer
from simplecv.data.skeleton.coco_133 import COCO_133_IDS
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence
from simplecv.rerun_log_utils import log_pinhole
from simplecv.video_encoder import VideoCodecChoice, VideoEncoder
from simplecv.video_io import TorchCodecMultiVideoReader

from mv_api.api.full_exoego_pipeline import set_annotation_context
from mv_api.multiview_pose_estimator import MultiviewBodyTracker, MultiviewBodyTrackerConfig, MVHistory

WORLD_TAG: str = "world"
_VIDEO_DIR_CANDIDATES: tuple[str, ...] = ("videos_light", "videos", "videos_crf24")


def pinhole_entity(camera_name: str) -> str:
    """Return the Rerun entity path for one calibrated camera pinhole."""
    return f"{WORLD_TAG}/cameras/{camera_name}/pinhole"


@dataclass(frozen=True, slots=True)
class PoseCaptureSequence:
    """Minimal synchronized-sequence contract needed by MV-API pose capture."""

    name: str
    camera_names: list[str]
    video_paths: list[Path]
    fps: float
    frame_count: int
    frame_start: int


@dataclass(frozen=True, slots=True)
class CalibratedMammaCapture:
    """A validated MAMMA sequence and its aligned MV-API camera objects."""

    sequence: PoseCaptureSequence
    pinholes: list[PinholeParameters]
    projection_matrices: Float32[ndarray, "camera 3 4"]


@dataclass(frozen=True, slots=True)
class ReprojectionSummary:
    """Aggregate pixel residuals between observed and reprojected keypoints."""

    mean_px: float
    median_px: float
    p95_px: float
    valid_observation_count: int
    per_camera_mean_px: Float32[ndarray, "camera"]
    per_camera_valid_observation_count: list[int]

    def to_json_dict(self) -> dict[str, float | int | list[float] | list[int]]:
        """Return a JSON-compatible metrics mapping."""
        data: dict[str, float | int | list[float] | list[int]] = asdict(self)
        data["per_camera_mean_px"] = self.per_camera_mean_px.tolist()
        return data


@dataclass(frozen=True, slots=True)
class PoseCaptureConfig:
    """CLI configuration for one exact-window multiview pose run."""

    capture_dir: Path
    """MAMMA capture with ``meta`` calibration and GPU-decodable ``videos`` proxies."""
    output_npz: Path
    """Destination for the full 2D/3D pose sequence and reprojections."""
    output_metrics_json: Path
    """Destination for aggregate reprojection metrics and runtime metadata."""
    output_rrd: Path
    """Destination for the self-contained Rerun pose-overlay recording."""
    expected_frame_count: int = 121
    """Exact synchronized frame count required by the experiment gate."""
    confidence_threshold: float = 0.3
    """Minimum 2D confidence included in reprojection metrics and visualization."""
    backend: str = "tensorrt"
    """PoseKit backend, normally ``tensorrt`` for the RTX 5090 run."""
    decode_device: str = "cuda"
    """TorchCodec decode device; ``cuda`` selects NVDEC."""


class _PoseVideoLogger:
    """Encode GPU RGB frames into compact H.264 Rerun video streams."""

    def __init__(self, capture: CalibratedMammaCapture) -> None:
        self.capture: CalibratedMammaCapture = capture
        self.encoders: dict[str, VideoEncoder] = {
            name: VideoEncoder(codec=VideoCodecChoice.H264, fps=capture.sequence.fps)
            for name in capture.sequence.camera_names
        }

    def setup(self) -> None:
        """Log the static coordinate system, calibrated pinholes, and video streams."""
        rr.log(WORLD_TAG, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for pinhole in self.capture.pinholes:
            log_pinhole(
                pinhole,
                cam_log_path=Path(f"world/cameras/{pinhole.name}"),
                image_plane_distance=0.4,
                static=True,
            )
            rr.log(f"{pinhole_entity(pinhole.name)}/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

    @staticmethod
    def _rgb_to_yuv420(
        frame_rgb: UInt8[torch.Tensor, "3 h w"],
    ) -> tuple[
        UInt8[ndarray, "h w"],
        UInt8[ndarray, "h2 w2"],
        UInt8[ndarray, "h2 w2"],
    ]:
        """Convert RGB to BT.601 limited-range YUV420 on the GPU."""
        rgb: torch.Tensor = frame_rgb.float()
        red: torch.Tensor
        green: torch.Tensor
        blue: torch.Tensor
        red, green, blue = rgb[0], rgb[1], rgb[2]
        y_plane: torch.Tensor = (
            16.0 + 0.256788 * red + 0.504129 * green + 0.097906 * blue
        ).round_().clamp_(16, 235).to(torch.uint8)
        u_full: torch.Tensor = 128.0 - 0.148223 * red - 0.290993 * green + 0.439216 * blue
        v_full: torch.Tensor = 128.0 + 0.439216 * red - 0.367788 * green - 0.071427 * blue
        uv_small: torch.Tensor = torch_functional.avg_pool2d(
            torch.stack([u_full, v_full]).unsqueeze(0), kernel_size=2
        )[0].round_().clamp_(16, 240).to(torch.uint8)
        return (
            y_plane.contiguous().cpu().numpy(),
            uv_small[0].contiguous().cpu().numpy(),
            uv_small[1].contiguous().cpu().numpy(),
        )

    def log_frames(
        self,
        frames_rgb: list[UInt8[torch.Tensor, "3 h w"]],
    ) -> None:
        """Encode and log one synchronized frame from every camera."""
        for name, frame_rgb in zip(self.capture.sequence.camera_names, frames_rgb, strict=True):
            y_plane, u_plane, v_plane = self._rgb_to_yuv420(frame_rgb)
            for pts, packet in self.encoders[name].encode_yuv_planes(y_plane, u_plane, v_plane):
                rr.set_time("time", duration=pts / self.capture.sequence.fps)
                rr.log(f"{pinhole_entity(name)}/video", rr.VideoStream.from_fields(sample=packet))

    def flush(self) -> None:
        """Drain every encoder into its Rerun stream."""
        for name, encoder in self.encoders.items():
            for pts, packet in encoder.flush():
                rr.set_time("time", duration=pts / self.capture.sequence.fps)
                rr.log(f"{pinhole_entity(name)}/video", rr.VideoStream.from_fields(sample=packet))


def load_calibrated_mamma_capture(
    capture_dir: Path,
    *,
    expected_frame_count: int | None = None,
) -> CalibratedMammaCapture:
    """Load a MAMMA NPZ rig and verify calibration/video/frame alignment."""
    meta_dir: Path = capture_dir / "meta"
    global_path: Path = meta_dir / "global.npz"
    if not global_path.is_file():
        raise FileNotFoundError(f"Missing MAMMA global calibration: {global_path}.")
    camera_paths: list[Path] = sorted(path for path in meta_dir.glob("*.npz") if path.name != "global.npz")
    if not camera_paths:
        raise FileNotFoundError(f"No per-camera MAMMA calibrations under {meta_dir}.")
    video_dir: Path | None = next(
        (capture_dir / candidate for candidate in _VIDEO_DIR_CANDIDATES if (capture_dir / candidate).is_dir()),
        None,
    )
    if video_dir is None:
        raise FileNotFoundError(f"No videos directory among {_VIDEO_DIR_CANDIDATES} under {capture_dir}.")

    global_data = np.load(global_path, allow_pickle=True)
    camera_names: list[str] = []
    pinholes: list[PinholeParameters] = []
    video_paths: list[Path] = []
    for camera_path in camera_paths:
        camera_data = np.load(camera_path, allow_pickle=True)
        camera_name: str = str(camera_data["cam_name"]) if "cam_name" in camera_data.files else camera_path.stem
        k_matrix: Float64[ndarray, "3 3"] = np.asarray(camera_data["cam_int"], dtype=np.float64)
        world_to_camera: Float64[ndarray, "4 4"] = np.asarray(camera_data["cam_ext"], dtype=np.float64)
        if np.abs(world_to_camera[:3, 3]).max() > 200.0:
            world_to_camera = world_to_camera.copy()
            world_to_camera[:3, 3] /= 1000.0
        width: int = int(camera_data["cam_img_w"])
        height: int = int(camera_data["cam_img_h"])
        video_path: Path = video_dir / f"{camera_name}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video for calibrated camera {camera_name}: {video_path}.")
        camera_names.append(camera_name)
        video_paths.append(video_path)
        pinholes.append(
            PinholeParameters(
                name=camera_name,
                intrinsics=Intrinsics.from_k_matrix(
                    camera_conventions="RDF",
                    k_matrix=k_matrix,
                    width=width,
                    height=height,
                ),
                extrinsics=Extrinsics(
                    cam_R_world=world_to_camera[:3, :3],
                    cam_t_world=world_to_camera[:3, 3],
                ),
            )
        )

    sequence = PoseCaptureSequence(
        name=str(global_data["seq_name"]),
        camera_names=camera_names,
        video_paths=video_paths,
        fps=float(global_data["fps"]),
        frame_count=int(global_data["frame_end"]) - int(global_data["frame_start"]),
        frame_start=int(global_data["frame_start"]),
    )
    if expected_frame_count is not None and sequence.frame_count != expected_frame_count:
        raise ValueError(
            f"Expected {expected_frame_count} synchronized frames, but metadata declares {sequence.frame_count}."
        )
    if sequence.frame_start != 0:
        raise ValueError(
            "Pose capture expects already-trimmed videos with frame_start=0; "
            f"metadata declares {sequence.frame_start}."
        )
    projection_matrices: Float32[ndarray, "camera 3 4"] = np.stack(
        [pinhole.projection_matrix for pinhole in pinholes]
    ).astype(np.float32)
    return CalibratedMammaCapture(
        sequence=sequence,
        pinholes=pinholes,
        projection_matrices=projection_matrices,
    )


def project_keypoints(
    *,
    joints3d: Float32[ndarray, "frame joint 4"],
    projection_matrices: Float32[ndarray, "camera 3 4"],
) -> Float32[ndarray, "frame camera joint 2"]:
    """Project homogeneous 3D keypoints into every calibrated camera."""
    xyz_h: Float32[ndarray, "frame joint 4"] = np.concatenate(
        [joints3d[..., :3], np.ones((*joints3d.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )
    uvw: Float32[ndarray, "frame camera joint 3"] = np.einsum(
        "cij,fkj->fcki", projection_matrices, xyz_h, optimize=True
    ).astype(np.float32, copy=False)
    depth: Float32[ndarray, "frame camera joint 1"] = uvw[..., 2:3]
    valid_depth: Bool[ndarray, "frame camera joint 1"] = np.isfinite(depth) & (np.abs(depth) > 1e-8)
    projected: Float32[ndarray, "frame camera joint 2"] = np.full(uvw[..., :2].shape, np.nan, dtype=np.float32)
    np.divide(uvw[..., :2], depth, out=projected, where=valid_depth)
    return projected


def compute_reprojection_summary(
    *,
    observed_joints2d: Float32[ndarray, "frame camera joint 3"],
    projected_joints2d: Float32[ndarray, "frame camera joint 2"],
    confidence_threshold: float,
) -> ReprojectionSummary:
    """Summarize residuals for finite observations above a confidence floor."""
    if observed_joints2d.shape[:-1] != projected_joints2d.shape[:-1]:
        raise ValueError(
            "Observed/projected pose dimensions do not align: "
            f"{observed_joints2d.shape} vs {projected_joints2d.shape}."
        )
    finite: Bool[ndarray, "frame camera joint"] = np.all(
        np.isfinite(observed_joints2d[..., :2]) & np.isfinite(projected_joints2d), axis=-1
    )
    valid: Bool[ndarray, "frame camera joint"] = finite & (
        observed_joints2d[..., 2] >= confidence_threshold
    )
    residuals: Float32[ndarray, "frame camera joint"] = np.linalg.norm(
        observed_joints2d[..., :2] - projected_joints2d, axis=-1
    ).astype(np.float32, copy=False)
    selected: Float32[ndarray, "valid"] = residuals[valid]
    if selected.size == 0:
        raise ValueError("No finite 2D observations satisfy the confidence threshold.")

    per_camera_mean: Float32[ndarray, "camera"] = np.full(observed_joints2d.shape[1], np.nan, dtype=np.float32)
    per_camera_count: list[int] = []
    for camera_idx in range(observed_joints2d.shape[1]):
        camera_selected: Float32[ndarray, "valid"] = residuals[:, camera_idx, :][valid[:, camera_idx, :]]
        per_camera_count.append(int(camera_selected.size))
        if camera_selected.size:
            per_camera_mean[camera_idx] = float(np.mean(camera_selected, dtype=np.float64))

    return ReprojectionSummary(
        mean_px=float(np.mean(selected, dtype=np.float64)),
        median_px=float(np.median(selected)),
        p95_px=float(np.percentile(selected, 95.0)),
        valid_observation_count=int(selected.size),
        per_camera_mean_px=per_camera_mean,
        per_camera_valid_observation_count=per_camera_count,
    )


def _pose_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """Place the calibrated 3D scene beside all eight synchronized image views."""
    camera_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=f"/{pinhole_entity(name)}", name=name) for name in camera_names
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=f"/{WORLD_TAG}", name="Calibrated 3D pose"),
            rrb.Grid(*camera_views, grid_columns=4, name="Eight synchronized views"),
            column_shares=[2, 5],
        ),
        collapse_panels=True,
    )


def _log_pose_frame(
    *,
    frame_idx: int,
    fps: float,
    camera_names: list[str],
    joints2d: Float32[ndarray, "camera joint 3"],
    joints3d: Float32[ndarray, "joint 4"],
    reprojected2d: Float32[ndarray, "camera joint 2"],
    confidence_threshold: float,
) -> None:
    """Log one triangulated skeleton and observed/projected 2D overlays."""
    rr.set_time("time", duration=frame_idx / fps)
    confidences3d: Float32[ndarray, "joint"] = joints3d[:, 3]
    valid3d: Bool[ndarray, "joint"] = (
        np.all(np.isfinite(joints3d[:, :3]), axis=-1) & (confidences3d >= confidence_threshold)
    )
    positions3d: Float32[ndarray, "joint 3"] = np.where(valid3d[:, None], joints3d[:, :3], np.nan)
    rr.log(
        f"{WORLD_TAG}/pose/coco133",
        Points3DWithConfidence(
            positions=positions3d,
            confidences=np.where(valid3d, confidences3d, 0.0),
            class_ids=int(Coco133AnnotationLayer.TRIANGULATED_3D),
            keypoint_ids=COCO_133_IDS,
            show_labels=False,
            radii=0.012,
        ),
    )

    for camera_idx, camera_name in enumerate(camera_names):
        observed_confidence: Float32[ndarray, "joint"] = joints2d[camera_idx, :, 2]
        valid2d: Bool[ndarray, "joint"] = (
            np.all(np.isfinite(joints2d[camera_idx, :, :2]), axis=-1)
            & (observed_confidence >= confidence_threshold)
        )
        observed_uv: Float32[ndarray, "joint 2"] = np.where(
            valid2d[:, None], joints2d[camera_idx, :, :2], np.nan
        )
        projected_uv: Float32[ndarray, "joint 2"] = np.where(
            valid2d[:, None], reprojected2d[camera_idx], np.nan
        )
        base_path: str = pinhole_entity(camera_name)
        rr.log(
            f"{base_path}/pose/observed",
            Points2DWithConfidence(
                positions=observed_uv,
                confidences=np.where(valid2d, observed_confidence, 0.0),
                class_ids=int(Coco133AnnotationLayer.RAW_2D),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                radii=2.5,
            ),
        )
        rr.log(
            f"{base_path}/pose/reprojected",
            Points2DWithConfidence(
                positions=projected_uv,
                confidences=np.where(valid2d, observed_confidence, 0.0),
                class_ids=int(Coco133AnnotationLayer.PROJECTED_2D),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                radii=1.5,
            ),
        )


def run_pose_capture(config: PoseCaptureConfig) -> ReprojectionSummary:
    """Run exact-window GPU decode, whole-body inference, triangulation, and Rerun logging."""
    capture: CalibratedMammaCapture = load_calibrated_mamma_capture(
        config.capture_dir,
        expected_frame_count=config.expected_frame_count,
    )
    sequence: PoseCaptureSequence = capture.sequence
    config.output_npz.parent.mkdir(parents=True, exist_ok=True)
    config.output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_rrd.parent.mkdir(parents=True, exist_ok=True)

    tracker_config = MultiviewBodyTrackerConfig(
        mode="wholebody",
        backend=config.backend,
        device="cuda",
        keypoint_threshold=config.confidence_threshold,
        perform_tracking=True,
        verbose=False,
    )
    # Construct CUDA inference runtimes before NVDEC. TensorRT can replace the
    # active CUDA context while deserializing engines.
    tracker = MultiviewBodyTracker(tracker_config)
    reader = TorchCodecMultiVideoReader(
        list(sequence.video_paths),
        device=config.decode_device,
        seek_mode="exact",
    )
    if reader.frame_cnt != config.expected_frame_count:
        reader.close()
        raise ValueError(
            f"Decoded videos expose {reader.frame_cnt} frames, expected exactly {config.expected_frame_count}."
        )
    first_intrinsics: Intrinsics = capture.pinholes[0].intrinsics
    if reader.width != first_intrinsics.width or reader.height != first_intrinsics.height:
        reader.close()
        raise ValueError(
            "Video/calibration dimensions disagree: "
            f"decoder={reader.width}x{reader.height}, calibration="
            f"{first_intrinsics.width}x{first_intrinsics.height}."
        )

    recording_id: str = f"{sequence.name}-mvapi-coco133"
    rr.init("mamma-multiview-pose", recording_id=recording_id)
    rr.save(config.output_rrd)
    video_logger = _PoseVideoLogger(capture)
    video_logger.setup()
    set_annotation_context(recording=None)
    rr.send_blueprint(_pose_blueprint(sequence.camera_names))

    joints2d_frames: list[Float32[ndarray, "camera joint 3"]] = []
    joints3d_frames: list[Float32[ndarray, "joint 4"]] = []
    history = MVHistory()
    try:
        for frame_idx in range(config.expected_frame_count):
            frames_chw: list[UInt8[torch.Tensor, "3 h w"]] = reader.get_frames_at(
                [frame_idx] * len(sequence.video_paths)
            )
            video_logger.log_frames(frames_chw)
            frames_nhwc: UInt8[torch.Tensor, "camera h w 3"] = rearrange(
                torch.stack(frames_chw), "camera channel height width -> camera height width channel"
            )
            history = tracker(
                frames_rgb=frames_nhwc,
                pinhole_list=capture.pinholes,
                pred_state=history,
            )
            if history.uvc_t is None or history.xyzc_t is None:
                raise RuntimeError(f"Pose tracker returned no pose at frame {frame_idx}.")
            joints2d_frame: Float32[ndarray, "camera joint 3"] = history.uvc_t.copy()
            joints3d_frame: Float32[ndarray, "joint 4"] = history.xyzc_t.copy()
            reprojected_frame: Float32[ndarray, "camera joint 2"] = project_keypoints(
                joints3d=joints3d_frame[None],
                projection_matrices=capture.projection_matrices,
            )[0]
            joints2d_frames.append(joints2d_frame)
            joints3d_frames.append(joints3d_frame)
            _log_pose_frame(
                frame_idx=frame_idx,
                fps=sequence.fps,
                camera_names=sequence.camera_names,
                joints2d=joints2d_frame,
                joints3d=joints3d_frame,
                reprojected2d=reprojected_frame,
                confidence_threshold=config.confidence_threshold,
            )
    finally:
        reader.close()
        video_logger.flush()

    joints2d: Float32[ndarray, "frame camera joint 3"] = np.stack(joints2d_frames).astype(np.float32)
    joints3d: Float32[ndarray, "frame joint 4"] = np.stack(joints3d_frames).astype(np.float32)
    reprojected2d: Float32[ndarray, "frame camera joint 2"] = project_keypoints(
        joints3d=joints3d,
        projection_matrices=capture.projection_matrices,
    )
    summary: ReprojectionSummary = compute_reprojection_summary(
        observed_joints2d=joints2d,
        projected_joints2d=reprojected2d,
        confidence_threshold=config.confidence_threshold,
    )
    np.savez_compressed(
        config.output_npz,
        camera_names=np.asarray(sequence.camera_names),
        fps=np.float32(sequence.fps),
        frame_indices=np.arange(config.expected_frame_count, dtype=np.int32),
        joints2d=joints2d,
        joints3d=joints3d,
        reprojected2d=reprojected2d,
        projection_matrices=capture.projection_matrices,
    )
    metrics: dict[str, object] = {
        **summary.to_json_dict(),
        "camera_names": sequence.camera_names,
        "frame_count": config.expected_frame_count,
        "fps": sequence.fps,
        "confidence_threshold": config.confidence_threshold,
        "pose_backend": config.backend,
        "decode_device": config.decode_device,
        "decode_backend": "torchcodec-nvdec" if config.decode_device.startswith("cuda") else "torchcodec-cpu",
    }
    config.output_metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    recording: rr.RecordingStream | None = rr.get_global_data_recording()
    if recording is not None:
        recording.flush()
    return summary


def main() -> None:
    """Parse CLI options and run the calibrated MAMMA pose capture."""
    import tyro

    summary: ReprojectionSummary = run_pose_capture(tyro.cli(PoseCaptureConfig))
    print(json.dumps(summary.to_json_dict(), indent=2))


if __name__ == "__main__":
    main()
