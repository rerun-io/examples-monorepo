"""One-segment EgoHand comparison, using a single cached catalog window."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
import torch
from datafusion import col
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from PIL import Image
from rerun.catalog import CatalogClient
from serde import serde
from serde.json import to_json
from simplecv.catalog_video import CatalogVideo, catalog_codec, catalog_keyframes, packet_views
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rrd_query_utils import first_valid_value
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from posekit.models.egohand import (
    SOMA_HAND,
    EgoHandConfig,
    EgoHandDetector,
    EgoHandDetectorConfig,
    EgoHandPredictor,
    HandDetections,
    HandReconstruction,
)
from posekit.models.egohand_geometry import OAKCamera, coco_hand_indices, soma_to_hand21
from posekit.predictions import BoxDetections, Keypoints2d

SEGMENT = "selfcap__2025-12-03__d31bf1da__ep001"
CAMERA = "/world/rig_04/cam_00/pinhole"
VIDEO = f"{CAMERA}/video"


@dataclass(frozen=True, slots=True)
class Config:
    """Restricted experiment on the requested SelfCap episode."""

    output: Path = Path("/tmp/egohand")
    """Scratch folder for the cached window, report and reference image."""
    start_s: float = 2.0
    """First inference time on video_time in seconds."""
    end_s: float = 6.0
    """Last inference time on video_time in seconds."""
    stride: int = 1
    """Infer every Nth decoded frame; 1 uses all frames."""
    compare_sapiens: bool = False
    """Log Sapiens2 0.4B on the one reference frame; needs its cached weights."""
    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(application_id="egohand-selfcap", recording_id=SEGMENT, headless=True))
    """Shared viewer/save settings; pass --rr-config.save for the recording."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunReport:
    """Measured values for this one inference run."""

    segment: str
    """Only segment processed."""
    frames: int
    """Number of inferred RGB frames."""
    hands: int
    """Total reconstructed hands."""
    detector_median_ms: float
    """Median detector wall time, with CUDA synchronization."""
    inference_median_ms: float
    """Median complete hand predictor time."""
    stage_median_ms: dict[str, float]
    """Median upstream stage times, including perspective crops and forward."""
    peak_torch_allocated_mib: float
    """PyTorch peak allocated memory; excludes ORT and driver allocations."""
    peak_torch_reserved_mib: float
    """PyTorch peak reserved memory; excludes ORT and driver allocations."""
    quest_reprojection_median_px: float | None
    """Median same-time EgoHand/Quest hand21 distance; descriptive, not ground-truth accuracy."""
    quest_reprojection_pairs: int
    """Number of finite, positive-depth paired joints used."""
    opposite_side_median_px: float | None
    """Diagnostic distance after swapping side associations, without changing outputs."""
    opposite_side_pairs: int
    """Number of valid joints in the opposite-side diagnostic."""
    process_peak_gpu_mib: float | None = None
    """Optional external 200 ms nvidia-smi sample, including non-PyTorch allocations."""


def load_window(cache: Path) -> pa.Table:
    """Reuse the local stream cache or make one bounded catalog materialization."""
    if cache.is_file():
        return pq.read_table(cache)
    dataset = CatalogClient("rerun+http://dgx-spark:51235").get_dataset(name="selfcap-post-ep1")
    view = dataset.filter_segments(SEGMENT).filter_contents(
        [f"{CAMERA.rsplit('/pinhole', 1)[0]}/**", "/world/rig_04", "/world/rig_05", "/world/gt/coco133_xyz"]
    )
    table: pa.Table = (
        view.reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= 0)
        .filter(col("video_time").cast(pa.int64()) <= 6_200_000_000)
        .to_arrow_table()
    )
    pq.write_table(table, cache)
    return table


def camera_from_window(table: pa.Table) -> OAKCamera:
    """Read the catalog calibration; matrix components are column-major."""
    intrinsics: Float64[ndarray, "3 3"] = (
        np.asarray(first_valid_value(table[f"{CAMERA}:Pinhole:image_from_camera"]), dtype=np.float64).reshape(3, 3).T
    )
    distortion: Float64[ndarray, "14"] = np.asarray(
        first_valid_value(table[f"{CAMERA}:simplecv.components.DistortionCoefficients"]), dtype=np.float64
    ).reshape(14)
    resolution = first_valid_value(table[f"{CAMERA}:Pinhole:resolution"])
    return OAKCamera(intrinsics, distortion, int(resolution[0]), int(resolution[1]))


@torch.inference_mode()
def log_sapiens_reference(frame: UInt8[Tensor, "h w 3"], time_s: float, next_time_s: float) -> None:
    """Log the incumbent full-frame model at the exact EgoHand reference image."""
    from sapiens2_pose.api.runtime import clear_pose_model_cache

    from posekit.models.sapiens import SapiensPoseConfig

    height: int = int(frame.shape[0])
    width: int = int(frame.shape[1])
    boxes: BoxDetections = BoxDetections(
        torch.tensor([[0.0, 0.0, float(width), float(height)]], device=frame.device),
        torch.ones(1, device=frame.device),
        torch.zeros(1, dtype=torch.int64, device=frame.device),
    )
    model = SapiensPoseConfig(model_size="0.4B").setup()
    points: Keypoints2d = model(frame.unsqueeze(0), boxes)
    rr.set_time("video_time", duration=time_s)
    for side in ("left", "right"):
        indices: list[int] = list(coco_hand_indices(side))
        rr.log(f"/image/sapiens/{side}", rr.Points2D(points.xy[0, indices].cpu().numpy(), colors=(100, 255, 100), radii=3.0, show_labels=False))
        rr.log(f"/image/sapiens/{side}/scores", rr.AnyValues(confidence=points.scores[0, indices].cpu().numpy()))
    rr.set_time("video_time", duration=next_time_s)
    rr.log("/image/sapiens", rr.Clear(recursive=True))
    del model
    clear_pose_model_cache()
    torch.cuda.empty_cache()


def main(config: Config) -> None:
    """Decode, infer and log one calibrated window without catalog writes."""
    if not 0.0 <= config.start_s <= config.end_s <= 6.2 or config.stride < 1:
        raise ValueError("This experiment is restricted to 0..6.2 seconds with positive stride")
    config.output.mkdir(parents=True, exist_ok=True)
    table: pa.Table = load_window(config.output / "window.parquet")
    camera = camera_from_window(table)
    frames: pa.Table = table.filter(table[f"{VIDEO}:VideoStream:sample"].is_valid())
    times: Int64[ndarray, "n"] = frames["video_time"].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
    keys: list[bool] = catalog_keyframes(frames[f"{VIDEO}:VideoStream:is_keyframe"])
    first: int = next(index for index, flag in enumerate(keys) if flag)
    video: CatalogVideo = CatalogVideo(
        times[first:],
        packet_views(frames[f"{VIDEO}:VideoStream:sample"])[first:],
        keys[first:],
        catalog_codec(frames[f"{VIDEO}:VideoStream:codec"], VIDEO),
    )
    fps: int = round(1e9 / float(np.median(np.diff(video.t_ns))))
    decoder: VideoDecoder = VideoDecoder(
        wrap_mp4(video.samples, video.keyframes, fps, video.codec), device="cuda", dimension_order="NHWC", seek_mode="exact"
    )
    nearest: int = int(np.argmin(abs(video.t_ns - 4_144_000_000)))
    reference_frame: UInt8[Tensor, "h w 3"] = decoder.get_frame_at(nearest).data
    Image.fromarray(reference_frame.cpu().numpy()).save(config.output / "frame-4.144.png")
    if config.compare_sapiens:
        log_sapiens_reference(reference_frame, float(video.t_ns[nearest]) / 1e9, float(video.t_ns[nearest + 1]) / 1e9)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="OAK RGB + EgoHand / Quest", origin="/image", visual_bounds=rrb.VisualBounds2D(x_range=[0, 1280], y_range=[0, 720])
                ),
                rrb.Spatial3DView(
                    name="OAK camera space (metres)",
                    origin="/camera_space",
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(position=[0.0, 0.0, -0.7], look_target=[0.0, 0.15, 0.6], eye_up=[0.0, -1.0, 0.0]),
                ),
            ),
            collapse_panels=True,
        )
    )
    annotations: rr.AnnotationContext = rr.AnnotationContext(
        rr.ClassDescription(
            info=rr.AnnotationInfo(id=0, label="SOMA hand"),
            keypoint_annotations=[(i, name) for i, name in enumerate(SOMA_HAND.keypoint_names)],
            keypoint_connections=SOMA_HAND.links,
        )
    )
    rr.log("/image/egohand", annotations, static=True)
    rr.log("/camera_space/egohand", annotations, static=True)
    rr.log("/camera_space", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "/notes",
        rr.TextDocument(
            "Cyan: detector-labelled left. Orange: detector-labelled right. Magenta: Quest. Green: Sapiens2 at 4.128 s only. "
            "Detector handedness is reversed on this reference image. Raw labels are preserved. "
            "Quest opposite-side distances are a diagnostic, not an accuracy score."
        ),
        static=True,
    )
    # Original distorted pixels live in a flat 2D view, not an ideal pinhole frustum.
    weights: Path = config.output / "weights"
    detector: EgoHandDetector = EgoHandDetectorConfig(weights / "detector/hand_detector.onnx").setup()
    predictor: EgoHandPredictor = EgoHandConfig(weights / "egohand/egohand_s060000.ckpt", weights / "soma").setup()
    torch.cuda.reset_peak_memory_stats()
    detector_ms: list[float] = []
    inference_ms: list[float] = []
    differences: list[float] = []
    opposite_differences: list[float] = []
    hand_count: int = 0
    chosen: Int64[ndarray, "n"] = np.flatnonzero((video.t_ns >= int(config.start_s * 1e9)) & (video.t_ns <= int(config.end_s * 1e9)))[
        :: config.stride
    ]
    quest_column: str = "/world/gt/coco133_xyz:Points3D:positions"
    quest_rows: pa.Table = table.filter(table[quest_column].is_valid())
    quest_times: Int64[ndarray, "n"] = quest_rows["video_time"].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
    for index in chosen:
        frame: UInt8[Tensor, "h w 3"] = decoder.get_frame_at(int(index)).data
        batch: UInt8[Tensor, "1 h w 3"] = frame.unsqueeze(0)
        torch.cuda.synchronize()
        start: float = time.perf_counter()
        detections: HandDetections = detector(batch)
        torch.cuda.synchronize()
        detector_ms.append((time.perf_counter() - start) * 1000.0)
        start: float = time.perf_counter()
        result: HandReconstruction = predictor(batch, detections, camera)
        torch.cuda.synchronize()
        inference_ms.append((time.perf_counter() - start) * 1000.0)
        rr.set_time("video_time", duration=float(video.t_ns[index]) / 1e9)
        rr.log("/image/rgb", rr.Image(frame.cpu().numpy()).compress(jpeg_quality=90))
        rr.log("/image/egohand", rr.Clear(recursive=True))
        rr.log("/camera_space/egohand", rr.Clear(recursive=True))
        rr.log("/image/egohand", annotations)
        rr.log("/camera_space/egohand", annotations)
        rr.log("/image/quest", rr.Clear(recursive=True))
        rr.log("/camera_space/quest", rr.Clear(recursive=True))
        # The calibrated layer already places the OAK pose and Quest joints on video_time.
        quest: Float64[ndarray, "133 3"] | None = None
        if len(quest_times):
            qi: int = int(np.argmin(abs(quest_times - video.t_ns[index])))
            if abs(int(quest_times[qi]) - int(video.t_ns[index])) < 40_000_000:
                world_points: Float64[ndarray, "133 3"] = np.asarray(quest_rows[quest_column][qi].as_py(), dtype=np.float64)
                row_times: Int64[ndarray, "n"] = table["video_time"].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
                pose_mask: Bool[ndarray, "n"] = table["/world/rig_04:Transform3D:mat3x3"].is_valid().to_numpy()
                pose_indices: Int64[ndarray, "n"] = np.flatnonzero(pose_mask)
                if len(pose_indices):
                    pi: int = int(pose_indices[np.argmin(abs(row_times[pose_indices] - video.t_ns[index]))])
                    world_R_camera: Float64[ndarray, "3 3"] = (
                        np.asarray(table["/world/rig_04:Transform3D:mat3x3"][pi].as_py(), dtype=np.float64).reshape(3, 3).T
                    )
                    world_t_camera: Float64[ndarray, "3"] = np.asarray(
                        table["/world/rig_04:Transform3D:translation"][pi].as_py(), dtype=np.float64
                    ).reshape(3)
                    quest: Float64[ndarray, "133 3"] | None = (world_points - world_t_camera) @ world_R_camera
        xyz: Float32[Tensor, "n 25 3"] | None = result.keypoints.xyz_camera
        assert xyz is not None
        for hand, side in enumerate(result.sides):
            color: tuple[int, int, int] = (65, 210, 255) if side == "left" else (255, 165, 65)
            points: Float32[ndarray, "25 3"] = xyz[hand].cpu().numpy()
            pixels: Float32[ndarray, "25 2"] = result.keypoints.xy[hand].cpu().numpy()
            rr.log(f"/image/egohand/{side}", rr.Points2D(pixels, keypoint_ids=range(25), class_ids=0, colors=color, radii=3.0, show_labels=False))
            rr.log(
                f"/camera_space/egohand/{side}",
                rr.Points3D(points, keypoint_ids=range(25), class_ids=0, colors=color, radii=0.003, show_labels=False),
            )
            rr.log(
                f"/camera_space/egohand/{side}/mesh",
                rr.Mesh3D(
                    vertex_positions=result.vertices[hand].cpu().numpy(),
                    triangle_indices=predictor.predictor.model.soma_hand.faces(side),
                    albedo_factor=(*[c / 255.0 for c in color], 0.25),
                ),
            )
            rr.log(
                f"/camera_space/egohand/{side}/params",
                rr.AnyValues(
                    rotations=result.rotations[hand].cpu().numpy().reshape(-1),
                    identity=result.identity[hand].cpu().numpy().reshape(-1),
                    scale=float(result.scale[hand]),
                    translation=result.translation[hand].cpu().numpy().reshape(-1),
                    scale_parameters=result.scale_parameters[hand].cpu().numpy().reshape(-1) if result.scale_parameters is not None else [],
                ),
            )
            if quest is not None:
                target: Float64[ndarray, "21 3"] = quest[np.asarray(coco_hand_indices(side))]
                valid: Bool[ndarray, "21"] = np.isfinite(target).all(axis=1) & (target[:, 2] > 0.0)
                projected: Float64[ndarray, "21 2"] = camera.project(target)
                inferred: Float64[ndarray, "21 2"] = camera.project(soma_to_hand21(points).astype(np.float64))
                valid &= np.isfinite(inferred).all(axis=1) & np.isfinite(projected).all(axis=1)
                differences.extend(np.linalg.norm(inferred[valid] - projected[valid], axis=1).tolist())
                opposite: Float64[ndarray, "21 3"] = quest[np.asarray(coco_hand_indices("right" if side == "left" else "left"))]
                opposite_pixels: Float64[ndarray, "21 2"] = camera.project(opposite)
                opposite_valid: Bool[ndarray, "21"] = (
                    np.isfinite(opposite_pixels).all(axis=1) & np.isfinite(inferred).all(axis=1) & (opposite[:, 2] > 0.0)
                )
                opposite_differences.extend(np.linalg.norm(inferred[opposite_valid] - opposite_pixels[opposite_valid], axis=1).tolist())
                rr.log(f"/image/quest/{side}", rr.Points2D(projected[valid], colors=(255, 75, 180), radii=2.0))
                rr.log(f"/camera_space/quest/{side}", rr.Points3D(target[valid], colors=(255, 75, 180), radii=0.002))
        hand_count += len(result.sides)
        print(
            f"{video.t_ns[index] / 1e9:.3f}s: {len(result.sides)} hands; detector {detector_ms[-1]:.1f} ms; inference {inference_ms[-1]:.1f} ms",
            flush=True,
        )
    report: RunReport = RunReport(
        SEGMENT,
        len(chosen),
        hand_count,
        float(np.median(detector_ms)),
        float(np.median(inference_ms)),
        {key: float(np.median(values)) for key, values in predictor.stage_ms.items()},
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20,
        float(np.median(differences)) if differences else None,
        len(differences),
        float(np.median(opposite_differences)) if opposite_differences else None,
        len(opposite_differences),
    )
    (config.output / "report.json").write_text(to_json(report))
    recording: rr.RecordingStream | None = rr.get_global_data_recording()
    assert recording is not None
    recording.flush()
    print(to_json(report), flush=True)
