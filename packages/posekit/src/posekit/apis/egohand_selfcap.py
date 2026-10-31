"""One-segment EgoHand comparison, using a single cached catalog window."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
import torch
from datafusion import col
from jaxtyping import Float32, Float64, Int64, UInt8
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
from trtkit import TensorRtBackendConfig

from posekit.apis.egohand_comparison import ErrorSummary, quest_pixels, summarize
from posekit.models.egohand import (
    SOMA_HAND,
    EgoHandConfig,
    EgoHandDetector,
    EgoHandDetectorConfig,
    EgoHandPredictor,
    HandDetections,
    HandReconstruction,
)
from posekit.models.egohand_geometry import SOMA_HAND21, HandSide, OAKCamera, coco_hand_indices
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.skeletons import HAND_21

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
    compare_sapiens: bool = True
    """Compare Sapiens2 0.4B on every selected frame; needs its cached weights."""
    batch_frames: int = 16
    """Frames per inference call; full meshes and parameters are always retained."""
    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(application_id="egohand-selfcap", recording_id=SEGMENT, headless=True))
    """Shared viewer/save settings; pass --rr-config.save for the recording."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunReport:
    """Matched-frame, matched-joint comparison and measured runtime."""

    segment: str
    """Only segment processed."""
    frames: int
    """Number of common RGB frames."""
    hands: int
    """Reconstructed EgoHand instances."""
    errors: ErrorSummary
    """Both models scored on identical finite joint slots."""
    per_joint: dict[str, ErrorSummary]
    """One row per anatomical side and joint, including unmapped slots."""
    sharpness_groups: dict[str, ErrorSummary]
    """Lower and upper ROI sharpness thirds; a blur proxy, not manual labels."""
    coverage: dict[str, int]
    """Finite prediction counts out of all requested joint slots."""
    median_frame_ms: dict[str, float]
    """Detector/Sapiens amortized ms; predictor/stage medians. SOMA is nested."""
    peak_torch_allocated_mib: float
    """PyTorch allocation high-water mark."""
    peak_torch_reserved_mib: float
    """PyTorch reserved high-water mark."""
    process_peak_gpu_mib: float | None = None
    """Optional external nvidia-smi sample."""


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
def run_sapiens(frames: UInt8[Tensor, "n h w 3"]) -> tuple[Float32[ndarray, "n 2 21 2"], float]:
    """Production full-frame 0.4B model on the same images, in batches of four."""
    from sapiens2_pose.api.runtime import clear_pose_model_cache

    from posekit.models.sapiens import SapiensPoseConfig

    model = SapiensPoseConfig(model_size="0.4B").setup()
    output: Float32[ndarray, "n 2 21 2"] = np.full((len(frames), 2, 21, 2), np.nan, dtype=np.float32)
    torch.cuda.synchronize()
    started: float = time.perf_counter()
    for start in range(0, len(frames), 4):
        batch: UInt8[Tensor, "b h w 3"] = frames[start : start + 4]
        boxes: BoxDetections = BoxDetections(
            torch.tensor([[0.0, 0.0, float(batch.shape[2]), float(batch.shape[1])]], device=batch.device).repeat(len(batch), 1),
            torch.ones(len(batch), device=batch.device),
            torch.arange(len(batch), dtype=torch.int64, device=batch.device),
        )
        points: Keypoints2d = model(batch, boxes)
        sides: tuple[HandSide, HandSide] = ("left", "right")
        for side, name in enumerate(sides):
            output[start : start + len(batch), side] = points.xy[:, list(coco_hand_indices(name))].cpu().numpy()
    torch.cuda.synchronize()
    elapsed: float = (time.perf_counter() - started) * 1000.0 / len(frames)
    del model
    clear_pose_model_cache()
    torch.cuda.empty_cache()
    return output, elapsed


def main(config: Config) -> None:
    """Resolve hand identity, reconstruct and score one matched frame window."""
    if not 0.0 <= config.start_s <= config.end_s <= 6.2 or config.stride < 1 or config.batch_frames < 1:
        raise ValueError("This experiment is restricted to 0..6.2 seconds with positive stride")
    config.output.mkdir(parents=True, exist_ok=True)
    table: pa.Table = load_window(config.output / "window.parquet")
    camera: OAKCamera = camera_from_window(table)
    samples: pa.Table = table.filter(table[f"{VIDEO}:VideoStream:sample"].is_valid())
    times: Int64[ndarray, "v"] = samples["video_time"].to_numpy().view(np.int64)
    keys: list[bool] = catalog_keyframes(samples[f"{VIDEO}:VideoStream:is_keyframe"])
    first: int = next(i for i, key in enumerate(keys) if key)
    video: CatalogVideo = CatalogVideo(
        times[first:],
        packet_views(samples[f"{VIDEO}:VideoStream:sample"])[first:],
        keys[first:],
        catalog_codec(samples[f"{VIDEO}:VideoStream:codec"], VIDEO),
    )
    fps: int = round(1e9 / float(np.median(np.diff(video.t_ns))))
    decoder: VideoDecoder = VideoDecoder(
        wrap_mp4(video.samples, video.keyframes, fps, video.codec), device="cuda", dimension_order="NHWC", seek_mode="exact"
    )
    chosen: Int64[ndarray, "n"] = np.flatnonzero((video.t_ns >= int(config.start_s * 1e9)) & (video.t_ns <= int(config.end_s * 1e9)))[
        :: config.stride
    ]
    frame_times: Float64[ndarray, "n"] = video.t_ns[chosen].astype(np.float64) / 1e9
    frames: UInt8[Tensor, "n h w 3"] = decoder.get_frames_at(chosen.tolist()).data
    reference: int = int(np.argmin(abs(frame_times - 4.144)))
    Image.fromarray(frames[reference].cpu().numpy()).save(config.output / "frame-4.144.png")
    clock_path: Path = config.output / "clock.parquet"
    if not clock_path.exists():
        dataset = CatalogClient("rerun+http://dgx-spark:51235").get_dataset(name="selfcap-post-ep1")
        pq.write_table(dataset.filter_segments(SEGMENT).filter_contents(["/__properties/time_sync"]).reader(index=None).to_arrow_table(), clock_path)
    quest: Float32[ndarray, "n 2 21 2"] = quest_pixels(table, pq.read_table(clock_path), camera, frame_times)
    sapiens: Float32[ndarray, "n 2 21 2"] = np.full_like(quest, np.nan)
    sapiens_ms: float = 0.0
    if config.compare_sapiens:
        sapiens, sapiens_ms = run_sapiens(frames)
    weights: Path = config.output / "weights"
    detector: EgoHandDetector = EgoHandDetectorConfig(
        weights / "detector/hand_detector.onnx",
        backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=4, cache_dir=config.output / "engines"),
    ).setup()
    detector(frames[:8])
    torch.cuda.synchronize()
    started: float = time.perf_counter()
    detections: HandDetections = detector(frames)
    torch.cuda.synchronize()
    detector_ms: float = (time.perf_counter() - started) * 1000.0 / len(frames)
    predictor: EgoHandPredictor = EgoHandConfig(
        weights / "egohand/egohand_s060000.ckpt",
        weights / "soma",
        batch_size=32,
        gpu_crops=True,
        backbone_onnx=config.output / "dinov3-bf16.onnx",
        backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=32, cache_dir=config.output / "engines"),
    ).setup()
    ego: Float32[ndarray, "n 2 21 2"] = np.full_like(quest, np.nan)
    sharpness: Float64[ndarray, "n 2"] = np.full((len(frames), 2), np.nan)
    predictor_ms: list[float] = []
    results_by_frame: list[tuple[HandReconstruction, list[int]]] = []
    for start in range(0, len(frames), config.batch_frames):
        stop: int = min(start + config.batch_frames, len(frames))
        selected = torch.where((detections.boxes.frame_indices >= start) & (detections.boxes.frame_indices < stop))[0].tolist()
        batch_detections = HandDetections(
            BoxDetections(detections.boxes.xyxy[selected], detections.boxes.scores[selected], detections.boxes.frame_indices[selected] - start),
            tuple(detections.sides[i] for i in selected),
        )
        if start == 0:
            predictor(frames[start:stop], batch_detections, camera)
            predictor.stage_ms.clear()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        batch_result = predictor(frames[start:stop], batch_detections, camera)
        torch.cuda.synchronize()
        predictor_ms.extend([(time.perf_counter() - started) * 1000.0 / (stop - start)] * (stop - start))
        for index in range(stop - start):
            results_by_frame.append((batch_result, torch.where(batch_result.keypoints.frame_indices == index)[0].tolist()))
    hand_count: int = 0
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="Same-frame OAK: EgoHand / Sapiens2 / Quest",
                    origin="/image",
                    visual_bounds=rrb.VisualBounds2D(x_range=[0, camera.width], y_range=[0, camera.height]),
                ),
                rrb.Spatial3DView(
                    name="Identity-corrected EgoHand (OAK metres)",
                    origin="/camera_space",
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(position=[0.0, 0.0, -0.7], look_target=[0.0, 0.15, 0.6], eye_up=[0.0, -1.0, 0.0]),
                ),
            ),
            collapse_panels=True,
        )
    )
    annotations: rr.AnnotationContext = rr.AnnotationContext(
        rr.ClassDescription(info=rr.AnnotationInfo(id=0, label="SOMA hand"), keypoint_connections=SOMA_HAND.links)
    )
    rr.log("/camera_space", rr.ViewCoordinates.RDF, static=True)
    rr.log("/image/egohand", annotations, static=True)
    rr.log("/camera_space/egohand", annotations, static=True)
    rr.log(
        "/notes",
        rr.TextDocument(
            "Cyan left / orange right: EgoHand after upstream geometry + IoU identity assignment, before crop mirroring. Green: full-frame Sapiens2 0.4B. Magenta: Quest, using capture time + offset - 48 ms; interpolated OAK pose and 14-term projection. Metrics use identical finite joint slots."
        ),
        static=True,
    )
    for index in range(len(frames)):
        selected: list[int] = torch.where(detections.boxes.frame_indices == index)[0].tolist()
        local: HandDetections = HandDetections(
            BoxDetections(
                detections.boxes.xyxy[selected],
                detections.boxes.scores[selected],
                torch.zeros(len(selected), device=frames.device, dtype=torch.int64),
            ),
            tuple(detections.sides[i] for i in selected),
        )
        result, hand_indices = results_by_frame[index]
        rr.set_time("video_time", duration=float(frame_times[index]))
        rr.log("/image/rgb", rr.Image(frames[index].cpu().numpy()).compress(jpeg_quality=90))
        rr.log("/image/egohand", rr.Clear(recursive=True))
        rr.log("/camera_space/egohand", rr.Clear(recursive=True))
        rr.log("/image/egohand", annotations)
        rr.log("/camera_space/egohand", annotations)
        for hand in hand_indices:
            side = result.sides[hand]
            side_index: int = 0 if side == "left" else 1
            color: tuple[int, int, int] = (65, 210, 255) if side == "left" else (255, 165, 65)
            ego[index, side_index] = result.keypoints.xy[hand, list(SOMA_HAND21)].cpu().numpy()
            assert result.keypoints.xyz_camera is not None
            rr.log(
                f"/image/egohand/{side}",
                rr.Points2D(result.keypoints.xy[hand].cpu().numpy(), keypoint_ids=range(25), class_ids=0, colors=color, radii=3.0, show_labels=False),
            )
            rr.log(
                f"/camera_space/egohand/{side}",
                rr.Points3D(
                    result.keypoints.xyz_camera[hand].cpu().numpy(), keypoint_ids=range(25), class_ids=0, colors=color, radii=0.003, show_labels=False
                ),
            )
            rr.log(
                f"/camera_space/egohand/{side}/mesh",
                rr.Mesh3D(
                    vertex_positions=result.vertices[hand].cpu().numpy(),
                    triangle_indices=predictor.predictor.model.soma_hand.faces(side),
                    albedo_factor=(*[c / 255 for c in color], 0.25),
                ),
            )
            rr.log(
                f"/camera_space/egohand/{side}/params",
                rr.AnyValues(
                    rotations=result.rotations[hand].cpu().numpy().reshape(-1),
                    translation=result.translation[hand].cpu().numpy(),
                    identity=result.identity[hand].cpu().numpy(),
                    scale=float(result.scale[hand]),
                    scale_parameters=result.scale_parameters[hand].cpu().numpy() if result.scale_parameters is not None else [],
                ),
            )
        sides: tuple[HandSide, HandSide] = ("left", "right")
        for side, name in enumerate(sides):
            for source, positions, color in (("quest", quest, (255, 75, 180)), ("sapiens", sapiens, (100, 255, 100))):
                rr.log(f"/image/{source}/{name}", rr.Points2D(positions[index, side], colors=color, radii=2.0, show_labels=False))
        for box, name in zip(local.boxes.xyxy_numpy(), local.sides, strict=True):
            side_index = 0 if name == "left" else 1
            x0, y0, x1, y1 = np.rint(box).astype(int)
            crop = frames[index, max(0, y0) : min(camera.height, y1), max(0, x0) : min(camera.width, x1)].cpu().numpy()
            if crop.size:
                gray = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_RGB2GRAY)
                sharpness[index, side_index] = cv2.Laplacian(gray, cv2.CV_64F).var()
        hand_count += len(hand_indices)
        print(f"{frame_times[index]:.3f}s: {len(hand_indices)} hands; inference {predictor_ms[index]:.1f} ms", flush=True)
    per_joint: dict[str, ErrorSummary] = {}
    for side, name in enumerate(("left", "right")):
        for joint, joint_name in enumerate(HAND_21.keypoint_names):
            mask = np.zeros((len(frames), 2, 21), dtype=bool)
            mask[:, side, joint] = True
            per_joint[f"{name}/{joint_name}"] = summarize(ego, sapiens, quest, mask)
    low, high = np.nanpercentile(sharpness, [33, 67])
    groups: dict[str, ErrorSummary] = {
        "low_sharpness_third": summarize(ego, sapiens, quest, (sharpness <= low)[..., None]),
        "high_sharpness_third": summarize(ego, sapiens, quest, (sharpness >= high)[..., None]),
    }
    report: RunReport = RunReport(
        SEGMENT,
        len(frames),
        hand_count,
        summarize(ego, sapiens, quest),
        per_joint,
        groups,
        {
            "requested": len(frames) * 42,
            "egohand_finite": int(np.isfinite(ego).all(-1).sum()),
            "sapiens_finite": int(np.isfinite(sapiens).all(-1).sum()),
            "quest_finite": int(np.isfinite(quest).all(-1).sum()),
        },
        {
            "detector": detector_ms,
            "sapiens": sapiens_ms,
            "predictor": float(np.median(predictor_ms)),
            **{key: float(np.median(values)) for key, values in predictor.stage_ms.items()},
        },
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20,
    )
    (config.output / "report.json").write_text(to_json(report))
    recording: rr.RecordingStream | None = rr.get_global_data_recording()
    assert recording is not None
    recording.flush()
    print(to_json(report), flush=True)
