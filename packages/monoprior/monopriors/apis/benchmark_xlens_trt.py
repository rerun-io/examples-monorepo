"""Benchmark eager, frozen-geometry, and TensorRT X-Lens on a four-camera robocap fisheye rig.

Frames come from one robocap catalog segment at two network sizes: 896x504
(aspect-preserving resize of 1920x1080, a stage-1 training size) and 798x504
(centre-crop to 1.583:1 then resize, the paper's fisheye size). Every mode is
timed through the ``BaseRigDepthPredictor`` call, so the numbers include image
upload, normalisation, geometry-cache lookup, and output copies. Parity compares
each mode against the eager fp32 frozen-geometry path.
"""

import gc
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, cast, get_args, runtime_checkable

import cv2
import numpy as np
import torch
from beartype.roar import BeartypeException
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, Int64, Shaped, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from torch import Tensor

from monopriors.models.rig_depth import BaseRigDepthPredictor, RigDepthPrediction, XLensPredictor, XLensTrtPredictor, camera_type, unit_rays
from monopriors.models.rig_depth.xlens_trt import XLENS_CACHE_DIR, EngineProfile

Mode: TypeAlias = Literal["eager-bf16", "eager-frozen-bf16", "trt-rig-static", "trt-rig", "trt-dynamic"]
"""``trt-rig-static``: one static engine per rig (batch 1); ``trt-rig``: per-rig engine with a dynamic frameset batch;
``trt-dynamic``: the predictor's default profile, one engine over its view/resolution ranges. All TensorRT modes replay CUDA graphs."""
ALL_MODES: tuple[Mode, ...] = get_args(Mode)

TIMELINE: str = "video_time"
RIG_PATH: str = "world/rig_00"
TimedeltaNs: TypeAlias = Shaped[ndarray, " samples"]
"""Sample timestamps in timeline order, dtype ``timedelta64[ns]`` (jaxtyping has no timedelta dtype)."""


@runtime_checkable
class VideoFrame(Protocol):
    """Decoded RGB frame returned by TorchCodec."""

    data: UInt8[Tensor, "3 height width"]


@runtime_checkable
class SegmentVideoDecoder(Protocol):
    """TorchCodec operation used by the benchmark's frame sampler."""

    def get_frame_at(self, index: int) -> VideoFrame:
        """Decode one frame by sample index."""
        ...


DecoderBundle: TypeAlias = tuple[TimedeltaNs, list[bytes], list[bool], SegmentVideoDecoder]
"""What ``open_segment_decoder`` returns: sample times, raw samples, keyframe flags, and the decoder."""


def nearest_time_index(times: TimedeltaNs, target_ns: int) -> int:
    """Return the sample nearest one nanosecond timestamp, clamped to endpoints.

    Args:
        times: Sample timestamps in timeline order, ``timedelta64[ns]``.
        target_ns: The wanted time in nanoseconds on the same timeline.

    Returns:
        Index of the nearest sample.

    Raises:
        ValueError: If *times* is empty.
    """
    if len(times) == 0:
        raise ValueError("cannot sample an empty timestamp sequence")
    numeric_times: Int64[ndarray, "samples"] = np.asarray(times, dtype="timedelta64[ns]").astype(np.int64)
    insertion: int = int(np.searchsorted(numeric_times, target_ns))
    if insertion == 0:
        return 0
    if insertion == len(numeric_times):
        return len(numeric_times) - 1
    before: int = insertion - 1
    return before if target_ns - int(numeric_times[before]) <= int(numeric_times[insertion]) - target_ns else insertion


@dataclass(frozen=True, slots=True)
class Config:
    """X-Lens rig benchmark configuration."""

    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog server URL."""
    dataset: str = "robocap"
    """Catalog dataset name."""
    segment_id: str = "robocap__f408193e6447b3b0__s00000021"
    """Robocap segment recording id."""
    cams: tuple[str, ...] = ("cam_00", "cam_01", "cam_04", "cam_05")
    """Outward rig cameras."""
    settings: tuple[tuple[int, int], ...] = ((504, 896), (504, 798))
    """Network height and width per setting; frames are centre-cropped to the target aspect and resized."""
    off_opt: tuple[tuple[int, tuple[int, int]], ...] = ((2, (504, 798)), (4, (630, 1120)))
    """View count and network size of the dynamic-engine rows away from its tuning shape (the second is the catalog tool's default)."""
    batch_sizes: tuple[int, ...] = (2, 4)
    """Frameset batches timed through ``predict_batch`` on the ``trt-rig`` engine."""
    warmup_iters: int = 10
    """Untimed calls before each measurement (the first TensorRT call also exports and builds)."""
    timed_iters: int = 50
    """Timed calls per mode and setting."""
    modes: tuple[Mode, ...] = ALL_MODES
    """Modes measured in display order."""
    checkpoint: Path | None = None
    """Local X-Lens safetensors, or None for the pinned gated download."""
    cache_dir: Path = field(default_factory=lambda: XLENS_CACHE_DIR)
    """ONNX and engine cache root."""
    workspace_gib: float = 8.0
    """TensorRT builder workspace cap."""
    max_batch_size: int = 4
    """``trt-rig`` engine batch profile maximum."""


@dataclass(frozen=True, slots=True)
class RigFrames:
    """One frameset with its rig geometry at network resolution."""

    source: str
    """Where the frames came from."""
    images: UInt8[ndarray, "s h w 3"]
    """RGB views."""
    rays: Float32[ndarray, "s h w 3"]
    """Camera-frame unit rays."""
    cam_types: Int64[ndarray, "s"]
    """X-Lens camera ids."""
    cam_T_ref: Float64[ndarray, "s 4 4"] | None
    """Camera-to-reference poses."""


@dataclass(frozen=True, slots=True)
class TimingRow:
    """Per-frameset latency of one mode at one setting."""

    mode: Mode
    """Predictor mode."""
    views: int
    """Views per frameset."""
    image_hw: tuple[int, int]
    """Network height and width."""
    batch: int
    """Framesets per call."""
    source: str
    """Frame source."""
    mean_ms: float
    """Mean call time in milliseconds (per frameset when ``batch`` > 1)."""
    median_ms: float
    """Median call time in milliseconds (per frameset when ``batch`` > 1)."""
    min_ms: float
    """Fastest call in milliseconds (per frameset when ``batch`` > 1)."""


@dataclass(frozen=True, slots=True)
class ParityRow:
    """Agreement of one mode with eager fp32 at one setting."""

    mode: Mode
    """Predictor mode."""
    views: int
    """Views per frameset."""
    image_hw: tuple[int, int]
    """Network height and width."""
    median_abs_rel: float
    """Median absolute relative depth error over valid pixels."""
    p95_abs_rel: float
    """95th percentile absolute relative depth error."""
    scale_rel_diff: float
    """Relative difference of the metric scaling factor."""


def crop_fisheye(camera: Fisheye62Parameters, x0: int, y0: int, width: int, height: int) -> Fisheye62Parameters:
    """Copy a Kannala-Brandt camera for a pixel-aligned crop (principal point shifts, distortion unchanged)."""
    intrinsics: Intrinsics = camera.intrinsics
    if intrinsics.fl_x is None or intrinsics.fl_y is None or intrinsics.cx is None or intrinsics.cy is None:
        raise ValueError(f"{camera.name}: camera intrinsics are incomplete")
    cropped = Intrinsics(
        camera_conventions=intrinsics.camera_conventions,
        fl_x=float(intrinsics.fl_x),
        fl_y=float(intrinsics.fl_y),
        cx=float(intrinsics.cx) - x0,
        cy=float(intrinsics.cy) - y0,
        height=height,
        width=width,
    )
    return Fisheye62Parameters(name=camera.name, extrinsics=camera.extrinsics, intrinsics=cropped, distortion=camera.distortion)


def fit_to_network(frame: UInt8[ndarray, "h0 w0 3"], camera: Fisheye62Parameters, image_hw: tuple[int, int]) -> tuple[UInt8[ndarray, "h w 3"], Fisheye62Parameters]:
    """Centre-crop a frame and its camera to the network aspect, then resize both to the network size."""
    from monopriors.apis.rig_depth_catalog import rescaled_fisheye

    source_height: int = frame.shape[0]
    source_width: int = frame.shape[1]
    target_aspect: float = image_hw[1] / image_hw[0]
    crop_width: int = min(source_width, round(source_height * target_aspect))
    crop_height: int = min(source_height, round(crop_width / target_aspect))
    x0: int = (source_width - crop_width) // 2
    y0: int = (source_height - crop_height) // 2
    cropped: UInt8[ndarray, "hc wc 3"] = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]
    resized: UInt8[ndarray, "h w 3"] = cv2.resize(cropped, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_AREA)
    cropped_camera: Fisheye62Parameters = crop_fisheye(camera, x0, y0, crop_width, crop_height)
    return resized, rescaled_fisheye(cropped_camera, width=image_hw[1], height=image_hw[0])


def catalog_frames(config: Config, cams: tuple[str, ...]) -> tuple[dict[str, UInt8[ndarray, "h0 w0 3"]], dict[str, Fisheye62Parameters]]:
    """Decode the first shared frameset of the segment for ``cams`` and read their cameras."""
    from rerun.catalog import CatalogClient, DatasetEntry, DatasetView
    from simplecv.rerun_dataloader import open_segment_decoder

    from monopriors.apis.stereo_catalog import read_fisheye_camera

    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    view: DatasetView = dataset.filter_segments(config.segment_id)
    cameras: dict[str, Fisheye62Parameters] = {cam: read_fisheye_camera(view, cam) for cam in cams}
    device = torch.device("cuda")
    decoders: dict[str, DecoderBundle] = {
        cam: cast(DecoderBundle, open_segment_decoder(dataset, config.segment_id, f"{RIG_PATH}/{cam}/pinhole/video", TIMELINE, device, 30))
        for cam in cams
    }
    shared_start_ns: int = max(int(bundle[0][0].astype(np.int64)) for bundle in decoders.values())
    frames: dict[str, UInt8[ndarray, "h0 w0 3"]] = {}
    for cam in cams:
        index: int = nearest_time_index(decoders[cam][0], shared_start_ns)
        frame_chw: UInt8[Tensor, "3 h0 w0"] = decoders[cam][3].get_frame_at(index).data
        frames[cam] = rearrange(frame_chw, "c h w -> h w c").cpu().numpy()  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    return frames, cameras


def rig_frames(source: str, images: UInt8[ndarray, "s h w 3"], cameras: list[Fisheye62Parameters]) -> RigFrames:
    """Rig geometry arrays of ``cameras`` beside their images."""
    return RigFrames(
        source=source,
        images=images,
        rays=np.stack([unit_rays(camera) for camera in cameras]),
        cam_types=np.asarray([camera_type(camera) for camera in cameras], dtype=np.int64),
        cam_T_ref=np.stack([np.asarray(camera.extrinsics.world_T_cam, dtype=np.float64) for camera in cameras]),
    )


def catalog_rig_frames(
    frames: dict[str, UInt8[ndarray, "h0 w0 3"]], cameras: dict[str, Fisheye62Parameters], cams: tuple[str, ...], image_hw: tuple[int, int]
) -> RigFrames:
    """Fit decoded catalog frames and cameras of ``cams`` to one network size."""
    fitted: list[tuple[UInt8[ndarray, "h w 3"], Fisheye62Parameters]] = [fit_to_network(frames[cam], cameras[cam], image_hw) for cam in cams]
    return rig_frames("robocap catalog", np.stack([image for image, _ in fitted]), [camera for _, camera in fitted])


def synthetic_rig_frames(n_views: int, image_hw: tuple[int, int], seed: int = 3) -> RigFrames:
    """Equidistant fisheye rig (KB4 with zero higher-order terms) at 90-degree yaw steps with noise images."""
    height, width = image_hw
    generator: np.random.Generator = np.random.default_rng(seed)
    images: UInt8[ndarray, "s h w 3"] = generator.integers(0, 256, size=(n_views, height, width, 3), dtype=np.uint8)
    focal: float = width / np.pi
    cameras: list[Fisheye62Parameters] = []
    for view in range(n_views):
        yaw: float = view * np.pi / 2.0
        world_R_cam: Float64[ndarray, "3 3"] = np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]])
        cameras.append(
            Fisheye62Parameters(
                name=f"synthetic_{view}",
                extrinsics=Extrinsics(world_R_cam=world_R_cam, world_t_cam=world_R_cam @ np.array([0.0, 0.0, 0.05])),
                intrinsics=Intrinsics(camera_conventions="RDF", fl_x=focal, fl_y=focal, cx=width / 2.0, cy=height / 2.0, height=height, width=width),
                distortion=KannalaBrandtDistortion(k1=0.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0, p1=0.0, p2=0.0),
            )
        )
    return rig_frames("synthetic", images, cameras)


def build_predictor(mode: Mode, config: Config) -> BaseRigDepthPredictor:
    """Instantiate the predictor of one benchmark mode."""
    if mode == "eager-bf16":
        return XLensPredictor(device="cuda", checkpoint=config.checkpoint, amp="bf16", freeze_geometry=False)
    if mode == "eager-frozen-bf16":
        return XLensPredictor(device="cuda", checkpoint=config.checkpoint, amp="bf16", freeze_geometry=True)
    profile: EngineProfile = "dynamic" if mode == "trt-dynamic" else "rig"
    return XLensTrtPredictor(
        checkpoint=config.checkpoint,
        cache_dir=config.cache_dir,
        workspace_gib=config.workspace_gib,
        profile=profile,
        max_batch_size=config.max_batch_size if mode == "trt-rig" else 1,
    )


def time_calls(run: Callable[[], object], warmup_iters: int, timed_iters: int) -> list[float]:
    """Per-call milliseconds of ``run`` under explicit synchronisation, after untimed warm-ups."""
    for _warmup_index in range(warmup_iters):
        run()
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _timed_index in range(timed_iters):
        torch.cuda.synchronize()
        started: float = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    return samples_ms


def parity(mode: Mode, views: int, image_hw: tuple[int, int], prediction: RigDepthPrediction, reference: RigDepthPrediction) -> ParityRow:
    """Compare one prediction with the eager fp32 reference."""
    valid: Bool[Tensor, "s h w"] = (reference.depth_m > 0.0) & torch.isfinite(reference.depth_m) & torch.isfinite(prediction.depth_m)
    abs_rel: Float32[Tensor, "n"] = ((prediction.depth_m - reference.depth_m).abs() / reference.depth_m)[valid]
    return ParityRow(
        mode=mode,
        views=views,
        image_hw=image_hw,
        median_abs_rel=float(abs_rel.median()),
        p95_abs_rel=float(torch.quantile(abs_rel, 0.95)),
        scale_rel_diff=abs(prediction.scale / reference.scale - 1.0),
    )


def artifact_summary(predictor: XLensTrtPredictor) -> str:
    """ONNX size, engine size, and build time of the engine currently loaded."""
    engine_path: Path | None = predictor.engine_path
    if engine_path is None:
        return "no engine"
    manifest: dict[str, Any] = json.loads(engine_path.with_suffix(engine_path.suffix + ".json").read_text())
    onnx_path = Path(manifest["onnx_path"])
    onnx_bytes: int = onnx_path.stat().st_size + (onnx_path.with_suffix(".onnx.data").stat().st_size if onnx_path.with_suffix(".onnx.data").exists() else 0)
    return (
        f"onnx {onnx_path.name} {onnx_bytes / 1e6:.1f} MB; engine {engine_path.name} {engine_path.stat().st_size / 1e6:.1f} MB; "
        f"build {float(manifest['build_seconds']):.1f} s"
    )


def print_tables(timings: list[TimingRow], parities: list[ParityRow]) -> None:
    """Print the timing and parity tables as Markdown."""
    print("\n| Mode | Views | Network H×W | Batch | Source | mean ms/frameset | median ms | min ms | FPS (mean) | FPS (median) |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in timings:
        print(
            f"| {row.mode} | {row.views} | {row.image_hw[0]}×{row.image_hw[1]} | {row.batch} | {row.source} | {row.mean_ms:.2f} | {row.median_ms:.2f} | "
            f"{row.min_ms:.2f} | {1000.0 / row.mean_ms:.2f} | {1000.0 / row.median_ms:.2f} |"
        )
    print("\n| Mode vs eager fp32 (frozen) | Views | Network H×W | median abs-rel | p95 abs-rel | scale rel diff |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in parities:
        print(f"| {row.mode} | {row.views} | {row.image_hw[0]}×{row.image_hw[1]} | {row.median_abs_rel:.5f} | {row.p95_abs_rel:.5f} | {row.scale_rel_diff:.5f} |")


def main(config: Config) -> None:
    """Run every mode at every setting and print Markdown tables.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("the X-Lens benchmark requires CUDA")
    if config.timed_iters < 1 or config.warmup_iters < 0:
        raise ValueError("timed_iters must be positive and warmup_iters non-negative")

    source_frames: dict[str, UInt8[ndarray, "h0 w0 3"]] | None = None
    cameras: dict[str, Fisheye62Parameters] | None = None
    try:
        source_frames, cameras = catalog_frames(config, config.cams)
        first: UInt8[ndarray, "h0 w0 3"] = next(iter(source_frames.values()))
        print(f"catalog frames: {len(source_frames)} cameras at {first.shape[1]}x{first.shape[0]} from {config.segment_id}")
    except BeartypeException:
        raise
    except Exception as error:  # noqa: BLE001 — the catalog is optional; report and fall back.
        print(f"catalog unavailable ({type(error).__name__}: {error}); timing on synthetic equidistant-fisheye frames instead")

    def frames_for(views: int, image_hw: tuple[int, int]) -> RigFrames:
        if source_frames is not None and cameras is not None:
            return catalog_rig_frames(source_frames, cameras, config.cams[:views], image_hw)
        return synthetic_rig_frames(views, image_hw)

    def reference_for(frames: RigFrames) -> RigDepthPrediction:
        reference_predictor = XLensPredictor(device="cuda", checkpoint=config.checkpoint, amp="fp32", freeze_geometry=True)
        prediction: RigDepthPrediction = reference_predictor(frames.images, frames.rays, frames.cam_types, frames.cam_T_ref)
        del reference_predictor
        gc.collect()
        torch.cuda.empty_cache()
        return prediction

    def release() -> None:
        """Return freed GPU memory once the caller has dropped its own references."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    timings: list[TimingRow] = []
    parities: list[ParityRow] = []
    # One dynamic engine serves every shape below; it is built at the first setting and kept.
    dynamic: XLensTrtPredictor | None = cast(XLensTrtPredictor, build_predictor("trt-dynamic", config)) if "trt-dynamic" in config.modes else None

    def measure(mode: Mode, predictor: BaseRigDepthPredictor, frames: RigFrames, reference: RigDepthPrediction) -> None:
        views: int = int(frames.images.shape[0])
        image_hw: tuple[int, int] = (int(frames.images.shape[1]), int(frames.images.shape[2]))
        prediction: RigDepthPrediction = predictor(frames.images, frames.rays, frames.cam_types, frames.cam_T_ref)
        samples_ms: list[float] = time_calls(partial(predictor, frames.images, frames.rays, frames.cam_types, frames.cam_T_ref), config.warmup_iters, config.timed_iters)
        timings.append(
            TimingRow(
                mode=mode, views=views, image_hw=image_hw, batch=1, source=frames.source,
                mean_ms=statistics.fmean(samples_ms), median_ms=statistics.median(samples_ms), min_ms=min(samples_ms),
            )
        )
        parities.append(parity(mode, views, image_hw, prediction, reference))
        print(f"{mode} ({views}v {image_hw[0]}x{image_hw[1]}): mean {timings[-1].mean_ms:.2f} ms, median {timings[-1].median_ms:.2f} ms, peak torch {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        if isinstance(predictor, XLensTrtPredictor):
            print(f"  {artifact_summary(predictor)}")
            print(f"  {predictor.runtime_summary()}")
            if mode == "trt-rig":
                for batch in config.batch_sizes:
                    if batch <= predictor.max_batch_size:
                        stacked: UInt8[ndarray, "b s h w 3"] = np.stack([frames.images] * batch)
                        call_ms: list[float] = time_calls(
                            partial(predictor.predict_batch, stacked, frames.rays, frames.cam_types, frames.cam_T_ref), max(1, config.warmup_iters // 2), config.timed_iters
                        )
                        batched_ms: list[float] = [ms / batch for ms in call_ms]
                        timings.append(
                            TimingRow(
                                mode=mode, views=views, image_hw=image_hw, batch=batch, source=frames.source,
                                mean_ms=statistics.fmean(batched_ms), median_ms=statistics.median(batched_ms), min_ms=min(batched_ms),
                            )
                        )
                        print(f"  predict_batch B={batch}: mean {timings[-1].mean_ms:.2f} ms/frameset")

    for image_hw in config.settings:
        frames: RigFrames = frames_for(len(config.cams), image_hw)
        print(f"\n== {image_hw[0]}x{image_hw[1]}, {frames.images.shape[0]} views, source={frames.source}, cam_types={frames.cam_types.tolist()} ==")
        reference: RigDepthPrediction = reference_for(frames)
        for mode in config.modes:
            if mode == "trt-dynamic":
                if dynamic is not None:
                    measure(mode, dynamic, frames, reference)
                continue
            predictor: BaseRigDepthPredictor = build_predictor(mode, config)
            measure(mode, predictor, frames, reference)
            del predictor
            release()
        del reference
        release()

    if dynamic is not None:
        for views, image_hw in config.off_opt:
            if views > len(config.cams):
                print(f"skipping {views}-view row: only {len(config.cams)} cameras configured")
                continue
            frames = frames_for(views, image_hw)
            print(f"\n== off-optimum {views} views at {image_hw[0]}x{image_hw[1]}, source={frames.source} ==")
            reference = reference_for(frames)
            eager_frozen: BaseRigDepthPredictor = build_predictor("eager-frozen-bf16", config)
            measure("eager-frozen-bf16", eager_frozen, frames, reference)
            del eager_frozen
            release()
            measure("trt-dynamic", dynamic, frames, reference)
            del reference
            release()
    print_tables(timings, parities)
