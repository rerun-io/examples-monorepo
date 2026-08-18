"""In-process quantization and publication of trained gsplat products."""

import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import rerun as rr
import torch
from arkitscenes_download.ingest.depth import encode_depth_png
from arkitscenes_download.ingest.paths import (
    DEPTH_SPLAT_ULTRAWIDE,
    DEPTH_SPLAT_WIDE,
    ERROR_DEPTH_ULTRAWIDE_RECT,
    ERROR_NORMAL_ULTRAWIDE_RECT,
    ERROR_RGB_ULTRAWIDE_RECT,
    NORMALS_SPLAT_ULTRAWIDE_RECT,
    NORMALS_SPLAT_WIDE,
    RGB_SPLAT_ULTRAWIDE_RECT,
    TIMELINE,
)
from arkitscenes_download.ingest.recording import atomic_recording
from arkitscenes_download.schema import DEPTH_RANGE_MM
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetView

from gauss_surf.catalog import SegmentReader, register_layer
from gauss_surf.contracts import (
    PROMPTDA_DEPTH_BLOB_COLUMN,
    PROMPTDA_LAYER,
    RENDER_BACKGROUND,
    RGB_JPEG_QUALITY,
    SPLAT_DEPTH_LAYER,
    SPLAT_LAYER,
    SPLAT_TRIAGE_LAYER,
    ULTRAWIDE_DEPTH_LAYER,
    ULTRAWIDE_NORMALS_LAYER,
    CameraTag,
)
from gauss_surf.normals_encoding import decode_normal_codes, decode_normals_png
from gauss_surf.render_io import (
    RenderCamera,
    decode_rgb_image,
    decode_splat_depth_png,
    encode_rgb_jpeg,
    encode_rgb_png,
    load_render_cameras,
)
from gauss_surf.train_gsplat.cache import RasterCamera, raster_camera_from_render_camera
from gauss_surf.train_gsplat.core import opacity_export_mask
from gauss_surf.train_gsplat.layer_metrics import (
    DEPTH_ERROR_MAXIMUM_M,
    NORMAL_ERROR_MAXIMUM_DEGREES,
    RGB_ERROR_MAXIMUM,
    ULTRAWIDE_REFERENCE_COLUMN,
    CameraSummary,
    SplatTriageWriteStats,
    TriageErrors,
    TriageReferenceBlobs,
    TriageWriteStats,
    compute_triage_errors,
    load_reference_blobs,
    reference_blobs_at_component_timestamps,
    resize_reference_nearest,
    summarize_metrics,
)
from gauss_surf.train_gsplat.renderer import RenderOutput, render_splats
from gauss_surf.train_gsplat.splat_values import GaussianSplats, inria_colors_rgba, inria_quaternions_xyzw


def quantize_depth_m_to_mm(depth_m_hw: torch.Tensor) -> torch.Tensor:
    """Quantize metre depth on-device to uint16 millimetres with zero invalid."""
    valid_hw: torch.Tensor = torch.isfinite(depth_m_hw) & (depth_m_hw > 0.0)
    safe_m_hw: torch.Tensor = torch.where(valid_hw, depth_m_hw, torch.zeros_like(depth_m_hw))
    return torch.round((safe_m_hw * 1000.0).clamp(0.0, 65535.0)).to(dtype=torch.uint16)


def quantize_normal_to_uint8(normal_hw3: torch.Tensor) -> torch.Tensor:
    """Quantize signed normals on-device with round-to-nearest central code 128."""
    return torch.round((normal_hw3.clamp(-1.0, 1.0) + 1.0) * 127.5).to(dtype=torch.uint8)


@dataclass(frozen=True, slots=True)
class WideQuantizedRender:
    """One wide full-grid render after on-device storage quantization and D2H."""

    camera: RenderCamera
    """Camera identity, timestamp, pose, and intrinsics."""
    depth_mm_hw: UInt16[ndarray, "h w"]
    """Plane-pass z-depth in uint16 millimetres with zero invalid."""
    normal_rgb_hw3: UInt8[ndarray, "h w 3"]
    """Direct plane normal encoded to the lossless uint8 storage contract."""


@dataclass(frozen=True, slots=True)
class UltrawideQuantizedRender:
    """One ultrawide full-grid render after on-device storage quantization and D2H."""

    camera: RenderCamera
    """Camera identity, timestamp, pose, and intrinsics."""
    depth_mm_hw: UInt16[ndarray, "h w"]
    """Plane-pass z-depth in uint16 millimetres with zero invalid."""
    normal_rgb_hw3: UInt8[ndarray, "h w 3"]
    """Direct plane normal encoded to the lossless uint8 storage contract."""
    rgb_hw3: UInt8[ndarray, "h w 3"]
    """Fitted ultrawide RGB quantized to uint8."""


PublicationProduct: TypeAlias = WideQuantizedRender | UltrawideQuantizedRender
QuantizedDepthNormal: TypeAlias = tuple[
    RenderOutput,
    UInt16[ndarray, "h w"],
    UInt8[ndarray, "h w 3"],
]


@dataclass(frozen=True, slots=True)
class PublishStats:
    """In-process render, metric, recording, and registration outcomes."""

    wall_seconds: float
    """Complete in-process publication wall, including RRD writes and registration."""
    render_wall_seconds: float
    """Accumulated GPU render, quantization, and D2H wall."""
    frame_count: int
    """Number of full-grid cameras rendered."""
    frames_per_second: float
    """Full-grid product count divided by render wall."""
    gaussian_count: int
    """Opacity-filtered Gaussian count written to the splat layer."""
    d2h_transfers: dict[str, int]
    """D2H call counts by product signal."""
    wide_depth: CameraSummary
    """Wide plane-depth coverage and PromptDA comparison."""
    ultrawide_depth: CameraSummary
    """Ultrawide plane-depth coverage and mesh-depth comparison."""
    triage: SplatTriageWriteStats
    """Wide-normal counts plus ultrawide fit-error summaries."""
    rrd_paths: dict[str, str]
    """Registered layer names mapped to local RRD paths."""


@dataclass(slots=True)
class _DepthMetrics:
    """Streaming depth metric accumulator for one camera."""

    camera: CameraTag
    """Camera summarized by this accumulator."""
    reference_blobs: dict[int, bytes]
    """Reference depth PNGs keyed by exact timestamp."""
    coverage_fractions: list[float] = field(default_factory=list)
    """Per-frame valid fractions."""
    seen_timestamps: set[int] = field(default_factory=set)
    """Rendered timestamps used to audit reference completeness."""
    absolute_error_sum: float = 0.0
    """Float64-accumulated absolute depth error in metres."""
    joint_valid_pixels: int = 0
    """Joint-valid pixel count."""
    total_pixels: int = 0
    """All pixels in compared frames."""
    resized_reference_frames: int = 0
    """References resized to the rendered shape."""
    compared_frames: int = 0
    """Frames with an exact reference."""

    def add(self, timestamp_ns: int, splat_m_hw: Float32[ndarray, "h w"]) -> None:
        """Accumulate one quantized splat-depth product."""
        self.seen_timestamps.add(timestamp_ns)
        splat_valid_hw: ndarray = np.isfinite(splat_m_hw) & (splat_m_hw > 0.0)
        self.coverage_fractions.append(float(np.count_nonzero(splat_valid_hw) / splat_m_hw.size))
        reference_blob: bytes | None = self.reference_blobs.get(timestamp_ns)
        if reference_blob is None:
            return
        reference_m_hw: Float32[ndarray, "reference_h reference_w"] = decode_splat_depth_png(reference_blob)
        if reference_m_hw.shape != splat_m_hw.shape:
            self.resized_reference_frames += 1
            reference_m_hw = resize_reference_nearest(reference_m_hw, splat_m_hw.shape)
        reference_valid_hw: ndarray = np.isfinite(reference_m_hw) & (reference_m_hw > 0.0)
        joint_valid_hw: ndarray = splat_valid_hw & reference_valid_hw
        frame_joint_pixels: int = int(np.count_nonzero(joint_valid_hw))
        if frame_joint_pixels:
            errors_m: ndarray = np.abs(
                splat_m_hw[joint_valid_hw].astype(np.float64) - reference_m_hw[joint_valid_hw].astype(np.float64)
            )
            self.absolute_error_sum += float(np.sum(errors_m, dtype=np.float64))
        self.joint_valid_pixels += frame_joint_pixels
        self.total_pixels += splat_m_hw.size
        self.compared_frames += 1

    def summary(self) -> CameraSummary:
        """Validate completeness and return the final camera summary."""
        missing_timestamps: list[int] = [timestamp for timestamp in self.reference_blobs if timestamp not in self.seen_timestamps]
        if missing_timestamps:
            raise ValueError(f"{self.camera} reference has no exact render timestamp for {missing_timestamps[0]}")
        if not self.coverage_fractions:
            raise ValueError(f"no {self.camera} depth frames were rendered")
        coverage_n: Float32[ndarray, "n"] = np.asarray(self.coverage_fractions, dtype=np.float32)
        mae_m: float = self.absolute_error_sum / self.joint_valid_pixels if self.joint_valid_pixels else float("nan")
        joint_valid_fraction: float = self.joint_valid_pixels / self.total_pixels if self.total_pixels else float("nan")
        return CameraSummary(
            camera=self.camera,
            frame_count=len(self.coverage_fractions),
            compared_frames=self.compared_frames,
            coverage_min=float(np.min(coverage_n)),
            coverage_mean=float(np.mean(coverage_n)),
            coverage_max=float(np.max(coverage_n)),
            joint_valid_fraction=joint_valid_fraction,
            joint_valid_pixels=self.joint_valid_pixels,
            total_pixels=self.total_pixels,
            mae_m=mae_m,
            resized_reference_frames=self.resized_reference_frames,
        )


def _send_depth_batch(
    recording: rr.RecordingStream,
    entity_path: str,
    products: Sequence[PublicationProduct],
    blobs: list[bytes],
) -> None:
    """Send one encoded depth batch on the duration timeline."""
    timestamps_n: ndarray = np.asarray([product.camera.timestamp_ns for product in products], dtype=np.int64).astype("timedelta64[ns]")
    count: int = len(products)
    rr.send_columns(
        entity_path,
        indexes=[rr.TimeColumn(TIMELINE, duration=timestamps_n)],
        columns=rr.EncodedDepthImage.columns(
            blob=blobs,
            media_type=["image/png"] * count,
            meter=[1000.0] * count,
            depth_range=[DEPTH_RANGE_MM] * count,
        ),
        recording=recording,
    )


def _send_image_batch(
    recording: rr.RecordingStream,
    entity_path: str,
    products: Sequence[PublicationProduct],
    blobs: list[bytes],
    media_type: str,
) -> None:
    """Send one encoded image batch on the duration timeline."""
    timestamps_n: ndarray = np.asarray([product.camera.timestamp_ns for product in products], dtype=np.int64).astype("timedelta64[ns]")
    rr.send_columns(
        entity_path,
        indexes=[rr.TimeColumn(TIMELINE, duration=timestamps_n)],
        columns=rr.EncodedImage.columns(blob=blobs, media_type=[media_type] * len(products)),
        recording=recording,
    )


def metric_splats_from_live(splats: torch.nn.ParameterDict | dict[str, torch.Tensor]) -> GaussianSplats:
    """Convert opacity-filtered live metric tensors directly to Rerun values."""
    keep_n: torch.Tensor = opacity_export_mask(splats["opacities"].squeeze(-1))
    centers_n3: Float32[ndarray, "n 3"] = splats["means"][keep_n].detach().cpu().numpy().astype(np.float32, copy=False)
    log_scales_n3: Float32[ndarray, "n 3"] = splats["scales"][keep_n].detach().cpu().numpy().astype(np.float32, copy=False)
    quaternions_wxyz_n4: Float32[ndarray, "n 4"] = splats["quats"][keep_n].detach().cpu().numpy().astype(np.float32, copy=False)
    f_dc_n3: Float32[ndarray, "n 3"] = splats["sh0"][keep_n, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    opacity_logits_n: Float32[ndarray, "n"] = splats["opacities"][keep_n, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    sh_coefficients_n153: Float32[ndarray, "n 15 3"] = np.ascontiguousarray(
        splats["shN"][keep_n].detach().cpu().numpy(), dtype=np.float32
    )
    return GaussianSplats(
        centers_n3=centers_n3,
        scales_n3=np.exp(log_scales_n3).astype(np.float32, copy=False),
        quaternions_xyzw_n4=inria_quaternions_xyzw(quaternions_wxyz_n4),
        colors_rgba_n4=inria_colors_rgba(f_dc_n3, opacity_logits_n),
        sh_coefficients_n153=sh_coefficients_n153,
    )


def _write_metric_splats(rrd_path: Path, video_id: str, splats: GaussianSplats) -> None:
    """Write live metric-world Gaussian tensors as one static layer recording."""
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_recording(rrd_path, video_id, send_properties=False) as recording:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
        rr.log("world/splats", splats.as_archetype(), static=True, recording=recording)


def _quantized_depth_normal_render(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    render_camera: RenderCamera,
    background_3: torch.Tensor,
) -> QuantizedDepthNormal:
    """Render fused publication channels and transfer quantized depth and normal."""
    camera: RasterCamera = raster_camera_from_render_camera(render_camera)
    output: RenderOutput = render_splats(
        splats,
        camera,
        downscale=1,
        sh_degree=3,
        background_3=background_3,
        absgrad=False,
    )
    valid_hw1: torch.Tensor = output.surface_valid_hw1 & torch.isfinite(output.surface_depth_hw1) & (output.surface_depth_hw1 > 0.0)
    depth_m_hw: torch.Tensor = torch.where(
        valid_hw1.squeeze(-1), output.surface_depth_hw1.squeeze(-1), torch.zeros_like(output.surface_depth_hw1.squeeze(-1))
    )
    normal_hw3: torch.Tensor = torch.where(
        valid_hw1.expand_as(output.direct_normal_hw3), output.direct_normal_hw3, torch.zeros_like(output.direct_normal_hw3)
    )
    depth_mm_hw: UInt16[ndarray, "h w"] = quantize_depth_m_to_mm(depth_m_hw).cpu().numpy()
    normal_rgb_hw3: UInt8[ndarray, "h w 3"] = quantize_normal_to_uint8(normal_hw3).cpu().numpy()
    return output, depth_mm_hw, normal_rgb_hw3


def _quantized_wide_render(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    render_camera: RenderCamera,
    background_3: torch.Tensor,
) -> WideQuantizedRender:
    """Build one required depth-and-normal wide product."""
    rendered: QuantizedDepthNormal = _quantized_depth_normal_render(splats, render_camera, background_3)
    return WideQuantizedRender(camera=render_camera, depth_mm_hw=rendered[1], normal_rgb_hw3=rendered[2])


def _quantized_ultrawide_render(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    render_camera: RenderCamera,
    background_3: torch.Tensor,
) -> UltrawideQuantizedRender:
    """Build one required depth, normal, and RGB ultrawide product."""
    rendered: QuantizedDepthNormal = _quantized_depth_normal_render(splats, render_camera, background_3)
    rgb_hw3: UInt8[ndarray, "h w 3"] = (
        torch.round(rendered[0].rgb_hw3.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).cpu().numpy()
    )
    return UltrawideQuantizedRender(
        camera=render_camera,
        depth_mm_hw=rendered[1],
        normal_rgb_hw3=rendered[2],
        rgb_hw3=rgb_hw3,
    )


def _initialize_triage_recording(recording: rr.RecordingStream) -> None:
    """Write static product and fixed-scale diagnostic metadata."""
    recording.log(
        NORMALS_SPLAT_WIDE,
        rr.AnyValues(normal_convention="camera-space RDF, away-from-camera; direct Gaussian plane pass, no sign flip"),
        static=True,
    )
    recording.log(RGB_SPLAT_ULTRAWIDE_RECT, rr.AnyValues(jpeg_quality=RGB_JPEG_QUALITY), static=True)
    recording.log(
        NORMALS_SPLAT_ULTRAWIDE_RECT,
        rr.AnyValues(normal_convention="camera-space RDF, away-from-camera; direct Gaussian plane pass, no sign flip"),
        static=True,
    )
    for entity_path, quantity, maximum, unit in (
        (ERROR_RGB_ULTRAWIDE_RECT, "per-pixel mean absolute RGB error", RGB_ERROR_MAXIMUM, "normalized RGB [0,1]"),
        (ERROR_DEPTH_ULTRAWIDE_RECT, "absolute z-depth error", DEPTH_ERROR_MAXIMUM_M, "metres"),
        (ERROR_NORMAL_ULTRAWIDE_RECT, "normal angular error", NORMAL_ERROR_MAXIMUM_DEGREES, "degrees"),
    ):
        recording.log(
            entity_path,
            rr.AnyValues(
                colormap="viridis-like",
                error_quantity=quantity,
                error_range=[0.0, maximum],
                error_unit=unit,
                invalid_color="black",
            ),
            static=True,
        )


def _future_results(futures: list[Future[bytes]]) -> list[bytes]:
    """Collect encoded images in submission order."""
    return [future.result() for future in futures]


def publish_in_process(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    *,
    reader: SegmentReader,
    bundle_dir: Path,
    video_id: str,
    splat_output_dir: Path,
    depth_output_dir: Path,
    triage_output_dir: Path,
    batch_size: int = 8,
    encoder_workers: int = 8,
) -> PublishStats:
    """Render the full grid, write three RRDs, and register them from the training process."""
    if batch_size <= 0 or encoder_workers <= 0:
        raise ValueError("batch size and encoder worker count must be positive")
    cameras: list[RenderCamera] = load_render_cameras(bundle_dir / "cameras_all.json")
    wide_cameras: list[RenderCamera] = [camera for camera in cameras if camera.camera == "wide"]
    ultrawide_cameras: list[RenderCamera] = [camera for camera in cameras if camera.camera == "uw"]
    if not wide_cameras or not ultrawide_cameras:
        raise RuntimeError(
            f"full-grid manifest needs both cameras, got {len(wide_cameras)} wide and {len(ultrawide_cameras)} ultrawide"
        )
    reader.require_layers((PROMPTDA_LAYER, ULTRAWIDE_DEPTH_LAYER, ULTRAWIDE_NORMALS_LAYER))
    segment_view: DatasetView = reader.segment_view()
    wide_depth_references: dict[int, bytes] = reference_blobs_at_component_timestamps(
        segment_view,
        PROMPTDA_DEPTH_BLOB_COLUMN,
    )
    ultrawide_depth_references: dict[int, bytes] = reference_blobs_at_component_timestamps(segment_view, ULTRAWIDE_REFERENCE_COLUMN)
    triage_references: list[TriageReferenceBlobs] = load_reference_blobs(segment_view)
    triage_reference_by_timestamp: dict[int, TriageReferenceBlobs] = {
        reference.timestamp_ns: reference for reference in triage_references
    }
    if len(triage_reference_by_timestamp) != len(triage_references):
        raise ValueError("triage references contain duplicate timestamps")

    splat_output_dir.mkdir(parents=True, exist_ok=True)
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    triage_output_dir.mkdir(parents=True, exist_ok=True)
    splat_rrd_path: Path = splat_output_dir / f"{video_id}.rrd"
    depth_rrd_path: Path = depth_output_dir / f"{video_id}.rrd"
    triage_rrd_path: Path = triage_output_dir / f"{video_id}.rrd"

    started_at: float = time.perf_counter()
    rerun_splats: GaussianSplats = metric_splats_from_live(splats)
    _write_metric_splats(splat_rrd_path, video_id, rerun_splats)
    wide_depth_metrics = _DepthMetrics(camera="wide", reference_blobs=wide_depth_references)
    ultrawide_depth_metrics = _DepthMetrics(camera="uw", reference_blobs=ultrawide_depth_references)
    rgb_means: list[float] = []
    depth_means_m: list[float] = []
    normal_means_degrees: list[float] = []
    diagnostic_frames: int = 0
    render_wall_seconds: float = 0.0
    device: torch.device = splats["means"].device
    background_3: torch.Tensor = torch.tensor(RENDER_BACKGROUND, dtype=torch.float32, device=device)
    d2h_transfers: dict[str, int] = {"depth": 0, "normal": 0, "rgb": 0}

    with (
        atomic_recording(depth_rrd_path, video_id, send_properties=False) as depth_recording,
        atomic_recording(triage_rrd_path, video_id, send_properties=False) as triage_recording,
        ThreadPoolExecutor(max_workers=encoder_workers, thread_name_prefix="gauss-surf-png") as executor,
        torch.inference_mode(),
    ):
        _initialize_triage_recording(triage_recording)
        rendered_count: int = 0
        for batch_start in range(0, len(wide_cameras), batch_size):
            wide_batch: list[RenderCamera] = wide_cameras[batch_start : batch_start + batch_size]
            render_started_at: float = time.perf_counter()
            wide_products: list[WideQuantizedRender] = [
                _quantized_wide_render(splats, camera, background_3) for camera in wide_batch
            ]
            render_wall_seconds += time.perf_counter() - render_started_at
            d2h_transfers["depth"] += len(wide_products)
            d2h_transfers["normal"] += len(wide_products)

            wide_depth_futures: list[Future[bytes]] = [
                executor.submit(encode_depth_png, product.depth_mm_hw, level=1) for product in wide_products
            ]
            wide_normal_futures: list[Future[bytes]] = [
                executor.submit(encode_rgb_png, product.normal_rgb_hw3) for product in wide_products
            ]
            for product in wide_products:
                splat_depth_m_hw: Float32[ndarray, "h w"] = (
                    product.depth_mm_hw.astype(np.float32) / 1000.0
                ).astype(np.float32, copy=False)
                wide_depth_metrics.add(product.camera.timestamp_ns, splat_depth_m_hw)
            _send_depth_batch(
                depth_recording,
                DEPTH_SPLAT_WIDE,
                wide_products,
                _future_results(wide_depth_futures),
            )
            _send_image_batch(
                triage_recording,
                NORMALS_SPLAT_WIDE,
                wide_products,
                _future_results(wide_normal_futures),
                "image/png",
            )
            rendered_count += len(wide_products)
            elapsed: float = time.perf_counter() - started_at
            print(
                f"published {rendered_count}/{len(cameras)} full-grid frames "
                f"({rendered_count / elapsed:.2f} end-to-end frames/s)",
                flush=True,
            )

        for batch_start in range(0, len(ultrawide_cameras), batch_size):
            ultrawide_batch: list[RenderCamera] = ultrawide_cameras[batch_start : batch_start + batch_size]
            render_started_at = time.perf_counter()
            ultrawide_products: list[UltrawideQuantizedRender] = [
                _quantized_ultrawide_render(splats, camera, background_3) for camera in ultrawide_batch
            ]
            render_wall_seconds += time.perf_counter() - render_started_at
            d2h_transfers["depth"] += len(ultrawide_products)
            d2h_transfers["normal"] += len(ultrawide_products)
            d2h_transfers["rgb"] += len(ultrawide_products)

            ultrawide_depth_futures: list[Future[bytes]] = [
                executor.submit(encode_depth_png, product.depth_mm_hw, level=1) for product in ultrawide_products
            ]
            ultrawide_normal_futures: list[Future[bytes]] = [
                executor.submit(encode_rgb_png, product.normal_rgb_hw3) for product in ultrawide_products
            ]
            rgb_futures: list[Future[bytes]] = [
                executor.submit(encode_rgb_jpeg, product.rgb_hw3) for product in ultrawide_products
            ]
            rgb_error_futures: list[Future[bytes]] = []
            depth_error_futures: list[Future[bytes]] = []
            normal_error_futures: list[Future[bytes]] = []
            diagnostic_products: list[UltrawideQuantizedRender] = []
            for product in ultrawide_products:
                splat_depth_m_hw = (product.depth_mm_hw.astype(np.float32) / 1000.0).astype(np.float32, copy=False)
                ultrawide_depth_metrics.add(product.camera.timestamp_ns, splat_depth_m_hw)
                reference: TriageReferenceBlobs | None = triage_reference_by_timestamp.get(product.camera.timestamp_ns)
                if reference is None:
                    continue
                splat_normal_hw3: Float32[ndarray, "h w 3"] = decode_normal_codes(product.normal_rgb_hw3)
                source_rgb_hw3: UInt8[ndarray, "h w 3"] = decode_rgb_image(reference.rgb_blob)
                mesh_depth_m_hw: Float32[ndarray, "h w"] = decode_splat_depth_png(reference.depth_blob)
                moge_normal_hw3: Float32[ndarray, "h w 3"] = decode_normals_png(reference.normal_blob)
                errors: TriageErrors = compute_triage_errors(
                    source_rgb_hw3,
                    product.rgb_hw3,
                    mesh_depth_m_hw,
                    splat_depth_m_hw,
                    moge_normal_hw3,
                    splat_normal_hw3,
                )
                diagnostic_products.append(product)
                rgb_error_futures.append(executor.submit(encode_rgb_png, errors.rgb_map_hw3))
                depth_error_futures.append(executor.submit(encode_rgb_png, errors.depth_map_hw3))
                normal_error_futures.append(executor.submit(encode_rgb_png, errors.normal_map_hw3))
                rgb_means.append(errors.rgb_mean)
                depth_means_m.append(errors.depth_mean_m)
                normal_means_degrees.append(errors.normal_mean_degrees)
            _send_depth_batch(
                depth_recording,
                DEPTH_SPLAT_ULTRAWIDE,
                ultrawide_products,
                _future_results(ultrawide_depth_futures),
            )
            _send_image_batch(
                triage_recording,
                RGB_SPLAT_ULTRAWIDE_RECT,
                ultrawide_products,
                _future_results(rgb_futures),
                "image/jpeg",
            )
            _send_image_batch(
                triage_recording,
                NORMALS_SPLAT_ULTRAWIDE_RECT,
                ultrawide_products,
                _future_results(ultrawide_normal_futures),
                "image/png",
            )
            if diagnostic_products:
                _send_image_batch(
                    triage_recording,
                    ERROR_RGB_ULTRAWIDE_RECT,
                    diagnostic_products,
                    _future_results(rgb_error_futures),
                    "image/png",
                )
                _send_image_batch(
                    triage_recording,
                    ERROR_DEPTH_ULTRAWIDE_RECT,
                    diagnostic_products,
                    _future_results(depth_error_futures),
                    "image/png",
                )
                _send_image_batch(
                    triage_recording,
                    ERROR_NORMAL_ULTRAWIDE_RECT,
                    diagnostic_products,
                    _future_results(normal_error_futures),
                    "image/png",
                )
                diagnostic_frames += len(diagnostic_products)
            rendered_count += len(ultrawide_products)
            elapsed = time.perf_counter() - started_at
            print(
                f"published {rendered_count}/{len(cameras)} full-grid frames "
                f"({rendered_count / elapsed:.2f} end-to-end frames/s)",
                flush=True,
            )

    wide_summary: CameraSummary = wide_depth_metrics.summary()
    ultrawide_summary: CameraSummary = ultrawide_depth_metrics.summary()
    if diagnostic_frames != len(triage_references):
        raise RuntimeError(f"published {diagnostic_frames} triage frames for {len(triage_references)} references")
    triage_stats = SplatTriageWriteStats(
        wide_normal_frames=len(wide_cameras),
        ultrawide=TriageWriteStats(
            product_frames=len(ultrawide_cameras),
            diagnostic_frames=diagnostic_frames,
            rgb=summarize_metrics(rgb_means),
            depth=summarize_metrics(depth_means_m),
            normal=summarize_metrics(normal_means_degrees),
        ),
    )
    dataset: DatasetEntry = reader.dataset
    for rrd_path, layer_name in (
        (splat_rrd_path, SPLAT_LAYER),
        (depth_rrd_path, SPLAT_DEPTH_LAYER),
        (triage_rrd_path, SPLAT_TRIAGE_LAYER),
    ):
        register_layer(dataset, rrd_path, layer_name)
        print(f"registered layer: {layer_name} (REPLACE)", flush=True)
    wall_seconds: float = time.perf_counter() - started_at
    return PublishStats(
        wall_seconds=wall_seconds,
        render_wall_seconds=render_wall_seconds,
        frame_count=len(cameras),
        frames_per_second=len(cameras) / render_wall_seconds,
        gaussian_count=rerun_splats.count,
        d2h_transfers=d2h_transfers,
        wide_depth=wide_summary,
        ultrawide_depth=ultrawide_summary,
        triage=triage_stats,
        rrd_paths={
            SPLAT_LAYER: str(splat_rrd_path),
            SPLAT_DEPTH_LAYER: str(depth_rrd_path),
            SPLAT_TRIAGE_LAYER: str(triage_rrd_path),
        },
    )
