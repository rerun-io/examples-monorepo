"""Typed CPU and CUDA builders for catalog training samples."""

from time import perf_counter
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from arkitscenes_download.ingest.depth import ArkitDepthConfidence, decode_depth_png, decode_depth_png_fast, inflate_depth_png_rows
from jaxtyping import Float32, Int32, UInt8, UInt16
from numpy import ndarray
from torch import Tensor

from zipdepth.catalog.png import unfilter_up_cuda
from zipdepth.catalog.stats import BuilderStats
from zipdepth.catalog.targets import (
    AugmentPolicy,
    TargetMode,
    build_metric_training_sample,
    build_training_sample,
    build_training_sample_cuda,
    depth_span_ratio,
    depth_span_ratio_cuda,
)
from zipdepth.data.transforms import AlbumentationsWrapper


@runtime_checkable
class SampleBuilder(Protocol):
    """Build one collatable training sample from aligned catalog inputs."""

    stats: BuilderStats
    """Counters owned by this builder and its producer thread."""

    def __call__(
        self,
        frame_chw: UInt8[Tensor, "3 h w"],
        target_blob_bytes: UInt8[ndarray, "target_n"],
        prompt_blob_bytes: UInt8[ndarray, "prompt_n"],
        confidence: UInt8[ndarray, "confidence_n"] | None,
        quarter_turns: int,
        sample_seed: int,
    ) -> dict[str, Tensor] | None:
        """Build a sample, or return None when the flat-depth filter rejects it."""


class CpuSampleBuilder:
    """Decode depth and build samples on the CPU."""

    def __init__(self, transform: AlbumentationsWrapper, min_depth_span: float, target_mode: TargetMode = "ssi") -> None:
        """Configure deterministic transform and optional flat-depth filtering."""
        if min_depth_span < 0.0:
            raise ValueError("min_depth_span must be non-negative")
        if target_mode not in ("ssi", "metric"):
            raise ValueError(f"unknown target mode {target_mode!r}")
        self._transform: AlbumentationsWrapper = transform
        self._min_depth_span: float = min_depth_span
        self._target_mode: TargetMode = target_mode
        self.stats: BuilderStats = BuilderStats()
        """Counters local to this builder."""

    def __call__(
        self,
        frame_chw: UInt8[Tensor, "3 h w"],
        target_blob_bytes: UInt8[ndarray, "target_n"],
        prompt_blob_bytes: UInt8[ndarray, "prompt_n"],
        confidence: UInt8[ndarray, "confidence_n"] | None,
        quarter_turns: int,
        sample_seed: int,
    ) -> dict[str, Tensor] | None:
        """Decode, orient, filter, and transform one CPU sample."""
        del sample_seed
        started: float = perf_counter()
        depth_mm_hw: UInt16[ndarray, "h w"] | None = decode_depth_png_fast(target_blob_bytes)
        if depth_mm_hw is None:
            depth_mm_hw = decode_depth_png(target_blob_bytes)
            self.stats.png_fallbacks += 1
        prompt_depth_mm_hw: UInt16[ndarray, "prompt_h prompt_w"] | None = decode_depth_png_fast(prompt_blob_bytes)
        if prompt_depth_mm_hw is None:
            prompt_depth_mm_hw = decode_depth_png(prompt_blob_bytes)
            self.stats.png_fallbacks += 1
        if confidence is None:
            # Confidence was not fetched: trust every prompt pixel and let the model's
            # own 0.1-4 m range gate do the filtering.
            prompt_confidence_hw: UInt8[ndarray, "prompt_h prompt_w"] = np.full(
                prompt_depth_mm_hw.shape, ArkitDepthConfidence.HIGH, dtype=np.uint8
            )
        else:
            if confidence.size != prompt_depth_mm_hw.size:
                raise ValueError(f"prompt confidence has {confidence.size} values for prompt shape {prompt_depth_mm_hw.shape}")
            prompt_confidence_hw = confidence.reshape(prompt_depth_mm_hw.shape)
        if self._min_depth_span > 0.0 and depth_span_ratio(depth_mm_hw) < self._min_depth_span:
            self.stats.blob_decode += perf_counter() - started
            self.stats.skipped_flat_frames += 1
            return None
        self.stats.blob_decode += perf_counter() - started

        started = perf_counter()
        rgb_chw: UInt8[ndarray, "3 h w"] = frame_chw.cpu().numpy()
        rgb_hwc: UInt8[ndarray, "h w 3"] = np.moveaxis(rgb_chw, 0, -1)
        rgb_landscape_hwc: UInt8[ndarray, "landscape_h landscape_w 3"] = np.ascontiguousarray(np.rot90(rgb_hwc, quarter_turns))
        depth_landscape_mm_hw: UInt16[ndarray, "landscape_h landscape_w"] = np.ascontiguousarray(np.rot90(depth_mm_hw, quarter_turns))
        prompt_landscape_mm_hw: UInt16[ndarray, "192 256"] = np.ascontiguousarray(np.rot90(prompt_depth_mm_hw, quarter_turns))
        confidence_landscape_hw: UInt8[ndarray, "192 256"] = np.ascontiguousarray(np.rot90(prompt_confidence_hw, quarter_turns))
        if self._target_mode == "metric":
            sample: dict[str, Tensor] = build_metric_training_sample(
                rgb_landscape_hwc,
                depth_landscape_mm_hw,
                prompt_landscape_mm_hw,
                confidence_landscape_hw,
                self._transform,
            )
        else:
            sample = build_training_sample(
                rgb_landscape_hwc,
                depth_landscape_mm_hw,
                self._transform,
                prompt_depth_mm_hw=prompt_landscape_mm_hw,
                prompt_confidence_hw=confidence_landscape_hw,
            )
        self.stats.augment += perf_counter() - started
        self.stats.samples_built += 1
        return sample


class CudaSampleBuilder:
    """Inflate depth and build samples on one producer-owned CUDA stream."""

    def __init__(
        self,
        out_hw: tuple[int, int],
        policy: AugmentPolicy,
        min_depth_span: float,
        device: torch.device,
        target_mode: TargetMode = "ssi",
    ) -> None:
        """Configure output shape, augmentation, filtering, and CUDA state."""
        if out_hw[0] <= 0 or out_hw[1] <= 0:
            raise ValueError("output height and width must be positive")
        if min_depth_span < 0.0:
            raise ValueError("min_depth_span must be non-negative")
        if device.type != "cuda":
            raise ValueError("CudaSampleBuilder requires a CUDA device")
        if target_mode not in ("ssi", "metric"):
            raise ValueError(f"unknown target mode {target_mode!r}")
        self._out_hw: tuple[int, int] = out_hw
        self._policy: AugmentPolicy = policy
        self._min_depth_span: float = min_depth_span
        self._device: torch.device = device
        self._target_mode: TargetMode = target_mode
        self._generator: torch.Generator = torch.Generator()
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=device)
        self.stats: BuilderStats = BuilderStats()
        """Counters local to this builder."""

    def __call__(
        self,
        frame_chw: UInt8[Tensor, "3 h w"],
        target_blob_bytes: UInt8[ndarray, "target_n"],
        prompt_blob_bytes: UInt8[ndarray, "prompt_n"],
        confidence: UInt8[ndarray, "confidence_n"] | None,
        quarter_turns: int,
        sample_seed: int,
    ) -> dict[str, Tensor] | None:
        """Inflate, unfilter, filter, and augment one CUDA sample."""
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        frame_chw.record_stream(self._stream)
        started: float = perf_counter()
        with torch.cuda.stream(self._stream):
            inflated: tuple[UInt8[ndarray, "h row_bytes"], tuple[int, int]] | None = inflate_depth_png_rows(target_blob_bytes)
            if inflated is None:
                depth_mm_hw: UInt16[ndarray, "h w"] = decode_depth_png(target_blob_bytes)
                depth_mm_cuda_hw: Int32[Tensor, "h w"] = torch.from_numpy(depth_mm_hw).to(
                    device=self._device,
                    dtype=torch.int32,
                    non_blocking=True,
                )
                self.stats.png_fallbacks += 1
            else:
                filtered_hwb: UInt8[ndarray, "h row_bytes"] = inflated[0]
                filtered_cuda_hwb: UInt8[Tensor, "h row_bytes"] = torch.from_numpy(filtered_hwb).to(
                    device=self._device,
                    non_blocking=True,
                )
                depth_mm_cuda_hw = unfilter_up_cuda(filtered_cuda_hwb)
            prompt_depth_mm_hw: UInt16[ndarray, "prompt_h prompt_w"] | None = decode_depth_png_fast(prompt_blob_bytes)
            if prompt_depth_mm_hw is None:
                prompt_depth_mm_hw = decode_depth_png(prompt_blob_bytes)
                self.stats.png_fallbacks += 1
            if confidence is None:
                # Confidence was not fetched: trust every prompt pixel and let the
                # model's own 0.1-4 m range gate do the filtering.
                confidence = np.full(prompt_depth_mm_hw.size, ArkitDepthConfidence.HIGH, dtype=np.uint8)
            if confidence.size != prompt_depth_mm_hw.size:
                raise ValueError(f"prompt confidence has {confidence.size} values for prompt shape {prompt_depth_mm_hw.shape}")
            prompt_depth_cuda_hw: Int32[Tensor, "prompt_h prompt_w"] = torch.from_numpy(prompt_depth_mm_hw).to(
                device=self._device,
                dtype=torch.int32,
                non_blocking=True,
            )
            prompt_confidence_cuda_hw: UInt8[Tensor, "prompt_h prompt_w"] = torch.from_numpy(
                confidence.reshape(prompt_depth_mm_hw.shape)
            ).to(device=self._device, non_blocking=True)
            span_ratio: Float32[Tensor, ""] | None = (
                depth_span_ratio_cuda(depth_mm_cuda_hw) if self._min_depth_span > 0.0 else None
            )
        self.stats.blob_decode += perf_counter() - started

        self._generator.manual_seed(sample_seed)
        started = perf_counter()
        with torch.cuda.stream(self._stream):
            sample: dict[str, Tensor] = build_training_sample_cuda(
                frame_chw,
                depth_mm_cuda_hw,
                quarter_turns,
                self._out_hw,
                self._generator,
                self._policy,
                prompt_depth_mm_hw=prompt_depth_cuda_hw,
                prompt_confidence_hw=prompt_confidence_cuda_hw,
                target_mode=self._target_mode,
            )
            self._stream.synchronize()
        self.stats.augment += perf_counter() - started
        if span_ratio is not None and float(span_ratio.item()) < self._min_depth_span:
            self.stats.skipped_flat_frames += 1
            return None
        self.stats.samples_built += 1
        return sample
