"""Typed CPU and CUDA builders for catalog training samples."""

from time import perf_counter
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from arkitscenes_download.ingest.depth import ArkitDepthConfidence, decode_depth_png, decode_depth_png_fast, inflate_depth_png_rows
from jaxtyping import Bool, Float32, Int32, UInt8, UInt16
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
from zipdepth.catalog.ultrawide import (
    Camera,
    PromptPlacement,
    UltrawidePolicy,
    erode_valid,
    prompt_placement,
    valid_fraction,
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
        *,
        camera: Camera = "wide",
    ) -> dict[str, Tensor] | None:
        """Build a sample, or return None when a frame filter rejects it."""


def _resolve_placement(camera: Camera, policy: UltrawidePolicy | None) -> PromptPlacement | None:
    """Return the prompt placement one camera needs, or None for the full-canvas wide prompt.

    Raises:
        ValueError: If an ultrawide sample is requested from a builder that was
            not configured with an ultrawide policy.
    """
    if camera == "wide":
        return None
    if policy is None:
        raise ValueError("this builder was not configured with an ultrawide policy")
    return prompt_placement(policy.prompt_scale)


def _require_landscape_after_rotation(frame_chw: UInt8[Tensor, "3 h w"], quarter_turns: int) -> None:
    """Reject an ultrawide frame that the orientation turns do not bring to landscape.

    The ``ultrawide_depth`` layer stores 640x480 for landscape captures and
    480x640 for portrait ones and carries no orientation guard of its own, so a
    segment whose stored orientation disagrees with its capture property would
    otherwise train silently on a rotated image.

    Args:
        frame_chw: Decoded rectified frame with shape ``(3, H, W)``.
        quarter_turns: Counter-clockwise turns applied before resizing.

    Raises:
        ValueError: If the frame is not landscape after the turns.
    """
    height: int = int(frame_chw.shape[-2])
    width: int = int(frame_chw.shape[-1])
    if quarter_turns % 2 == 1:
        height, width = width, height
    if height >= width:
        raise ValueError(
            f"ultrawide frame {tuple(frame_chw.shape[-2:])} is {height}x{width} after {quarter_turns} quarter turns, which is not landscape"
        )


def _apply_ultrawide_mask_policy(sample: dict[str, Tensor], policy: UltrawidePolicy, stats: BuilderStats) -> dict[str, Tensor] | None:
    """Drop a sparse ultrawide frame, then erode the surviving target mask.

    The valid fraction is measured on the resized target before erosion, so the
    threshold describes the supervision the loss would actually see and does not
    move when the erosion radius changes.

    Args:
        sample: Built sample carrying ``target_valid`` (metric) or ``mask`` (SSI).
        policy: Minimum valid fraction and erosion radius.
        stats: Builder counters incremented when the frame is rejected.

    Returns:
        The sample with an eroded mask, or None when the frame is too sparse.
    """
    valid_key: str = "target_valid" if "target_valid" in sample else "mask"
    valid_chw: Bool[Tensor, "1 h w"] = sample[valid_key]
    if valid_fraction(valid_chw) < policy.min_valid_fraction:
        stats.skipped_low_valid_frames += 1
        return None
    sample[valid_key] = erode_valid(valid_chw, policy.valid_erosion_px)
    return sample


class CpuSampleBuilder:
    """Decode depth and build samples on the CPU."""

    def __init__(
        self,
        transform: AlbumentationsWrapper,
        min_depth_span: float,
        target_mode: TargetMode = "ssi",
        ultrawide_policy: UltrawidePolicy | None = None,
    ) -> None:
        """Configure deterministic transform, flat-depth filtering, and ultrawide policy."""
        if min_depth_span < 0.0:
            raise ValueError("min_depth_span must be non-negative")
        if target_mode not in ("ssi", "metric"):
            raise ValueError(f"unknown target mode {target_mode!r}")
        self._transform: AlbumentationsWrapper = transform
        self._min_depth_span: float = min_depth_span
        self._target_mode: TargetMode = target_mode
        self._ultrawide_policy: UltrawidePolicy | None = ultrawide_policy
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
        *,
        camera: Camera = "wide",
    ) -> dict[str, Tensor] | None:
        """Decode, orient, filter, and transform one CPU sample."""
        del sample_seed
        placement: PromptPlacement | None = _resolve_placement(camera, self._ultrawide_policy)
        if camera == "ultrawide":
            _require_landscape_after_rotation(frame_chw, quarter_turns)
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
                prompt_placement=placement,
            )
        else:
            sample = build_training_sample(
                rgb_landscape_hwc,
                depth_landscape_mm_hw,
                self._transform,
                prompt_depth_mm_hw=prompt_landscape_mm_hw,
                prompt_confidence_hw=confidence_landscape_hw,
                prompt_placement=placement,
            )
        self.stats.augment += perf_counter() - started
        if camera == "ultrawide" and self._ultrawide_policy is not None:
            accepted: dict[str, Tensor] | None = _apply_ultrawide_mask_policy(sample, self._ultrawide_policy, self.stats)
            if accepted is None:
                return None
            sample = accepted
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
        ultrawide_policy: UltrawidePolicy | None = None,
    ) -> None:
        """Configure output shape, augmentation, filtering, ultrawide policy, and CUDA state."""
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
        self._ultrawide_policy: UltrawidePolicy | None = ultrawide_policy
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
        *,
        camera: Camera = "wide",
    ) -> dict[str, Tensor] | None:
        """Inflate, unfilter, filter, and augment one CUDA sample."""
        placement: PromptPlacement | None = _resolve_placement(camera, self._ultrawide_policy)
        if camera == "ultrawide":
            _require_landscape_after_rotation(frame_chw, quarter_turns)
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
                prompt_placement=placement,
            )
            self._stream.synchronize()
        self.stats.augment += perf_counter() - started
        if span_ratio is not None and float(span_ratio.item()) < self._min_depth_span:
            self.stats.skipped_flat_frames += 1
            return None
        if camera == "ultrawide" and self._ultrawide_policy is not None:
            accepted: dict[str, Tensor] | None = _apply_ultrawide_mask_policy(sample, self._ultrawide_policy, self.stats)
            if accepted is None:
                return None
            sample = accepted
        self.stats.samples_built += 1
        return sample

    def build_decoded_batch(
        self,
        frames: list[UInt8[Tensor, "3 h w"]],
        targets_mm: list[Int32[Tensor, "h w"]],
        prompts_mm: list[Int32[Tensor, "prompt_h prompt_w"]],
        confidences: list[UInt8[Tensor, "confidence_n"] | None],
        quarter_turns: list[int],
        sample_seeds: list[int],
    ) -> list[dict[str, Tensor] | None]:
        """Build one decoded batch on the builder stream with one synchronization.

        Args:
            frames: CUDA uint8 RGB frames with shape ``(3, H, W)``.
            targets_mm: CUDA int32 dense target depths aligned with the frames.
            prompts_mm: CUDA int32 low-resolution prompt depths.
            confidences: Flattened uint8 prompt confidence, or None for all trusted.
            quarter_turns: Counter-clockwise orientation correction per sample.
            sample_seeds: Deterministic augmentation seed per sample.

        Returns:
            One existing-contract training sample per accepted input and None for
            each target rejected by the depth-span filter.

        Raises:
            ValueError: If the input lists differ in length or prompt confidence
                does not match its prompt depth.
        """
        batch_size: int = len(frames)
        if not all(
            len(values) == batch_size
            for values in (targets_mm, prompts_mm, confidences, quarter_turns, sample_seeds)
        ):
            raise ValueError("decoded sample batch inputs must have equal lengths")
        if batch_size == 0:
            return []

        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        frame: UInt8[Tensor, "3 h w"]
        target_mm: Int32[Tensor, "h w"]
        prompt_mm: Int32[Tensor, "prompt_h prompt_w"]
        for frame, target_mm, prompt_mm in zip(frames, targets_mm, prompts_mm, strict=True):
            frame.record_stream(self._stream)
            target_mm.record_stream(self._stream)
            prompt_mm.record_stream(self._stream)

        started: float = perf_counter()
        samples: list[dict[str, Tensor]] = []
        span_ratios: list[Float32[Tensor, ""] | None] = []
        with torch.cuda.stream(self._stream):
            confidence: UInt8[Tensor, "confidence_n"] | None
            turns: int
            sample_seed: int
            for frame, target_mm, prompt_mm, confidence, turns, sample_seed in zip(
                frames,
                targets_mm,
                prompts_mm,
                confidences,
                quarter_turns,
                sample_seeds,
                strict=True,
            ):
                if confidence is None:
                    prompt_confidence: UInt8[Tensor, "prompt_h prompt_w"] = torch.full(
                        prompt_mm.shape,
                        ArkitDepthConfidence.HIGH,
                        dtype=torch.uint8,
                        device=self._device,
                    )
                else:
                    if confidence.numel() != prompt_mm.numel():
                        raise ValueError(
                            f"prompt confidence has {confidence.numel()} values for prompt shape {tuple(prompt_mm.shape)}"
                        )
                    prompt_confidence = confidence.reshape(prompt_mm.shape).to(
                        device=self._device,
                        dtype=torch.uint8,
                        non_blocking=True,
                    )
                span_ratios.append(
                    depth_span_ratio_cuda(target_mm) if self._min_depth_span > 0.0 else None
                )
                self._generator.manual_seed(sample_seed)
                samples.append(
                    build_training_sample_cuda(
                        frame,
                        target_mm,
                        turns,
                        self._out_hw,
                        self._generator,
                        self._policy,
                        prompt_depth_mm_hw=prompt_mm,
                        prompt_confidence_hw=prompt_confidence,
                        target_mode=self._target_mode,
                    )
                )
        self._stream.synchronize()
        self.stats.augment += perf_counter() - started

        accepted: list[dict[str, Tensor] | None] = []
        sample: dict[str, Tensor]
        span_ratio: Float32[Tensor, ""] | None
        for sample, span_ratio in zip(samples, span_ratios, strict=True):
            if span_ratio is not None and float(span_ratio.item()) < self._min_depth_span:
                self.stats.skipped_flat_frames += 1
                accepted.append(None)
            else:
                self.stats.samples_built += 1
                accepted.append(sample)
        return accepted
