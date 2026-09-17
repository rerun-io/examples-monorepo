"""Evaluate metric ZipDepth-PromptDA and explicit reference diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, UInt8
from monopriors.models.depth_completion.base_completion_depth import MAX_METRIC_DEPTH_M, MIN_METRIC_DEPTH_M
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor, download_zipdepth_checkpoint
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader

from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, PromptDACatalog, load_promptda_catalog, split_holdout_segments
from zipdepth.catalog.targets import build_eval_transform
from zipdepth.catalog.ultrawide import (
    DEFAULT_ULTRAWIDE_PROMPT_SCALE,
    ULTRAWIDE_LAYER,
    CameraSelection,
    PromptPlacement,
    UltrawidePolicy,
    footprint_box,
    prompt_placement,
)
from zipdepth.data.transforms import AlbumentationsWrapper
from zipdepth.evaluation.edge_metrics import EdgeStratifiedResult, edge_stratified_mae
from zipdepth.evaluation.metrics import abs_relative_difference, align_depth_least_square, delta1_acc, disparity2depth, fit_affine_least_squares


def _mean_by_key(records: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every metric key over per-frame records."""
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


@dataclass(slots=True)
class EvalCatalogConfig:
    """Held-out catalog evaluation configuration."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset containing ARKitScenes and PromptDA layers."""
    save_dir: Path = Path("data/checkpoints/catalog")
    """Training output directory containing ``holdout_segments.json`` and receiving results."""
    segment_ids: list[str] | None = None
    """Explicit evaluation segment identifiers; overrides the saved holdout manifest."""
    checkpoint: Path | None = None
    """Checkpoint to evaluate; None uses released ZipDepth-base weights."""
    evaluation: Literal["metric", "affine-reference", "aligned-relative", "prompt-upsample"] = "metric"
    """Direct prompted model, closed-form untrained reference, legacy aligned diagnostic, or the
    zero-parameter bilinear prompt upsample -- the floor any trained student must beat."""
    height: int = 768
    """Deterministic evaluation image height."""
    width: int = 1024
    """Deterministic evaluation image width."""
    frame_stride: int = 10
    """Evaluate every Nth chosen PromptDA frame within each segment."""
    input_size: int = 768
    """Shorter image side the network runs at (``ZipDepthPredictor`` resizes to it, then upsamples the disparity).
    384 is the released model's training size; catalog fine-tunes train the network directly at ``height`` x ``width``,
    so score those with ``input_size=height`` to avoid a train/eval resolution mismatch."""
    max_segments: int | None = None
    """Optional limit on the selected evaluation segments."""
    holdout_count: int = 20
    """Deterministic fallback holdout size when no saved manifest exists."""
    holdout_seed: int = 0
    """Deterministic fallback holdout seed."""
    cameras: CameraSelection = "wide"
    """Cameras scored. Selecting the ultrawide adds a second, separately reported pass
    over the same segments; the wide numbers never change. Ultrawide scoring supports
    the ``metric`` and ``prompt-upsample`` evaluations only."""
    ultrawide_min_valid_fraction: float = 0.0
    """Ultrawide frames below this in-range target fraction are skipped. Zero scores
    every frame: dropping sparse frames is a training data-efficiency choice, not a
    property of the benchmark."""
    ultrawide_valid_erosion_px: int = 1
    """Ultrawide target-mask erosion radius; matches the training lane."""
    ultrawide_prompt_scale: float = DEFAULT_ULTRAWIDE_PROMPT_SCALE
    """Wide prompt footprint inside an ultrawide frame, as a fraction per axis."""


@dataclass(slots=True, frozen=True)
class MetricCatalogDepthMetrics:
    """Unaligned or explicitly diagnostic metric-depth scores."""

    abs_rel: float
    """Mean absolute relative depth error over valid pixels."""
    delta1: float
    """Fraction of valid depth ratios below 1.25."""
    mae: float
    """Mean absolute error in metres."""


def _mean_metrics(records: list[MetricCatalogDepthMetrics]) -> MetricCatalogDepthMetrics:
    """Average one non-empty list of per-frame metric records."""
    return MetricCatalogDepthMetrics(
        abs_rel=float(np.mean([record.abs_rel for record in records])),
        delta1=float(np.mean([record.delta1 for record in records])),
        mae=float(np.mean([record.mae for record in records])),
    )


def prompt_upsample_depth(
    prompt_depth_hw: Float32[ndarray, "192 256"],
    prompt_valid_hw: Bool[ndarray, "192 256"],
    *,
    height: int,
    width: int,
) -> Float32[ndarray, "h w"]:
    """Bilinearly upsample the LiDAR prompt to full resolution, ignoring holes.

    The zero-parameter control every trained student must beat. Invalid prompt
    pixels are excluded rather than treated as zero depth: depth and validity are
    resized together and divided, which is a normalized-convolution fill. Pixels
    with no valid support anywhere nearby take the frame's median valid depth, so
    the prediction is dense and is scored on the same pixels as a model's.

    Args:
        prompt_depth_hw: Prompt depth in metres with shape ``(192, 256)``.
        prompt_valid_hw: Prompt validity with shape ``(192, 256)``.
        height: Output height.
        width: Output width.

    Returns:
        Dense float32 depth in metres with shape ``(height, width)``.
    """
    valid_weight_hw: Float32[ndarray, "192 256"] = prompt_valid_hw.astype(np.float32)
    weighted_hw: Float32[ndarray, "192 256"] = (prompt_depth_hw * valid_weight_hw).astype(np.float32)
    weighted_up_hw: Float32[ndarray, "h w"] = cv2.resize(weighted_hw, (width, height), interpolation=cv2.INTER_LINEAR)
    weight_up_hw: Float32[ndarray, "h w"] = cv2.resize(valid_weight_hw, (width, height), interpolation=cv2.INTER_LINEAR)
    supported_hw: Bool[ndarray, "h w"] = weight_up_hw > 1.0e-3
    fallback: float = float(np.median(prompt_depth_hw[prompt_valid_hw])) if bool(prompt_valid_hw.any()) else 1.0
    upsampled_hw: Float32[ndarray, "h w"] = np.full((height, width), fallback, dtype=np.float32)
    upsampled_hw[supported_hw] = weighted_up_hw[supported_hw] / weight_up_hw[supported_hw]
    return upsampled_hw


def score_metric_depth(
    prediction_depth_hw: Float[ndarray, "h w"],
    target_depth_hw: Float[ndarray, "h w"],
    valid_hw: Bool[ndarray, "h w"],
) -> MetricCatalogDepthMetrics:
    """Score metric depth directly, without any scale or shift alignment."""
    if prediction_depth_hw.shape != target_depth_hw.shape or valid_hw.shape != target_depth_hw.shape:
        raise ValueError("prediction, target, and validity mask must have equal shapes")
    metric_valid_hw: Bool[ndarray, "h w"] = (
        valid_hw
        & np.isfinite(prediction_depth_hw)
        & np.isfinite(target_depth_hw)
        & (prediction_depth_hw > 0.0)
        & (target_depth_hw >= MIN_METRIC_DEPTH_M)
        & (target_depth_hw <= MAX_METRIC_DEPTH_M)
    )
    if int(np.count_nonzero(metric_valid_hw)) < 10:
        raise ValueError("at least ten valid metric-depth pixels are required")
    return MetricCatalogDepthMetrics(
        abs_rel=abs_relative_difference(prediction_depth_hw, target_depth_hw, metric_valid_hw),
        delta1=delta1_acc(prediction_depth_hw, target_depth_hw, metric_valid_hw),
        mae=float(np.mean(np.abs(prediction_depth_hw[metric_valid_hw] - target_depth_hw[metric_valid_hw]))),
    )


def affine_reference_depth(
    relative_inverse_depth_hw: Float32[ndarray, "h w"],
    prompt_depth_hw: Float32[ndarray, "192 256"],
    prompt_valid_hw: Bool[ndarray, "192 256"],
) -> Float32[ndarray, "h w"]:
    """Fit released relative inverse depth to valid LiDAR and return metres.

    This is the untrained reference only. It is not the prompted model.
    """
    if prompt_depth_hw.shape != (192, 256) or prompt_valid_hw.shape != (192, 256):
        raise ValueError("prompt depth and validity must both have shape 192x256")
    relative_prompt_hw: Float32[ndarray, "192 256"] = cv2.resize(
        relative_inverse_depth_hw,
        (256, 192),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    fit_hw: Bool[ndarray, "192 256"] = (
        prompt_valid_hw
        & np.isfinite(relative_prompt_hw)
        & np.isfinite(prompt_depth_hw)
        & (prompt_depth_hw >= MIN_METRIC_DEPTH_M)
        & (prompt_depth_hw <= MAX_METRIC_DEPTH_M)
    )
    if int(np.count_nonzero(fit_hw)) < 10:
        raise ValueError("at least ten valid LiDAR prompt pixels are required")
    x_n: Float[ndarray, "n"] = relative_prompt_hw[fit_hw].astype(np.float64)
    y_n: Float[ndarray, "n"] = (1.0 / prompt_depth_hw[fit_hw]).astype(np.float64)
    x_mean: float = float(np.mean(x_n))
    centered_x_n: Float[ndarray, "n"] = x_n - x_mean
    variance: float = float(np.sum(centered_x_n * centered_x_n))
    if variance <= 1.0e-12:
        raise ValueError("relative inverse depth is constant over the valid LiDAR prompt")
    scale: float
    shift: float
    scale, shift = fit_affine_least_squares(y_n, x_n)
    aligned_inverse_hw: Float32[ndarray, "h w"] = np.clip(
        scale * relative_inverse_depth_hw.astype(np.float64) + shift,
        0.25,
        10.0,
    ).astype(np.float32)
    return np.reciprocal(aligned_inverse_hw).astype(np.float32)


def aligned_inverse_diagnostic(
    prediction_depth_hw: Float[ndarray, "h w"],
    target_depth_hw: Float[ndarray, "h w"],
    valid_hw: Bool[ndarray, "h w"],
) -> MetricCatalogDepthMetrics:
    """Score after affine inverse-depth alignment, explicitly as a diagnostic."""
    fit_hw: Bool[ndarray, "h w"] = (
        valid_hw
        & np.isfinite(prediction_depth_hw)
        & np.isfinite(target_depth_hw)
        & (prediction_depth_hw > 0.0)
        & (target_depth_hw >= MIN_METRIC_DEPTH_M)
        & (target_depth_hw <= MAX_METRIC_DEPTH_M)
    )
    if int(np.count_nonzero(fit_hw)) < 10:
        raise ValueError("at least ten valid metric-depth pixels are required")
    prediction_inverse_hw: Float[ndarray, "h w"] = np.zeros_like(prediction_depth_hw)
    target_inverse_hw: Float[ndarray, "h w"] = np.zeros_like(target_depth_hw)
    prediction_inverse_hw[fit_hw] = 1.0 / prediction_depth_hw[fit_hw]
    target_inverse_hw[fit_hw] = 1.0 / target_depth_hw[fit_hw]
    aligned_inverse_hw: Float[ndarray, "h w"] = align_depth_least_square(target_inverse_hw, prediction_inverse_hw, fit_hw)
    aligned_depth_hw: Float[ndarray, "h w"] = disparity2depth(aligned_inverse_hw)
    return score_metric_depth(aligned_depth_hw, target_depth_hw, fit_hw)


@dataclass(slots=True, frozen=True)
class FootprintSplitMetrics:
    """Ultrawide metric scores split by the wide prompt's footprint.

    ``outside`` is the headline: those pixels have no prompt at all, so they show
    what the model learned to do with an unprompted part of a prompted frame.
    """

    whole: MetricCatalogDepthMetrics
    """Scores over every valid pixel in the frame."""
    inside: MetricCatalogDepthMetrics
    """Scores inside the prompt footprint."""
    outside: MetricCatalogDepthMetrics
    """Scores outside the prompt footprint."""
    inside_prompt_upsample: MetricCatalogDepthMetrics
    """Zero-parameter bilinear prompt-upsample floor inside the footprint."""


def _mean_footprint_metrics(records: list[FootprintSplitMetrics]) -> FootprintSplitMetrics:
    """Average one non-empty list of footprint-split records field by field."""
    return FootprintSplitMetrics(
        whole=_mean_metrics([record.whole for record in records]),
        inside=_mean_metrics([record.inside for record in records]),
        outside=_mean_metrics([record.outside for record in records]),
        inside_prompt_upsample=_mean_metrics([record.inside_prompt_upsample for record in records]),
    )


def _footprint_metrics_report(prefix: str, metrics: FootprintSplitMetrics) -> dict[str, float]:
    """Flatten footprint-split metrics into prefixed report keys."""
    named: dict[str, MetricCatalogDepthMetrics] = {
        "whole": metrics.whole,
        "inside": metrics.inside,
        "outside": metrics.outside,
        "inside_prompt_upsample": metrics.inside_prompt_upsample,
    }
    return {
        f"{prefix}_{region}_{field}": getattr(record, field)
        for region, record in named.items()
        for field in ("abs_rel", "delta1", "mae")
    }


def footprint_mask(placement: PromptPlacement, height: int, width: int) -> Bool[ndarray, "h w"]:
    """Return the output pixels the scaled wide prompt covers.

    Args:
        placement: Prompt block position inside the prompt canvas.
        height: Output image height.
        width: Output image width.

    Returns:
        Boolean mask with shape ``(height, width)`` that is True inside the footprint.
    """
    box: tuple[int, int, int, int] = footprint_box(placement, (height, width))
    inside_hw: Bool[ndarray, "h w"] = np.zeros((height, width), dtype=bool)
    inside_hw[box[0] : box[2], box[1] : box[3]] = True
    return inside_hw


def score_footprint_split(
    prediction_depth_hw: Float32[ndarray, "h w"],
    target_depth_hw: Float32[ndarray, "h w"],
    target_valid_hw: Bool[ndarray, "h w"],
    prompt_depth_hw: Float32[ndarray, "192 256"],
    prompt_valid_hw: Bool[ndarray, "192 256"],
    inside_hw: Bool[ndarray, "h w"],
) -> FootprintSplitMetrics | None:
    """Score one ultrawide frame whole, inside, and outside the prompt footprint.

    Args:
        prediction_depth_hw: Predicted metric depth in metres.
        target_depth_hw: Raycast ultrawide target depth in metres.
        target_valid_hw: Eroded in-range target validity.
        prompt_depth_hw: Padded prompt canvas depth in metres.
        prompt_valid_hw: Padded prompt canvas validity.
        inside_hw: Output pixels the prompt footprint covers.

    Returns:
        The four scored regions, or None when either region lacks enough valid
        pixels to score.
    """
    height: int = target_depth_hw.shape[0]
    width: int = target_depth_hw.shape[1]
    baseline_depth_hw: Float32[ndarray, "h w"] = prompt_upsample_depth(prompt_depth_hw, prompt_valid_hw, height=height, width=width)
    try:
        return FootprintSplitMetrics(
            whole=score_metric_depth(prediction_depth_hw, target_depth_hw, target_valid_hw),
            inside=score_metric_depth(prediction_depth_hw, target_depth_hw, target_valid_hw & inside_hw),
            outside=score_metric_depth(prediction_depth_hw, target_depth_hw, target_valid_hw & ~inside_hw),
            inside_prompt_upsample=score_metric_depth(baseline_depth_hw, target_depth_hw, target_valid_hw & inside_hw),
        )
    except ValueError:
        return None


def _selected_segments(config: EvalCatalogConfig, catalog: PromptDACatalog) -> list[str]:
    """Resolve explicit segments or the saved training holdout manifest."""
    if config.segment_ids is None:
        manifest: Path = config.save_dir / "holdout_segments.json"
        try:
            with manifest.open() as file:
                loaded: object = json.load(file)
        except FileNotFoundError:
            if not 0 < config.holdout_count <= len(catalog.segment_ids):
                raise ValueError(f"holdout_count must be in [1, {len(catalog.segment_ids)}]") from None
            segment_ids: list[str] = split_holdout_segments(catalog.segment_ids, config.holdout_count, config.holdout_seed)[1]
        else:
            if not isinstance(loaded, list) or not all(isinstance(value, str) for value in loaded):
                raise ValueError(f"holdout manifest must contain a JSON string list: {manifest}")
            segment_ids = list(loaded)
    else:
        segment_ids = list(config.segment_ids)
    catalog.require_segments(segment_ids)
    if config.max_segments is not None:
        if config.max_segments <= 0:
            raise ValueError("max_segments must be positive when set")
        segment_ids = segment_ids[: config.max_segments]
    if not segment_ids:
        raise RuntimeError("no evaluation segments selected")
    return segment_ids


def _evaluate_ultrawide(
    config: EvalCatalogConfig,
    catalog: PromptDACatalog,
    segment_ids: list[str],
    prompted_model: ZipDepthPrompt | None,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    """Score the rectified ultrawide camera with the wide prompt padded into its footprint.

    Args:
        config: Evaluation settings, including the ultrawide mask and prompt policy.
        catalog: Connected PromptDA catalog metadata.
        segment_ids: Segments to score; those without the ultrawide layer are skipped.
        prompted_model: Fused prompted model, or None to score the prompt-upsample floor.
        device: Device the model runs on.

    Returns:
        Per-segment reports and the overall footprint-split summary.

    Raises:
        RuntimeError: If no ultrawide frame could be scored.
    """
    # Catalog-only imports stay local so pure metric tests run outside the catalog lane.
    from zipdepth.catalog.builders import CpuSampleBuilder
    from zipdepth.catalog.dataset import CatalogPromptDepthDataset

    transform: AlbumentationsWrapper = build_eval_transform(config.height, config.width)
    policy: UltrawidePolicy = UltrawidePolicy(
        min_valid_fraction=config.ultrawide_min_valid_fraction,
        valid_erosion_px=config.ultrawide_valid_erosion_px,
        prompt_scale=config.ultrawide_prompt_scale,
    )
    placement: PromptPlacement = prompt_placement(policy.prompt_scale)
    inside_hw: Bool[ndarray, "h w"] = footprint_mask(placement, config.height, config.width)
    covered_ids: list[str] = [segment_id for segment_id in segment_ids if ULTRAWIDE_LAYER in catalog.row_by_id[segment_id].layer_names]
    missing_count: int = len(segment_ids) - len(covered_ids)
    if missing_count:
        print(f"ultrawide: skipping {missing_count} segment(s) without the {ULTRAWIDE_LAYER!r} layer")

    per_segment: list[dict[str, float | int | str]] = []
    all_records: list[FootprintSplitMetrics] = []
    segment_id: str
    for segment_id in covered_ids:
        dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
            catalog.dataset_entry,
            [segment_id],
            catalog.row_by_id,
            device=device,
            builder_factory=lambda: CpuSampleBuilder(transform, min_depth_span=0.0, target_mode="metric", ultrawide_policy=policy),
            shuffle_buffer_size=1,
            frame_stride=config.frame_stride,
            num_producers=1,
            prefetch_samples=4,
            cameras="ultrawide",
        )
        loader: DataLoader[dict[str, Tensor]] = DataLoader(dataset, batch_size=1, num_workers=0)
        segment_records: list[FootprintSplitMetrics] = []
        batch: dict[str, Tensor]
        for batch in loader:
            target_depth_hw: Float32[ndarray, "h w"] = batch["target_depth"][0, 0].numpy().astype(np.float32, copy=False)
            target_valid_hw: Bool[ndarray, "h w"] = batch["target_valid"][0, 0].numpy().astype(bool, copy=False)
            prompt_depth_hw: Float32[ndarray, "192 256"] = batch["prompt_depth"][0, 0].numpy().astype(np.float32, copy=False)
            prompt_valid_hw: Bool[ndarray, "192 256"] = batch["prompt_valid"][0, 0].numpy().astype(bool, copy=False)
            if prompted_model is not None:
                image_bchw: UInt8[Tensor, "b 3 h w"] = batch["image"].to(device=device, non_blocking=True)
                prompt_depth_bchw: Float32[Tensor, "b 1 192 256"] = batch["prompt_depth"].to(device=device, non_blocking=True)
                with torch.inference_mode():
                    prediction_depth_hw: Float32[ndarray, "h w"] = prompted_model(image_bchw, prompt_depth_bchw)[0, 0].float().cpu().numpy()
            else:
                prediction_depth_hw = prompt_upsample_depth(
                    prompt_depth_hw, prompt_valid_hw, height=config.height, width=config.width
                )
            scored: FootprintSplitMetrics | None = score_footprint_split(
                prediction_depth_hw, target_depth_hw, target_valid_hw, prompt_depth_hw, prompt_valid_hw, inside_hw
            )
            if scored is not None:
                segment_records.append(scored)
        if not segment_records:
            print(f"{segment_id}: no scorable ultrawide frames")
            continue
        segment_metrics: FootprintSplitMetrics = _mean_footprint_metrics(segment_records)
        segment_report: dict[str, float | int | str] = {
            "segment_id": segment_id,
            "frame_count": len(segment_records),
            **_footprint_metrics_report("ultrawide", segment_metrics),
        }
        per_segment.append(segment_report)
        all_records.extend(segment_records)
        print(
            f"{segment_id}: ultrawide AbsRel whole={segment_metrics.whole.abs_rel:.6f} "
            f"inside={segment_metrics.inside.abs_rel:.6f} outside={segment_metrics.outside.abs_rel:.6f} "
            f"(inside floor={segment_metrics.inside_prompt_upsample.abs_rel:.6f}) frames={len(segment_records)}"
        )

    if not all_records:
        raise RuntimeError("ultrawide evaluation yielded no valid frames")
    overall_metrics: FootprintSplitMetrics = _mean_footprint_metrics(all_records)
    overall: dict[str, float | int] = {
        "ultrawide_frame_count": len(all_records),
        **_footprint_metrics_report("ultrawide", overall_metrics),
    }
    print(
        f"overall ultrawide: whole AbsRel={overall_metrics.whole.abs_rel:.6f} MAE={overall_metrics.whole.mae:.6f}m | "
        f"inside AbsRel={overall_metrics.inside.abs_rel:.6f} MAE={overall_metrics.inside.mae:.6f}m "
        f"(prompt-upsample floor AbsRel={overall_metrics.inside_prompt_upsample.abs_rel:.6f} "
        f"MAE={overall_metrics.inside_prompt_upsample.mae:.6f}m) | "
        f"outside AbsRel={overall_metrics.outside.abs_rel:.6f} MAE={overall_metrics.outside.mae:.6f}m "
        f"delta1={overall_metrics.outside.delta1:.6f} frames={len(all_records)}"
    )
    return per_segment, overall


def main(config: EvalCatalogConfig) -> Path:
    """Evaluate selected segments, print explicit metric/diagnostic summaries, and write JSON."""
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if config.height <= 0 or config.width <= 0 or config.input_size <= 0:
        raise ValueError("height, width, and input_size must be positive")
    if config.cameras not in ("wide", "ultrawide", "both"):
        raise ValueError(f"unknown camera selection {config.cameras!r}")
    scores_ultrawide: bool = config.cameras in ("ultrawide", "both")
    if scores_ultrawide and config.evaluation not in ("metric", "prompt-upsample"):
        raise ValueError(f"ultrawide scoring supports the metric and prompt-upsample evaluations, not {config.evaluation!r}")
    catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
    segment_ids: list[str] = _selected_segments(config, catalog)
    transform: AlbumentationsWrapper = build_eval_transform(config.height, config.width)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: Path | None = None
    output_tag: str = "prompt-baseline"
    prompted_model: ZipDepthPrompt | None = None
    relative_predictor: ZipDepthPredictor | None = None
    if config.evaluation != "prompt-upsample":
        checkpoint = config.checkpoint or download_zipdepth_checkpoint()
        output_tag = f"{checkpoint.stem}_net{config.input_size}"
        if config.evaluation == "metric":
            prompted_model = load_zipdepth_prompt(checkpoint).to(device).fuse_for_inference()
        else:
            relative_predictor = ZipDepthPredictor(
                device="cuda" if device.type == "cuda" else "cpu",
                checkpoint=checkpoint,
                input_size=config.input_size,
            )

    # Catalog-only imports stay local so pure metric tests run outside the catalog lane.
    from zipdepth.catalog.builders import CpuSampleBuilder
    from zipdepth.catalog.dataset import CatalogPromptDepthDataset

    per_segment: list[dict[str, float | int | str]] = []
    all_metric_records: list[MetricCatalogDepthMetrics] = []
    all_edge_records: list[dict[str, float]] = []
    all_diagnostic_records: list[MetricCatalogDepthMetrics] = []
    wide_segment_ids: list[str] = segment_ids if config.cameras in ("wide", "both") else []
    segment_id: str
    for segment_id in wide_segment_ids:
        dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
            catalog.dataset_entry,
            [segment_id],
            catalog.row_by_id,
            device=device,
            builder_factory=lambda: CpuSampleBuilder(transform, min_depth_span=0.0, target_mode="metric"),
            shuffle_buffer_size=1,
            frame_stride=config.frame_stride,
            num_producers=1,
            prefetch_samples=4,
        )
        loader: DataLoader[dict[str, Tensor]] = DataLoader(dataset, batch_size=1, num_workers=0)
        segment_metric_records: list[MetricCatalogDepthMetrics] = []
        segment_edge_records: list[dict[str, float]] = []
        segment_diagnostic_records: list[MetricCatalogDepthMetrics] = []
        skipped_frame_count: int = 0
        batch: dict[str, Tensor]
        for batch in loader:
            target_depth_hw: Float32[ndarray, "h w"] = batch["target_depth"][0, 0].numpy().astype(np.float32, copy=False)
            target_valid_hw: Bool[ndarray, "h w"] = batch["target_valid"][0, 0].numpy().astype(bool, copy=False)
            prompt_depth_hw: Float32[ndarray, "192 256"] = batch["prompt_depth"][0, 0].numpy().astype(np.float32, copy=False)
            prompt_valid_hw: Bool[ndarray, "192 256"] = batch["prompt_valid"][0, 0].numpy().astype(bool, copy=False)

            if config.evaluation == "prompt-upsample":
                # Zero-parameter control: bilinearly resize the confidence-masked
                # LiDAR prompt to full resolution. A trained student that does not
                # beat this has learned nothing beyond upsampling its own input.
                prediction_depth_hw: Float32[ndarray, "h w"] = prompt_upsample_depth(
                    prompt_depth_hw, prompt_valid_hw, height=config.height, width=config.width
                )
            elif prompted_model is not None:
                image_bchw: UInt8[Tensor, "b 3 h w"] = batch["image"].to(device=device, non_blocking=True)
                # Raw prompt, matching both training and the teacher's own input.
                prompt_depth_bchw: Float32[Tensor, "b 1 192 256"] = batch["prompt_depth"].to(device=device, non_blocking=True)
                with torch.inference_mode():
                    prediction_depth_hw = prompted_model(image_bchw, prompt_depth_bchw)[0, 0].float().cpu().numpy()
            else:
                if relative_predictor is None:
                    raise RuntimeError("relative predictor was not initialized")
                image_chw: UInt8[ndarray, "3 h w"] = batch["image"][0].numpy()
                rgb_hwc: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(np.moveaxis(image_chw, 0, -1))
                relative_inverse_hw: Float32[ndarray, "h w"] = relative_predictor(rgb_hwc, None).disparity
                if config.evaluation == "affine-reference":
                    try:
                        prediction_depth_hw = affine_reference_depth(relative_inverse_hw, prompt_depth_hw, prompt_valid_hw)
                    except ValueError as error:
                        print(f"{segment_id}: skipping frame without a solvable LiDAR fit ({error})")
                        skipped_frame_count += 1
                        continue
                else:
                    prediction_depth_hw = np.reciprocal(np.clip(relative_inverse_hw, 1.0e-6, None)).astype(np.float32)

            if config.evaluation != "aligned-relative":
                metric: MetricCatalogDepthMetrics = score_metric_depth(prediction_depth_hw, target_depth_hw, target_valid_hw)
                segment_metric_records.append(metric)
                # Skip the stratified block when the prediction IS the baseline: the
                # ratios would compare an upsampler against itself and print 1.0x.
                if config.evaluation != "prompt-upsample":
                    height, width = target_depth_hw.shape
                    baseline_depth_hw: Float32[ndarray, "h w"] = prompt_upsample_depth(
                        prompt_depth_hw, prompt_valid_hw, height=height, width=width
                    )
                    stratified: EdgeStratifiedResult | None = edge_stratified_mae(
                        prediction_depth_hw, target_depth_hw, target_valid_hw, baseline_depth_hw
                    )
                    if stratified is not None:
                        if stratified.baseline is None:
                            raise AssertionError("a supplied baseline must produce baseline metrics")
                        segment_edge_records.append(
                            {
                                "edge_mae": stratified.prediction.edge_mae_m,
                                "flat_mae": stratified.prediction.flat_mae_m,
                                "ewmae": stratified.prediction.ewmae_m,
                                "baseline_edge_mae": stratified.baseline.edge_mae_m,
                                "baseline_flat_mae": stratified.baseline.flat_mae_m,
                                "baseline_ewmae": stratified.baseline.ewmae_m,
                            }
                        )
            diagnostic: MetricCatalogDepthMetrics = aligned_inverse_diagnostic(
                prediction_depth_hw,
                target_depth_hw,
                target_valid_hw,
            )
            segment_diagnostic_records.append(diagnostic)

        if not segment_diagnostic_records:
            print(f"{segment_id}: no scorable frames (skipped={skipped_frame_count})")
            continue
        frame_count: int = len(segment_diagnostic_records)
        segment_diagnostic: MetricCatalogDepthMetrics = _mean_metrics(segment_diagnostic_records)
        segment_report: dict[str, float | int | str] = {
            "segment_id": segment_id,
            "frame_count": frame_count,
            "skipped_frame_count": skipped_frame_count,
            "aligned_inverse_diagnostic_abs_rel": segment_diagnostic.abs_rel,
            "aligned_inverse_diagnostic_delta1": segment_diagnostic.delta1,
            "aligned_inverse_diagnostic_mae": segment_diagnostic.mae,
        }
        if segment_metric_records:
            segment_metric: MetricCatalogDepthMetrics = _mean_metrics(segment_metric_records)
            segment_report.update(
                {
                    "metric_abs_rel": segment_metric.abs_rel,
                    "metric_delta1": segment_metric.delta1,
                    "metric_mae": segment_metric.mae,
                }
            )
            print(
                f"{segment_id}: metric AbsRel={segment_report['metric_abs_rel']:.6f} "
                f"delta1={segment_report['metric_delta1']:.6f} MAE={segment_report['metric_mae']:.6f}m frames={frame_count}"
            )
            all_metric_records.extend(segment_metric_records)
            all_edge_records.extend(segment_edge_records)
            if segment_edge_records:
                segment_stratified_means: dict[str, float] = _mean_by_key(segment_edge_records)
                segment_report.update(segment_stratified_means)
                print(
                    f"{segment_id}: edge MAE={segment_stratified_means['edge_mae']:.6f}m "
                    f"flat MAE={segment_stratified_means['flat_mae']:.6f}m EWMAE={segment_stratified_means['ewmae']:.6f}m "
                    f"(baseline={segment_stratified_means['baseline_ewmae']:.6f}m)"
                )
        else:
            print(
                f"{segment_id}: aligned_inverse_diagnostic AbsRel={segment_report['aligned_inverse_diagnostic_abs_rel']:.6f} "
                f"delta1={segment_report['aligned_inverse_diagnostic_delta1']:.6f} frames={frame_count}"
            )
        per_segment.append(segment_report)
        all_diagnostic_records.extend(segment_diagnostic_records)

    if not all_diagnostic_records and wide_segment_ids:
        raise RuntimeError("evaluation yielded no valid frames")
    overall: dict[str, float | int] = {}
    if all_diagnostic_records:
        overall_diagnostic: MetricCatalogDepthMetrics = _mean_metrics(all_diagnostic_records)
        overall.update(
            {
                "frame_count": len(all_diagnostic_records),
                "aligned_inverse_diagnostic_abs_rel": overall_diagnostic.abs_rel,
                "aligned_inverse_diagnostic_delta1": overall_diagnostic.delta1,
                "aligned_inverse_diagnostic_mae": overall_diagnostic.mae,
            }
        )
    if all_metric_records:
        overall_metric: MetricCatalogDepthMetrics = _mean_metrics(all_metric_records)
        overall.update(
            {
                "metric_abs_rel": overall_metric.abs_rel,
                "metric_delta1": overall_metric.delta1,
                "metric_mae": overall_metric.mae,
            }
        )
        print(
            f"overall metric: AbsRel={overall['metric_abs_rel']:.6f} delta1={overall['metric_delta1']:.6f} "
            f"MAE={overall['metric_mae']:.6f}m frames={overall['frame_count']}"
        )
        if all_edge_records:
            stratified_means: dict[str, float] = _mean_by_key(all_edge_records)
            stratified_means["edge_ratio_vs_baseline"] = stratified_means["baseline_edge_mae"] / max(stratified_means["edge_mae"], 1e-9)
            stratified_means["flat_ratio_vs_baseline"] = stratified_means["baseline_flat_mae"] / max(stratified_means["flat_mae"], 1e-9)
            overall.update(stratified_means)
            print(
                f"overall edge-stratified: edge MAE={stratified_means['edge_mae']:.6f}m "
                f"(baseline/pred {stratified_means['edge_ratio_vs_baseline']:.3f}x) | "
                f"flat MAE={stratified_means['flat_mae']:.6f}m "
                f"(baseline/pred {stratified_means['flat_ratio_vs_baseline']:.3f}x) | "
                f"EWMAE={stratified_means['ewmae']:.6f}m (baseline={stratified_means['baseline_ewmae']:.6f}m)"
            )
    if all_diagnostic_records:
        print(
            f"overall aligned_inverse_diagnostic: AbsRel={overall['aligned_inverse_diagnostic_abs_rel']:.6f} "
            f"delta1={overall['aligned_inverse_diagnostic_delta1']:.6f} MAE={overall['aligned_inverse_diagnostic_mae']:.6f}m"
        )

    ultrawide_per_segment: list[dict[str, float | int | str]] = []
    if scores_ultrawide:
        ultrawide_result: tuple[list[dict[str, float | int | str]], dict[str, float | int]] = _evaluate_ultrawide(
            config, catalog, segment_ids, prompted_model, device
        )
        ultrawide_per_segment = ultrawide_result[0]
        overall.update(ultrawide_result[1])

    config.save_dir.mkdir(parents=True, exist_ok=True)
    output: Path = config.save_dir / f"eval_{config.evaluation}_{output_tag}.json"
    report: dict[str, object] = {
        "config": {**asdict(config), "checkpoint": str(checkpoint) if checkpoint is not None else None, "segment_ids": segment_ids},
        "per_segment": per_segment,
        "ultrawide_per_segment": ultrawide_per_segment,
        "overall": overall,
    }
    with output.open("w") as file:
        json.dump(report, file, indent=2, default=str)
    return output
