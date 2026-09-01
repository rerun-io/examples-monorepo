"""Evaluate metric ZipDepth-PromptDA and explicit reference diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Literal

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, UInt8
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor, download_zipdepth_checkpoint
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader

from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, PromptDACatalog, load_promptda_catalog
from zipdepth.catalog.targets import build_eval_transform
from zipdepth.data.transforms import AlbumentationsWrapper
from zipdepth.evaluation.metrics import abs_relative_difference, align_depth_least_square, delta1_acc, disparity2depth

EDGE_QUANTILE: float = 0.90
"""Teacher-gradient quantile defining the edge stratum in stratified metrics."""


def edge_stratified_mae(
    prediction_depth_hw: Float32[ndarray, "h w"],
    target_depth_hw: Float32[ndarray, "h w"],
    target_valid_hw: Bool[ndarray, "h w"],
    prompt_depth_hw: Float32[ndarray, "192 256"],
) -> dict[str, float] | None:
    """MAE split by teacher depth edges, for the prediction and a bilinear prompt baseline.

    Edges are the top valid pixels by teacher spatial gradient magnitude (above the
    ``EDGE_QUANTILE`` per-frame threshold). The baseline is the raw prompt bilinearly
    upsampled to target size — the depth any phone gives for free — so the derived
    baseline/prediction ratios read as edge and flat improvement over doing nothing.
    Whole-frame AbsRel is blind to this stratum: edges are ~10% of pixels.

    Returns:
        Edge/flat MAE for the prediction and the baseline, or None when a stratum
        is empty or the frame has too few valid pixels to threshold.
    """
    gradient_hw: Float32[ndarray, "h w"] = np.zeros_like(target_depth_hw)
    gradient_x: Float32[ndarray, "h w_minus_one"] = np.abs(np.diff(target_depth_hw, axis=1))
    gradient_y: Float32[ndarray, "h_minus_one w"] = np.abs(np.diff(target_depth_hw, axis=0))
    gradient_hw[:, :-1] += gradient_x
    gradient_hw[:, 1:] += gradient_x
    gradient_hw[:-1, :] += gradient_y
    gradient_hw[1:, :] += gradient_y
    valid_gradients: Float32[ndarray, "n_valid"] = gradient_hw[target_valid_hw]
    if valid_gradients.size < 100:
        return None
    threshold: float = float(np.quantile(valid_gradients, EDGE_QUANTILE))
    edge_hw: Bool[ndarray, "h w"] = target_valid_hw & (gradient_hw >= threshold)
    flat_hw: Bool[ndarray, "h w"] = target_valid_hw & (gradient_hw < threshold)
    if not edge_hw.any() or not flat_hw.any():
        return None
    height, width = target_depth_hw.shape
    baseline_hw: Float32[ndarray, "h w"] = cv2.resize(prompt_depth_hw, (width, height), interpolation=cv2.INTER_LINEAR)
    prediction_error_hw: Float32[ndarray, "h w"] = np.abs(prediction_depth_hw - target_depth_hw)
    baseline_error_hw: Float32[ndarray, "h w"] = np.abs(baseline_hw - target_depth_hw)
    return {
        "edge_mae": float(prediction_error_hw[edge_hw].mean()),
        "flat_mae": float(prediction_error_hw[flat_hw].mean()),
        "baseline_edge_mae": float(baseline_error_hw[edge_hw].mean()),
        "baseline_flat_mae": float(baseline_error_hw[flat_hw].mean()),
    }


@dataclass(slots=True)
class EvalCatalogConfig:
    """Held-out catalog evaluation configuration."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = "arkitscenes-v2"
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


@dataclass(slots=True, frozen=True)
class CatalogDepthMetrics:
    """Depth metrics after scale-and-shift alignment in inverse-depth space."""

    abs_rel: float
    """Mean absolute relative depth error over valid pixels."""
    delta1: float
    """Fraction of valid depth ratios below 1.25."""


@dataclass(slots=True, frozen=True)
class MetricCatalogDepthMetrics:
    """Unaligned or explicitly diagnostic metric-depth scores."""

    abs_rel: float
    """Mean absolute relative depth error over valid pixels."""
    delta1: float
    """Fraction of valid depth ratios below 1.25."""
    mae: float
    """Mean absolute error in metres."""


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
    valid_f32_hw: Float32[ndarray, "192 256"] = prompt_valid_hw.astype(np.float32)
    weighted_hw: Float32[ndarray, "192 256"] = (prompt_depth_hw * valid_f32_hw).astype(np.float32)
    weighted_up_hw: Float32[ndarray, "h w"] = cv2.resize(weighted_hw, (width, height), interpolation=cv2.INTER_LINEAR)
    weight_up_hw: Float32[ndarray, "h w"] = cv2.resize(valid_f32_hw, (width, height), interpolation=cv2.INTER_LINEAR)
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
        & (target_depth_hw >= 0.1)
        & (target_depth_hw <= 4.0)
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
        & (prompt_depth_hw >= 0.1)
        & (prompt_depth_hw <= 4.0)
    )
    if int(np.count_nonzero(fit_hw)) < 10:
        raise ValueError("at least ten valid LiDAR prompt pixels are required")
    x_n: Float[ndarray, "n"] = relative_prompt_hw[fit_hw].astype(np.float64)
    y_n: Float[ndarray, "n"] = (1.0 / prompt_depth_hw[fit_hw]).astype(np.float64)
    x_mean: float = float(np.mean(x_n))
    y_mean: float = float(np.mean(y_n))
    centered_x_n: Float[ndarray, "n"] = x_n - x_mean
    variance: float = float(np.sum(centered_x_n * centered_x_n))
    if variance <= 1.0e-12:
        raise ValueError("relative inverse depth is constant over the valid LiDAR prompt")
    scale: float = float(np.sum(centered_x_n * (y_n - y_mean)) / variance)
    shift: float = y_mean - scale * x_mean
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
        & (target_depth_hw >= 0.1)
        & (target_depth_hw <= 4.0)
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


def align_and_score_inverse_depth(
    prediction_hw: Float[ndarray, "h w"],
    target_hw: Float[ndarray, "h w"],
    valid_hw: Bool[ndarray, "h w"],
) -> CatalogDepthMetrics:
    """Align and score affine-invariant inverse depth.

    A closed-form least-squares scale and shift fits ``prediction`` to target
    inverse depth over finite, positive, valid pixels. Both aligned disparities
    are converted to depth before standard AbsRel and delta1 are computed over
    that same mask. At least ten valid pixels are required, matching the
    vendored evaluation helper and protocol.

    Args:
        prediction_hw: Predicted floating inverse depth with shape ``(H, W)``.
        target_hw: PromptDA floating inverse depth with shape ``(H, W)``.
        valid_hw: Boolean validity mask with shape ``(H, W)``.

    Returns:
        Aligned AbsRel and delta1 metrics.

    Raises:
        ValueError: If shapes differ or fewer than ten pixels are valid after filtering.
    """
    if prediction_hw.shape != target_hw.shape or valid_hw.shape != target_hw.shape:
        raise ValueError("prediction, target, and validity mask must have equal shapes")
    fit_hw: Bool[ndarray, "h w"] = valid_hw & np.isfinite(prediction_hw) & np.isfinite(target_hw) & (target_hw > 0.0)
    if int(np.count_nonzero(fit_hw)) < 10:
        raise ValueError("at least ten valid inverse-depth pixels are required")
    aligned_hw: Float[ndarray, "h w"] = align_depth_least_square(target_hw, prediction_hw, fit_hw)
    metric_hw: Bool[ndarray, "h w"] = fit_hw & np.isfinite(aligned_hw) & (aligned_hw > 0.0)
    prediction_depth_hw: Float[ndarray, "h w"] = disparity2depth(aligned_hw)
    target_depth_hw: Float[ndarray, "h w"] = disparity2depth(target_hw)
    return CatalogDepthMetrics(
        abs_rel=abs_relative_difference(prediction_depth_hw, target_depth_hw, metric_hw),
        delta1=delta1_acc(prediction_depth_hw, target_depth_hw, metric_hw),
    )


def _selected_segments(config: EvalCatalogConfig, catalog: PromptDACatalog) -> list[str]:
    """Resolve explicit segments or the saved training holdout manifest."""
    if config.segment_ids is None:
        manifest: Path = config.save_dir / "holdout_segments.json"
        if manifest.is_file():
            with manifest.open() as file:
                loaded: object = json.load(file)
            if not isinstance(loaded, list) or not all(isinstance(value, str) for value in loaded):
                raise ValueError(f"holdout manifest must contain a JSON string list: {manifest}")
            segment_ids: list[str] = list(loaded)
        else:
            if not 0 < config.holdout_count <= len(catalog.segment_ids):
                raise ValueError(f"holdout_count must be in [1, {len(catalog.segment_ids)}]")
            holdout_set: set[str] = set(Random(config.holdout_seed).sample(sorted(catalog.segment_ids), config.holdout_count))
            segment_ids = sorted(holdout_set)
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


def main(config: EvalCatalogConfig) -> Path:
    """Evaluate selected segments, print explicit metric/diagnostic summaries, and write JSON."""
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if config.height <= 0 or config.width <= 0 or config.input_size <= 0:
        raise ValueError("height, width, and input_size must be positive")
    catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
    segment_ids: list[str] = _selected_segments(config, catalog)
    transform: AlbumentationsWrapper = build_eval_transform(config.height, config.width)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: Path = config.checkpoint or download_zipdepth_checkpoint()
    prompted_model: ZipDepthPrompt | None = None
    relative_predictor: ZipDepthPredictor | None = None
    if config.evaluation == "metric":
        prompted_model = load_zipdepth_prompt(checkpoint).to(device).fuse_for_inference()
    elif config.evaluation != "prompt-upsample":
        # The prompt-upsample control needs no network at all.
        relative_predictor = ZipDepthPredictor(
            device="cuda" if device.type == "cuda" else "cpu",
            checkpoint=checkpoint,
            input_size=config.input_size,
        )

    # Catalog-only imports stay local so pure metric tests run outside the catalog lane.
    from zipdepth.catalog.builders import CpuSampleBuilder
    from zipdepth.catalog.dataset import CatalogPromptDepthDataset

    per_segment: list[dict[str, float | int | str]] = []
    all_metric_abs_rel: list[float] = []
    all_metric_delta1: list[float] = []
    all_metric_mae: list[float] = []
    all_edge_records: list[dict[str, float]] = []
    all_diagnostic_abs_rel: list[float] = []
    all_diagnostic_delta1: list[float] = []
    all_diagnostic_mae: list[float] = []
    segment_id: str
    for segment_id in segment_ids:
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
        segment_metric_abs_rel: list[float] = []
        segment_metric_delta1: list[float] = []
        segment_metric_mae: list[float] = []
        segment_edge_records: list[dict[str, float]] = []
        segment_diagnostic_abs_rel: list[float] = []
        segment_diagnostic_delta1: list[float] = []
        segment_diagnostic_mae: list[float] = []
        skipped_frame_count: int = 0
        batch: dict[str, Tensor]
        for batch in loader:
            image_chw: UInt8[ndarray, "3 h w"] = batch["image"][0].numpy()
            rgb_hw3: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(np.moveaxis(image_chw, 0, -1))
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
                image_b3hw: Tensor = batch["image"].to(device=device, non_blocking=True)
                # Raw prompt, matching both training and the teacher's own input.
                prompt_depth_b1hw: Tensor = batch["prompt_depth"].to(device=device, non_blocking=True)
                with torch.inference_mode():
                    prediction_depth_hw = prompted_model(image_b3hw, prompt_depth_b1hw)[0, 0].float().cpu().numpy()
            else:
                if relative_predictor is None:
                    raise RuntimeError("relative predictor was not initialized")
                relative_inverse_hw: Float32[ndarray, "h w"] = relative_predictor(rgb_hw3, None).disparity
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
                segment_metric_abs_rel.append(metric.abs_rel)
                segment_metric_delta1.append(metric.delta1)
                segment_metric_mae.append(metric.mae)
                stratified: dict[str, float] | None = edge_stratified_mae(
                    prediction_depth_hw, target_depth_hw, target_valid_hw, prompt_depth_hw
                )
                if stratified is not None:
                    segment_edge_records.append(stratified)
            diagnostic: MetricCatalogDepthMetrics = aligned_inverse_diagnostic(
                prediction_depth_hw,
                target_depth_hw,
                target_valid_hw,
            )
            segment_diagnostic_abs_rel.append(diagnostic.abs_rel)
            segment_diagnostic_delta1.append(diagnostic.delta1)
            segment_diagnostic_mae.append(diagnostic.mae)

        if not segment_diagnostic_abs_rel:
            print(f"{segment_id}: no scorable frames (skipped={skipped_frame_count})")
            continue
        frame_count: int = len(segment_diagnostic_abs_rel)
        segment_report: dict[str, float | int | str] = {
            "segment_id": segment_id,
            "frame_count": frame_count,
            "skipped_frame_count": skipped_frame_count,
            "aligned_inverse_diagnostic_abs_rel": float(np.mean(segment_diagnostic_abs_rel)),
            "aligned_inverse_diagnostic_delta1": float(np.mean(segment_diagnostic_delta1)),
            "aligned_inverse_diagnostic_mae": float(np.mean(segment_diagnostic_mae)),
        }
        if segment_metric_abs_rel:
            segment_report.update(
                {
                    "metric_abs_rel": float(np.mean(segment_metric_abs_rel)),
                    "metric_delta1": float(np.mean(segment_metric_delta1)),
                    "metric_mae": float(np.mean(segment_metric_mae)),
                }
            )
            print(
                f"{segment_id}: metric AbsRel={segment_report['metric_abs_rel']:.6f} "
                f"delta1={segment_report['metric_delta1']:.6f} MAE={segment_report['metric_mae']:.6f}m frames={frame_count}"
            )
            all_metric_abs_rel.extend(segment_metric_abs_rel)
            all_metric_delta1.extend(segment_metric_delta1)
            all_metric_mae.extend(segment_metric_mae)
            all_edge_records.extend(segment_edge_records)
            if segment_edge_records:
                segment_report.update(
                    {key: float(np.mean([record[key] for record in segment_edge_records])) for key in segment_edge_records[0]}
                )
        else:
            print(
                f"{segment_id}: aligned_inverse_diagnostic AbsRel={segment_report['aligned_inverse_diagnostic_abs_rel']:.6f} "
                f"delta1={segment_report['aligned_inverse_diagnostic_delta1']:.6f} frames={frame_count}"
            )
        per_segment.append(segment_report)
        all_diagnostic_abs_rel.extend(segment_diagnostic_abs_rel)
        all_diagnostic_delta1.extend(segment_diagnostic_delta1)
        all_diagnostic_mae.extend(segment_diagnostic_mae)

    if not all_diagnostic_abs_rel:
        raise RuntimeError("evaluation yielded no valid frames")
    overall: dict[str, float | int] = {
        "frame_count": len(all_diagnostic_abs_rel),
        "aligned_inverse_diagnostic_abs_rel": float(np.mean(all_diagnostic_abs_rel)),
        "aligned_inverse_diagnostic_delta1": float(np.mean(all_diagnostic_delta1)),
        "aligned_inverse_diagnostic_mae": float(np.mean(all_diagnostic_mae)),
    }
    if all_metric_abs_rel:
        overall.update(
            {
                "metric_abs_rel": float(np.mean(all_metric_abs_rel)),
                "metric_delta1": float(np.mean(all_metric_delta1)),
                "metric_mae": float(np.mean(all_metric_mae)),
            }
        )
        print(
            f"overall metric: AbsRel={overall['metric_abs_rel']:.6f} delta1={overall['metric_delta1']:.6f} "
            f"MAE={overall['metric_mae']:.6f}m frames={overall['frame_count']}"
        )
        if all_edge_records:
            stratified_means: dict[str, float] = {
                key: float(np.mean([record[key] for record in all_edge_records])) for key in all_edge_records[0]
            }
            stratified_means["edge_ratio_vs_baseline"] = stratified_means["baseline_edge_mae"] / max(stratified_means["edge_mae"], 1e-9)
            stratified_means["flat_ratio_vs_baseline"] = stratified_means["baseline_flat_mae"] / max(stratified_means["flat_mae"], 1e-9)
            overall.update(stratified_means)
            print(
                f"overall edge-stratified: edge MAE={stratified_means['edge_mae']:.6f}m "
                f"(baseline/pred {stratified_means['edge_ratio_vs_baseline']:.3f}x) | "
                f"flat MAE={stratified_means['flat_mae']:.6f}m "
                f"(baseline/pred {stratified_means['flat_ratio_vs_baseline']:.3f}x)"
            )
    print(
        f"overall aligned_inverse_diagnostic: AbsRel={overall['aligned_inverse_diagnostic_abs_rel']:.6f} "
        f"delta1={overall['aligned_inverse_diagnostic_delta1']:.6f} MAE={overall['aligned_inverse_diagnostic_mae']:.6f}m"
    )

    config.save_dir.mkdir(parents=True, exist_ok=True)
    output: Path = config.save_dir / f"eval_{config.evaluation}_{checkpoint.stem}_net{config.input_size}.json"
    report: dict[str, object] = {
        "config": {**asdict(config), "checkpoint": str(checkpoint), "segment_ids": segment_ids},
        "per_segment": per_segment,
        "overall": overall,
    }
    with output.open("w") as file:
        json.dump(report, file, indent=2, default=str)
    return output
