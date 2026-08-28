"""Evaluate ZipDepth inverse depth on held-out catalog PromptDA targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, UInt8
from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor, download_zipdepth_checkpoint
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader

from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, PromptDACatalog, load_promptda_catalog
from zipdepth.catalog.targets import INV_DEPTH_SCALE, build_eval_transform
from zipdepth.data.transforms import AlbumentationsWrapper
from zipdepth.evaluation.metrics import abs_relative_difference, align_depth_least_square, delta1_acc, disparity2depth


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
    """Trainable checkpoint to evaluate; None uses released ZipDepth-base weights."""
    height: int = 768
    """Deterministic evaluation image height."""
    width: int = 1024
    """Deterministic evaluation image width."""
    frame_stride: int = 10
    """Evaluate every Nth chosen PromptDA frame within each segment."""
    input_size: int = 384
    """Shorter image side the network runs at (``ZipDepthPredictor`` resizes to it, then upsamples the disparity).
    384 is the released model's training size; catalog fine-tunes train the network directly at ``height`` x ``width``,
    so score those with ``input_size=height`` to avoid a train/eval resolution mismatch."""
    max_segments: int | None = None
    """Optional limit on the selected evaluation segments."""


@dataclass(slots=True, frozen=True)
class CatalogDepthMetrics:
    """Depth metrics after scale-and-shift alignment in inverse-depth space."""

    abs_rel: float
    """Mean absolute relative depth error over valid pixels."""
    delta1: float
    """Fraction of valid depth ratios below 1.25."""


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
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open() as file:
            loaded: object = json.load(file)
        if not isinstance(loaded, list) or not all(isinstance(value, str) for value in loaded):
            raise ValueError(f"holdout manifest must contain a JSON string list: {manifest}")
        segment_ids: list[str] = list(loaded)
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
    """Evaluate selected segments, print summaries, and write a JSON report."""
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
    segment_ids: list[str] = _selected_segments(config, catalog)
    transform: AlbumentationsWrapper = build_eval_transform(config.height, config.width)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: Path = config.checkpoint or download_zipdepth_checkpoint()
    predictor: ZipDepthPredictor = ZipDepthPredictor(
        device="cuda" if device.type == "cuda" else "cpu", checkpoint=checkpoint, input_size=config.input_size
    )

    # Catalog-only import stays local so pure metric tests run in zipdepth-dev.
    from zipdepth.catalog.builders import CpuSampleBuilder
    from zipdepth.catalog.dataset import CatalogPromptDepthDataset

    per_segment: list[dict[str, float | int | str]] = []
    all_abs_rel: list[float] = []
    all_delta1: list[float] = []
    segment_id: str
    for segment_id in segment_ids:
        dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
            catalog.dataset_entry,
            [segment_id],
            catalog.row_by_id,
            device=device,
            builder_factory=lambda: CpuSampleBuilder(transform, min_depth_span=1.25),
            shuffle_buffer_size=1,
            frame_stride=config.frame_stride,
            num_producers=1,
            prefetch_samples=4,
        )
        loader: DataLoader[dict[str, Tensor]] = DataLoader(dataset, batch_size=1, num_workers=0)
        segment_abs_rel: list[float] = []
        segment_delta1: list[float] = []
        batch: dict[str, Tensor]
        for batch in loader:
            image_chw: UInt8[ndarray, "3 h w"] = batch["image"][0].numpy()
            rgb_hw3: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(np.moveaxis(image_chw, 0, -1))
            target_hw: Float32[ndarray, "h w"] = batch["depth"][0, 0].numpy().astype(np.float32) / INV_DEPTH_SCALE
            valid_hw: Bool[ndarray, "h w"] = batch["mask"][0, 0].numpy().astype(bool, copy=False)
            prediction_hw: Float32[ndarray, "h w"] = predictor(rgb_hw3, None).disparity
            metrics: CatalogDepthMetrics = align_and_score_inverse_depth(prediction_hw, target_hw, valid_hw)
            segment_abs_rel.append(metrics.abs_rel)
            segment_delta1.append(metrics.delta1)
        if not segment_abs_rel:
            print(f"{segment_id}: no frames")
            continue
        mean_abs_rel: float = float(np.mean(segment_abs_rel))
        mean_delta1: float = float(np.mean(segment_delta1))
        frame_count: int = len(segment_abs_rel)
        print(f"{segment_id}: AbsRel={mean_abs_rel:.6f} delta1={mean_delta1:.6f} frames={frame_count}")
        per_segment.append({"segment_id": segment_id, "abs_rel": mean_abs_rel, "delta1": mean_delta1, "frame_count": frame_count})
        all_abs_rel.extend(segment_abs_rel)
        all_delta1.extend(segment_delta1)

    if not all_abs_rel:
        raise RuntimeError("evaluation yielded no valid frames")
    overall_abs_rel: float = float(np.mean(all_abs_rel))
    overall_delta1: float = float(np.mean(all_delta1))
    overall_frames: int = len(all_abs_rel)
    print(f"overall: AbsRel={overall_abs_rel:.6f} delta1={overall_delta1:.6f} frames={overall_frames}")

    config.save_dir.mkdir(parents=True, exist_ok=True)
    output: Path = config.save_dir / f"eval_{checkpoint.stem}_net{config.input_size}.json"
    report: dict[str, object] = {
        "config": {**asdict(config), "checkpoint": str(checkpoint)},
        "per_segment": per_segment,
        "overall": {"abs_rel": overall_abs_rel, "delta1": overall_delta1, "frame_count": overall_frames},
    }
    with output.open("w") as file:
        json.dump(report, file, indent=2, default=str)
    return output
