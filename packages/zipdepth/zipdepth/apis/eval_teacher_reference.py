"""Reference ceiling for the ultrawide lane: PromptDA-L beside the ZipDepth students.

The ultrawide lane prompts a wide LiDAR map that only covers the central
``1 / ULTRAWIDE_FOV_RATIO`` of the frame, so the periphery is unprompted. This
module answers what a much larger prompted model does with exactly the same
input: it builds the lane's own ultrawide samples once and runs PromptDA-L and
every requested ZipDepth-PromptDA checkpoint over the identical frames, scoring
all of them with :func:`score_footprint_split`.

Two extra diagnostics come out of the same pass:

* the zero-parameter bilinear prompt-upsample floor inside the footprint, which
  :func:`score_footprint_split` already reports; and
* the **unreachable periphery**: valid target pixels outside the footprint whose
  depth lies outside the prompt's own ``[min, max]``. A ZipDepth-PromptDA head
  with ``range_margin_m = 0`` cannot emit those depths at all, so splitting the
  periphery AbsRel by reachability separates the output clamp from the rest of
  the periphery error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8
from monopriors.models.depth_completion.base_completion_depth import MAX_METRIC_DEPTH_M, MIN_METRIC_DEPTH_M
from monopriors.models.depth_completion.prompt_da import PromptDAConfig, PromptDAPredictor, network_image_hw
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader
from trtkit import BackendConfig, TensorRtBackendConfig, TorchBackendConfig

from zipdepth.apis.eval_catalog import (
    FootprintSplitMetrics,
    MetricCatalogDepthMetrics,
    footprint_mask,
    mean_footprint_metrics,
    score_footprint_split,
    score_metric_depth,
)
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, PromptDACatalog, load_promptda_catalog, split_holdout_segments
from zipdepth.catalog.targets import build_eval_transform
from zipdepth.catalog.ultrawide import (
    DEFAULT_ULTRAWIDE_PROMPT_SCALE,
    ULTRAWIDE_LAYER,
    PromptPlacement,
    UltrawidePolicy,
    prompt_placement,
)
from zipdepth.data.transforms import AlbumentationsWrapper

TEACHER_NAME: str = "promptda-l"
"""Report key for PromptDA-L on the lane's zero-padded prompt canvas."""
TEACHER_REPLICATED_NAME: str = "promptda-l-replicated"
"""Report key for PromptDA-L on a canvas whose periphery replicates the block edge."""
PROMPTDA_CAPTURE_HW: tuple[int, int] = (1440, 1920)
"""ARKitScenes wide capture size PromptDA-L's network resolution is derived from."""


@dataclass(slots=True)
class TeacherReferenceConfig:
    """Ultrawide teacher-versus-student comparison over one holdout set."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset containing ARKitScenes and PromptDA layers."""
    save_dir: Path = Path("data/checkpoints/zdpda-uw-v1")
    """Training output directory whose ``holdout_segments.json`` selects the frames."""
    segment_ids: list[str] | None = None
    """Explicit segment identifiers; overrides the saved holdout manifest."""
    student_checkpoints: list[Path] = field(default_factory=list)
    """ZipDepth-PromptDA checkpoints scored on the same frames, keyed by parent/stem."""
    include_replicated_teacher: bool = True
    """Also score PromptDA-L on an edge-replicated prompt canvas.

    PromptDA normalizes its prompt by the canvas ``[min, max]`` and de-normalizes
    its output with the same pair, so the lane's zero padding forces ``min = 0``
    and hands the periphery a literal "zero metres" prompt. This second pass keeps
    the frame, the block, and the network identical and only replaces the padding,
    which separates that input artifact from the difficulty of the periphery.
    """
    promptda_backend: Literal["torch", "tensorrt"] = "torch"
    """PromptDA-L runtime. ``torch`` is eager fp32; ``tensorrt`` reuses a cached fp16 engine."""
    promptda_autocast: Literal["fp32", "fp16", "bf16"] = "fp32"
    """Autocast precision for the torch backend only."""
    promptda_max_image_size: int = 1008
    """Longest PromptDA-L network side; the released default for a 1440x1920 capture."""
    batch_size: int = 8
    """Frames per forward pass; also the TensorRT profile batch."""
    height: int = 768
    """Ultrawide sample height, matching the lane."""
    width: int = 1024
    """Ultrawide sample width, matching the lane."""
    frame_stride: int = 10
    """Score every Nth chosen ultrawide frame within each segment."""
    max_segments: int | None = None
    """Optional limit on the selected segments."""
    holdout_count: int = 20
    """Deterministic fallback holdout size when no saved manifest exists."""
    holdout_seed: int = 0
    """Deterministic fallback holdout seed."""
    ultrawide_min_valid_fraction: float = 0.0
    """Ultrawide frames below this in-range target fraction are skipped."""
    ultrawide_valid_erosion_px: int = 1
    """Ultrawide target-mask erosion radius; matches the training lane."""
    ultrawide_prompt_scale: float = DEFAULT_ULTRAWIDE_PROMPT_SCALE
    """Wide prompt footprint inside an ultrawide frame, as a fraction per axis."""
    output_json: Path = Path("data/teacher_reference.json")
    """Destination for the full report."""
    dump_frames_dir: Path | None = None
    """When set, writes one landscape and one portrait frame as ``.npz`` for visualization."""


@dataclass(frozen=True, slots=True)
class PeripheryReachability:
    """Periphery error split by whether the prompt's own range can reach the target.

    ``unreachable`` pixels lie outside the prompt's ``[min, max]``, which is
    exactly the interval a ``range_margin_m = 0`` head is clamped to.
    """

    unreachable_fraction: float
    """Share of valid periphery pixels the prompt range cannot reach."""
    unreachable_abs_rel: float | None
    """AbsRel restricted to unreachable periphery pixels, or None when too few."""
    reachable_abs_rel: float | None
    """AbsRel restricted to the remaining periphery pixels, or None when too few."""


def replicate_prompt_canvas(prompt_bchw: Float32[Tensor, "b 1 192 256"], placement: PromptPlacement) -> Float32[Tensor, "b 1 192 256"]:
    """Refill a padded prompt canvas by replicating the block's edge outward.

    Args:
        prompt_bchw: Padded prompt canvas in metres, zero outside the block.
        placement: Where the block sits inside the canvas.

    Returns:
        A canvas of the same shape whose periphery repeats the nearest block pixel.
    """
    canvas_height: int = int(prompt_bchw.shape[-2])
    canvas_width: int = int(prompt_bchw.shape[-1])
    block_bchw: Float32[Tensor, "b 1 block_h block_w"] = prompt_bchw[
        ..., placement.top : placement.top + placement.height, placement.left : placement.left + placement.width
    ]
    padding: tuple[int, int, int, int] = (
        placement.left,
        canvas_width - placement.left - placement.width,
        placement.top,
        canvas_height - placement.top - placement.height,
    )
    return torch.nn.functional.pad(block_bchw, padding, mode="replicate")


def prompt_output_range(prompt_depth_hw: Float32[ndarray, "192 256"]) -> tuple[float, float]:
    """Return the prompt's own ``[min, max]`` under the shared metric window.

    This mirrors :meth:`ZipDepthPrompt.forward_with_range` exactly: validity comes
    from the raw prompt canvas, not from the builder's confidence mask, and an
    empty prompt falls back to the full window.

    Args:
        prompt_depth_hw: Padded prompt canvas depth in metres.

    Returns:
        The lower and upper edge of the un-widened output range, in metres.
    """
    valid_hw: Bool[ndarray, "192 256"] = (
        np.isfinite(prompt_depth_hw) & (prompt_depth_hw >= MIN_METRIC_DEPTH_M) & (prompt_depth_hw <= MAX_METRIC_DEPTH_M)
    )
    if not bool(valid_hw.any()):
        return MIN_METRIC_DEPTH_M, MAX_METRIC_DEPTH_M
    return float(prompt_depth_hw[valid_hw].min()), float(prompt_depth_hw[valid_hw].max())


def score_periphery_reachability(
    prediction_depth_hw: Float32[ndarray, "h w"],
    target_depth_hw: Float32[ndarray, "h w"],
    periphery_hw: Bool[ndarray, "h w"],
    prompt_range: tuple[float, float],
) -> PeripheryReachability:
    """Split periphery AbsRel by whether the prompt range contains the target depth.

    Args:
        prediction_depth_hw: Predicted metric depth in metres.
        target_depth_hw: Raycast ultrawide target depth in metres.
        periphery_hw: Valid target pixels outside the prompt footprint.
        prompt_range: The prompt's own ``(min, max)`` in metres.

    Returns:
        The unreachable share and the AbsRel on either side of the split.
    """
    unreachable_hw: Bool[ndarray, "h w"] = periphery_hw & ((target_depth_hw < prompt_range[0]) | (target_depth_hw > prompt_range[1]))
    reachable_hw: Bool[ndarray, "h w"] = periphery_hw & ~unreachable_hw
    periphery_count: int = int(np.count_nonzero(periphery_hw))
    return PeripheryReachability(
        unreachable_fraction=float(np.count_nonzero(unreachable_hw) / periphery_count) if periphery_count else 0.0,
        unreachable_abs_rel=_abs_rel_or_none(prediction_depth_hw, target_depth_hw, unreachable_hw),
        reachable_abs_rel=_abs_rel_or_none(prediction_depth_hw, target_depth_hw, reachable_hw),
    )


def _abs_rel_or_none(
    prediction_depth_hw: Float32[ndarray, "h w"],
    target_depth_hw: Float32[ndarray, "h w"],
    valid_hw: Bool[ndarray, "h w"],
) -> float | None:
    """AbsRel over a pixel subset, or None when it holds fewer than ten valid pixels."""
    try:
        return score_metric_depth(prediction_depth_hw, target_depth_hw, valid_hw).abs_rel
    except ValueError:
        return None


def _mean_or_none(values: list[float | None]) -> float | None:
    """Mean of the present values, or None when every frame lacked the subset."""
    present: list[float] = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _metrics_report(metrics: FootprintSplitMetrics) -> dict[str, float]:
    """Flatten one model's footprint split into ``region_field`` report keys."""
    named: dict[str, MetricCatalogDepthMetrics] = {
        "whole": metrics.whole,
        "inside": metrics.inside,
        "outside": metrics.outside,
        "inside_prompt_upsample": metrics.inside_prompt_upsample,
    }
    return {f"{region}_{field}": getattr(record, field) for region, record in named.items() for field in ("abs_rel", "delta1", "mae")}


def _selected_segments(config: TeacherReferenceConfig, catalog: PromptDACatalog) -> list[str]:
    """Resolve explicit segments or the saved training holdout manifest."""
    if config.segment_ids is None:
        manifest: Path = config.save_dir / "holdout_segments.json"
        try:
            with manifest.open() as file:
                loaded: object = json.load(file)
        except FileNotFoundError:
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


def load_teacher(config: TeacherReferenceConfig) -> PromptDAPredictor:
    """Build PromptDA-L at its released network resolution.

    The lane's ultrawide sample is 768x1024, the same 4:3 aspect as the 1440x1920
    ARKitScenes capture PromptDA-L's 756x1008 network size is derived from, so the
    predictor's own aspect-preserving resize is all that is needed. The padded
    192x256 prompt canvas is passed through untouched.

    Args:
        config: Backend, precision, and batch settings.

    Returns:
        A predictor taking uint8 BHWC frames and float32 B x 192 x 256 prompts.
    """
    image_hw: tuple[int, int] = network_image_hw(PROMPTDA_CAPTURE_HW, config.promptda_max_image_size)
    backend: BackendConfig
    if config.promptda_backend == "torch":
        backend = TorchBackendConfig(autocast=config.promptda_autocast, max_batch_size=config.batch_size)
    else:
        backend = TensorRtBackendConfig(max_batch_size=config.batch_size, opt_batch_size=config.batch_size)
    return PromptDAConfig(model_type="large", backend=backend, image_height=image_hw[0], image_width=image_hw[1]).setup()


def main(config: TeacherReferenceConfig) -> Path:
    """Score PromptDA-L and every student checkpoint on identical ultrawide frames."""
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.height <= 0 or config.width <= 0:
        raise ValueError("height and width must be positive")

    # Catalog-only imports stay local, matching the rest of the evaluation lane.
    from zipdepth.catalog.builders import CpuSampleBuilder
    from zipdepth.catalog.dataset import CatalogPromptDepthDataset

    catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
    segment_ids: list[str] = _selected_segments(config, catalog)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform: AlbumentationsWrapper = build_eval_transform(config.height, config.width)
    policy: UltrawidePolicy = UltrawidePolicy(
        min_valid_fraction=config.ultrawide_min_valid_fraction,
        valid_erosion_px=config.ultrawide_valid_erosion_px,
        prompt_scale=config.ultrawide_prompt_scale,
    )
    placement: PromptPlacement = prompt_placement(policy.prompt_scale)
    inside_hw: Bool[ndarray, "h w"] = footprint_mask(placement, config.height, config.width)

    teacher: PromptDAPredictor = load_teacher(config)
    students: dict[str, ZipDepthPrompt] = {}
    checkpoint: Path
    for checkpoint in config.student_checkpoints:
        name: str = f"{checkpoint.parent.name}/{checkpoint.stem}"
        students[name] = load_zipdepth_prompt(checkpoint).to(device).eval().fuse_for_inference()
        print(f"student {name}: range margin {students[name].range_margin_m:.2f} m")
    teacher_names: list[str] = [TEACHER_NAME, TEACHER_REPLICATED_NAME] if config.include_replicated_teacher else [TEACHER_NAME]
    model_names: list[str] = [*teacher_names, *students]

    covered_ids: list[str] = [segment_id for segment_id in segment_ids if ULTRAWIDE_LAYER in catalog.row_by_id[segment_id].layer_names]
    missing_count: int = len(segment_ids) - len(covered_ids)
    if missing_count:
        print(f"skipping {missing_count} segment(s) without the {ULTRAWIDE_LAYER!r} layer")

    records: dict[str, list[FootprintSplitMetrics]] = {name: [] for name in model_names}
    reachability: dict[str, list[PeripheryReachability]] = {name: [] for name in model_names}
    # PromptDA normalizes and de-normalizes by these, so they explain the two teacher rows.
    canvas_ranges: dict[str, list[tuple[float, float]]] = {name: [] for name in teacher_names}
    per_segment: list[dict[str, object]] = []
    teacher_seconds: float = 0.0
    teacher_frames: int = 0
    dumped_orientations: set[str] = set()

    segment_id: str
    for segment_id in covered_ids:
        orientation: str = catalog.row_by_id[segment_id].orientation
        dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
            catalog.dataset_entry,
            [segment_id],
            catalog.row_by_id,
            device=device,
            builder_factory=lambda: CpuSampleBuilder(transform, min_depth_span=0.0, target_mode="metric", ultrawide_policy=policy),
            shuffle_buffer_size=1,
            frame_stride=config.frame_stride,
            num_producers=1,
            prefetch_samples=4 * config.batch_size,
            cameras="ultrawide",
        )
        loader: DataLoader[dict[str, Tensor]] = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
        segment_records: dict[str, list[FootprintSplitMetrics]] = {name: [] for name in model_names}
        batch: dict[str, Tensor]
        for batch in loader:
            image_bchw: UInt8[Tensor, "b 3 h w"] = batch["image"].to(device=device, non_blocking=True)
            prompt_bchw: Float32[Tensor, "b 1 192 256"] = batch["prompt_depth"].to(device=device, non_blocking=True)
            predictions: dict[str, Float32[ndarray, "b h w"]] = {}

            teacher_prompts: dict[str, Float32[Tensor, "b 1 192 256"]] = {TEACHER_NAME: prompt_bchw}
            if config.include_replicated_teacher:
                teacher_prompts[TEACHER_REPLICATED_NAME] = replicate_prompt_canvas(prompt_bchw, placement)
            rgb_bhwc: UInt8[Tensor, "b h w 3"] = rearrange(image_bchw, "b c h w -> b h w c").contiguous()
            teacher_name: str
            teacher_prompt_bchw: Float32[Tensor, "b 1 192 256"]
            for teacher_name, teacher_prompt_bchw in teacher_prompts.items():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started: float = perf_counter()
                teacher_depth_bhw: Float32[Tensor, "b h w"] = teacher(
                    rgb_bhwc, rearrange(teacher_prompt_bchw, "b 1 h w -> b h w").contiguous()
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if teacher_name == TEACHER_NAME:
                    teacher_seconds += perf_counter() - started
                    teacher_frames += int(image_bchw.shape[0])
                predictions[teacher_name] = teacher_depth_bhw.float().cpu().numpy()

            student_name: str
            student: ZipDepthPrompt
            for student_name, student in students.items():
                with torch.inference_mode():
                    predictions[student_name] = student(image_bchw, prompt_bchw)[:, 0].float().cpu().numpy()

            index: int
            for index in range(int(image_bchw.shape[0])):
                target_depth_hw: Float32[ndarray, "h w"] = batch["target_depth"][index, 0].numpy().astype(np.float32, copy=False)
                target_valid_hw: Bool[ndarray, "h w"] = batch["target_valid"][index, 0].numpy().astype(bool, copy=False)
                prompt_depth_hw: Float32[ndarray, "192 256"] = batch["prompt_depth"][index, 0].numpy().astype(np.float32, copy=False)
                prompt_valid_hw: Bool[ndarray, "192 256"] = batch["prompt_valid"][index, 0].numpy().astype(bool, copy=False)
                scored: dict[str, FootprintSplitMetrics] = {}
                name_iter: str
                for name_iter in model_names:
                    split: FootprintSplitMetrics | None = score_footprint_split(
                        predictions[name_iter][index], target_depth_hw, target_valid_hw, prompt_depth_hw, prompt_valid_hw, inside_hw
                    )
                    if split is None:
                        break
                    scored[name_iter] = split
                if len(scored) != len(model_names):
                    # Every model must score, or the comparison stops being frame-identical.
                    continue
                periphery_hw: Bool[ndarray, "h w"] = target_valid_hw & ~inside_hw
                prompt_range: tuple[float, float] = prompt_output_range(prompt_depth_hw)
                for name_iter in model_names:
                    segment_records[name_iter].append(scored[name_iter])
                    reachability[name_iter].append(
                        score_periphery_reachability(predictions[name_iter][index], target_depth_hw, periphery_hw, prompt_range)
                    )
                for teacher_name in teacher_names:
                    canvas_hw: Float32[Tensor, "192 256"] = teacher_prompts[teacher_name][index, 0]
                    canvas_ranges[teacher_name].append((float(canvas_hw.min().item()), float(canvas_hw.max().item())))
                if config.dump_frames_dir is not None and orientation not in dumped_orientations:
                    dumped_orientations.add(orientation)
                    config.dump_frames_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        config.dump_frames_dir / f"{orientation}_{segment_id}.npz",
                        image_hwc=np.moveaxis(batch["image"][index].numpy(), 0, -1),
                        target_depth_hw=target_depth_hw,
                        target_valid_hw=target_valid_hw,
                        prompt_depth_hw=prompt_depth_hw,
                        **{f"pred_{name_iter.replace('/', '_')}_hw": predictions[name_iter][index] for name_iter in model_names},
                    )

        frame_count: int = len(segment_records[TEACHER_NAME])
        if not frame_count:
            print(f"{segment_id}: no scorable ultrawide frames")
            continue
        segment_report: dict[str, object] = {"segment_id": segment_id, "orientation": orientation, "frame_count": frame_count}
        name_key: str
        for name_key in model_names:
            records[name_key].extend(segment_records[name_key])
            segment_report[name_key] = _metrics_report(mean_footprint_metrics(segment_records[name_key]))
        teacher_mean: FootprintSplitMetrics = mean_footprint_metrics(segment_records[TEACHER_NAME])
        print(
            f"{segment_id} ({orientation}, {frame_count} frames): {TEACHER_NAME} AbsRel "
            f"whole={teacher_mean.whole.abs_rel:.4f} inside={teacher_mean.inside.abs_rel:.4f} outside={teacher_mean.outside.abs_rel:.4f}"
        )
        per_segment.append(segment_report)

    if not records[TEACHER_NAME]:
        raise RuntimeError("no ultrawide frame could be scored")

    splits: dict[str, FootprintSplitMetrics] = {name: mean_footprint_metrics(records[name]) for name in model_names}
    overall: dict[str, dict[str, float | None]] = {}
    name_key2: str
    for name_key2 in model_names:
        entry: dict[str, float | None] = {
            **_metrics_report(splits[name_key2]),
            "periphery_unreachable_fraction": float(np.mean([item.unreachable_fraction for item in reachability[name_key2]])),
            "periphery_unreachable_abs_rel": _mean_or_none([item.unreachable_abs_rel for item in reachability[name_key2]]),
            "periphery_reachable_abs_rel": _mean_or_none([item.reachable_abs_rel for item in reachability[name_key2]]),
        }
        if name_key2 in canvas_ranges:
            entry["prompt_canvas_min_m"] = float(np.mean([pair[0] for pair in canvas_ranges[name_key2]]))
            entry["prompt_canvas_max_m"] = float(np.mean([pair[1] for pair in canvas_ranges[name_key2]]))
        overall[name_key2] = entry
    # The prompt-upsample floor depends on the input only, so every model's split carries the same one.
    floor: MetricCatalogDepthMetrics = mean_footprint_metrics(records[TEACHER_NAME]).inside_prompt_upsample
    overall["prompt-upsample"] = {"inside_abs_rel": floor.abs_rel, "inside_delta1": floor.delta1, "inside_mae": floor.mae}

    seconds_per_frame: float = teacher_seconds / teacher_frames if teacher_frames else 0.0
    report: dict[str, object] = {
        "frame_count": len(records[TEACHER_NAME]),
        "segment_count": len(per_segment),
        "frame_stride": config.frame_stride,
        "promptda_backend": config.promptda_backend,
        "promptda_autocast": config.promptda_autocast if config.promptda_backend == "torch" else None,
        "promptda_network_hw": list(network_image_hw(PROMPTDA_CAPTURE_HW, config.promptda_max_image_size)),
        "sample_hw": [config.height, config.width],
        "batch_size": config.batch_size,
        "promptda_seconds_per_frame": seconds_per_frame,
        "student_checkpoints": [str(path) for path in config.student_checkpoints],
        "overall": overall,
        "per_segment": per_segment,
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    with config.output_json.open("w") as file:
        json.dump(report, file, indent=2)

    print(f"\n{len(records[TEACHER_NAME])} frames over {len(per_segment)} segments")
    print(f"{'model':32s} {'whole':>24s} {'inside':>24s} {'outside':>24s}")
    print(f"{'':32s} " + " ".join(f"{'AbsRel  delta1     MAE':>24s}" for _ in ("whole", "inside", "outside")))
    for name_key2 in model_names:
        cells: str = " ".join(
            f"{record.abs_rel:8.4f} {record.delta1:7.4f} {record.mae:7.4f}"
            for record in (splits[name_key2].whole, splits[name_key2].inside, splits[name_key2].outside)
        )
        print(f"{name_key2:32s} {cells}")
    print(f"{'prompt-upsample (inside only)':32s} {floor.abs_rel:8.4f} {floor.delta1:7.4f} {floor.mae:7.4f}")
    print("\nperiphery reachability (target outside the prompt's own [min, max])")
    for name_key2 in model_names:
        reach: list[PeripheryReachability] = reachability[name_key2]
        print(
            f"{name_key2:32s} unreachable={float(np.mean([item.unreachable_fraction for item in reach])):.4f} "
            f"AbsRel unreachable={_mean_or_none([item.unreachable_abs_rel for item in reach])} "
            f"reachable={_mean_or_none([item.reachable_abs_rel for item in reach])}"
        )
    print("\nprompt canvas range PromptDA normalizes and de-normalizes by")
    for name_key2 in teacher_names:
        print(
            f"{name_key2:32s} min={float(np.mean([pair[0] for pair in canvas_ranges[name_key2]])):.4f} m "
            f"max={float(np.mean([pair[1] for pair in canvas_ranges[name_key2]])):.4f} m"
        )
    print(f"\nPromptDA-L {config.promptda_backend}: {1000.0 * seconds_per_frame:.1f} ms/frame")
    print(f"wrote {config.output_json}")
    return config.output_json


__all__ = (
    "TEACHER_NAME",
    "TEACHER_REPLICATED_NAME",
    "PeripheryReachability",
    "TeacherReferenceConfig",
    "load_teacher",
    "main",
    "prompt_output_range",
    "replicate_prompt_canvas",
    "score_periphery_reachability",
)
