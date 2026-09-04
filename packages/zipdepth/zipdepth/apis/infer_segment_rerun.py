"""Run ZipDepth-PromptDA over one whole catalog segment and log it to Rerun.

The evaluation lane reports one number per segment; this tool shows the frames
behind it. It streams a segment through the **same** sample builders the trainer
and the evaluation use -- eval transform, no augmentation, the ultrawide prompt
scaled into its footprint on a zero canvas -- runs the prompted model on batches
of those samples, and logs every input, prediction, target, and per-frame score
on the capture's own ``video_time`` timeline.

Both cameras are logged in the lane's ``768x1024`` network frame, which is where
the model sees them and where the footprint split is defined. Portrait captures
are rotated to landscape by the builders, so the logged ``Pinhole`` is the stored
calibration carried through the same quarter turns and rescaled to that frame.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
import torch
from arkitscenes_download.ingest.paths import PINHOLE_ULTRAWIDE_RECT, PINHOLE_WIDE, PINHOLE_WIDE_LOWRES, TIMELINE
from arkitscenes_download.schema import DEPTH_RANGE_MM
from jaxtyping import Bool, Float32, Float64, UInt8, UInt16
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rrd_query_utils import first_valid_value
from torch import Tensor
from torch.utils.data import DataLoader

from zipdepth.apis.eval_catalog import (
    FootprintSplitMetrics,
    MetricCatalogDepthMetrics,
    footprint_mask,
    mean_metrics,
    score_footprint_split,
    score_metric_depth,
)
from zipdepth.catalog.builders import CpuSampleBuilder
from zipdepth.catalog.dataset import CatalogPromptDepthDataset
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, PromptDACatalog, SegmentRow, load_promptda_catalog
from zipdepth.catalog.targets import build_eval_transform
from zipdepth.catalog.ultrawide import (
    DEFAULT_ULTRAWIDE_PROMPT_SCALE,
    ULTRAWIDE_LAYER,
    Camera,
    CameraSelection,
    PromptPlacement,
    UltrawidePolicy,
    footprint_box,
    prompt_placement,
)

NETWORK_HW: tuple[int, int] = (768, 1024)
"""Network input and output size the ultrawide lane trains and evaluates at."""
ERROR_RANGE_MM: tuple[float, float] = (0.0, 500.0)
"""Fixed colormap range for the absolute-error images, in millimetres."""
ULTRAWIDE_ROOT: str = f"/{PINHOLE_ULTRAWIDE_RECT}"
"""Entity carrying the rectified ultrawide pinhole and its images."""
WIDE_ROOT: str = f"/{PINHOLE_WIDE}"
"""Entity carrying the wide pinhole and its images."""
WIDE_PROMPT_ENTITY: str = f"/{PINHOLE_WIDE_LOWRES}/prompt"
"""Entity carrying the raw 192x256 ARKit LiDAR prompt."""
METRICS_ROOT: str = "/metrics"
"""Root of the per-frame scalar series."""


@dataclass(slots=True)
class InferSegmentRerunConfig:
    """One-segment ZipDepth-PromptDA demo configuration."""

    rr_config: RerunTyroConfig
    """Viewer, save, connect, and serve behaviour shared across SimpleCV tools."""
    video_id: str
    """Catalog segment (ARKitScenes video id) to run."""
    checkpoint: Path = Path("data/checkpoints/zdpda-uw-v1/final_model.pth")
    """Prompted checkpoint to run; its recorded ``range_margin_m`` is honoured."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset containing the ARKitScenes, PromptDA, and ultrawide layers."""
    cameras: CameraSelection = "both"
    """Cameras to run. Each is streamed and logged as its own pass over the segment."""
    frame_stride: int = 1
    """Keep every Nth chosen frame of each camera; 10 keeps a 1 Hz demo."""
    max_frames: int | None = None
    """Stop each camera after this many logged frames; None runs the whole segment."""
    batch_size: int = 8
    """Frames per forward pass, matching the lane's deployment batch."""


@dataclass(frozen=True, slots=True)
class CameraSummary:
    """Mean per-frame scores for one camera over one segment."""

    camera: Camera
    """Camera the scored frames came from."""
    frame_count: int
    """Frames that had enough valid target pixels to score."""
    regions: dict[str, MetricCatalogDepthMetrics]
    """Mean metrics per region. The wide camera reports ``whole`` only; the
    ultrawide adds the prompt-footprint split and its prompt-upsample floor."""


def demo_blueprint() -> rrb.Blueprint:
    """Lay the demo out as an ultrawide row, a wide row, and the metric series.

    Returns:
        A blueprint with one 2D view per logged image and one time-series view
        over every per-frame scalar.
    """
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=f"{ULTRAWIDE_ROOT}/rgb", name="ultrawide rgb"),
                rrb.Spatial2DView(origin=f"{ULTRAWIDE_ROOT}/prompt_footprint", name="prompt footprint"),
                rrb.Spatial2DView(origin=f"{ULTRAWIDE_ROOT}/depth_pred", name="ultrawide pred"),
                rrb.Spatial2DView(origin=f"{ULTRAWIDE_ROOT}/depth_target", name="ultrawide target"),
                rrb.Spatial2DView(origin=f"{ULTRAWIDE_ROOT}/abs_error", name="ultrawide |error|"),
                name="ultrawide",
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin=f"{WIDE_ROOT}/rgb", name="wide rgb"),
                rrb.Spatial2DView(origin=f"{WIDE_ROOT}/depth_pred", name="wide pred"),
                rrb.Spatial2DView(origin=f"{WIDE_ROOT}/depth_target", name="wide target"),
                rrb.Spatial2DView(origin=f"{WIDE_ROOT}/abs_error", name="wide |error|"),
                name="wide",
            ),
            rrb.TimeSeriesView(origin=METRICS_ROOT, name="per-frame metrics"),
            row_shares=[3.0, 3.0, 2.0],
        ),
        collapse_panels=False,
    )


def summary_table(summaries: list[CameraSummary]) -> str:
    """Render the per-camera, per-region metric means as a fixed-width table.

    Args:
        summaries: One entry per camera that scored at least one frame.

    Returns:
        A printable table, one row per camera and region.
    """
    header: str = f"{'camera':<10} {'region':<22} {'frames':>6} {'AbsRel':>9} {'delta1':>9} {'MAE (m)':>9}"
    lines: list[str] = [header, "-" * len(header)]
    summary: CameraSummary
    for summary in summaries:
        region: str
        metrics: MetricCatalogDepthMetrics
        for region, metrics in summary.regions.items():
            lines.append(
                f"{summary.camera:<10} {region:<22} {summary.frame_count:>6d} "
                f"{metrics.abs_rel:>9.4f} {metrics.delta1:>9.4f} {metrics.mae:>9.4f}"
            )
    return "\n".join(lines)


def depth_mm(depth_m_hw: Float32[ndarray, "h w"]) -> UInt16[ndarray, "h w"]:
    """Quantize metric depth to the ARKitScenes layers' uint16 millimetre encoding."""
    return np.clip(depth_m_hw * 1000.0, 0.0, 65535.0).astype(np.uint16)


def landscape_pinhole(
    dataset_entry: DatasetEntry,
    segment_id: str,
    entity: str,
    quarter_turns: int,
    out_hw: tuple[int, int],
) -> tuple[Float64[ndarray, "3 3"], tuple[int, int]]:
    """Read one static ``Pinhole`` and carry it into the logged landscape frame.

    The builders rotate a stored portrait frame counter-clockwise by
    ``quarter_turns`` and then resize to ``out_hw``; the intrinsics take the same
    two steps. One counter-clockwise turn of a ``width x height`` image maps a
    stored pixel ``(x, y)`` to ``(y, width - 1 - x)``, which swaps the focal
    lengths and moves the principal point. The rescale ignores the half-pixel
    centre convention, a sub-pixel effect at these ratios.

    Args:
        dataset_entry: Connected catalog dataset.
        segment_id: Segment whose calibration is read.
        entity: Entity carrying the ``Pinhole``, without a leading slash.
        quarter_turns: Counter-clockwise turns that bring the frame to landscape.
        out_hw: Size the images are logged at, as ``(height, width)``.

    Returns:
        The float64 ``image_from_camera`` matrix and the logged ``(width, height)``.

    Raises:
        ValueError: If the segment has no ``Pinhole`` on that entity.
    """
    intrinsics_column: str = f"/{entity}:Pinhole:image_from_camera"
    resolution_column: str = f"/{entity}:Pinhole:resolution"
    # The ingest logs the calibration per frame rather than statically, so this
    # reads the timeline and keeps the first row that carries it.
    table: pa.Table = (
        dataset_entry.filter_segments(segment_id)
        .filter_contents([f"/{entity}"])
        .reader(index=TIMELINE, fill_latest_at=True)
        .select(TIMELINE, intrinsics_column, resolution_column)
        .to_arrow_table()
    )
    if table.num_rows == 0:
        raise ValueError(f"segment {segment_id} has no Pinhole on {entity!r}")
    # Rerun stores the 3x3 column-major, so the read side transposes it.
    image_from_camera: Float64[ndarray, "3 3"] = (
        np.asarray(first_valid_value(table.column(intrinsics_column), component_name=intrinsics_column), dtype=np.float64).reshape(3, 3).T
    )
    resolution_wh: Float64[ndarray, "2"] = np.asarray(
        first_valid_value(table.column(resolution_column), component_name=resolution_column), dtype=np.float64
    ).reshape(2)
    width: float = float(resolution_wh[0])
    height: float = float(resolution_wh[1])
    for _ in range(quarter_turns % 4):
        image_from_camera = np.array(
            [
                [image_from_camera[1, 1], 0.0, image_from_camera[1, 2]],
                [0.0, image_from_camera[0, 0], width - 1.0 - image_from_camera[0, 2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        width, height = height, width
    image_from_camera[0] *= out_hw[1] / width
    image_from_camera[1] *= out_hw[0] / height
    return image_from_camera, (out_hw[1], out_hw[0])


def log_camera_calibration(catalog: PromptDACatalog, video_id: str, quarter_turns: int, camera: Camera) -> None:
    """Log one camera's static pinhole, and the ultrawide prompt-footprint outline."""
    entity: str = PINHOLE_ULTRAWIDE_RECT if camera == "ultrawide" else PINHOLE_WIDE
    root: str = ULTRAWIDE_ROOT if camera == "ultrawide" else WIDE_ROOT
    pinhole: tuple[Float64[ndarray, "3 3"], tuple[int, int]] = landscape_pinhole(
        catalog.dataset_entry, video_id, entity, quarter_turns, NETWORK_HW
    )
    rr.log(root, rr.Pinhole(image_from_camera=pinhole[0], resolution=pinhole[1], camera_xyz=rr.ViewCoordinates.RDF), static=True)
    if camera != "ultrawide":
        return
    box: tuple[int, int, int, int] = footprint_box(prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE), NETWORK_HW)
    rr.log(
        f"{root}/rgb/prompt_footprint_box",
        rr.Boxes2D(
            mins=[[box[1], box[0]]],
            sizes=[[box[3] - box[1], box[2] - box[0]]],
            colors=[[255, 214, 0]],
            labels=["wide prompt footprint"],
        ),
        static=True,
    )


def run_camera(
    config: InferSegmentRerunConfig,
    catalog: PromptDACatalog,
    model: ZipDepthPrompt,
    device: torch.device,
    camera: Camera,
) -> CameraSummary | None:
    """Stream one camera's frames of the segment, predict, log, and score them.

    Args:
        config: Segment, checkpoint, and streaming settings.
        catalog: Connected catalog metadata.
        model: Fused prompted model in evaluation mode.
        device: Device the model and the segment decoder run on.
        camera: Camera streamed on this pass.

    Returns:
        The camera's mean metrics, or None when no frame could be scored.
    """
    # Evaluation policy: score every frame (a sparse-frame drop is a training
    # data-efficiency choice), with the training lane's mask erosion.
    policy: UltrawidePolicy = UltrawidePolicy(min_valid_fraction=0.0, valid_erosion_px=1, prompt_scale=DEFAULT_ULTRAWIDE_PROMPT_SCALE)
    placement: PromptPlacement = prompt_placement(policy.prompt_scale)
    inside_hw: Bool[ndarray, "h w"] = footprint_mask(placement, NETWORK_HW[0], NETWORK_HW[1])
    root: str = ULTRAWIDE_ROOT if camera == "ultrawide" else WIDE_ROOT
    dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
        catalog.dataset_entry,
        [config.video_id],
        catalog.row_by_id,
        device=device,
        builder_factory=lambda: CpuSampleBuilder(
            build_eval_transform(NETWORK_HW[0], NETWORK_HW[1]),
            min_depth_span=0.0,
            target_mode="metric",
            ultrawide_policy=policy,
        ),
        shuffle_buffer_size=1,
        frame_stride=config.frame_stride,
        num_producers=1,
        prefetch_samples=4 * config.batch_size,
        cameras=camera,
        emit_timestamps=True,
    )
    loader: DataLoader[dict[str, Tensor]] = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)

    records: dict[str, list[MetricCatalogDepthMetrics]] = {}
    logged_count: int = 0
    started: float = perf_counter()
    batch: dict[str, Tensor]
    for batch in loader:
        if config.max_frames is not None and logged_count >= config.max_frames:
            break
        image_bchw: UInt8[Tensor, "b 3 h w"] = batch["image"].to(device=device, non_blocking=True)
        prompt_depth_bchw: Float32[Tensor, "b 1 192 256"] = batch["prompt_depth"].to(device=device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction_bhw: Float32[ndarray, "b h w"] = model(image_bchw, prompt_depth_bchw)[:, 0].float().cpu().numpy()

        index: int
        for index in range(prediction_bhw.shape[0]):
            if config.max_frames is not None and logged_count >= config.max_frames:
                break
            prediction_hw: Float32[ndarray, "h w"] = prediction_bhw[index]
            target_depth_hw: Float32[ndarray, "h w"] = batch["target_depth"][index, 0].numpy().astype(np.float32, copy=False)
            target_valid_hw: Bool[ndarray, "h w"] = batch["target_valid"][index, 0].numpy().astype(bool, copy=False)
            prompt_depth_hw: Float32[ndarray, "192 256"] = batch["prompt_depth"][index, 0].numpy().astype(np.float32, copy=False)
            prompt_valid_hw: Bool[ndarray, "192 256"] = batch["prompt_valid"][index, 0].numpy().astype(bool, copy=False)
            absolute_error_hw: Float32[ndarray, "h w"] = np.where(target_valid_hw, np.abs(prediction_hw - target_depth_hw), 0.0).astype(
                np.float32
            )

            rr.set_time(TIMELINE, duration=np.timedelta64(int(batch["video_time_ns"][index].item()), "ns"))
            rgb_hwc: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(np.moveaxis(batch["image"][index].numpy(), 0, -1))
            rr.log(f"{root}/rgb", rr.Image(rgb_hwc).compress(jpeg_quality=90))
            rr.log(f"{root}/depth_pred", rr.DepthImage(depth_mm(prediction_hw), meter=1000.0, depth_range=DEPTH_RANGE_MM))
            rr.log(
                f"{root}/depth_target",
                rr.DepthImage(depth_mm(np.where(target_valid_hw, target_depth_hw, 0.0)), meter=1000.0, depth_range=DEPTH_RANGE_MM),
            )
            rr.log(
                f"{root}/abs_error",
                rr.DepthImage(depth_mm(absolute_error_hw), meter=1000.0, depth_range=ERROR_RANGE_MM, colormap="turbo"),
            )
            prompt_entity: str = f"{root}/prompt_footprint" if camera == "ultrawide" else WIDE_PROMPT_ENTITY
            rr.log(prompt_entity, rr.DepthImage(depth_mm(prompt_depth_hw), meter=1000.0, depth_range=DEPTH_RANGE_MM))

            frame_regions: dict[str, MetricCatalogDepthMetrics] = {}
            if camera == "ultrawide":
                split: FootprintSplitMetrics | None = score_footprint_split(
                    prediction_hw, target_depth_hw, target_valid_hw, prompt_depth_hw, prompt_valid_hw, inside_hw
                )
                if split is not None:
                    frame_regions = {
                        "whole": split.whole,
                        "inside": split.inside,
                        "outside": split.outside,
                        "inside_prompt_upsample": split.inside_prompt_upsample,
                    }
            else:
                try:
                    frame_regions = {"whole": score_metric_depth(prediction_hw, target_depth_hw, target_valid_hw)}
                except ValueError:
                    frame_regions = {}
            region: str
            metrics: MetricCatalogDepthMetrics
            for region, metrics in frame_regions.items():
                records.setdefault(region, []).append(metrics)
                rr.log(f"{METRICS_ROOT}/{camera}/abs_rel/{region}", rr.Scalars(metrics.abs_rel))
                rr.log(f"{METRICS_ROOT}/{camera}/delta1/{region}", rr.Scalars(metrics.delta1))
                rr.log(f"{METRICS_ROOT}/{camera}/mae_m/{region}", rr.Scalars(metrics.mae))
            logged_count += 1
            if logged_count % 25 == 0:
                print(f"{config.video_id} {camera}: logged {logged_count} frames in {perf_counter() - started:.1f}s")

    print(f"{config.video_id} {camera}: {logged_count} frames logged in {perf_counter() - started:.1f}s ({dataset.stats.samples_built} built)")
    if not records:
        print(f"{config.video_id} {camera}: no scorable frames")
        return None
    return CameraSummary(
        camera=camera,
        frame_count=len(records["whole"]),
        regions={region: mean_metrics(values) for region, values in records.items()},
    )


def main(config: InferSegmentRerunConfig) -> None:
    """Run the configured cameras of one segment into Rerun and report the means.

    Raises:
        ValueError: If a streaming setting is not positive.
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If no camera produced a scorable frame.
    """
    if config.frame_stride <= 0 or config.batch_size <= 0:
        raise ValueError("frame_stride and batch_size must be positive")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames must be positive when set")
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"prompted checkpoint is not a file: {config.checkpoint}")
    catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
    catalog.require_segments([config.video_id])
    segment_row: SegmentRow = catalog.row_by_id[config.video_id]
    cameras: list[Camera] = ["wide", "ultrawide"]
    if config.cameras == "wide":
        cameras = ["wide"]
    elif config.cameras == "ultrawide":
        cameras = ["ultrawide"]
    if "ultrawide" in cameras and ULTRAWIDE_LAYER not in segment_row.layer_names:
        print(f"{config.video_id}: no {ULTRAWIDE_LAYER!r} layer, running the wide camera only")
        cameras = [camera for camera in cameras if camera != "ultrawide"]
    if not cameras:
        raise RuntimeError(f"{config.video_id}: no camera left to run")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: ZipDepthPrompt = load_zipdepth_prompt(config.checkpoint).to(device).eval().fuse_for_inference()
    print(f"{config.video_id}: {segment_row.orientation} capture, {config.checkpoint}, range margin {model.range_margin_m:.2f} m on {device}")
    # Same rule the dataset applies: a stored portrait frame is turned back to landscape.
    quarter_turns: int = 0 if segment_row.orientation == "landscape" else (-segment_row.orientation_quarter_turns_ccw) % 4

    rr.send_blueprint(demo_blueprint())
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    summaries: list[CameraSummary] = []
    camera: Camera
    for camera in cameras:
        log_camera_calibration(catalog, config.video_id, quarter_turns, camera)
        summary: CameraSummary | None = run_camera(config, catalog, model, device, camera)
        if summary is not None:
            summaries.append(summary)
    if not summaries:
        raise RuntimeError(f"{config.video_id}: no camera produced a scorable frame")

    print(summary_table(summaries))
    if config.rr_config.save is None:
        return
    report: dict[str, object] = {
        "video_id": config.video_id,
        "checkpoint": str(config.checkpoint),
        "cameras": config.cameras,
        "frame_stride": config.frame_stride,
        "orientation": segment_row.orientation,
        "summaries": [
            {
                "camera": summary.camera,
                "frame_count": summary.frame_count,
                "regions": {region: asdict(metrics) for region, metrics in summary.regions.items()},
            }
            for summary in summaries
        ],
    }
    report_path: Path = config.rr_config.save.with_suffix(".summary.json")
    with report_path.open("w") as file:
        json.dump(report, file, indent=2)
    print(f"wrote {report_path}")
