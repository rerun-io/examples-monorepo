"""Per-scene comparison rows (ours and published) read from the run layout, logged as Rerun data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from serde import serde

from gsplat_rust_renderer.nvs_protocol import read_json

RunDataset: TypeAlias = Literal["mipnerf360", "dtu", "tnt"]
PAPER_DATASET: dict[RunDataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tanks_and_temples"}
# FP32 learned state per primitive (contract run-layout.md, Deploy MB).
PRIMITIVE_BYTES: dict[str, int] = {"brush": 236, "v1": 80, "v2": 116, "v3": 260}
METRICS_ENTITY: str = "/metrics/table"
METRICS_TIMELINE: str = "row"
# Table rows in display order. The label column sorts first: the dataframe orders columns by name.
# The table sizes columns to their headers, so the header is as wide as the longest label.
METRIC_LABEL: str = "(metric and unit)"
METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("psnr", "PSNR dB"),
    ("ssim", "SSIM"),
    ("lpips_vgg", "LPIPS-VGG"),
    ("viewer_ms_median", "viewer ms p50"),
    ("viewer_ms_p95", "viewer ms p95"),
    ("viewer_gpu_mb", "viewer GPU MB"),
    ("fps", "render FPS"),
    ("train_min", "train min"),
    ("train_vram_gib", "train GiB"),
    ("deploy_mb", "model MB"),
    ("primitives", "primitives"),
)


@serde
@dataclass(frozen=True, slots=True)
class RunMetrics:
    """`metrics.json` / `published_metrics.json` of one run (read partially)."""

    psnr: float | None
    """Mean PSNR over the test views, dB."""
    ssim: float | None
    """Mean SSIM."""
    lpips_vgg: float | None
    """Mean LPIPS-VGG."""
    scorer: str
    """`shared-png16` or `upstream-float`."""
    fps: float | None = None
    """Render FPS under the run-layout protocol, when measured."""


@serde
@dataclass(frozen=True, slots=True)
class TrainRecord:
    """`train.json` of a run we trained (read partially)."""

    loop_seconds: float
    """Optimisation loop wall time."""
    peak_vram_gib: float | None
    """Peak training process GPU memory."""
    primitives: int
    """Final primitive count."""


@serde
@dataclass(frozen=True, slots=True)
class BrushSummary:
    """`brush-summary.json` (Brush's own 8-bit eval at the last step, read partially)."""

    brush_eval_psnr: float
    """Brush in-process eval PSNR."""
    brush_eval_ssim: float
    """Brush in-process eval SSIM."""


@serde
@dataclass(frozen=True, slots=True)
class ParitySummary:
    """WGSL render of a published model against GT."""

    mean_psnr_wgsl_gt: float
    """Mean PSNR of our WGSL renders against GT, dB."""


@serde
@dataclass(frozen=True, slots=True)
class ParityReport:
    """`parity.json` of a `-pub-wgsl` run (read partially)."""

    summary: ParitySummary
    """Means over the test views."""


@serde
@dataclass(frozen=True, slots=True)
class PaperScene:
    """One published checkpoint in `paper-numbers.json` `hf_nvs_per_scene` (authors' A5000 numbers)."""

    psnr: float
    """Mean test PSNR, dB."""
    ssim: float
    """Mean test SSIM."""
    lpips_vgg: float
    """Mean test LPIPS-VGG."""
    primitives: int
    """Primitive count."""
    learned_model_bytes: int
    """FP32 learned state in bytes."""
    train_seconds_summary: float | None
    """Training time, seconds."""
    peak_gpu_used_gb_legacy: float | None
    """Peak GPU memory during training."""


@serde
@dataclass(frozen=True, slots=True)
class PaperNumbers:
    """`specs/paper-numbers.json` (read partially)."""

    hf_nvs_per_scene: dict[str, dict[str, dict[str, PaperScene]]]
    """dataset -> scene -> `v{1,2,3}-{fixed,grown}` -> published numbers."""


@serde
@dataclass(frozen=True, slots=True)
class ProbeFrame:
    """One measured `gsplat probe` frame (read partially)."""

    ms: float
    """Full headless viewer frame, milliseconds."""


@serde
@dataclass(frozen=True, slots=True)
class ProbeReport:
    """`gsplat probe --out` report (read partially)."""

    adapter: str
    """GPU adapter name."""
    frames: list[ProbeFrame]
    """Measured frames, whole 300-frame windows."""
    gpu_bytes_scene: int
    """Compute scene uploads, bytes."""
    gpu_bytes_views: int
    """Compute per-view scratch and targets, bytes."""


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One table row; `None` cells stay empty."""

    method: str
    """Model name."""
    source: str
    """Where the numbers come from (scorer or publication)."""
    psnr: float | None = None
    """Test PSNR, dB."""
    ssim: float | None = None
    """Test SSIM."""
    lpips_vgg: float | None = None
    """Test LPIPS-VGG."""
    train_min: float | None = None
    """Training loop minutes."""
    train_vram_gib: float | None = None
    """Peak training GPU memory, GiB."""
    fps: float | None = None
    """Renderer FPS over the test views (run-layout protocol)."""
    deploy_mb: float | None = None
    """FP32 learned state, MB."""
    primitives: int | None = None
    """Primitive count."""
    viewer_ms_median: float | None = None
    """`gsplat probe` full viewer frame on the orbit, median ms."""
    viewer_ms_p95: float | None = None
    """`gsplat probe` full viewer frame on the orbit, 95th percentile ms."""
    viewer_gpu_mb: float | None = None
    """Compute renderer GPU memory in the viewer (scene + view), MB."""


def read_record[Record](cls: type[Record], path: Path) -> Record | None:
    """Decode an optional JSON record; a missing file is `None`, a malformed one names the file."""
    return read_json(cls, path) if path.is_file() else None


def probe_cells(path: Path | None) -> dict[str, float]:
    """Median/p95 frame time and compute GPU memory from a `gsplat probe` report."""
    report: ProbeReport | None = read_record(ProbeReport, path) if path is not None else None
    if report is None:
        return {}
    frame_ms: Float64[ndarray, "n"] = np.array([frame.ms for frame in report.frames], dtype=np.float64)
    return {
        "viewer_ms_median": float(np.median(frame_ms)),
        "viewer_ms_p95": float(np.percentile(frame_ms, 95.0)),
        "viewer_gpu_mb": (report.gpu_bytes_scene + report.gpu_bytes_views) / 1e6,
    }


def run_row(run_dir: Path, method: str, primitive: str, primitives: int, probe: Path | None = None) -> MetricRow:
    """Our row for one run directory. PSNR source order: shared scorer, WGSL parity, Brush's own eval.

    Args:
        run_dir: `runs/<dataset>/<scene>/<method>`.
        method: display name.
        primitive: key of PRIMITIVE_BYTES (`brush`, `v1`, `v2`, `v3`).
        primitives: primitive count of the model shown.
        probe: optional `gsplat probe` report for this model.
    """
    metrics: RunMetrics | None = read_record(RunMetrics, run_dir / "metrics.json")
    parity: ParityReport | None = read_record(ParityReport, run_dir / "parity.json")
    brush: BrushSummary | None = read_record(BrushSummary, run_dir / "brush-summary.json")
    train: TrainRecord | None = read_record(TrainRecord, run_dir / "train.json")
    psnr: float | None = None
    ssim: float | None = None
    lpips_vgg: float | None = None
    fps: float | None = None
    source: str = "unscored"
    if metrics is not None:
        psnr, ssim, lpips_vgg, fps, source = metrics.psnr, metrics.ssim, metrics.lpips_vgg, metrics.fps, f"ours, {metrics.scorer}"
    elif parity is not None:
        psnr, source = parity.summary.mean_psnr_wgsl_gt, "ours, WGSL vs GT"
    elif brush is not None:
        psnr, ssim, source = brush.brush_eval_psnr, brush.brush_eval_ssim, "ours, Brush eval 8-bit"
    return MetricRow(
        method=method,
        source=source,
        psnr=psnr,
        ssim=ssim,
        lpips_vgg=lpips_vgg,
        fps=fps,
        train_min=train.loop_seconds / 60.0 if train is not None else None,
        train_vram_gib=train.peak_vram_gib if train is not None else None,
        deploy_mb=primitives * PRIMITIVE_BYTES[primitive] / 1e6,
        primitives=primitives,
        **probe_cells(probe),
    )


def paper_row(paper_numbers: Path, dataset: RunDataset, scene: str, variant: str, method: str) -> MetricRow:
    """The authors' published per-scene numbers for one checkpoint (`v1-fixed`, ...)."""
    numbers: PaperNumbers | None = read_record(PaperNumbers, paper_numbers)
    if numbers is None:
        raise ValueError(f"Missing {paper_numbers}")
    published: PaperScene = numbers.hf_nvs_per_scene[PAPER_DATASET[dataset]][scene][variant]
    return MetricRow(
        method=method,
        source="published, A5000",
        psnr=published.psnr,
        ssim=published.ssim,
        lpips_vgg=published.lpips_vgg,
        train_min=published.train_seconds_summary / 60.0 if published.train_seconds_summary is not None else None,
        train_vram_gib=published.peak_gpu_used_gb_legacy,
        deploy_mb=published.learned_model_bytes / 1e6,
        primitives=published.primitives,
    )


def log_metric_table(rows: list[MetricRow], recording: rr.RecordingStream | None = None) -> None:
    """A paper-style table: one row per metric on the `row` sequence timeline, one column per model.

    Column names are `<model> · <source>`; a metric no model has is left out, a missing cell stays empty.
    """
    shown: list[tuple[str, str]] = [(name, label) for name, label in METRIC_ROWS if any(getattr(row, name) is not None for row in rows)]
    for index, (name, label) in enumerate(shown):
        values: rr.AnyValues = rr.AnyValues().with_component_from_data(METRIC_LABEL, label)
        for row in rows:
            value: float | int | None = getattr(row, name)
            if value is not None:
                values = values.with_component_from_data(f"{row.method} · {row.source}", round(float(value), 4))
        rr.set_time(METRICS_TIMELINE, sequence=index, recording=recording)
        rr.log(METRICS_ENTITY, values, recording=recording)
    rr.reset_time(recording=recording)
