"""Per-scene comparison rows for the side-by-side recording: our numbers and the published ones from the benchmark
table (`runs/TABLE.json`, written by the hexels package), viewer timing from `gsplat probe` / `gsplat speed`."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json

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
    ("renderer_ms_median", "renderer ms p50"),
    ("fps", "test-view FPS"),
    ("train_min", "train min"),
    ("train_vram_gib", "train GiB"),
    ("deploy_mb", "model MB"),
    ("primitives", "primitives"),
)


def read_json[Record](cls: type[Record], path: Path) -> Record:
    """Decode a JSON record; malformed input names the file."""
    try:
        return from_json(cls, path.read_text())
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


@serde
@dataclass(frozen=True, slots=True)
class PublishedCells:
    """`paper` of a TABLE.json scene row: the published 30k checkpoint of the same variant (read partially)."""

    psnr: float
    """PSNR, dB."""
    ssim: float
    """SSIM."""
    lpips_vgg: float
    """LPIPS-VGG."""
    deploy_mb: float
    """Learned FP32 bytes / 1e6."""
    primitives: int
    """Hexels."""
    train_min: float | None
    """Training minutes on the authors' A5000 (30k steps)."""


@serde
@dataclass(frozen=True, slots=True)
class SceneCells:
    """One (dataset, scene, method) row of TABLE.json (read partially); unscored cells are null."""

    dataset: str
    """Run-layout dataset."""
    scene: str
    """Scene name."""
    method: str
    """Run method."""
    psnr: float | None
    """Mean PSNR, dB (shared scorer)."""
    ssim: float | None
    """Mean SSIM."""
    lpips_vgg: float | None
    """Mean LPIPS-VGG."""
    fps: float | None
    """Paper-protocol render FPS over the test views."""
    train_min: float | None
    """Optimisation loop minutes."""
    vram_gib: float | None
    """Peak training VRAM, GiB."""
    deploy_mb: float | None
    """Primitives x FP32 bytes / 1e6."""
    paper: PublishedCells | None
    """Published checkpoint of the same variant (Hexels rows)."""


@serde
@dataclass(frozen=True, slots=True)
class BenchTable:
    """`runs/TABLE.json` (read partially): one row per protocol scene and method, complete or not."""

    scenes: list[SceneCells]
    """Per-scene rows."""


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


@serde
@dataclass(frozen=True, slots=True)
class SpeedBackend:
    """One renderer in a `gsplat speed --out` report (read partially)."""

    implementation: str
    """`ours` (gsplat-core), `brush` or `native`."""
    median_of_repeat_medians_ms: float
    """Renderer frame time, median of the repeat medians."""


@serde
@dataclass(frozen=True, slots=True)
class SpeedReport:
    """`gsplat speed --out` report (read partially)."""

    backends: list[SpeedBackend]
    """Measured renderers."""


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
    renderer_ms_median: float | None = None
    """`gsplat speed --impl ours` on the orbit (renderer only), median of repeat medians, ms."""


def scene_cells(table: BenchTable, dataset: str, scene: str, method: str) -> SceneCells | None:
    """The TABLE.json row of one run, None when the table has no such method."""
    return next((row for row in table.scenes if (row.dataset, row.scene, row.method) == (dataset, scene, method)), None)


def run_row(table: BenchTable, cells: SceneCells, name: str, primitives: int, probe: Path | None, speed: Path | None) -> MetricRow:
    """Our row for one run: its TABLE.json cells (null until the shared scorer scored it) and viewer timing.

    A run without its own render FPS (Brush renders its test views in the trainer) takes it from `<method>-wgsl`,
    the same model rendered by gsplat-core.

    Args:
        table: The benchmark table.
        cells: The run's row of `table`.
        name: Display name of the model.
        primitives: Primitive count of the model shown.
        probe: Optional `gsplat probe` report for this model.
        speed: Optional `gsplat speed` report for this model.
    """
    rerender: SceneCells | None = scene_cells(table, cells.dataset, cells.scene, f"{cells.method}-wgsl")
    fps: float | None = cells.fps if cells.fps is not None or rerender is None else rerender.fps
    viewer_ms: Float64[ndarray, "n"] | None = None
    viewer_gpu_mb: float | None = None
    if probe is not None:
        report: ProbeReport = read_json(ProbeReport, probe)
        viewer_ms = np.array([frame.ms for frame in report.frames], dtype=np.float64)
        viewer_gpu_mb = (report.gpu_bytes_scene + report.gpu_bytes_views) / 1e6
    ours: list[SpeedBackend] = [b for b in read_json(SpeedReport, speed).backends if b.implementation == "ours"] if speed is not None else []
    return MetricRow(
        method=name,
        source="ours, shared-png16" if cells.psnr is not None else "unscored",
        psnr=cells.psnr,
        ssim=cells.ssim,
        lpips_vgg=cells.lpips_vgg,
        fps=fps,
        train_min=cells.train_min,
        train_vram_gib=cells.vram_gib,
        deploy_mb=cells.deploy_mb,
        primitives=primitives,
        viewer_ms_median=None if viewer_ms is None else float(np.median(viewer_ms)),
        viewer_ms_p95=None if viewer_ms is None else float(np.percentile(viewer_ms, 95.0)),
        viewer_gpu_mb=viewer_gpu_mb,
        renderer_ms_median=ours[0].median_of_repeat_medians_ms if ours else None,
    )


def paper_row(cells: SceneCells, name: str) -> MetricRow:
    """The authors' published per-scene numbers for the checkpoint beside a Hexels row."""
    if cells.paper is None:
        raise ValueError(f"TABLE.json has no published checkpoint beside {cells.dataset}/{cells.scene}/{cells.method}")
    return MetricRow(
        method=name,
        source="published, A5000",
        psnr=cells.paper.psnr,
        ssim=cells.paper.ssim,
        lpips_vgg=cells.paper.lpips_vgg,
        train_min=cells.paper.train_min,
        deploy_mb=cells.paper.deploy_mb,
        primitives=cells.paper.primitives,
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
