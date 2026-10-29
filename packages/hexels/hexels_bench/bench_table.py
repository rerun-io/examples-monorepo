"""Benchmark table: `runs/` -> `TABLE.json` (the paper's Table 2/3 columns, the published numbers beside ours)
and a Rerun metrics recording `table.rrd`.

Per-scene rows carry only what that run measured (Brush training has no FPS of its own: `brush-7k-wgsl`
re-renders the same PLY and carries it). Hexels rows sit beside the published 30k checkpoint of the same
variant on the same scene (HF catalog, `hexels_bench.paper` `hf_nvs_per_scene`). Dataset means average the
representative scenes (`layout.REPRESENTATIVE_SCENES`) and sit beside the mean of the published per-scene values over
those same scenes, never beside the paper's 34-scene dataset means.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

import rerun as rr
import rerun.blueprint as rrb
from serde import serde
from serde.json import to_json

from hexels_bench.bench import incomplete_reason
from hexels_bench.layout import PROTOCOL_SCENES, REPRESENTATIVE_SCENES, UPSTREAM_DATASET, Dataset, read_json
from hexels_bench.methods import BENCHMARK_METHODS, METHODS, Method
from hexels_bench.metrics import MetricsFile, TrainRecord, ply_vertex_count
from hexels_bench.paper import PaperNumbers, PublishedNvs

PAPER_DATASET_ORDER: tuple[Dataset, ...] = ("tnt", "mipnerf360", "dtu")
"""The paper's column order (Table 2/3): T&T, Mip-NeRF 360, DTU."""
PUBLISHED_TOLERANCE_DB: float = 0.05
"""Published 30k checkpoints re-rendered by us must match the paper's per-scene PSNR this closely."""
Status: TypeAlias = Literal["complete", "incomplete", "missing"]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PaperScene:
    """The published 30k checkpoint beside one of our scene rows."""

    model: str
    """Published variant, e.g. v1-fixed."""
    views: int
    """Held-out views."""
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
    """Training minutes of the authors' run (30k steps; hardware in `BenchTable.published_hardware`)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SceneRow:
    """One (dataset, scene, method) run."""

    dataset: Dataset
    """Run-layout dataset."""
    scene: str
    """Scene name."""
    method: Method
    """Run method."""
    status: Status
    """complete = scored on every held-out view; incomplete = started; missing = no run directory."""
    reason: str | None
    """Why the run is not complete."""
    views: int | None
    """Scored views."""
    train_min: float | None
    """Optimisation loop minutes (train.json loop_seconds)."""
    total_min: float | None
    """Training process minutes (loading and export included)."""
    vram_gib: float | None
    """Peak training VRAM, GiB."""
    fps: float | None
    """Paper-protocol render FPS."""
    psnr: float | None
    """Mean PSNR, dB (shared scorer)."""
    ssim: float | None
    """Mean SSIM."""
    lpips_vgg: float | None
    """Mean LPIPS-VGG."""
    deploy_mb: float | None
    """Primitives x FP32 bytes / 1e6."""
    ply_mb: float | None
    """model.ply size, MB."""
    primitives: int | None
    """Primitives in model.ply."""
    sfm_points: int | None
    """Scene SfM points (the cap)."""
    paper: PaperScene | None
    """Published checkpoint of the same variant (Hexels rows)."""
    delta_psnr: float | None
    """Ours minus published PSNR, dB."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PaperMean:
    """Mean of the published per-scene values over the scenes of one of our means."""

    model: str
    """Published variant, e.g. v3-fixed."""
    scenes: int
    """Scenes averaged."""
    psnr: float
    """PSNR, dB."""
    ssim: float
    """SSIM."""
    lpips_vgg: float
    """LPIPS-VGG."""
    deploy_mb: float
    """Learned FP32 MB."""
    primitives_k: float
    """Hexels, thousands."""
    train_min: float | None
    """Training minutes of the authors' runs (30k steps; hardware in `BenchTable.published_hardware`)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MeanRow:
    """Mean of one method over the representative scenes of a dataset that it completed."""

    dataset: Dataset
    """Run-layout dataset."""
    method: Method
    """Run method."""
    scenes: int
    """Complete scenes averaged."""
    total: int
    """Representative scenes of the dataset."""
    complete: bool
    """Every representative scene is complete."""
    train_min: float | None
    """Mean optimisation loop minutes."""
    total_min: float | None
    """Mean training process minutes."""
    vram_gib: float | None
    """Mean peak training VRAM, GiB."""
    fps: float | None
    """Mean of the per-scene median FPS (the paper's aggregation)."""
    psnr: float | None
    """Mean PSNR, dB."""
    ssim: float | None
    """Mean SSIM."""
    lpips_vgg: float | None
    """Mean LPIPS-VGG."""
    deploy_mb: float | None
    """Mean deploy MB."""
    ply_mb: float | None
    """Mean model.ply MB."""
    primitives_k: float | None
    """Mean primitives, thousands."""
    paper: PaperMean | None
    """Published per-scene values of the same variant, averaged over the same representative scenes."""
    delta_psnr: float | None
    """Ours minus published PSNR, dB (meaningful when `complete`)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BenchTable:
    """`runs/TABLE.json`."""

    format: str
    """Schema name."""
    generated: str
    """Local ISO time."""
    runs_root: str
    """Runs directory read."""
    published_hardware: str
    """GPU of the published numbers as paper-numbers.json records it (`source.hardware_paper`)."""
    mean_scenes: dict[str, list[str]]
    """Scenes each dataset mean averages (`layout.REPRESENTATIVE_SCENES` by default)."""
    scenes: list[SceneRow]
    """Per-scene rows, datasets in queue order, scenes in upstream order, methods in queue order."""
    means: list[MeanRow]
    """Per-dataset means, datasets in the paper's column order."""
    flags: list[str]
    """Numbers outside the expected band (published checkpoints vs paper, view counts)."""


def mean(values: list[float | None]) -> float | None:
    present: list[float] = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def published_scene(paper: PaperNumbers, dataset: Dataset, scene: str, method: Method) -> PaperScene | None:
    found: str | None = METHODS[method].variant
    model: PublishedNvs | None = paper.hf_nvs_per_scene.get(UPSTREAM_DATASET[dataset], {}).get(scene, {}).get(f"{found}-fixed") if found else None
    if model is None:
        return None
    seconds: float | None = model.train_seconds_summary
    return PaperScene(
        model=f"{found}-fixed",
        views=model.views,
        psnr=model.psnr,
        ssim=model.ssim,
        lpips_vgg=model.lpips_vgg,
        deploy_mb=model.learned_model_bytes / 1e6,
        primitives=model.primitives,
        train_min=None if seconds is None else seconds / 60.0,
    )


def scene_row(runs: Path, dataset: Dataset, scene: str, method: Method, paper: PaperNumbers) -> SceneRow:
    run_dir: Path = runs / dataset / scene / method
    published: PaperScene | None = published_scene(paper, dataset, scene, method)
    reason: str | None = incomplete_reason(run_dir, method) if run_dir.is_dir() else "no run directory"
    status: Status = "complete" if reason is None else ("incomplete" if run_dir.is_dir() else "missing")
    metrics: MetricsFile | None = read_json(MetricsFile, run_dir / "metrics.json") if status == "complete" else None
    train: TrainRecord | None = read_json(TrainRecord, run_dir / "train.json") if metrics is not None and (run_dir / "train.json").is_file() else None
    primitives: int | None = None
    if metrics is not None:
        primitives = train.primitives if train is not None else ply_vertex_count(run_dir / "model.ply")
    return SceneRow(
        dataset=dataset,
        scene=scene,
        method=method,
        status=status,
        reason=reason,
        views=None if metrics is None else metrics.views,
        train_min=None if train is None else train.loop_seconds / 60.0,
        total_min=None if train is None else train.total_seconds / 60.0,
        vram_gib=None if train is None else train.peak_vram_gib,
        fps=None if metrics is None else metrics.fps,
        psnr=None if metrics is None else metrics.psnr,
        ssim=None if metrics is None else metrics.ssim,
        lpips_vgg=None if metrics is None else metrics.lpips_vgg,
        deploy_mb=None if metrics is None else metrics.deploy_mb,
        ply_mb=None if metrics is None else metrics.ply_mb,
        primitives=primitives,
        sfm_points=paper.sfm_points.get(UPSTREAM_DATASET[dataset], {}).get(scene),
        paper=published,
        delta_psnr=None if metrics is None or published is None else metrics.psnr - published.psnr,
    )


def mean_row(dataset: Dataset, method: Method, rows: list[SceneRow]) -> MeanRow:
    """Mean of `rows` (one method, the representative scenes of one dataset) over its complete scenes."""
    done: list[SceneRow] = [row for row in rows if row.status == "complete"]
    published: list[PaperScene] = [row.paper for row in rows if row.paper is not None]
    paper_mean: PaperMean | None = None
    if published and len(published) == len(rows):
        paper_mean = PaperMean(
            model=published[0].model,
            scenes=len(published),
            psnr=sum(scene.psnr for scene in published) / len(published),
            ssim=sum(scene.ssim for scene in published) / len(published),
            lpips_vgg=sum(scene.lpips_vgg for scene in published) / len(published),
            deploy_mb=sum(scene.deploy_mb for scene in published) / len(published),
            primitives_k=sum(scene.primitives for scene in published) / len(published) / 1000.0,
            train_min=mean([scene.train_min for scene in published]),
        )
    psnr: float | None = mean([row.psnr for row in done])
    primitives: float | None = mean([None if row.primitives is None else float(row.primitives) for row in done])
    return MeanRow(
        dataset=dataset,
        method=method,
        scenes=len(done),
        total=len(rows),
        complete=len(done) == len(rows),
        train_min=mean([row.train_min for row in done]),
        total_min=mean([row.total_min for row in done]),
        vram_gib=mean([row.vram_gib for row in done]),
        fps=mean([row.fps for row in done]),
        psnr=psnr,
        ssim=mean([row.ssim for row in done]),
        lpips_vgg=mean([row.lpips_vgg for row in done]),
        deploy_mb=mean([row.deploy_mb for row in done]),
        ply_mb=mean([row.ply_mb for row in done]),
        primitives_k=None if primitives is None else primitives / 1000.0,
        paper=paper_mean,
        delta_psnr=None if psnr is None or paper_mean is None else psnr - paper_mean.psnr,
    )


def flags_for(rows: list[SceneRow]) -> list[str]:
    flags: list[str] = []
    for row in rows:
        if row.status != "complete" or row.paper is None:
            continue
        if METHODS[row.method].published and row.delta_psnr is not None and abs(row.delta_psnr) > PUBLISHED_TOLERANCE_DB:
            flags.append(
                f"{row.dataset}/{row.scene}/{row.method}: PSNR {row.psnr:.3f} vs published {row.paper.psnr:.3f} ({row.delta_psnr:+.3f} dB, band {PUBLISHED_TOLERANCE_DB} dB)"
            )
        if row.views != row.paper.views:
            flags.append(f"{row.dataset}/{row.scene}/{row.method}: {row.views} scored views, published checkpoint has {row.paper.views}")
    return flags


def build_table(
    runs: Path,
    paper: PaperNumbers,
    datasets: tuple[Dataset, ...] = ("dtu", "mipnerf360", "tnt"),
    scenes: dict[Dataset, tuple[str, ...]] | None = None,
    mean_scenes: dict[Dataset, tuple[str, ...]] | None = None,
    methods: tuple[Method, ...] = BENCHMARK_METHODS,
) -> BenchTable:
    """Read every run directory into scene rows, dataset means and flags.

    Args:
        runs: The runs root (<dataset>/<scene>/<method>/).
        paper: Published numbers (`hexels_bench.paper`).
        datasets: Datasets of the scene rows, in this order.
        scenes: Scenes per dataset for the rows; defaults to every protocol scene.
        mean_scenes: Scenes per dataset the means average; defaults to the representative scenes.
        methods: Methods per scene.
    """
    row_scenes: dict[Dataset, tuple[str, ...]] = scenes if scenes is not None else dict(PROTOCOL_SCENES)
    averaged: dict[Dataset, tuple[str, ...]] = mean_scenes if mean_scenes is not None else dict(REPRESENTATIVE_SCENES)
    rows: list[SceneRow] = [
        scene_row(runs, dataset, scene, method, paper) for dataset in datasets for scene in row_scenes[dataset] for method in methods
    ]
    means: list[MeanRow] = [
        mean_row(dataset, method, [scene_row(runs, dataset, scene, method, paper) for scene in averaged[dataset]])
        for dataset in PAPER_DATASET_ORDER
        if dataset in datasets
        for method in methods
    ]
    return BenchTable(
        format="hexels-bench-table-3",
        generated=datetime.now().isoformat(timespec="seconds"),
        runs_root=str(runs),
        published_hardware=paper.source.hardware_paper,
        mean_scenes={dataset: list(averaged[dataset]) for dataset in datasets},
        scenes=rows,
        means=means,
        flags=flags_for(rows),
    )


def write_table_json(table: BenchTable, path: Path) -> None:
    path.write_text(to_json(table) + "\n")


def any_values(**values: float | int | str | None) -> rr.AnyValues:
    """AnyValues of the scored fields: unscored (None) and non-finite numbers are left out."""
    result: rr.AnyValues = rr.AnyValues()
    for key, value in values.items():
        if value is not None and not (isinstance(value, float) and not math.isfinite(value)):
            result = result.with_component_from_data(key, value)
    return result


Column: TypeAlias = dict[str, float | None]
"""One model's values in a paper-style table, keyed by TABLE_ROWS field."""


def scene_columns(rows: list[SceneRow]) -> dict[str, Column]:
    """Columns of one scene: every complete method, then each published checkpoint once."""
    columns: dict[str, Column] = {}
    for row in rows:
        if row.status == "complete":
            columns[row.method] = {
                "psnr": row.psnr, "ssim": row.ssim, "lpips_vgg": row.lpips_vgg, "fps": row.fps, "train_min": row.train_min,
                "total_min": row.total_min, "vram_gib": row.vram_gib, "deploy_mb": row.deploy_mb,
                "primitives_k": None if row.primitives is None else row.primitives / 1000.0,
            }  # fmt: skip
        if row.paper is not None:
            columns[f"published {row.paper.model} 30k"] = {
                "psnr": row.paper.psnr, "ssim": row.paper.ssim, "lpips_vgg": row.paper.lpips_vgg, "train_min": row.paper.train_min,
                "deploy_mb": row.paper.deploy_mb, "primitives_k": row.paper.primitives / 1000.0,
            }  # fmt: skip
    return columns


def mean_columns(means: list[MeanRow]) -> dict[str, Column]:
    """Columns of one dataset's means: every method with a complete scene, then each published mean once."""
    columns: dict[str, Column] = {}
    for entry in means:
        if entry.scenes > 0:
            columns[entry.method] = {
                "psnr": entry.psnr, "ssim": entry.ssim, "lpips_vgg": entry.lpips_vgg, "fps": entry.fps, "train_min": entry.train_min,
                "total_min": entry.total_min, "vram_gib": entry.vram_gib, "deploy_mb": entry.deploy_mb, "primitives_k": entry.primitives_k,
                "scenes": float(entry.scenes),
            }  # fmt: skip
        if entry.paper is not None:
            columns[f"published {entry.paper.model} 30k"] = {
                "psnr": entry.paper.psnr, "ssim": entry.paper.ssim, "lpips_vgg": entry.paper.lpips_vgg, "train_min": entry.paper.train_min,
                "deploy_mb": entry.paper.deploy_mb, "primitives_k": entry.paper.primitives_k, "scenes": float(entry.paper.scenes),
            }  # fmt: skip
    return columns


TABLE_LABEL: str = "(metric and unit)"
"""Label column; it sorts first because the dataframe orders columns by name."""
TABLE_ROWS: tuple[tuple[str, str], ...] = (
    ("psnr", "PSNR dB"),
    ("ssim", "SSIM"),
    ("lpips_vgg", "LPIPS-VGG"),
    ("fps", "render FPS (paper protocol)"),
    ("train_min", "train min (loop)"),
    ("total_min", "train min (process)"),
    ("vram_gib", "train peak VRAM GiB"),
    ("deploy_mb", "deploy MB (FP32)"),
    ("primitives_k", "primitives K"),
    ("scenes", "scenes averaged"),
)
"""Paper-style table rows (sequence timeline `row`)."""


def rounded(value: float | None) -> float | None:
    """Four decimals for display cells (TABLE.json keeps full precision)."""
    return None if value is None else round(value, 4)


def log_paper_table(recording: rr.RecordingStream, entity: str, columns: dict[str, Column]) -> None:
    """Rows = metrics on the `row` timeline, one column per model."""
    for index, (field_name, label) in enumerate(TABLE_ROWS):
        recording.set_time("row", sequence=index)
        cells: dict[str, float | int | str | None] = {name: rounded(column.get(field_name)) for name, column in columns.items()}
        recording.log(entity, any_values(**{TABLE_LABEL: label}, **cells))


def table_blueprint(tables: dict[str, list[str]]) -> rrb.Blueprint:
    """One tab per dataset: its means table above its per-scene tables."""
    titles: dict[str, str] = {"tnt": "Tanks and Temples", "mipnerf360": "Mip-NeRF 360", "dtu": "DTU"}
    query = rrb.archetypes.DataframeQuery(timeline="row")
    tabs: list[rrb.Container] = [
        rrb.Vertical(
            *(rrb.DataframeView(name=f"{titles[dataset]}: {part}", origin=f"/table/{dataset}/{part}", query=query) for part in parts),
            name=titles[dataset],
        )
        for dataset, parts in tables.items()
    ]
    return rrb.Blueprint(rrb.Tabs(*tabs), collapse_panels=True)


def write_table_rrd(table: BenchTable, path: Path) -> None:
    """Log the table as data.

    `/metrics/<dataset>/<method>`: every complete scene as AnyValues on the `scene` timeline (index in upstream
    order). `/table/<dataset>/<scene>` and `/table/<dataset>/mean`: paper-style tables
    (rows = metrics on the `row` timeline, one column per model, published 30k beside ours) for the representative
    scenes, shown by the default blueprint.
    """
    recording = rr.RecordingStream("hexels_bench_table", recording_id="hexels-bench-table")
    recording.save(path)
    for row in table.scenes:
        if row.status != "complete":
            continue
        recording.set_time("scene", sequence=PROTOCOL_SCENES[row.dataset].index(row.scene))
        values: rr.AnyValues = any_values(
            scene=row.scene, psnr=row.psnr, ssim=row.ssim, lpips_vgg=row.lpips_vgg, train_min=row.train_min, total_min=row.total_min,
            vram_gib=row.vram_gib, fps=row.fps, deploy_mb=row.deploy_mb, ply_mb=row.ply_mb, primitives=row.primitives,
            published_psnr=None if row.paper is None else row.paper.psnr,
        )  # fmt: skip
        recording.log(f"metrics/{row.dataset}/{row.method}", values)
    recording.reset_time()
    tables: dict[str, list[str]] = {}
    for dataset in PAPER_DATASET_ORDER:
        if dataset not in table.mean_scenes:
            continue
        log_paper_table(recording, f"table/{dataset}/mean", mean_columns([entry for entry in table.means if entry.dataset == dataset]))
        for scene in table.mean_scenes[dataset]:
            log_paper_table(
                recording, f"table/{dataset}/{scene}", scene_columns([row for row in table.scenes if row.dataset == dataset and row.scene == scene])
            )
        tables[dataset] = ["mean", *table.mean_scenes[dataset]]
    recording.send_blueprint(table_blueprint(tables))
    recording.flush()
