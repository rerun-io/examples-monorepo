"""Collect every scored surface run into `runs/SURFACE_TABLE.json`, with the paper's per-scene number beside it.

Rows come from `runs/<dataset>/<scene>/<method>/surface*/surface_metrics.json` (+ fusion.json, depth_parity.json);
paper values from `hexels_bench.paper` (`hf_surface_main_per_scene`, the published surface
checkpoints' own evaluation).
"""

from dataclasses import dataclass, field
from pathlib import Path

from serde import serde
from serde.json import to_json

from hexels_bench.layout import UPSTREAM_DATASET, Roots, read_json
from hexels_bench.paper import PAPER_NUMBERS, PaperNumbers, PublishedSurface
from hexels_bench.surface_records import DepthParity, FusionReport, SurfaceMetrics


@dataclass
class Config:
    """Write runs/SURFACE_TABLE.json from the scored surface runs."""

    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SurfaceRow:
    """One scored surface run."""

    dataset: str
    """Run-layout dataset name."""
    scene: str
    """Upstream scene name."""
    method: str
    """Run-layout method."""
    depth: str
    """Fused depth: <source>/<mode> (wgsl/median, upstream-cuda/median, wgsl/expected)."""
    split: str
    """all or train views."""
    triangles: int
    """fuse_post.ply triangles."""
    cd_mm: float | None
    """DTU Chamfer (mm)."""
    acc_mm: float | None
    """DTU accuracy (mm)."""
    comp_mm: float | None
    """DTU completeness (mm)."""
    f1: float | None
    """T&T F1 (%)."""
    precision: float | None
    """T&T precision (%)."""
    recall: float | None
    """T&T recall (%)."""
    paper_cd_mm: float | None
    """Paper per-scene Chamfer of the published surface checkpoint."""
    paper_f1: float | None
    """Paper per-scene F1 of the published surface checkpoint."""
    depth_vs_cuda_mean_abs: float | None
    """CUDA method: mean |median depth CUDA - WGSL| (scene units)."""


def rows(roots: Roots, paper: dict[str, dict[str, PublishedSurface]]) -> list[SurfaceRow]:
    found: list[SurfaceRow] = []
    for metrics_path in sorted(roots.runs.glob("*/*/*/surface*/surface_metrics.json")):
        metrics: SurfaceMetrics = read_json(SurfaceMetrics, metrics_path)
        fusion: FusionReport = read_json(FusionReport, metrics_path.parent / "fusion.json")
        parity_path: Path = metrics_path.parent / "depth_parity.json"
        parity: DepthParity | None = read_json(DepthParity, parity_path) if parity_path.is_file() else None
        reference: PublishedSurface | None = paper.get(UPSTREAM_DATASET[metrics.dataset], {}).get(metrics.scene)
        found.append(
            SurfaceRow(
                dataset=metrics.dataset,
                scene=metrics.scene,
                method=metrics.method,
                depth=f"{fusion.depth_source}/{fusion.depth_mode}",
                split=fusion.split,
                triangles=fusion.triangles,
                cd_mm=metrics.cd_mm,
                acc_mm=metrics.acc_mm,
                comp_mm=metrics.comp_mm,
                f1=metrics.f1,
                precision=metrics.precision,
                recall=metrics.recall,
                paper_cd_mm=None if reference is None else reference.cd_mm,
                paper_f1=None if reference is None else reference.f1_pct,
                depth_vs_cuda_mean_abs=None if parity is None else parity.median_abs_mean,
            )
        )
    return found


def main(config: Config) -> None:
    table: list[SurfaceRow] = rows(config.roots, read_json(PaperNumbers, PAPER_NUMBERS).hf_surface_main_per_scene)
    out: Path = config.roots.runs / "SURFACE_TABLE.json"
    out.write_text(to_json(table) + "\n")
    for row in table:
        score: str = f"CD {row.cd_mm:.4f} mm (paper {row.paper_cd_mm})" if row.cd_mm is not None else f"F1 {row.f1:.2f}% (paper {row.paper_f1})"
        print(f"{row.dataset}/{row.scene} {row.method} [{row.depth}, {row.split}]: {score}, {row.triangles:,} triangles")
    print(f"wrote {out} ({len(table)} rows)")
