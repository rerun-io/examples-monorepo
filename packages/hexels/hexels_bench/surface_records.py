"""The JSON records of a surface run: `fusion.json` and `depth_parity.json` (`apis.surface_run`) and
`surface_metrics.json` (`apis.surface_eval`). No Open3D here, so the table reads them without it."""

from dataclasses import dataclass

from serde import serde

from hexels_bench.layout import Dataset


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FusionReport:
    """`fusion.json`: what was fused and how (upstream's extraction.json fields that apply)."""

    dataset: str
    """Run-layout dataset name."""
    scene: str
    """Upstream scene name."""
    method: str
    """Run-layout method."""
    depth_source: str
    """wgsl (gsplat render --surface) or upstream-cuda."""
    depth_mode: str
    """median or expected."""
    alpha_threshold: float
    """Depth is zeroed where alpha <= this."""
    split: str
    """all (every registered view) or train (the NVS training views)."""
    masked: bool
    """True when the images' alpha masks zeroed depth (DTU)."""
    voxel_size: float
    """TSDF voxel."""
    sdf_trunc: float
    """TSDF truncation."""
    depth_trunc: float
    """Maximum fused depth."""
    views: int
    """Integrated views."""
    valid_depth_fraction: float
    """Fraction of fused pixels with 0 < depth < depth_trunc."""
    raw_triangles: int
    """Triangles of fuse.ply."""
    vertices: int
    """Vertices of fuse_post.ply."""
    triangles: int
    """Triangles of fuse_post.ply."""
    fusion_seconds: float
    """Integration + marching cubes + post-processing wall time (includes reading or rendering frames)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DepthParity:
    """`depth_parity.json`: upstream CUDA vs WGSL surface frames of the same checkpoint (all views, or one view)."""

    views: int
    """Compared views."""
    pixels: int
    """All pixels of those views."""
    both_hit: int
    """Pixels where both median depths are positive."""
    median_abs_mean: float
    """Mean |median depth CUDA - WGSL| over `both_hit` pixels (scene units)."""
    median_abs_max: float
    """Largest such difference."""
    median_over_tolerance: int
    """`both_hit` pixels whose difference exceeds `surface_run.PARITY_TOLERANCE`."""
    hit_mismatch: int
    """Pixels where exactly one renderer has a median depth."""
    alpha_abs_mean: float
    """Mean |alpha CUDA - WGSL| over all pixels."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SurfaceMetrics:
    """`surface_metrics.json`; unscored fields are null."""

    dataset: Dataset
    """Run-layout dataset name."""
    scene: str
    """Upstream scene name."""
    method: str
    """Run-layout method."""
    evaluator: str
    """2dgs-eval_dtu or 2dgs-eval_tnt."""
    cd_mm: float | None
    """DTU Chamfer distance (mm)."""
    acc_mm: float | None
    """DTU accuracy (mean d2s, mm)."""
    comp_mm: float | None
    """DTU completeness (mean s2d, mm)."""
    precision: float | None
    """T&T precision at tau, percent."""
    recall: float | None
    """T&T recall at tau, percent."""
    f1: float | None
    """T&T F1 at tau, percent."""
    tau: float | None
    """T&T distance threshold (scene units of the GT)."""
