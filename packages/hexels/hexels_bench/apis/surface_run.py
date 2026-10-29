"""Meshes from rendered depth, per scene and method: write the surface cameras, render the surface
recording with `gsplat render --surface` (or render upstream CUDA frames in process), and fuse with upstream's
Open3D TSDF settings (`hexels_bench.surface`).

Writes `runs/<dataset>/<scene>/<method>/surface/{cameras.json, surface.rrd, renders/, fuse.ply,
fuse_post.ply, fusion.json, log.txt}`; the CUDA method also writes `depth_parity.json` against the WGSL
recording of the same checkpoint. Scenes with a fusion.json are skipped unless `--force`.
"""

import fcntl
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import open3d as o3d
from jaxtyping import Bool, Float32
from numpy import ndarray
from serde import serde
from serde.json import to_json

from hexels_bench.apis.export import export_run
from hexels_bench.cameras import CamerasFile, ContractCamera, find_image, load_contract, write_cameras_file
from hexels_bench.checkpoint import UpstreamRunConfig, find_checkpoint, load_run_config
from hexels_bench.layout import Dataset, Roots
from hexels_bench.surface import (
    DepthMode,
    FusionResult,
    FusionView,
    TsdfSettings,
    alpha_mask,
    fuse_tsdf,
    post_process_mesh,
    scene_cameras,
    surface_depth,
    tsdf_settings,
)
from hexels_bench.surface_frames import SurfaceFrame, rrd_frames
from hexels_bench.upstream import surface_frames

SurfaceMethod: TypeAlias = Literal[
    "hexels-v3-surf-30k-pub-wgsl",
    "hexels-v3-surf-30k-pub-cuda",
    "hexels-v3-surf-30k",
    "brush-7k",
    "hexels-v1-7k",
    "hexels-v3-7k",
    "hexels-v1-30k-pub-wgsl",
    "hexels-v3-30k-pub-wgsl",
]
PUBLISHED_SURFACE: tuple[SurfaceMethod, ...] = ("hexels-v3-surf-30k-pub-wgsl", "hexels-v3-surf-30k-pub-cuda")
"""The paper's surface checkpoints (all views): one PLY, depth from WGSL or from upstream CUDA."""
GSPLAT: Path = Path(__file__).resolve().parents[3] / "gsplat-rust-renderer" / "target" / "release" / "gsplat"


@dataclass
class Config:
    """Fuse TSDF meshes from rendered depth (published surface checkpoints, or the 7k NVS runs on their train views)."""

    dataset: Dataset
    """Run-layout dataset name."""
    scenes: list[str]
    """Upstream scene names."""
    methods: list[SurfaceMethod] = field(default_factory=lambda: ["hexels-v3-surf-30k-pub-wgsl"])
    """Methods to fuse; NVS methods (7k runs, published NVS checkpoints) need their run's model.ply and fuse their training views."""
    depth_mode: DepthMode | None = None
    """Override the method's depth (Hexels: median, the paper's; Brush: expected); writes to surface-<mode>/."""
    gsplat: Path = GSPLAT
    """Release `gsplat` binary (pixi run -e gsplat-rust-renderer gsplat-build, or cargo build -p gsplat-cli --release)."""
    gpu_lock: Path | None = Path.home() / "hexels-work" / "gpu.lock"
    """flock held around each WGSL render; None to skip."""
    force: bool = False
    """Redo scenes that already have a fusion.json."""
    rerender: bool = False
    """Render again even when the run already has a complete surface.rrd (renders are deterministic)."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


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


PARITY_TOLERANCE: float = 1e-3
"""Median-depth difference (scene units) above which a pixel counts as different in depth_parity.json."""


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
    """`both_hit` pixels whose difference exceeds PARITY_TOLERANCE."""
    hit_mismatch: int
    """Pixels where exactly one renderer has a median depth."""
    alpha_abs_mean: float
    """Mean |alpha CUDA - WGSL| over all pixels."""


def frame_parity(frame: SurfaceFrame, reference: SurfaceFrame) -> DepthParity:
    both: Bool[ndarray, "h w"] = (frame.median_depth > 0) & (reference.median_depth > 0)
    difference: Float32[ndarray, "k"] = np.abs(frame.median_depth[both] - reference.median_depth[both])
    return DepthParity(
        views=1,
        pixels=frame.alpha.size,
        both_hit=int(difference.size),
        median_abs_mean=float(difference.mean()) if difference.size else 0.0,
        median_abs_max=float(difference.max()) if difference.size else 0.0,
        median_over_tolerance=int(np.count_nonzero(difference > PARITY_TOLERANCE)),
        hit_mismatch=int(np.count_nonzero((frame.median_depth > 0) != (reference.median_depth > 0))),
        alpha_abs_mean=float(np.abs(frame.alpha - reference.alpha).mean()),
    )


def total_parity(rows: list[DepthParity]) -> DepthParity:
    both_hit: int = sum(row.both_hit for row in rows)
    pixels: int = sum(row.pixels for row in rows)
    return DepthParity(
        views=len(rows),
        pixels=pixels,
        both_hit=both_hit,
        median_abs_mean=sum(row.median_abs_mean * row.both_hit for row in rows) / max(both_hit, 1),
        median_abs_max=max(row.median_abs_max for row in rows),
        median_over_tolerance=sum(row.median_over_tolerance for row in rows),
        hit_mismatch=sum(row.hit_mismatch for row in rows),
        alpha_abs_mean=sum(row.alpha_abs_mean * row.pixels for row in rows) / max(pixels, 1),
    )


@dataclass(frozen=True, slots=True)
class SurfacePlan:
    """Inputs of one (scene, method) surface run."""

    out_dir: Path
    """Where the outputs go."""
    ply: Path
    """Model to render."""
    cameras: list[ContractCamera]
    """Views to render and fuse, in order."""
    split: Literal["all", "train"]
    """Which views those are."""
    scene_dir: Path
    """Scene directory (upstream source_path)."""
    images_dir: Path
    """Scene images (alpha masks)."""
    depth_mode: DepthMode
    """Which depth to fuse."""
    alpha_threshold: float
    """Zero depth where alpha <= this (0 for Hexels as upstream; 0.5 for Brush, spec R7)."""


def default_depth_mode(method: SurfaceMethod) -> DepthMode:
    """Hexels fuse median depth (the paper's); Brush splats have only the alpha-weighted expected depth (spec R7)."""
    return "expected" if method.startswith("brush") else "median"


def surface_out_dir(roots: Roots, dataset: Dataset, scene: str, method: SurfaceMethod, depth_mode: DepthMode | None) -> Path:
    """`<run>/surface/` for the method's own depth mode, `<run>/surface-<mode>/` for an override."""
    if depth_mode is None or depth_mode == default_depth_mode(method):
        return roots.surface_dir(dataset, scene, method)
    return roots.run_dir(dataset, scene, method) / f"surface-{depth_mode}"


def surface_model_dir(roots: Roots, dataset: Dataset, scene: str, method: SurfaceMethod) -> Path | None:
    """Upstream run directory (config.json, checkpoints/) of an all-view surface method; None for NVS methods.
    `hexels-v3-surf-30k` is our own run of the paper's surface recipe (`scripts.train_scene --all-views --surface`)."""
    if method in PUBLISHED_SURFACE:
        return roots.published_surface_model(dataset, scene)
    if method == "hexels-v3-surf-30k":
        return roots.run_dir(dataset, scene, method) / "upstream"
    return None


def plan_surface(roots: Roots, dataset: Dataset, scene: str, method: SurfaceMethod, depth_mode: DepthMode | None) -> SurfacePlan:
    """All-view surface runs (published or ours) fuse every registered view loaded with the run's own settings;
    NVS runs fuse their training views (the contract's `geometry.training`)."""
    brush: bool = method.startswith("brush")
    mode: DepthMode = depth_mode if depth_mode is not None else default_depth_mode(method)
    out_dir: Path = surface_out_dir(roots, dataset, scene, method, depth_mode)
    model_dir: Path | None = surface_model_dir(roots, dataset, scene, method)
    if model_dir is not None:
        config: UpstreamRunConfig = load_run_config(model_dir / "config.json")
        if config.eval:
            raise ValueError(f"{model_dir}: not an all-view checkpoint")
        source: Path = Path(config.source_path)
        scene_dir: Path = source if source.is_absolute() else roots.data / source.relative_to("data")
        images: str = config.images if config.images is not None else "images"
        ply_method: SurfaceMethod = "hexels-v3-surf-30k-pub-wgsl" if method in PUBLISHED_SURFACE else method
        return SurfacePlan(
            out_dir=out_dir,
            ply=roots.run_dir(dataset, scene, ply_method) / "model.ply",
            cameras=scene_cameras(scene_dir, images, config.resolution),
            split="all",
            scene_dir=scene_dir,
            images_dir=scene_dir / images,
            depth_mode=mode,
            alpha_threshold=0.0,
        )
    contract = load_contract(roots.contract(dataset, scene))
    return SurfacePlan(
        out_dir=out_dir,
        ply=roots.run_dir(dataset, scene, method) / "model.ply",
        cameras=contract.geometry.training,
        split="train",
        scene_dir=roots.scene_dir(dataset, scene),
        images_dir=roots.scene_dir(dataset, scene) / contract.settings.images,
        depth_mode=mode,
        alpha_threshold=0.5 if brush else 0.0,
    )


def render_surface(config: Config, plan: SurfacePlan, method: str, log: Path) -> Path:
    """`gsplat render --surface` of every planned view (black background, as upstream); returns the recording."""
    rrd: Path = plan.out_dir / "surface.rrd"
    command: list[str] = [
        str(config.gsplat), "render", "--ply", str(plan.ply), "--camera", str(plan.out_dir / "cameras.json"),
        "--output-dir", str(plan.out_dir / "renders"), "--surface", str(rrd), "--surface-name", method, "--background", "0,0,0",
    ]  # fmt: skip
    with log.open("a") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        if config.gpu_lock is None:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        else:
            with config.gpu_lock.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    return rrd


def fusion_views(frames: Iterator[SurfaceFrame], plan: SurfacePlan) -> Iterator[FusionView]:
    """Masked, alpha-cut depth per frame, checked against the planned camera order."""
    for camera, frame in zip(plan.cameras, frames, strict=True):
        if frame.name != camera.name:
            raise ValueError(f"frame {frame.name} arrived for camera {camera.name}")
        mask: Bool[ndarray, "h w"] | None = alpha_mask(find_image(plan.images_dir, camera.name), camera.width, camera.height)
        depth: Float32[ndarray, "h w"] = surface_depth(
            frame.median_depth, frame.expected_depth, frame.alpha, plan.depth_mode, plan.alpha_threshold, mask
        )
        yield FusionView(camera=camera, rgb=frame.rgb, depth=depth)


def compare_frames(frames: Iterator[SurfaceFrame], references: Iterator[SurfaceFrame], rows: list[DepthParity]) -> Iterator[SurfaceFrame]:
    """Pass `frames` through, appending each frame's parity against the matching reference frame to `rows`."""
    for frame, reference in zip(frames, references, strict=True):
        rows.append(frame_parity(frame, reference))
        yield frame


def run_one(config: Config, dataset: Dataset, scene: str, method: SurfaceMethod) -> FusionReport | None:
    plan: SurfacePlan = plan_surface(config.roots, dataset, scene, method, config.depth_mode)
    if (plan.out_dir / "fusion.json").is_file() and not config.force:
        print(f"skip {dataset}/{scene}/{method}: {plan.out_dir / 'fusion.json'} exists")
        return None
    started: float = time.perf_counter()
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    log: Path = plan.out_dir / "log.txt"
    model_dir: Path | None = surface_model_dir(config.roots, dataset, scene, method)
    if model_dir is not None and not plan.ply.is_file():
        export_run(model_dir, plan.ply)
    background: list[float] = [0.0, 0.0, 0.0]
    write_cameras_file(
        plan.out_dir / "cameras.json",
        CamerasFile(format="hexels-cameras-1", dataset=dataset, scene=scene, split=plan.split, background=background, cameras=plan.cameras),
    )
    parity_rows: list[DepthParity] = []
    if method == "hexels-v3-surf-30k-pub-cuda":
        assert model_dir is not None
        frames: Iterator[SurfaceFrame] = surface_frames(model_dir, find_checkpoint(model_dir), plan.scene_dir)
        wgsl_rrd: Path = config.roots.surface_dir(dataset, scene, "hexels-v3-surf-30k-pub-wgsl") / "surface.rrd"
        if wgsl_rrd.is_file():
            frames = compare_frames(frames, rrd_frames(wgsl_rrd, "hexels-v3-surf-30k-pub-wgsl", plan.cameras), parity_rows)
        source: str = "upstream-cuda"
    else:
        rrd: Path = plan.out_dir / "surface.rrd"
        if config.rerender or not rrd.is_file():
            rrd = render_surface(config, plan, method, log)
        frames = rrd_frames(rrd, method, plan.cameras)
        source = "wgsl"
    settings: TsdfSettings = tsdf_settings(dataset, scene)
    result: FusionResult = fuse_tsdf(fusion_views(frames, plan), settings)
    o3d.io.write_triangle_mesh(str(plan.out_dir / "fuse.ply"), result.mesh)
    raw_triangles: int = len(result.mesh.triangles)
    mesh: o3d.geometry.TriangleMesh = post_process_mesh(result.mesh, settings.clusters_to_keep)
    o3d.io.write_triangle_mesh(str(plan.out_dir / "fuse_post.ply"), mesh)
    report = FusionReport(
        dataset=dataset,
        scene=scene,
        method=method,
        depth_source=source,
        depth_mode=plan.depth_mode,
        alpha_threshold=plan.alpha_threshold,
        split=plan.split,
        masked=dataset == "dtu",
        voxel_size=settings.voxel_size,
        sdf_trunc=settings.sdf_trunc,
        depth_trunc=settings.depth_trunc,
        views=result.views,
        valid_depth_fraction=result.valid_depth_fraction,
        raw_triangles=raw_triangles,
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        fusion_seconds=time.perf_counter() - started,
    )
    (plan.out_dir / "fusion.json").write_text(to_json(report) + "\n")
    if parity_rows:
        (plan.out_dir / "depth_parity.json").write_text(to_json(total_parity(parity_rows)) + "\n")
    print(f"{dataset}/{scene}/{method}: {report.views} views, {report.triangles:,} triangles, {report.fusion_seconds:.1f} s")
    return report


def main(config: Config) -> None:
    for scene in config.scenes:
        for method in config.methods:
            run_one(config, config.dataset, scene, method)
