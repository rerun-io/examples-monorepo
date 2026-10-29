"""Brush 3DGS | Hexels side by side: both views track one logged orbit camera on the `frame` timeline.

Brush enters as native `GaussianSplats3D`, Hexels as `Hexels3D` (stock Rerun shows its Points3D fallback).
Below the views, a DataframeView lists our per-scene numbers (from the benchmark's `TABLE.json`) beside the
published ones. The `brush` and `hexels` layouts keep one view, the shape `gsplat probe` measures.
"""

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.gaussians3d import log_ply
from gsplat_rust_renderer.hexels3d import Hexels3D, log_hexels, read_hexel_ply
from gsplat_rust_renderer.hexels_metrics import (
    METRICS_ENTITY,
    METRICS_TIMELINE,
    BenchTable,
    MetricRow,
    SceneCells,
    log_metric_table,
    paper_row,
    read_json,
    run_row,
    scene_cells,
)
from gsplat_rust_renderer.orbit import CameraSpec, Orbit, orbit_from_cameras, orbit_specs, write_orbit_json

Dataset: TypeAlias = Literal["mipnerf360", "dtu", "tnt"]
Layout: TypeAlias = Literal["compare", "brush", "hexels"]
UPSTREAM_DATASET: dict[Dataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tanks_and_temples"}
"""Run-layout dataset name -> dataset directory of the published camera contracts."""
BRUSH_ENTITY: str = "/world/brush"
HEXELS_ENTITY: str = "/world/hexels"
ORBIT_ENTITY: str = "/world/orbit_cam"
ORBIT_TIMELINE: str = "frame"


@dataclass
class CompareConfig:
    """Log a Brush 3DGS PLY and a hexel PLY of one scene side by side on a shared orbit."""

    rr_config: RerunTyroConfig
    """Viewer wiring; `--rr-config.save <scene>-compare.rrd` writes the recording."""
    dataset: Dataset = "mipnerf360"
    """Run-layout dataset name."""
    scene: str = "garden"
    """Upstream scene name."""
    brush_method: str = "brush-7k"
    """Run directory of the Brush 3DGS PLY (left view)."""
    hexels_method: str = "hexels-v1-30k-pub-wgsl"
    """Run directory of the hexel PLY (right view)."""
    layout: Layout = "compare"
    """`compare` (two views and the table) or one view (`brush`, `hexels`) for `gsplat probe`."""
    frames: int = 300
    """Orbit poses; one per timeline step. 300 makes `gsplat probe`'s 300-frame windows whole orbits."""
    width: int = 1920
    """Orbit camera resolution (orbit.json and the logged pinhole)."""
    height: int = 1080
    """Orbit camera resolution (orbit.json and the logged pinhole)."""
    fps: float = 30.0
    """Timeline playback rate; 60 advances one pose per headless probe frame."""
    brush_probe: Path | None = None
    """`gsplat probe` report of the `brush` layout, shown in the table and its view title."""
    hexels_probe: Path | None = None
    """`gsplat probe` report of the `hexels` layout, shown in the table and its view title."""
    brush_speed: Path | None = None
    """`gsplat speed --impl ours` report of the Brush PLY on orbit.json, shown in the table."""
    hexels_speed: Path | None = None
    """`gsplat speed --impl ours` report of the hexel PLY on orbit.json, shown in the table."""
    timing_gpu: str = "RTX 5090"
    """GPU of the probe and speed reports, named in the table title."""
    test_view_fps: bool = True
    """Show the run's test-view render FPS; it was measured on the RTX 5090, so drop it for other hosts."""
    runs_root: Path = field(default_factory=lambda: Path.home() / "hexels-work/runs")
    """Run layout root (contracts/run-layout.md)."""
    contracts_root: Path = field(default_factory=lambda: Path.home() / "gsplat-modern-data/hexels-pretrained/cameras/nvs")
    """Published camera contracts: `<paper dataset>/<scene>/contract.json`."""
    table: Path | None = None
    """Benchmark table with our and the published per-scene numbers (`hexels-bench-table`); default `<runs_root>/TABLE.json`."""


@serde
@dataclass(frozen=True, slots=True)
class ContractCamera:
    """One camera of the published camera contract (read partially; 2DGS conventions)."""

    name: str
    """Image stem."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    fov_x: float
    """Horizontal field of view in radians; the principal point is the image centre."""
    fov_y: float
    """Vertical field of view in radians."""
    world_view_transform: Float64[ndarray, "4 4"]
    """Transposed OpenCV world-to-camera matrix (row-vector convention)."""

    def world_from_camera(self) -> Float64[ndarray, "4 4"]:
        """OpenCV (right/down/forward) camera-to-world pose."""
        return np.linalg.inv(self.world_view_transform.T)


@serde
@dataclass(frozen=True, slots=True)
class ContractGeometry:
    """Camera records per split."""

    training: list[ContractCamera]
    """Training cameras."""
    test: list[ContractCamera]
    """Held-out cameras."""


@serde
@dataclass(frozen=True, slots=True)
class CameraContract:
    """HF jhu-ccvl/Hexels `cameras/nvs/<dataset>/<scene>/contract.json` (third-party document, read partially)."""

    geometry: ContractGeometry
    """Camera records."""


def display_name(method: str) -> str:
    """`brush-7k` -> `Brush 3DGS 7k`, `hexels-v1-30k-pub-wgsl` -> `Hexels V1 30k published`."""
    match = re.fullmatch(r"(brush|hexels)(?:-(v\d))?-(\d+k)(-pub)?.*", method)
    if match is None:
        return method
    kind, variant, steps, published = match.groups()
    name: str = "Brush 3DGS" if kind == "brush" else f"Hexels {variant.upper() if variant else ''}".strip()
    return f"{name} {steps}{' published' if published else ''}"


def view_name(row: MetricRow, primitive: str) -> str:
    """`<model> · <count> <primitive>`, plus the viewer FPS of the model's probe when there is one."""
    name: str = f"{row.method} · {row.primitives:,} {primitive}"
    return name if row.viewer_ms_median is None else f"{name} · {1000.0 / row.viewer_ms_median:.0f} FPS viewer"


def vertex_count(path: Path) -> int:
    """`element vertex <n>` from a PLY header."""
    with path.open("rb") as file:
        for line in file:
            if line.startswith(b"element vertex "):
                return int(line.split()[2])
            if line.strip() == b"end_header":
                break
    raise ValueError(f"{path}: no vertex element")


def log_cameras(contract: CameraContract, orbit: Orbit, specs: list[CameraSpec]) -> None:
    """Static test frustums, and the orbit camera that both views track on the `frame` timeline."""
    for camera in contract.geometry.test:
        pose: Float64[ndarray, "4 4"] = camera.world_from_camera()
        path: str = f"/world/cameras/test/{camera.name}"
        rr.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=True)
        focal: list[float] = [0.5 * camera.width / np.tan(0.5 * camera.fov_x), 0.5 * camera.height / np.tan(0.5 * camera.fov_y)]
        rr.log(path, rr.Pinhole(focal_length=focal, width=camera.width, height=camera.height, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.1), static=True)
    spec: CameraSpec = specs[0]
    # Both views must contain the tracked camera; a transparent, zero-width frustum keeps its lines out of the pixels.
    rr.log(ORBIT_ENTITY, rr.Pinhole(focal_length=[spec.fx, spec.fy], width=spec.width, height=spec.height, camera_xyz=rr.ViewCoordinates.RDF,
                                    image_plane_distance=0.05, color=[0, 0, 0, 0], line_width=0.0), static=True)
    for index, pose in enumerate(orbit.world_from_cameras):
        rr.set_time(ORBIT_TIMELINE, sequence=index)
        rr.log(ORBIT_ENTITY, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
    rr.reset_time()


def splat_view(name: str, entity: str) -> rrb.Spatial3DView:
    """A black-background 3D view of one model that takes the orbit camera's pose."""
    return rrb.Spatial3DView(
        name=name, origin="/world", contents=[f"{entity}/**", ORBIT_ENTITY], line_grid=False,
        background=rrb.Background(color=(0, 0, 0), kind="SolidColor"), eye_controls=rrb.EyeControls3D(tracking_entity=ORBIT_ENTITY),
    )


def compare_blueprint(layout: Layout, brush_name: str, hexels_name: str, fps: float, timing_gpu: str) -> rrb.Blueprint:
    """Brush left, Hexels right, metrics below; the orbit plays in a loop."""
    brush: rrb.Spatial3DView = splat_view(brush_name, BRUSH_ENTITY)
    hexels: rrb.Spatial3DView = splat_view(hexels_name, HEXELS_ENTITY)
    time_panel: rrb.TimePanel = rrb.TimePanel(timeline=ORBIT_TIMELINE, play_state="playing", loop_mode="all", fps=fps, state="collapsed")
    if layout != "compare":
        return rrb.Blueprint(brush if layout == "brush" else hexels, time_panel, collapse_panels=True)
    table: rrb.DataframeView = rrb.DataframeView(
        name=f"Per-scene metrics: viewer and renderer timing on {timing_gpu}, quality and training on RTX 5090, published on A5000", origin=METRICS_ENTITY,
        query=rrb.archetypes.DataframeQuery(timeline=METRICS_TIMELINE),
    )
    # The table needs about 350 px for its twelve rows at 1080p.
    return rrb.Blueprint(rrb.Vertical(rrb.Horizontal(brush, hexels), table, row_shares=[1.85, 1]), time_panel, collapse_panels=True)


def main(config: CompareConfig) -> None:
    """Log both models, the orbit (also written to `<run>/orbit.json`), the table and the blueprint."""
    scene_dir: Path = config.runs_root / config.dataset / config.scene
    contract: CameraContract = read_json(CameraContract, config.contracts_root / UPSTREAM_DATASET[config.dataset] / config.scene / "contract.json")
    cameras: list[ContractCamera] = contract.geometry.training + contract.geometry.test
    orbit: Orbit = orbit_from_cameras(np.stack([camera.world_from_camera() for camera in cameras]), config.frames)
    fov_y: float = float(np.median([camera.fov_y for camera in contract.geometry.test]))
    specs: list[CameraSpec] = orbit_specs(orbit, config.width, config.height, fov_y)
    write_orbit_json(scene_dir / "orbit.json", specs)

    brush_ply: Path = scene_dir / config.brush_method / "model.ply"
    hexels_ply: Path = scene_dir / config.hexels_method / "model.ply"
    brush_count: int = vertex_count(brush_ply)
    hexels: Hexels3D = read_hexel_ply(hexels_ply)
    variant: str = hexels.variant
    table: BenchTable = read_json(BenchTable, config.table if config.table is not None else config.runs_root / "TABLE.json")
    brush_cells: SceneCells | None = scene_cells(table, config.dataset, config.scene, config.brush_method)
    hexels_cells: SceneCells | None = scene_cells(table, config.dataset, config.scene, config.hexels_method)
    if brush_cells is None or hexels_cells is None:
        raise ValueError(f"TABLE.json has no {config.dataset}/{config.scene} rows for {config.brush_method} and {config.hexels_method}")
    brush: MetricRow = run_row(table, brush_cells, display_name(config.brush_method), brush_count, config.brush_probe, config.brush_speed)
    hexel: MetricRow = run_row(table, hexels_cells, display_name(config.hexels_method), len(hexels.centers), config.hexels_probe, config.hexels_speed)
    if not config.test_view_fps:
        brush, hexel = replace(brush, fps=None), replace(hexel, fps=None)
    brush_name: str = view_name(brush, "splats")
    hexels_name: str = view_name(hexel, "hexels")
    if config.layout == "compare":
        # Rows first: the table sizes its columns to the content of its first frame.
        log_metric_table([brush, hexel, paper_row(hexels_cells, f"Paper Hexels {variant.upper()} 30k")])
    rr.send_blueprint(compare_blueprint(config.layout, brush_name, hexels_name, config.fps, config.timing_gpu))
    log_cameras(contract, orbit, specs)
    if config.layout != "hexels":
        log_ply(brush_ply, entity=BRUSH_ENTITY)
    if config.layout != "brush":
        log_hexels(hexels, HEXELS_ENTITY)
    recording: rr.RecordingStream | None = rr.get_global_data_recording()
    if recording is not None:
        recording.flush(timeout_sec=120.0)
    print(f"logged {config.dataset}/{config.scene}: {brush_name} | {hexels_name}, {config.frames}-pose orbit -> {scene_dir / 'orbit.json'}")
