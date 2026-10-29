"""Per-scene surface recording for the viewer and the report: the fused meshes, depth and normal maps of
a few views, the cameras, the reference geometry (DTU) and the surface metrics, with a default blueprint.

Writes `runs/<dataset>/<scene>/<scene>-surface.rrd` (application `hexels-surface`, recording
`<dataset>-<scene>`). Timeline `view` = the view's index in the first method's cameras.json.
Depth uses one Turbo range per scene (2nd/98th percentile of the first method's depth); normals are
world space, `round(255 (0.5 n + 0.5))`, black where nothing was hit.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray

from hexels_bench.apis.surface_run import SURFACE_METHODS, SurfaceMethod, SurfacePlan, plan_surface
from hexels_bench.cameras import ContractCamera, find_image, load_target
from hexels_bench.layout import Dataset, Roots
from hexels_bench.surface import camera_from_world, normals_from_depth, surface_depth
from hexels_bench.surface_frames import rrd_views

MAP_PARTS: tuple[str, ...] = ("rgb", "median_depth", "expected_depth", "alpha", "normal")


@dataclass
class Config:
    """Write one scene's surface recording from its surface runs."""

    dataset: Dataset
    """Run-layout dataset name."""
    scene: str
    """Upstream scene name."""
    methods: list[SurfaceMethod] = field(default_factory=lambda: ["hexels-v3-surf-30k-pub-wgsl", "hexels-v3-surf-30k-pub-cuda"])
    """Methods to show; the first one sets the depth range. Methods without surface.rrd show only their mesh."""
    views: int = 4
    """Views with depth and normal maps, evenly spaced over the views every map method rendered."""
    max_triangles: int = 2_000_000
    """Display meshes above this are decimated (quadric); metrics always use the full mesh."""
    reference_points: int = 1_000_000
    """DTU reference scan points shown (random subset)."""
    out: Path | None = None
    """Defaults to runs/<dataset>/<scene>/<scene>-surface.rrd."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def normal_rgb(normal: Float32[ndarray, "h w 3"], hit: ndarray) -> UInt8[ndarray, "h w 3"]:
    """World normals to RGB8 (`0.5 n + 0.5`), black where `hit` is false or the normal is zero."""
    length: Float32[ndarray, "h w 1"] = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit: Float32[ndarray, "h w 3"] = normal / np.maximum(length, np.float32(1e-12))
    rgb: UInt8[ndarray, "h w 3"] = np.round(255 * (0.5 * unit + 0.5)).astype(np.uint8)
    return np.where((hit & (length[..., 0] > 0))[..., None], rgb, np.uint8(0))


def world_from_camera(camera: ContractCamera) -> Float64[ndarray, "4 4"]:
    return np.linalg.inv(camera_from_world(camera))


def log_camera(camera: ContractCamera) -> None:
    pose: Float64[ndarray, "4 4"] = world_from_camera(camera)
    focal_x: float = 0.5 * camera.width / math.tan(0.5 * camera.fov_x)
    focal_y: float = 0.5 * camera.height / math.tan(0.5 * camera.fov_y)
    rr.log("world/camera", rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
    rr.log(
        "world/camera",
        rr.Pinhole(
            focal_length=[focal_x, focal_y],
            principal_point=[camera.width / 2, camera.height / 2],
            resolution=[camera.width, camera.height],
            image_plane_distance=0.1,
        ),
    )


def log_mesh(path: Path, entity: str, max_triangles: int) -> int:
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    mesh.compute_vertex_normals()
    colors: UInt8[ndarray, "n 3"] = np.round(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    rr.log(
        entity,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
            triangle_indices=np.asarray(mesh.triangles, dtype=np.uint32),
            vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
            vertex_colors=colors if len(colors) else None,
        ),
        static=True,
    )
    return len(mesh.triangles)


def log_dtu_reference(roots: Roots, scene: str, points: int) -> None:
    """The DTU scan in the training frame: `(x - t) / s` with the scene's `scale_mat_0` (what the culler inverts)."""
    cloud = o3d.io.read_point_cloud(str(roots.data / "dtu_official" / "Points" / "stl" / f"stl{int(scene.removeprefix('scan')):03d}_total.ply"))
    world: Float64[ndarray, "n 3"] = np.asarray(cloud.points)
    if len(world) > points:
        world = world[np.random.default_rng(0).choice(len(world), points, replace=False)]
    with np.load(roots.data / "dtu" / scene / "cameras.npz") as cameras:
        scale: Float64[ndarray, "4 4"] = cameras["scale_mat_0"].astype(np.float64)
    local: Float32[ndarray, "n 3"] = ((world - scale[:3, 3]) / scale[0, 0]).astype(np.float32)
    rr.log("world/reference", rr.Points3D(local, colors=[150, 150, 150], radii=rr.Radius.ui_points(1.0)), static=True)


def blueprint(
    map_methods: list[SurfaceMethod],
    mesh_methods: list[SurfaceMethod],
    up: Float64[ndarray, "3"],
    center: Float64[ndarray, "3"],
    eye: Float64[ndarray, "3"],
    reference: bool,
) -> rrb.Blueprint:
    def image_view(entity: str, name: str) -> rrb.Spatial2DView:
        return rrb.Spatial2DView(origin="/world/camera", contents=[entity], name=name)

    rows: list[rrb.Horizontal] = [
        rrb.Horizontal(
            image_view("/world/camera/gt", "GT"), *[image_view(f"/world/camera/{m}/rgb", f"{SURFACE_METHODS[m].label} RGB") for m in map_methods]
        ),
        rrb.Horizontal(*[image_view(f"/world/camera/{m}/depth", f"{SURFACE_METHODS[m].label} depth") for m in map_methods]),
        rrb.Horizontal(*[image_view(f"/world/camera/{m}/normal", f"{SURFACE_METHODS[m].label} normal") for m in map_methods]),
        rrb.Horizontal(*[image_view(f"/world/camera/{m}/normal_from_depth", f"{SURFACE_METHODS[m].label} normal from depth") for m in map_methods]),
    ]
    eye_controls = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=eye, look_target=center, eye_up=up)
    meshes: list[rrb.Spatial3DView] = [
        rrb.Spatial3DView(
            origin="/world", contents=[f"/world/mesh/{m}", "/world/camera"], name=f"{SURFACE_METHODS[m].label} mesh", eye_controls=eye_controls
        )
        for m in mesh_methods
    ]
    if mesh_methods and reference:
        contents: list[str] = [f"/world/mesh/{mesh_methods[0]}", "/world/reference"]
        meshes.append(rrb.Spatial3DView(origin="/world", contents=contents, name="reference scan", eye_controls=eye_controls))
    return rrb.Blueprint(rrb.Horizontal(rrb.Vertical(*rows), rrb.Tabs(*meshes), column_shares=[2, 3]), collapse_panels=True)


def main(config: Config) -> None:
    roots: Roots = config.roots
    plans: dict[SurfaceMethod, SurfacePlan] = {method: plan_surface(roots, config.dataset, config.scene, method, None) for method in config.methods}
    map_methods: list[SurfaceMethod] = [m for m in config.methods if (plans[m].out_dir / "surface.rrd").is_file()]
    mesh_methods: list[SurfaceMethod] = [m for m in config.methods if (plans[m].out_dir / "fuse_post.ply").is_file()]
    first: SurfacePlan = plans[config.methods[0]]
    shared: list[str] = [c.name for c in first.cameras if all(c.name in {d.name for d in plans[m].cameras} for m in map_methods)]
    chosen: list[str] = [shared[round(i * (len(shared) - 1) / max(config.views - 1, 1))] for i in range(min(config.views, len(shared)))]
    out: Path = config.out if config.out is not None else roots.runs / config.dataset / config.scene / f"{config.scene}-surface.rrd"

    maps: dict[str, dict[str, dict[str, ndarray]]] = {}
    for method in map_methods:
        plan: SurfacePlan = plans[method]
        index: dict[str, int] = {c.name: i for i, c in enumerate(plan.cameras)}
        maps[method] = {
            plan.cameras[view].name: parts
            for view, parts in rrd_views(plan.out_dir / "surface.rrd", method, MAP_PARTS, plan.cameras, keep={index[name] for name in chosen})
        }
    reference_depth: list[ndarray] = [
        np.where(parts["alpha"] > plans[map_methods[0]].alpha_threshold, parts[f"{plans[map_methods[0]].depth_mode}_depth"], 0) for parts in maps[map_methods[0]].values()
    ] if map_methods else []  # fmt: skip
    positive: ndarray = np.concatenate([d[d > 0] for d in reference_depth]) if reference_depth else np.ones(1)
    depth_range: tuple[float, float] = (float(np.percentile(positive, 2)), float(np.percentile(positive, 98)))

    rr.init("hexels-surface", recording_id=f"{config.dataset}-{config.scene}")
    rr.save(out)
    centers: Float64[ndarray, "n 3"] = np.asarray([world_from_camera(c)[:3, 3] for c in first.cameras])
    rr.log("world/cameras", rr.Points3D(centers, colors=[255, 200, 0], radii=rr.Radius.ui_points(3.0)), static=True)
    triangles: dict[str, int] = {m: log_mesh(plans[m].out_dir / "fuse_post.ply", f"world/mesh/{m}", config.max_triangles) for m in mesh_methods}
    if config.dataset == "dtu":
        log_dtu_reference(roots, config.scene, config.reference_points)
    for method in config.methods:
        metrics_path: Path = plans[method].out_dir / "surface_metrics.json"
        fusion_path: Path = plans[method].out_dir / "fusion.json"
        values: dict = {k: v for k, v in json.loads(metrics_path.read_text()).items() if isinstance(v, float)} if metrics_path.is_file() else {}
        if fusion_path.is_file():
            values["triangles"] = float(json.loads(fusion_path.read_text())["triangles"])
        if values:
            rr.log(f"metrics/surface/{method}", rr.AnyValues(**values), static=True)

    camera_index: dict[str, int] = {c.name: i for i, c in enumerate(first.cameras)}
    for name in chosen:
        camera: ContractCamera = next(c for c in first.cameras if c.name == name)
        rr.set_time("view", sequence=camera_index[name])
        log_camera(camera)
        rr.log("world/camera/gt", rr.Image(load_target(find_image(first.images_dir, name), camera.width, camera.height)))
        for method in map_methods:
            plan = plans[method]
            parts: dict[str, ndarray] = maps[method][name]
            depth: Float32[ndarray, "h w"] = surface_depth(
                parts["median_depth"], parts["expected_depth"], parts["alpha"], plan.depth_mode, plan.alpha_threshold, None
            )
            hit: ndarray = depth > 0
            rr.log(f"world/camera/{method}/rgb", rr.Image(parts["rgb"]))
            rr.log(f"world/camera/{method}/depth", rr.DepthImage(depth, meter=1.0, colormap="turbo", depth_range=depth_range))
            rr.log(f"world/camera/{method}/normal", rr.Image(normal_rgb(parts["normal"] * 2 - 1, parts["alpha"] > 0)))
            rr.log(f"world/camera/{method}/normal_from_depth", rr.Image(normal_rgb(normals_from_depth(depth, camera), hit)))

    up: Float64[ndarray, "3"] = -np.mean([world_from_camera(c)[:3, 1] for c in first.cameras], axis=0)
    look: Float64[ndarray, "3"] = np.median(centers, axis=0)
    if mesh_methods:
        vertices: ndarray = np.asarray(o3d.io.read_triangle_mesh(str(plans[mesh_methods[0]].out_dir / "fuse_post.ply")).vertices)
        look = np.median(vertices, axis=0)
    eye: Float64[ndarray, "3"] = look + 0.8 * (np.median(centers, axis=0) - look)
    rr.send_blueprint(blueprint(map_methods, mesh_methods, up / np.linalg.norm(up), look, eye, reference=config.dataset == "dtu"))
    rr.disconnect()
    print(f"wrote {out}: views {chosen}, meshes {triangles}, depth range {depth_range[0]:.3f}-{depth_range[1]:.3f}")
