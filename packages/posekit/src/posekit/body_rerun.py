"""Rerun logging for whole-body 3D people (SAM 3D Body): image overlays, MHR-70 joints, meshes, optional depth clouds.

Moved from ``packages/sam3d-body`` (``sam3d_body.api.visualization``) with the same entity paths and blueprint, so
recordings look as they did: ``/world/cam/pinhole/{image,pred/...}`` for 2D, ``/world/pred/{kpts3d,mesh}_<i>`` for 3D
and ``/world/depth/*_point_cloud`` for MoGe-2 depth.
"""

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Int32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import log_pinhole

from posekit.models.sam3d_body import Sam3dBodyReconstruction
from posekit.skeletons import MHR_70

BOX_PALETTE: UInt8[ndarray, "8 4"] = np.array(
    [
        [255, 99, 71, 255],
        [65, 105, 225, 255],
        [60, 179, 113, 255],
        [255, 215, 0, 255],
        [138, 43, 226, 255],
        [255, 140, 0, 255],
        [220, 20, 60, 255],
        [70, 130, 180, 255],
    ],
    dtype=np.uint8,
)
"""Per-person colours: tomato, royal blue, sea green, gold, blue violet, dark orange, crimson, steel blue."""
SEG_CLASS_OFFSET: int = 100
"""Segmentation class ids: 100 is background, 101.. are people (clear of the MHR-70 keypoint ids)."""
SEG_OVERLAY_ALPHA: int = 242
"""Segmentation overlay opacity (0-255)."""
MAX_POINT_CLOUD_POINTS: int = 50_000
"""Depth clouds above this size are voxel-downsampled."""
MIN_DEPTH_CONFIDENCE: float = 0.5
"""Depth pixels below this MoGe-2 confidence are dropped."""
KEYPOINT_IDS: list[int] = list(range(MHR_70.num_keypoints))


def compute_vertex_normals(vertices: Float32[ndarray, "v 3"], faces: Int32[ndarray, "f 3"], eps: float = 1e-12) -> Float32[ndarray, "v 3"]:
    """Area-weighted unit vertex normals of one triangle mesh; zero for degenerate vertices."""
    corners: Float32[ndarray, "f 3 3"] = vertices[faces]
    face_normals: Float32[ndarray, "f 3"] = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals: Float32[ndarray, "v 3"] = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths: Float32[ndarray, "v 1"] = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(lengths > eps, normals / np.maximum(lengths, eps), 0.0).astype(np.float32)


def in_image(uv: Float32[ndarray, "k 2"], height: int, width: int) -> Float32[ndarray, "k 2"]:
    """``uv`` with off-image rows set to NaN, which Rerun hides."""
    visible: Float32[ndarray, "k 2"] = uv.astype(np.float32, copy=True)
    outside: Bool[ndarray, "k"] = (visible[:, 0] < 0.0) | (visible[:, 0] >= width) | (visible[:, 1] < 0.0) | (visible[:, 1] >= height)
    visible[outside] = np.nan
    return visible


def set_annotation_context() -> None:
    """MHR-70 joint names and links for class 0, and one segmentation class per person colour."""
    person = rr.ClassDescription(
        info=rr.AnnotationInfo(id=0, label="Person", color=(0, 0, 255)),
        keypoint_annotations=[rr.AnnotationInfo(id=index, label=name) for index, name in enumerate(MHR_70.keypoint_names)],
        keypoint_connections=list(MHR_70.links),
    )
    segments: list[rr.ClassDescription] = [rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET, label="Background", color=(64, 64, 64, 0)))]
    for index, color in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
        segments.append(
            rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET + index, label=f"Person-{index}", color=(*color, SEG_OVERLAY_ALPHA)))
        )
    rr.log("/", rr.AnnotationContext([person, *segments]), static=True)


def log_people(
    people: Sam3dBodyReconstruction,
    rgb: UInt8[ndarray, "h w 3"] | tuple[int, int],
    parent_log_path: Path,
    *,
    masks: Bool[ndarray, "n h w"] | None = None,
    depth_meters: Float32[ndarray, "h w"] | None = None,
    depth_confidence: Float32[ndarray, "h w"] | None = None,
    depth_intrinsics: Float32[ndarray, "3 3"] | None = None,
) -> None:
    """Log one frame: image, boxes, 2D joints, segmentation, 3D joints and meshes, and optional depth clouds.

    Args:
        people: The frame's people (every row belongs to this frame).
        rgb: The frame, logged as an image; or only its ``(height, width)`` when the source is a video
            logged once as a VideoStream at ``<parent>/cam/pinhole/image``.
        parent_log_path: Root entity, ``/world`` in the apps.
        masks: Each person's mask, row-aligned with ``people``.
        depth_meters: Metric depth of the frame (MoGe-2), for the depth point clouds.
        depth_confidence: Its per-pixel confidence.
        depth_intrinsics: The intrinsics the depth was estimated with.

    Raises:
        ValueError: If depth is given without the frame, which colours the depth clouds.
    """
    frame: UInt8[ndarray, "h w 3"] | None = None if isinstance(rgb, tuple) else rgb
    frame_hw: tuple[int, int] = rgb if isinstance(rgb, tuple) else (int(rgb.shape[0]), int(rgb.shape[1]))
    height: int = frame_hw[0]
    width: int = frame_hw[1]
    if frame is None and depth_meters is not None:
        raise ValueError("Depth clouds need the frame for their colours; pass the image, not only its size.")
    cam_log_path: Path = parent_log_path / "cam"
    pinhole_log_path: Path = cam_log_path / "pinhole"
    pred_log_path: Path = pinhole_log_path / "pred"
    mesh_root_path: Path = parent_log_path / "pred"
    # The pinhole uses the model's focal length about the image centre, as its 2D keypoints do.
    focal: float = float(people.focal_length[0]) if people.num_people else float(np.hypot(height, width))
    intrinsics = Intrinsics(camera_conventions="RDF", fl_x=focal, fl_y=focal, cx=width / 2.0, cy=height / 2.0, height=height, width=width)
    extrinsics = Extrinsics(world_R_cam=np.eye(3, dtype=np.float32), world_t_cam=np.zeros(3, dtype=np.float32))
    log_pinhole(camera=PinholeParameters(intrinsics=intrinsics, extrinsics=extrinsics, name="pinhole"), cam_log_path=cam_log_path)
    rr.log(str(pred_log_path), rr.Clear(recursive=True))
    if frame is not None:
        rr.log(str(pinhole_log_path / "image"), rr.Image(frame, color_model=rr.ColorModel.RGB).compress(jpeg_quality=90))
    rr.log(str(mesh_root_path), rr.Clear(recursive=True))

    boxes: Float32[ndarray, "n 4"] = people.boxes.cpu().numpy()
    uv: Float32[ndarray, "n 70 2"] = people.keypoints.xy.cpu().numpy()
    assert people.keypoints.xyz_camera is not None
    xyz: Float32[ndarray, "n 70 3"] = people.keypoints.xyz_camera.cpu().numpy()
    vertices: Float32[ndarray, "n v 3"] = people.vertices.cpu().numpy()
    faces: Int32[ndarray, "f 3"] = people.faces.cpu().numpy().astype(np.int32)
    segmentation: UInt8[ndarray, "h w"] = np.full((height, width), SEG_CLASS_OFFSET, dtype=np.uint8)
    people_mask: Bool[ndarray, "h w"] = np.zeros((height, width), dtype=bool)
    for index in range(people.num_people):
        color: UInt8[ndarray, "4"] = BOX_PALETTE[index % len(BOX_PALETTE)]
        rr.log(
            f"{pred_log_path}/bbox_{index}",
            rr.Boxes2D(array=boxes[index], array_format=rr.Box2DFormat.XYXY, class_ids=0, colors=color[None], show_labels=True),
        )
        rr.log(f"{pred_log_path}/uv_{index}", rr.Points2D(positions=in_image(uv[index], height, width), keypoint_ids=KEYPOINT_IDS, class_ids=0, colors=(0, 255, 0)))
        if masks is not None:
            people_mask |= masks[index]
            segmentation = np.where(masks[index], np.uint8(SEG_CLASS_OFFSET + index + 1), segmentation)
        rr.log(f"{mesh_root_path}/kpts3d_{index}", rr.Points3D(positions=xyz[index], keypoint_ids=KEYPOINT_IDS, class_ids=0, colors=(0, 255, 0)))
        rr.log(
            f"{mesh_root_path}/mesh_{index}",
            rr.Mesh3D(
                vertex_positions=vertices[index],
                triangle_indices=faces,
                vertex_normals=compute_vertex_normals(vertices[index], faces),
                albedo_factor=(float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0, 0.35),
            ),
        )
    if np.any(segmentation != SEG_CLASS_OFFSET):
        rr.log(f"{pred_log_path}/segmentation_ids", rr.SegmentationImage(segmentation))
    if frame is not None and depth_meters is not None and depth_confidence is not None and depth_intrinsics is not None:
        log_depth_clouds(frame, depth_meters, depth_confidence, depth_intrinsics, people_mask, parent_log_path / "depth")


def log_depth_clouds(
    rgb: UInt8[ndarray, "h w 3"],
    depth_meters: Float32[ndarray, "h w"],
    confidence: Float32[ndarray, "h w"],
    intrinsics: Float32[ndarray, "3 3"],
    people_mask: Bool[ndarray, "h w"],
    depth_log_root: Path,
) -> None:
    """Full, background-only and people-only depth point clouds, without flying pixels or low-confidence depth."""
    import open3d as o3d
    from monopriors.depth_utils import depth_edges_mask
    from simplecv.ops.pc_utils import estimate_voxel_size

    height: int = int(rgb.shape[0])
    width: int = int(rgb.shape[1])
    rr.log(str(depth_log_root), rr.Clear(recursive=True))
    depth: Float32[ndarray, "h w"] = np.nan_to_num(depth_meters.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth = depth * np.logical_not(depth_edges_mask(depth, threshold=0.01)) * (confidence >= MIN_DEPTH_CONFIDENCE)
    grid_u, grid_v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points: Float32[ndarray, "h w 3"] = np.stack(
        [(grid_u - intrinsics[0, 2]) * depth / intrinsics[0, 0], (grid_v - intrinsics[1, 2]) * depth / intrinsics[1, 1], depth], axis=-1
    )
    valid: Bool[ndarray, "h w"] = (depth > 0.0) & np.isfinite(depth)
    for name, keep in (("full_depth", valid), ("background_depth", valid & ~people_mask), ("foreground_depth", valid & people_mask)):
        if not np.any(keep):
            continue
        positions: Float32[ndarray, "p 3"] = points[keep]
        colors: UInt8[ndarray, "p 3"] = rgb[keep]
        if len(positions) > MAX_POINT_CLOUD_POINTS:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(positions)
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
            cloud = cloud.voxel_down_sample(estimate_voxel_size(positions, target_points=MAX_POINT_CLOUD_POINTS, tolerance=0.25))
            positions = np.asarray(cloud.points, dtype=np.float32)
            colors = (np.asarray(cloud.colors, dtype=np.float32) * 255.0).astype(np.uint8)
        rr.log(f"{depth_log_root}/{name}_point_cloud", rr.Points3D(positions=positions, colors=colors))


def export_meshes_to_glb(people: Sam3dBodyReconstruction, output_dir: Path, *, center_mesh: bool = True) -> list[Path]:
    """Write one coloured GLB per person (a PLY if Open3D cannot write GLB) and return the paths."""
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    faces: Int32[ndarray, "f 3"] = people.faces.cpu().numpy().astype(np.int32)
    written: list[Path] = []
    for index, vertices in enumerate(people.vertices.cpu().numpy()):
        exported: Float32[ndarray, "v 3"] = vertices - vertices.mean(axis=0, keepdims=True) if center_mesh else vertices
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(exported.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(compute_vertex_normals(exported, faces).astype(np.float64))
        color: Float32[ndarray, "3"] = BOX_PALETTE[index % len(BOX_PALETTE), :3].astype(np.float32) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(color[None], len(exported), axis=0).astype(np.float64))
        for path in (output_dir / f"person_{index:02d}.glb", output_dir / f"person_{index:02d}.ply"):
            if o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, write_vertex_normals=True, write_vertex_colors=True):
                written.append(path)
                break
    return written


def create_view(log_depth: bool = True) -> rrb.ContainerLike:
    """Image overlays beside a 3D view; with depth, three 3D tabs (full, background, people)."""
    view_2d = rrb.Vertical(
        contents=[
            rrb.Spatial2DView(name="image", origin="/world/cam/pinhole", contents=["/world/cam/pinhole/image", "/world/cam/pinhole/pred/segmentation_ids"]),
            rrb.Spatial2DView(
                name="mhr",
                origin="/world/cam/pinhole",
                contents=["/world/cam/pinhole/image", "/world/cam/pinhole/pred/**", "- /world/cam/pinhole/pred/segmentation_ids/**"],
            ),
        ],
    )
    if not log_depth:
        view_3d = rrb.Spatial3DView(name="mhr_3d", line_grid=rrb.LineGrid3D(visible=False))
        return rrb.Tabs(contents=[rrb.Horizontal(contents=[view_2d, view_3d], column_shares=[2, 3])], name="sam-3d-body-demo")
    depth_views: list[rrb.Spatial3DView] = [
        rrb.Spatial3DView(
            name=f"Depth ({label})",
            origin="/world",
            contents=["/world/cam/**", "/world/pred/**", f"/world/depth/{cloud}_point_cloud"],
            line_grid=rrb.LineGrid3D(visible=False),
        )
        for label, cloud in (("full", "full_depth"), ("background", "background_depth"), ("foreground", "foreground_depth"))
    ]
    main_view = rrb.Horizontal(contents=[view_2d, rrb.Tabs(name="3D depth views", contents=depth_views)], column_shares=[2, 3])
    return rrb.Tabs(contents=[main_view], name="sam-3d-body-demo")
