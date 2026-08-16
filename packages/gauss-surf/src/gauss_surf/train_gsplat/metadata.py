"""Create the camera and seed metadata needed by direct gsplat training."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from arkitscenes_download.ingest.paths import (
    PINHOLE_ULTRAWIDE_RECT,
    PROMPTDA_MESH,
    SHARPNESS_ULTRAWIDE,
    TIMELINE,
)
from jaxtyping import Float32, UInt8
from numpy import ndarray

from gauss_surf.catalog import (
    SegmentReader,
    TimedeltaNs,
    _matrix_from_cell,
    _resolution_from_cell,
    table_timestamps,
)
from gauss_surf.contracts import (
    FRAME_SELECTION_LAYER,
    MOGE_NORMALS_LAYER,
    PROMPTDA_DEPTH_BLOB_COLUMN,
    PROMPTDA_LAYER,
    ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
    ULTRAWIDE_DEPTH_LAYER,
    ULTRAWIDE_NORMALS_LAYER,
    WIDE_CHOSEN_SHARPNESS_COLUMN,
    WIDE_INTRINSICS_COLUMN,
    WIDE_RESOLUTION_COLUMN,
    CameraTag,
)
from gauss_surf.uw_geometry import CameraPoses, CameraPoseTrack, load_camera_pose_track, resolve_camera_poses

UW_GRID_COLUMN: str = f"/{SHARPNESS_ULTRAWIDE}:Scalars:scalars"
UW_K_COLUMN: str = f"/{PINHOLE_ULTRAWIDE_RECT}:Pinhole:image_from_camera"
UW_RESOLUTION_COLUMN: str = f"/{PINHOLE_ULTRAWIDE_RECT}:Pinhole:resolution"
MESH_VERTICES_COLUMN: str = f"/{PROMPTDA_MESH}:Mesh3D:vertex_positions"
MESH_COLORS_COLUMN: str = f"/{PROMPTDA_MESH}:Mesh3D:vertex_colors"
METADATA_FILENAMES: tuple[str, ...] = ("transforms.json", "cameras_all.json", "seed.ply")
BUNDLE_MANIFEST_FILENAME: str = "bundle_manifest.json"
BUNDLE_MANIFEST_SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class MetadataStats:
    """Counts written for one fresh segment's direct-training metadata."""

    wide_frames: int
    ultrawide_frames: int
    holdout_frames: int
    seed_vertices: int
    render_wide_frames: int
    render_ultrawide_frames: int


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one metadata file."""
    with path.open("rb") as input_file:
        return hashlib.file_digest(input_file, "sha256").hexdigest()


def metadata_files_complete(bundle_dir: Path, *, allow_legacy: bool = False) -> bool:
    """Return whether a complete, internally consistent metadata generation exists."""
    if not all((bundle_dir / name).is_file() for name in METADATA_FILENAMES):
        return False
    manifest_path: Path = bundle_dir / BUNDLE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return allow_legacy
    try:
        manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest: dict[str, Any] = {
            "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
            "files": {name: {"sha256": _sha256_file(bundle_dir / name)} for name in METADATA_FILENAMES},
        }
    except (json.JSONDecodeError, OSError):
        return False
    return manifest == expected_manifest


def holdout_wide_indices(frame_count: int, interval: int = 8) -> tuple[int, ...]:
    """Return deterministic zero-based wide holdouts at a one-based cadence."""
    if frame_count < 0:
        raise ValueError("frame count must be non-negative")
    if interval <= 0:
        raise ValueError("holdout interval must be positive")
    return tuple(range(interval - 1, frame_count, interval))


def _opencv_to_opengl(world_from_camera_44: Float32[ndarray, "4 4"]) -> Float32[ndarray, "4 4"]:
    """Convert an RDF OpenCV camera transform to the OpenGL convention."""
    world_from_opengl_44: Float32[ndarray, "4 4"] = np.asarray(world_from_camera_44, dtype=np.float32).copy()
    world_from_opengl_44[:, 1:3] *= -1.0
    return world_from_opengl_44


def _camera_entry(
    *,
    stem: str,
    camera: CameraTag,
    timestamp_ns: int,
    K_33: Float32[ndarray, "3 3"],
    image_wh: tuple[int, int],
    world_from_camera_44: Float32[ndarray, "4 4"],
) -> dict[str, Any]:
    """Build one image-free direct-render camera record."""
    return {
        "stem": stem,
        "camera": camera,
        "timestamp_ns": timestamp_ns,
        "fl_x": float(K_33[0, 0]),
        "fl_y": float(K_33[1, 1]),
        "cx": float(K_33[0, 2]),
        "cy": float(K_33[1, 2]),
        "w": image_wh[0],
        "h": image_wh[1],
        "transform_matrix": _opencv_to_opengl(world_from_camera_44).tolist(),
    }


def _load_seed_mesh(reader: SegmentReader) -> tuple[Float32[ndarray, "n 3"], ndarray | None]:
    """Read PromptDA mesh positions and optional packed RGBA colors."""
    mesh_table: pa.Table = (
        reader.segment_view()
        .reader(index=TIMELINE, fill_latest_at=True)
        .select(TIMELINE, MESH_VERTICES_COLUMN, MESH_COLORS_COLUMN)
        .limit(1)
        .to_arrow_table()
    )
    if mesh_table.num_rows != 1:
        raise SystemExit("catalog segment has no PromptDA TSDF mesh")
    mesh_row: dict[str, Any] = mesh_table.to_pylist()[0]
    vertices_n3: Float32[ndarray, "n 3"] = np.asarray(mesh_row[MESH_VERTICES_COLUMN], dtype=np.float32)
    packed_colors_n: ndarray | None = None
    if mesh_row[MESH_COLORS_COLUMN] is not None:
        packed_colors_n = np.asarray(mesh_row[MESH_COLORS_COLUMN], dtype=np.uint32)
    return vertices_n3, packed_colors_n


def _write_seed_ply(path: Path, vertices_n3: Float32[ndarray, "n 3"], packed_rgba_n: ndarray | None) -> int:
    """Write PromptDA vertices as a metric colored binary PLY."""
    if vertices_n3.ndim != 2 or vertices_n3.shape[1] != 3:
        raise ValueError(f"seed vertices must have shape (N, 3), got {vertices_n3.shape}")
    vertex_count: int = len(vertices_n3)
    colors_n3: UInt8[ndarray, "n 3"] = np.full((vertex_count, 3), 128, dtype=np.uint8)
    if packed_rgba_n is not None:
        packed_n: ndarray = np.asarray(packed_rgba_n, dtype=np.uint32).reshape(-1)
        if len(packed_n) != vertex_count:
            raise ValueError("seed vertex colors do not match the vertex count")
        colors_n3[:, 0] = (packed_n >> 24).astype(np.uint8)
        colors_n3[:, 1] = (packed_n >> 16).astype(np.uint8)
        colors_n3[:, 2] = (packed_n >> 8).astype(np.uint8)
    vertex_dtype: np.dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    records_n: ndarray = np.empty(vertex_count, dtype=vertex_dtype)
    records_n["x"], records_n["y"], records_n["z"] = vertices_n3[:, 0], vertices_n3[:, 1], vertices_n3[:, 2]
    records_n["red"], records_n["green"], records_n["blue"] = colors_n3[:, 0], colors_n3[:, 1], colors_n3[:, 2]
    header: str = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output_file:
        output_file.write(header.encode("ascii"))
        records_n.tofile(output_file)
    temporary_path.replace(path)
    return vertex_count


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write one stable JSON metadata file."""
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def prepare_training_metadata(reader: SegmentReader, bundle_dir: Path) -> MetadataStats:
    """Build camera manifests and a metric seed directly from one live segment."""
    reader.require_layers(
        (FRAME_SELECTION_LAYER, MOGE_NORMALS_LAYER, PROMPTDA_LAYER, ULTRAWIDE_DEPTH_LAYER, ULTRAWIDE_NORMALS_LAYER)
    )
    wide_table: pa.Table = reader.chosen_table(
        WIDE_CHOSEN_SHARPNESS_COLUMN,
        (WIDE_INTRINSICS_COLUMN, WIDE_RESOLUTION_COLUMN),
    )
    uw_table: pa.Table = reader.chosen_table(ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN, (UW_K_COLUMN, UW_RESOLUTION_COLUMN))
    wide_timestamps_n: TimedeltaNs = table_timestamps(wide_table)
    uw_timestamps_n: TimedeltaNs = table_timestamps(uw_table)
    pose_track: CameraPoseTrack = load_camera_pose_track(reader)
    poses: CameraPoses = resolve_camera_poses(pose_track, wide_timestamps_n, uw_timestamps_n, exact_wide=True)
    holdout_indices: set[int] = set(holdout_wide_indices(wide_table.num_rows))

    training_frames: list[dict[str, Any]] = []
    for frame_index, (timestamp, row) in enumerate(zip(wide_timestamps_n, wide_table.to_pylist(), strict=True)):
        entry: dict[str, Any] = _camera_entry(
            stem=f"wide_{frame_index:06d}",
            camera="wide",
            timestamp_ns=int(timestamp.astype(np.int64)),
            K_33=_matrix_from_cell(row[WIDE_INTRINSICS_COLUMN], WIDE_INTRINSICS_COLUMN),
            image_wh=_resolution_from_cell(row[WIDE_RESOLUTION_COLUMN], WIDE_RESOLUTION_COLUMN),
            world_from_camera_44=poses.world_from_wide_n44[frame_index],
        )
        training_frames.append(
            {
                **entry,
                "file_path": f"images/{entry['stem']}.png",
                "holdout": frame_index in holdout_indices,
            }
        )
    for frame_index, (timestamp, row) in enumerate(zip(uw_timestamps_n, uw_table.to_pylist(), strict=True)):
        entry = _camera_entry(
            stem=f"uw_{frame_index:06d}",
            camera="uw",
            timestamp_ns=int(timestamp.astype(np.int64)),
            K_33=_matrix_from_cell(row[UW_K_COLUMN], UW_K_COLUMN),
            image_wh=_resolution_from_cell(row[UW_RESOLUTION_COLUMN], UW_RESOLUTION_COLUMN),
            world_from_camera_44=poses.world_from_ultrawide_n44[frame_index],
        )
        training_frames.append({**entry, "file_path": f"images_uw/{entry['stem']}.jpg", "holdout": False})

    wide_render_table: pa.Table = reader.chosen_table(
        PROMPTDA_DEPTH_BLOB_COLUMN,
        (WIDE_INTRINSICS_COLUMN, WIDE_RESOLUTION_COLUMN),
    )
    uw_render_table: pa.Table = reader.chosen_table(UW_GRID_COLUMN, ())
    wide_render_timestamps_n: TimedeltaNs = table_timestamps(wide_render_table)
    uw_render_timestamps_n: TimedeltaNs = table_timestamps(uw_render_table)
    render_poses: CameraPoses = resolve_camera_poses(
        pose_track,
        wide_render_timestamps_n,
        uw_render_timestamps_n,
        exact_wide=False,
    )
    uw_rows: list[dict[str, Any]] = uw_table.to_pylist()
    uw_intrinsics_n33: Float32[ndarray, "n_uw 3 3"] = np.stack(
        [_matrix_from_cell(row[UW_K_COLUMN], UW_K_COLUMN) for row in uw_rows]
    )
    uw_resolutions: list[tuple[int, int]] = [
        _resolution_from_cell(row[UW_RESOLUTION_COLUMN], UW_RESOLUTION_COLUMN) for row in uw_rows
    ]
    if not np.allclose(uw_intrinsics_n33, uw_intrinsics_n33[0][None], atol=1e-5, rtol=0.0):
        raise SystemExit("catalog segment has varying rectified-ultrawide intrinsics")
    if any(resolution != uw_resolutions[0] for resolution in uw_resolutions):
        raise SystemExit("catalog segment has varying rectified-ultrawide resolutions")

    render_frames: list[dict[str, Any]] = []
    for frame_index, (timestamp, row) in enumerate(
        zip(wide_render_timestamps_n, wide_render_table.to_pylist(), strict=True)
    ):
        render_frames.append(
            _camera_entry(
                stem=f"wide_all_{frame_index:06d}",
                camera="wide",
                timestamp_ns=int(timestamp.astype(np.int64)),
                K_33=_matrix_from_cell(row[WIDE_INTRINSICS_COLUMN], WIDE_INTRINSICS_COLUMN),
                image_wh=_resolution_from_cell(row[WIDE_RESOLUTION_COLUMN], WIDE_RESOLUTION_COLUMN),
                world_from_camera_44=render_poses.world_from_wide_n44[frame_index],
            )
        )
    for frame_index, timestamp in enumerate(uw_render_timestamps_n):
        render_frames.append(
            _camera_entry(
                stem=f"uw_all_{frame_index:06d}",
                camera="uw",
                timestamp_ns=int(timestamp.astype(np.int64)),
                K_33=uw_intrinsics_n33[0],
                image_wh=uw_resolutions[0],
                world_from_camera_44=render_poses.world_from_ultrawide_n44[frame_index],
            )
        )

    vertices_n3: Float32[ndarray, "n 3"]
    packed_colors_n: ndarray | None
    vertices_n3, packed_colors_n = _load_seed_mesh(reader)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    seed_vertices: int = _write_seed_ply(bundle_dir / "seed.ply", vertices_n3, packed_colors_n)
    _write_json(
        bundle_dir / "transforms.json",
        {
            "camera_model": "OPENCV",
            "ply_file_path": "seed.ply",
            "frames": training_frames,
        },
    )
    _write_json(
        bundle_dir / "cameras_all.json",
        {
            "schema_version": 1,
            "camera_model": "OPENCV",
            "counts": {
                "wide": wide_render_table.num_rows,
                "uw": uw_render_table.num_rows,
                "total": len(render_frames),
            },
            "ultrawide_pose_staleness_ms": {
                "minimum": float(np.min(poses.ultrawide_staleness_ms_n)),
                "mean": float(np.mean(poses.ultrawide_staleness_ms_n)),
                "maximum": float(np.max(poses.ultrawide_staleness_ms_n)),
            },
            "frames": render_frames,
        },
    )
    _write_json(
        bundle_dir / BUNDLE_MANIFEST_FILENAME,
        {
            "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
            "files": {name: {"sha256": _sha256_file(bundle_dir / name)} for name in METADATA_FILENAMES},
        },
    )
    return MetadataStats(
        wide_frames=wide_table.num_rows,
        ultrawide_frames=uw_table.num_rows,
        holdout_frames=len(holdout_indices),
        seed_vertices=seed_vertices,
        render_wide_frames=wide_render_table.num_rows,
        render_ultrawide_frames=uw_render_table.num_rows,
    )
