"""Writers for the four artifacts a cuSFM run leaves behind.

Replaces `kpmap_to_colmap`, `extract_pose_from_map_main` and
`update_keyframe_pose_main` (docs/spec/export.md), plus the runtime log that
`pycusfm/command_runner.py` appends.

| Artifact | Written by | cuSFM equivalent |
|---|---|---|
| `<out>/sparse/` COLMAP text model | `write_colmap_model` | `kpmap_to_colmap` |
| `<out>/output_poses/**.tum` | `write_pose_files` | `extract_pose_from_map_main` |
| `<out>/kpmap/keyframes/frames_meta.json` | `write_optimised_frames_meta` | `keypoints_mapper_main` / `update_keyframe_pose_main` |
| `<out>/runtime.csv` | `append_runtime_record` | `CommandRunner` |

## Deliberate differences from the blob's output

The TUM files and `frames_meta.json` are reproduced to float64 noise — composing
the rotation through scipy rather than Eigen moves the last bit, worst case
4e-16 over Galileo's 15 merged lines, while the timestamp column matches as text.
The COLMAP model is **semantically** equivalent, not byte-identical, for four
reasons:

1. **Text formatting.** cuSFM writes `ofs.precision(15)`, default float format,
   and a trailing space after every field. COLMAP's own writer uses its own
   precision and no trailing space. Both parse to the same numbers.
2. **Line order.** cuSFM emits `unordered_map` bucket order; COLMAP sorts by id.
   No reader cares, and matching the bucket order is not reproducible anyway.
3. **Extra files.** COLMAP 4.2's `write_text` also emits `rigs.txt` and
   `frames.txt`. cuSFM's export predates the rig data model and flattens the rig
   away, writing one independent pose per image. Emitting a real rig is an
   improvement, not a fidelity requirement (export.md §8).
4. **`points3D.txt` `ERROR`.** cuSFM hardcodes `2.0` for every point because
   `MapKeyPoints` has no error field; a real reconstruction carries the measured
   value. Anything reading that column as a quality signal was already misled.

One cuSFM bug is deliberately **not** reproduced: its TUM reader truncates
`seconds * 1e6`, so odd microsecond timestamps come back one low and frames get
silently dropped (export.md §5.4). `colsfm.geometry.parse_tum_line` rounds.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pycolmap

from colsfm.frames_meta import FramesMeta, InitialPoseType, KeyframeMeta, write_frames_meta
from colsfm.geometry import TumPose, format_tum_line

SPARSE_DIR_NAME: Final[str] = "sparse"
"""Subdirectory of the workspace holding the COLMAP model."""

OUTPUT_POSES_DIR_NAME: Final[str] = "output_poses"
"""Subdirectory holding the TUM trajectories."""

MERGED_POSE_FILE_NAME: Final[str] = "merged_pose_file.tum"
"""Name of the single merged trajectory, written in `output_poses` itself."""

KEYFRAME_METADATA_SUBPATH: Final[Path] = Path("kpmap") / "keyframes" / "frames_meta.json"
"""Where the optimised metadata lands inside the workspace."""

RUNTIME_CSV_NAME: Final[str] = "runtime.csv"
"""Per-stage runtime log, appended to by every stage."""

RUNTIME_CSV_HEADER: Final[tuple[str, str]] = ("command", "runtime_seconds")
"""Exactly the header `pycusfm/command_runner.py` writes."""

DEFAULT_SESSION_NAME: Final[str] = "0"
"""Session subdirectory used when `camera_params_id_to_session_name` is empty."""


@dataclass(frozen=True, slots=True)
class RuntimeRecord:
    """One row of `runtime.csv`."""

    command: str
    """`"<binary_name> <space-joined args>"`, exactly as the runner formats it."""
    runtime_seconds: float
    """Wall-clock seconds the stage took."""


def colour_points_from_images(reconstruction: pycolmap.Reconstruction, image_root: Path, num_threads: int = -1) -> int:
    """Give every 3D point the colour of the imagery that observes it.

    cuSFM colours points in the **extractor**: `feature_extractor_main --output_rgb`
    fills `KeypointVector.r/g/b` per keypoint and the mapper copies it through
    (keypoints_mapper_main.md §5.7). COLMAP's ALIKED extractor stores no colour,
    so the same information is recovered here instead, at export time, from the
    same JPEGs — `Reconstruction.extract_colors_for_all_images` averages each
    point's observations. Where the blob samples one pixel per keypoint, this
    averages over the track, which is strictly the better estimate and costs
    0.4 s over Galileo's 226 images.

    Without it a `points3D.txt` carries `0 0 0` everywhere and every viewer falls
    back to a flat colour.

    Args:
        reconstruction: The model to colour, edited in place.
        image_root: Directory the images' `name` values are relative to.
        num_threads: Decoding threads; -1 lets COLMAP use every core.

    Returns:
        Points that ended up with a non-black colour. Points whose every
        observation falls outside its image keep `(0, 0, 0)`.

    Raises:
        FileNotFoundError: When `image_root` is not a directory.
    """
    if not image_root.is_dir():
        raise FileNotFoundError(f"No image directory at {image_root}")
    reconstruction.extract_colors_for_all_images(str(image_root), num_threads)
    return sum(1 for point in reconstruction.points3D.values() if any(int(channel) for channel in point.color))


def write_colmap_model(output_dir: Path, reconstruction: pycolmap.Reconstruction) -> Path:
    """Write a COLMAP text model into `<output_dir>/sparse/`.

    Args:
        output_dir: Workspace root; the `sparse` subdirectory is created.
        reconstruction: The reconstruction to serialise.

    Returns:
        The directory the model was written to.
    """
    sparse_dir: Path = output_dir / SPARSE_DIR_NAME
    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(str(sparse_dir))
    return sparse_dir


def _tum_poses(frames_meta: FramesMeta, keyframes: Sequence[KeyframeMeta], in_vehicle_frame: bool) -> list[TumPose]:
    """Collect poses into cuSFM's timestamp-keyed map and flatten it.

    cuSFM stores them in a `std::map<uint64 microseconds, SE3>`, so the output is
    sorted ascending by timestamp and duplicate timestamps collapse with the last
    write winning (export.md §4.4).

    Args:
        frames_meta: The collection the keyframes belong to.
        keyframes: Keyframes to export, in collection order.
        in_vehicle_frame: Export `world_T_vehicle` rather than `world_T_cam`.

    Returns:
        Poses sorted ascending by timestamp, one per unique microsecond.
    """
    by_timestamp: dict[int, pycolmap.Rigid3d] = {}
    for keyframe in keyframes:
        pose: pycolmap.Rigid3d = frames_meta.world_T_vehicle(keyframe) if in_vehicle_frame else keyframe.world_T_cam
        by_timestamp[keyframe.timestamp_microseconds] = pose
    return [
        TumPose(timestamp_microseconds=timestamp, world_T_body=by_timestamp[timestamp]) for timestamp in sorted(by_timestamp)
    ]


def write_tum_file(path: Path, poses: Sequence[TumPose]) -> None:
    """Write a TUM trajectory in cuSFM's exact formatting.

    Args:
        path: Destination; parent directories are created.
        poses: Poses to write, already ordered and deduplicated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [format_tum_line(pose.timestamp_microseconds, pose.world_T_body) for pose in poses]
    path.write_text("".join(f"{line}\n" for line in lines))


def session_name(frames_meta: FramesMeta, camera_params_id: int) -> str:
    """Resolve the session subdirectory for one camera.

    Args:
        frames_meta: The collection.
        camera_params_id: Camera whose session to look up.

    Returns:
        The mapped session name, or `"0"` when the map is empty — cuSFM logs
        `camera_params_id_to_session_name is empty, generating fake session names`
        in that case (export.md §5.2).
    """
    return frames_meta.session_name_by_camera_params_id.get(camera_params_id, DEFAULT_SESSION_NAME)


def write_pose_files(
    output_dir: Path,
    frames_meta: FramesMeta,
    *,
    export_pose_in_vehicle_frame: bool = True,
) -> list[Path]:
    """Write the per-camera and merged TUM trajectories.

    Per camera the file is `<output_dir>/output_poses/<session>/camera_name-<sensor_name>_pose_file.tum`;
    the merged trajectory lands in `output_poses` itself, not in the session
    subdirectory (export.md §4.4).

    With `export_pose_in_vehicle_frame=False` the merged file is meaningless — it
    mixes every camera's world pose into one timestamp-keyed map — so it is only
    written in vehicle mode.

    Args:
        output_dir: Workspace root.
        frames_meta: The collection to export.
        export_pose_in_vehicle_frame: Write `world_T_vehicle` rather than `world_T_cam`.

    Returns:
        Every file written, per-camera files first.
    """
    poses_dir: Path = output_dir / OUTPUT_POSES_DIR_NAME
    by_camera: dict[int, list[KeyframeMeta]] = {}
    for keyframe in frames_meta.keyframes:
        by_camera.setdefault(keyframe.camera_params_id, []).append(keyframe)

    written: list[Path] = []
    for camera_params_id in sorted(by_camera):
        sensor_name: str = frames_meta.cameras[camera_params_id].sensor_name
        path: Path = poses_dir / session_name(frames_meta, camera_params_id) / f"camera_name-{sensor_name}_pose_file.tum"
        write_tum_file(path, _tum_poses(frames_meta, by_camera[camera_params_id], export_pose_in_vehicle_frame))
        written.append(path)

    # `CusfmRunner` always passes `--also_merge_poses`, so the merged file is not a knob.
    if export_pose_in_vehicle_frame:
        merged_path: Path = poses_dir / MERGED_POSE_FILE_NAME
        write_tum_file(merged_path, _tum_poses(frames_meta, frames_meta.keyframes, True))
        written.append(merged_path)
    return written


def write_optimised_frames_meta(
    output_dir: Path,
    frames_meta: FramesMeta,
    world_T_cam_by_keyframe_id: Mapping[int, pycolmap.Rigid3d],
    initial_pose_type: InitialPoseType = "ALIGNMENT",
) -> FramesMeta:
    """Write `<output_dir>/kpmap/keyframes/frames_meta.json` with optimised poses.

    Everything except `camera_to_world` and `initial_pose_type` is copied from the
    input collection verbatim, which is exactly what the blob's own diff shows
    (export.md §5.2: 32 of 32 keyframes differ in the pose and in nothing else).

    The re-posed collection is returned rather than the path — the path is
    `output_dir / KEYFRAME_METADATA_SUBPATH` by definition, whereas `write_pose_files`
    needs the very collection that was written, and building it twice is how the
    exported poses and the exported TUM files drift apart.

    Args:
        output_dir: Workspace root.
        frames_meta: The input collection.
        world_T_cam_by_keyframe_id: Optimised camera poses keyed by keyframe id.
        initial_pose_type: Provenance to stamp; the mapper writes `ALIGNMENT`.

    Returns:
        The collection that was written.
    """
    optimised: FramesMeta = frames_meta.with_camera_to_world(world_T_cam_by_keyframe_id, initial_pose_type)
    write_frames_meta(output_dir / KEYFRAME_METADATA_SUBPATH, optimised)
    return optimised


def append_runtime_record(output_dir: Path, record: RuntimeRecord) -> Path:
    """Append one stage's runtime to `<output_dir>/runtime.csv`.

    Mirrors `pycusfm/command_runner.py`: the same two columns, the header written
    only when the file does not yet exist, and `csv.writer`'s default dialect so
    a command containing a comma is quoted the same way.

    Args:
        output_dir: Directory holding the log.
        record: The row to append.

    Returns:
        The path written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path: Path = output_dir / RUNTIME_CSV_NAME
    exists: bool = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(list(RUNTIME_CSV_HEADER))
        writer.writerow([record.command, record.runtime_seconds])
    return path


def read_runtime_records(path: Path) -> list[RuntimeRecord]:
    """Read a `runtime.csv` back.

    Args:
        path: The log file.

    Returns:
        One record per data row, in file order.

    Raises:
        ValueError: When the header is not the two columns the runner writes.
    """
    with path.open(newline="") as handle:
        rows: list[list[str]] = list(csv.reader(handle))
    if not rows or tuple(rows[0]) != RUNTIME_CSV_HEADER:
        raise ValueError(f"{path} is not a cuSFM runtime log; header was {rows[0] if rows else None}")
    return [RuntimeRecord(command=row[0], runtime_seconds=float(row[1])) for row in rows[1:]]
