"""Write the ``handtrack_v1`` and ``detnet_v1`` layer rrds of a tracker run, a merged standalone rrd per chosen clip,
and (``--register``) register the layers on the catalog's ``dataforge-umetrack``, the dataforge way.

Layer files: ``<layers-root>/<layer>/<segment>.rrd`` (application ``dataforge``, the segment's recording id, no
recording properties), written atomically; an existing file is kept unless ``--force``. A standalone clip is
``<export-dir>/<export-name>__<segment>.rrd``: the segment's base and ground-truth layers (read from the files the
catalog serves), the ``handtrack_v1`` layer and the handtrack blueprint, as one recording.

The ground truth (rig, headset poses, hand poses, the subject's profile) is read from the catalog, never from raw files.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TypeAlias
from urllib.parse import unquote, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
import torch
from dataforge import schema, writing
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView, OnDuplicateSegmentLayer
from rerun.chunk import LazyChunkStream, RrdReader
from scipy.spatial.transform import Rotation
from serde.json import from_json
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy

from handtrack import rerun_layers
from handtrack.blueprint import handtrack_blueprint
from handtrack.geometry.camera import CameraRig
from handtrack.results import SegmentTrack, load_track

HANDTRACK_LAYER: str = "handtrack_v1"
DETNET_LAYER: str = "detnet_v1"
GT_LAYERS: tuple[str, ...] = ("base", "hand_pose", "hand_mesh", "projections")
"""The segment's layers a standalone clip carries beside ``handtrack_v1``."""
PROFILE_ENVELOPE: str = '"hand_model"'
"""SHOW3D wraps the profile in a ``hand_model`` envelope; UmeTrack stores the bare model."""


@dataclass
class Config:
    """Export a tracker run's Rerun layers and standalone clips; optionally register the layers."""

    run_dir: Path | None = None
    """Tracker run directory holding ``<segment>.npz`` + ``.json`` (``SegmentTrack``) for ``handtrack_v1``."""
    detnet_dir: Path | None = None
    """DetNet-alone outputs in the same form, for ``detnet_v1``."""
    segments: tuple[str, ...] = ()
    """Only these segments; all the run's segments when empty."""
    clips: tuple[str, ...] = ()
    """Segments to export as standalone rrds (must be in the run)."""
    layers_root: Path = Path("/home/pablo/handtrack-data/layers")
    export_dir: Path = Path("/home/pablo/handtrack-data/export")
    export_name: str = "handtrack"
    """Prefix of the standalone clip file names."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog the ground truth is read from (read only)."""
    dataset: str = "dataforge-umetrack"
    register: bool = False
    """Register every layer file of the selected segments on ``register_url``'s ``dataset`` (replacing an earlier copy)."""
    register_url: str | None = None
    """Catalog to register on; ``catalog_url`` when None."""
    register_blueprint: bool = False
    """Also make the handtrack blueprint the dataset's default (it replaces dataforge's until dataforge re-registers)."""
    force: bool = False
    """Rewrite layer files that exist."""


def _dense(column: pa.ChunkedArray, width: int) -> Float32[ndarray, "f w"]:
    """A one-instance-per-row fixed-size-list column as rows of ``width`` floats; NaN on null or empty rows."""
    lists: pa.ListArray = column.combine_chunks()
    present: Bool[ndarray, "f"] = lists.value_lengths().fill_null(0).to_numpy() > 0
    rows: Float32[ndarray, "f w"] = np.full((len(lists), width), np.nan, dtype=np.float32)
    values: Float32[ndarray, "m"] = np.asarray(lists.flatten(recursive=True).to_numpy(zero_copy_only=False), dtype=np.float32)
    rows[present] = values.reshape(-1, width)
    return rows


StaticValue: TypeAlias = list[float] | int | str
"""What one instance of the static columns read here holds: a vector, a relation enum, a document."""


def _first(table: pa.Table, column: str) -> StaticValue:
    """The single value of a static one-instance column."""
    return table[column][0].as_py()[0]


def read_rig(static: pa.Table, rig: int = rerun_layers.RIG, num_cameras: int = rerun_layers.NUM_CAMERAS) -> CameraRig:
    """UmeTrack's four Fisheye62 cameras from the static columns (cam_from_rig and K are column-major)."""
    cam_from_rig: Float32[ndarray, "c 4 4"] = np.tile(np.eye(4, dtype=np.float32), (num_cameras, 1, 1))
    focal: Float32[ndarray, "c 2"] = np.zeros((num_cameras, 2), dtype=np.float32)
    principal: Float32[ndarray, "c 2"] = np.zeros((num_cameras, 2), dtype=np.float32)
    size: Float32[ndarray, "c 2"] = np.zeros((num_cameras, 2), dtype=np.float32)
    coefficients: Float32[ndarray, "c 8"] = np.zeros((num_cameras, 8), dtype=np.float32)
    for camera in range(num_cameras):
        node: str = schema.cam_path(rig, camera)
        pinhole: str = schema.pinhole_path(rig, camera)
        if _first(static, f"{node}:Transform3D:relation") != 2:
            raise ValueError(f"{node}: expected a ChildFromParent (cam_from_rig) extrinsic")
        cam_from_rig[camera, :3, :3] = np.asarray(_first(static, f"{node}:Transform3D:mat3x3"), dtype=np.float32).reshape(3, 3).T
        cam_from_rig[camera, :3, 3] = _first(static, f"{node}:Transform3D:translation")
        intrinsics: Float32[ndarray, "3 3"] = np.asarray(_first(static, f"{pinhole}:Pinhole:image_from_camera"), dtype=np.float32).reshape(3, 3).T
        focal[camera] = intrinsics[0, 0], intrinsics[1, 1]
        principal[camera] = intrinsics[0, 2], intrinsics[1, 2]
        size[camera] = _first(static, f"{pinhole}:Pinhole:resolution")
        coefficients[camera] = _first(static, f"{pinhole}:simplecv.components.DistortionCoefficients")
    return CameraRig(
        names=tuple(schema.cam_path(rig, camera) for camera in range(num_cameras)),
        image_size=torch.from_numpy(size),
        cam_from_rig=torch.from_numpy(cam_from_rig),
        focal=torch.from_numpy(focal),
        principal=torch.from_numpy(principal),
        fisheye62=torch.from_numpy(coefficients),
    )


def _poses(quaternion_xyzw: Float32[ndarray, "f 4"], translation: Float32[ndarray, "f 3"]) -> tuple[Float32[ndarray, "f 3 3"], Bool[ndarray, "f"]]:
    """Rotation matrices of the finite rows (identity elsewhere) and the finite-row mask."""
    valid: Bool[ndarray, "f"] = np.isfinite(quaternion_xyzw).all(axis=1) & np.isfinite(translation).all(axis=1)
    rotation: Float32[ndarray, "f 3 3"] = np.tile(np.eye(3, dtype=np.float32), (len(valid), 1, 1))
    if valid.any():
        rotation[valid] = Rotation.from_quat(quaternion_xyzw[valid].astype(np.float64)).as_matrix().astype(np.float32)
    return rotation, valid


def read_ground_truth(entry: DatasetEntry, segment: str) -> rerun_layers.GroundTruth:
    """One static and one temporal query: the rig, the headset path, both hands' poses and the subject's hand model."""
    view: DatasetView = entry.filter_segments(segment)
    rig_path: str = schema.rig_path(rerun_layers.RIG)
    cameras: list[str] = [schema.cam_path(rerun_layers.RIG, camera) for camera in range(rerun_layers.NUM_CAMERAS)]
    static: pa.Table = (
        view.filter_contents([*cameras, *(f"{camera}/pinhole" for camera in cameras), schema.hand_profile_path()]).reader(index=None).to_arrow_table()
    )
    entities: list[str] = [rig_path, *(schema.hands_path(side) + "/**" for side in rerun_layers.SIDES)]
    temporal: pa.Table = view.filter_contents(entities).reader(index=schema.TIMELINE).to_arrow_table().sort_by(schema.TIMELINE)
    video_time_ns: Int64[ndarray, "f"] = temporal[schema.TIMELINE].combine_chunks().cast(pa.int64()).to_numpy()
    frame_index: Int64[ndarray, "f"] = temporal[schema.FRAME_INDEX].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    rig_translation: Float32[ndarray, "f 3"] = _dense(temporal[f"{rig_path}:Transform3D:translation"], 3)
    rig_rotation_valid: tuple[Float32[ndarray, "f 3 3"], Bool[ndarray, "f"]] = _poses(_dense(temporal[f"{rig_path}:Transform3D:quaternion"], 4), rig_translation)
    world_from_rig: Float32[ndarray, "f 4 4"] = rerun_layers.rigid_transforms(rig_rotation_valid[0], rig_translation)
    world_from_rig[~rig_rotation_valid[1]] = np.nan
    rotation: Float32[ndarray, "f 2 3 3"] = np.zeros((len(video_time_ns), 2, 3, 3), dtype=np.float32)
    translation: Float32[ndarray, "f 2 3"] = np.zeros((len(video_time_ns), 2, 3), dtype=np.float32)
    joint_angles: Float32[ndarray, "f 2 22"] = np.zeros((len(video_time_ns), 2, 22), dtype=np.float32)
    present: Bool[ndarray, "f 2"] = np.zeros((len(video_time_ns), 2), dtype=bool)
    for index, side in enumerate(rerun_layers.SIDES):
        wrist: str = schema.hand_wrist_path(side)
        hand_translation: Float32[ndarray, "f 3"] = _dense(temporal[f"{wrist}:Transform3D:translation"], 3)
        hand_rotation_valid: tuple[Float32[ndarray, "f 3 3"], Bool[ndarray, "f"]] = _poses(_dense(temporal[f"{wrist}:Transform3D:quaternion"], 4), hand_translation)
        angles: Float32[ndarray, "f 22"] = _dense(temporal[f"{schema.hand_joint_angles_path(side)}:joint_angles"], 22)
        confidence: Float32[ndarray, "f 1"] = _dense(temporal[f"{schema.hand_confidence_path(side)}:Scalars:scalars"], 1)
        present[:, index] = hand_rotation_valid[1] & np.isfinite(angles).all(axis=1) & (np.nan_to_num(confidence[:, 0]) > 0)
        rotation[:, index] = hand_rotation_valid[0]
        translation[:, index] = np.nan_to_num(hand_translation)
        joint_angles[:, index] = np.nan_to_num(angles)
    profile: StaticValue = _first(static, f"{schema.hand_profile_path()}:TextDocument:text")
    if not isinstance(profile, str) or PROFILE_ENVELOPE in profile[:64]:
        raise ValueError(f"{segment}: expected UmeTrack's bare hand model profile")
    return rerun_layers.GroundTruth(
        segment=segment,
        rig=read_rig(static),
        video_time_ns=video_time_ns,
        frame_index=frame_index,
        world_from_rig=world_from_rig,
        rotation=rotation,
        translation=translation,
        joint_angles=joint_angles,
        present=present,
        model=from_json(HandModelNumpy, profile),
    )


def layer_path(layers_root: Path, layer: str, segment: str) -> Path:
    return layers_root / layer / f"{segment}.rrd"


def write_layer_files(config: Config, entry: DatasetEntry, tracks: dict[str, Path], detections: dict[str, Path]) -> None:
    """Write the missing (or, with ``--force``, every) layer file of the selected segments."""
    for segment in sorted(set(tracks) | set(detections)):
        started: float = perf_counter()
        handtrack_target: Path = layer_path(config.layers_root, HANDTRACK_LAYER, segment)
        if segment in tracks and not writing.should_skip(handtrack_target, force=config.force):
            track: SegmentTrack = load_track(tracks[segment])
            truth: rerun_layers.GroundTruth = read_ground_truth(entry, segment)
            with writing.atomic_recording(handtrack_target, recording_id=segment, send_properties=False) as recording:
                rerun_layers.write_handtrack_layer(recording, track, truth, rerun_layers.prediction_model(track, truth))
        detnet_target: Path = layer_path(config.layers_root, DETNET_LAYER, segment)
        if segment in detections and not writing.should_skip(detnet_target, force=config.force):
            with writing.atomic_recording(detnet_target, recording_id=segment, send_properties=False) as recording:
                rerun_layers.write_detnet_layer(recording, load_track(detections[segment]))
        print(f"{segment}: layers in {perf_counter() - started:.1f} s")


def storage_paths(entry: DatasetEntry, segment: str) -> dict[str, Path]:
    """The file behind each registered layer of a segment."""
    row: dict[str, list[str]] = (
        entry.segment_table().to_arrow_table().filter(pc.field("rerun_segment_id") == segment).select(["rerun_layer_names", "rerun_storage_urls"]).to_pylist()[0]
    )
    return {layer: Path(unquote(urlparse(url).path)) for layer, url in zip(row["rerun_layer_names"], row["rerun_storage_urls"], strict=True)}


def export_clip(entry: DatasetEntry, segment: str, handtrack_rrd: Path, target: Path) -> None:
    """One standalone recording: the segment's base and ground-truth layers, ``handtrack_v1`` and the handtrack blueprint."""
    sources: dict[str, Path] = storage_paths(entry, segment)
    missing: list[str] = [layer for layer in GT_LAYERS if layer not in sources]
    if missing:
        raise ValueError(f"{segment}: the catalog has no {missing} layer")
    streams: list[LazyChunkStream] = [RrdReader(path).stream() for path in (*(sources[layer] for layer in GT_LAYERS), handtrack_rrd)]
    with writing.atomic_write(target) as temp_path, rr.RecordingStream(application_id=writing.APPLICATION_ID, recording_id=segment, send_properties=False) as recording:
        recording.save(temp_path, default_blueprint=handtrack_blueprint(), write_footer=True)
        rr.send_chunks(LazyChunkStream.merge(*streams), recording=recording)


def register_layers(config: Config, segments: list[str]) -> None:
    """Register the layer files of ``segments`` under their layer names, replacing an earlier registration of the same layer."""
    url: str = config.register_url or config.catalog_url
    entry: DatasetEntry = CatalogClient(url).get_dataset(config.dataset)
    for layer in (HANDTRACK_LAYER, DETNET_LAYER):
        candidates: list[Path] = [layer_path(config.layers_root, layer, segment) for segment in segments]
        files: list[Path] = [path for path in candidates if path.is_file()]
        if files:
            entry.register([path.resolve().as_uri() for path in files], layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
            print(f"registered {len(files)} {layer} rrds on {config.dataset} at {url}")
    if config.register_blueprint:
        # A registered file is never rewritten (writing.py): every run registers a fresh stamped file.
        stamp: str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        blueprint_file: Path = config.layers_root / "blueprints" / f"{config.dataset}__handtrack__{stamp}.rbl"
        with writing.atomic_write(blueprint_file) as temp_path:
            handtrack_blueprint().save(writing.APPLICATION_ID, str(temp_path))
        entry.register_blueprint(blueprint_file.resolve().as_uri(), set_default=True)
        print(f"registered the handtrack blueprint as {config.dataset}'s default")


def _run_files(directory: Path | None, segments: tuple[str, ...]) -> dict[str, Path]:
    if directory is None:
        return {}
    found: dict[str, Path] = {path.stem: path for path in sorted(directory.glob("*.npz"))}
    return {segment: path for segment, path in found.items() if not segments or segment in segments}


def main(config: Config) -> None:
    tracks: dict[str, Path] = _run_files(config.run_dir, config.segments)
    detections: dict[str, Path] = _run_files(config.detnet_dir, config.segments)
    unknown_clips: list[str] = [clip for clip in config.clips if clip not in tracks]
    if unknown_clips:
        raise ValueError(f"clips without a track in {config.run_dir}: {unknown_clips}")
    entry: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    write_layer_files(config, entry, tracks, detections)
    for clip in config.clips:
        started: float = perf_counter()
        target: Path = config.export_dir / f"{config.export_name}__{clip}.rrd"
        export_clip(entry, clip, layer_path(config.layers_root, HANDTRACK_LAYER, clip), target)
        print(f"{clip}: standalone {target} ({target.stat().st_size / 1e6:.1f} MB) in {perf_counter() - started:.1f} s")
    if config.register:
        register_layers(config, sorted(set(tracks) | set(detections)))

