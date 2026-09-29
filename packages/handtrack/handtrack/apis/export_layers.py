"""Write the ``handtrack_v1`` and ``detnet_v1`` layer rrds of a tracker run, a merged standalone rrd per chosen clip,
and (``--register``) register the layers on the catalog's ``dataforge-umetrack``, the dataforge way.

Layer files: ``<layers-root>/<layer>/<segment>.rrd`` (application ``dataforge``, the segment's recording id, no
recording properties), written atomically; an existing file is kept unless ``--force``. A standalone clip is
``<export-dir>/<export-name>__<segment>.rrd``: the segment's base and ground-truth layers (read from the files the
catalog serves), the ``handtrack_v1`` layer and the handtrack blueprint, as one recording.

The ground truth (rig, headset poses, hand poses, the subject's profile) is read from the catalog through
``handtrack.data.catalog``'s readers, never from raw files.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from dataforge import schema, writing
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.chunk import LazyChunkStream, RrdReader

from handtrack import rerun_layers
from handtrack.blueprint import handtrack_blueprint
from handtrack.data import catalog
from handtrack.data.catalog import HandTimeline, SegmentInfo
from handtrack.hand.pose import hand_model_numpy_from_profile
from handtrack.labels.validity import SHOW3D_CONFIDENCE_THRESHOLD
from handtrack.results import SegmentTrack, load_track

HANDTRACK_LAYER: str = "handtrack_v1"
DETNET_LAYER: str = "detnet_v1"
GT_LAYERS: tuple[str, ...] = ("base", "hand_pose", "hand_mesh", "projections")
"""The segment's layers a standalone clip carries beside ``handtrack_v1``."""


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
    catalog_url: str = catalog.CATALOG_URL
    """Catalog the ground truth is read from (read only)."""
    dataset: str = catalog.UMETRACK
    register: bool = False
    """Register every layer file of the selected segments on ``register_url``'s ``dataset`` (replacing an earlier copy)."""
    register_url: str | None = None
    """Catalog to register on; ``catalog_url`` when None."""
    register_blueprint: bool = False
    """Also make the handtrack blueprint the dataset's default (it replaces dataforge's until dataforge re-registers)."""
    force: bool = False
    """Rewrite layer files that exist."""


def read_ground_truth(entry: DatasetEntry, info: SegmentInfo) -> rerun_layers.GroundTruth:
    """The catalog's static row and hand timeline (``handtrack.data.catalog``) as numpy, plus the frame index and the subject's numpy model.

    The temporal query names the rig and the three hand entities it reads (``/**`` would also pull the ``hand_mesh`` layer).
    """
    statics: pa.Table = catalog.read_statics(entry, info)
    entities: list[str] = [
        catalog.layout_for(info.dataset).rig,
        *(f"{schema.hands_path(side)}/{part}" for side in rerun_layers.SIDES for part in ("confidence", "joint_angles", "wrist")),
    ]
    table: pa.Table = entry.filter_segments(info.segment_id).filter_contents(entities).reader(index=schema.TIMELINE).to_arrow_table().sort_by(schema.TIMELINE)
    timeline: HandTimeline = catalog.hand_timeline(table, statics, info)
    return rerun_layers.GroundTruth(
        segment=info.segment_id,
        rig=catalog.read_rig(statics, info)[0],
        video_time_ns=timeline.video_time_ns,
        frame_index=table[schema.FRAME_INDEX].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64),
        world_from_rig=timeline.world_from_rig.numpy(),
        rotation=np.stack([pose.rotation.numpy() for pose in timeline.poses], axis=1),
        translation=np.stack([pose.translation.numpy() for pose in timeline.poses], axis=1),
        joint_angles=np.stack([pose.joint_angles.numpy() for pose in timeline.poses], axis=1),
        present=(timeline.has_pose & timeline.headset_valid[:, None]
                 & (timeline.confidence > (SHOW3D_CONFIDENCE_THRESHOLD if catalog.is_show3d(info.dataset) else 0.0))).numpy(),
        model=hand_model_numpy_from_profile(catalog.static_text(statics, catalog.PROFILE_COLUMN, f"{info.dataset} {info.segment_id}")),
    )


def layer_path(layers_root: Path, layer: str, segment: str) -> Path:
    return layers_root / layer / f"{segment}.rrd"


def write_layer_files(config: Config, entry: DatasetEntry, infos: dict[str, SegmentInfo], tracks: dict[str, Path], detections: dict[str, Path]) -> None:
    """Write the missing (or, with ``--force``, every) layer file of the selected segments."""
    for segment in sorted(set(tracks) | set(detections)):
        started: float = perf_counter()
        handtrack_target: Path = layer_path(config.layers_root, HANDTRACK_LAYER, segment)
        if segment in tracks and not writing.should_skip(handtrack_target, force=config.force):
            track: SegmentTrack = load_track(tracks[segment])
            truth: rerun_layers.GroundTruth = read_ground_truth(entry, infos[segment])
            with writing.atomic_recording(handtrack_target, recording_id=segment, send_properties=False) as recording:
                rerun_layers.write_handtrack_layer(recording, track, truth, rerun_layers.prediction_model(track, truth))
        detnet_target: Path = layer_path(config.layers_root, DETNET_LAYER, segment)
        if segment in detections and not writing.should_skip(detnet_target, force=config.force):
            with writing.atomic_recording(detnet_target, recording_id=segment, send_properties=False) as recording:
                rerun_layers.write_detnet_layer(recording, load_track(detections[segment]), read_ground_truth(entry, infos[segment]))
        print(f"{segment}: layers in {perf_counter() - started:.1f} s")


def storage_paths(segment_table: pa.Table, segment: str) -> dict[str, Path]:
    """The file behind each registered layer of a segment, from the dataset's segment table."""
    row: dict[str, list[str]] = segment_table.filter(pc.field("rerun_segment_id") == segment).select(["rerun_layer_names", "rerun_storage_urls"]).to_pylist()[0]
    return {layer: Path(unquote(urlparse(url).path)) for layer, url in zip(row["rerun_layer_names"], row["rerun_storage_urls"], strict=True)}


def export_clip(segment_table: pa.Table, segment: str, handtrack_rrd: Path, target: Path, dataset: str = catalog.UMETRACK) -> None:
    """One standalone recording: the segment's base and ground-truth layers, ``handtrack_v1`` and the handtrack blueprint."""
    sources: dict[str, Path] = storage_paths(segment_table, segment)
    required: tuple[str, ...] = ("base", "hand_pose") if catalog.is_show3d(dataset) else GT_LAYERS
    missing: list[str] = [layer for layer in required if layer not in sources]
    if missing:
        raise ValueError(f"{segment}: the catalog has no {missing} layer")
    streams: list[LazyChunkStream] = [RrdReader(path).stream() for path in (*(sources[layer] for layer in GT_LAYERS if layer in sources), handtrack_rrd)]
    with writing.atomic_write(target) as temp_path, rr.RecordingStream(application_id=writing.APPLICATION_ID, recording_id=segment, send_properties=False) as recording:
        recording.save(temp_path, default_blueprint=handtrack_blueprint(dataset), write_footer=True)
        rr.send_chunks(LazyChunkStream.merge(*streams), recording=recording)


def validate_base_segments(table: pa.Table, segments: list[str], dataset: str) -> None:
    """Reject inputs without base video before writing or registering derived layers."""
    bases: set[str] = {
        segment for segment, layers in zip(table["rerun_segment_id"].to_pylist(), table["rerun_layer_names"].to_pylist(), strict=True)
        if "base" in (layers or [])
    }
    missing: list[str] = sorted(set(segments) - bases)
    if missing:
        raise ValueError(f"{dataset} has no base segment for: {missing}")


def register_layers(config: Config, segments: list[str]) -> None:
    """Register the layer files of ``segments`` under their layer names, replacing an earlier registration of the same layer."""
    url: str = config.register_url or config.catalog_url
    entry: DatasetEntry = CatalogClient(url).get_dataset(config.dataset)
    validate_base_segments(entry.segment_table().to_arrow_table(), segments, config.dataset)
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
            handtrack_blueprint(config.dataset).save(writing.APPLICATION_ID, str(temp_path))
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
    segment_table: pa.Table = entry.segment_table().to_arrow_table()
    infos: dict[str, SegmentInfo] = {info.segment_id: info for info in catalog.segment_infos(config.dataset, segment_table)}
    selected: list[str] = sorted(set(tracks) | set(detections))
    validate_base_segments(segment_table, selected, config.dataset)
    if config.register and config.register_url and config.register_url != config.catalog_url:
        destination: DatasetEntry = CatalogClient(config.register_url).get_dataset(config.dataset)
        validate_base_segments(destination.segment_table().to_arrow_table(), selected, config.dataset)
    write_layer_files(config, entry, infos, tracks, detections)
    for clip in config.clips:
        started: float = perf_counter()
        target: Path = config.export_dir / f"{config.export_name}__{clip}.rrd"
        export_clip(segment_table, clip, layer_path(config.layers_root, HANDTRACK_LAYER, clip), target, config.dataset)
        print(f"{clip}: standalone {target} ({target.stat().st_size / 1e6:.1f} MB) in {perf_counter() - started:.1f} s")
    if config.register:
        register_layers(config, sorted(set(tracks) | set(detections)))
