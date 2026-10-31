"""Copy a served SelfCap base into a new, migrated local RRD.

Video packets stay encoded. Existing chunks pass through unchanged except for
video clocks and the time_sync JSON. The shared dataforge hand logger supplies
raw Quest hands on both clocks. This module never connects to a catalog.
"""

import math
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
from dataforge import paths, schema
from dataforge.quest_joint_poses import quest_hand_path
from dataforge.datasets.selfcap import exo_devices, log_quest_hands, raw_calibration_chunks, read_quest_hands
from dataforge.schema import CAPTURE_TIMELINE
from dataforge.video_encoding import read_video_start_ns
from dataforge.writing import atomic_write, recording_to
from jaxtyping import Int64
from numpy import ndarray
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from serde.json import to_json

from selfcap_curate.records import HAND_LATENCY_MS, RgbModel, TimeSync, read_frozen_sync

TIME_SYNC_ENTITY: str = "/__properties/time_sync"
PATCH_VERSION: int = 1


def raw_video(raw_dir: Path, name: str) -> Path:
    """Resolve legacy raw video metadata; fresh bases never use this path."""
    if name.startswith("exo/"):
        device = name.removeprefix("exo/")
        return raw_dir / "exo" / device / f"{device}.mp4"
    kind, stream = name.split("_", 1)
    return raw_dir / ("ego" if kind == "oak" else kind) / f"{stream}.mp4"


PatchMode: TypeAlias = Literal["fresh", "migrate-keep-clock", "migrate-realign"]


@dataclass(frozen=True, slots=True)
class Inventory:
    """Facts gathered in the one source-recording scan."""

    original: TimeSync | None
    """Embedded legacy clock record, if present."""
    present: set[str]
    """Temporal entity paths, including sparse marker chunks."""
    entities: tuple[str, ...]
    """Every source entity, sorted for the blueprint-envelope filter."""
    starts: dict[str, int]
    """First capture stamp per video in fresh mode."""


def scan(chunks: Iterable[Chunk], source: Path, *, capture_starts: bool) -> Inventory:
    """Inspect properties, temporal membership and first samples in one pass."""
    records: list[TimeSync] = []
    property_found: bool = False
    present: set[str] = set()
    entities: set[str] = set()
    starts: dict[str, int] = {}
    for chunk in chunks:
        entity = str(chunk.entity_path)
        entities.add(entity)
        if entity == TIME_SYNC_ENTITY:
            property_found = True
            batch = chunk.to_record_batch()
            if "record_json" in batch.schema.names:
                for values in batch.column("record_json").to_pylist():
                    if values:
                        records.append(read_frozen_sync(source, text=values[0]))
        if not chunk.is_static:
            present.add(entity)
            if capture_starts and entity.endswith("/pinhole/video"):
                stamp = int(np.min(np.asarray(chunk.to_record_batch().column(CAPTURE_TIMELINE).cast(pa.int64()))))
                starts[entity] = min(starts.get(entity, stamp), stamp)
    if len(records) > 1 or (property_found and not records):
        raise ValueError(f"{source}: expected one time_sync record_json, got {len(records)}")
    return Inventory(records[0] if records else None, present, tuple(sorted(entities)), starts)


@dataclass(frozen=True, slots=True)
class Retime:
    """Resolved clock operation for one temporal entity."""

    restore_ns: int
    """Container start added to capture_time and any retained video_time."""
    source_clock: Literal["capture_time", "video_time"]
    """Align from restored capture_time or preserve the already-aligned clock."""
    offset: int | RgbModel
    """Rounded constant nanoseconds, or the unchanged RGB drift calculation."""
    extra_ns: int = 0
    """Additional Quest-video shift only."""


def rewrite(chunk: Chunk, operation: Retime, source: Path) -> list[Chunk]:
    """Replace only the two clocks; retain column fields, RowIds and payloads."""
    batch = chunk.to_record_batch()
    capture_index, video_index = (batch.schema.get_field_index(name) for name in (CAPTURE_TIMELINE, schema.TIMELINE))
    if capture_index < 0 or video_index < 0:
        raise ValueError(f"{source}: {chunk.entity_path} lacks capture_time or video_time")
    capture: Int64[ndarray, "n"] = np.asarray(batch.column(capture_index).cast(pa.int64())) + operation.restore_ns
    if operation.source_clock == CAPTURE_TIMELINE:
        offset = (
            np.rint(operation.offset.offset_ms(capture / 1e9) * 1e6).astype(np.int64) if isinstance(operation.offset, RgbModel) else operation.offset
        )
        aligned = capture + offset
    else:
        aligned = np.asarray(batch.column(video_index).cast(pa.int64())) + operation.restore_ns
    aligned = aligned + operation.extra_ns
    batch = batch.set_column(capture_index, batch.schema.field(capture_index), pa.array(capture, type=batch.schema.field(capture_index).type))
    batch = batch.set_column(video_index, batch.schema.field(video_index), pa.array(aligned, type=batch.schema.field(video_index).type))
    return Chunk.from_record_batch(batch)


def patch(
    served_base: Path,
    raw_episode_dir: Path | None,
    sync_record: TimeSync,
    *,
    output_path: Path,
    mode: PatchMode,
    quest_video_delta_ms: float | None = None,
) -> Path:
    """Apply the stated fresh alignment, legacy clock restoration, or legacy realignment."""
    paths.require_outside([output_path], roots=[paths.NAS_ROOT, *([raw_episode_dir] if raw_episode_dir is not None else [])])
    if output_path.exists():
        raise FileExistsError(output_path)
    if quest_video_delta_ms is not None and not math.isfinite(quest_video_delta_ms):
        raise ValueError("quest_video_delta_ms must be finite")
    if sync_record.offsets_ms is None:
        raise ValueError("cannot patch a failed alignment record")
    reader: RrdReader = RrdReader(served_base)
    stores = reader.recordings()
    if len(stores) != 1 or stores[0].recording_id != sync_record.episode_key.segment:
        raise ValueError(f"{served_base}: expected the single SelfCap recording {sync_record.episode_key.segment}")
    source = stores[0]
    inventory = scan(reader.stream(store=source), served_base, capture_starts=mode == "fresh")
    original = inventory.original
    if original is not None and (original.base_patch_version is not None or original.video_start_ms is not None):
        raise ValueError(f"{served_base}: already patched")
    if original is not None and original.key != sync_record.key:
        raise ValueError(f"{served_base}: supplied sync record has the wrong episode key")
    if mode not in ("fresh", "migrate-keep-clock", "migrate-realign"):
        raise ValueError(f"unknown patch mode: {mode}")
    if (mode == "fresh") != (original is None):
        raise ValueError(f"{served_base}: {mode} requires {'no' if mode == 'fresh' else 'an embedded'} time_sync record")
    aligning: bool = mode != "migrate-keep-clock"
    pose_ms: float | None = sync_record.offsets_ms.get("quest_pose")
    if pose_ms is None or not math.isfinite(pose_ms):
        raise ValueError("Quest hands require a finite quest_pose offset")
    if original is not None and raw_episode_dir is None:
        raise ValueError("legacy served-base migration requires raw_episode_dir")
    phones: tuple[str, ...] = tuple(name.removeprefix("exo/") for name in sync_record.offsets_ms if name.startswith("exo/"))
    devices = exo_devices(raw_episode_dir) if original is not None and raw_episode_dir is not None else sorted(phones)
    if not devices:
        raise ValueError(f"{raw_episode_dir}: no readable exo videos and calibrations")
    oak_rig: int = len(devices)
    videos: dict[str, str] = {schema.video_path(rig, 0): f"exo/{device}" for rig, device in enumerate(devices)}
    videos.update(
        {
            schema.video_path(oak_rig + offset, cam): f"{kind}_{name}"
            for offset, kind, names in ((0, "oak", ("rgb", "left", "right")), (1, "quest", ("left", "right")))
            for cam, name in enumerate(names)
        }
    )
    present = inventory.present
    videos = {entity: name for entity, name in videos.items() if entity in present}
    starts: dict[str, int] = (
        {entity: read_video_start_ns(raw_video(raw_episode_dir, name)) for entity, name in videos.items()}
        if original is not None and raw_episode_dir is not None
        else {}
    )
    if mode == "fresh":
        starts = {entity: inventory.starts[entity] for entity in videos}
    if raw_episode_dir is not None and original is None:
        for entity, name in videos.items():
            expected: int = read_video_start_ns(raw_video(raw_episode_dir, name))
            if starts[entity] != expected:
                raise ValueError(f"{served_base}: {name} start mismatch: {starts[entity]} != {expected}")
    updated: TimeSync = replace(
        sync_record,
        video_start_ms={name: starts[entity] * 1e-6 for entity, name in videos.items()},
        quest_video_delta_ms=quest_video_delta_ms,
        base_patch_version=PATCH_VERSION,
        hand_latency_ms=HAND_LATENCY_MS,
    )
    record_json: str = to_json(updated)
    quest_delta_ns: int = round((quest_video_delta_ms or 0.0) * 1e6)

    csv_offsets: dict[str, float | None] = {
        schema.gyro_path(oak_rig, 0): sync_record.offsets_ms.get("oak_imu"),
        schema.accel_path(oak_rig, 0): sync_record.offsets_ms.get("oak_imu"),
        schema.rig_path(oak_rig + 1): pose_ms,
        schema.coco133_xyz_path(): pose_ms - HAND_LATENCY_MS,  # bases written before the joint poses were ingested
        quest_hand_path("left"): pose_ms - HAND_LATENCY_MS,
        quest_hand_path("right"): pose_ms - HAND_LATENCY_MS,
    }
    hand_entities: set[str] = {schema.coco133_xyz_path(), quest_hand_path("left"), quest_hand_path("right")}
    if unmapped := {entity for entity in present if entity.endswith("/pinhole/video")} - videos.keys():
        raise ValueError(f"{served_base}: video {sorted(unmapped)[0]} has no raw-file mapping")
    if aligning and (unknown := present - videos.keys() - csv_offsets.keys()):
        raise ValueError(f"{served_base}: unknown temporal entity {sorted(unknown)}")
    has_hands: bool = bool(hand_entities & present)
    if not has_hands and raw_episode_dir is None:
        raise ValueError("fresh base requires ingested Quest hands")
    plan: dict[str, Retime] = {}
    for entity, name in videos.items():
        restored = starts[entity] if mode != "fresh" else 0
        stream = "quest_video" if name.startswith("quest_") else name
        offset: int | RgbModel = 0
        if aligning:
            offset = sync_record.oak_rgb_model if name == "oak_rgb" else round((sync_record.offsets_ms.get(stream) or 0.0) * 1e6)
        plan[entity] = Retime(
            restored,
            CAPTURE_TIMELINE if aligning else schema.TIMELINE,
            offset,
            quest_delta_ns if name.startswith("quest_") else 0,
        )
    for entity, offset_ms in csv_offsets.items():
        if entity in present and (aligning or entity in hand_entities):
            plan[entity] = Retime(0, CAPTURE_TIMELINE, round((offset_ms or 0.0) * 1e6))

    def migrate(chunk: Chunk) -> Iterable[Chunk]:
        entity = str(chunk.entity_path)
        if entity == TIME_SYNC_ENTITY:
            if aligning:
                return []
            batch = chunk.to_record_batch()
            if "record_json" not in batch.schema.names:
                return [chunk]
            index = batch.schema.get_field_index("record_json")
            batch = batch.set_column(
                index, batch.schema.field(index), pa.array([[record_json]] * batch.num_rows, type=batch.schema.field(index).type)
            )
            return Chunk.from_record_batch(batch)
        if chunk.is_static or (operation := plan.get(entity)) is None:
            return [chunk]
        return rewrite(chunk, operation, served_base)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".selfcap-patch-", dir=output_path.parent) as staging, atomic_write(output_path) as temp_output:
        migrated: LazyChunkStream = reader.stream(store=source).flat_map(migrate)
        if original is not None:
            assert raw_episode_dir is not None
            migrated = LazyChunkStream.merge(
                migrated,
                LazyChunkStream.from_iter(
                    raw_calibration_chunks(raw_episode_dir, [v.removeprefix("exo/") for v in videos.values() if v.startswith("exo/")])
                ),
            )
        if not has_hands:
            assert raw_episode_dir is not None
            hands_path: Path = Path(staging) / "hands.rrd"
            with recording_to(hands_path, application_id=source.application_id, recording_id=source.recording_id, send_properties=False) as recording:
                log_quest_hands(recording, read_quest_hands(raw_episode_dir / "quest"), shift_ns=round((pose_ms - HAND_LATENCY_MS) * 1e6))
            hands_reader: RrdReader = RrdReader(hands_path)
            migrated = LazyChunkStream.merge(migrated, hands_reader.stream())
        if original is None or aligning:
            property_chunk: Chunk = Chunk.from_property(
                "time_sync",
                rr.AnyValues(record_json=record_json, method=updated.method, aligned_clock=updated.aligned_clock, flags=",".join(updated.flags)),
            )
            migrated = LazyChunkStream.merge(migrated, LazyChunkStream.from_iter([property_chunk]))
        migrated_path: Path = Path(staging) / "migrated.rrd"
        migrated.write_rrd(migrated_path, application_id=source.application_id, recording_id=source.recording_id)
        # Copy blueprint envelopes, including activation commands, verbatim.
        merge_inputs: list[Path] = [migrated_path]
        if reader.blueprints():
            blueprint_path: Path = Path(staging) / "blueprints.rbl"
            entities = inventory.entities
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rerun",
                    "rrd",
                    "filter",
                    str(served_base),
                    "--output",
                    str(blueprint_path),
                    *(arg for entity in entities for arg in ("--drop-entity", entity)),
                ],
                check=True,
                capture_output=True,
            )
            merge_inputs.append(blueprint_path)
        subprocess.run(
            [sys.executable, "-m", "rerun", "rrd", "route", "--output", str(temp_output), *(str(path) for path in merge_inputs)],
            check=True,
            capture_output=True,
        )
    return output_path
