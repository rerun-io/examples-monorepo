"""Union of mask layers (track 3's contract v2) into one layer: what SLAM and colsfm read when several maskers run.

Every input layer holds ``<camera>/pinhole/people_mask`` ``SegmentationImage``s (nonzero = masked) on the base
recording's timeline, each at its own resolution with an optional static ``Transform3D(scale=[sx, sy, 1])`` to image
pixels (none = scale 1). Per camera the union is written on the coarsest of the input grids: each finer mask is
max-pooled onto it (a coarse cell is masked when any of its pixels is), so nothing is upsampled and no masked pixel
is lost. The union exists at the stamps every layer has a mask on (a stride-5 and a stride-10 layer give every 10th
frame). The inputs are already dilated; the union adds none. Layers are read a chunk at a time and held bit-packed,
so a 17.8 min clip with two 480 px layers needs a few hundred MB per camera.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from posekit.mask_layer import MASK_ANNOTATION, MASK_NAME


@dataclass
class MaskUnionConfig:
    """OR several contract-v2 mask layers of one recording into one layer."""

    layers: tuple[Path, ...]
    """Mask layers of the same recording (same recording id and timeline)."""
    rr_config: RerunTyroConfig
    """Output sink: ``--rr-config.save <union.rrd> --rr-config.headless --rr-config.recording-id <id>
    --rr-config.application-id dataforge``."""
    timeline: str = "video_time"
    """Timeline of the masks."""


def pool_any(mask: Bool[ndarray, "*n h w"], factor: int) -> Bool[ndarray, "*n hc wc"]:
    """Max-pool binary masks by a whole ``factor`` (a cell is set when any of its pixels is).

    Raises:
        ValueError: If the mask size is not a multiple of ``factor``.
    """
    height, width = mask.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"a {width}x{height} mask does not pool by {factor}")
    return mask.reshape(*mask.shape[:-2], height // factor, factor, width // factor, factor).any(axis=(-3, -1))


def layer_statics(store: rrc.LazyStore) -> dict[str, Float64[ndarray, "2"]]:
    """Per mask entity (one with ``SegmentationImage`` buffers), its static scale ``[sx, sy]`` to image pixels (1 without one)."""
    scales: dict[str, Float64[ndarray, "2"]] = {
        column.entity_path: np.ones(2)
        for column in store.schema().component_columns()
        if column.component == "SegmentationImage:buffer" and column.entity_path.endswith(f"/pinhole/{MASK_NAME}")
    }
    for chunk in store.stream().filter(content=list(scales), is_static=True).to_chunks():
        batch = chunk.to_record_batch()
        if "Transform3D:scale" in batch.schema.names:
            scales[str(chunk.entity_path)] = np.asarray(batch.column("Transform3D:scale").to_pylist()[0][0][:2], dtype=np.float64)
    return scales


def layer_masks(
    store: rrc.LazyStore, entity: str, timeline: str, cell_px: Float64[ndarray, "2"], target_cell_px: Float64[ndarray, "2"]
) -> Iterator[tuple[int, Bool[ndarray, "h w"]]]:
    """One entity's masks with their stamps (ns) in file order, pooled from cells of ``cell_px`` image pixels to cells of
    ``target_cell_px``, a chunk at a time.

    Raises:
        ValueError: If the target cell is not a whole multiple of this layer's cell.
    """
    ratio: Float64[ndarray, "2"] = target_cell_px / cell_px
    factor: int = int(round(float(ratio[0])))
    if abs(ratio[0] - factor) > 1e-6 or abs(ratio[1] - factor) > 1e-6:
        raise ValueError(f"{entity}: cells of {target_cell_px} px are not a whole multiple of {cell_px} px")
    for chunk in store.stream().filter(content=[entity], is_static=False):
        batch = chunk.to_record_batch()
        if "SegmentationImage:buffer" not in batch.schema.names:
            continue
        stamps: Int64[ndarray, " n"] = np.asarray(batch.column(timeline).to_numpy(zero_copy_only=False)).view(np.int64)
        fmt: dict = batch.column("SegmentationImage:format").to_pylist()[0][0]
        values: ndarray = batch.column("SegmentationImage:buffer").flatten().flatten().to_numpy(zero_copy_only=False)
        stacked: Bool[ndarray, "n h w"] = values.reshape(len(stamps), fmt["height"], fmt["width"]) > 0
        yield from zip(stamps.tolist(), pool_any(stacked, factor), strict=True)


def main(config: MaskUnionConfig) -> None:
    """Write the union layer and print the per-camera stamp counts."""
    start: float = time.perf_counter()
    readers: list[rrc.RrdReader] = [rrc.RrdReader(path) for path in config.layers]
    ids: set[str] = {entry.recording_id for reader in readers for entry in reader.recordings()}
    if ids != {config.rr_config.recording_id}:
        raise ValueError(f"the layers belong to {sorted(ids)}; pass --rr-config.recording-id with the one they share")
    stores: list[rrc.LazyStore] = [reader.store() for reader in readers]
    statics: list[dict[str, Float64[ndarray, "2"]]] = [layer_statics(store) for store in stores]
    entities: list[str] = sorted(set.intersection(*(set(s) for s in statics)))
    if not entities:
        raise ValueError("the layers share no mask entity")
    for entity in entities:
        target_cell_px: Float64[ndarray, "2"] = np.max(np.stack([s[entity] for s in statics]), axis=0)
        per_layer: list[dict[int, UInt8[ndarray, " b"]]] = []
        shapes: set[tuple[int, int]] = set()
        for store, s in zip(stores, statics, strict=True):
            packed: dict[int, UInt8[ndarray, " b"]] = {}
            for stamp, mask in layer_masks(store, entity, config.timeline, s[entity], target_cell_px):
                packed[stamp] = np.packbits(mask)
                shapes.add(mask.shape)
            per_layer.append(packed)
        if len(shapes) > 1:
            raise ValueError(f"{entity}: the layers pool to different grids {sorted(shapes)}")
        stamps: list[int] = sorted(set.intersection(*(set(m) for m in per_layer)))
        rr.log(entity, MASK_ANNOTATION, static=True)
        rr.log(entity, rr.Transform3D(scale=[float(target_cell_px[0]), float(target_cell_px[1]), 1.0]), static=True)
        height, width = next(iter(shapes), (0, 0))
        for stamp in stamps:
            packed_union: UInt8[ndarray, " b"] = np.bitwise_or.reduce([m[stamp] for m in per_layer])
            union: UInt8[ndarray, "h w"] = np.unpackbits(packed_union, count=height * width).reshape(height, width)
            rr.set_time(config.timeline, duration=np.timedelta64(stamp, "ns"))
            rr.log(entity, rr.SegmentationImage(union))
        sizes: str = ", ".join(str(len(m)) for m in per_layer)
        print(f"{entity}: layers have {sizes} masks; union on {len(stamps)} shared stamps, cell {target_cell_px.tolist()} px")
    recording: rr.RecordingStream | None = rr.get_data_recording()
    assert recording is not None
    recording.flush()
    print(f"union written in {time.perf_counter() - start:.1f} s")
