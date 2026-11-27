"""Read the Insta360 cube-face layer (and its people-mask layer) from a local ``.rrd``.

Faces are the camera nodes ``/world/rig_NN/cam_NN`` whose static ``name`` the layer's ``cube`` property lists in
``faces`` (dataforge's ``insta360`` cube layer); other cameras (the raw lenses) are skipped. Geometry comes from the
layer's static ``Pinhole`` and ``Transform3D`` (face-from-rig), never from constants.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from jaxtyping import Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.camera_parameters import PinholeParameters
from simplecv.catalog_calibration import read_camera_calibration
from simplecv.catalog_layers import segment_properties
from simplecv.catalog_video_reader import ChunkedVideoDecoder, open_decoders, select_decode_device
from simplecv.rrd_query_utils import first_valid_value

from monopriors.cube_faces import CubeFace

CAMERA_NODE_PATTERN: re.Pattern[str] = re.compile(r"^/world/rig_\d+/cam_\d+$")
"""Camera nodes as simplecv's catalog discovery names them."""


@dataclass(frozen=True, slots=True)
class LayerSegment:
    """One recording of a local ``.rrd`` served as a catalog dataset."""

    dataset: DatasetEntry
    """The dataset on the private server."""
    segment_id: str
    """The single segment (recording id) of the file."""


@contextmanager
def open_rrd(path: Path, name: str) -> Iterator[LayerSegment]:
    """Serve one ``.rrd`` on a private in-process server for the duration of the block.

    Raises:
        ValueError: If the file holds other than exactly one recording.
    """
    with rr.server.Server(datasets={name: [str(path)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset(name)
        segment_ids: list[str] = list(dataset.segment_ids())
        if len(segment_ids) != 1:
            raise ValueError(f"{path}: expected one recording, found {len(segment_ids)}")
        yield LayerSegment(dataset=dataset, segment_id=segment_ids[0])


def read_cube_faces(layer: LayerSegment) -> tuple[CubeFace, ...]:
    """The cube faces of the layer, ordered by entity, with K, size and face-from-rig rotation.

    Raises:
        ValueError: If the layer's ``cube`` property lists no faces, or its camera nodes do not name exactly those faces.
    """
    entities: list[str] = sorted(
        {
            f"/{str(column.entity_path).lstrip('/')}"
            for column in layer.dataset.schema().component_columns()
            if str(column.component) == "name" and CAMERA_NODE_PATTERN.match(f"/{str(column.entity_path).lstrip('/')}")
        }
    )
    faces: tuple[str, ...] = tuple(str(name) for name in segment_properties(layer.dataset, layer.segment_id, "cube").get("faces", []))
    if not faces:
        raise ValueError(f"segment {layer.segment_id}: no property:cube:faces (a cube layer written before dataforge recorded its faces)")
    table: pa.Table = layer.dataset.filter_segments(layer.segment_id).filter_contents(entities).reader(index=None).to_arrow_table()
    names: dict[str, str] = {}
    for entity in entities:
        value: object = first_valid_value(table.column(f"{entity}:name"), component_name=f"{entity}:name")
        name: str = str(np.asarray(value, dtype=str).reshape(-1)[0])
        if name in faces:
            names[entity] = name
    if sorted(names.values()) != sorted(faces):
        raise ValueError(f"segment {layer.segment_id}: the cube property lists faces {list(faces)}, the camera nodes name {sorted(names.values())}")
    calibration: dict[str, PinholeParameters] = read_camera_calibration(layer.dataset, layer.segment_id, list(names))
    return tuple(
        CubeFace(
            entity=entity,
            name=name,
            k_matrix=np.asarray(calibration[entity].intrinsics.k_matrix, dtype=np.float64),
            width=int(calibration[entity].intrinsics.width),
            height=int(calibration[entity].intrinsics.height),
        )
        for entity, name in names.items()
    )


@dataclass(frozen=True, slots=True)
class FaceFrameset:
    """The decoded images of all faces at one ``video_time`` stamp."""

    t_ns: int
    """The frameset's stamp on the layer's timeline."""
    rgb_by_face: dict[str, UInt8[ndarray, "h w 3"]]
    """RGB image per face name."""


def iter_framesets(layer: LayerSegment, faces: tuple[CubeFace, ...], timeline: str, wanted_t_ns: set[int] | None = None) -> Iterator[FaceFrameset]:
    """Decode all faces in lockstep (NVDEC whenever CUDA is available) and yield the framesets (all, or those in ``wanted_t_ns``).

    Raises:
        ValueError: If the faces do not share the same stamps, or a frame has another size than its face's Pinhole.
    """
    device: torch.device = torch.device(select_decode_device())
    decoders: tuple[ChunkedVideoDecoder, ...] = open_decoders(
        layer.dataset, layer.segment_id, [f"{face.entity}/pinhole/video" for face in faces], timeline, device
    )
    stamps: Int64[ndarray, " n_frames"] = decoders[0].index.t_ns.copy()
    for face, decoder in zip(faces, decoders, strict=True):
        if not np.array_equal(decoder.index.t_ns, stamps):
            raise ValueError(f"{face.name}: its frame stamps differ from {faces[0].name}'s; all faces of a frameset share one stamp")
    for position, t_ns in enumerate(stamps.tolist()):
        if wanted_t_ns is not None and t_ns not in wanted_t_ns:
            continue
        rgb_by_face: dict[str, UInt8[ndarray, "h w 3"]] = {}
        for face, decoder in zip(faces, decoders, strict=True):
            frame: UInt8[torch.Tensor, "3 h w"] = decoder.get_frame_at(position)
            if tuple(frame.shape[1:]) != (face.height, face.width):
                raise ValueError(f"{face.name}: decoded {frame.shape[2]}x{frame.shape[1]}, the Pinhole says {face.width}x{face.height}")
            rgb_by_face[face.name] = frame.permute(1, 2, 0).cpu().numpy()
        yield FaceFrameset(t_ns=int(t_ns), rgb_by_face=rgb_by_face)


def read_people_masks(layer: LayerSegment, face: CubeFace, timeline: str, wanted_t_ns: set[int]) -> dict[int, UInt8[ndarray, "h w"]]:
    """The face's ``people_mask`` (1 = person) at the wanted stamps; stamps without a mask are absent.

    Raises:
        ValueError: If a mask has another size than the face.
    """
    entity: str = f"{face.entity}/pinhole/people_mask"
    buffer_column: str = f"{entity}:SegmentationImage:buffer"
    table: pa.Table = (
        layer.dataset.filter_segments(layer.segment_id)
        .filter_contents([entity])
        .reader(index=timeline)
        .select(timeline, buffer_column)
        .to_arrow_table()
    )
    table = table.filter(table[buffer_column].is_valid())
    stamps: Int64[ndarray, " n"] = table[timeline].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
    buffers: pa.ListArray = table[buffer_column].combine_chunks()
    masks: dict[int, UInt8[ndarray, "h w"]] = {}
    for row, t_ns in enumerate(stamps.tolist()):
        if t_ns not in wanted_t_ns:
            continue
        flat: UInt8[ndarray, " n_pixels"] = np.asarray(buffers[row].values.flatten(), dtype=np.uint8)
        if flat.size != face.width * face.height:
            raise ValueError(f"{entity} @ {t_ns}: {flat.size} pixels, the face is {face.width}x{face.height}")
        masks[int(t_ns)] = flat.reshape(face.height, face.width)
    return masks
