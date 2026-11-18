"""Lens-plane masks to cube-face masks: remap the people masks of the two Insta360 lenses through the cube face tables.

Reads (a) the base recording (the two Mei lens cameras), (b) a mask layer with ``SegmentationImage`` uint8 at
``/world/rig_00/cam_0{0,1}/pinhole/people_mask`` at lens resolution on ``video_time``, and (c) the cube layer (each
face's ``Pinhole``, face-from-IMU ``Transform3D`` and ``source_lens``, and the face frame stamps). Writes a layer with
``SegmentationImage`` at ``/world/rig_00/cam_NN/pinhole/people_mask`` (NN = 02..11) on every face frame stamp, remapped
with nearest-neighbour interpolation; pixels outside the lens field are 0. A face stamp without a mask on its lens is
an error.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from dataforge import schema, writing
from dataforge.datasets import insta360_cube as cube
from dataforge.datasets.insta360 import RIG, LensCameras, cameras_from_recording

MASK_NAME: str = "people_mask"
"""The mask entity under each ``pinhole``, on the lenses and on the faces."""
FACE_CAMS: range = range(2, 12)
"""``cam_02 .. cam_11``: the cube faces."""


@dataclass
class Config:
    """Remap a lens-plane mask layer onto the cube faces."""

    base: Path
    """The base recording (lens cameras)."""
    masks: Path
    """The lens mask layer (``cam_0{0,1}/pinhole/people_mask``)."""
    faces: Path
    """The cube layer (face geometry and stamps)."""
    output: Path
    """The face mask layer to write."""


def face_statics(faces: Path) -> dict[int, dict[str, object]]:
    """Each face camera's static components, ``{cam: {component: value}}``."""
    statics: dict[int, dict[str, object]] = {cam: {} for cam in FACE_CAMS}
    entities: dict[str, int] = {path: cam for cam in FACE_CAMS for path in (schema.cam_path(RIG, cam), schema.pinhole_path(RIG, cam))}
    for chunk in rrc.RrdReader(faces).stream():
        if chunk.is_static and chunk.entity_path in entities:
            batch = chunk.to_record_batch()
            statics[entities[chunk.entity_path]].update(
                {name: batch.column(name).to_pylist()[0] for name in batch.schema.names if not name.startswith("rerun")}
            )
    return statics


def faces_from_layer(faces: Path, lenses: LensCameras) -> dict[int, cube.CubeFace]:
    """Each face camera of a cube layer as a ``CubeFace``, its rotation in its lens from the two stored poses.

    Raises:
        ValueError: If a face lacks its pinhole, pose or source lens.
    """
    out: dict[int, cube.CubeFace] = {}
    for cam, values in face_statics(faces).items():
        missing: list[str] = [
            key for key in ("Pinhole:image_from_camera", "Pinhole:resolution", "Transform3D:mat3x3", "source_lens", "name") if not values.get(key)
        ]
        if missing:
            raise ValueError(f"{faces}: cam_{cam:02d} lacks {missing}")
        lens: int = int(values["source_lens"][0])  # pyrefly: ignore[bad-index]
        # Rerun stores 3x3 matrices column-major.
        k: Float64[ndarray, "3 3"] = np.asarray(values["Pinhole:image_from_camera"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        face_R_rig: Float64[ndarray, "3 3"] = np.asarray(values["Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        width, height = (int(v) for v in values["Pinhole:resolution"][0])  # pyrefly: ignore[bad-index]
        name: str = str(values["name"][0])  # pyrefly: ignore[bad-index]
        lens_from_face: Float64[ndarray, "3 3"] = lenses.lens_from_rig[lens][:3, :3] @ face_R_rig.T
        out[cam] = cube.CubeFace(
            lens=lens, direction=name.rsplit("_", 1)[-1], lens_from_face=lens_from_face, intrinsics=k, width=width, height=height
        )
    return out


def face_stamps(faces: Path) -> Int64[ndarray, " k"]:
    """The face frame stamps (``video_time``, ns) of ``cam_02``; the cube layer gives every face the same ones."""
    entity: str = schema.video_path(RIG, FACE_CAMS[0])
    stamps: list[int] = []
    for chunk in rrc.RrdReader(faces).stream():
        if chunk.entity_path == entity and not chunk.is_static:
            batch = chunk.to_record_batch()
            if "VideoStream:sample" in batch.schema.names:
                stamps += batch.column("video_time").cast("int64").to_pylist()
    return np.unique(np.asarray(stamps, dtype=np.int64))


def lens_masks(masks: Path, stamps: set[int]) -> Iterator[tuple[int, int, UInt8[ndarray, "h w"]]]:
    """``(lens, stamp, mask)`` for every lens mask row of ``masks`` whose stamp is in ``stamps``."""
    entities: dict[str, int] = {f"{schema.pinhole_path(RIG, lens)}/{MASK_NAME}": lens for lens in range(2)}
    for chunk in rrc.RrdReader(masks).stream():
        if chunk.entity_path not in entities or chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        if "SegmentationImage:buffer" not in batch.schema.names:
            continue
        times: list[int] = batch.column("video_time").cast("int64").to_pylist()
        buffers = batch.column("SegmentationImage:buffer").to_pylist()
        formats = batch.column("SegmentationImage:format").to_pylist()
        for stamp, buffer, image_format in zip(times, buffers, formats, strict=True):
            if stamp in stamps and buffer:
                fmt: dict[str, int] = image_format[0]
                yield entities[chunk.entity_path], stamp, np.asarray(buffer[0], dtype=np.uint8).reshape(fmt["height"], fmt["width"])


def main(config: Config) -> None:
    """Write the face mask layer.

    Raises:
        ValueError: If a face stamp has no mask on its lens, or a lens mask is not at the lens resolution.
    """
    lenses: LensCameras = cameras_from_recording(config.base)
    faces: dict[int, cube.CubeFace] = faces_from_layer(config.faces, lenses)
    tables: dict[int, cube.RemapTable] = {cam: cube.remap_table(face, lenses.cameras[face.lens]) for cam, face in faces.items()}
    stamps: Int64[ndarray, " k"] = face_stamps(config.faces)
    seen: dict[int, set[int]] = {0: set(), 1: set()}
    lens_size: tuple[int, int] = (lenses.size[1], lenses.size[0])
    with writing.atomic_recording(config.output, recording_id=config.base.stem, send_properties=False) as recording:
        for lens, stamp, mask in lens_masks(config.masks, {int(s) for s in stamps}):
            if mask.shape != lens_size:
                raise ValueError(f"lens {lens} mask at {stamp} is {mask.shape}, the lens is {lens_size}")
            if stamp in seen[lens]:
                continue
            seen[lens].add(stamp)
            recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            for cam, face in faces.items():
                if face.lens == lens:
                    rr.log(
                        f"{schema.pinhole_path(RIG, cam)}/{MASK_NAME}", rr.SegmentationImage(cube.remap_mask(mask, tables[cam])), recording=recording
                    )
        for lens, found in seen.items():
            absent: list[int] = sorted({int(s) for s in stamps} - found)
            if absent:
                raise ValueError(f"lens {lens}: {len(absent)} face stamps have no mask in {config.masks}, first {absent[:5]}")
