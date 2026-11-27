"""Lens-plane masks to cubemap masks: join the people masks of the two Insta360 lenses into the six cube faces.

Reads (a) the base recording (the two Mei lens cameras), (b) a mask layer with ``SegmentationImage`` uint8 at
``/world/rig_00/cam_0{0,1}/pinhole/people_mask`` on ``video_time`` (mask contract v2: at the resolution it was computed at,
with a static ``Transform3D`` scale to the lens image; no transform = lens resolution), and (c) the cube layer (each
face's name, ``Pinhole`` and face-from-IMU ``Transform3D``, and the face frame stamps). Writes a layer with
``SegmentationImage`` at ``/world/rig_00/cam_NN/pinhole/people_mask`` (NN = 02..07) on every face frame stamp, joined
from the two lens masks through the faces' tables with nearest-neighbour interpolation: one gather per face from the
native-resolution masks (``insta360_cube.MaskJoin``, on the GPU when there is one). Every face pixel lies inside the
lens it is read from (``insta360_cube.face_table`` checks it), so the mask is the people mask alone. A face stamp
without a mask on both lenses is an error.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.chunk as rrc
import torch
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from simplecv.run_record import RunStart, start_run, write_run_record

from dataforge import paths, schema, writing
from dataforge.datasets import insta360_cube as cube
from dataforge.datasets.insta360 import FACE_CAMS, RIG, LensCameras, cameras_from_recording, layer_device


@dataclass
class Config:
    """Join a lens-plane mask layer into the cube faces."""

    base: Path
    """The base recording (lens cameras)."""
    masks: Path
    """The lens mask layer (``cam_0{0,1}/pinhole/people_mask``)."""
    faces: Path
    """The cube layer (face geometry and stamps)."""
    output: Path
    """The face mask layer to write."""
    device: Literal["cuda", "cpu"] | None = None
    """Where the faces are gathered (``insta360_cube.MaskJoin``); default ``cuda`` when torch sees a GPU."""


def face_statics(faces: Path) -> dict[int, dict[str, object]]:
    """Each face camera's static components, ``{cam: {component: value}}``."""
    statics: dict[int, dict[str, object]] = {cam: {} for cam in FACE_CAMS}
    entities: dict[str, int] = {path: cam for cam in FACE_CAMS for path in (schema.cam_path(RIG, cam), schema.pinhole_path(RIG, cam))}
    for chunk in rrc.RrdReader(faces).store().stream().filter(is_static=True, content=list(entities)):
        if chunk.entity_path in entities:
            batch = chunk.to_record_batch()
            statics[entities[chunk.entity_path]].update(
                {name: batch.column(name).to_pylist()[0] for name in batch.schema.names if not name.startswith("rerun")}
            )
    return statics


def faces_from_layer(faces: Path, lenses: LensCameras) -> dict[int, cube.CubeFace]:
    """Each face camera of a cube layer as a ``CubeFace``: its rotation in the cube frame from the stored pose.

    Raises:
        ValueError: If a face lacks its name, pinhole or pose, or its pinhole is not the cubemap's.
    """
    out: dict[int, cube.CubeFace] = {}
    lens0_R_rig: Float64[ndarray, "3 3"] = lenses.lens_from_rig[0][:3, :3]
    for cam, values in face_statics(faces).items():
        missing: list[str] = [key for key in ("Pinhole:image_from_camera", "Pinhole:resolution", "Transform3D:mat3x3", "name") if not values.get(key)]
        if missing:
            raise ValueError(f"{faces}: cam_{cam:02d} lacks {missing}")
        # Rerun stores 3x3 matrices column-major.
        k: Float64[ndarray, "3 3"] = np.asarray(values["Pinhole:image_from_camera"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        face_R_rig: Float64[ndarray, "3 3"] = np.asarray(values["Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T  # pyrefly: ignore[bad-index]
        width, height = (int(v) for v in values["Pinhole:resolution"][0])  # pyrefly: ignore[bad-index]
        name: str = str(values["name"][0])  # pyrefly: ignore[bad-index]
        face = cube.CubeFace(name=name, cube_from_face=lens0_R_rig @ face_R_rig.T, size=width)
        if width != height or not np.allclose(k, face.intrinsics):
            raise ValueError(f"{faces}: cam_{cam:02d} pinhole {k.tolist()} at {width}x{height} is not a cubemap face")
        out[cam] = face
    return out


def face_stamps(faces: Path) -> Int64[ndarray, " k"]:
    """The face frame stamps (``video_time``, ns) of the first face; the cube layer gives every face the same ones."""
    entity: str = schema.video_path(RIG, FACE_CAMS[0])
    stamps: list[int] = []
    for chunk in rrc.RrdReader(faces).store().stream().filter(is_static=False, content=entity):
        if chunk.entity_path == entity:
            batch = chunk.to_record_batch()
            if "VideoStream:sample" in batch.schema.names:  # the stamp column only: the samples are not read out
                stamps += batch.column("video_time").cast("int64").to_pylist()
    return np.unique(np.asarray(stamps, dtype=np.int64))


def lens_masks(masks: Path, stamps: set[int]) -> Iterator[tuple[int, int, UInt8[ndarray, "h w"]]]:
    """``(lens, stamp, mask)`` for every lens mask row of ``masks`` whose stamp is in ``stamps``."""
    entities: dict[str, int] = {schema.people_mask_path(RIG, lens): lens for lens in range(2)}
    for chunk in rrc.RrdReader(masks).stream():
        if chunk.entity_path not in entities or chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        if "SegmentationImage:buffer" not in batch.schema.names:
            continue
        times: list[int] = batch.column("video_time").cast("int64").to_pylist()
        buffers = batch.column("SegmentationImage:buffer")  # list<list<uint8>>: one image buffer per row
        formats = batch.column("SegmentationImage:format").to_pylist()
        for row, (stamp, image_format) in enumerate(zip(times, formats, strict=True)):
            if stamp in stamps and image_format:
                fmt: dict[str, int] = image_format[0]
                # Zero-copy view of the row's uint8 buffer; a Python-list round trip cost 60 % of the remap.
                pixels: UInt8[ndarray, " n"] = buffers[row].values[0].values.to_numpy(zero_copy_only=True)
                yield entities[chunk.entity_path], stamp, pixels.reshape(fmt["height"], fmt["width"])


def lens_mask_annotation(masks: Path) -> rr.AnnotationContext | None:
    """The lens masks' own ``AnnotationContext`` (posekit logs one on each lens mask: kept clear, masked tinted), to
    draw the face masks the same way; None when the mask layer has none."""
    entity: str = schema.people_mask_path(RIG, 0)
    for chunk in rrc.RrdReader(masks).store().stream().filter(is_static=True, content=[entity]).to_chunks():
        batch = chunk.to_record_batch()
        if chunk.entity_path == entity and "AnnotationContext:context" in batch.schema.names:
            entries = batch.column("AnnotationContext:context").to_pylist()[0][0]
            return rr.AnnotationContext([(entry["class_id"], entry["class_description"]["info"]["label"], entry["class_description"]["info"]["color"]) for entry in entries])
    return None


def lens_mask_scales(masks: Path) -> dict[int, tuple[float, float]]:
    """Each lens mask entity's static ``Transform3D`` scale ``(sx, sy)``; a lens without one is absent (scale 1)."""
    entities: dict[str, int] = {schema.people_mask_path(RIG, lens): lens for lens in range(2)}
    scales: dict[int, tuple[float, float]] = {}
    # The manifest's lazy store reads only the chunks it keeps (a full stream decodes every mask: 14 s on clip 007).
    for chunk in rrc.RrdReader(masks).store().stream().filter(is_static=True, content=list(entities)):
        if chunk.entity_path in entities:
            batch = chunk.to_record_batch()
            if "Transform3D:scale" in batch.schema.names and (value := batch.column("Transform3D:scale").to_pylist()[0]):
                scales[entities[chunk.entity_path]] = (float(value[0][0]), float(value[0][1]))
    return scales


def check_mask_covers_lens(shape: tuple[int, int], scale: tuple[float, float], image_shape: tuple[int, int]) -> None:
    """Mask contract v2: an ``hm x wm`` mask times its scale is the lens image size to within half a pixel.

    Raises:
        ValueError: If it is not.
    """
    (hm, wm), (h, w) = shape, image_shape
    if abs(wm * scale[0] - w) >= 0.5 or abs(hm * scale[1] - h) >= 0.5:
        raise ValueError(f"a {wm}x{hm} mask at scale {scale} does not cover the {w}x{h} lens image")


def main(config: Config) -> None:
    """Write the face mask layer.

    Raises:
        ValueError: If a face stamp has no mask on both lenses, or a lens mask and its scale do not cover the lens image.
    """
    run: RunStart = start_run(config)
    lenses: LensCameras = cameras_from_recording(config.base)
    faces: dict[int, cube.CubeFace] = faces_from_layer(config.faces, lenses)
    device: torch.device = layer_device(config.device)
    join = cube.MaskJoin([cube.face_table(face, lenses.cameras, lenses.lens1_from_lens0, lenses.size) for face in faces.values()], lenses.size, device)
    wanted: set[int] = {int(s) for s in face_stamps(config.faces)}
    lens_shape: tuple[int, int] = (lenses.size[1], lenses.size[0])
    scales: dict[int, tuple[float, float]] = lens_mask_scales(config.masks)
    pending: dict[int, dict[int, UInt8[ndarray, "h w"]]] = {}
    done: set[int] = set()
    recording_id: str = rrc.RrdReader(config.base).recordings()[0].recording_id
    annotation: rr.AnnotationContext | None = lens_mask_annotation(config.masks)
    with writing.atomic_recording(config.output, recording_id=recording_id, send_properties=False) as recording:
        for cam in faces:
            if annotation is not None:
                rr.log(schema.people_mask_path(RIG, cam), annotation, static=True, recording=recording)
        for lens, stamp, mask in lens_masks(config.masks, wanted):
            if stamp in done:
                continue
            check_mask_covers_lens(mask.shape, scales.get(lens, (1.0, 1.0)), lens_shape)
            pending.setdefault(stamp, {})[lens] = mask
            if len(pending[stamp]) < 2:
                pending[stamp][lens] = mask.copy()  # the view would hold its whole Arrow chunk until the pair completes
                continue
            pair: dict[int, UInt8[ndarray, "h w"]] = pending.pop(stamp)
            done.add(stamp)
            recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            for cam, joined in zip(faces, join.faces(pair[0], pair[1]), strict=True):
                rr.log(schema.people_mask_path(RIG, cam), rr.SegmentationImage(joined), recording=recording)
        absent: list[int] = sorted(wanted - done)
        if absent:
            raise ValueError(f"{len(absent)} face stamps have no mask on both lenses in {config.masks}, first {absent[:5]}")
        write_run_record(recording, run.finish(paths.MASKS_LAYER, recording_id))
