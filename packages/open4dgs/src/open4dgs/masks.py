"""Foreground (union of people) mattes: BiRefNet on a person crop of the stored frames, a ``people_matte`` catalog layer, and its reader.

The matte model is BiRefNet (``ZhengPeng7/BiRefNet``, the weights fdanyone already pins under
``$FDANYONE_MODEL_DIR/birefnet``), run in fp16 at 1024x1024 on a crop around the people: the crop is the box of the
segment's SMPL-X vertices projected with the camera's pinhole, padded so hair, hands and loose clothes stay inside.
The crop keeps the studio hardware and other cameras out of the matte and gives a small person the full model resolution.

Layer: one ``EncodedImage`` (8-bit grayscale PNG, soft matte, 255 = person) per camera and frame at
``/world/rig_NN/cam_00/pinhole/people_matte`` (posekit's mask layer contract, ``posekit.mask_layer.MATTE_NAME``) on the
``frame_index`` timeline, at the STORED video size and grid
(no undistortion), so it overlays the stored video in the viewer. ``read_clip_masks`` applies the clip's undistortion
and resize (``catalog_clip.resample_images``), so mask and frame pixels agree.
"""

import importlib.util
import os
import sys
import types
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import torch
import torch.nn.functional as F
from datafusion import col
from jaxtyping import Float16, Float32, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_layers import atomic_recording, ensure_dataset, register_layer
from torch import Tensor

from open4dgs.catalog_clip import FRAME_TIMELINE, RESAMPLE_CHUNK, MultiViewClip, resample_images, roi_grid, undistortion_grid
from open4dgs.properties import MatteLayerProperties, any_values

MATTE_NAME: str = "people_matte"
"""Leaf entity of a camera's matte under ``<camera>/pinhole``: ``posekit.mask_layer.MATTE_NAME`` (posekit is not in the
open4dgs envs; ``tests/test_masks.py`` pins the two equal where posekit imports)."""
MATTE_LAYER: str = MATTE_NAME
"""Catalog layer name (and recording property) of the foreground mattes."""
MODEL_NAME: str = "birefnet"
"""Recorded in the layer's properties."""
MODEL_SIZE: int = 1024
"""BiRefNet's input side (fdanyone ``FOREGROUND.image_size``)."""
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DECODE_THREADS: int = min(16, os.cpu_count() or 1)
"""PNG decode threads of ``read_clip_masks`` (``cv2.imdecode`` releases the GIL)."""


MESH_PAD_FRACTION: float = 0.15
"""Crop margin around projected SMPL-X vertices, as a fraction of their box (hair, loose clothes); ``person_boxes``' default for meshes."""
MESH_PAD_PX: float = 24.0
"""... plus this many pixels (a lens shift of up to 21 px on MAMMA: the box uses the pinhole on distorted frames)."""


def mask_entity(camera_name: str) -> str:
    """The layer entity of a camera's foreground matte, e.g. ``/world/rig_01/cam_00/pinhole/people_matte``."""
    return f"/world/{camera_name}/pinhole/{MATTE_NAME}"


def load_birefnet(model_dir: Path, device: torch.device) -> torch.nn.Module:
    """BiRefNet from a local Hugging Face snapshot, fp16, eval mode.

    The snapshot's ``birefnet.py`` imports ``kornia.filters.laplacian`` but calls it only while training; when kornia is
    absent a stand-in module that raises on use satisfies the import (transformers imports every module a remote file names).
    """
    from transformers import AutoModelForImageSegmentation

    if importlib.util.find_spec("kornia") is None and "kornia" not in sys.modules:
        kornia = types.ModuleType("kornia")
        filters = types.ModuleType("kornia.filters")

        def laplacian(*_args: object, **_kwargs: object) -> Tensor:
            raise RuntimeError("kornia stand-in: BiRefNet's laplacian runs only in training")

        filters.laplacian = laplacian  # type: ignore[attr-defined]
        kornia.filters = filters  # type: ignore[attr-defined]
        sys.modules["kornia"] = kornia
        sys.modules["kornia.filters"] = filters
    model = AutoModelForImageSegmentation.from_pretrained(str(model_dir.expanduser().resolve()), local_files_only=True, trust_remote_code=True)
    return model.eval().half().to(device)


def person_boxes(
    vertices: Float32[ndarray, "frames verts 3"],
    world_T_cam: Float32[ndarray, "4 4"],
    K: Float32[ndarray, "3 3"],
    image_wh: tuple[int, int],
    pad_fraction: float,
    pad_px: float,
) -> Float32[ndarray, "frames 4"]:
    """Per frame, the padded box ``x0, y0, x1, y1`` (pixels, clipped to the image) of the vertices in front of the camera.

    ``vertices`` is the union of people at each frame (NaN rows are ignored). A frame with no projected vertex gets a
    NaN box; a frame whose box misses the image gets an empty box (``x0 == x1``).
    """
    cam_T_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_T_cam.astype(np.float64)).astype(np.float32)
    points: Float32[ndarray, "frames verts 3"] = vertices @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
    in_front: np.ndarray = np.isfinite(points).all(axis=-1) & (points[..., 2] > 1e-3)
    uv: Float32[ndarray, "frames verts 2"] = (points @ K.T)[..., :2] / np.where(in_front, points[..., 2], 1.0)[..., None]
    boxes: Float32[ndarray, "frames 4"] = np.full((len(vertices), 4), np.nan, dtype=np.float32)
    width, height = image_wh
    for frame in range(len(vertices)):
        if not in_front[frame].any():
            continue
        lo: Float32[ndarray, "2"] = uv[frame][in_front[frame]].min(axis=0)
        hi: Float32[ndarray, "2"] = uv[frame][in_front[frame]].max(axis=0)
        pad: Float32[ndarray, "2"] = (hi - lo) * pad_fraction + pad_px
        lo = np.clip(lo - pad, 0, [width, height])
        hi = np.clip(hi + pad, 0, [width, height])
        boxes[frame] = [lo[0], lo[1], max(hi[0], lo[0]), max(hi[1], lo[1])]
    return boxes


def fill_missing_boxes(boxes: Float32[ndarray, "frames 4"]) -> Float32[ndarray, "frames 4"]:
    """Give each NaN box the union of the nearest valid boxes before and after it (bodies stored every 3rd frame)."""
    valid: np.ndarray = np.isfinite(boxes).all(axis=1)
    if valid.all() or not valid.any():
        return boxes
    filled: Float32[ndarray, "frames 4"] = boxes.copy()
    rows: Int64[ndarray, " v"] = np.flatnonzero(valid)
    for frame in np.flatnonzero(~valid):
        after: int = int(np.searchsorted(rows, frame))
        near: list[int] = [int(rows[i]) for i in (after - 1, after) if 0 <= i < len(rows)]
        filled[frame] = [boxes[near, 0].min(), boxes[near, 1].min(), boxes[near, 2].max(), boxes[near, 3].max()]
    return filled


def predict_mattes(
    model: torch.nn.Module,
    frames: UInt8[Tensor, "n h w 3"],
    boxes: Float32[ndarray, "n 4"],
    device: torch.device,
    batch_size: int = 8,
) -> UInt8[Tensor, "n h w"]:
    """BiRefNet mattes (255 = person) of each frame inside its box; zero outside it, the whole frame for a NaN box."""
    height, width = frames.shape[1], frames.shape[2]
    mean: Tensor = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std: Tensor = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out: UInt8[Tensor, "n h w"] = torch.zeros((len(frames), height, width), dtype=torch.uint8)
    crops: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not np.isfinite(box).all():
            crops.append((0, 0, width, height))
        else:
            crops.append((int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3]))))
    todo: list[int] = [i for i, (x0, y0, x1, y1) in enumerate(crops) if x1 - x0 >= 8 and y1 - y0 >= 8]
    for start in range(0, len(todo), batch_size):
        batch: list[int] = todo[start : start + batch_size]
        inputs: list[Tensor] = []
        for i in batch:
            x0, y0, x1, y1 = crops[i]
            crop: Tensor = frames[i, y0:y1, x0:x1].to(device).permute(2, 0, 1)[None].float() / 255.0
            inputs.append(F.interpolate(crop, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", antialias=True, align_corners=False))
        with torch.inference_mode():
            logits: Tensor = model(((torch.cat(inputs) - mean) / std).half())[-1]
        probs: Tensor = logits.float().sigmoid()
        for slot, i in enumerate(batch):
            x0, y0, x1, y1 = crops[i]
            matte: Tensor = F.interpolate(probs[slot : slot + 1], size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0]
            out[i, y0:y1, x0:x1] = (matte * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
    return out


def camera_matte(
    model: torch.nn.Module,
    frames: UInt8[Tensor, "n h w 3"],
    box_points: Float32[ndarray, "n k 3"],
    world_T_cam: Float32[ndarray, "4 4"],
    K: Float32[ndarray, "3 3"],
    image_wh: tuple[int, int],
    pad_fraction: float,
    pad_px: float,
    device: torch.device,
    batch_size: int = 8,
) -> UInt8[Tensor, "n h w"]:
    """The person matte of one camera's frames: BiRefNet inside the padded box of ``box_points`` (world, per frame: mesh
    vertices or joint proxies) projected into the camera; a frame without a box takes the union of its nearest boxes.

    The one matte pipeline: ``clip_mattes`` (``open4dgs-masks``, SelfCap) and the fdanyone app call it."""
    boxes: Float32[ndarray, "n 4"] = fill_missing_boxes(person_boxes(box_points, world_T_cam, K, image_wh, pad_fraction, pad_px))
    return predict_mattes(model, frames, boxes, device, batch_size)


def clip_mattes(
    clip: MultiViewClip,
    box_points: Callable[[int], Float32[ndarray, "n k 3"]],
    model: torch.nn.Module,
    pad_fraction: float,
    pad_px: float,
    device: torch.device,
    batch_size: int = 8,
) -> list[UInt8[Tensor, "n h w"]]:
    """``camera_matte`` of every clip camera; ``box_points(camera)`` gives that camera's box points per frame."""
    mattes: list[UInt8[Tensor, "n h w"]] = []
    for cam, name in enumerate(clip.camera_names):
        matte: UInt8[Tensor, "n h w"] = camera_matte(
            model, clip.frames[cam], box_points(cam), clip.world_T_cam[cam].numpy(), clip.K[cam].numpy(), clip.sizes[cam], pad_fraction, pad_px, device, batch_size
        )
        print(f"{name}: foreground fraction {matte.float().mean().item() / 255:.4f}")
        mattes.append(matte)
    return mattes


def write_mask_layer(
    path: Path,
    segment_id: str,
    camera_names: Sequence[str],
    frame_indices: Int64[ndarray, " t"],
    mattes: Sequence[UInt8[Tensor, "t h w"]],
    properties: MatteLayerProperties,
) -> None:
    """Write the ``people_matte`` layer recording (recording id = ``segment_id``) with one grayscale PNG per camera and frame."""
    with atomic_recording(path, application_id="open4dgs-masks", recording_id=segment_id, send_properties=False) as rec:
        rec.send_property(MATTE_LAYER, any_values(properties.to_properties()))
        for camera, matte in zip(camera_names, mattes, strict=True):
            entity: str = mask_entity(camera)
            blobs: list[bytes] = []
            for frame in matte.numpy():
                ok, png = cv2.imencode(".png", frame)
                if not ok:
                    raise RuntimeError(f"PNG encode failed for {entity}")
                blobs.append(png.tobytes())
            rec.send_columns(
                entity,
                indexes=[rr.TimeColumn(FRAME_TIMELINE, sequence=frame_indices)],
                columns=rr.EncodedImage.columns(blob=blobs, media_type=["image/png"] * len(blobs)),
            )


def register_mask_layer(catalog_url: str, dataset: str, path: Path) -> None:
    """Register ``path`` as layer ``people_matte`` of its segment (replacing an older ``people_matte`` layer of that segment)
    on ``dataset``, got or created (a companion dataset such as ``open4dgs-mamma``; ``simplecv.catalog_layers.ensure_dataset``)."""
    register_layer(ensure_dataset(CatalogClient(catalog_url), dataset), [path], MATTE_LAYER)


def decode_mask_pngs(blobs: Sequence[bytes], threads: int = DECODE_THREADS) -> list[np.ndarray | None]:
    """``cv2.imdecode`` (grayscale) of each blob on a thread pool, in order; None where a blob does not decode."""

    def decode(blob: bytes) -> np.ndarray | None:
        return cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # type: ignore[arg-type]

    if threads <= 1:
        return [decode(blob) for blob in blobs]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(decode, blobs))


def resample_masks(
    stored: UInt8[Tensor, "t h w"],
    grid: Float32[Tensor, "1 h2 w2 2"] | None,
    target_wh: tuple[int, int],
    compute_device: torch.device,
    device: torch.device,
    dtype: torch.dtype,
) -> Float32[Tensor, "t h3 w3"] | Float16[Tensor, "t h3 w3"]:
    """Stored-grid 8-bit mattes to [0, 1] on the clip grid (``resample_images``), ``RESAMPLE_CHUNK`` frames at a time on
    ``compute_device``; the result lands on ``device`` as ``dtype``."""
    return torch.cat(
        [
            resample_images(stored[start : start + RESAMPLE_CHUNK, None].to(compute_device).float() / 255.0, grid, target_wh)[:, 0]
            .clamp(0.0, 1.0)
            .to(device=device, dtype=dtype)
            for start in range(0, len(stored), RESAMPLE_CHUNK)
        ]
    ).contiguous()


def read_clip_masks(
    clip: MultiViewClip,
    catalog_url: str,
    dataset: str,
    segment_id: str,
) -> tuple[Float32[Tensor, "frames h w"], ...]:
    """The ``people_matte`` layer for ``clip``: per camera, soft foreground in [0, 1], on the clip's exact pixel grid.

    The stored-grid PNGs go through the clip's own undistortion grid and resize (``resample_images``), so mask pixel
    ``(y, x)`` covers the same ray as ``clip.frames[c][t, y, x]``. A clip read with ``roi`` uses the same combined
    rotation + zoom + undistortion grid as its frames (``catalog_clip.roi_grid``); the grid is the one the frame read
    cached on ``cuda`` (the reader's default ``decode_device``). The grid is built and the masks are resampled on
    ``cuda`` when available (else the CPU), ``RESAMPLE_CHUNK`` frames at a time; the float32 masks land on the CPU.

    Args:
        clip: The clip the masks align with (from ``read_catalog_clip``).
        catalog_url: Catalog holding the ``people_matte`` layer.
        dataset: Its dataset.
        segment_id: The clip's segment.

    Raises:
        ValueError: If the clip lacks ``stored_sizes``/``lenses``, or the layer misses a camera or a clip frame.
    """
    if len(clip.stored_sizes) != len(clip.camera_names) or len(clip.lenses) != len(clip.camera_names):
        raise ValueError("clip has no stored_sizes/lenses: read it with open4dgs.catalog_clip.read_catalog_clip")
    compute_on: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry: DatasetEntry = CatalogClient(catalog_url).get_dataset(dataset)
    entities: list[str] = [mask_entity(camera) for camera in clip.camera_names]
    frames: list[int] = clip.frame_indices.tolist()
    table: pa.Table = (
        entry.filter_segments(segment_id)
        .filter_contents(entities)
        .reader(index=FRAME_TIMELINE)
        .filter((col(FRAME_TIMELINE) >= min(frames)) & (col(FRAME_TIMELINE) <= max(frames)))
        .select(FRAME_TIMELINE, *[f"{entity}:EncodedImage:blob" for entity in entities])
        .to_arrow_table()
    )
    row_of: dict[int, int] = {int(frame): row for row, frame in enumerate(table.column(FRAME_TIMELINE).to_numpy())}
    out: list[Tensor] = []
    for camera, entity in enumerate(entities):
        column: str = f"{entity}:EncodedImage:blob"
        if column not in table.column_names:
            raise ValueError(f"{dataset}/{segment_id}: layer {MATTE_LAYER!r} has no {entity}")
        cells: list[object | None] = table.column(column).to_pylist()
        stored_wh: tuple[int, int] = clip.stored_sizes[camera]
        blobs: list[bytes] = []
        for frame in frames:
            row: int | None = row_of.get(frame)
            cell: object | None = None if row is None else cells[row]
            if isinstance(cell, list):  # a list-of-blobs cell
                cell = cell[0] if cell else None
            if cell is None:
                raise ValueError(f"{dataset}/{segment_id}: {entity} has no mask at frame {frame} (window {frames[0]}..{frames[-1]} not covered)")
            blobs.append(bytes(cell))  # type: ignore[arg-type]
        stack: UInt8[ndarray, "t h w"] = np.empty((len(frames), stored_wh[1], stored_wh[0]), dtype=np.uint8)
        for slot, (frame, image) in enumerate(zip(frames, decode_mask_pngs(blobs), strict=True)):
            if image is None or (image.shape[1], image.shape[0]) != stored_wh:
                raise ValueError(f"{entity} frame {frame}: mask is not a {stored_wh} PNG")
            stack[slot] = image
        lens = clip.lenses[camera]
        grid: Float32[Tensor, "1 h w 2"] | None
        if clip.roi is not None:
            grid = roi_grid(lens, clip.roi.cameras[camera], compute_on)
        else:
            grid = None if lens is None else undistortion_grid(lens, stored_wh, compute_on)
        out.append(resample_masks(torch.from_numpy(stack), grid, clip.sizes[camera], compute_on, torch.device("cpu"), torch.float32))
    return tuple(out)
