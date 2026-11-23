"""MoGe-2 prior maps for a clip and their catalog layer ``moge2_prior`` (the loss is ``open4dgs.prior``).

Prior maps: MoGe-2 (``moge-2-vitl-normal``) on the EXACT frames the trainer sees (``read_catalog_clip``: undistorted,
resized to ``max_side``), with the calibrated horizontal field of view given to MoGe's shift recovery instead of
letting it guess the focal. Layer ``moge2_prior``: per camera and frame ``/world/rig_NN/cam_00/pinhole/moge2/depth``
(``EncodedDepthImage``, 16-bit PNG in mm, 0 = invalid, ``meter=1000``, ``depth_range`` set) and ``.../moge2/normals``
(``EncodedImage`` RGB PNG ``127.5 + 127.5 n``, black = invalid) on ``frame_index``, at the CLIP's pixel grid (not the
stored video's), with recording property ``moge2_prior`` (model, window, max_side, image sizes).
"""

import dataclasses
import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from datafusion import col
from jaxtyping import Float32, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_layers import atomic_recording, ensure_dataset, register_layer
from simplecv.catalog_rig_layout import catalog_components
from torch import Tensor

from open4dgs.catalog_clip import FRAME_TIMELINE, MultiViewClip
from open4dgs.prior import ClipPriors, encode_normals

ClipReader: TypeAlias = Literal["catalog", "selfcap"]
"""Which reader made the clip: ``catalog_clip.read_catalog_clip`` or ``selfcap_clip.read_timed_clip``."""
LOADERS: dict[ClipReader, str] = {
    "catalog": "open4dgs.catalog_clip.read_catalog_clip (undistorted)",
    "selfcap": "open4dgs.selfcap_clip.read_timed_clip",
}
"""The ``loader`` the layer records for each reader."""
PRIOR_LAYER: str = "moge2_prior"
"""Catalog layer name of the MoGe-2 prior maps."""
MODEL_NAME: str = "moge-2-vitl-normal"
"""Recorded in the layer's properties."""
ENGINE_PREDICTOR: str = "MoGeV2TrtPredictor dynamic"
"""Layer property ``predictor`` of maps from monopriors' dynamic-shape TensorRT engine (one per GPU and detail level, built once)."""
TORCH_PREDICTOR: str = "MoGeV2TorchPredictor fp16 autocast"
"""Layer property ``predictor`` of maps from the torch predictor (a clip with a camera outside the engine's aspect range)."""


def compute_clip_priors(
    clip: MultiViewClip, camera_names: Sequence[str], predictor: object, batch_size: int = 8, device: torch.device | None = None
) -> ClipPriors:
    """MoGe-2 maps for ``camera_names`` at every clip frame, from the clip's own (undistorted, resized) frames."""
    dev: torch.device = device if device is not None else torch.device("cuda")
    depths: list[Tensor] = []
    normals: list[Tensor] = []
    for name in camera_names:
        cam: int = clip.camera_names.index(name)
        # The calibrated focal fixes MoGe's shift recovery (metrictrail ``recover_metric_depth`` pattern); MoGe assumes a
        # centred principal point.
        fov_x: float = math.degrees(2.0 * math.atan(clip.frames[cam].shape[2] / (2.0 * float(clip.K[cam, 0, 0]))))
        d_parts: list[Tensor] = []
        n_parts: list[Tensor] = []
        for start in range(0, clip.frames[cam].shape[0], batch_size):
            batch: Tensor = clip.frames[cam][start : start + batch_size].to(dev)
            out = predictor.predict_geometry(batch, fov_x=fov_x)  # type: ignore[attr-defined]
            valid: Tensor = (out.mask_bhw > 0.5) & (out.depth_bhw > 0)
            d_parts.append(torch.where(valid, out.depth_bhw, 0.0).half().cpu())
            n_parts.append(encode_normals(out.normals_bhw3, valid).cpu())
        depths.append(torch.cat(d_parts))
        normals.append(torch.cat(n_parts))
    return ClipPriors(camera_names=tuple(camera_names), frame_indices=clip.frame_indices.clone(), depth=tuple(depths), normals=tuple(normals))


def as_stored(priors: ClipPriors) -> ClipPriors:
    """``priors`` as ``read_clip_priors`` hands them back from the layer: depth rounded to whole millimetres (the 16-bit
    PNG), then float16; normals are stored losslessly. A run that computes its maps trains on the same values as the
    next run that reads them."""
    depth: tuple[Tensor, ...] = tuple(((d.float() * 1000.0).round().clamp(0, 65535) / 1000.0).half() for d in priors.depth)
    return dataclasses.replace(priors, depth=depth)


def layer_properties(
    clip: MultiViewClip, source: str, max_side: int | None, resolution_level: int, *, reader: ClipReader, predictor: str
) -> dict[str, str | int | float]:
    """The ``moge2_prior`` recording property: model, source, window, sizes, the reader that made ``clip`` and the camera
    grid (``roi_properties``)."""
    return {
        "model": MODEL_NAME,
        "predictor": predictor,
        "resolution_level": resolution_level,
        "fov_source": "calibrated fx (fov_x = 2 atan(w / 2 fx))",
        "source": source,
        "frame_start": int(clip.frame_indices[0]),
        "frame_end": int(clip.frame_indices[-1]),
        "num_frames": len(clip.frame_indices),
        "max_side": max_side or 0,
        "cameras": ",".join(clip.camera_names),
        "image_sizes": ",".join(f"{w}x{h}" for w, h in clip.sizes),
        "loader": LOADERS[reader],
        **roi_properties(clip),
    }


def compute_prior_layer(
    clip: MultiViewClip,
    segment_id: str,
    *,
    source: str,
    catalog_url: str,
    dataset: str,
    output_dir: Path,
    max_side: int | None,
    reader: ClipReader,
    register: bool = True,
    batch_size: int = 8,
    resolution_level: int = 9,
) -> tuple[ClipPriors, Path, dict[str, float]]:
    """MoGe-2 (vitl) on every clip camera and frame, written as the ``moge2_prior`` layer recording and registered on
    ``catalog_url``/``dataset`` (get-or-create; replaces the segment's previous layer).

    The file is ``<output_dir>/<dataset>/<segment_id>.moge2_prior-<grid>.rrd``, ``<grid>`` a hash of the window, cameras,
    sizes and ROI grid, so a layer of another grid is never overwritten on disk. Returns the maps as stored
    (``as_stored``), the path and the timings (``moge_s``, ``images``, ``write_s``, ``register_s``). The predictor is
    freed before returning.

    The maps come from monopriors' dynamic TensorRT engine (the first call on a GPU builds it, ~90 s) when it serves every
    camera (``MoGeV2TrtPredictor.serves``: aspects 1:2 to 2:1), else from the torch predictor for the whole clip (wide ROI
    crops); property ``predictor`` records which.
    """
    from monopriors.models.moge_v2.predictor import MoGeV2TorchPredictor, MoGeV2TrtPredictor

    engine = MoGeV2TrtPredictor(encoder="vitl", heads="geometry", resolution_level=resolution_level, batch_size=batch_size)
    on_engine: bool = all(engine.serves((height, width)) for width, height in clip.sizes)
    predictor: MoGeV2TrtPredictor | MoGeV2TorchPredictor = (
        engine if on_engine else MoGeV2TorchPredictor(encoder="vitl", heads="geometry", resolution_level=resolution_level)
    )
    torch.cuda.synchronize()
    t0: float = time.perf_counter()
    priors: ClipPriors = compute_clip_priors(clip, clip.camera_names, predictor, batch_size=batch_size)
    torch.cuda.synchronize()
    timings: dict[str, float] = {"moge_s": time.perf_counter() - t0, "images": float(len(clip.camera_names) * len(clip.frame_indices))}
    del predictor, engine
    torch.cuda.empty_cache()
    properties = layer_properties(
        clip, source, max_side, resolution_level, reader=reader, predictor=ENGINE_PREDICTOR if on_engine else TORCH_PREDICTOR
    )
    grid: str = hashlib.sha1(
        "|".join(str(properties[k]) for k in ("frame_start", "frame_end", "cameras", "image_sizes", "max_side", "roi", "roi_cameras")).encode()
    ).hexdigest()[:10]
    path: Path = output_dir / dataset / f"{segment_id}.{PRIOR_LAYER}-{grid}.rrd"
    t0 = time.perf_counter()
    write_prior_layer(path, segment_id, priors, properties)
    timings["write_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if register:
        register_prior_layer(catalog_url, dataset, path)
    timings["register_s"] = time.perf_counter() - t0
    return as_stored(priors), path, timings


def select_cameras(priors: ClipPriors, camera_names: Sequence[str]) -> ClipPriors:
    """The maps of ``camera_names`` (each must be in ``priors``)."""
    slots: list[int] = [priors.camera_names.index(name) for name in camera_names]
    return ClipPriors(tuple(camera_names), priors.frame_indices, tuple(priors.depth[i] for i in slots), tuple(priors.normals[i] for i in slots))


def roi_signature(clip: MultiViewClip, camera_names: Sequence[str]) -> dict[str, str]:
    """Per camera, the grid a prior map belongs to: ``none`` for a real camera, else the ROI pad, cap, size, ``K`` and rotation.

    Written into the layer's properties (``roi``, ``roi_cameras``); ``read_clip_priors`` compares it with the clip it reads into.
    """
    if clip.roi is None:
        return dict.fromkeys(camera_names, "none")
    signatures: dict[str, str] = {}
    for name in camera_names:
        c: int = clip.camera_names.index(name)
        w, h = clip.sizes[c]
        k: Float32[ndarray, "3 3"] = clip.K[c].numpy()
        r: str = ",".join(f"{v:.4f}" for v in clip.roi.cameras[c].cam_R_virtual.ravel())
        signatures[name] = (
            f"pad={clip.roi.pad:g};max_side={clip.roi.max_side};{w}x{h};K={k[0, 0]:.2f},{k[1, 1]:.2f},{k[0, 2]:.2f},{k[1, 2]:.2f};R={r}"
        )
    return signatures


def roi_properties(clip: MultiViewClip) -> dict[str, str]:
    """Layer properties that pin the prior to the clip's camera grid (see ``roi_signature``)."""
    signatures: dict[str, str] = roi_signature(clip, clip.camera_names)
    roi: str = "none" if clip.roi is None else f"people pad={clip.roi.pad:g} max_side={clip.roi.max_side}"
    return {"roi": roi, "roi_cameras": "|".join(f"{name}@{sig}" for name, sig in signatures.items())}


def check_roi(clip: MultiViewClip, camera_names: Sequence[str], stored: str | None, source: str) -> None:
    """Raise when the layer's ``roi_cameras`` property (None: a layer written before ROI cameras, real cameras) does not
    describe ``clip``'s cameras ``camera_names``.

    Raises:
        ValueError: On a camera whose grid differs (an ROI clip read against a full-frame layer, another pad, cap or window).
    """
    stored_of: dict[str, str] = dict(item.split("@", 1) for item in stored.split("|")) if stored else {}
    for name, expected in roi_signature(clip, camera_names).items():
        got: str = stored_of.get(name, "none")
        if got != expected:
            raise ValueError(f"{source}: prior layer camera {name} was computed on grid {got!r}, the clip camera is {expected!r}")


def prior_entity(camera_name: str, kind: Literal["depth", "normals"]) -> str:
    """``/world/rig_NN/cam_00/pinhole/moge2/depth`` or ``.../normals``."""
    return f"/world/{camera_name}/pinhole/moge2/{kind}"


def write_prior_layer(path: Path, segment_id: str, priors: ClipPriors, properties: dict[str, str | int | float]) -> None:
    """Write the ``moge2_prior`` layer recording (recording id = ``segment_id``)."""
    with atomic_recording(path, application_id="open4dgs-moge2-prior", recording_id=segment_id, send_properties=False) as rec:
        rec.send_property(PRIOR_LAYER, rr.AnyValues(**properties))  # type: ignore[bad-argument-type]
        frames: Int64[ndarray, " t"] = priors.frame_indices.numpy()
        for camera, depth, normals in zip(priors.camera_names, priors.depth, priors.normals, strict=True):
            depth_mm: np.ndarray = (depth.float() * 1000.0).round().clamp(0, 65535).numpy().astype(np.uint16)
            valid_mm: np.ndarray = depth_mm[depth_mm > 0]
            lo, hi = (float(np.percentile(valid_mm, 1)), float(np.percentile(valid_mm, 99))) if valid_mm.size else (0.0, 1.0)
            depth_blobs: list[bytes] = []
            normal_blobs: list[bytes] = []
            for f in range(len(frames)):
                ok_d, png_d = cv2.imencode(".png", depth_mm[f])
                ok_n, png_n = cv2.imencode(".png", np.ascontiguousarray(normals[f].numpy()[..., ::-1]))  # cv2 writes BGR
                if not (ok_d and ok_n):
                    raise RuntimeError(f"PNG encode failed for {camera} frame {int(frames[f])}")
                depth_blobs.append(png_d.tobytes())
                normal_blobs.append(png_n.tobytes())
            index = [rr.TimeColumn(FRAME_TIMELINE, sequence=frames)]
            rec.send_columns(
                prior_entity(camera, "depth"),
                indexes=index,
                columns=rr.EncodedDepthImage.columns(
                    blob=depth_blobs, media_type=["image/png"] * len(frames), meter=[1000.0] * len(frames), depth_range=[[lo, hi]] * len(frames)
                ),
            )
            rec.send_columns(
                prior_entity(camera, "normals"), indexes=index, columns=rr.EncodedImage.columns(blob=normal_blobs, media_type=["image/png"] * len(frames))
            )


def register_prior_layer(catalog_url: str, dataset: str, path: Path) -> None:
    """Register ``path`` as layer ``moge2_prior`` of its segment on ``dataset`` (``ensure_dataset``: get-or-create with one retry; replaces)."""
    register_layer(ensure_dataset(CatalogClient(catalog_url), dataset), [path], PRIOR_LAYER)


def _cell_bytes(cell: object | None) -> bytes | None:
    if isinstance(cell, list):
        cell = cell[0] if cell else None
    return None if cell is None else bytes(cell)  # type: ignore[arg-type]


def read_clip_priors(clip: MultiViewClip, catalog_url: str, dataset: str, segment_id: str, camera_names: Sequence[str] | None = None) -> ClipPriors:
    """The ``moge2_prior`` layer for ``clip`` (default: every clip camera), checked against the clip's frames and sizes.

    Raises:
        ValueError: If a camera or a clip frame is missing from the layer, or a map's size is not the clip camera's size
            (the layer was computed for another ``max_side``).
    """
    names: list[str] = list(camera_names) if camera_names is not None else list(clip.camera_names)
    client = CatalogClient(catalog_url)  # held for the whole read: a dropped client ends the queries
    entry: DatasetEntry = client.get_dataset(dataset)
    entities: list[str] = [prior_entity(name, kind) for name in names for kind in ("depth", "normals")]
    frames: list[int] = clip.frame_indices.tolist()
    columns: list[str] = [f"{prior_entity(name, 'depth')}:EncodedDepthImage:blob" for name in names] + [
        f"{prior_entity(name, 'normals')}:EncodedImage:blob" for name in names
    ]
    view = entry.filter_segments(segment_id).filter_contents(entities)
    properties: pa.Table = entry.segment_table().to_arrow_table()
    roi_column: str = f"property:{PRIOR_LAYER}:roi_cameras"
    stored: str | None = None
    if roi_column in properties.column_names:
        segment_column: str = "rerun_segment_id"
        rows: list[dict[str, object]] = [r for r in properties.select([segment_column, roi_column]).to_pylist() if r[segment_column] == segment_id]
        cell: object = rows[0][roi_column] if rows else None
        while isinstance(cell, list):
            cell = cell[0] if cell else None
        stored = None if cell is None else str(cell)
    check_roi(clip, names, stored, f"{dataset}/{segment_id}")
    available: set[str] = {f"{c.entity_path}:{c.component}" for c in catalog_components(view.schema())}
    missing: list[str] = [c for c in columns if c not in available]
    if missing:
        raise ValueError(f"{dataset}/{segment_id}: layer {PRIOR_LAYER!r} lacks {missing[:3]} (cameras not computed?)")
    table: pa.Table = (
        view.reader(index=FRAME_TIMELINE)
        .filter((col(FRAME_TIMELINE) >= min(frames)) & (col(FRAME_TIMELINE) <= max(frames)))
        .select(FRAME_TIMELINE, *columns)
        .to_arrow_table()
    )
    row_of: dict[int, int] = {int(frame): row for row, frame in enumerate(table.column(FRAME_TIMELINE).to_numpy())}
    depths: list[Tensor] = []
    normals: list[Tensor] = []
    for name in names:
        width, height = clip.sizes[clip.camera_names.index(name)]
        d_cells: list[object | None] = table.column(f"{prior_entity(name, 'depth')}:EncodedDepthImage:blob").to_pylist()
        n_cells: list[object | None] = table.column(f"{prior_entity(name, 'normals')}:EncodedImage:blob").to_pylist()
        d_stack: np.ndarray = np.empty((len(frames), height, width), dtype=np.float16)
        n_stack: np.ndarray = np.empty((len(frames), height, width, 3), dtype=np.uint8)
        for slot, frame in enumerate(frames):
            row: int | None = row_of.get(frame)
            d_blob: bytes | None = None if row is None else _cell_bytes(d_cells[row])
            n_blob: bytes | None = None if row is None else _cell_bytes(n_cells[row])
            if d_blob is None or n_blob is None:
                raise ValueError(f"{dataset}/{segment_id}: {name} has no prior at frame {frame} (window {frames[0]}..{frames[-1]} not covered)")
            depth_mm: np.ndarray | None = cv2.imdecode(np.frombuffer(d_blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            normal_bgr: np.ndarray | None = cv2.imdecode(np.frombuffer(n_blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if depth_mm is None or normal_bgr is None or depth_mm.shape != (height, width) or normal_bgr.shape[:2] != (height, width):
                got = None if depth_mm is None else depth_mm.shape[::-1]
                raise ValueError(f"{name} frame {frame}: prior is {got}, the clip camera is {(width, height)} (layer computed for another max_side?)")
            d_stack[slot] = depth_mm.astype(np.float32) / 1000.0
            n_stack[slot] = normal_bgr[..., ::-1]
        depths.append(torch.from_numpy(d_stack))
        normals.append(torch.from_numpy(n_stack))
    return ClipPriors(camera_names=tuple(names), frame_indices=clip.frame_indices.clone(), depth=tuple(depths), normals=tuple(normals))
