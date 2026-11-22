"""Compute foreground mattes for a catalog segment's frame window once and register them as the ``people_matte`` layer.

Frames are the STORED frames (no undistortion), so the layer overlays the stored video; ``open4dgs.masks.read_clip_masks``
maps it onto a clip's undistorted grid. Layers of a segment on a read-only catalog (Spark MAMMA) go on a companion
dataset on the local catalog with the same segment id.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient
from simplecv.catalog_rig_layout import catalog_components, parse_rig_layout
from torch import Tensor

from open4dgs.catalog_clip import MultiViewClip, read_catalog_clip
from open4dgs.masks import (
    MASKS_LAYER,
    MODEL_NAME,
    fill_missing_boxes,
    load_birefnet,
    person_boxes,
    predict_mattes,
    register_mask_layer,
    write_mask_layer,
)

LOCAL_CATALOG_URL: str = "rerun+http://pablo-dl-server:51235"


@dataclass
class Config:
    catalog_url: str = LOCAL_CATALOG_URL
    """Catalog that holds the segment's video."""
    dataset: str = "fdanyone-views"
    segment_id: str = "pexels-2785536__v06-seed42"
    frame_start: int = 0
    num_frames: int = 121
    cameras: list[str] | None = None
    """Camera names such as ``rig_01/cam_00``; default every camera."""
    register_url: str = LOCAL_CATALOG_URL
    """Catalog the layer is registered on."""
    register_dataset: str | None = None
    """Dataset the layer goes on; default the source dataset when it is on ``register_url``, else ``open4dgs-mamma``."""
    register: bool = True
    output_dir: Path = Path("/mnt/nas/ssd-8tb/open4dgs/masks")
    """The layer recording goes to ``<output_dir>/<register_dataset>/<segment_id>.people_matte.rrd``."""
    model_dir: Path = Path(os.environ.get("FDANYONE_MODEL_DIR", "/mnt/nas/ssd-8tb/fdanyone-rerun/models")) / "birefnet"
    camera_group: int = 4
    """Cameras decoded at a time (stored MAMMA frames are 9 MB each)."""
    batch_size: int = 8
    pad_fraction: float = 0.15
    """Crop padding around the projected SMPL-X box, as a fraction of its size (hair, loose clothes) ..."""
    pad_px: float = 24.0
    """... plus this many pixels (lens shift of up to 21 px on MAMMA: the box uses the pinhole on distorted frames)."""


def main(config: Config) -> None:
    target: str = config.register_dataset or (config.dataset if config.catalog_url == config.register_url else "open4dgs-mamma")
    entry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    cameras: list[str] = config.cameras or [
        camera.name for camera in parse_rig_layout(catalog_components(entry.filter_segments(config.segment_id).schema())).cameras
    ]
    device = torch.device("cuda")
    model: torch.nn.Module = load_birefnet(config.model_dir, device)
    mattes: list[UInt8[Tensor, "t h w"]] = []
    frame_indices: ndarray | None = None
    model_seconds: float = 0.0
    started: float = perf_counter()
    for start in range(0, len(cameras), config.camera_group):
        group: list[str] = cameras[start : start + config.camera_group]
        clip: MultiViewClip = read_catalog_clip(
            config.catalog_url, config.dataset, config.segment_id, config.frame_start, config.num_frames, cameras=group, undistort=False
        )
        if clip.people is None:
            raise ValueError(f"{config.segment_id} has no body meshes: the person crop needs them")
        vertices: Float32[ndarray, "t v 3"] = clip.people.vertices.reshape(len(clip.frame_indices), -1, 3).numpy()
        frame_indices = clip.frame_indices.numpy()
        for camera in range(len(group)):
            boxes: Float32[ndarray, "t 4"] = fill_missing_boxes(
                person_boxes(
                    vertices, clip.world_T_cam[camera].numpy(), clip.K[camera].numpy(), clip.sizes[camera], config.pad_fraction, config.pad_px
                )
            )
            tic: float = perf_counter()
            matte: UInt8[Tensor, "t h w"] = predict_mattes(model, clip.frames[camera], boxes, device, config.batch_size)
            model_seconds += perf_counter() - tic
            mattes.append(matte)
            print(f"{group[camera]}: foreground fraction {matte.float().mean().item() / 255:.4f}")
        del clip
    assert frame_indices is not None
    count: int = len(cameras) * len(frame_indices)
    print(f"{count} masks: {count / model_seconds:.1f} masks/s in the model, {count / (perf_counter() - started):.1f} masks/s with decode")
    path: Path = config.output_dir / target / f"{config.segment_id}.{MASKS_LAYER}.rrd"
    write_mask_layer(
        path,
        config.segment_id,
        cameras,
        frame_indices.astype(np.int64),
        mattes,
        {
            "model": MODEL_NAME,
            "frame_start": int(frame_indices[0]),
            "num_frames": len(frame_indices),
            "source_catalog": config.catalog_url,
            "source_dataset": config.dataset,
            "source_segment": config.segment_id,
            "crop": f"smplx box pad {config.pad_fraction} + {config.pad_px} px",
        },
    )
    path.parent.chmod(0o775)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if config.register:
        register_mask_layer(config.register_url, target, path)
        print(f"registered layer {MASKS_LAYER} of {target}/{config.segment_id} on {config.register_url}")
