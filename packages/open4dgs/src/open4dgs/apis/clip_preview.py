"""Read a catalog segment as a training clip and log it to Rerun: cameras, undistorted frames, SMPL-X meshes.

The recording shows exactly what a trainer gets from ``read_catalog_clip``: each camera as ``Transform3D`` + ``Pinhole``
(the returned undistorted K and ``world_T_cam``), its frames as JPEG ``EncodedImage`` under the pinhole, and every
person's mesh on the ``frame_index`` timeline.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import rerun as rr
import torch
from simplecv.rerun_log_utils import RerunTyroConfig

from open4dgs.catalog_clip import SPARK_CATALOG, MultiViewClip, read_catalog_clip

JPEG_QUALITY: int = 90
"""Quality of the logged frames (a 32-camera clip at 1028 px is 1.2 GB raw per 16 frames)."""


@dataclass
class Config:
    rr_config: RerunTyroConfig
    catalog_url: str = SPARK_CATALOG
    dataset: str = "dataforge-mamma-markerless-dance"
    segment_id: str = "mamma-markerless-dance__140725_Breakdancemarkers_Improv_2_03686_1_1_1"
    frame_start: int = 0
    num_frames: int = 16
    frame_step: int = 1
    cameras: list[str] | None = None
    """Camera names such as ``rig_01/cam_00``; default every camera."""
    max_side: int | None = 1028
    body_source: str | None = None
    """``gt`` or ``pred/mamma``; default prefers ``gt``."""
    image_cameras: int = field(default=32)
    """Log frames for the first this-many cameras (all cameras still get a frustum)."""


def log_clip(clip: MultiViewClip, image_cameras: int) -> None:
    """Log a clip on the ``frame_index`` timeline (cameras static, frames and meshes per frame)."""
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for index, name in enumerate(clip.camera_names):
        world_T_cam: np.ndarray = clip.world_T_cam[index].numpy().astype(np.float64)
        rr.log(f"/world/{name}", rr.Transform3D(translation=world_T_cam[:3, 3], mat3x3=world_T_cam[:3, :3]), static=True)
        rr.log(
            f"/world/{name}/pinhole",
            rr.Pinhole(
                image_from_camera=clip.K[index].numpy(),
                resolution=list(clip.sizes[index]),
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
            ),
            static=True,
        )
    frame_indices: list[int] = clip.frame_indices.tolist()
    for slot, frame in enumerate(frame_indices):
        rr.set_time("frame_index", sequence=frame)
        for index, name in enumerate(clip.camera_names[:image_cameras]):
            rr.log(f"/world/{name}/pinhole/image", rr.Image(clip.frames[index][slot].numpy(), color_model="RGB").compress(jpeg_quality=JPEG_QUALITY))
        if clip.people is None:
            continue
        for person, person_name in enumerate(clip.people.names):
            path: str = f"/world/{clip.people.source}/people/{person_name}/body/mesh"
            if bool(clip.people.present[slot, person]):
                rr.log(path, rr.Mesh3D(vertex_positions=clip.people.vertices[slot, person].numpy(), triangle_indices=clip.people.faces.numpy()))
            else:
                rr.log(path, rr.Clear(recursive=False))


def main(config: Config) -> None:
    started: float = perf_counter()
    clip: MultiViewClip = read_catalog_clip(
        config.catalog_url,
        config.dataset,
        config.segment_id,
        config.frame_start,
        config.num_frames,
        config.frame_step,
        config.cameras,
        config.max_side,
        body_source=config.body_source,
    )
    load_s: float = perf_counter() - started
    gpu_peak_gb: float = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    people: str = (
        "none" if clip.people is None else f"{len(clip.people.names)} from /world/{clip.people.source} ({int(clip.people.present.sum())} meshes)"
    )
    print(
        f"clip: {len(clip.camera_names)} cams x {len(clip.frame_indices)} frames at {sorted(set(clip.sizes))}, "
        f"{clip.nbytes / 1e9:.2f} GB frames on {clip.frames[0].device}, people {people}; "
        f"load {load_s:.1f} s, torch GPU peak {gpu_peak_gb:.2f} GB"
    )
    log_clip(clip, config.image_cameras)
