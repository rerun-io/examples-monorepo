"""``open4dgs-init-roma``: the upstream RoMa v2 + MEMFOF init of a catalog segment window as an ``init_roma`` layer.

Held-out cameras (``select_holdout(all cameras, num_holdout, seed)``) never reach the init. The cloud is written with
``log_init_cloud`` to ``<output_dir>/<segment>__roma__f<start>-<n>.rrd`` (recording id = segment id) and registered as
layer ``init_roma`` on the local companion dataset (``open4dgs-mamma`` for MAMMA segments, ``fdanyone-views`` for
4DAnyone segments). The trainer reads it back with ``open4dgs.init_cloud.read_init_cloud(<rrd>)``.
"""

import dataclasses
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32
from numpy import ndarray
from rerun.catalog import CatalogClient
from scipy.spatial import KDTree
from simplecv.catalog_layers import atomic_recording, ensure_dataset, register_layer
from torch import Tensor

from open4dgs.catalog_clip import LOCAL_CATALOG, SPARK_CATALOG, MultiViewClip, read_catalog_clip
from open4dgs.holdout import select_holdout
from open4dgs.init_cloud import InitCloud, log_init_cloud
from open4dgs.masks import read_clip_masks
from open4dgs.roma_init import RomaInitConfig, run_roma_init

LAYER: str = "init_roma"
"""Catalog layer name of this init."""


@dataclass
class Config:
    catalog_url: str = SPARK_CATALOG
    """Catalog that holds the source segment (read only)."""
    dataset: str = "dataforge-mamma-markerless-dance"
    segment_id: str = "mamma-markerless-dance__140725_Breakdancemarkers_Improv_2_03686_1_1_1"
    frame_start: int = 99
    """First frame of the training window (source ``frame_index``)."""
    num_frames: int = 60
    """Frames in the training window."""
    num_holdout: int = 3
    holdout_seed: int = 0
    init_cameras: int | None = None
    """Use this many training cameras, spread by farthest-point sampling of the camera centres; None uses all."""
    max_side: int | None = 1028
    """Longer side of the frames the init sees (``read_catalog_clip``)."""
    masks: Literal["catalog", "mesh", "none"] = "catalog"
    """Foreground: the companion dataset's ``people_matte`` layer, or keep only points near the SMPL-X mesh, or nothing."""
    mesh_radius_m: float = 0.3
    """``masks="mesh"``: keep points within this distance of the nearest mesh vertex."""
    local_catalog_url: str = LOCAL_CATALOG
    """Catalog that takes the layer (and holds the masks layer)."""
    companion_dataset: str | None = None
    """Dataset that takes the layer; None picks ``open4dgs-mamma`` for MAMMA segments, else ``dataset``."""
    output_dir: Path = Path("/mnt/nas/ssd-8tb/open4dgs/init")
    register: bool = True
    keyframe_step: int = 3
    camera_neighbors: int = 4
    wide_baseline_per_camera: int = 1
    roma_v2_setting: str = "fast"
    memfof_max_side: int = 1920
    subkeyframe_min_speed: float | None = None
    """Upstream's speed-aware sub-keyframes: None keeps its default (on), 0 turns them off."""
    extra_args: list[str] = field(default_factory=list)
    """More upstream flags, verbatim."""


def companion_dataset(config: Config) -> str:
    """The local dataset that takes layers for ``config.dataset``."""
    if config.companion_dataset is not None:
        return config.companion_dataset
    return "open4dgs-mamma" if config.dataset.startswith("dataforge-mamma") else config.dataset


def spread_cameras(names: tuple[str, ...], centres: Float32[ndarray, "c 3"], count: int) -> tuple[str, ...]:
    """``count`` cameras by farthest-point sampling of their centres (start: the camera farthest from the mean), in input order."""
    if count >= len(names):
        return names
    chosen: list[int] = [int(np.argmax(np.linalg.norm(centres - centres.mean(0), axis=1)))]
    distance: ndarray = np.linalg.norm(centres - centres[chosen[0]], axis=1)
    while len(chosen) < count:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(centres - centres[chosen[-1]], axis=1))
    return tuple(names[index] for index in sorted(chosen))


def keep_near_mesh(cloud: InitCloud, clip: MultiViewClip, radius_m: float) -> InitCloud:
    """Rows within ``radius_m`` of the SMPL-X vertices at the nearest clip frame that has a mesh."""
    if clip.people is None:
        raise ValueError("masks='mesh' needs a segment with SMPL-X meshes")
    present: ndarray = clip.people.present.any(dim=1).numpy()
    mesh_frames: ndarray = clip.frame_indices.numpy()[present]
    keep: ndarray = np.zeros(cloud.num_rows, dtype=bool)
    for keyframe in cloud.keyframes.tolist():
        rows: ndarray = np.flatnonzero(cloud.frames == keyframe)
        slot: int = int(np.flatnonzero(present)[np.argmin(np.abs(mesh_frames - (cloud.source_frame_start + keyframe)))])
        vertices: ndarray = clip.people.vertices[slot][clip.people.present[slot]].reshape(-1, 3).numpy()
        distance, _ = KDTree(vertices).query(cloud.positions[rows], k=1)
        keep[rows] = distance <= radius_m

    def kept(value: ndarray | None) -> ndarray | None:
        return value[keep] if value is not None else None

    return InitCloud(
        positions=cloud.positions[keep],
        velocities=cloud.velocities[keep],
        colors=cloud.colors[keep],
        times=cloud.times[keep],
        frames=cloud.frames[keep],
        frame_start=cloud.frame_start,
        frame_end=cloud.frame_end,
        keyframe_step=cloud.keyframe_step,
        fps=cloud.fps,
        source_frame_start=cloud.source_frame_start,
        dataset=cloud.dataset,
        segment_id=cloud.segment_id,
        method=f"{cloud.method}+mesh{radius_m:g}m",
        durations=kept(cloud.durations),
        has_velocity=kept(cloud.has_velocity),
        is_static=kept(cloud.is_static),
        dynamic_protect=kept(cloud.dynamic_protect),
        dynamic_allowed=kept(cloud.dynamic_allowed),
        stage_keep=kept(cloud.stage_keep),
    )


class _VramSampler:
    """Peak GPU memory used on the device (nvidia-smi, every 0.5 s): the whole-device figure, other processes included."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            output: str = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, check=False
            ).stdout
            if output.strip():
                self.samples.append(int(output.split()[0]))
            self._stop.wait(0.5)

    def __enter__(self) -> "_VramSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def main(config: Config) -> Path:
    started: float = time.perf_counter()
    step: int = config.keyframe_step
    frame_end: int = config.frame_start + config.num_frames - 1
    # Camera names first (a one-frame read), so held-out cameras are never decoded for the init.
    probe: MultiViewClip = read_catalog_clip(config.catalog_url, config.dataset, config.segment_id, config.frame_start, 1, max_side=64)
    train, holdout = select_holdout(probe.camera_names, config.num_holdout, config.holdout_seed)
    train_index: list[int] = [probe.camera_names.index(name) for name in train]
    centres: Float32[ndarray, "c 3"] = probe.world_T_cam[train_index, :3, 3].numpy()
    init_names: tuple[str, ...] = spread_cameras(train, centres, config.init_cameras) if config.init_cameras is not None else train
    print(f"[init-roma] held out {holdout}; init on {len(init_names)} of {len(train)} training cameras: {init_names}", flush=True)

    # The clip reaches one keyframe step past both window ends: MEMFOF tracks each keyframe over (t - step, t, t + step).
    clip: MultiViewClip = read_catalog_clip(
        config.catalog_url,
        config.dataset,
        config.segment_id,
        config.frame_start - step,
        config.num_frames + 2 * step,
        cameras=init_names,
        max_side=config.max_side,
    )
    masks: tuple[Tensor, ...] | None = None
    mask_frames: list[int] | None = None
    if config.masks == "catalog":
        # The masks layer covers the training window; the tracking margin frames go without masks.
        inside: Tensor = (clip.frame_indices >= config.frame_start) & (clip.frame_indices <= frame_end)
        window: MultiViewClip = dataclasses.replace(
            clip, frames=tuple(f[inside] for f in clip.frames), frame_indices=clip.frame_indices[inside], people=None
        )
        masks = read_clip_masks(window, config.local_catalog_url, companion_dataset(config), config.segment_id)
        mask_frames = window.frame_indices.tolist()
    read_s: float = time.perf_counter() - started

    roma_config: RomaInitConfig = RomaInitConfig(
        frame_start=config.frame_start,
        frame_end=frame_end,
        keyframe_step=step,
        camera_neighbors=config.camera_neighbors,
        wide_baseline_per_camera=config.wide_baseline_per_camera,
        roma_v2_setting=config.roma_v2_setting,
        memfof_max_side=config.memfof_max_side,
        subkeyframe_min_speed=config.subkeyframe_min_speed,
        extra_args=tuple(config.extra_args),
    )
    torch.cuda.reset_peak_memory_stats()
    init_started: float = time.perf_counter()
    with _VramSampler() as vram:
        cloud: InitCloud = run_roma_init(clip, masks, roma_config, dataset=config.dataset, segment_id=config.segment_id, mask_frames=mask_frames)
    init_s: float = time.perf_counter() - init_started
    if config.masks == "mesh":
        cloud = keep_near_mesh(cloud, clip, config.mesh_radius_m)
    counts: dict[int, int] = {int(k): int((cloud.frames == k).sum()) for k in cloud.keyframes.tolist()}
    print(
        f"[init-roma] {cloud.num_rows} rows over {len(counts)} keyframes (logical frame: rows) {counts}; read {read_s:.1f} s, init {init_s:.1f} s, "
        f"torch peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, device peak {max(vram.samples, default=0)} MiB "
        f"(before {vram.samples[0] if vram.samples else 0} MiB)",
        flush=True,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(config.output_dir, 0o775)
    rrd_path: Path = config.output_dir / f"{config.segment_id}__roma__f{config.frame_start}-{config.num_frames}.rrd"
    with atomic_recording(rrd_path, application_id=companion_dataset(config), recording_id=config.segment_id, send_properties=False) as recording:
        log_init_cloud(cloud, recording)
        recording.send_property(
            "init_roma",
            rr.AnyValues(
                holdout=list(holdout),
                init_cameras=list(init_names),
                num_holdout=config.num_holdout,
                holdout_seed=config.holdout_seed,
                max_side=config.max_side or 0,
                masks=config.masks,
                camera_neighbors=config.camera_neighbors,
                wide_baseline_per_camera=config.wide_baseline_per_camera,
                roma_v2_setting=config.roma_v2_setting,
                memfof_max_side=config.memfof_max_side,
                subkeyframe_min_speed=-1.0 if config.subkeyframe_min_speed is None else config.subkeyframe_min_speed,
                extra_args=" ".join(config.extra_args),
                init_seconds=init_s,
            ),
        )
    print(f"[init-roma] wrote {rrd_path}", flush=True)

    if config.register:
        register_layer(ensure_dataset(CatalogClient(config.local_catalog_url), companion_dataset(config)), [rrd_path], LAYER)
        print(f"[init-roma] registered layer {LAYER} of {config.segment_id} on {config.local_catalog_url} / {companion_dataset(config)}", flush=True)
    return rrd_path
