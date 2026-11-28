"""SelfCap (4 exo phones, no body mesh) preparation: ``people_matte`` layer, ``moge2_prior`` layer and an init cloud .rrd.

Layers go on the local companion dataset ``open4dgs-selfcap`` (same segment id as the read-only Spark segment), on the
``frame_index`` timeline whose values are the reference phone's packet numbers (``selfcap_clip.read_timed_clip``).

- ``masks`` stage: the ``people_matte`` layer, BiRefNet on the stored 1920x1080 frames, cropped by the projected joints box (``joint_box_points``) padded
  more than the SMPL-X box (joints are not a surface: hair, hands, clothes and the head under the headset).
- ``prior``: MoGe-2 metric depth + normals on the training grid (``max_side``), every phone.
- ``init``: ``moge_lift_init`` (or ``capsule_init``) from the training phones only; written as an init-cloud .rrd that
  ``apis.train`` reads with ``--init-rrd``.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from simplecv.catalog_layers import atomic_recording

from open4dgs.birefnet_trt import load_matte_model
from open4dgs.catalog_clip import LOCAL_CATALOG, SPARK_CATALOG
from open4dgs.init_cloud import log_init_cloud
from open4dgs.masks import (
    MATTE_LAYER,
    MODEL_NAME,
    clip_mattes,
    read_clip_masks,
    register_mask_layer,
    write_mask_layer,
)
from open4dgs.prior import ClipPriors
from open4dgs.prior_layer import compute_prior_layer, read_clip_priors
from open4dgs.properties import MatteLayerProperties
from open4dgs.selfcap_clip import TimedClip, read_timed_clip
from open4dgs.selfcap_init import capsule_init, joint_box_points, moge_lift_init

COMPANION_DATASET: str = "open4dgs-selfcap"
WORLD_UP: np.ndarray = np.array([0.0, 1.0, 0.0])
"""SelfCap world (Quest head_pose frame) is Y up."""


@dataclass
class Config:
    stage: Literal["masks", "prior", "init"] = "masks"
    catalog_url: str = SPARK_CATALOG
    dataset: str = "selfcap-easy26-v4"
    segment_id: str = "selfcap__2025-12-02__91e9bbba__ep001"
    time_start_s: float = 39.0
    """First frame time on ``video_time`` (the reference phone's first frame at or after it)."""
    num_frames: int = 60
    max_side: int | None = 960
    """Training grid (``prior`` and ``init``); ``masks`` always run on the stored frames."""
    register_url: str = LOCAL_CATALOG
    register_dataset: str = COMPANION_DATASET
    register: bool = True
    output_root: Path = Path("/mnt/nas/ssd-8tb/open4dgs")
    model_dir: Path = Path(os.environ.get("FDANYONE_MODEL_DIR", "/mnt/nas/ssd-8tb/fdanyone-rerun/models")) / "birefnet"
    pad_fraction: float = 0.25
    pad_px: float = 48.0
    holdout_cameras: tuple[str, ...] = ("rig_01/cam_00",)
    """``init``: these phones never enter the init."""
    init_method: Literal["moge", "capsule"] = "moge"
    keyframe_step: int = 3
    points_per_view: int = 6_000
    fit_scope: Literal["camera", "keyframe"] = "camera"
    """``init`` (moge): one depth fit per camera over the window, or one per camera and keyframe."""
    init_name: str = "moge"
    """``init``: file ``<output_root>/init/open4dgs-selfcap/<segment>__<init_name>.rrd``."""
    roi: bool = False
    """``prior``/``init``: read the clip with person-centred virtual cameras (the prior must match the training grid; the layer file is
    named by a hash of that grid, ``prior_layer.compute_prior_layer``, and replaces the registered ``moge2_prior`` layer)."""
    png_dir: Path | None = Path("/mnt/nas/ssd-8tb/scratch/open4dgs/runs/s1")


def _masks(config: Config) -> None:
    timed: TimedClip = read_timed_clip(config.catalog_url, config.dataset, config.segment_id, config.time_start_s, config.num_frames)
    clip = timed.clip
    device = torch.device("cuda")
    model = load_matte_model(config.model_dir, device)
    started: float = time.perf_counter()
    joints = timed.joints
    assert joints is not None
    mattes = clip_mattes(
        clip, lambda cam: joint_box_points(joints.at(timed.times_s[cam]), WORLD_UP), model, config.pad_fraction, config.pad_px, device, batch_size=2
    )
    print(f"{len(mattes) * config.num_frames} masks in {time.perf_counter() - started:.1f} s")
    path: Path = config.output_root / "masks" / config.register_dataset / f"{config.segment_id}.{MATTE_LAYER}.rrd"
    write_mask_layer(
        path,
        config.segment_id,
        clip.camera_names,
        clip.frame_indices.numpy(),
        mattes,
        MatteLayerProperties(
            model=MODEL_NAME,
            frame_start=int(clip.frame_indices[0]),
            num_frames=config.num_frames,
            time_start_s=float(timed.times_s[0, 0]),
            source_catalog=config.catalog_url,
            source_dataset=config.dataset,
            source_segment=config.segment_id,
            crop=f"projected joints + head proxies box pad {config.pad_fraction} + {config.pad_px} px",
        ),
    )
    path.parent.chmod(0o775)
    if config.png_dir is not None:
        slots = [0, config.num_frames // 2, config.num_frames - 1]
        rows = []
        for cam in range(len(clip.camera_names)):
            tiles = []
            for f in slots:
                rgb = clip.frames[cam][f].numpy()
                m = mattes[cam][f].numpy()[..., None] / 255.0
                tiles.append(cv2.resize(np.concatenate([rgb, (rgb * m).astype(np.uint8)], axis=1), (960, 270)))
            rows.append(np.concatenate(tiles, axis=1))
        cv2.imwrite(str(config.png_dir / f"s1-masks-contact-{config.segment_id[-17:]}.jpg"), np.concatenate(rows)[..., ::-1])
    if config.register:
        register_mask_layer(config.register_url, config.register_dataset, path)
        print(f"registered masks on {config.register_dataset}/{config.segment_id}")


def _prior(config: Config) -> None:
    """MoGe-2 maps of every phone on the training grid, through the trainer's own path (``prior_layer.compute_prior_layer``):
    the layer records the clip's camera grid (``roi`` / ``roi_cameras``), so ``read_clip_priors`` accepts it for this grid only."""
    timed: TimedClip = read_timed_clip(
        config.catalog_url, config.dataset, config.segment_id, config.time_start_s, config.num_frames, max_side=config.max_side, roi=config.roi
    )
    clip = timed.clip
    _, path, timings = compute_prior_layer(
        clip,
        config.segment_id,
        source=f"{config.catalog_url}/{config.dataset}",
        catalog_url=config.register_url,
        dataset=config.register_dataset,
        output_dir=config.output_root / "prior",
        max_side=config.max_side,
        reader="selfcap",
        register=config.register,
        batch_size=2,
    )
    print(f"MoGe-2 {timings['images']:.0f} images in {timings['moge_s']:.1f} s -> {path}")
    path.parent.chmod(0o775)
    if config.register:
        print(f"registered moge2_prior on {config.register_dataset}/{config.segment_id}")


def _init(config: Config) -> None:
    timed: TimedClip = read_timed_clip(
        config.catalog_url, config.dataset, config.segment_id, config.time_start_s, config.num_frames, max_side=config.max_side, roi=config.roi
    )
    clip = timed.clip
    assert timed.joints is not None
    train_names: tuple[str, ...] = tuple(name for name in clip.camera_names if name not in config.holdout_cameras)
    started: float = time.perf_counter()
    if config.init_method == "capsule":
        cloud = capsule_init(
            clip, train_names, timed.joints_at_frames(), dataset=config.dataset, segment_id=config.segment_id, keyframe_step=config.keyframe_step
        )
    else:
        masks = read_clip_masks(clip, config.register_url, config.register_dataset, config.segment_id)
        priors: ClipPriors = read_clip_priors(clip, config.register_url, config.register_dataset, config.segment_id, train_names)
        depths = {name: priors.depth[i] for i, name in enumerate(priors.camera_names)}
        joints_per_camera = np.stack([timed.joints.at(timed.times_s[c]) for c in range(len(clip.camera_names))])
        cloud, fits = moge_lift_init(
            clip,
            train_names,
            masks,
            depths,
            joints_per_camera,
            dataset=config.dataset,
            segment_id=config.segment_id,
            keyframe_step=config.keyframe_step,
            points_per_view=config.points_per_view,
            fit_scope=config.fit_scope,
        )
        for name, rows in fits.items():
            a = np.array([r[0] for r in rows[:-1]])
            b = np.array([r[1] for r in rows[:-1]])
            n = np.array([r[2] for r in rows[:-1]])
            print(
                f"{name}: per-keyframe a median {np.median(a):.3f} [{a.min():.3f}, {a.max():.3f}], b median {np.median(b):.3f} m, joints/keyframe median {np.median(n):.0f}; "
                f"pooled a {rows[-1][0]:.3f} b {rows[-1][1]:.3f} m"
            )
    print(f"init {cloud.method}: {cloud.num_rows:,} rows over {len(cloud.keyframes)} keyframes in {time.perf_counter() - started:.1f} s")
    path: Path = config.output_root / "init" / config.register_dataset / f"{config.segment_id}__{config.init_name}.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_recording(path, application_id="open4dgs-init", recording_id=config.segment_id, send_properties=False) as rec:
        log_init_cloud(cloud, rec)
    path.chmod(0o644)
    path.parent.chmod(0o775)
    print(f"wrote {path}")
    if config.png_dir is not None:
        rows = []
        for f in (0, config.num_frames // 2 // config.keyframe_step * config.keyframe_step):
            sel = cloud.frames == f
            tiles = []
            for cam in range(len(clip.camera_names)):
                img = np.ascontiguousarray(clip.frames[cam][f].numpy() // 2)
                cam_T_world = np.linalg.inv(clip.world_T_cam[cam].numpy().astype(np.float64))
                p = cloud.positions[sel] @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
                uv = (p @ clip.K[cam].numpy().T)[:, :2] / np.maximum(p[:, 2:3], 1e-6)
                ok = (p[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0] - 1)
                order = np.argsort(-p[ok, 2])
                col = (cloud.colors[sel][ok][order] * 255).astype(np.uint8)
                img[uv[ok][order, 1].astype(int), uv[ok][order, 0].astype(int)] = col
                label = clip.camera_names[cam] + (" HELD OUT" if clip.camera_names[cam] in config.holdout_cameras else "")
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                tiles.append(img)
            rows.append(np.concatenate(tiles, axis=1))
        cv2.imwrite(str(config.png_dir / f"s1-init-{config.init_name}.jpg"), np.concatenate(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(config: Config) -> None:
    {"masks": _masks, "prior": _prior, "init": _init}[config.stage](config)
