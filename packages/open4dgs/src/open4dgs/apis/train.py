"""Train a person from a catalog clip and report held-out numbers in one command.

catalog clip -> seeded held-out split -> masks (catalog ``masks`` layer, or the SMPL-X stand-in) -> body-surface
init (training cameras only) -> TRiGS in memory -> held-out + training-camera metrics, orbit -> run recording
``<runs_root>/<run_name>/run.rrd`` (+ ``trigs_ckpt.pt``) -> segment of the local dataset ``open4dgs-runs``.
"""

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import torch
from PIL import Image
from simplecv.rerun_log_utils import RerunTyroConfig

from open4dgs.body_init import body_surface_init, mesh_silhouette_masks
from open4dgs.catalog_clip import MultiViewClip, read_catalog_clip
from open4dgs.evaluate import CameraRenders, Metrics, evaluate_cameras, mean_metrics, render_orbit
from open4dgs.holdout import select_holdout
from open4dgs.init_cloud import InitCloud, log_init_cloud
from open4dgs.run_log import RUNS_DATASET, log_camera_eval, log_cameras, log_people, log_scalars, log_video, register_run
from open4dgs.trigs_adapter import TrainConfig, TrainedClip, train_clip, untrained_clip

LOCAL_CATALOG: str = "rerun+http://pablo-dl-server:51235"


@dataclass
class Config:
    """One training run."""

    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(headless=True))
    """Rerun sinks; with no ``save`` the run writes ``<runs_root>/<run_name>/run.rrd`` (recording id = run name)."""
    catalog_url: str = LOCAL_CATALOG
    """Catalog that holds the segment's frames, cameras and meshes."""
    dataset: str = "fdanyone-views"
    segment_id: str = "pexels-2785536__v06-seed42"
    frame_start: int = 0
    """First source frame_index (MAMMA: a multiple of 3, the mesh stride)."""
    num_frames: int = 30
    cameras: tuple[str, ...] | None = None
    """Cameras to read (``rig_00/cam_00`` ...); None reads every camera."""
    num_holdout: int = 1
    holdout_seed: int = 0
    max_side: int | None = 640
    """Longer side of the training frames; None keeps the stored size."""
    mask_source: Literal["layer", "mesh"] = "layer"
    """``layer``: the catalog ``masks`` layer (open4dgs.masks); ``mesh``: dilated SMPL-X silhouette (stand-in, development only)."""
    mask_catalog_url: str | None = None
    """Catalog of the masks layer (default ``catalog_url``; MAMMA masks live on the local ``open4dgs-mamma``)."""
    mask_dataset: str | None = None
    """Dataset of the masks layer (default ``dataset``)."""
    samples_per_person: int = 30_000
    """Body-surface samples per person per keyframe."""
    keyframe_step: int = 3
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = "fdanyone-v06-smoke"
    runs_root: Path = Path("/mnt/nas/ssd-8tb/open4dgs/runs")
    png_dir: Path | None = None
    """Also save a held-out gt|render pair and an orbit frame as PNGs here."""
    eval_train_cameras: bool = True
    """Score the training cameras too (train/held-out gap)."""
    register: bool = True
    runs_catalog_url: str = LOCAL_CATALOG


def _read_masks(config: Config, clip: MultiViewClip) -> tuple[torch.Tensor, ...]:
    if config.mask_source == "mesh":
        return mesh_silhouette_masks(clip)
    from open4dgs.masks import read_clip_masks

    return read_clip_masks(clip, config.mask_catalog_url or config.catalog_url, config.mask_dataset or config.dataset, config.segment_id)


def _save_pngs(png_dir: Path, run_name: str, clip: MultiViewClip, holdout: list[CameraRenders], orbit: torch.Tensor) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    mid: int = len(clip.frame_indices) // 2
    for renders in holdout:
        cam: int = clip.camera_names.index(renders.camera)
        pair = np.concatenate([clip.frames[cam][mid].numpy(), renders.render[mid].numpy()], axis=1)
        Image.fromarray(pair).save(png_dir / f"{run_name}-holdout-{renders.camera.replace('/', '_')}-f{int(clip.frame_indices[mid])}.png")
    for f in (0, len(orbit) // 4, len(orbit) // 2):
        Image.fromarray(orbit[f].numpy()).save(png_dir / f"{run_name}-orbit-{f:03d}.png")


def main(config: Config) -> dict[str, object]:
    """Run the whole pipeline; returns the summary (also printed as JSON and stored as the ``summary`` property)."""
    device: torch.device = torch.device("cuda")
    run_dir: Path = config.runs_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.rr_config.save is None:
        rr.init("open4dgs-runs", recording_id=config.run_name, default_enabled=True, strict=True)
        rr.save(run_dir / "run.rrd")
    recording: rr.RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]
    timings: dict[str, float] = {}
    t0: float = time.perf_counter()
    clip: MultiViewClip = read_catalog_clip(
        config.catalog_url, config.dataset, config.segment_id, config.frame_start, config.num_frames, cameras=config.cameras, max_side=config.max_side
    )
    timings["read_clip_s"] = time.perf_counter() - t0
    train_names, holdout_names = select_holdout(clip.camera_names, config.num_holdout, config.holdout_seed)
    t0 = time.perf_counter()
    masks = _read_masks(config, clip)
    timings["masks_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    init: InitCloud = body_surface_init(
        clip,
        train_names,
        dataset=config.dataset,
        segment_id=config.segment_id,
        samples_per_person=config.samples_per_person,
        keyframe_step=config.keyframe_step,
    )
    timings["init_s"] = time.perf_counter() - t0
    print(
        f"[open4dgs] clip {len(clip.camera_names)} cams x {len(clip.frame_indices)} frames, sizes {sorted(set(clip.sizes))}; held out {holdout_names}; init rows {init.num_rows:,}"
    )

    metrics: Metrics = Metrics(device)
    baseline: TrainedClip = untrained_clip(clip, init, config.train, holdout_names, device)
    init_holdout = mean_metrics(
        [row for cam in evaluate_cameras(baseline, clip, masks, holdout_names, metrics, keep_renders=False) for row in cam.metrics]
    )
    del baseline
    torch.cuda.empty_cache()

    trained: TrainedClip = train_clip(clip, masks, init, config.train, holdout_names, device=device)
    trained.save(run_dir / "trigs_ckpt.pt")
    t0 = time.perf_counter()
    holdout: list[CameraRenders] = evaluate_cameras(trained, clip, masks, holdout_names, metrics)
    train_eval: list[CameraRenders] = (
        evaluate_cameras(trained, clip, masks, train_names, metrics, keep_renders=False) if config.eval_train_cameras else []
    )
    orbit, _ = render_orbit(trained, clip, train_names, device)
    timings["eval_s"] = time.perf_counter() - t0

    holdout_mean = mean_metrics([row for cam in holdout for row in cam.metrics])
    train_mean = mean_metrics([row for cam in train_eval for row in cam.metrics]) if train_eval else {}
    losses = trained.scalars.get("loss/rgb", [])
    summary: dict[str, object] = {
        "holdout": holdout_mean,
        "holdout_init": init_holdout,
        "train": train_mean,
        "it_per_s": config.train.steps / trained.train_seconds,
        "train_s": trained.train_seconds,
        "peak_gpu_gb": trained.peak_gpu_bytes / 1e9,
        "init_rows": init.num_rows,
        "final_gaussians": int(trained.params["means"].shape[0]),
        "loss_rgb_first": losses[0][1] if losses else math.nan,
        "loss_rgb_last": losses[-1][1] if losses else math.nan,
        **timings,
    }

    log_cameras(recording, clip, holdout_names)
    log_people(recording, clip)
    log_init_cloud(init, recording)
    for renders in holdout:
        log_camera_eval(recording, clip, masks, renders, held_out=True)
    for renders in train_eval:
        for row_name in ("crop_psnr", "fg_psnr", "full_psnr"):
            recording.send_columns(
                f"/eval/train/{renders.camera.replace('/', '_')}/metrics/{row_name}",
                indexes=[rr.TimeColumn("frame_index", sequence=[row.frame_index for row in renders.metrics])],
                columns=rr.Scalars.columns(scalars=np.asarray([getattr(row, row_name) for row in renders.metrics], dtype=np.float64)),
            )
    log_video(recording, "/eval/orbit/render", orbit, clip.frame_indices.tolist(), clip.fps)
    log_scalars(recording, trained.scalars)
    run_properties = {
        "dataset": config.dataset,
        "segment_id": config.segment_id,
        "catalog_url": config.catalog_url,
        "frame_start": int(clip.frame_indices[0]),
        "frame_end": int(clip.frame_indices[-1]),
        "num_frames": len(clip.frame_indices),
        "cameras": list(clip.camera_names),
        "train_cameras": list(train_names),
        "holdout_cameras": list(holdout_names),
        "holdout_seed": config.holdout_seed,
        "max_side": config.max_side or 0,
        "init_method": init.method,
        "samples_per_person": config.samples_per_person,
        "keyframe_step": config.keyframe_step,
        "mask_source": config.mask_source if config.mask_source == "mesh" else f"layer:{config.mask_dataset or config.dataset}/masks",
        **{f"train_{k}": v for k, v in dataclasses.asdict(config.train).items()},
    }
    recording.send_property("run", rr.AnyValues(**run_properties))  # pyrefly: ignore[bad-argument-type]
    flat: dict[str, float] = {f"{group}_{k}": float(v) for group in ("holdout", "holdout_init", "train") for k, v in summary[group].items()}  # type: ignore[union-attr]
    flat.update({k: float(v) for k, v in summary.items() if isinstance(v, int | float)})
    recording.send_property("summary", rr.AnyValues(**flat))  # pyrefly: ignore[bad-argument-type]
    recording.flush()
    if config.png_dir is not None:
        _save_pngs(config.png_dir, config.run_name, clip, holdout, orbit)
    if config.register and config.rr_config.save is None:
        rr.disconnect()
        register_run(config.runs_catalog_url, run_dir / "run.rrd", RUNS_DATASET)
    print(json.dumps(summary, indent=2, default=float))
    return summary
