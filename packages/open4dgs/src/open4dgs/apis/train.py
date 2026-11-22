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
import rerun.blueprint as rrb
import torch
from PIL import Image
from simplecv.rerun_log_utils import RerunTyroConfig

from open4dgs.body_init import body_surface_init, mesh_silhouette_masks
from open4dgs.catalog_clip import MultiViewClip, read_catalog_clip
from open4dgs.evaluate import CameraGeometry, CameraRenders, Metrics, evaluate_cameras, evaluate_geometry, mean_geometry, mean_metrics, render_orbit
from open4dgs.holdout import select_holdout
from open4dgs.init_cloud import InitCloud, log_init_cloud
from open4dgs.prior import ClipPriors, MonoPriorLoss, PriorConfig, decode_normals, read_clip_priors
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
    holdout_cameras: tuple[str, ...] | None = None
    """Explicit held-out cameras (e.g. a split chosen on the full rig while ``cameras`` reads a subset); overrides ``num_holdout``/``holdout_seed``."""
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
    """Also save held-out gt|render pairs (training cameras when none is held out) at three frames and four orbit frames as PNGs here."""
    eval_train_cameras: bool = True
    """Score the training cameras too (train/held-out gap)."""
    register: bool = True
    runs_catalog_url: str = LOCAL_CATALOG
    prior: bool = False
    """Train with the MoGe-2 depth + normal prior (layer ``moge2_prior``, ``open4dgs-prior``) on the training cameras."""
    prior_catalog_url: str | None = None
    """Catalog of the ``moge2_prior`` layer (default ``mask_catalog_url``, then ``catalog_url``)."""
    prior_dataset: str | None = None
    """Dataset of the ``moge2_prior`` layer (default ``mask_dataset``, then ``dataset``)."""
    prior_config: PriorConfig = field(default_factory=PriorConfig)
    geometry: bool = True
    """Score rendered depth against the SMPL-X mesh depth at the held-out cameras (mesh frames only)."""


def _read_masks(config: Config, clip: MultiViewClip) -> tuple[torch.Tensor, ...]:
    if config.mask_source == "mesh":
        return mesh_silhouette_masks(clip)
    from open4dgs.masks import read_clip_masks

    return read_clip_masks(clip, config.mask_catalog_url or config.catalog_url, config.mask_dataset or config.dataset, config.segment_id)


def _save_pngs(png_dir: Path, run_name: str, clip: MultiViewClip, pairs: list[CameraRenders], orbit: torch.Tensor, kind: str) -> None:
    """gt | render pairs at three frames per camera, and four orbit frames."""
    png_dir.mkdir(parents=True, exist_ok=True)
    n: int = len(clip.frame_indices)
    for renders in pairs:
        cam: int = clip.camera_names.index(renders.camera)
        for f in (n // 6, n // 2, (5 * n) // 6):
            pair = np.concatenate([clip.frames[cam][f].numpy(), renders.render[f].numpy()], axis=1)
            Image.fromarray(pair).save(png_dir / f"{run_name}-{kind}-{renders.camera.replace('/', '_')}-f{int(clip.frame_indices[f])}.png")
    for f in (0, len(orbit) // 4, len(orbit) // 2, (3 * len(orbit)) // 4):
        Image.fromarray(orbit[f].numpy()).save(png_dir / f"{run_name}-orbit-{f:03d}.png")


def _depth_png(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Turbo-coloured depth in [lo, hi] (black = no depth)."""
    import cv2

    scaled: np.ndarray = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb: np.ndarray = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    return np.where((depth > 0)[..., None], rgb, 0).astype(np.uint8)


def _normal_png(normals: np.ndarray) -> np.ndarray:
    return np.where((np.abs(normals).sum(-1) > 0)[..., None], (127.5 + 127.5 * normals).round().clip(0, 255), 0).astype(np.uint8)


def _geometry_images(clip: MultiViewClip, cam: CameraGeometry, slot: int, priors: ClipPriors | None) -> dict[str, np.ndarray]:
    """Images of one held-out camera at one mesh frame: rendered depth + depth normals, mesh depth, MoGe-2 depth + normals."""
    from open4dgs.prior import camera_normals

    f: int = cam.frame_slots[slot]
    index: int = clip.camera_names.index(cam.camera)
    K: torch.Tensor = clip.K[index]
    mesh: np.ndarray = cam.mesh_depth[slot].numpy()
    ref: np.ndarray = mesh[mesh > 0] if (mesh > 0).any() else cam.depth[slot].numpy().reshape(-1)
    lo, hi = float(np.percentile(ref, 1)) - 0.3, float(np.percentile(ref, 99)) + 0.3
    out: dict[str, np.ndarray] = {
        "render_depth": _depth_png(cam.depth[slot].numpy(), lo, hi),
        "render_normals": _normal_png(camera_normals(cam.depth[slot], K).numpy()),
        "mesh_depth": _depth_png(mesh, lo, hi),
    }
    if priors is not None and cam.camera in priors.camera_names:
        p: int = priors.camera_names.index(cam.camera)
        moge: np.ndarray = priors.depth[p][f].float().numpy()
        valid = moge[moge > 0]
        out["moge2_depth"] = _depth_png(moge, float(np.percentile(valid, 1)), float(np.percentile(valid, 99))) if valid.size else np.zeros_like(out["mesh_depth"])
        out["moge2_normals"] = _normal_png(decode_normals(priors.normals[p][f])[0].numpy())
    return out


def _log_geometry(recording: rr.RecordingStream, clip: MultiViewClip, geometry: list[CameraGeometry], priors: ClipPriors | None) -> None:
    """Held-out rendered depth / depth normals / mesh depth (+ MoGe-2 maps when present) and the geometry scalars."""
    for cam in geometry:
        base: str = f"/eval/holdout/{cam.camera.replace('/', '_')}/geometry"
        for slot, f in enumerate(cam.frame_slots):
            recording.set_time("frame_index", sequence=int(clip.frame_indices[f]))
            for name, image in _geometry_images(clip, cam, slot, priors).items():
                recording.log(f"{base}/{name}", rr.Image(image).compress(jpeg_quality=90))
        for name in ("median_abs_cm", "within_3cm", "median_normal_deg"):
            recording.send_columns(
                f"{base}/metrics/{name}",
                indexes=[rr.TimeColumn("frame_index", sequence=[row.frame_index for row in cam.metrics])],
                columns=rr.Scalars.columns(scalars=np.asarray([getattr(row, name) for row in cam.metrics], dtype=np.float64)),
            )


def _save_geometry_pngs(png_dir: Path, run_name: str, clip: MultiViewClip, geometry: list[CameraGeometry], priors: ClipPriors | None) -> None:
    for cam in geometry:
        if not cam.frame_slots:
            continue
        slot: int = len(cam.frame_slots) // 2
        images = _geometry_images(clip, cam, slot, priors)
        rgb: np.ndarray = clip.frames[clip.camera_names.index(cam.camera)][cam.frame_slots[slot]].numpy()
        row = np.concatenate([rgb, *images.values()], axis=1)
        frame: int = int(clip.frame_indices[cam.frame_slots[slot]])
        Image.fromarray(row).save(png_dir / f"{run_name}-geometry-{cam.camera.replace('/', '_')}-f{frame}-{'_'.join(images)}.png")


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
    if config.holdout_cameras is not None:
        missing: set[str] = set(config.holdout_cameras) - set(clip.camera_names)
        if missing:
            raise ValueError(f"holdout cameras {sorted(missing)} are not in the clip {clip.camera_names}")
        holdout_names: tuple[str, ...] = config.holdout_cameras
        train_names: tuple[str, ...] = tuple(sorted(name for name in clip.camera_names if name not in holdout_names))
    else:
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

    prior_loss: MonoPriorLoss | None = None
    holdout_priors: ClipPriors | None = None
    if config.prior:
        prior_url: str = config.prior_catalog_url or config.mask_catalog_url or config.catalog_url
        prior_dataset: str = config.prior_dataset or config.mask_dataset or config.dataset
        t0 = time.perf_counter()
        train_priors: ClipPriors = read_clip_priors(clip, prior_url, prior_dataset, config.segment_id, train_names)
        try:  # held-out maps only for the side-by-side in the run recording; never handed to the trainer
            holdout_priors = read_clip_priors(clip, prior_url, prior_dataset, config.segment_id, holdout_names)
        except ValueError as error:
            print(f"[open4dgs] no held-out prior maps for the recording: {error}")
        timings["prior_read_s"] = time.perf_counter() - t0
        positions: tuple[int, ...] = tuple(clip.camera_names.index(name) for name in train_names)
        prior_loss = MonoPriorLoss(config.prior_config, train_priors, tuple(masks[c] for c in positions), positions, clip.K)
    trained: TrainedClip = train_clip(clip, masks, init, config.train, holdout_names, device=device, prior=prior_loss)
    trained.save(run_dir / "trigs_ckpt.pt")
    t0 = time.perf_counter()
    holdout: list[CameraRenders] = evaluate_cameras(trained, clip, masks, holdout_names, metrics)
    train_eval: list[CameraRenders] = (
        # With no held-out camera the training renders are kept for the gt | render PNGs.
        evaluate_cameras(trained, clip, masks, train_names, metrics, keep_renders=not holdout_names) if config.eval_train_cameras else []
    )
    orbit, _ = render_orbit(trained, clip, train_names, device)
    geometry: list[CameraGeometry] = evaluate_geometry(trained, clip, masks, holdout_names, device) if config.geometry else []
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
        "geometry_holdout": mean_geometry([row for cam in geometry for row in cam.metrics]),
        **{f"{tag.replace('/', '_')}_{edge}": rows[i][1] for tag in ("prior/depth", "prior/normal") if (rows := trained.scalars.get(tag)) for edge, i in (("first", 0), ("last", -1))},
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
    _log_geometry(recording, clip, geometry, holdout_priors)
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
        "prior": config.prior,
        **({f"prior_{k}": v for k, v in dataclasses.asdict(config.prior_config).items()} if config.prior else {}),
        "mask_source": config.mask_source if config.mask_source == "mesh" else f"layer:{config.mask_dataset or config.dataset}/masks",
        **{f"train_{k}": v for k, v in dataclasses.asdict(config.train).items()},
    }
    # Default layout: held-out gt / render / error (first two held-out cameras), the orbit, the 3D scene and the metrics.
    eval_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=f"/eval/holdout/{name.replace('/', '_')}/{kind}", name=f"{name} {kind}")
        for name in holdout_names[:2]
        for kind in ("gt", "render", "error")
    ]
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*eval_views, rrb.Spatial2DView(origin="/eval/orbit/render", name="orbit")),
                rrb.Horizontal(rrb.Spatial3DView(origin="/world", name="scene"), rrb.TimeSeriesView(origin="/eval", name="metrics")),
            ),
            collapse_panels=True,
        )
    )
    recording.send_property("run", rr.AnyValues(**run_properties))  # pyrefly: ignore[bad-argument-type]
    flat: dict[str, float] = {f"{group}_{k}": float(v) for group in ("holdout", "holdout_init", "train", "geometry_holdout") for k, v in summary[group].items()}  # type: ignore[union-attr]
    flat.update({k: float(v) for k, v in summary.items() if isinstance(v, int | float)})
    recording.send_property("summary", rr.AnyValues(**flat))  # pyrefly: ignore[bad-argument-type]
    recording.flush()
    if config.png_dir is not None:
        _save_pngs(config.png_dir, config.run_name, clip, holdout or train_eval, orbit, "holdout" if holdout else "train")
        _save_geometry_pngs(config.png_dir, config.run_name, clip, geometry, holdout_priors)
    if config.rr_config.save is None:
        rr.disconnect()  # closes run.rrd before chmod and registration
    # NAS rule: files 644, directories 775.
    run_dir.chmod(0o775)
    for path in run_dir.iterdir():
        path.chmod(0o644)
    if config.register and config.rr_config.save is None:
        register_run(config.runs_catalog_url, run_dir / "run.rrd", RUNS_DATASET)
    print(json.dumps(summary, indent=2, default=float))
    return summary
