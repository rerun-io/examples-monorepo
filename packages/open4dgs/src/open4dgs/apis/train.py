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
from open4dgs.compensation import BrightnessSplit, HoldoutFit, camera_brightness, evaluate_holdout_fit
from open4dgs.evaluate import (
    BASELINES,
    HEADLINE,
    CameraGeometry,
    CameraRenders,
    Metrics,
    evaluate_cameras,
    evaluate_geometry,
    mean_geometry,
    mean_metrics,
    render_orbit,
)
from open4dgs.holdout import select_holdout
from open4dgs.init_cloud import InitCloud, check_init_matches_run, log_init_cloud, read_init_cloud, read_layer_init_cloud
from open4dgs.prior import ClipPriors, MonoPriorLoss, PriorConfig, decode_normals, read_clip_priors
from open4dgs.run_log import RUNS_DATASET, log_camera_eval, log_cameras, log_metric_rows, log_people, log_scalars, log_video, register_run
from open4dgs.splat_export import up_view_coordinates
from open4dgs.trigs_adapter import CameraCompensation, TrainConfig, TrainedClip, train_clip, untrained_clip

LOCAL_CATALOG: str = "rerun+http://pablo-dl-server:51235"


@dataclass
class Config:
    """One training run."""

    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(headless=True))
    """Rerun sinks. ``save`` defaults to ``<runs_root>/<run_name>/run.rrd`` and ``recording_id`` to the run name; the saved file is always registered (``--no-register`` turns that off)."""
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
    roi: Literal["people"] | None = None
    """``people``: person-centred virtual cameras (roi_cameras.py) instead of the full frame; metrics are then on the ROI views."""
    roi_pad: float = 0.15
    """ROI margin around the people's angular box (fraction)."""
    mask_source: Literal["layer", "mesh"] = "layer"
    """``layer``: the catalog ``masks`` layer (open4dgs.masks); ``mesh``: dilated SMPL-X silhouette (stand-in, development only)."""
    mask_catalog_url: str | None = None
    """Catalog of the masks layer (default ``catalog_url``; MAMMA masks live on the local ``open4dgs-mamma``)."""
    mask_dataset: str | None = None
    """Dataset of the masks layer (default ``dataset``)."""
    init: Literal["body", "layer"] = "body"
    """``body``: body-surface samples on the SMPL-X meshes; ``layer``: the segment's init cloud layer ``init_layer`` on the mask catalog/dataset (its held-out cameras must equal the run's)."""
    init_layer: str = "init_roma"
    """Catalog layer of the init cloud for ``--init layer``."""
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
    """Register the saved run recording on ``runs_catalog_url`` / ``open4dgs-runs`` (the one switch; ``--no-register`` turns it off)."""
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
    selfcap_time_start_s: float | None = None
    """Read a SelfCap segment (phones on ``video_time``, ``open4dgs.selfcap_clip``) from this time instead of ``frame_start``."""
    selfcap_roi: bool = False
    """With ``selfcap_time_start_s``: person-centred virtual cameras framing the joints (``read_timed_clip(roi=True)``)."""
    init_rrd: Path | None = None
    """Use this init-cloud recording (``open4dgs.apis.selfcap --stage init``) instead of ``init`` (body surface or layer)."""
    selfcap_capture_times: bool = False
    """With ``selfcap_time_start_s``: give the trainer each phone's own frame times (``video_time``, the dataset's sync)
    instead of the reference grid; the nearest-frame pick leaves a sync residual of up to half a frame (14 ms here)."""
    holdout_fit_steps: int = 300
    """Held-out test-time fit of the compensations the run learned (``open4dgs.compensation``): steps per held-out camera
    on its even frames, both protocols scored on its odd frames; 0 = off. Runs only with ``train.learn_appearance`` or
    ``train.learn_time_offsets``."""


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
        out["moge2_depth"] = (
            _depth_png(moge, float(np.percentile(valid, 1)), float(np.percentile(valid, 99))) if valid.size else np.zeros_like(out["mesh_depth"])
        )
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


def split_cameras(
    camera_names: tuple[str, ...], holdout_cameras: tuple[str, ...] | None, num_holdout: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(train names sorted, held-out names): the explicit ``holdout_cameras`` when given, else ``select_holdout``."""
    if holdout_cameras is None:
        return select_holdout(camera_names, num_holdout, seed)
    missing: set[str] = set(holdout_cameras) - set(camera_names)
    if missing:
        raise ValueError(f"holdout cameras {sorted(missing)} are not in the clip {camera_names}")
    return tuple(sorted(set(camera_names) - set(holdout_cameras))), holdout_cameras


def main(config: Config) -> dict[str, object]:
    """Run the whole pipeline; returns the summary (also printed as JSON and stored as the ``summary`` property)."""
    device: torch.device = torch.device("cuda")
    run_dir: Path = config.runs_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # The run recording always goes through RerunTyroConfig: its save target defaults to <run_dir>/run.rrd and its
    # recording id (= the catalog segment id) to the run name. Rebuilding it re-runs __post_init__ (rr.init + sinks).
    rr_config: RerunTyroConfig = dataclasses.replace(
        config.rr_config,
        application_id=RUNS_DATASET,
        recording_id=config.rr_config.recording_id or config.run_name,
        save=config.rr_config.save or run_dir / "run.rrd",
    )
    rrd_path: Path = rr_config.save  # type: ignore[assignment]
    recording: rr.RecordingStream = rr_config.rec_stream
    timings: dict[str, float] = {}
    t0: float = time.perf_counter()
    capture_times: np.ndarray | None = None
    if config.selfcap_time_start_s is not None:
        from open4dgs.selfcap_clip import read_timed_clip

        timed = read_timed_clip(
            config.catalog_url,
            config.dataset,
            config.segment_id,
            config.selfcap_time_start_s,
            config.num_frames,
            cameras=config.cameras,
            max_side=config.max_side,
            roi=config.selfcap_roi,
        )
        clip: MultiViewClip = timed.clip
        if config.selfcap_capture_times:
            capture_times = timed.times_s
            print(f"[open4dgs] capture times: offset to the reference grid (ms) {np.round((timed.times_s - timed.times_s[0]) * 1e3, 1).max(axis=1).tolist()}")
    else:
        if config.selfcap_capture_times:
            raise ValueError("--selfcap-capture-times needs --selfcap-time-start-s")
        clip = read_catalog_clip(
            config.catalog_url,
            config.dataset,
            config.segment_id,
            config.frame_start,
            config.num_frames,
            cameras=config.cameras,
            max_side=config.max_side,
            roi=config.roi,
            roi_pad=config.roi_pad,
        )
    timings["read_clip_s"] = time.perf_counter() - t0
    train_names, holdout_names = split_cameras(clip.camera_names, config.holdout_cameras, config.num_holdout, config.holdout_seed)
    t0 = time.perf_counter()
    masks = _read_masks(config, clip)
    timings["masks_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    init: InitCloud
    if config.init_rrd is not None:
        init = read_init_cloud(config.init_rrd)
        if init.frame_end != len(clip.frame_indices) - 1 or init.source_frame_start != int(clip.frame_indices[0]):
            raise ValueError(f"{config.init_rrd}: init window {init.source_frame_start}+{init.frame_end + 1} does not match the clip")
    elif config.init == "layer":
        init, init_holdout = read_layer_init_cloud(
            config.mask_catalog_url or config.catalog_url, config.mask_dataset or config.dataset, config.segment_id, config.init_layer
        )
        check_init_matches_run(init, init_holdout, holdout_names, int(clip.frame_indices[0]), len(clip.frame_indices))
        if init.fps != clip.fps:
            # Layer times are normalised over the window and velocities are per video frame, so fps only maps time to
            # seconds; it must be the clip's, which the trainer renders with (frame / clip.fps). The MAMMA init_roma
            # layer records 60 while the segment's capture fps is 30: without this, frames past 0.95 s have no Gaussians.
            print(f"[open4dgs] init layer fps {init.fps} != clip fps {clip.fps}: using the clip fps")
            init = dataclasses.replace(init, fps=float(clip.fps))
    else:
        init = body_surface_init(
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
    trained: TrainedClip = train_clip(clip, masks, init, config.train, holdout_names, device=device, prior=prior_loss, capture_times=capture_times)
    trained.save(run_dir / "trigs_ckpt.pt")
    t0 = time.perf_counter()
    holdout: list[CameraRenders] = evaluate_cameras(trained, clip, masks, holdout_names, metrics, with_baselines=True)
    train_eval: list[CameraRenders] = (
        # With no held-out camera the training renders are kept for the gt | render PNGs.
        evaluate_cameras(trained, clip, masks, train_names, metrics, keep_renders=not holdout_names) if config.eval_train_cameras else []
    )
    orbit_centre: torch.Tensor | None = torch.from_numpy(np.median(init.positions, axis=0)) if clip.people is None else None
    orbit, _ = render_orbit(trained, clip, train_names, device, centre=orbit_centre)
    geometry: list[CameraGeometry] = evaluate_geometry(trained, clip, masks, holdout_names, device) if config.geometry else []
    timings["eval_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    train_appearance: list[CameraRenders] = (
        evaluate_cameras(trained, clip, masks, train_names, metrics, keep_renders=False, appearance=True)
        if config.train.learn_appearance and config.eval_train_cameras
        else []
    )
    fits: list[HoldoutFit] = (
        evaluate_holdout_fit(
            trained, clip, masks, holdout_names, metrics, config.holdout_fit_steps, config.train.learn_appearance, config.train.learn_time_offsets
        )
        if config.holdout_fit_steps > 0 and (config.train.learn_appearance or config.train.learn_time_offsets)
        else []
    )
    all_frames: list[int] = list(range(trained.num_frames))
    brightness: dict[str, BrightnessSplit] = {name: camera_brightness(trained, clip, masks, name, all_frames) for name in holdout_names}
    for fit in fits:
        odd: list[int] = list(range(1, trained.num_frames, 2))
        brightness[f"{fit.camera}:odd"] = camera_brightness(trained, clip, masks, fit.camera, odd)
        brightness[f"{fit.camera}:odd_fit"] = camera_brightness(trained, clip, masks, fit.camera, odd, fit.compensation)
    learned: list[CameraCompensation] = trained.learned()
    timings["compensation_eval_s"] = time.perf_counter() - t0

    holdout_mean = mean_metrics([row for cam in holdout for row in cam.metrics])
    train_mean = mean_metrics([row for cam in train_eval for row in cam.metrics]) if train_eval else {}
    losses = trained.scalars.get("loss/rgb", [])
    summary: dict[str, object] = {
        "holdout": holdout_mean,
        "holdout_init": init_holdout,
        **{f"holdout_floor_{kind}": mean_metrics([row for cam in holdout for row in cam.baselines[kind]]) for kind in BASELINES},
        "train": train_mean,
        **({"train_appearance": mean_metrics([row for cam in train_appearance for row in cam.metrics])} if train_appearance else {}),
        **(
            {
                "holdout_odd_identity": mean_metrics([row for fit in fits for row in fit.identity.metrics]),
                "holdout_odd_fit": mean_metrics([row for fit in fits for row in fit.fitted.metrics]),
            }
            if fits
            else {}
        ),
        "brightness": {name: dataclasses.asdict(split) for name, split in brightness.items()},
        "learned": {name: dataclasses.asdict(comp) for name, comp in zip(clip.camera_names, learned, strict=True)},
        "holdout_fitted": {fit.camera: dataclasses.asdict(fit.compensation) for fit in fits},
        "it_per_s": config.train.steps / trained.train_seconds,
        "train_s": trained.train_seconds,
        "peak_gpu_gb": trained.peak_gpu_bytes / 1e9,
        "init_rows": init.num_rows,
        "final_gaussians": int(trained.params["means"].shape[0]),
        "loss_rgb_first": losses[0][1] if losses else math.nan,
        "loss_rgb_last": losses[-1][1] if losses else math.nan,
        "geometry_holdout": mean_geometry([row for cam in geometry for row in cam.metrics]),
        **{
            f"{tag.replace('/', '_')}_{edge}": rows[i][1]
            for tag in ("prior/depth", "prior/normal")
            if (rows := trained.scalars.get(tag))
            for edge, i in (("first", 0), ("last", -1))
        },
        **timings,
    }

    # The world's up axis from the mean camera up (the segments declare none): the person stands upright in the 3D view.
    view_coordinates = up_view_coordinates(clip.world_T_cam.float().numpy())
    recording.log("/world", view_coordinates, static=True)
    log_cameras(recording, clip, holdout_names)
    log_people(recording, clip)
    log_init_cloud(init, recording)
    for renders in holdout:
        log_camera_eval(recording, clip, masks, renders, held_out=True)
    for renders in train_eval:
        log_metric_rows(recording, f"/eval/train/{renders.camera.replace('/', '_')}/metrics", renders.metrics, (*HEADLINE, "fg_psnr", "alpha_iou"))
    log_video(recording, "/eval/orbit/render", orbit, clip.frame_indices.tolist(), clip.fps)
    _log_geometry(recording, clip, geometry, holdout_priors)
    log_scalars(recording, trained.scalars)
    _log_compensation(recording, clip.camera_names, holdout_names, learned, fits, config.train)
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
        "roi": f"{config.roi}:{config.roi_pad}" if config.roi else "full",
        "init_method": f"rrd:{init.method}" if config.init_rrd is not None else init.method if config.init == "body" else f"layer:{config.init_layer}:{init.method}",
        "init_rrd": str(config.init_rrd or ""),
        "selfcap_roi": config.selfcap_roi,
        "selfcap_time_start_s": config.selfcap_time_start_s if config.selfcap_time_start_s is not None else -1.0,
        "view_coordinates": str(view_coordinates),
        "samples_per_person": config.samples_per_person,
        "keyframe_step": init.keyframe_step,
        "prior": config.prior,
        **({f"prior_{k}": v for k, v in dataclasses.asdict(config.prior_config).items()} if config.prior else {}),
        "mask_source": config.mask_source if config.mask_source == "mesh" else f"layer:{config.mask_dataset or config.dataset}/masks",
        **{f"train_{k}": ",".join(str(x) for x in v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(config.train).items() if v is not None},
        "train_time_offset_clamp_ms_resolved": float(trained.trigs_config.camera_time_offset_clamp_s) * 1e3,
        "rs_signs_per_camera": ",".join(f"{v:g}" for v in (config.train.rs_signs(len(clip.camera_names)) or ())),
        "selfcap_capture_times": config.selfcap_capture_times,
        "holdout_fit_steps": config.holdout_fit_steps if fits else 0,
        "holdout_protocol": "identity (gain 1, bias 0, gauge offset) on all frames; test-time fit on even frames scored on odd frames"
        if fits
        else "identity (gain 1, bias 0, gauge offset)",
    }
    # Default layout: held-out gt / render / error (first two held-out cameras), the orbit, the 3D scene and the metrics.
    eval_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=f"/eval/holdout/{name.replace('/', '_')}/{kind}", name=f"{name} {kind}")
        for name in holdout_names[:2]
        for kind in ("gt", "render", "error")
    ]
    # Metric panels: PSNR / SSIM / LPIPS of crop_raw (the headline) with both floors (held-out cameras, else training ones).
    scored: list[str] = [f"/eval/{'holdout' if holdout else 'train'}/{name.replace('/', '_')}" for name in (holdout_names or train_names)]

    def metric_view(metric: str, extra: tuple[str, ...] = ()) -> rrb.TimeSeriesView:
        contents: list[str] = [f"{base}/metrics/{name}" for base in scored for name in (metric, *extra)]
        contents += [f"{base}/floor/{kind}/{metric}" for base in scored for kind in BASELINES]
        return rrb.TimeSeriesView(origin="/eval", contents=contents, name=metric.replace("crop_raw_", "").upper())

    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*eval_views, rrb.Spatial2DView(origin="/eval/orbit/render", name="orbit")),
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/world", name="scene"),
                    metric_view(HEADLINE[0], ("fg_psnr",)),
                    metric_view(HEADLINE[1]),
                    metric_view(HEADLINE[2]),
                ),
            ),
            collapse_panels=True,
        )
    )
    recording.send_property("run", rr.AnyValues(**run_properties))  # pyrefly: ignore[bad-argument-type]
    groups: tuple[str, ...] = ("holdout", "holdout_init", *(f"holdout_floor_{kind}" for kind in BASELINES), "train", "geometry_holdout")
    groups += tuple(g for g in ("train_appearance", "holdout_odd_identity", "holdout_odd_fit") if g in summary)
    flat: dict[str, float] = {f"{group}_{k}": float(v) for group in groups for k, v in summary[group].items()}  # type: ignore[union-attr]
    flat.update({f"brightness_{name.replace('/', '_')}_{k}": float(v) for name, split in brightness.items() for k, v in dataclasses.asdict(split).items()})
    flat.update({k: float(v) for k, v in summary.items() if isinstance(v, int | float)})
    recording.send_property("summary", rr.AnyValues(**flat))  # pyrefly: ignore[bad-argument-type]
    recording.flush()
    if config.png_dir is not None:
        _save_pngs(config.png_dir, config.run_name, clip, holdout or train_eval, orbit, "holdout" if holdout else "train")
        for fit in fits:  # gt | identity | fitted at the middle odd frame
            j: int = len(fit.fitted.metrics) // 2
            f_clip: int = 2 * j + 1
            cam_index: int = clip.camera_names.index(fit.camera)
            strip = np.concatenate([clip.frames[cam_index][f_clip].numpy(), fit.identity.render[j].numpy(), fit.fitted.render[j].numpy()], axis=1)
            Image.fromarray(strip).save(config.png_dir / f"{config.run_name}-holdoutfit-{fit.camera.replace('/', '_')}-f{int(clip.frame_indices[f_clip])}.png")
        _save_geometry_pngs(config.png_dir, config.run_name, clip, geometry, holdout_priors)
    rr.disconnect()  # closes the .rrd before chmod and registration
    # NAS rule: files 644, directories 775.
    run_dir.chmod(0o775)
    for path in (*run_dir.iterdir(), rrd_path):
        path.chmod(0o644)
    if config.register:
        register_run(config.runs_catalog_url, rrd_path, RUNS_DATASET)
    print(json.dumps(summary, indent=2, default=float))
    print(headline_line(summary))
    return summary


def _log_compensation(
    recording: rr.RecordingStream,
    camera_names: tuple[str, ...],
    holdout_names: tuple[str, ...],
    learned: list[CameraCompensation],
    fits: list[HoldoutFit],
    train: TrainConfig,
) -> None:
    """The learned (and held-out fitted) per-camera values: a markdown table, one scalar per camera and quantity, a property."""
    fitted: dict[str, CameraCompensation] = {fit.camera: fit.compensation for fit in fits}
    lines: list[str] = [
        f"appearance {'on' if train.learn_appearance else 'off'}; time offsets {'on' if train.learn_time_offsets else 'off'}; "
        f"rolling shutter {'off' if not train.rs_readout_ms else ','.join(f'{r:g}' for r in train.rs_readout_ms) + ' ms ' + train.rs_axis}",
        "",
        "| camera | role | offset ms | gain r g b | bias r g b |",
        "|---|---|---|---|---|",
    ]
    for name, comp in zip(camera_names, learned, strict=True):
        rows: list[tuple[str, CameraCompensation]] = [("held out (identity)" if name in holdout_names else "train", comp)]
        if name in fitted:
            rows.append(("held out (test-time fit)", fitted[name]))
        for role, value in rows:
            lines.append(
                f"| {name} | {role} | {value.offset_s * 1e3:+.2f} | {' '.join(f'{g:.3f}' for g in value.gain)} | {' '.join(f'{b:+.3f}' for b in value.bias)} |"
            )
    recording.log("/compensation/learned", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)
    for name, comp in zip(camera_names, learned, strict=True):
        base: str = f"/compensation/{name.replace('/', '_')}"
        recording.log(f"{base}/offset_ms", rr.Scalars(comp.offset_s * 1e3), static=True)
        for channel, gain, bias in zip("rgb", comp.gain, comp.bias, strict=True):
            recording.log(f"{base}/gain_{channel}", rr.Scalars(gain), static=True)
            recording.log(f"{base}/bias_{channel}", rr.Scalars(bias), static=True)
    values: dict[str, list[str] | list[float]] = {
        "camera": list(camera_names),
        "offset_ms": [c.offset_s * 1e3 for c in learned],
        **{f"gain_{ch}": [c.gain[i] for c in learned] for i, ch in enumerate("rgb")},
        **{f"bias_{ch}": [c.bias[i] for c in learned] for i, ch in enumerate("rgb")},
    }
    recording.send_property("compensation", rr.AnyValues(**values))  # pyrefly: ignore[bad-argument-type]


def headline_line(summary: dict[str, object]) -> str:
    """One line: held-out crop_raw PSNR/SSIM/LPIPS, with the init and both floors beside it, plus fg PSNR and alpha IoU."""

    def triple(group: str) -> str:
        values = summary.get(group)
        if not isinstance(values, dict) or not values:
            return "n/a"
        return "/".join(f"{values[name]:.3f}" if name != HEADLINE[0] else f"{values[name]:.2f}" for name in HEADLINE) + f" fg {values['fg_psnr']:.2f} iou {values['alpha_iou']:.3f}"

    groups = ("holdout", "holdout_init", *(f"holdout_floor_{kind}" for kind in BASELINES), "train")
    return "[open4dgs] crop_raw PSNR/SSIM/LPIPS " + "; ".join(f"{group} {triple(group)}" for group in groups)
