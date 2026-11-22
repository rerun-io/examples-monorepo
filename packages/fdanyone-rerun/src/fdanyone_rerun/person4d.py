"""The 4D person: train Open4DGS / TRiGS on a finished run's generated views and log it, as Animated SOG, into the run's recording.

run directory (``cameras.json``, ``videos/NN.mp4``, ``gvhmr/``) -> every generated view, resized so its longer side is
``max_side``, with the GVHMR SMPL-X body as the clip's person (:func:`read_run_clip`) -> BiRefNet mattes in a crop
around the projected body (``open4dgs.masks``; the weights 4DAnyone already downloads) -> body-surface init ->
optional MoGe-2 prior -> TRiGS (``open4dgs.trigs_adapter.train_clip``) -> ASOG bytes (``open4dgs.asog.encode_asog``) ->
``world/person/splats``: the gsplat viewer's ``AnimatedSogSplats3D`` payload plus a one-pose stock preview, static,
``frame_origin`` 0, so the custom viewer plays it on the run's ``time`` and ``frame`` timelines. The recording is the
one output: no ``.asog``, checkpoint or mask files.

While it trains, :func:`train_person` reports step, it/s and ETA, logs the RGB loss and, every ``preview_every``
steps, a render of the front view on the ``train_step`` timeline. Both ``fdanyone-run --train`` and the app run it
through :class:`fdanyone_rerun.stream.RunStreamer`.
"""

import gc
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from gsplat_rust_renderer.asog_decode import parse_asog
from gsplat_rust_renderer.sog3d import PREVIEW_ROWS, log_asog
from jaxtyping import Float32, UInt8
from open4dgs.asog import AsogSettings, encode_asog
from open4dgs.body_init import body_surface_init
from open4dgs.catalog_clip import ClipPeople, MultiViewClip, fit_max_side, resample_images, scale_intrinsics
from open4dgs.init_cloud import InitCloud
from open4dgs.masks import fill_missing_boxes, load_birefnet, person_boxes, predict_mattes, resample_masks
from open4dgs.prior import ClipPriors, MonoPriorLoss, PriorConfig
from open4dgs.splat_export import ASOG_CUT_FORMAT
from open4dgs.trigs_adapter import TrainConfig, TrainedClip, train_clip
from torch import Tensor

from fdanyone.errors import RunCancelled
from fdanyone.video import read_rgb_video
from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.outputs import CameraRig, RunMetadata, read_camera_rig, read_run_metadata
from fdanyone_rerun.rerun_log import PERSON, TRAIN_LOSS, TRAIN_PREVIEW, TRAIN_TIMELINE, camera_entity, load_body

MASK_PAD_FRACTION: float = 0.15
"""BiRefNet crop margin around the projected body, as a fraction of its box (``open4dgs-masks``' default)."""
MASK_PAD_PX: float = 24.0
"""... plus this many pixels."""
Report = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PersonConfig:
    """Train the 4D person of a 4DAnyone run (Open4DGS / TRiGS) and log it as Animated SOG."""

    train: TrainConfig = field(default_factory=lambda: TrainConfig(steps=25_000, cameras_per_step=8))
    """Trainer settings: those of the measured runs on the 18-view example (12 train + 4 held out, prior off): 25,000 steps
    23.07 dB held out (539 s on the 5090), 5,000 steps 22.44 dB."""
    max_side: int | None = None
    """Longer side of the training frames; None keeps the generated 704x1280 (as the measured runs)."""
    prior: bool = True
    """MoGe-2 depth + normal prior on every view (MoGe-2 runs first, 16-22 images/s; open4dgs' default); off = no extra estimator model."""
    prior_config: PriorConfig = field(default_factory=PriorConfig)
    """Prior weights and schedule (fractions of ``train.steps``)."""
    asog: AsogSettings = field(default_factory=AsogSettings)
    """ASOG encoder preset (``band-safe``: held-out PSNR within 1 % of the uncompressed model)."""
    samples_per_person: int = 8_000
    """Body-surface init samples per keyframe."""
    keyframe_step: int = 3
    """Frames between init keyframes."""
    preview_every: int = 500
    """Steps between preview renders (0: none)."""
    report_every: int = 100
    """Steps between progress lines and loss points."""


@dataclass(frozen=True, slots=True)
class PersonSummary:
    """What a training run took and produced."""

    steps: int
    it_per_s: float
    rows: int
    """Gaussians in the trained model."""
    asog_rows: int
    """Gaussians in the ASOG (after the encoder's pruning)."""
    asog_bytes: int
    peak_gpu_bytes: int
    """Peak CUDA memory allocated while training."""
    seconds: dict[str, float]
    """Wall time per stage: read, masks, init, prior, train, encode, log."""


def read_run_clip(output_dir: Path, max_side: int | None, device: torch.device) -> tuple[MultiViewClip, tuple[Float32[Tensor, "frames h w"], ...]]:
    """Every generated view of the run as a training clip, and each view's soft person matte on the clip's grid.

    Frames and mattes are computed at the stored 704x1280 and resized together (``resample_images``), so their pixels agree.
    """
    rig: CameraRig = read_camera_rig(output_dir)
    metadata: RunMetadata = read_run_metadata(output_dir)
    models: Path = model_dir()
    body = load_body(output_dir, models)
    num_frames: int = metadata.input.num_frames
    if len(body.vertices) != num_frames:
        raise ValueError(f"{output_dir}: the body has {len(body.vertices)} frames, the clip {num_frames}")
    vertices: Float32[np.ndarray, "frames verts 3"] = body.vertices.astype(np.float32)
    matte_model: torch.nn.Module = load_birefnet(models / "birefnet", device)
    frames: list[UInt8[Tensor, "frames h w 3"]] = []
    masks: list[Float32[Tensor, "frames h w"]] = []
    K: list[np.ndarray] = []
    sizes: list[tuple[int, int]] = []
    try:
        for camera in rig.cameras:
            stored_wh: tuple[int, int] = (camera.image_width, camera.image_height)
            target_wh: tuple[int, int] = fit_max_side(*stored_wh, max_side)
            stored: UInt8[Tensor, "frames h w 3"] = torch.from_numpy(np.stack(read_rgb_video(output_dir / camera.video, expected_frames=num_frames)))
            boxes = fill_missing_boxes(
                person_boxes(
                    vertices, camera.world_T_cam.astype(np.float32), camera.intrinsics.astype(np.float32), stored_wh, MASK_PAD_FRACTION, MASK_PAD_PX
                )
            )
            mattes: UInt8[Tensor, "frames h w"] = predict_mattes(matte_model, stored, boxes, device)
            masks.append(resample_masks(mattes, None, target_wh, device, torch.device("cpu"), torch.float32))
            resized: list[Tensor] = [
                resample_images(stored[start : start + 16].to(device).permute(0, 3, 1, 2).float(), None, target_wh)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                for start in range(0, num_frames, 16)
            ]
            frames.append(torch.cat(resized).contiguous())
            K.append(scale_intrinsics(camera.intrinsics, stored_wh, target_wh))
            sizes.append(stored_wh)
    finally:
        del matte_model
        gc.collect()
        torch.cuda.empty_cache()
    people: ClipPeople = ClipPeople(
        source="pred/fdanyone",
        names=("person_00",),
        vertices=torch.from_numpy(vertices)[:, None],
        present=torch.ones((num_frames, 1), dtype=torch.bool),
        faces=torch.from_numpy(body.faces.astype(np.uint32)),
    )
    clip: MultiViewClip = MultiViewClip(
        camera_names=tuple(camera_entity(camera.camera_id).removeprefix("world/") for camera in rig.cameras),
        frames=tuple(frames),
        valid=tuple(torch.ones(f.shape[1:3], dtype=torch.bool) for f in frames),
        K=torch.from_numpy(np.stack(K)).float(),
        world_T_cam=torch.from_numpy(np.stack([camera.world_T_cam for camera in rig.cameras])).float(),
        fps=float(metadata.input.frame_rate),
        frame_indices=torch.arange(num_frames, dtype=torch.int64),
        people=people,
        stored_sizes=tuple(sizes),
        lenses=(None,) * len(rig.cameras),
    )
    return clip, tuple(masks)


def _prior_loss(clip: MultiViewClip, masks: tuple[Tensor, ...], config: PersonConfig) -> MonoPriorLoss:
    """MoGe-2 maps of every view (all of them train) and the prior loss on them; the predictor is freed before training."""
    from monopriors.models.moge_v2.predictor import MoGeV2TorchPredictor
    from open4dgs.prior_layer import compute_clip_priors

    predictor = MoGeV2TorchPredictor(encoder="vitl", heads="geometry", resolution_level=9)
    priors: ClipPriors = compute_clip_priors(clip, clip.camera_names, predictor)
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    positions: tuple[int, ...] = tuple(range(len(clip.camera_names)))
    return MonoPriorLoss(config.prior_config.at_steps(config.train.steps), priors, masks, positions, clip.K)


def free_gpu() -> tuple[float, float]:
    """Collect garbage and return cached CUDA blocks; returns (allocated, reserved) GB after."""
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9


def train_person(
    output_dir: Path, config: PersonConfig, recording: rr.RecordingStream, *, report: Report = print, cancel: threading.Event | None = None
) -> PersonSummary:
    """Train the person of the run at ``output_dir`` and log it into ``recording`` (see the module docstring).

    ``report`` gets the progress lines; a set ``cancel`` stops the run at the next step (``RunCancelled``).
    """
    device: torch.device = torch.device("cuda")
    seconds: dict[str, float] = {}
    allocated, reserved = free_gpu()
    report(f"training the 4D person: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved before it starts")

    def stage(name: str, started: float) -> None:
        if cancel is not None and cancel.is_set():
            raise RunCancelled("stopped")
        seconds[name] = time.perf_counter() - started

    started: float = time.perf_counter()
    clip, masks = read_run_clip(output_dir, config.max_side, device)
    stage("read_and_masks", started)
    report(
        f"{len(clip.camera_names)} views x {len(clip.frame_indices)} frames at {clip.sizes[0][0]}x{clip.sizes[0][1]}, mattes in {seconds['read_and_masks']:.0f} s"
    )
    started = time.perf_counter()
    init: InitCloud = body_surface_init(
        clip,
        clip.camera_names,
        dataset="fdanyone",
        segment_id=output_dir.name,
        samples_per_person=config.samples_per_person,
        keyframe_step=config.keyframe_step,
    )
    stage("init", started)
    prior: MonoPriorLoss | None = None
    if config.prior:
        started = time.perf_counter()
        report("MoGe-2 depth + normal prior on every view")
        prior = _prior_loss(clip, masks, config)
        stage("prior", started)

    steps: int = config.train.steps
    front: int = 0
    middle: int = len(clip.frame_indices) // 2
    tick: list[float] = [time.perf_counter(), 0.0]  # time and step of the last report

    def on_step(done: int, live: TrainedClip) -> None:
        if cancel is not None and cancel.is_set():
            raise RunCancelled("stopped")
        if done % config.report_every == 0 or done == steps:
            now: float = time.perf_counter()
            rate: float = (done - tick[1]) / max(now - tick[0], 1e-9)
            tick[0], tick[1] = now, float(done)
            report(f"train step {done:,} / {steps:,}: {rate:.0f} it/s, ETA {(steps - done) / max(rate, 1e-9):.0f} s")
            losses = live.scalars.get("loss/rgb")
            if losses:
                recording.set_time(TRAIN_TIMELINE, sequence=done)
                recording.log(TRAIN_LOSS, rr.Scalars(losses[-1][1]))
                recording.reset_time()
        if config.preview_every > 0 and (done % config.preview_every == 0 or done == steps):
            rgb, _ = live.render(live.cameras, front, middle)
            recording.set_time(TRAIN_TIMELINE, sequence=done)
            recording.log(TRAIN_PREVIEW, rr.Image((rgb * 255).round().byte().cpu().numpy()).compress(jpeg_quality=85))
            recording.reset_time()

    started = time.perf_counter()
    trained: TrainedClip = train_clip(clip, masks, init, config.train, (), device=device, prior=prior, on_step=on_step)
    seconds["train"] = time.perf_counter() - started
    rows: int = int(trained.params["means"].shape[0])
    params: dict[str, Tensor] = {name: value.detach().cpu() for name, value in trained.params.items()}
    it_per_s: float = steps / trained.train_seconds
    peak: int = trained.peak_gpu_bytes
    del trained, prior, masks
    free_gpu()

    started = time.perf_counter()
    asog, stats = encode_asog(params, clip.fps, len(clip.frame_indices), config.asog)
    stage("encode", started)
    started = time.perf_counter()
    log_asog(PERSON, parse_asog(asog, f"{output_dir.name} person"), 0.0, ASOG_CUT_FORMAT, PREVIEW_ROWS, fps=clip.fps, recording=recording)
    seconds["log"] = time.perf_counter() - started
    summary: PersonSummary = PersonSummary(
        steps=steps, it_per_s=it_per_s, rows=rows, asog_rows=stats.count, asog_bytes=len(asog), peak_gpu_bytes=peak, seconds=seconds
    )
    report(
        f"4D person: {steps:,} steps at {it_per_s:.0f} it/s, {rows:,} Gaussians -> ASOG {stats.count:,} rows, {len(asog) / 1e6:.1f} MB; "
        f"peak {peak / 1e9:.1f} GB; " + ", ".join(f"{name} {value:.0f} s" for name, value in seconds.items())
    )
    return summary
