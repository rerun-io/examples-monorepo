"""The 4D person: train Open4DGS / TRiGS on a finished run's generated views and log it, as Animated SOG, into the run's recording.

run directory (``cameras.json``, ``videos/NN.mp4``, ``gvhmr/``) -> every generated view at its stored 704x1280 (or
resized with ``max_side``), with the GVHMR SMPL-X body as the clip's person (:func:`read_run_clip`) -> BiRefNet mattes
in a crop around the projected body (``open4dgs.masks``; the weights 4DAnyone already downloads) -> body-surface init
-> optional MoGe-2 prior -> TRiGS (``open4dgs.trigs_adapter.train_clip``) -> ASOG bytes (``open4dgs.asog.encode_asog``)
-> ``world/person/splats``: the gsplat viewer's ``AnimatedSogSplats3D`` payload plus a one-pose stock preview, static,
``frame_origin`` 0, so the custom viewer plays it on the run's ``time`` and ``frame`` timelines. The recording is the
one output: no ``.asog``, checkpoint or mask files.

:func:`train_person` is the one path of ``fdanyone-run --train``, ``fdanyone-log-run --train`` and the app: it shows
the training layout (step, it/s and ETA, the RGB loss and, every ``preview_every`` steps, a render of the front view
on ``train_step``), trains, logs the person and switches to the 4D person layout.
"""

import dataclasses
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from gsplat_rust_renderer.asog_decode import parse_asog
from gsplat_rust_renderer.sog3d import PREVIEW_ROWS, log_asog
from jaxtyping import Float32, UInt8
from open4dgs.asog import AsogSettings, encode_asog
from open4dgs.birefnet_trt import load_matte_model
from open4dgs.body_init import body_surface_init
from open4dgs.catalog_clip import RESAMPLE_CHUNK, ClipPeople, MultiViewClip, fit_max_side, resample_frames, scale_intrinsics
from open4dgs.init_cloud import InitCloud
from open4dgs.masks import MESH_PAD_FRACTION, MESH_PAD_PX, MatteNet, camera_matte, resample_masks
from open4dgs.prior import ClipPriors, MonoPriorLoss, PriorConfig
from open4dgs.splat_export import ASOG_CUT_FORMAT
from open4dgs.trigs_adapter import TrainConfig, TrainedClip, train_clip
from torch import Tensor

from fdanyone.errors import raise_if_stopped
from fdanyone.video import read_rgb_video
from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.core import free_gpu
from fdanyone_rerun.outputs import CameraRig, RunMetadata, read_camera_rig, read_run_metadata
from fdanyone_rerun.rerun_log import (
    PERSON,
    TRAIN_LOSS,
    TRAIN_PREVIEW,
    TRAIN_TIMELINE,
    BodyTrack,
    camera_entity,
    load_body,
    person_blueprint,
    training_blueprint,
)

Report = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PersonConfig:
    """Train the 4D person of a 4DAnyone run (Open4DGS / TRiGS) and log it as Animated SOG."""

    train: TrainConfig = field(default_factory=lambda: TrainConfig(steps=20_000, schedule_steps=25_000, cameras_per_step=8))
    """Trainer settings: 20,000 steps of the 25,000-step schedule (``TrainConfig.schedule_steps``). Run C (18-view example,
    12 train + 4 held out, prior on, stride 2; Spark 2026-11-27): the full 25,000 steps 23.38 dB / LPIPS 0.137 held out,
    stopped at 20,000 -0.1 % PSNR / -1.6 % LPIPS / geometry within 0.3 % for 20 % fewer steps; a schedule compressed to
    18,000 steps loses 4.8 % LPIPS. Set ``--person.train.schedule-steps`` with ``--person.train.steps`` (None = the
    schedule spans the steps). ``log_every`` is :attr:`report_every`'s."""
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
    """Steps between progress lines and loss points; the trainer records its scalars as often (``train.log_every``)."""

    def __post_init__(self) -> None:
        if self.report_every < 1:
            raise ValueError(f"report_every must be at least 1 (it is also the trainer's log_every), got {self.report_every}")

    def with_steps(self, steps: int) -> "PersonConfig":
        """This config with ``train.steps`` replaced and the schedule spanning them (the app's step presets)."""
        return dataclasses.replace(self, train=dataclasses.replace(self.train, steps=steps, schedule_steps=None))


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
    """Wall time per stage: read_and_masks, init, prior (when on), train, encode, log."""


def read_run_clip(
    output_dir: Path, max_side: int | None, device: torch.device, cancel: threading.Event | None = None
) -> tuple[MultiViewClip, tuple[Float32[Tensor, "frames h w"], ...]]:
    """Every generated view of the run as a training clip, and each view's soft person matte on the clip's grid.

    Mattes are computed at the stored 704x1280 and resized like the frames (``resample_images``), so their pixels agree.
    A set ``cancel`` stops before the next view.
    """
    rig: CameraRig = read_camera_rig(output_dir)
    metadata: RunMetadata = read_run_metadata(output_dir)
    models: Path = model_dir()
    num_frames: int = metadata.input.num_frames

    def decode(video: str) -> UInt8[Tensor, "frames h w 3"]:
        return torch.from_numpy(np.stack(read_rgb_video(output_dir / video, expected_frames=num_frames)))

    # PyAV decodes without the GIL: every view decodes on its own thread, and the body is rebuilt on another, while the
    # matte model loads and the earlier views' mattes run, instead of one after another on the critical path.
    pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=len(rig.cameras) + 1)
    pending_body: Future[BodyTrack] = pool.submit(load_body, output_dir, models)
    decoded: list[Future[UInt8[Tensor, "frames h w 3"]]] = [pool.submit(decode, camera.video) for camera in rig.cameras]
    frames: list[UInt8[Tensor, "frames h w 3"]] = []
    masks: list[Float32[Tensor, "frames h w"]] = []
    K: list[np.ndarray] = []
    sizes: list[tuple[int, int]] = []
    matte_model: MatteNet | None = None
    try:
        matte_model = load_matte_model(models / "birefnet", device)
        body: BodyTrack = pending_body.result()
        if len(body.vertices) != num_frames:
            raise ValueError(f"{output_dir}: the body has {len(body.vertices)} frames, the clip {num_frames}")
        vertices: Float32[np.ndarray, "frames verts 3"] = body.vertices.astype(np.float32)
        for camera in rig.cameras:
            raise_if_stopped(cancel)
            stored_wh: tuple[int, int] = (camera.image_width, camera.image_height)
            target_wh: tuple[int, int] = fit_max_side(*stored_wh, max_side)
            stored: UInt8[Tensor, "frames h w 3"] = decoded.pop(0).result()  # popped: a resized view frees its full-size frames
            mattes: UInt8[Tensor, "frames h w"] = camera_matte(
                matte_model,
                stored,
                vertices,
                camera.world_T_cam.astype(np.float32),
                camera.intrinsics.astype(np.float32),
                stored_wh,
                MESH_PAD_FRACTION,
                MESH_PAD_PX,
                device,
            )
            masks.append(resample_masks(mattes, None, target_wh, device, torch.device("cpu"), torch.float32))
            if target_wh != stored_wh:
                stored = torch.cat(
                    [
                        resample_frames(stored[start : start + RESAMPLE_CHUNK].to(device).permute(0, 3, 1, 2), None, target_wh)
                        .permute(0, 2, 3, 1)
                        .cpu()
                        for start in range(0, num_frames, RESAMPLE_CHUNK)
                    ]
                ).contiguous()
            frames.append(stored)
            K.append(scale_intrinsics(camera.intrinsics, stored_wh, target_wh))
            sizes.append(stored_wh)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        del matte_model
        free_gpu()
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


def _prior_loss(clip: MultiViewClip, masks: tuple[Tensor, ...], config: PersonConfig, report: Report) -> MonoPriorLoss:
    """MoGe-2 maps of every view (all of them train) and the prior loss on them; the predictor is freed before training.

    MoGe-2 runs through TensorRT when monopriors' engine cache already holds one for these frames on this machine (built
    by an earlier ``MoGeV2TrtPredictor`` run at the same size and settings), else through torch: a click never waits for an
    engine build.
    """
    from monopriors.models.moge_v2.predictor import MoGeV2TorchPredictor, MoGeV2TrtPredictor
    from open4dgs.prior_layer import compute_clip_priors

    width, height = clip.sizes[0]
    trt: MoGeV2TrtPredictor = MoGeV2TrtPredictor(encoder="vitl", heads="geometry")
    engine: bool = trt.has_engine((height, width))
    predictor: MoGeV2TrtPredictor | MoGeV2TorchPredictor = trt if engine else MoGeV2TorchPredictor(encoder="vitl", heads="geometry")
    report(f"MoGe-2 depth + normal prior on every view ({'TensorRT' if engine else 'torch'})")
    priors: ClipPriors = compute_clip_priors(clip, clip.camera_names, predictor)
    del predictor, trt
    free_gpu()
    positions: tuple[int, ...] = tuple(range(len(clip.camera_names)))
    return MonoPriorLoss(config.prior_config.at_steps(config.train.horizon), priors, masks, positions, clip.K)


def train_person(
    output_dir: Path, config: PersonConfig, recording: rr.RecordingStream, *, report: Report = print, cancel: threading.Event | None = None
) -> PersonSummary:
    """Train the person of the run at ``output_dir`` and log it into ``recording`` (see the module docstring).

    ``report`` gets the progress lines; a set ``cancel`` stops the run before the next view, stage or step (``RunCancelled``).
    """
    recording.send_blueprint(training_blueprint())
    device: torch.device = torch.device("cuda")
    seconds: dict[str, float] = {}

    @contextmanager
    def stage(name: str) -> Iterator[None]:
        """One stage of the run: a Stop is honoured before it starts; its wall time goes to ``seconds[name]``."""
        raise_if_stopped(cancel)
        started: float = time.perf_counter()
        yield
        seconds[name] = time.perf_counter() - started

    allocated, reserved = free_gpu()
    report(f"training the 4D person: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved before it starts")
    with stage("read_and_masks"):
        clip, masks = read_run_clip(output_dir, config.max_side, device, cancel)
    report(
        f"{len(clip.camera_names)} views x {len(clip.frame_indices)} frames at {clip.sizes[0][0]}x{clip.sizes[0][1]}, mattes in {seconds['read_and_masks']:.0f} s"
    )
    with stage("init"):
        init: InitCloud = body_surface_init(
            clip,
            clip.camera_names,
            dataset="fdanyone",
            segment_id=output_dir.name,
            samples_per_person=config.samples_per_person,
            keyframe_step=config.keyframe_step,
        )
    prior: MonoPriorLoss | None = None
    if config.prior:
        with stage("prior"):
            prior = _prior_loss(clip, masks, config, report)

    steps: int = config.train.steps
    middle: int = len(clip.frame_indices) // 2
    last_time: float = time.perf_counter()
    last_step: int = 0

    def on_step(done: int, live: TrainedClip) -> None:
        nonlocal last_time, last_step
        raise_if_stopped(cancel)
        if done % config.report_every == 0 or done == steps:
            now: float = time.perf_counter()
            rate: float = (done - last_step) / max(now - last_time, 1e-9)
            last_time, last_step = now, done
            report(f"train step {done:,} / {steps:,}: {rate:.0f} it/s, ETA {(steps - done) / max(rate, 1e-9):.0f} s")
            losses = live.scalars.get("loss/rgb")
            if losses:
                recording.set_time(TRAIN_TIMELINE, sequence=losses[-1][0])
                recording.log(TRAIN_LOSS, rr.Scalars(losses[-1][1]))
                recording.reset_time()
        if config.preview_every > 0 and (done % config.preview_every == 0 or done == steps):
            rgb, _ = live.render(live.cameras, 0, middle)  # the front view
            recording.set_time(TRAIN_TIMELINE, sequence=done)
            recording.log(TRAIN_PREVIEW, rr.Image((rgb * 255).round().byte().cpu().numpy()).compress(jpeg_quality=85))
            recording.reset_time()

    with stage("train"):
        train: TrainConfig = dataclasses.replace(config.train, log_every=config.report_every)  # a loss point per report
        trained: TrainedClip = train_clip(clip, masks, init, train, (), device=device, prior=prior, on_step=on_step)
    rows: int = int(trained.params["means"].shape[0])
    params: dict[str, Tensor] = {name: value.detach().cpu() for name, value in trained.params.items()}
    it_per_s: float = steps / trained.train_seconds
    peak: int = trained.peak_gpu_bytes
    fps, num_frames = clip.fps, len(clip.frame_indices)
    del trained, prior, masks, clip, init
    free_gpu()

    with stage("encode"):
        asog, stats = encode_asog(params, fps, num_frames, config.asog)
    with stage("log"):
        log_asog(PERSON, parse_asog(asog, f"{output_dir.name} person"), 0.0, ASOG_CUT_FORMAT, PREVIEW_ROWS, fps=fps, recording=recording)
    recording.send_blueprint(person_blueprint(read_camera_rig(output_dir)))
    report(
        f"4D person: {steps:,} steps at {it_per_s:.0f} it/s, {rows:,} Gaussians -> ASOG {stats.count:,} rows, {len(asog) / 1e6:.1f} MB; "
        f"peak {peak / 1e9:.1f} GB; " + ", ".join(f"{name} {value:.0f} s" for name, value in seconds.items())
    )
    return PersonSummary(steps=steps, it_per_s=it_per_s, rows=rows, asog_rows=stats.count, asog_bytes=len(asog), peak_gpu_bytes=peak, seconds=seconds)
