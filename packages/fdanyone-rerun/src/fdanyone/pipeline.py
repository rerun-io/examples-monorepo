"""Top-level inference orchestration."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event

from fdanyone.assets import (
    CHECKPOINT,
    HF_REPO_ID,
    HF_REVISION,
    TURBO_LORA,
    TURBO_LORA_NAME,
    TURBO_LORA_SHA256,
    resolve_base_assets,
    resolve_checkpoint,
    resolve_foreground_model,
    resolve_regressor,
    resolve_turbo_lora,
)
from fdanyone.config import BASE24, INFERENCE, RANK64_DELTA4
from fdanyone.device import CUDA_ALLOCATOR_CONF, WAIT_FOR_REQUEST, select_cuda_devices
from fdanyone.download import create_classic_gvhmr_links
from fdanyone.errors import ConfigurationError, RunCancelled
from fdanyone.io import remove_tree, resolve_output_path, write_json
from fdanyone.motion.gvhmr import validate_gvhmr
from fdanyone.motion.result import MotionResult
from fdanyone.output_directory import OutputDirectory
from fdanyone.run_request import save_run_request
from fdanyone.video import (
    decode_canonical_clip,
    validate_clip_options,
    validate_required_video_codecs,
    verify_lossless_video,
    write_gvhmr_video,
)
from fdanyone.views import ViewPlan, resolve_view_plan

LOGGER = logging.getLogger("fdanyone")
PROGRESS = logging.getLogger("fdanyone.progress")
PROGRESS.addHandler(logging.NullHandler())
PROGRESS.propagate = False


def _resolve_output_dir(output_dir: str | None, video_path: str) -> Path:
    """Resolve the output directory without following its final path component."""

    target = Path("data/fdanyone") / Path(video_path).stem if output_dir is None else Path(output_dir)
    return resolve_output_path(target)


def _discard_scratch(path: Path) -> None:
    """Best-effort cleanup that can never invalidate a published result.

    Some network filesystems keep an open, hidden tombstone after a file is
    unlinked.  Such a tombstone may remain ``EBUSY`` until this process exits,
    so cleanup must not be part of the atomic publication transaction.
    """

    try:
        remove_tree(path)
    except OSError as exc:
        LOGGER.warning(
            "Could not remove temporary files at %s (%s). "
            "The result is unaffected; the hidden scratch directory can be removed after this process exits.",
            path,
            exc,
        )


def _worker_environment() -> dict[str, str]:
    """Give the short-lived GVHMR workers this checkout and stable CUDA flags."""

    environment = os.environ.copy()
    environment.update(
        {
            "TORCH_CUDNN_V8_API_DISABLED": "1",
            "CUDNN_FRONTEND_DISABLE": "1",
            "CUDNN_LOGINFO_DBG": "0",
            "CUDNN_LOGDEST_DBG": "stderr",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    environment.pop("PYTHONHOME", None)
    # The 4 GiB split policy is specific to the long-lived DiT process. These
    # short-lived preprocessing workers use unrelated allocation shapes.
    environment.pop(CUDA_ALLOCATOR_CONF, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    return environment


def _inline_workers() -> bool:
    """Run the motion and skeleton workers in this process (``FDANYONE_INLINE_WORKERS=1``).

    For hosts where a grandchild process cannot use the GPU (a Hugging Face
    ZeroGPU worker). The subprocess default keeps each worker's CUDA flags and
    frees its memory with the process. Inline, GVHMR's motion recovery changes
    the process working directory (gvhmr_imports) while it runs, and Stop waits
    for an inline worker to finish.
    """

    return os.environ.get("FDANYONE_INLINE_WORKERS") == "1"


def _start_worker(module: str, worker_python: str, *preload: str) -> subprocess.Popen | None:
    """Start a worker ahead of its request so its interpreter and imports overlap this process's work (None inline).

    ``preload`` is passed to the worker's import step. The worker blocks on stdin until ``_run_worker`` sends the
    request; ``_discard_worker`` ends one that was never used.
    """

    if _inline_workers():
        return None
    return subprocess.Popen(
        [worker_python, "-m", module, WAIT_FOR_REQUEST, *preload],
        env=_worker_environment(),
        stdin=subprocess.PIPE,
        text=True,
    )


def _discard_worker(child: subprocess.Popen | None) -> None:
    """End a started worker that never received a request (closing stdin makes it exit); reap it either way."""

    if child is None or child.poll() is not None:
        return
    try:
        if child.stdin is not None and not child.stdin.closed:
            child.stdin.close()
        child.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        child.kill()
        child.wait()


def _run_worker(
    module: str,
    request_path: Path,
    worker_python: str,
    cancel: Event | None,
    started: subprocess.Popen | None = None,
    while_running: Callable[[], None] | None = None,
) -> None:
    """Run one worker entry point, as a subprocess or (inline) in this process.

    ``while_running`` is called once the worker has its request (inline: before it runs), in this thread.

    The subprocess is this call's own (``started``, from ``_start_worker``, or a new one): when ``cancel`` is set (or
    this thread raises), it gets SIGTERM, then SIGKILL after 10 s, and it is reaped before the error propagates
    (RunCancelled on cancel). It stays in this process group, so a Ctrl-C in the terminal reaches it too. An inline
    worker cannot be interrupted; it runs to its end.
    """

    if not _inline_workers():
        stop = Event() if cancel is None else cancel
        if started is None:
            command = [worker_python, "-m", module, str(request_path)]
            child = subprocess.Popen(command, env=_worker_environment())
        else:
            command, child = list(started.args), started
        try:
            if started is not None and child.stdin is not None:
                try:
                    child.stdin.write(f"{request_path}\n")
                    child.stdin.close()
                except BrokenPipeError:
                    pass  # the worker already exited: its return code below is the error
            if while_running is not None:
                while_running()
            while child.poll() is None:
                if stop.wait(0.25):
                    raise RunCancelled(f"stopped during {module}")
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
        return
    import gc
    import importlib

    import torch

    if while_running is not None:
        while_running()
    importlib.import_module(module).main(str(request_path))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_motion(
    *,
    working_video: Path,
    output_dir: Path,
    gvhmr_root: Path,
    device: str,
    worker_python: str,
    clip_metadata: Path,
    cancel: Event | None,
    started: subprocess.Popen | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / ".motion-worker-request.json"
    result_dir = output_dir / "result"
    write_json(
        request_path,
        {
            "gvhmr_root": str(gvhmr_root),
            "working_video": str(working_video),
            "clip_metadata": str(clip_metadata),
            "output_dir": str(output_dir / "runtime"),
            "result_dir": str(result_dir),
            "device": device,
        },
    )
    try:
        _run_worker("fdanyone.motion.worker", request_path, worker_python, cancel, started)
    finally:
        request_path.unlink(missing_ok=True)
    return MotionResult.load(result_dir)


def _build_conditioning(
    *,
    working_video: Path,
    clip_metadata: Path,
    motion_result_dir: Path,
    view_plan: ViewPlan,
    output_dir: Path,
    regressor: Path,
    foreground_model: Path,
    gvhmr_root: Path,
    device: str,
    worker_python: str,
    cancel: Event | None,
    started: subprocess.Popen | None = None,
    while_running: Callable[[], None] | None = None,
):
    from fdanyone.skeleton.pipeline import Conditioning

    request_path = output_dir.parent / ".skeleton-worker-request.json"
    write_json(
        request_path,
        {
            "working_video": str(working_video),
            "clip_metadata": str(clip_metadata),
            "motion_result_dir": str(motion_result_dir),
            "regressor_path": str(regressor),
            "foreground_model_path": str(foreground_model),
            "gvhmr_root": str(gvhmr_root),
            "output_dir": str(output_dir),
            "device": device,
            "view_plan": view_plan.to_dict(),
        },
    )
    try:
        _run_worker("fdanyone.skeleton.worker", request_path, worker_python, cancel, started, while_running)
    finally:
        request_path.unlink(missing_ok=True)
    return Conditioning.load(output_dir)


def run_pipeline(
    *,
    video_path: str,
    output_dir: str | None,
    model_dir: str,
    checkpoint_path: str | None,
    mhr70_regressor_path: str | None,
    gvhmr_root: str,
    gpu_ids: list[int] | None,
    attention_backend: str,
    start_time: float,
    target_fps: str | int | float,
    seed: int,
    views_per_layer: int,
    layer_pitches: list[int],
    start_yaw: int,
    yaw_span: int,
    enable_rcp: bool,
    enable_tcr: bool,
    enable_turbo: bool,
    cancel: Event | None = None,
) -> dict:
    """Execute inference for one clip, retaining reusable motion; setting ``cancel`` stops a running worker subprocess."""

    request_options = locals().copy()
    request_options.pop("cancel")
    pipeline_started = time.monotonic()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    PROGRESS.info("Checking input and settings", extra={"fraction": 0.02})
    if seed < 0:
        raise ConfigurationError(f"seed must be non-negative, got {seed}.")
    if not isinstance(enable_turbo, bool):
        raise ConfigurationError(f"enable_turbo must be True or False, got {enable_turbo!r}.")
    denoising_profile = RANK64_DELTA4 if enable_turbo else BASE24
    view_plan = resolve_view_plan(
        views_per_layer=views_per_layer,
        layer_pitches=layer_pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        enable_rcp=enable_rcp,
        enable_tcr=enable_tcr,
    )
    canonical_fps = validate_clip_options(
        start_time=start_time,
        fps=None if str(target_fps).lower() == "auto" else target_fps,
    )
    checkpoint = resolve_checkpoint(checkpoint_path, model_dir=model_dir) if checkpoint_path is not None else None
    destination = _resolve_output_dir(output_dir, video_path)
    clip_name = Path(video_path).stem
    output = OutputDirectory(destination)
    # Check before downloads and decoding, then again when locking the output.
    output.validate_available()
    validate_required_video_codecs()
    devices = select_cuda_devices(gpu_ids)
    device = devices[0]

    from fdanyone.vendor.diffsynth.models.wan_video_dit import get_attention_backend

    # Resolve once, before downloading assets or preparing conditioning. Every
    # DiT, including spawned replicas, receives this concrete backend.
    attention_backend = get_attention_backend(attention_backend)
    LOGGER.info("Using attention backend: %s", attention_backend)

    PROGRESS.info("Preparing model assets", extra={"fraction": 0.05})
    # The weights come from the pixi download tasks; GVHMR reads some of them, SMPL-X included, from its own tree.
    create_classic_gvhmr_links(model_dir, gvhmr_root)
    turbo_lora = resolve_turbo_lora(model_dir) if enable_turbo else None
    gvhmr_root, gvhmr_revision = validate_gvhmr(gvhmr_root)
    worker_python = os.path.abspath(sys.executable)

    if checkpoint is None:
        checkpoint = resolve_checkpoint(model_dir=model_dir)
    base_assets = resolve_base_assets(model_dir)
    regressor = resolve_regressor(mhr70_regressor_path, model_dir=model_dir)
    foreground_model = resolve_foreground_model(model_dir)
    PROGRESS.info("Preparing the 121-frame clip", extra={"fraction": 0.10})
    # Both workers start now: their interpreters and imports (~4 s each) overlap the clip preparation and GVHMR.
    motion_worker = (
        None if output.motion_dir.exists() else _start_worker("fdanyone.motion.worker", worker_python, str(gvhmr_root))
    )
    skeleton_worker = _start_worker("fdanyone.skeleton.worker", worker_python)
    prefetch = None
    scratch = None
    try:
        clip = decode_canonical_clip(
            video_path,
            num_frames=INFERENCE.num_frames,
            start_time=start_time,
            fps=canonical_fps,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f".{clip_name}.scratch-", dir=destination.parent))
        clip_metadata = scratch / "canonical_clip.json"
        clip.write_metadata(clip_metadata)
        working_video = write_gvhmr_video(clip, scratch / "canonical_clip.mp4")

        with output.stage() as work:
            # The canonical clip (frame k at k / fps from start_time) rides on the record: observers log it on the run clock.
            PROGRESS.info("Recovering human motion with GVHMR", extra={"fraction": 0.15, "canonical_clip": str(working_video)})
            if output.motion_dir.exists():
                LOGGER.info("Reusing GVHMR motion from %s", output.motion_dir)
                motion = MotionResult.load(output.motion_dir)
            else:
                save_run_request(destination, request_options)
                motion = _run_motion(
                    working_video=working_video,
                    output_dir=scratch / "gvhmr",
                    gvhmr_root=gvhmr_root,
                    device=device,
                    worker_python=worker_python,
                    clip_metadata=clip_metadata,
                    cancel=cancel,
                    started=motion_worker,
                )
            if motion.gvhmr_revision != gvhmr_revision:
                raise ConfigurationError(
                    f"GVHMR motion has revision {motion.gvhmr_revision}, expected {gvhmr_revision}. "
                    "Choose a new --output_dir to recover motion with the current GVHMR version."
                )
            motion.validate_against_clip(clip)
            if output.motion_dir.exists():
                save_run_request(destination, request_options)
            else:
                output.save_motion(motion)

            # Record the published identity only for the published checkpoint; an
            # explicit override must not claim the frozen Hugging Face coordinates.
            if checkpoint_path is None:
                model_identity = {"checkpoint": CHECKPOINT, "repo_id": HF_REPO_ID, "revision": HF_REVISION}
            else:
                model_identity = {"checkpoint": checkpoint.name, "source": "local_override"}
            if turbo_lora is not None:
                model_identity["turbo_lora"] = {
                    "name": TURBO_LORA_NAME,
                    "file": TURBO_LORA,
                    "sha256": TURBO_LORA_SHA256,
                }

            # Heavy rendering and generation are imported only after the motion
            # contract has been materialized, keeping CLI/help and CPU tests light.
            from fdanyone.model.inference import generate_views
            from fdanyone.output_writer import write_output

            if len(devices) == 1:
                # Conditioning runs in a worker process; load the DiT and VAE here meanwhile.
                from fdanyone.model.loader import GenerationPrefetch

                prefetch = GenerationPrefetch(
                    checkpoint_path=checkpoint,
                    turbo_lora_path=turbo_lora,
                    profile=denoising_profile,
                    attention_backend=attention_backend,
                    vae_path=base_assets.vae,
                )
            conditioning = _build_conditioning(
                working_video=working_video,
                clip_metadata=clip_metadata,
                motion_result_dir=output.motion_dir,
                view_plan=view_plan,
                output_dir=scratch / "conditioning",
                regressor=regressor,
                foreground_model=foreground_model,
                gvhmr_root=gvhmr_root,
                device=device,
                worker_python=worker_python,
                cancel=cancel,
                started=skeleton_worker,
                # The stage's observers (the body rebuild and its Rerun log, ~2 s) run while the worker works.
                while_running=lambda: PROGRESS.info("Building foreground masks and skeletons", extra={"fraction": 0.30}),
            )
            if conditioning.num_frames != len(clip.frames) or (
                conditioning.fps_num,
                conditioning.fps_den,
            ) != (
                clip.fps_num,
                clip.fps_den,
            ):
                raise ConfigurationError("Skeleton conditioning does not match the canonical clip timeline.")
            # Re-decode the worker-produced source before it becomes a model tensor.
            verify_lossless_video(clip, conditioning.source_video)
            # Generation reads the verified working video. Only lightweight
            # provenance is needed from the original full-resolution clip now.
            clip_info = clip.info
            del clip
            PROGRESS.info("Generating target-view videos", extra={"fraction": 0.45})
            generated = generate_views(
                conditioning=conditioning,
                checkpoint_path=checkpoint,
                turbo_lora_path=turbo_lora,
                denoising_profile=denoising_profile,
                attention_backend=attention_backend,
                assets=base_assets,
                output_dir=scratch / "generation",
                devices=devices,
                seed=seed,
                prefetch=prefetch,
            )
            PROGRESS.info("Saving and validating generated videos", extra={"fraction": 0.92})
            summary = write_output(
                clip=clip_info,
                conditioning=conditioning,
                generated=generated,
                destination=work,
                motion=motion,
                model_identity=model_identity,
                pipeline_started=pipeline_started,
            )
    finally:
        _discard_worker(motion_worker)
        _discard_worker(skeleton_worker)
        if prefetch is not None:
            prefetch.close()
        if scratch is not None:
            _discard_scratch(scratch)
    summary["output_dir"] = str(destination)
    PROGRESS.info("Inference complete", extra={"fraction": 1.0})
    return summary
