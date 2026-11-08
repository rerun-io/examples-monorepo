"""MASt3R-SLAM inference pipeline: settings, model loading, the tracking loop and the CLI.

This module implements the main tracking loop (frontend); the backend global
optimisation process lives in ``backend_lifecycle.py``.  The two run concurrently:

.. code-block:: text

    Main Process (Frontend / Tracker)          Backend Process (Global Optimizer)
    ┌──────────────────────────────┐           ┌──────────────────────────────┐
    │ for each frame:              │           │ while not terminated:        │
    │  1. Load image from dataset  │  shared   │  1. Poll for new keyframes   │
    │  2. Encode with MASt3R       │  state    │  2. Run image retrieval      │
    │  3. Match against keyframe   │◄────────►│  3. Build factor graph edges │
    │  4. Estimate pose (GN)       │  shared   │  4. Gauss-Newton global opt  │
    │  5. Decide if new keyframe   │  kfs      │  5. Update keyframe poses    │
    │  6. Log to Rerun             │           │  6. Handle relocalization    │
    └──────────────────────────────┘           └──────────────────────────────┘

The ``SlamBackend`` context manager (see ``backend_lifecycle.py``) owns the
backend process, the ``mp.Manager`` server, and all shared-memory buffers.
It guarantees cleanup on all exit paths (normal completion, exceptions,
Ctrl+C, or Gradio stop).

State machine modes (``Mode`` enum):
    - **INIT**: First frame — run monocular inference to bootstrap the map.
    - **TRACKING**: Normal operation — match each frame against the last
      keyframe, estimate its pose, decide whether to promote it to a new
      keyframe.
    - **RELOC**: Tracking lost — run monocular inference and ask the backend
      to relocalize against the retrieval database.
    - **TERMINATED**: Pipeline finished or interrupted — backend exits its loop.
"""

import contextlib
import queue
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import lietorch_mojo as lietorch
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.multiprocessing as mp
import tyro
from jaxtyping import Float, Float32, Int
from mast3r.model import AsymmetricMASt3R
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

from mast3r_slam.async_logger import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_PANEL_HZ,
    AsyncRerunLogger,
    compute_orient,
    snapshot_current_frame,
    snapshot_edges,
    snapshot_keyframe,
)
from mast3r_slam.backend_lifecycle import SlamBackend
from mast3r_slam.config import SlamConfig, SlamPreset, flat_config
from mast3r_slam.dataloader import MonocularDataset, load_dataset
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates
from mast3r_slam.frame_prefetch import FramePrefetcher
from mast3r_slam.log_events import (
    KeyframeSnapshot,
    LogCurrentFrame,
    LogEvent,
    LogMapUpdate,
    LogText,
)
from mast3r_slam.mast3r_utils import (
    compile_tracking_forward,
    enable_trt_engines,
    load_mast3r,
    mast3r_inference_mono,
    tracking_engine_size,
    trt_engines_enabled,
    use_tracking_engine,
)
from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio
from mast3r_slam.rerun_log_utils import (
    FRAME_TIMELINE,
    LOG_PATH,
    PARENT_LOG_PATH,
    VIDEO_TIMELINE,
    create_blueprints,
    log_video_for_dataset,
)
from mast3r_slam.tracker import FrameTracker


def format_time(seconds: float) -> str:
    """Format a duration in seconds as mm:ss.

    Args:
        seconds: Number of seconds.

    Returns:
        A string in ``"mm:ss"`` format.
    """
    minutes: int = int(seconds // 60)
    seconds_rem: int = int(seconds % 60)
    return f"{minutes:02d}:{seconds_rem:02d}"


DEVICE: str = "cuda:0"
APPLICATION_ID: str = "mast3r-slam"
"""Recording id of the Gradio app (the CLI's comes from ``--rr-config.application-id``)."""


@dataclass(frozen=True)
class SlamSettings:
    """What to run and how; the CLI adds the Rerun target (:class:`InferenceConfig`), the app streams its own."""

    dataset: str = "data/normal-apt-tour.mp4"
    """Path to the input dataset or video file."""
    slam: SlamPreset = field(default_factory=SlamConfig.base)
    """SLAM hyper-parameters: a preset (the subcommand), any field overridden with ``--slam.<group>.<field>``."""
    img_size: Literal[224, 512] = 512
    """Target image size for MASt3R encoder."""
    max_frames: int | None = None
    """Stop after processing this many frames (None = process all)."""
    ns_save_path: None | Path = None
    """Optional path to export keyframes in NerfStudio format."""
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    """Smallest MASt3R confidence (1 = none) a pixel needs to show in the depth panels and to fuse into the map
    surface. Lower it when a hard clip (water, sky, low texture) shows black depth."""
    live_panel_hz: float = DEFAULT_PANEL_HZ
    """Most current-frame panels (image, depth, pointmap, confidence) logged per second; 0 logs every frame. The
    camera pose, the path and the map are logged for every frame."""
    compile_model: bool = False
    """Run the tracker's MASt3R forward as compiled CUDA graphs: about 2x faster per frame, but the first tracked
    frame waits for the compile (about 30 s cold, 14 s with a warm inductor cache)."""
    trt_model: bool = True
    """Run tracking's MASt3R forward and the backend's pair decoder as bf16 TensorRT engines, one pair per frame size
    (512 keeps the video's aspect ratio, so each aspect ratio is its own size). The first run at a new size on a
    machine exports and builds its engines (about 3 min at 224, longer at 512); later runs load them from the cache.
    ``--no-trt-model`` runs eager fp32."""


@tyro.conf.configure(tyro.conf.CascadeSubcommandArgs)
@dataclass(frozen=True, kw_only=True)
class InferenceConfig(SlamSettings):
    """Configuration for a MASt3R-SLAM inference run from the command line.

    The first argument names a preset (``base`` when omitted); ``--slam.<group>.<field>`` overrides any of its fields,
    and the other flags may come before or after the preset name.
    """

    rr_config: RerunTyroConfig
    """Rerun recording configuration (save path, application id, etc.)."""


CURRENT_FRAME_BACKLOG: int = 32
"""Logger events queued before the tracking thread stops enqueueing current frames (map events are always queued)."""
STATUS_INTERVAL_S: float = 0.1
"""Shortest wall-clock time between two status yields of :func:`run_slam_pipeline`. Tracking runs inside the generator,
so the caller's work per yield stalls it: the Gradio app streams the recording to its viewer on every yield, which
cost about 5 ms per frame when every frame was yielded."""


def load_slam_model(settings: SlamSettings, device: str = DEVICE) -> AsymmetricMASt3R:
    """Set up the process and load MASt3R for :func:`run_slam_pipeline`, with the forward ``settings`` asks for.

    Uses the ``spawn`` start method (the backend process shares CUDA tensors), TF32 matmuls, and shares the model's
    weights so the backend process reads them without a copy (CUDA IPC).
    """
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    model: AsymmetricMASt3R = load_mast3r(device=device)
    model.share_memory()
    if settings.compile_model:
        compile_tracking_forward(model)
    elif settings.trt_model:
        enable_trt_engines(model)
    return model


def load_slam_dataset(settings: SlamSettings) -> MonocularDataset:
    """Load the dataset, subsampled as the run will read it."""
    return load_dataset(settings.dataset, settings.img_size, settings.slam)


@torch.no_grad()
def run_slam_pipeline(
    settings: SlamSettings,
    model: AsymmetricMASt3R,
    *,
    device: str = DEVICE,
    rr_application_id: str | None = None,
    recording: rr.RecordingStream | None = None,
) -> Generator[str, None, None]:
    """Framework-agnostic SLAM tracking loop, shared by the CLI and the Gradio app.

    Logging is performed asynchronously by an ``AsyncRerunLogger`` thread.
    The pipeline thread creates lightweight CPU snapshots (events) and
    enqueues them; the logger thread does all the expensive work (JPEG
    compression, focal estimation, ``rr.log()`` calls).

    The generator manages the full lifecycle:

    1. Load the dataset; record the SLAM config.
    2. Set up Rerun (video log, blueprint, async logger).
    3. Check calibration.
    4. Spawn and manage backend via ``SlamBackend`` context manager.
    5. Run frame-by-frame tracking loop.
    6. Log FPS metrics and timing.
    7. Export keyframes to NerfStudio format (if ``settings.ns_save_path`` is set).
    8. Wait for backend to finish (``ctx.join()``).

    Nerfstudio export happens inside the generator (before
    ``SlamBackend.__exit__``) because ``SharedKeyframes`` depends on the
    ``mp.Manager`` which is shut down on exit. Closing the generator early
    (Ctrl+C, the app's Stop button) runs the same cleanup.

    Args:
        settings: What to run.
        model: Model from :func:`load_slam_model`.
        device: Torch device string.
        rr_application_id: When set, backend subprocess connects to Rerun via gRPC.
        recording: Rerun recording to pass to the async logger thread
            (Gradio).  ``None`` uses the global recording (CLI).

    Yields:
        Status message string, at most every ``STATUS_INTERVAL_S`` and after the last frame.

    Raises:
        ValueError: If the config asks for calibration and the dataset has none.
    """
    # ── Load dataset, record the SLAM config ──────────────────────────────
    cfg: SlamConfig = settings.slam
    dataset: MonocularDataset = load_slam_dataset(settings)
    rr.log(LOG_PATH, rr.TextLog(f"Dataset: {settings.dataset}", level="INFO"))
    # The resolved config, one component per field (``tracking.max_iters``): the recording says how it was made.
    config_values: rr.AnyValues = rr.AnyValues()
    for key, value in flat_config(cfg).items():
        config_values.with_component_from_data(key, value)
    rr.send_property("slam_config", config_values)
    frame_timestamps_ns: Int[ndarray, "num_frames"] | None = log_video_for_dataset(dataset)

    # ── Rerun visualisation setup ──────────────────────────────────────────
    active_timeline: str = VIDEO_TIMELINE if frame_timestamps_ns is not None else FRAME_TIMELINE
    h: int
    w: int
    h, w = dataset.get_img_shape()[0]
    if trt_engines_enabled(model):
        # Before any pipeline thread starts: loading captures the engine's CUDA graph.
        use_tracking_engine(model, (h, w))
    blueprint: rrb.Blueprint = create_blueprints(PARENT_LOG_PATH, timeline=active_timeline)
    rr.send_blueprint(blueprint)

    # Async logger: events are enqueued here, processed in a background thread.
    # Used as a context manager further down so LogTerminate + join is automatic.
    # Unbounded, so the tracking thread never blocks on the logger: map events (keyframes, poses, edges) are never
    # dropped, and current-frame events are only enqueued while the backlog is short (CURRENT_FRAME_BACKLOG).
    event_queue: queue.Queue[LogEvent] = queue.Queue()
    async_logger: AsyncRerunLogger = AsyncRerunLogger(
        event_queue, PARENT_LOG_PATH, conf_threshold=settings.conf_threshold, panel_hz=settings.live_panel_hz, recording=recording,
    )

    # ── Camera calibration (optional) ──────────────────────────────────────
    use_calib: bool = cfg.use_calib
    if use_calib and not dataset.has_calib():
        raise ValueError(f"The SLAM config sets use_calib, but {settings.dataset} has no calibration")
    K: Float[Tensor, "3 3"] | None = None
    if use_calib:
        assert dataset.camera_intrinsics is not None
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32)

    # ── Main tracking loop ─────────────────────────────────────────────────
    # SlamBackend.__enter__ creates the mp.Manager, shared keyframe/state
    # buffers, and spawns the backend process.  __exit__ guarantees cleanup
    # (backend shutdown, manager shutdown, GPU memory release).
    # AsyncRerunLogger.__exit__ sends LogTerminate and joins the thread.
    n_frames: int = len(dataset) if settings.max_frames is None else min(settings.max_frames, len(dataset))
    with (
        SlamBackend(
            cfg, model, h, w, K, device=device, rr_application_id=rr_application_id, pair_decoder_hw=tracking_engine_size(model)
        ) as ctx,
        async_logger,
        FramePrefetcher(dataset, n_frames, cfg.dataset.img_downsample, device=device) as frames,
    ):
        assert ctx.keyframes is not None
        assert ctx.states is not None
        keyframes: SharedKeyframes = ctx.keyframes
        states: SharedStates = ctx.states

        tracker: FrameTracker = FrameTracker(model, keyframes, cfg, device)

        i: int = 0
        fps_timer: float = time.time()
        last_status: float = 0.0
        start_time: float = timer()
        last_orient_n_kf: int = 0
        prev_edge_count: int = 0
        # Dataset frame id of the keyframe whose image the 2D panel shows
        # (set by every LogMapUpdate with a new keyframe and by every
        # LogCurrentFrame that carried a keyframe image).
        shown_last_kf_id: int | None = None

        while True:
            # Re-raise a crash of the backend process (BackendError) or of the logger thread.
            ctx.check_backend()
            async_logger.check()

            mode: Mode = states.get_mode()

            # Stop condition: processed all requested frames.
            if i == n_frames:
                # A fast frontend outruns the backend: let it optimise the keyframes still queued, so the final
                # map is the same however fast tracking ran.
                backlog: int = len(states.global_optimizer_tasks)
                drain_start: float = time.perf_counter()
                ctx.wait_for_backlog(0)
                if backlog > 0:
                    print(f"Backend finished {backlog} queued keyframes in {time.perf_counter() - drain_start:.2f} s")
                states.set_mode(Mode.TERMINATED)
                break

            # Resolve timestamp for this frame.
            ts_ns: int | None = (
                int(frame_timestamps_ns[i]) if frame_timestamps_ns is not None and i < len(frame_timestamps_ns) else None
            )

            # Initialise pose: identity for the first frame, otherwise use the
            # last tracked pose from shared state.
            world_sim3_cam: lietorch.Sim3 = (
                lietorch.Sim3.Identity(1, device=device) if i == 0 else states.get_frame().world_sim3_cam
            )
            frame: Frame = frames.get()
            frame.world_sim3_cam = world_sim3_cam

            add_new_kf: bool = False

            X: Float[Tensor, "hw 3"]
            C: Float[Tensor, "hw 1"]
            match mode:
                case Mode.INIT:
                    # Bootstrap: mono inference gives the first frame its pointmap and features; it is the first
                    # keyframe. TRACKING is set before the keyframe is queued: the backend skips its queue in INIT.
                    X, C = mast3r_inference_mono(model, frame, cfg.dataset.img_downsample)
                    frame.update_pointmap(X, C, cfg.tracking)
                    states.set_mode(Mode.TRACKING)
                    states.set_frame(frame)
                    add_new_kf = True

                case Mode.TRACKING:
                    # Normal tracking: match this frame against the last keyframe,
                    # estimate its relative pose via Gauss-Newton, and decide
                    # whether the overlap is low enough to warrant a new keyframe.
                    try_reloc: bool
                    add_new_kf, try_reloc = tracker.track(frame)
                    if try_reloc:
                        # Too few matches — tracking is lost, switch to reloc mode.
                        states.set_mode(Mode.RELOC)
                    states.set_frame(frame)

                case Mode.RELOC:
                    # Relocalization: run mono inference to get features, then the
                    # backend process will try to match against the retrieval DB.
                    X, C = mast3r_inference_mono(model, frame, cfg.dataset.img_downsample)
                    frame.update_pointmap(X, C, cfg.tracking)
                    states.set_frame(frame)
                    states.queue_reloc()
                    # In single-threaded mode, block until reloc completes.
                    while cfg.single_thread:
                        with states.lock:
                            if states.reloc_sem.value == 0:
                                break
                        time.sleep(0.01)

                case _:
                    raise RuntimeError(f"Invalid mode: {mode!r}")

            # If the tracker decided this frame should be a new keyframe,
            # add it to the shared buffer and queue it for the backend to
            # build factor graph edges and run global optimisation.
            if add_new_kf:
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                ctx.wait_for_backlog()

            # ── Async logging: create events and enqueue ──────────────────
            # 1. Collect structural changes (non-droppable)
            new_kfs: list[KeyframeSnapshot] = []
            pose_updates: list[tuple[int, Float32[ndarray, "8"]]] = []
            edge_pos: tuple[Float32[ndarray, "n 3"], Float32[ndarray, "n 3"]] | None = None
            orient_data: tuple[Float32[ndarray, "3 3"], Float32[ndarray, "3"]] | None = None

            new_kf_idx: int | None = None
            if add_new_kf:
                new_kf_idx = len(keyframes) - 1
                new_kfs.append(snapshot_keyframe(frame, new_kf_idx))

            dirty_idx: Int[Tensor, "n_dirty"] = keyframes.get_dirty_idx()
            if dirty_idx.numel() > 0:
                for idx_val in dirty_idx.tolist():
                    kf_idx_dirty: int = int(idx_val)
                    if new_kf_idx is not None and kf_idx_dirty == new_kf_idx:
                        continue
                    sim3_cpu: Float32[ndarray, "8"] = (
                        keyframes.world_sim3_cam[kf_idx_dirty, 0].cpu().numpy().astype(np.float32)
                    )
                    pose_updates.append((kf_idx_dirty, sim3_cpu))

            # Edge update: resnapshot when edge count changes OR when poses
            # were refined (global optimization moves endpoints without
            # adding/removing factors).
            with states.lock:
                current_edge_count: int = len(states.edges_ii)
            if current_edge_count != prev_edge_count or pose_updates:
                edge_pos = snapshot_edges(states, keyframes)
                prev_edge_count = current_edge_count

            # Orient update (when keyframe count changes)
            with keyframes.lock:
                n_kf: int = len(keyframes)
            if n_kf > 0 and n_kf != last_orient_n_kf:
                orient_data = compute_orient(keyframes, n_kf)
                last_orient_n_kf = n_kf

            if new_kfs or pose_updates or edge_pos is not None or orient_data is not None:
                event_queue.put(LogMapUpdate(
                    frame_idx=i, timestamp_ns=ts_ns,
                    new_keyframes=new_kfs, pose_updates=pose_updates,
                    edge_positions=edge_pos, orient=orient_data,
                ))
            if add_new_kf:
                shown_last_kf_id = frame.frame_id

            # 2. Current frame (droppable) — skipped, snapshot included, while the logger is behind
            if event_queue.qsize() < CURRENT_FRAME_BACKLOG:
                current_event: LogCurrentFrame = snapshot_current_frame(frame, i, ts_ns, keyframes, shown_last_kf_id)
                event_queue.put_nowait(current_event)
                shown_last_kf_id = current_event.last_kf_frame_id

            # FPS text log
            if i % 30 == 0 and i > 0:
                FPS: float = i / (time.time() - fps_timer)
                event_queue.put(LogText(path=LOG_PATH, message=f"FPS: {FPS:.1f}"))
            i += 1

            if time.monotonic() - last_status >= STATUS_INTERVAL_S or i == n_frames:
                last_status = time.monotonic()
                yield f"Processing frame {i}/{n_frames}"

        # ── Post-loop: flush logger, log timing, export nerfstudio ────────
        # Nerfstudio export MUST happen before ctx.join() / __exit__ because
        # SharedKeyframes uses mp.Manager-backed values (e.g. n_size) that
        # become invalid after the manager shuts down.
        event_queue.put(LogText(path=LOG_PATH, message=f"Inference time: {format_time(timer() - start_time)}"))
        event_queue.put(LogText(path=LOG_PATH, message=f"Processed {len(keyframes)} keyframes"))

        if settings.ns_save_path is not None:
            pcd = save_kf_to_nerfstudio(
                ns_save_path=settings.ns_save_path,
                keyframes=keyframes,
            )
            rr.log(
                f"{PARENT_LOG_PATH}/final_pointcloud",
                rr.Points3D(positions=pcd.points, colors=pcd.colors),
            )

        ctx.join()

    # async_logger.__exit__ has now run: LogTerminate sent, thread joined.

    # SlamBackend.__exit__ has now run: backend terminated, manager shut down,
    # GPU memory released via torch.cuda.empty_cache() + gc.collect().


def mast3r_slam_inference(inf_config: InferenceConfig) -> None:
    """Run the full MASt3R-SLAM inference pipeline from the command line.

    Loads the model with :func:`load_slam_model` and runs :func:`run_slam_pipeline` to the end; the Gradio app
    calls the same two functions.

    Args:
        inf_config: Inference configuration dataclass (typically from ``tyro.cli``).
    """
    model: AsymmetricMASt3R = load_slam_model(inf_config)

    # Pass application_id to backend so it can connect to the same Rerun viewer via gRPC.
    # Only works when the main process uses spawn/serve/connect (not save-to-file).
    rr_app_id: str | None = None if inf_config.rr_config.save is not None else inf_config.rr_config.application_id
    for _msg in run_slam_pipeline(inf_config, model, rr_application_id=rr_app_id):
        pass

    rr.log(LOG_PATH, rr.TextLog("Done", level="INFO"))
