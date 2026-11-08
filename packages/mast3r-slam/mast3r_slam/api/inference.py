"""MASt3R-SLAM inference pipeline.

This module implements the main tracking loop (frontend) and the backend
global optimisation process.  The two run concurrently:

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
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import lietorch_mojo as lietorch
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.multiprocessing as mp
from jaxtyping import Float, Float32, Int
from mast3r.model import AsymmetricMASt3R
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

from mast3r_slam.async_logger import (
    AsyncRerunLogger,
    compute_orient,
    snapshot_current_frame,
    snapshot_edges,
    snapshot_keyframe,
)
from mast3r_slam.backend_lifecycle import SlamBackend
from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import MonocularDataset, load_dataset
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates
from mast3r_slam.frame_prefetch import FramePrefetcher
from mast3r_slam.global_opt import FactorGraph
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
    load_retriever,
    mast3r_inference_mono,
    tracking_engine_size,
    trt_engines_enabled,
    use_pair_decoder_engine,
    use_tracking_engine,
)
from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio
from mast3r_slam.rerun_log_utils import (
    FRAME_TIMELINE,
    VIDEO_TIMELINE,
    create_blueprints,
    log_video_for_dataset,
)
from mast3r_slam.retrieval_database import RetrievalDatabase
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
PARENT_LOG_PATH: Path = Path("world")


@dataclass
class SlamSettings:
    """What to run and how; the CLI adds the Rerun target (:class:`InferenceConfig`), the app streams its own."""

    dataset: str = "data/normal-apt-tour.mp4"
    """Path to the input dataset or video file."""
    config: str = "config/base.yaml"
    """Path to the SLAM config YAML file."""
    img_size: Literal[224, 512] = 512
    """Target image size for MASt3R encoder."""
    subsample: int | None = None
    """Keep every n-th frame (None = the config's ``dataset.subsample``)."""
    max_frames: int | None = None
    """Stop after processing this many frames (None = process all)."""
    ns_save_path: None | Path = None
    """Optional path to export keyframes in NerfStudio format."""
    compile_model: bool = False
    """Run the tracker's MASt3R forward as compiled CUDA graphs: about 2x faster per frame, but the first tracked
    frame waits for the compile (about 30 s cold, 14 s with a warm inductor cache)."""
    trt_model: bool = True
    """Run tracking's MASt3R forward and the backend's pair decoder as bf16 TensorRT engines, one pair per frame size
    (512 keeps the video's aspect ratio, so each aspect ratio is its own size). The first run at a new size on a
    machine exports and builds its engines (about 3 min at 224, longer at 512); later runs load them from the cache.
    ``--no-trt-model`` runs eager fp32."""


@dataclass(kw_only=True)
class InferenceConfig(SlamSettings):
    """Configuration for a MASt3R-SLAM inference run from the command line."""

    rr_config: RerunTyroConfig
    """Rerun recording configuration (save path, application id, etc.)."""


CURRENT_FRAME_BACKLOG: int = 32
"""Logger events queued before the tracking thread stops enqueueing current frames (map events are always queued)."""


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
    """Load the SLAM config (the global ``config``) and the dataset, subsampled as the run will read it."""
    load_config(settings.config)
    if settings.subsample is not None:
        config["dataset"]["subsample"] = settings.subsample
    dataset: MonocularDataset = load_dataset(settings.dataset, img_size=settings.img_size)
    dataset.subsample(config["dataset"]["subsample"])
    return dataset


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

    1. Load SLAM config and dataset.
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
        Status message string after each processed frame.

    Raises:
        ValueError: If the config asks for calibration and the dataset has none.
    """
    # ── Load SLAM config and dataset ───────────────────────────────────────
    log_path: str = f"{PARENT_LOG_PATH}/logs"
    dataset: MonocularDataset = load_slam_dataset(settings)
    rr.log(log_path, rr.TextLog(f"Dataset: {settings.dataset}", level="INFO"))
    rr.log(log_path, rr.TextLog(f"Config: {config}", level="DEBUG"))
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
        event_queue, PARENT_LOG_PATH, recording=recording,
    )

    # ── Camera calibration (optional) ──────────────────────────────────────
    use_calib: bool = config["use_calib"]
    if use_calib and not dataset.has_calib():
        raise ValueError(f"{settings.config} sets use_calib, but {settings.dataset} has no calibration")
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
            settings.config, model, h, w, K, device=device, rr_application_id=rr_application_id, pair_decoder_hw=tracking_engine_size(model)
        ) as ctx,
        async_logger,
        FramePrefetcher(dataset, n_frames, device=device) as frames,
    ):
        assert ctx.keyframes is not None
        assert ctx.states is not None
        keyframes: SharedKeyframes = ctx.keyframes
        states: SharedStates = ctx.states

        tracker: FrameTracker = FrameTracker(model, keyframes, device)

        i: int = 0
        fps_timer: float = time.time()
        start_time: float = timer()
        last_orient_n_kf: int = 0
        prev_edge_count: int = 0
        # Dataset frame id of the keyframe whose image the 2D panel shows
        # (set by every LogMapUpdate with a new keyframe and by every
        # LogCurrentFrame that carried a keyframe image).
        shown_last_kf_id: int | None = None

        while True:
            # Check if the backend process crashed (raises BackendError if so).
            ctx.check_backend()

            mode: Mode = states.get_mode()

            # Stop condition: processed all requested frames.
            if i == n_frames:
                # A fast frontend outruns the backend: let it optimise the keyframes still queued, so the final
                # map is the same however fast tracking ran.
                backlog: int = len(states.global_optimizer_tasks)
                drain_start: float = time.perf_counter()
                while len(states.global_optimizer_tasks) > 0:
                    ctx.check_backend()
                    time.sleep(0.01)
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
                    X, C = mast3r_inference_mono(model, frame)
                    frame.update_pointmap(X, C)
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
                    X, C = mast3r_inference_mono(model, frame)
                    frame.update_pointmap(X, C)
                    states.set_frame(frame)
                    states.queue_reloc()
                    # In single-threaded mode, block until reloc completes.
                    while config["single_thread"]:
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
                wait_for_backend(states, ctx)

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
                event_queue.put(LogText(path=log_path, message=f"FPS: {FPS:.1f}"))
            i += 1

            yield f"Processing frame {i}/{n_frames}"

        # ── Post-loop: flush logger, log timing, export nerfstudio ────────
        # Nerfstudio export MUST happen before ctx.join() / __exit__ because
        # SharedKeyframes uses mp.Manager-backed values (e.g. n_size) that
        # become invalid after the manager shuts down.
        event_queue.put(LogText(path=log_path, message=f"Inference time: {format_time(timer() - start_time)}"))
        event_queue.put(LogText(path=log_path, message=f"Processed {len(keyframes)} keyframes"))

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

    rr.log(f"{PARENT_LOG_PATH}/logs", rr.TextLog("Done", level="INFO"))


def relocalization(
    frame: Frame,
    keyframes: SharedKeyframes,
    factor_graph: FactorGraph,
    retrieval_database: RetrievalDatabase,
) -> bool:
    """Attempt relocalization of a frame against the keyframe database.

    Called by the **backend process** when the frontend signals ``Mode.RELOC``.
    Queries the retrieval database for visually similar keyframes, attempts to
    build factor graph edges via symmetric MASt3R matching, and if successful,
    copies the pose from the matched keyframe and runs global optimisation.

    Args:
        frame: The current lost frame (with features from mono inference).
        keyframes: Shared keyframe buffer (locked during the entire operation
            to prevent the frontend from modifying keyframes concurrently).
        factor_graph: The global factor graph to add reloc edges to.
        retrieval_database: Image retrieval database for finding candidates.

    Returns:
        True if relocalization succeeded (edges added, pose copied), False otherwise.
    """
    with keyframes.lock:
        kf_idx: list[int] = []
        retrieval_inds: list[int] = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure: bool = False
        if kf_idx:
            keyframes.append(frame)
            n_kf: int = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx: list[int] = [n_kf - 1] * len(kf_idx)
            rr.log("/world/logs", rr.TextLog(f"Relocalizing against kf {n_kf - 1} and {kf_idx}", level="INFO"))
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                rr.log("/world/logs", rr.TextLog("Relocalized successfully", level="INFO"))
                successful_loop_closure = True
                # Copy the pose from the matched keyframe as initial estimate.
                keyframes.update_world_sim3_cams(
                    lietorch.Sim3(keyframes.world_sim3_cam[kf_idx[0]].clone()),
                    torch.tensor([n_kf - 1], device=keyframes.world_sim3_cam.device, dtype=torch.long),
                )
            else:
                keyframes.pop_last()
                rr.log("/world/logs", rr.TextLog("Failed to relocalize", level="WARN"))

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def wait_for_backend(states: SharedStates, ctx: SlamBackend) -> None:
    """Block while the backend has more keyframes queued than ``max_backend_backlog`` (0 when single-threaded).

    A fast frontend otherwise runs several keyframes ahead of the backend, most of all while
    the backend warms up, and global optimisation then starts from stale poses: some runs settled in a wrong
    minimum (final keyframe ATE 0.5-12 on a path of 12).
    """
    max_backlog: int = 0 if config["single_thread"] else int(config.get("max_backend_backlog", 2))
    while len(states.global_optimizer_tasks) > max_backlog:
        ctx.check_backend()
        time.sleep(0.002)


def run_backend(
    config_path: str,
    model: AsymmetricMASt3R,
    states: SharedStates,
    keyframes: SharedKeyframes,
    K: Float[Tensor, "3 3"] | None,
    rr_application_id: str | None = None,
    pair_decoder_hw: tuple[int, int] | None = None,
) -> None:
    """Backend process main loop: graph construction, retrieval, and global optimisation.

    This function runs in a **separate process** spawned by ``SlamBackend``.
    It continuously polls for new keyframes queued by the frontend, builds
    pairwise matching factors, queries the retrieval database for loop closures,
    and runs Gauss-Newton global optimisation to refine all keyframe poses.

    The loop exits when the frontend sets ``Mode.TERMINATED`` in shared state.

    Args:
        config_path: Path to the SLAM config YAML (reloaded in this process).
        model: Shared MASt3R model (parameters in shared GPU memory).
        states: Shared mutable state for mode, task queue, and current frame.
        keyframes: Shared keyframe buffer (poses, pointmaps, features).
        K: Camera intrinsic matrix, or None for uncalibrated mode.
        rr_application_id: When set, initialize Rerun in this subprocess and
            connect via gRPC so ``rr.TextLog`` entries reach the main viewer.
        pair_decoder_hw: When set, decode keyframe pairs with the TensorRT pair decoder for this image size
            (the frontend built it; see ``prepare_trt_engines``).
    """
    load_config(config_path)
    if pair_decoder_hw is not None:
        use_pair_decoder_engine(model, pair_decoder_hw)

    # Connect to the main process's Rerun viewer for TextLog output.
    if rr_application_id is not None:
        rr.init(rr_application_id, spawn=False)
        rr.connect_grpc()

    device: str = keyframes.device
    factor_graph: FactorGraph = FactorGraph(model, keyframes, K, device)
    retrieval_database: RetrievalDatabase = load_retriever(model)

    mode: Mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()

        # Wait for the frontend to finish initialisation.
        if mode == Mode.INIT:
            time.sleep(0.01)
            continue

        # Handle relocalization requests from the frontend.
        if mode == Mode.RELOC:
            frame: Frame = states.get_frame()
            success: bool = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue

        # Check if there's a new keyframe to process.
        idx: int = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # ── Graph construction for the new keyframe ────────────────────────
        # Connect to the previous consecutive keyframe(s) and to any
        # visually similar keyframes found via retrieval.
        kf_idx: list[int] = []
        n_consec: int = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)

        frame = keyframes[idx]
        retrieval_inds: list[int] = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        # Log loop closure candidates (excluding the consecutive neighbour).
        lc_inds: set[int] = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            rr.log("/world/logs", rr.TextLog(f"Database retrieval {idx}: {lc_inds}", level="INFO"))

        # Deduplicate and remove self-references.
        kf_idx_set: set[int] = set(kf_idx)
        kf_idx_set.discard(idx)
        kf_idx = list(kf_idx_set)
        frame_idx: list[int] = [idx] * len(kf_idx)

        # Run symmetric MASt3R matching to build factor graph edges.
        if kf_idx:
            factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])

        # Publish edge list to shared state (for Rerun visualisation).
        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        # ── Global Gauss-Newton optimisation ───────────────────────────────
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        # Mark this keyframe as processed.
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
