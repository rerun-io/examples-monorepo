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
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import lietorch
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.multiprocessing as mp
from jaxtyping import Float, Int
from mast3r.model import AsymmetricMASt3R
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

from mast3r_slam.backend_lifecycle import SlamBackend
from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import MonocularDataset, load_dataset
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio
from mast3r_slam.rerun_log_utils import (
    FRAME_TIMELINE,
    VIDEO_TIMELINE,
    RerunLogger,
    create_blueprints,
    log_video_for_dataset,
)
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.perf import BenchmarkRecorder, load_jsonl_rows, summarize_benchmark, timed_section, write_summary_markdown


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


@dataclass
class InferenceConfig:
    """Configuration for a MASt3R-SLAM inference run."""

    rr_config: RerunTyroConfig
    """Rerun recording configuration (save path, application id, etc.)."""
    dataset: str = "data/normal-apt-tour.mp4"
    """Path to the input dataset or video file."""
    config: str = "config/base.yaml"
    """Path to the SLAM config YAML file."""
    save_as: str = "default"
    """Subdirectory name for saving results under ``logs/``."""
    no_viz: bool = False
    """If True, skip launching visualisation processes."""
    disable_logging: bool = False
    """If True, skip Rerun video/frame logging to isolate inference throughput."""
    img_size: Literal[224, 512] = 512
    """Target image size for MASt3R encoder."""
    max_frames: int | None = None
    """Stop after processing this many frames (None = process all)."""
    ns_save_path: None | Path = None
    """Optional path to export keyframes in NerfStudio format."""
    benchmark_dir: Path | None = None
    """Optional output directory for per-stage benchmark traces and summaries."""


def mast3r_slam_inference(inf_config: InferenceConfig) -> None:
    """Run the full MASt3R-SLAM inference pipeline.

    This is the main entry point for CLI inference.  It:

    1. Loads the MASt3R model and dataset.
    2. Spawns a backend process inside a ``SlamBackend`` context manager.
    3. Runs the tracking loop frame-by-frame in the main process.
    4. Saves results (trajectory, reconstruction, keyframe images) on completion.

    The ``SlamBackend`` context manager ensures the backend process, manager
    server, and GPU memory are always cleaned up — even on exceptions or Ctrl+C.

    Args:
        inf_config: Inference configuration dataclass (typically from ``tyro.cli``).
    """
    # CUDA multiprocessing requires "spawn" start method (not "fork").
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn")

    # Enable TF32 for faster matmuls on Ampere+ GPUs; disable autograd
    # since the entire pipeline runs in inference mode.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device: str = "cuda:0"

    # ── Load SLAM config and dataset ───────────────────────────────────────
    parent_log_path: Path = Path("/world")
    log_path: str = f"{parent_log_path}/logs"
    load_config(inf_config.config)
    if inf_config.benchmark_dir is not None:
        config["benchmark_dir"] = str(inf_config.benchmark_dir)
    else:
        config.pop("benchmark_dir", None)
    frontend_benchmark: BenchmarkRecorder | None = None
    backend_benchmark_path: Path | None = None
    if inf_config.benchmark_dir is not None:
        inf_config.benchmark_dir.mkdir(parents=True, exist_ok=True)
        frontend_benchmark = BenchmarkRecorder(inf_config.benchmark_dir / "frontend.jsonl")
        backend_benchmark_path = inf_config.benchmark_dir / "backend.jsonl"
    logging_enabled: bool = not inf_config.disable_logging
    if logging_enabled:
        rr.log(log_path, rr.TextLog(f"Dataset: {inf_config.dataset}", level="INFO"))
        rr.log(log_path, rr.TextLog(f"Config: {config}", level="DEBUG"))

    dataset: MonocularDataset = load_dataset(inf_config.dataset, img_size=inf_config.img_size)
    dataset.subsample(config["dataset"]["subsample"])
    frame_timestamps_ns: Int[ndarray, "num_frames"] | None = None
    if logging_enabled:
        frame_timestamps_ns = log_video_for_dataset(
            dataset,
            parent_log_path / "current_camera" / "pinhole" / "video",
            timeline=VIDEO_TIMELINE,
        )

    # ── Rerun visualisation setup ──────────────────────────────────────────
    active_timeline: str = VIDEO_TIMELINE if frame_timestamps_ns is not None else FRAME_TIMELINE
    rr_logger: RerunLogger | None = None
    if logging_enabled:
        rr_logger = RerunLogger(parent_log_path, timeline=active_timeline)
        blueprint: rrb.Blueprint = create_blueprints(parent_log_path, timeline=active_timeline, n_keyframes=0)
        rr.send_blueprint(blueprint)

    h: int
    w: int
    h, w = dataset.get_img_shape()[0]

    # ── Load MASt3R model and share weights across processes ───────────────
    # share_memory() makes the model's parameters accessible to the backend
    # process without copying them (uses CUDA IPC for GPU tensors).
    model = load_mast3r(device=device)
    model.share_memory()

    # ── Camera calibration (optional) ──────────────────────────────────────
    has_calib: bool = dataset.has_calib()
    use_calib: bool = config["use_calib"]
    if use_calib and not has_calib:
        rr.log(log_path, rr.TextLog("No calibration provided for this dataset!", level="WARN"))
        sys.exit(0)
    K: Float[Tensor, "3 3"] | None = None
    if use_calib:
        assert dataset.camera_intrinsics is not None
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32)

    # ── Main tracking loop ─────────────────────────────────────────────────
    # SlamBackend.__enter__ creates the mp.Manager, shared keyframe/state
    # buffers, and spawns the backend process.  __exit__ guarantees cleanup
    # (backend shutdown, manager shutdown, GPU memory release).
    # Pass application_id to backend so it can connect to the same Rerun viewer via gRPC.
    # Only works when the main process uses spawn/serve/connect (not save-to-file).
    rr_app_id: str | None = None if inf_config.rr_config.save is not None else inf_config.rr_config.application_id
    with SlamBackend(
        inf_config.config,
        model,
        h,
        w,
        K,
        device=device,
        rr_application_id=rr_app_id,
        benchmark_dir=str(inf_config.benchmark_dir) if inf_config.benchmark_dir is not None else None,
    ) as ctx:
        assert ctx.keyframes is not None
        assert ctx.states is not None
        keyframes: SharedKeyframes = ctx.keyframes
        states: SharedStates = ctx.states
        tracker: FrameTracker = FrameTracker(model, keyframes, device)

        i: int = 0
        fps_timer: float = time.time()
        start_time: float = timer()

        while True:
            frame_profile: dict[str, float | int | str] = {}
            # Check if the backend process crashed (raises BackendError if so).
            ctx.check_backend()

            if logging_enabled:
                rr.set_time(FRAME_TIMELINE, sequence=i)
                if frame_timestamps_ns is not None and i < len(frame_timestamps_ns):
                    rr.set_time(VIDEO_TIMELINE, duration=1e-9 * float(frame_timestamps_ns[i]))
            mode: Mode = states.get_mode()

            # Stop condition: processed all requested frames.
            n_frames: int = len(dataset) if inf_config.max_frames is None else min(inf_config.max_frames, len(dataset))
            if i == n_frames:
                states.set_mode(Mode.TERMINATED)
                break

            with timed_section(frame_profile, "dataset_read_ms", sync_cuda=False):
                _, rgb = dataset[i]

            # Initialise pose: identity for the first frame, otherwise use the
            # last tracked pose from shared state.
            world_sim3_cam: lietorch.Sim3 = (
                lietorch.Sim3.Identity(1, device=device) if i == 0 else states.get_frame().world_sim3_cam
            )
            with timed_section(frame_profile, "create_frame_ms", sync_cuda=True):
                frame: Frame = create_frame(i, rgb, world_sim3_cam, img_size=dataset.img_size, device=device)

            add_new_kf: bool = False

            # INIT is handled separately because it uses `continue` to skip
            # the keyframe-selection and logging below.
            if mode == Mode.INIT:
                # Bootstrap: run MASt3R mono inference to get initial 3D points
                # and features.  The first frame is always a keyframe.
                X_init: Float[Tensor, "hw 3"]
                C_init: Float[Tensor, "hw 1"]
                init_profile: dict[str, float | int | str] = {}
                with timed_section(frame_profile, "init_total_ms", sync_cuda=True):
                    X_init, C_init = mast3r_inference_mono(model, frame, profile=init_profile)
                    frame.update_pointmap(X_init, C_init)
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes) - 1)
                    states.set_mode(Mode.TRACKING)
                    states.set_frame(frame)
                frame_profile.update({f"init_{k}": v for k, v in init_profile.items()})
                if logging_enabled:
                    assert rr_logger is not None
                    rr_logger.log_frame(frame, keyframes, states)
                    frame_profile.update(rr_logger.last_profile)
                else:
                    frame_profile["logging_total_ms"] = 0.0
                frame_profile["frame_idx"] = i
                frame_profile["mode"] = "INIT"
                frame_profile["n_keyframes"] = len(keyframes)
                frame_profile["go_queue_len"] = len(states.global_optimizer_tasks)
                frame_profile["frame_total_ms"] = (
                    float(frame_profile.get("dataset_read_ms", 0.0))
                    + float(frame_profile.get("create_frame_ms", 0.0))
                    + float(frame_profile.get("init_total_ms", 0.0))
                    + float(frame_profile.get("logging_total_ms", 0.0))
                )
                if frontend_benchmark is not None:
                    frontend_benchmark.append(frame_profile)
                i += 1
                continue

            match mode:
                case Mode.TRACKING:
                    # Normal tracking: match this frame against the last keyframe,
                    # estimate its relative pose via Gauss-Newton, and decide
                    # whether the overlap is low enough to warrant a new keyframe.
                    match_info: list
                    try_reloc: bool
                    with timed_section(frame_profile, "tracking_total_ms", sync_cuda=True):
                        add_new_kf, match_info, try_reloc = tracker.track(frame)
                    if try_reloc:
                        # Too few matches — tracking is lost, switch to reloc mode.
                        states.set_mode(Mode.RELOC)
                    states.set_frame(frame)
                    frame_profile.update(tracker.last_profile)

                case Mode.RELOC:
                    # Relocalization: run mono inference to get features, then the
                    # backend process will try to match against the retrieval DB.
                    X: Float[Tensor, "hw 3"]
                    C: Float[Tensor, "hw 1"]
                    reloc_profile: dict[str, float | int | str] = {}
                    with timed_section(frame_profile, "reloc_total_ms", sync_cuda=True):
                        X, C = mast3r_inference_mono(model, frame, profile=reloc_profile)
                        frame.update_pointmap(X, C)
                        states.set_frame(frame)
                        states.queue_reloc()
                    # In single-threaded mode, block until reloc completes.
                    while config["single_thread"]:
                        with states.lock:
                            if states.reloc_sem.value == 0:
                                break
                        time.sleep(0.01)
                    frame_profile.update({f"reloc_{k}": v for k, v in reloc_profile.items()})

                case _:
                    raise RuntimeError(f"Invalid mode: {mode!r}")

            # If the tracker decided this frame should be a new keyframe,
            # add it to the shared buffer and queue it for the backend to
            # build factor graph edges and run global optimisation.
            if add_new_kf:
                with timed_section(frame_profile, "queue_keyframe_ms", sync_cuda=False):
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes) - 1)
                # In single-threaded mode, block until the backend finishes.
                while config["single_thread"]:
                    with states.lock:
                        if len(states.global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)

            # Log current frame, all keyframes, camera path, and factor graph
            # edges to the Rerun viewer.
            if logging_enabled:
                assert rr_logger is not None
                rr_logger.log_frame(frame, keyframes, states)
                frame_profile.update(rr_logger.last_profile)
            else:
                frame_profile["logging_total_ms"] = 0.0
            if logging_enabled and i % 30 == 0:
                FPS: float = i / (time.time() - fps_timer)
                rr.log(log_path, rr.TextLog(f"FPS: {FPS:.1f}", level="INFO"))
            frame_profile["frame_idx"] = i
            frame_profile["mode"] = mode.name
            frame_profile["n_keyframes"] = len(keyframes)
            frame_profile["go_queue_len"] = len(states.global_optimizer_tasks)
            frame_profile["frame_total_ms"] = (
                float(frame_profile.get("dataset_read_ms", 0.0))
                + float(frame_profile.get("create_frame_ms", 0.0))
                + float(frame_profile.get("tracking_total_ms", 0.0))
                + float(frame_profile.get("reloc_total_ms", 0.0))
                + float(frame_profile.get("queue_keyframe_ms", 0.0))
                + float(frame_profile.get("logging_total_ms", 0.0))
            )
            if frontend_benchmark is not None:
                frontend_benchmark.append(frame_profile)
            i += 1

        if inf_config.ns_save_path is not None:
            pcd = save_kf_to_nerfstudio(
                ns_save_path=inf_config.ns_save_path,
                keyframes=keyframes,
            )
            if logging_enabled:
                rr.log(
                    f"{parent_log_path}/final_pointcloud",
                    rr.Points3D(positions=pcd.points, colors=pcd.colors),
                )

        if logging_enabled:
            rr.log(log_path, rr.TextLog("Done", level="INFO"))
            rr.log(log_path, rr.TextLog(f"Inference time: {format_time(timer() - start_time)}", level="INFO"))
            rr.log(log_path, rr.TextLog(f"Processed {len(keyframes)} keyframes", level="INFO"))

        # Wait for the backend to finish its last optimisation task before
        # the context manager shuts it down.
        ctx.join()

    if frontend_benchmark is not None and inf_config.benchmark_dir is not None:
        frontend_benchmark.flush()
        backend_rows = load_jsonl_rows(backend_benchmark_path) if backend_benchmark_path is not None else []
        summary = summarize_benchmark(frontend_benchmark._rows, backend_rows)
        write_summary_markdown(summary, inf_config.benchmark_dir / "summary.md")
        (inf_config.benchmark_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # SlamBackend.__exit__ has now run: backend terminated, manager shut down,
    # GPU memory released via torch.cuda.empty_cache() + gc.collect().
    if logging_enabled and not inf_config.no_viz:
        rr.log(log_path, rr.TextLog("All visualization processes terminated", level="INFO"))


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
                keyframes.world_sim3_cam[n_kf - 1] = keyframes.world_sim3_cam[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                rr.log("/world/logs", rr.TextLog("Failed to relocalize", level="WARN"))

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(
    config_path: str,
    model: AsymmetricMASt3R,
    states: SharedStates,
    keyframes: SharedKeyframes,
    K: Float[Tensor, "3 3"] | None,
    rr_application_id: str | None = None,
    benchmark_dir: str | None = None,
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
    """
    load_config(config_path)
    benchmark_recorder: BenchmarkRecorder | None = None
    if benchmark_dir:
        benchmark_recorder = BenchmarkRecorder(Path(benchmark_dir) / "backend.jsonl")

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
        if mode == Mode.INIT or states.is_paused():
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
        backend_profile: dict[str, float | int | str] = {"task_keyframe_idx": idx}

        # ── Graph construction for the new keyframe ────────────────────────
        # Connect to the previous consecutive keyframe(s) and to any
        # visually similar keyframes found via retrieval.
        kf_idx: list[int] = []
        n_consec: int = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)

        frame = keyframes[idx]
        with timed_section(backend_profile, "retrieval_total_ms", sync_cuda=True):
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
            with timed_section(backend_profile, "add_factors_total_ms", sync_cuda=True):
                factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])
            backend_profile.update(factor_graph.last_profile)

        # Publish edge list to shared state (for Rerun visualisation).
        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        # ── Global Gauss-Newton optimisation ───────────────────────────────
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()
        backend_profile.update(factor_graph.last_profile)
        backend_profile["retrieval_candidates"] = len(retrieval_inds)
        backend_profile["num_connected_keyframes"] = len(kf_idx)
        backend_profile["task_total_ms"] = (
            float(backend_profile.get("retrieval_total_ms", 0.0))
            + float(backend_profile.get("add_factors_total_ms", 0.0))
            + float(backend_profile.get("global_opt_total_ms", 0.0))
        )
        if benchmark_recorder is not None:
            benchmark_recorder.append(backend_profile)

        # Mark this keyframe as processed.
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
    if benchmark_recorder is not None:
        benchmark_recorder.flush()
