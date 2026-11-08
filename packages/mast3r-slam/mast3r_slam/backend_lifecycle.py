"""The MASt3R-SLAM backend process: its lifecycle and its main loop.

``SlamBackend`` owns the backend ``mp.Process``, the ``mp.Manager`` server, and the shared
state objects (``SharedKeyframes``, ``SharedStates``).  Guarantees cleanup
on all exit paths — normal completion, exceptions, Ctrl+C, or Gradio stop.
``run_backend`` is the process's loop: graph construction, retrieval, global
optimisation and relocalization.

Usage::

    with SlamBackend(config_path, model, h, w, K) as ctx:
        keyframes = ctx.keyframes
        states = ctx.states
        tracker = FrameTracker(model, keyframes, device)
        for frame in frames:  # e.g. a FramePrefetcher
            ctx.check_backend()  # raises BackendError if backend died
            tracker.track(frame)
    # __exit__ handles: signal backend, join/terminate/kill, manager shutdown, GPU cleanup
"""

import gc
import multiprocessing
import time
import traceback
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue as MPQueue
from queue import Empty
from types import TracebackType

import lietorch_mojo as lietorch
import rerun as rr
import torch
import torch.multiprocessing as mp
from jaxtyping import Float
from mast3r.model import AsymmetricMASt3R
from torch import Tensor

from mast3r_slam.config import config, load_config
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import load_retriever, use_pair_decoder_engine
from mast3r_slam.rerun_log_utils import LOG_PATH
from mast3r_slam.retrieval_database import RetrievalDatabase

try:
    from beartype.roar import BeartypeException
except ImportError:
    BeartypeException = ()

class BackendError(RuntimeError):
    """Raised when the backend process dies with an exception."""


class SlamBackend:
    """Context manager that owns the backend process and manager lifecycle.

    Attributes:
        states: Shared mutable state between tracker and backend.
        keyframes: Shared keyframe buffer.
    """

    def __init__(
        self,
        config_path: str,
        model: AsymmetricMASt3R,
        h: int,
        w: int,
        K: Float[Tensor, "3 3"] | None,
        device: str = "cuda",
        rr_application_id: str | None = None,
        pair_decoder_hw: tuple[int, int] | None = None,
    ) -> None:
        self._config_path: str = config_path
        self._model: AsymmetricMASt3R = model
        self._h: int = h
        self._w: int = w
        self._K: Float[Tensor, "3 3"] | None = K
        self._device: str = device
        self._rr_application_id: str | None = rr_application_id
        """When set, the backend subprocess will call ``rr.init`` + ``rr.connect_grpc``
        so that ``rr.TextLog`` entries reach the same viewer as the main process."""
        self._pair_decoder_hw: tuple[int, int] | None = pair_decoder_hw
        """When set, the backend decodes keyframe pairs with the TensorRT pair decoder for this image size."""

        self._manager: SyncManager | None = None
        self._backend: mp.Process | None = None
        self._error_queue: MPQueue | None = None

        self.states: SharedStates | None = None
        self.keyframes: SharedKeyframes | None = None

    def __enter__(self) -> "SlamBackend":
        self._manager = mp.Manager()
        self._error_queue = MPQueue(ctx=multiprocessing.get_context())

        self.keyframes = SharedKeyframes(self._manager, self._h, self._w, device=self._device)
        self.states = SharedStates(self._manager, self._h, self._w, device=self._device)

        if self._K is not None and config["use_calib"]:
            self.keyframes.set_intrinsics(self._K)

        self._backend = mp.Process(
            target=_backend_entry,
            args=(
                self._config_path,
                self._model,
                self.states,
                self.keyframes,
                self._K,
                self._error_queue,
                self._rr_application_id,
                self._pair_decoder_hw,
            ),
            daemon=False,
        )
        self._backend.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self._shutdown_backend()
        self._shutdown_manager()
        self._cleanup_gpu()

    def _shutdown_backend(self) -> None:
        """Signal backend to stop, then escalate: join → terminate → kill."""
        if self._backend is None or not self._backend.is_alive():
            return

        # Step 1: Ask politely via shared mode flag
        if self.states is not None:
            try:
                self.states.set_mode(Mode.TERMINATED)
            except BeartypeException:
                raise
            except (OSError, BrokenPipeError, EOFError):
                pass  # Manager might already be dead

        # Step 2: Wait for graceful exit
        self._backend.join(timeout=5.0)

        # Step 3: Terminate
        if self._backend.is_alive():
            self._backend.terminate()
            self._backend.join(timeout=3.0)

        # Step 4: Kill (last resort)
        if self._backend.is_alive():
            self._backend.kill()
            self._backend.join(timeout=2.0)

    def _shutdown_manager(self) -> None:
        """Shut down the mp.Manager server process."""
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except BeartypeException:
                raise
            except (OSError, BrokenPipeError, EOFError):
                pass
            self._manager = None

    def _cleanup_gpu(self) -> None:
        """Release cached GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def check_backend(self) -> None:
        """Call periodically from the tracking loop.

        Raises:
            BackendError: If the backend process has died.
        """
        if self._backend is None:
            return
        if self._backend.is_alive():
            return

        # Backend died — try to read error details from queue (non-blocking).
        error_msg: str = "Backend process died unexpectedly"
        if self._error_queue is not None:
            try:
                exc_type_name: str
                tb_str: str
                exc_type_name, tb_str = self._error_queue.get_nowait()
                error_msg = f"Backend process crashed with {exc_type_name}:\n{tb_str}"
            except Empty:
                pass  # Backend died without writing to the queue

        raise BackendError(error_msg)

    def wait_for_backlog(self, max_backlog: int | None = None) -> None:
        """Block while the backend has more than ``max_backlog`` keyframes queued.

        ``None`` means the config's ``max_backend_backlog`` (0 when single-threaded). A fast frontend otherwise runs
        several keyframes ahead of the backend, most of all while the backend warms up, and global optimisation then
        starts from stale poses: some runs settled in a wrong minimum (final keyframe ATE 0.5-12 on a path of 12).

        Raises:
            BackendError: If the backend process dies while the frontend waits.
        """
        if max_backlog is None:
            max_backlog = 0 if config["single_thread"] else int(config.get("max_backend_backlog", 2))
        assert self.states is not None
        while len(self.states.global_optimizer_tasks) > max_backlog:
            self.check_backend()
            time.sleep(0.002)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the backend process to finish normally."""
        if self._backend is not None:
            self._backend.join(timeout=timeout)


def _backend_entry(
    config_path: str,
    model: AsymmetricMASt3R,
    states: SharedStates,
    keyframes: SharedKeyframes,
    K: Float[Tensor, "3 3"] | None,
    error_queue: MPQueue,
    rr_application_id: str | None = None,
    pair_decoder_hw: tuple[int, int] | None = None,
) -> None:
    """Entry point for the backend subprocess.

    Wraps ``run_backend`` with error reporting via the queue so the main
    process can detect and surface backend crashes.
    """
    try:
        run_backend(config_path, model, states, keyframes, K, rr_application_id=rr_application_id, pair_decoder_hw=pair_decoder_hw)
    except KeyboardInterrupt:
        pass  # Normal Ctrl+C propagation, not an error
    except BeartypeException:
        raise
    except Exception as exc:
        tb_str: str = traceback.format_exc()
        try:
            error_queue.put((type(exc).__name__, tb_str))
        except BeartypeException:
            raise
        except (OSError, BrokenPipeError, EOFError):
            pass
        print(f"[Backend FATAL] {tb_str}", flush=True)


def _retrieve(retrieval_database: RetrievalDatabase, frame: Frame, add_after_query: bool) -> list[int]:
    """Keyframes similar to ``frame`` (the config's ``retrieval`` k and threshold); adds ``frame`` when asked."""
    return retrieval_database.update(
        frame,
        add_after_query=add_after_query,
        k=config["retrieval"]["k"],
        min_thresh=config["retrieval"]["min_thresh"],
    )


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
        kf_idx: list[int] = _retrieve(retrieval_database, frame, add_after_query=False)
        if not kf_idx:
            return False
        keyframes.append(frame)
        n_kf: int = len(keyframes)
        frame_idx: list[int] = [n_kf - 1] * len(kf_idx)
        rr.log(LOG_PATH, rr.TextLog(f"Relocalizing against kf {n_kf - 1} and {kf_idx}", level="INFO"))
        if not factor_graph.add_factors(frame_idx, kf_idx, config["reloc"]["min_match_frac"], is_reloc=config["reloc"]["strict"]):
            keyframes.pop_last()
            rr.log(LOG_PATH, rr.TextLog("Failed to relocalize", level="WARN"))
            return False
        _retrieve(retrieval_database, frame, add_after_query=True)
        rr.log(LOG_PATH, rr.TextLog("Relocalized successfully", level="INFO"))
        # Copy the pose from the matched keyframe as initial estimate.
        keyframes.update_world_sim3_cams(
            lietorch.Sim3(keyframes.world_sim3_cam[kf_idx[0]].clone()),
            torch.tensor([n_kf - 1], device=keyframes.world_sim3_cam.device, dtype=torch.long),
        )
        factor_graph.solve()
        return True


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
        # Connect to the previous keyframe and to any visually similar keyframes found via retrieval.
        frame = keyframes[idx]
        retrieval_inds: list[int] = _retrieve(retrieval_database, frame, add_after_query=True)

        # Log loop closure candidates (excluding the consecutive neighbour).
        lc_inds: set[int] = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            rr.log(LOG_PATH, rr.TextLog(f"Database retrieval {idx}: {lc_inds}", level="INFO"))

        # Deduplicate and remove self-references.
        kf_idx_set: set[int] = set(retrieval_inds)
        if idx > 0:
            kf_idx_set.add(idx - 1)
        kf_idx_set.discard(idx)
        kf_idx: list[int] = list(kf_idx_set)
        frame_idx: list[int] = [idx] * len(kf_idx)

        # Run symmetric MASt3R matching to build factor graph edges.
        if kf_idx:
            factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])

        # Publish edge list to shared state (for Rerun visualisation).
        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        # ── Global Gauss-Newton optimisation ───────────────────────────────
        factor_graph.solve()

        # Mark this keyframe as processed.
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
