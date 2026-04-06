"""Context manager for the MASt3R-SLAM backend process lifecycle.

Owns the backend ``mp.Process``, the ``mp.Manager`` server, and the shared
state objects (``SharedKeyframes``, ``SharedStates``).  Guarantees cleanup
on all exit paths — normal completion, exceptions, Ctrl+C, or Gradio stop.

Usage::

    with SlamBackend(config_path, model, h, w, K) as ctx:
        keyframes = ctx.keyframes
        states = ctx.states
        tracker = FrameTracker(model, keyframes, device)
        for frame in dataset:
            ctx.check_backend()  # raises BackendError if backend died
            tracker.track(frame)
    # __exit__ handles: signal backend, join/terminate/kill, manager shutdown, GPU cleanup
"""

import gc
import multiprocessing
import traceback
from multiprocessing.managers import SyncManager
from queue import Empty
from types import TracebackType

import torch
import torch.multiprocessing as mp
from jaxtyping import Float
from torch import Tensor

from mast3r_slam.config import config
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates


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
        model: object,
        h: int,
        w: int,
        K: Float[Tensor, "3 3"] | None,
        device: str = "cuda",
    ) -> None:
        self._config_path: str = config_path
        self._model: object = model
        self._h: int = h
        self._w: int = w
        self._K: Float[Tensor, "3 3"] | None = K
        self._device: str = device

        self._manager: SyncManager | None = None
        self._backend: mp.Process | None = None
        self._error_queue: multiprocessing.Queue | None = None

        self.states: SharedStates | None = None
        self.keyframes: SharedKeyframes | None = None

    def __enter__(self) -> "SlamBackend":
        self._manager = mp.Manager()
        self._error_queue = multiprocessing.Queue()

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
            ),
            daemon=False,
        )
        self._backend.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
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
            except Exception:
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
            except Exception:
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

    @property
    def is_alive(self) -> bool:
        """Whether the backend process is still running."""
        return self._backend is not None and self._backend.is_alive()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the backend process to finish normally."""
        if self._backend is not None:
            self._backend.join(timeout=timeout)


def _backend_entry(
    config_path: str,
    model: object,
    states: SharedStates,
    keyframes: SharedKeyframes,
    K: Float[Tensor, "3 3"] | None,
    error_queue: multiprocessing.Queue,
) -> None:
    """Entry point for the backend subprocess.

    Wraps ``run_backend`` with error reporting via the queue so the main
    process can detect and surface backend crashes.
    """
    try:
        from mast3r_slam.api.inference import run_backend

        run_backend(config_path, model, states, keyframes, K)
    except KeyboardInterrupt:
        pass  # Normal Ctrl+C propagation, not an error
    except Exception as exc:
        tb_str: str = traceback.format_exc()
        try:
            error_queue.put((type(exc).__name__, tb_str))
        except Exception:
            pass
        print(f"[Backend FATAL] {tb_str}", flush=True)
