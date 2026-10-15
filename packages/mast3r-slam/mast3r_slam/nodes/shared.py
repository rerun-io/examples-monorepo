"""The graph's ``shared`` resource: keyframe buffer and current-frame state in shared memory, seen by both processes.

The frontend (main process) and the backend (its own process) exchange the map through these buffers: CUDA tensors cross
by CUDA IPC, the small fields live in an ``mp.Manager``. This is the feedback path of the SLAM loop (poses, mode,
relocalisation, the backend's acknowledgements), so the graph itself stays acyclic.
"""

import multiprocessing
from collections.abc import Callable
from multiprocessing.managers import SyncManager

import lietorch
import torch
import torch.multiprocessing  # noqa: F401  (registers the CUDA IPC reductions: shared tensors cross processes uncopied)
from jaxtyping import Float, Int
from torch import Tensor

from mast3r_slam.types import Frame, Mode


class SharedStates:
    """Shared mutable state between the tracker and backend processes.

    Holds the current frame data, mode, and synchronisation primitives in
    shared memory so they can be accessed across ``torch.multiprocessing``
    process boundaries.
    """

    def __init__(
        self,
        manager,
        h: int,
        w: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> None:
        self.lock = manager.RLock()
        self.mode = manager.Value("i", Mode.INIT)

        self.feat_dim: int = 1024
        self.num_patches: int = h * w // (16 * 16)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx: Int[Tensor, "1"] = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.rgb_tensor: Float[Tensor, "3 h w"] = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.rgb: Float[Tensor, "h w 3"] = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape: Int[Tensor, "1 2"] = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape: Int[Tensor, "1 2"] = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.world_sim3_cam: Float[Tensor, "1 8"] = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X: Float[Tensor, "hw 3"] = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C: Float[Tensor, "hw 1"] = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat: Float[Tensor, "1 n_patches feat_dim"] = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos: Int[Tensor, "1 n_patches 2"] = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # fmt: on

    def set_frame(self, frame: Frame) -> None:
        """Copy the frame's data into the shared tensors under the lock."""
        assert frame.X_canon is not None
        assert frame.C is not None
        assert frame.feat is not None
        assert frame.pos is not None
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.rgb_tensor[:] = frame.rgb_tensor
            self.rgb[:] = frame.rgb
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.world_sim3_cam[:] = frame.world_sim3_cam.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos

    def get_frame(self) -> Frame:
        """Read the current frame, retaining views of the shared tensors."""
        with self.lock:
            frame: Frame = Frame(
                int(self.dataset_idx[0]),
                self.rgb_tensor,
                self.img_shape,
                self.img_true_shape,
                self.rgb,
                lietorch.Sim3(self.world_sim3_cam),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            return frame

    def get_mode(self) -> Mode:
        """Return the current operating mode under the lock."""
        with self.lock:
            return self.mode.value

    def set_mode(self, mode: Mode) -> None:
        """Set the operating mode under the lock."""
        with self.lock:
            self.mode.value = mode


class SharedKeyframes:
    """Thread-safe shared-memory buffer of keyframes.

    Pre-allocates fixed-size tensors in shared memory for up to ``buffer``
    keyframes.  All reads and writes are protected by a reentrant lock so
    the tracker and backend processes can safely interleave access.
    """

    def __init__(
        self,
        manager,
        h: int,
        w: int,
        buffer: int = 512,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        use_calib: bool = False,
    ) -> None:
        self.use_calib: bool = use_calib
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.buffer: int = buffer
        self.device: str = device

        self.feat_dim: int = 1024
        self.num_patches: int = h * w // (16 * 16)

        # fmt:off
        self.dataset_idx: Int[Tensor, "buf"] = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.rgb_tensor: Float[Tensor, "buf 3 h w"] = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.rgb: Float[Tensor, "buf h w 3"] = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape: Int[Tensor, "buf 1 2"] = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape: Int[Tensor, "buf 1 2"] = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.world_sim3_cam: Float[Tensor, "buf 1 sim3_dim"] = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()
        self.X: Float[Tensor, "buf hw 3"] = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C: Float[Tensor, "buf hw 1"] = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.N: Int[Tensor, "buf"] = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates: Int[Tensor, "buf"] = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.feat: Float[Tensor, "buf 1 n_patches feat_dim"] = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos: Int[Tensor, "buf 1 n_patches 2"] = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        self.K: Float[Tensor, "3 3"] = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        # fmt: on

    def __getitem__(self, idx) -> Frame:
        """Read a keyframe by buffer index, retaining views of the shared tensors."""
        with self.lock:
            kf: Frame = Frame(
                int(self.dataset_idx[idx]),
                self.rgb_tensor[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.rgb[idx],
                lietorch.Sim3(self.world_sim3_cam[idx]),
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if self.use_calib:
                kf.K = self.K
            return kf

    def __setitem__(self, idx: int, value: Frame) -> int:
        """Write a frame into the shared buffer and return its index; refuse a full buffer."""
        assert value.X_canon is not None
        assert value.C is not None
        assert value.feat is not None
        assert value.pos is not None
        if idx >= self.buffer:
            raise RuntimeError(f"the keyframe buffer is full ({self.buffer} slots): raise SlamConfig.keyframe_buffer or shorten the input")
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            self.dataset_idx[idx] = value.frame_id
            self.rgb_tensor[idx] = value.rgb_tensor
            self.rgb[idx] = value.rgb
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.world_sim3_cam[idx] = value.world_sim3_cam.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            return idx

    def __len__(self) -> int:
        """Return the number of keyframes currently stored."""
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame) -> None:
        """Append a keyframe to the end of the buffer."""
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self) -> None:
        """Remove the last keyframe by decrementing the size counter."""
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Frame | None:
        """Return the most recently appended keyframe, or None if empty."""
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_world_sim3_cams(self, world_sim3_cams: lietorch.Sim3, idx: Int[Tensor, "n"]) -> None:
        """Overwrite the poses for a set of keyframes.

        Args:
            world_sim3_cams: New Sim3 poses to write.
            idx: Tensor of buffer indices corresponding to each pose.
        """
        with self.lock:
            self.world_sim3_cam[idx] = world_sim3_cams.data

    def set_intrinsics(self, K: Float[Tensor, "3 3"]) -> None:
        """Set the shared camera intrinsic matrix (requires ``use_calib`` config).

        Args:
            K: 3x3 intrinsic matrix.
        """
        assert self.use_calib
        with self.lock:
            self.K[:] = K


class SharedSlam:
    """The ``shared`` graph resource: keyframe buffer + current-frame state over one manager (``close`` shuts it down).

    The backend's acknowledgements live here too: the frontend numbers its tasks 1, 2, ... in hand-over order (the graph's
    ``task`` queue is FIFO), the backend counts the ones it applied in ``completed``, so task ``n`` is applied once
    ``completed >= n``; each count wakes a frontend waiting in ``wait``.
    """

    def __init__(self, h: int, w: int, buffer: int, use_calib: bool, device: str = "cuda") -> None:
        manager: SyncManager = multiprocessing.get_context("spawn").Manager()
        self.manager: SyncManager | None = manager
        self.keyframes: SharedKeyframes = SharedKeyframes(manager, h, w, buffer, device=device, use_calib=use_calib)
        self.states: SharedStates = SharedStates(manager, h, w, device=device)
        self.completed = manager.Value("i", 0)
        self.acknowledged = manager.Condition()

    def __getstate__(self) -> dict[str, object]:
        """The buffers and manager proxies cross to the backend process; the manager itself stays with its owner."""
        return self.__dict__ | {"manager": None}

    def reset(self) -> None:
        """A new run: no keyframes, mode INIT, no task applied."""
        with self.keyframes.lock, self.states.lock:
            self.keyframes.n_size.value = 0
            self.states.mode.value = Mode.INIT
        with self.acknowledged:
            self.completed.value = 0

    def publish(self) -> None:
        """Make this process's queued GPU writes to the buffers visible to the other process before telling it so: CUDA IPC
        does not order work across processes (without it the backend relocalised the previous current frame)."""
        if self.keyframes.device.startswith("cuda"):
            torch.cuda.synchronize(self.keyframes.device)

    def acknowledge(self) -> None:
        """The backend applied its oldest task: publish the task's GPU writes, then count it and wake the waiting frontend."""
        self.publish()
        with self.acknowledged:
            self.completed.value += 1
            self.acknowledged.notify_all()

    def wait(self, task: int, aborted: Callable[[], bool]) -> bool:
        """Block until task ``task`` is applied (True) or the run is ``aborted()`` (False; ``wake`` ends a wait in progress)."""
        with self.acknowledged:
            self.acknowledged.wait_for(lambda: aborted() or self.completed.value >= task)
            return self.completed.value >= task

    def wake(self) -> None:
        """Wake a waiting frontend so it checks ``aborted`` again (the run failed)."""
        with self.acknowledged:
            self.acknowledged.notify_all()

    def close(self) -> None:
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None
