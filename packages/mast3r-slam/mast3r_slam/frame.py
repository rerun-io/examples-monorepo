import dataclasses
from enum import Enum
from typing import Literal

import lietorch
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from mast3r_slam.config import config
from mast3r_slam.image_types import RgbNormalized
from mast3r_slam.image_utils import ResizedImage, resize_img


class Mode(Enum):
    """Operating mode for the SLAM system."""

    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


class FilteringMode(Enum):
    """Strategy for updating a frame's canonical 3D point map.

    When a new observation arrives (from matching or mono inference), this
    controls how the existing point map is combined with the new one.
    """

    FIRST = "first"
    """Keep only the second observation (first update after init), ignore all later ones."""
    RECENT = "recent"
    """Always replace the point map with the most recent observation."""
    BEST_SCORE = "best_score"
    """Keep the observation whose scalar confidence score (median or mean) is highest."""
    INDEP_CONF = "indep_conf"
    """Per-point update: for each pixel, keep whichever observation has higher confidence."""
    WEIGHTED_POINTMAP = "weighted_pointmap"
    """Confidence-weighted running average of 3D points in Cartesian space."""
    WEIGHTED_SPHERICAL = "weighted_spherical"
    """Confidence-weighted running average in spherical coordinates (r, phi, theta)."""


class FilteringScore(Enum):
    """Scalar score function used by ``FilteringMode.BEST_SCORE``."""

    MEDIAN = "median"
    """Median of per-point confidence values."""
    MEAN = "mean"
    """Mean of per-point confidence values."""


def _cartesian_to_spherical(P: Float[Tensor, "hw 3"]) -> Float[Tensor, "hw 3"]:
    """Convert Cartesian (x, y, z) to spherical (r, phi, theta) coordinates.

    Args:
        P: Points in Cartesian coordinates, shape (hw, 3).

    Returns:
        Points in spherical coordinates (r, phi, theta), shape (hw, 3).
    """
    r: Float[Tensor, "hw 1"] = torch.linalg.norm(P, dim=-1, keepdim=True)
    x: Float[Tensor, "hw 1"]
    y: Float[Tensor, "hw 1"]
    z: Float[Tensor, "hw 1"]
    x, y, z = torch.tensor_split(P, 3, dim=-1)
    phi: Float[Tensor, "hw 1"] = torch.atan2(y, x)
    theta: Float[Tensor, "hw 1"] = torch.acos(z / r)
    spherical: Float[Tensor, "hw 3"] = torch.cat((r, phi, theta), dim=-1)
    return spherical


def _spherical_to_cartesian(spherical: Float[Tensor, "hw 3"]) -> Float[Tensor, "hw 3"]:
    """Convert spherical (r, phi, theta) to Cartesian (x, y, z) coordinates.

    Args:
        spherical: Points in spherical coordinates (r, phi, theta), shape (hw, 3).

    Returns:
        Points in Cartesian coordinates, shape (hw, 3).
    """
    r: Float[Tensor, "hw 1"]
    phi: Float[Tensor, "hw 1"]
    theta: Float[Tensor, "hw 1"]
    r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
    x: Float[Tensor, "hw 1"] = r * torch.sin(theta) * torch.cos(phi)
    y: Float[Tensor, "hw 1"] = r * torch.sin(theta) * torch.sin(phi)
    z: Float[Tensor, "hw 1"] = r * torch.cos(theta)
    P: Float[Tensor, "hw 3"] = torch.cat((x, y, z), dim=-1)
    return P


@dataclasses.dataclass
class Frame:
    """A single image frame with associated 3D data and pose.

    Stores the RGB image, its shape metadata, the canonical 3D point map,
    confidence values, MASt3R features, and the world-to-camera Sim3 pose.
    """

    frame_id: int
    """Index of this frame in the dataset sequence."""
    rgb_tensor: Float[Tensor, "*batch 3 h w"]
    """ImageNet-normalized RGB tensor, (1, 3, h, w) or (3, h, w) when read from shared buffer."""
    img_shape: Int[Tensor, "*batch 2"]
    """(height, width) of the processed image after optional downsampling."""
    img_true_shape: Int[Tensor, "*batch 2"]
    """(height, width) of the image before any downsampling."""
    rgb: Float[Tensor, "*batch h w 3"]
    """Normalized RGB image in [0, 1] range, HWC layout on CPU."""
    world_sim3_cam: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    """World-from-camera Sim3 pose."""
    X_canon: Float[Tensor, "hw 3"] | None = None
    """Canonical 3D point map, shape (h*w, 3)."""
    C: Float[Tensor, "hw 1"] | None = None
    """Per-point confidence values, shape (h*w, 1)."""
    feat: Float[Tensor, "1 n_patches feat_dim"] | None = None
    """Encoded MASt3R feature tokens."""
    pos: Int[Tensor, "1 n_patches 2"] | None = None
    """Positional encodings for feature patches."""
    N: int = 0
    """Number of accumulated point map observations."""
    N_updates: int = 0
    """Total number of point map update calls."""
    K: Float[Tensor, "3 3"] | None = None
    """Camera intrinsic matrix (only set when using calibration)."""
    score: Float[Tensor, ""] | None = None
    """Scalar filtering score (set by ``best_score`` filtering mode)."""

    def get_score(self, C: Float[Tensor, "hw 1"]) -> Float[Tensor, ""]:
        """Compute a scalar filtering score from confidence values.

        Args:
            C: Per-point confidence tensor of shape (hw, 1).

        Returns:
            Scalar score tensor (median or mean of C, depending on config).
        """
        score_fn: FilteringScore = FilteringScore(config["tracking"]["filtering_score"])
        match score_fn:
            case FilteringScore.MEDIAN:
                return torch.median(C)
            case FilteringScore.MEAN:
                return torch.mean(C)
            case _:
                raise ValueError(f"Unknown filtering score: {score_fn}")

    def update_pointmap(
        self,
        X: Float[Tensor, "hw 3"],
        C: Float[Tensor, "hw 1"],
    ) -> None:
        """Update the canonical point map using the configured filtering strategy.

        See ``FilteringMode`` for a description of each strategy.

        Args:
            X: New 3D point map of shape (h*w, 3).
            C: New confidence values of shape (h*w, 1).
        """
        mode: FilteringMode = FilteringMode(config["tracking"]["filtering_mode"])

        # First observation — always accept regardless of mode.
        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if mode is FilteringMode.BEST_SCORE:
                self.score = self.get_score(C)
            return

        match mode:
            case FilteringMode.FIRST:
                # Accept only the second observation (first update after init).
                if self.N_updates == 1:
                    self.X_canon = X.clone()
                    self.C = C.clone()
                    self.N = 1

            case FilteringMode.RECENT:
                # Always overwrite with the latest observation.
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1

            case FilteringMode.BEST_SCORE:
                # Keep whichever full observation has the highest aggregate score.
                new_score: Float[Tensor, ""] = self.get_score(C)
                assert self.score is not None
                if new_score > self.score:
                    self.X_canon = X.clone()
                    self.C = C.clone()
                    self.N = 1
                    self.score = new_score

            case FilteringMode.INDEP_CONF:
                # Per-point cherry-pick: keep the higher-confidence point at each pixel.
                assert self.C is not None
                assert self.X_canon is not None
                new_mask: Bool[Tensor, "hw 1"] = C > self.C
                self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
                self.C[new_mask] = C[new_mask]
                self.N = 1

            case FilteringMode.WEIGHTED_POINTMAP:
                # Confidence-weighted running average in Cartesian space.
                assert self.C is not None
                assert self.X_canon is not None
                self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
                self.C = self.C + C
                self.N += 1

            case FilteringMode.WEIGHTED_SPHERICAL:
                # Confidence-weighted running average in spherical (r, phi, theta) space.
                assert self.C is not None
                assert self.X_canon is not None
                spherical1: Float[Tensor, "hw 3"] = _cartesian_to_spherical(self.X_canon)
                spherical2: Float[Tensor, "hw 3"] = _cartesian_to_spherical(X)
                spherical: Float[Tensor, "hw 3"] = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)
                self.X_canon = _spherical_to_cartesian(spherical)
                self.C = self.C + C
                self.N += 1

        self.N_updates += 1

    def get_average_conf(self) -> Float[Tensor, "hw 1"] | None:
        """Return confidence divided by observation count, or None if no confidence."""
        return self.C / self.N if self.C is not None else None


def create_frame(
    i: int,
    rgb: RgbNormalized,
    world_sim3_cam: lietorch.Sim3,
    img_size: Literal[224, 512] = 512,
    device: str = "cuda:0",
) -> Frame:
    """Create a Frame from a raw image dict and a Sim3 pose.

    Args:
        i: Frame index in the dataset.
        rgb: Normalized RGB image passed to ``resize_img``.
        world_sim3_cam: World-from-camera Sim3 pose.
        img_size: Target image size for MASt3R encoder (224 or 512).
        device: Torch device string.

    Returns:
        A fully constructed Frame ready for tracking.
    """
    resized: ResizedImage = resize_img(rgb, img_size)
    rgb_tensor: Float[Tensor, "1 3 h w"] = resized.rgb_tensor.to(device=device)
    img_shape: Int[Tensor, "1 2"] = torch.tensor(resized.true_shape, device=device)
    img_true_shape: Int[Tensor, "1 2"] = img_shape.clone()
    rgb_cropped: Float[Tensor, "h w 3"] = torch.from_numpy(resized.rgb_uint8) / 255.0
    downsample: int = config["dataset"]["img_downsample"]
    if downsample > 1:
        rgb_cropped = rgb_cropped[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame: Frame = Frame(i, rgb_tensor, img_shape, img_true_shape, rgb_cropped, world_sim3_cam)
    return frame


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
        self.h: int = h
        self.w: int = w
        self.dtype: torch.dtype = dtype
        self.device: str = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.INIT)
        self.reloc_sem = manager.Value("i", 0)
        self.global_optimizer_tasks = manager.list()
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

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
        """Copy the frame's data into shared memory tensors.

        Args:
            frame: The Frame whose data should be written.
        """
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
        """Read the current frame from shared memory.

        Returns:
            A Frame constructed from the shared state tensors.
        """
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

    def queue_global_optimization(self, idx: int) -> None:
        """Enqueue a keyframe index for global optimization.

        Args:
            idx: Keyframe index to optimize.
        """
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self) -> None:
        """Increment the relocalization semaphore by one."""
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self) -> None:
        """Decrement the relocalization semaphore if positive."""
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self) -> Mode:
        """Return the current operating mode.

        Returns:
            The current Mode enum value.
        """
        with self.lock:
            return self.mode.value

    def set_mode(self, mode: Mode) -> None:
        """Set the operating mode.

        Args:
            mode: The new Mode to set.
        """
        with self.lock:
            self.mode.value = mode

    def is_paused(self) -> bool:
        """Return whether the system is currently paused."""
        with self.lock:
            return self.paused.value == 1


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
    ) -> None:
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.h: int = h
        self.w: int = w
        self.buffer: int = buffer
        self.dtype: torch.dtype = dtype
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
        self.is_dirty: Bool[Tensor, "buf 1"] = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        self.K: Float[Tensor, "3 3"] = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        # fmt: on

    def __getitem__(self, idx) -> Frame:
        """Retrieve a keyframe by buffer index.

        Args:
            idx: Buffer index of the keyframe.

        Returns:
            A Frame constructed from the shared tensors at the given index.
        """
        with self.lock:
            # put all of the data into a frame
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
            if config["use_calib"]:
                kf.K = self.K
            return kf

    def __setitem__(self, idx: int, value: Frame) -> int:
        """Write a Frame into the shared buffer at the given index.

        Args:
            idx: Buffer index to write to.
            value: The Frame to store.

        Returns:
            The index that was written.
        """
        assert value.X_canon is not None
        assert value.C is not None
        assert value.feat is not None
        assert value.pos is not None
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # set the attributes
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
            self.is_dirty[idx] = True
            return idx

    def __len__(self) -> int:
        """Return the number of keyframes currently stored."""
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame) -> None:
        """Append a keyframe to the end of the buffer.

        Args:
            value: The Frame to append.
        """
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self) -> None:
        """Remove the last keyframe by decrementing the size counter."""
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Frame | None:
        """Return the most recently appended keyframe, or None if empty.

        Returns:
            The last Frame in the buffer, or None when the buffer is empty.
        """
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
            self.is_dirty[idx] = True

    def get_dirty_idx(self) -> Int[Tensor, "n_dirty"]:
        """Return indices of keyframes modified since the last call, then clear the dirty flags.

        Returns:
            1-D tensor of dirty keyframe buffer indices.
        """
        with self.lock:
            idx: Int[Tensor, "n_dirty"] = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K: Float[Tensor, "3 3"]) -> None:
        """Set the shared camera intrinsic matrix (requires ``use_calib`` config).

        Args:
            K: 3x3 intrinsic matrix.
        """
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K
