"""robocap-live's hands-layer writer (Rust, ``crates/robocap-live-py``), built in place by the ``robocap-live-build`` task.

``HandsLayer`` runs robocap-live's own pipeline (the scheduler of ``robocap-live --source replay --slam reference``) on the
framesets pushed into it.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64, UInt8

CAMERA_NAMES: list[str]
FULL_WIDTH: int
FULL_HEIGHT: int

class Rig:
    """The six calibrated cameras of a RoboCap rig, in ``CAMERA_NAMES`` order."""

    def __init__(
        self,
        *,
        names: list[str],
        resolution_wh: Float64[np.ndarray, "c 2"],
        cam_from_rig: Float64[np.ndarray, "c 4 4"],
        focal: Float64[np.ndarray, "c 2"],
        principal: Float64[np.ndarray, "c 2"],
        fisheye62: Float64[np.ndarray, "c 8"],
        source: str,
        device: str,
    ) -> None: ...
    @property
    def names(self) -> list[str]: ...
    @property
    def source(self) -> str: ...
    @property
    def device(self) -> str: ...
    def to_json(self) -> str: ...

class HandStageTimes:
    """The tracker's split of a step (or of a whole layer), milliseconds."""

    @property
    def detnet_ms(self) -> float: ...
    @property
    def crops_ms(self) -> float: ...
    @property
    def keynet_ms(self) -> float: ...
    @property
    def fit_ms(self) -> float: ...
    @property
    def tracker_ms(self) -> float: ...

class LayerSummary:
    """A finished layer's totals."""

    @property
    def framesets(self) -> int: ...
    @property
    def with_pose(self) -> int: ...
    @property
    def held_pose(self) -> int: ...
    @property
    def tracked(self) -> tuple[int, int]: ...
    @property
    def reported(self) -> tuple[int, int]: ...
    @property
    def scale(self) -> float: ...
    @property
    def scale_final(self) -> bool: ...
    @property
    def hand_stages(self) -> HandStageTimes: ...
    def stage_total_ms(self, stage: str) -> float:
        """Summed wall time of ``downsample``, ``pose_wait``, ``hands``, ``output`` or ``end_to_end``; the stages overlap."""
    @property
    def log_worker_ms_mean(self) -> float: ...

class HandsLayer:
    """One segment's hands layer: ``push`` every frameset in time order, then ``finish``."""

    def __init__(
        self,
        rig: Rig,
        output: Path,
        recording_id: str,
        reference_t_ns: Int64[np.ndarray, " p"],
        reference_world_from_rig: Float64[np.ndarray, "p 4 4"],
        *,
        nets: str = "ort",
        models_dir: Path | None = None,
        device: str = "auto",
        ort_dylib: Path | None = None,
        ort_threads: int = 0,
        overlays: str = "debug",
    ) -> None: ...
    @property
    def nets(self) -> str: ...
    def push(self, t_ns: int, frames: Sequence[UInt8[np.ndarray, "h w"] | None], camera_t_ns: Sequence[int] | None = None) -> int:
        """Queue Rust-owned frame snapshots; source arrays may be mutated or reused after this returns."""
    def finish(self) -> LayerSummary: ...
    def abort(self) -> None:
        """Stop the pipeline without finishing the layer (its file is left incomplete); a no-op once finished or aborted."""
