"""Typed readers for the output directory upstream 4DAnyone writes.

Upstream owns these files (``cameras.json``, ``metadata.json``); we read the fields we use and allow the rest.
"""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json


@serde
@dataclass(frozen=True, slots=True)
class CameraRecord:
    """One generated view's OpenCV camera in the canonical human world."""

    camera_id: int
    """View index; the video is ``videos/<camera_id:02d>.mp4``."""
    layer_index: int
    """Pitch ring of the view."""
    pitch: int
    """Ring pitch, degrees; positive is above the subject."""
    yaw: float
    """Yaw on the ring, degrees; 0 faces the person."""
    K: list[list[float]]
    """3x3 pixel intrinsics."""
    camera_to_world: list[list[float]]
    """4x4 pose, OpenCV camera axes (RDF)."""
    image_width: int
    """Video width, pixels."""
    image_height: int
    """Video height, pixels."""
    video: str
    """Generated video, relative to the output directory."""
    skeleton_video: str
    """Pose-conditioning video, relative to the output directory."""

    @property
    def intrinsics(self) -> Float64[ndarray, "3 3"]:
        """``K`` as an array."""
        return np.asarray(self.K, dtype=np.float64)

    @property
    def world_T_cam(self) -> Float64[ndarray, "4 4"]:
        """``camera_to_world`` as an array."""
        return np.asarray(self.camera_to_world, dtype=np.float64)


@serde
@dataclass(frozen=True, slots=True)
class CameraRig:
    """``cameras.json``: the N-camera rig of the generated views."""

    camera_model: str
    """Always ``OPENCV``."""
    cameras: list[CameraRecord]
    """One record per generated view, in view order."""
    front_camera_ids: list[int]
    """Views that face the person."""

    def __post_init__(self) -> None:
        if self.camera_model != "OPENCV":
            raise ValueError(f"expected an OPENCV camera rig, got {self.camera_model}")
        if [camera.camera_id for camera in self.cameras] != list(range(len(self.cameras))):
            raise ValueError("cameras are not in view order")


@serde
@dataclass(frozen=True, slots=True)
class InputRecord:
    """``metadata.json`` ``input``: the canonical clip cut from the input video."""

    filename: str
    """Input video file name."""
    fps: str
    """Canonical clip rate, ``num/den``."""
    start_time_seconds: float
    """Clip start on the input timeline."""
    num_frames: int
    """Canonical clip length (121)."""
    width: int
    """Input width, pixels."""
    height: int
    """Input height, pixels."""

    @property
    def frame_rate(self) -> Fraction:
        """``fps`` as a fraction."""
        return Fraction(self.fps)


@serde
@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """``metadata.json`` ``generation``: what the denoiser ran and how long it took."""

    seed: int
    """Noise seed."""
    attention_backend: str
    """Resolved attention backend (``sageattention``, ``sdpa``, ``flash_attn_3``)."""
    inference_steps: int
    """Denoising steps (4 for Turbo)."""
    elapsed_seconds: dict[str, float]
    """Wall time per generation stage."""
    total_elapsed_seconds: float
    """Wall time of the whole pipeline, from the start of run_pipeline."""
    peak_vram_allocated_bytes: int
    """Peak CUDA memory allocated during generation."""
    peak_vram_reserved_bytes: int
    """Peak CUDA memory reserved during generation."""


@serde
@dataclass(frozen=True, slots=True)
class RunMetadata:
    """``metadata.json``: the fields of upstream's run summary that we read."""

    input: InputRecord
    """The canonical clip."""
    generation: GenerationRecord
    """The generation stage."""


def read_camera_rig(output_dir: Path) -> CameraRig:
    """Read ``<output_dir>/cameras.json``."""
    path: Path = output_dir / "cameras.json"
    try:
        return from_json(CameraRig, path.read_text(encoding="utf-8"))
    except (SerdeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: not a 4DAnyone camera rig: {error}") from error


def read_run_metadata(output_dir: Path) -> RunMetadata:
    """Read ``<output_dir>/metadata.json``."""
    path: Path = output_dir / "metadata.json"
    try:
        return from_json(RunMetadata, path.read_text(encoding="utf-8"))
    except (SerdeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: not a 4DAnyone run summary: {error}") from error
