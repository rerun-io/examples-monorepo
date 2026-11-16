"""Reusable GVHMR result stored as JSON plus safetensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from fdanyone.errors import FourDAnyoneError

SMPL_PARAMETER_NAMES = ("body_pose", "betas", "global_orient", "transl")
SMPL_PARAMETER_WIDTHS = {"body_pose": 63, "betas": 10, "global_orient": 3, "transl": 3}


@dataclass(frozen=True)
class MotionResult:
    gvhmr_revision: str
    fps: Fraction
    frame_timestamps_sec: tuple[float, ...]
    source_frame_indices: tuple[int, ...]
    source_pts: tuple[int | None, ...]
    source_size_bytes: int
    source_mtime_ns: int
    image_height: int
    image_width: int
    smpl_params_global: dict[str, Any]
    smpl_params_incam: dict[str, Any]
    K_fullimg: Any
    observed_keypoints_2d: Any
    motion_world: str = "gvhmr_gravity_aligned_y_up"

    @property
    def num_frames(self) -> int:
        return len(self.frame_timestamps_sec)

    def validate(self, expected_frames: int = 121) -> None:
        try:
            import torch
        except ImportError as exc:
            raise FourDAnyoneError("PyTorch is required to validate motion tensors.") from exc
        if self.motion_world != "gvhmr_gravity_aligned_y_up":
            raise FourDAnyoneError(f"Unknown motion world convention: {self.motion_world!r}.")
        if (
            not isinstance(self.gvhmr_revision, str)
            or len(self.gvhmr_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.gvhmr_revision.lower())
        ):
            raise FourDAnyoneError("MotionResult requires a 40-character GVHMR git revision.")
        if self.fps <= 0:
            raise FourDAnyoneError(f"MotionResult FPS must be positive, got {self.fps}.")
        if self.num_frames != expected_frames:
            raise FourDAnyoneError(f"MotionResult has {self.num_frames} frames, expected {expected_frames}.")
        for name, values in (
            ("source_frame_indices", self.source_frame_indices),
            ("source_pts", self.source_pts),
        ):
            if len(values) != self.num_frames:
                raise FourDAnyoneError(f"MotionResult {name} has {len(values)} values, expected {self.num_frames}.")
        expected_timestamps = tuple(float(Fraction(index, 1) / self.fps) for index in range(self.num_frames))
        if self.frame_timestamps_sec != expected_timestamps:
            raise FourDAnyoneError("MotionResult timestamps are not the exact zero-based CFR timeline.")
        if any(index < 0 for index in self.source_frame_indices) or any(
            right < left for left, right in zip(self.source_frame_indices, self.source_frame_indices[1:], strict=False)
        ):
            raise FourDAnyoneError("MotionResult source-frame indices must be non-negative and monotonic.")
        if self.source_size_bytes <= 0 or self.source_mtime_ns <= 0:
            raise FourDAnyoneError("MotionResult has an invalid source-file identity.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise FourDAnyoneError("MotionResult image dimensions must be positive.")
        for group_name, parameters in (
            ("smpl_params_global", self.smpl_params_global),
            ("smpl_params_incam", self.smpl_params_incam),
        ):
            if set(parameters) != set(SMPL_PARAMETER_NAMES):
                raise FourDAnyoneError(
                    f"MotionResult {group_name} must contain {SMPL_PARAMETER_NAMES}, got {tuple(parameters)}."
                )
            for name, tensor in parameters.items():
                expected_shape = (self.num_frames, SMPL_PARAMETER_WIDTHS[name])
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                    raise FourDAnyoneError(f"{group_name}.{name} must have shape {expected_shape}.")
                if not bool(torch.isfinite(tensor).all()):
                    raise FourDAnyoneError(f"{group_name}.{name} contains non-finite values.")
        if not isinstance(self.K_fullimg, torch.Tensor) or tuple(self.K_fullimg.shape) != (self.num_frames, 3, 3):
            raise FourDAnyoneError(f"K_fullimg must have shape ({self.num_frames}, 3, 3).")
        if not bool(torch.isfinite(self.K_fullimg).all()):
            raise FourDAnyoneError("K_fullimg contains non-finite values.")
        expected_keypoint_shape = (self.num_frames, 17, 3)
        if (
            not isinstance(self.observed_keypoints_2d, torch.Tensor)
            or tuple(self.observed_keypoints_2d.shape) != expected_keypoint_shape
        ):
            raise FourDAnyoneError(f"observed_keypoints_2d must have shape {expected_keypoint_shape}.")
        if not bool(torch.isfinite(self.observed_keypoints_2d).all()):
            raise FourDAnyoneError("observed_keypoints_2d contains non-finite values.")

    def validate_against_clip(self, clip) -> None:
        """Reject a cached result produced from a different video timeline."""

        self.validate(expected_frames=len(clip.frames))
        expected_timestamps = tuple(float(frame.canonical_timestamp) for frame in clip.frames)
        expected_indices = tuple(frame.source_index for frame in clip.frames)
        expected_pts = tuple(frame.source_pts for frame in clip.frames)
        if self.fps != clip.fps:
            raise FourDAnyoneError(f"Motion FPS {self.fps} does not match canonical FPS {clip.fps}.")
        if self.frame_timestamps_sec != expected_timestamps:
            raise FourDAnyoneError("Motion timestamps do not match the canonical clip.")
        if self.source_frame_indices != expected_indices or self.source_pts != expected_pts:
            raise FourDAnyoneError("Motion source-frame identity does not match the canonical clip.")
        if (self.source_size_bytes, self.source_mtime_ns) != (
            clip.source_size_bytes,
            clip.source_mtime_ns,
        ):
            raise FourDAnyoneError("Cached GVHMR motion belongs to a different source file.")

    def save(self, directory: str | Path) -> Path:
        from safetensors.torch import save_file

        self.validate()
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        tensor_path = root / "motion.safetensors"
        tensors = {
            **{
                f"smpl_params_global.{name}": self.smpl_params_global[name].detach().cpu().contiguous()
                for name in SMPL_PARAMETER_NAMES
            },
            **{
                f"smpl_params_incam.{name}": self.smpl_params_incam[name].detach().cpu().contiguous()
                for name in SMPL_PARAMETER_NAMES
            },
            "K_fullimg": self.K_fullimg.detach().cpu().contiguous(),
            "observed_keypoints_2d": self.observed_keypoints_2d.detach().cpu().contiguous(),
        }
        save_file(tensors, str(tensor_path))
        metadata = {
            "gvhmr_revision": self.gvhmr_revision,
            "num_frames": self.num_frames,
            "fps_num": self.fps.numerator,
            "fps_den": self.fps.denominator,
            "frame_timestamps_sec": self.frame_timestamps_sec,
            "source_frame_indices": self.source_frame_indices,
            "source_pts": self.source_pts,
            "source_size_bytes": self.source_size_bytes,
            "source_mtime_ns": self.source_mtime_ns,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "motion_world": self.motion_world,
            "tensor_file": tensor_path.name,
        }
        (root / "motion.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return root

    @classmethod
    def load(cls, directory: str | Path) -> MotionResult:
        from safetensors.torch import load_file

        root = Path(directory).expanduser().resolve()
        metadata = json.loads((root / "motion.json").read_text())
        tensors = load_file(str(root / metadata["tensor_file"]), device="cpu")
        # ``safetensors`` may keep the source file memory-mapped for as long as
        # returned tensors own that storage.  Motion results live until final
        # export, which in turn can leave an in-use ``.efc_*`` tombstone on
        # CPFS/NFS when scratch cleanup unlinks the backing file.  The payload
        # is tiny compared with the generated videos, so take ownership here
        # and close the file mapping at this explicit process boundary.
        owned = {name: tensor.clone() for name, tensor in tensors.items()}
        del tensors
        result = cls(
            gvhmr_revision=str(metadata["gvhmr_revision"]),
            fps=Fraction(int(metadata["fps_num"]), int(metadata["fps_den"])),
            frame_timestamps_sec=tuple(float(value) for value in metadata["frame_timestamps_sec"]),
            source_frame_indices=tuple(int(value) for value in metadata["source_frame_indices"]),
            source_pts=tuple(None if value is None else int(value) for value in metadata["source_pts"]),
            source_size_bytes=int(metadata["source_size_bytes"]),
            source_mtime_ns=int(metadata["source_mtime_ns"]),
            image_height=int(metadata["image_height"]),
            image_width=int(metadata["image_width"]),
            smpl_params_global={name: owned[f"smpl_params_global.{name}"] for name in SMPL_PARAMETER_NAMES},
            smpl_params_incam={name: owned[f"smpl_params_incam.{name}"] for name in SMPL_PARAMETER_NAMES},
            K_fullimg=owned["K_fullimg"],
            observed_keypoints_2d=owned["observed_keypoints_2d"],
            motion_world=str(metadata["motion_world"]),
        )
        result.validate()
        return result
