import pathlib
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch
from jaxtyping import UInt8
from numpy import ndarray

from mast3r_slam.config import config
from mast3r_slam.dataloader import Intrinsics, MonocularDataset
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import constrain_points_to_ray
from mast3r_slam.lietorch_utils import as_SE3


@runtime_checkable
class _HasSaveAs(Protocol):
    """Protocol for objects with a ``save_as`` attribute."""

    save_as: str


@runtime_checkable
class _HasDatasetPath(Protocol):
    """Protocol for objects with a ``dataset_path`` attribute."""

    dataset_path: pathlib.Path


def prepare_savedir(args: _HasSaveAs, dataset: MonocularDataset) -> tuple[pathlib.Path, str]:
    """Create the log directory and derive a sequence name from the dataset path.

    Args:
        args: Argument namespace containing a ``save_as`` attribute.
        dataset: Dataset object with a ``dataset_path`` attribute.

    Returns:
        A tuple of (save_dir, seq_name) where save_dir is the created
        ``pathlib.Path`` and seq_name is the stem of the dataset path.
    """
    save_dir: pathlib.Path = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name: str = dataset.dataset_path.stem
    return save_dir, seq_name


def save_ATE(
    logdir: str | pathlib.Path,
    logfile: str,
    timestamps: list | np.ndarray,
    frames: SharedKeyframes,
    intrinsics: Intrinsics | None = None,
) -> None:
    """Save Absolute Trajectory Error poses to a TUM-format text file.

    Each line is: ``timestamp tx ty tz qx qy qz qw``.

    Args:
        logdir: Directory to write the trajectory file into.
        logfile: File name for the trajectory file.
        timestamps: Per-frame timestamps indexed by ``frame_id``.
        frames: Shared keyframe buffer to read poses from.
        intrinsics: Optional camera intrinsics for calibrated pose refinement.
    """
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile_path: pathlib.Path = logdir / logfile
    with open(logfile_path, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            # TODO: calibrated pose refinement not yet implemented
            world_se3_cam = as_SE3(keyframe.world_sim3_cam)
            x, y, z, qx, qy, qz, qw = world_se3_cam.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(
    savedir: str | pathlib.Path,
    filename: str,
    timestamps: list | np.ndarray,
    keyframes: SharedKeyframes,
) -> None:
    """Save the full 3D reconstruction (poses + point maps) to a ``.pt`` file.

    Args:
        savedir: Directory to save the reconstruction file into.
        filename: File name for the ``.pt`` file.
        timestamps: Per-frame timestamps indexed by ``frame_id``.
        keyframes: Shared keyframe buffer to read data from.
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    reconstruction: dict[int, dict] = {}
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            assert keyframe.X_canon is not None
            assert keyframe.K is not None
            kf_img_shape: tuple[int, int] = (int(keyframe.img_shape.flatten()[0]), int(keyframe.img_shape.flatten()[1]))
            X_canon = constrain_points_to_ray(
                kf_img_shape, keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)

        assert keyframe.X_canon is not None
        assert keyframe.C is not None
        t = timestamps[keyframe.frame_id]
        reconstruction[i] = {
            "frame_id": i,
            "timestamp": t,
            "world_sim3_cam": keyframe.world_sim3_cam.cpu(),
            "X": keyframe.X_canon.cpu(),
            "X_canon": keyframe.X_canon.cpu(),
            "C": keyframe.C.cpu(),
        }
    torch.save(reconstruction, savedir / filename)


def save_keyframes(
    savedir: str | pathlib.Path,
    timestamps: list | np.ndarray,
    keyframes: SharedKeyframes,
) -> None:
    """Save keyframe images as PNG files.

    Args:
        savedir: Directory to save keyframe images into.
        timestamps: Per-frame timestamps indexed by ``frame_id``.
        keyframes: Shared keyframe buffer to read images from.
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename: pathlib.Path = savedir / f"{t}.png"
        bgr_img: UInt8[ndarray, "h w 3"] = cv2.cvtColor(
            (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(str(filename), bgr_img)
