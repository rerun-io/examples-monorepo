import pathlib

import cv2
import numpy as np
import torch
from jaxtyping import UInt8

from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray


def prepare_savedir(args: object, dataset: object) -> tuple[pathlib.Path, str]:
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
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
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
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)

        t = timestamps[keyframe.frame_id]
        reconstruction[i] = {
            "frame_id": i,
            "timestamp": t,
            "T_WC": keyframe.T_WC.cpu(),
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
        bgr_img: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(
            (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(str(filename), bgr_img)
