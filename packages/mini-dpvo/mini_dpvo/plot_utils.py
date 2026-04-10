from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D

# from evo.tools import plot
from jaxtyping import Float64, Int
from numpy import ndarray


def make_traj(args: tuple[Float64[ndarray, "n 7"], Float64[ndarray, "n"]] | PoseTrajectory3D) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj: Float64[ndarray, "n 7"] = args[0]
        tstamps: Float64[ndarray, "n"] = args[1]
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def best_plotmode(traj: PoseTrajectory3D) -> Any:
    variances: Float64[ndarray, "3"] = np.var(traj.positions_xyz, axis=0)
    sorted_indices: Int[ndarray, "3"] = np.argsort(variances)
    _: int = int(sorted_indices[0])
    i1: int = int(sorted_indices[1])
    i2: int = int(sorted_indices[2])
    plot_axes: str = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)

def plot_trajectory(
    pred_traj: tuple[Float64[ndarray, "n 7"], Float64[ndarray, "n"]] | PoseTrajectory3D,
    gt_traj: tuple[Float64[ndarray, "n 7"], Float64[ndarray, "n"]] | PoseTrajectory3D | None = None,
    title: str = "",
    filename: str = "",
    align: bool = True,
    correct_scale: bool = True,
) -> None:
    pred_traj_aligned: PoseTrajectory3D = make_traj(pred_traj)

    gt_traj_aligned: PoseTrajectory3D | None = None
    if gt_traj is not None:
        gt_traj_aligned = make_traj(gt_traj)
        gt_traj_aligned, pred_traj_aligned = sync.associate_trajectories(gt_traj_aligned, pred_traj_aligned)

        if align:
            pred_traj_aligned.align(gt_traj_aligned, correct_scale=correct_scale)

    plot_collection: Any = plot.PlotCollection("PlotCol")
    fig: matplotlib.figure.Figure = plt.figure(figsize=(8, 8))
    plot_mode: Any = best_plotmode(gt_traj_aligned if (gt_traj_aligned is not None) else pred_traj_aligned)
    ax: matplotlib.axes.Axes = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj_aligned is not None:
        plot.traj(ax, plot_mode, gt_traj_aligned, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj_aligned, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

def save_trajectory_tum_format(
    traj: tuple[Float64[ndarray, "n 7"], Float64[ndarray, "n"]] | PoseTrajectory3D,
    filename: str,
) -> None:
    traj_obj: PoseTrajectory3D = make_traj(traj)
    tostr: Callable[[ndarray], str] = lambda a: ' '.join(map(str, a))
    with Path(filename).open('w') as f:
        for i in range(traj_obj.num_poses):
            f.write(f"{traj_obj.timestamps[i]} {tostr(traj_obj.positions_xyz[i])} {tostr(traj_obj.orientations_quat_wxyz[i][[1,2,3,0]])}\n")
    print(f"Saved {filename}")
