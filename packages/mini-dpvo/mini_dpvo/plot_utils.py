"""Visualization and trajectory I/O helpers for DPVO evaluation.

Provides utilities to convert DPVO output (poses as ``[tx, ty, tz, qw, qx, qy, qz]``
arrays) into ``evo`` :class:`PoseTrajectory3D` objects, plot predicted vs.
ground-truth trajectories, and export to TUM format for benchmarking.

Requires the ``evo`` evaluation toolkit (https://github.com/MichaelGrupp/evo).
"""

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
    """Convert a ``(poses, timestamps)`` tuple into an ``evo`` trajectory.

    If ``args`` is already a :class:`PoseTrajectory3D`, a deep copy is
    returned to avoid mutating the caller's data during alignment.

    The pose convention is ``[tx, ty, tz, qw, qx, qy, qz]`` --
    positions followed by a wxyz quaternion, matching the lietorch /
    DPVO output format.

    Args:
        args: Either a 2-tuple of (poses array ``[n, 7]``, timestamps
            array ``[n]``), or an existing ``PoseTrajectory3D``.

    Returns:
        A fresh ``PoseTrajectory3D`` instance.
    """
    if isinstance(args, tuple):
        traj: Float64[ndarray, "n 7"] = args[0]
        tstamps: Float64[ndarray, "n"] = args[1]
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def best_plotmode(traj: PoseTrajectory3D) -> Any:
    """Choose the 2-D projection plane that captures the most trajectory variance.

    Computes per-axis variance of the 3-D positions and selects the two
    axes with the highest variance as the plot axes.  This avoids
    degenerate top-down or side views for trajectories that are
    predominantly planar.

    Args:
        traj: Trajectory whose positions are analysed.

    Returns:
        An ``evo.tools.plot.PlotMode`` enum value (e.g. ``PlotMode.xy``).
    """
    variances: Float64[ndarray, "3"] = np.var(traj.positions_xyz, axis=0)
    sorted_indices: Int[ndarray, "3"] = np.argsort(variances)
    _: int = int(sorted_indices[0])
    i1: int = int(sorted_indices[1])
    i2: int = int(sorted_indices[2])
    # Highest-variance axis first (horizontal), second-highest vertical.
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
    """Plot predicted trajectory (and optionally ground truth) and save to file.

    When a ground-truth trajectory is provided the two are first
    associated by timestamp, then the prediction is Sim(3)-aligned to
    the ground truth (Umeyama alignment), optionally correcting scale.

    The function automatically selects the best 2-D projection plane
    via :func:`best_plotmode`.

    Args:
        pred_traj: Predicted trajectory as a ``(poses, timestamps)`` tuple
            or ``PoseTrajectory3D``.
        gt_traj: Optional ground-truth trajectory in the same format.
        title: Plot title string.
        filename: Output file path (e.g. ``"traj.png"`` or ``"traj.pdf"``).
        align: Whether to Sim(3)-align the prediction to ground truth.
        correct_scale: Whether to correct scale during alignment.  Only
            meaningful when ``align=True``.
    """
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
    """Save a trajectory in TUM RGB-D benchmark format.

    Each line of the output file contains::

        timestamp tx ty tz qx qy qz qw

    Note the quaternion order is ``[qx, qy, qz, qw]`` (Hamilton, scalar-last)
    as required by the TUM benchmark, whereas internally DPVO / ``evo`` use
    ``[qw, qx, qy, qz]`` (scalar-first).  The conversion is handled here
    via index reordering ``[1, 2, 3, 0]``.

    Args:
        traj: Trajectory as a ``(poses, timestamps)`` tuple or
            ``PoseTrajectory3D``.
        filename: Output file path.
    """
    traj_obj: PoseTrajectory3D = make_traj(traj)
    def tostr(a: ndarray) -> str:
        return ' '.join(map(str, a))
    with Path(filename).open('w') as f:
        for i in range(traj_obj.num_poses):
            # Reorder quaternion from [qw, qx, qy, qz] to [qx, qy, qz, qw]
            f.write(f"{traj_obj.timestamps[i]} {tostr(traj_obj.positions_xyz[i])} {tostr(traj_obj.orientations_quat_wxyz[i][[1,2,3,0]])}\n")
    print(f"Saved {filename}")
