"""Score TUM trajectories against KITTI odometry ground truth (unaligned, SE(3), Sim(3)); see `colsfm.kitti_eval`."""

from colsfm.kitti_eval import Config, main

__all__ = ["Config", "main"]
