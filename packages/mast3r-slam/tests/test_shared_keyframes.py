"""SharedKeyframes regression tests."""

import multiprocessing as mp

import lietorch
import torch

from mast3r_slam.frame import Frame, SharedKeyframes


def test_update_world_sim3_cams_marks_keyframes_dirty() -> None:
    """Backend pose updates must mark keyframes dirty for async camera replay."""
    manager = mp.Manager()
    try:
        h, w = 16, 16
        keyframes = SharedKeyframes(manager, h, w, buffer=4, device="cpu")
        frame = Frame(
            frame_id=0,
            rgb_tensor=torch.zeros(1, 3, h, w),
            img_shape=torch.tensor([[h, w]], dtype=torch.int),
            img_true_shape=torch.tensor([[h, w]], dtype=torch.int),
            rgb=torch.zeros(h, w, 3),
            world_sim3_cam=lietorch.Sim3.Identity(1, device="cpu"),
            X_canon=torch.zeros(h * w, 3),
            C=torch.ones(h * w, 1),
            feat=torch.zeros(1, 1, keyframes.feat_dim),
            pos=torch.zeros(1, 1, 2, dtype=torch.long),
        )
        keyframes.append(frame)

        # Clear the dirty bit from the initial append so this only checks backend writes.
        keyframes.get_dirty_idx()

        updated_pose = lietorch.Sim3(torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]], dtype=torch.float32))
        keyframes.update_world_sim3_cams(updated_pose, torch.tensor([0], dtype=torch.long))

        dirty_idx = keyframes.get_dirty_idx()
        assert dirty_idx.tolist() == [0]
    finally:
        manager.shutdown()
