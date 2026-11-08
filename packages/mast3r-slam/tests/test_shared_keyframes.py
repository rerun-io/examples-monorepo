"""SharedKeyframes regression tests."""

import multiprocessing as mp

import lietorch_mojo as lietorch
import torch

from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates


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


def test_update_pointmap_keeps_the_backend_pose() -> None:
    """Tracking's write-back of a refined pointmap must not overwrite a pose the backend wrote meanwhile."""
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
        backend_pose = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0]]])
        keyframes.update_world_sim3_cams(lietorch.Sim3(backend_pose), torch.tensor([0], dtype=torch.long))

        # The tracker refined the pointmap of a keyframe it read before the backend's write.
        frame.X_canon = torch.full((h * w, 3), 2.0)
        frame.C = torch.full((h * w, 1), 3.0)
        frame.N = 2
        frame.N_updates = 2
        keyframes.update_pointmap(0, frame)

        assert torch.equal(keyframes.world_sim3_cam[0], backend_pose[0])
        assert torch.equal(keyframes.X[0], frame.X_canon)
        assert torch.equal(keyframes.C[0], frame.C)
        assert (int(keyframes.N[0]), int(keyframes.N_updates[0])) == (2, 2)
    finally:
        manager.shutdown()


def test_reloc_requests_are_counted_and_a_late_success_never_overwrites_terminated() -> None:
    """The backend relocalises only while a request is pending, and its TRACKING after a success must not undo a
    TERMINATED the frontend set meanwhile (both left the backend running after the run: the CLI hung, the app leaked it)."""
    from mast3r_slam.frame import Mode

    manager = mp.Manager()
    try:
        states = SharedStates(manager, 16, 16, device="cpu")
        assert not states.has_reloc_request()
        states.queue_reloc()
        assert states.has_reloc_request()
        states.dequeue_reloc()
        assert not states.has_reloc_request()

        states.set_mode(Mode.RELOC)
        assert states.set_mode_if(Mode.RELOC, Mode.TRACKING)
        assert states.get_mode() is Mode.TRACKING
        states.set_mode(Mode.TERMINATED)
        assert not states.set_mode_if(Mode.RELOC, Mode.TRACKING)
        assert states.get_mode() is Mode.TERMINATED
    finally:
        manager.shutdown()
