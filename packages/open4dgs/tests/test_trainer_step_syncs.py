"""The train step never waits on the host: GPU-resident frames/masks (trigs_adapter)."""

import pytest
import torch

trigs = pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")


def _clip_and_masks(num_frames: int = 3):
    from open4dgs.catalog_clip import MultiViewClip

    names = ("rig_01/cam_00", "rig_00/cam_00", "rig_02/cam_00")
    gen = torch.Generator().manual_seed(0)
    frames = tuple(torch.randint(0, 256, (num_frames, 8, 12, 3), dtype=torch.uint8, generator=gen) for _ in names)
    masks = tuple(torch.rand(num_frames, 8, 12, generator=gen) for _ in names)
    clip = MultiViewClip(
        camera_names=names,
        frames=frames,
        valid=tuple(torch.ones(8, 12, dtype=torch.bool) for _ in names),
        K=torch.eye(3).expand(len(names), 3, 3).clone(),
        world_T_cam=torch.eye(4).expand(len(names), 4, 4).clone(),
        fps=30.0,
        frame_indices=torch.arange(num_frames),
        people=None,
    )
    return clip, masks


def test_trainer_frames_device_none_keeps_tensors_and_blanks_holdout() -> None:
    from open4dgs.trigs_adapter import resident_device, trainer_frames

    clip, masks = _clip_and_masks()
    gt_images, gt_alphas, positions = trainer_frames(clip, masks, ("rig_02/cam_00",))
    assert positions == (2,)
    for f in range(3):
        for c in range(3):
            if c == 2:
                assert gt_images[f][c].shape == clip.frames[c][f].shape and int(gt_images[f][c].sum()) == 0
                assert float(gt_alphas[f][c].sum()) == 0.0 and gt_alphas[f][c].dtype == torch.float32
            else:
                assert gt_images[f][c].data_ptr() == clip.frames[c][f].data_ptr()  # a view, not a copy
                assert torch.equal(gt_alphas[f][c], masks[c][f])
    assert resident_device(clip, masks, ("rig_02/cam_00",), torch.device("cpu")) is None


@pytest.mark.integration
def test_trainer_frames_on_cuda_equal_the_cpu_ones() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from open4dgs.trigs_adapter import resident_device, trainer_frames

    clip, masks = _clip_and_masks()
    dev = torch.device("cuda")
    assert resident_device(clip, masks, ("rig_02/cam_00",), dev) == dev
    cpu_images, cpu_alphas, _ = trainer_frames(clip, masks, ("rig_02/cam_00",))
    gpu_images, gpu_alphas, _ = trainer_frames(clip, masks, ("rig_02/cam_00",), dev)
    for f in range(3):
        for c in range(3):
            assert gpu_images[f][c].is_cuda and gpu_alphas[f][c].is_cuda and gpu_images[f][c].dtype == torch.uint8
            # What the trainer computes from each (trainer.py: .to(device).float().div(255), .to(device).float()).
            assert torch.equal(gpu_images[f][c].to(dev).float().div(255.0).cpu(), cpu_images[f][c].to(dev).float().div(255.0).cpu())
            assert torch.equal(gpu_alphas[f][c].to(dev).float().cpu(), cpu_alphas[f][c].to(dev).float().cpu())
