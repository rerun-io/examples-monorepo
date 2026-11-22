"""The train step never waits on the host: GPU-resident frames/masks (trigs_adapter), recipe patch 0004 (host-sync fixes
inside train_step + the _const_tensor seam) and open4dgs.trigs_trainer (int camera sizes, the constant-tensor cache).

The maths must not change: the pieces replaced are checked here against the expressions they replace.
"""

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


def test_camera_width_height_are_python_ints_equal_to_the_tensors() -> None:
    from open4dgs.trigs_trainer import HostSizedCameraConfig

    widths, heights = torch.tensor([12, 1028, 7]), torch.tensor([8, 752, 5])
    cams = HostSizedCameraConfig(viewmats=torch.eye(4).expand(3, 4, 4).clone(), Ks=torch.eye(3).expand(3, 3, 3).clone(), widths=widths, heights=heights)
    for c in range(3):
        assert type(cams.width_for(c)) is int and cams.width_for(c) == int(widths[c].item())
        assert type(cams.height_for(c)) is int and cams.height_for(c) == int(heights[c].item())
    assert cams.size_for(1) == (1028, 752)


def test_camera_config_from_matrices_builds_the_host_sized_subclass() -> None:
    from open4dgs.trigs_adapter import camera_config
    from open4dgs.trigs_trainer import HostSizedCameraConfig

    cams = camera_config(torch.eye(4).expand(2, 4, 4).clone(), torch.eye(3).expand(2, 3, 3).clone(), [(12, 8), (7, 5)], torch.device("cpu"))
    assert isinstance(cams, HostSizedCameraConfig) and cams.size_for(1) == (7, 5) and type(cams.width_for(0)) is int


def test_stock_const_tensor_seam_is_torch_tensor() -> None:
    from trigs.core.trainer import TrigsTrainer

    built = TrigsTrainer(device="cpu")._const_tensor((1.0, 0.5), torch.device("cpu"), torch.float32)
    assert torch.equal(built, torch.tensor((1.0, 0.5)))


def test_const_tensor_equals_torch_tensor_and_is_built_once() -> None:
    from open4dgs.trigs_trainer import Open4dgsTrainer

    trainer = Open4dgsTrainer(device="cpu")
    for value in (0.0, 1.0 / 30.0, 59.0 / 30.0, (0.0, 0.0, 0.0), (1.0, 0.5, 0.25)):
        for dtype in (torch.float32, torch.float64):
            built = trainer._const_tensor(value, torch.device("cpu"), dtype)
            assert torch.equal(built, torch.tensor(value, dtype=dtype)) and built.dtype == dtype
            assert trainer._const_tensor(value, torch.device("cpu"), dtype) is built


@pytest.mark.parametrize("scale", [1.0, 255.0])
def test_mask_rescale_is_the_old_branch(scale: float) -> None:
    gen = torch.Generator().manual_seed(1)
    mask = torch.rand(16, 16, generator=gen) * scale
    old = mask / 255.0 if mask.max() > 1.0 else mask
    new = mask / torch.where(mask.amax() > 1.0, 255.0, 1.0)
    assert torch.equal(old, new)


def test_gpu_loss_sums_equal_the_python_float_sums() -> None:
    """float64 tensor sums of float32 losses, divided and cast to float32, equal torch.tensor(python-float sum / n)."""
    gen = torch.Generator().manual_seed(2)
    for n_group in (1, 3, 8):
        losses = [torch.rand((), generator=gen) * 0.05 for _ in range(n_group)]
        py_sum = 0.0
        t_sum = torch.zeros((), dtype=torch.float64)
        for loss in losses:
            py_sum += float(loss)
            t_sum += loss.double()
        assert torch.equal((t_sum / n_group).float(), torch.tensor(py_sum / n_group))
