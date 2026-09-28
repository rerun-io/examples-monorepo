import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from torch import Tensor

from handtrack.labels.crops import (
    CropJitter,
    apply_affine,
    boundary_occlusion,
    count_inside_crop,
    crop_boxes,
    crop_from_net,
    cut_crops,
    sample_jitter,
    scale_intensity,
)


def test_box_edges_mirror_and_inverse() -> None:
    boxes: Float32[Tensor, '2 4'] = torch.tensor([[100.0, 200.0, 196.0, 296.0]]).repeat(2, 1)
    points: Float32[Tensor, '2 3 2'] = torch.tensor([[[100.0, 200.0], [196.0, 296.0], [148.0, 248.0]]]).repeat(2, 1, 1)
    affine: Float32[Tensor, '2 3 3'] = crop_from_net(boxes, torch.tensor([False, True]))
    mapped: Float32[Tensor, '2 3 2'] = apply_affine(affine, points)
    torch.testing.assert_close(mapped[0], torch.tensor([[-0.5, -0.5], [95.5, 95.5], [47.5, 47.5]]))
    torch.testing.assert_close(mapped[1, :, 0], 95.0 - mapped[0, :, 0])
    torch.testing.assert_close((mapped[1, :, 0] + 0.5) / 96, 1 - (mapped[0, :, 0] + 0.5) / 96)
    torch.testing.assert_close(apply_affine(torch.linalg.inv(affine), mapped), points)
    torch.testing.assert_close(crop_boxes(torch.tensor([[148.0, 248.0, 40.0]])), boxes[:1])


def test_neutral_jitter_and_known_rotation_shift_scale() -> None:
    boxes: Float32[Tensor, '1 4'] = torch.tensor([[0.0, 0.0, 96.0, 96.0]])
    mirror: Bool[Tensor, '1'] = torch.tensor([False])
    neutral: CropJitter = CropJitter(torch.zeros(1), torch.ones(1), torch.zeros(1, 2))
    torch.testing.assert_close(crop_from_net(boxes, mirror, neutral), crop_from_net(boxes, mirror))
    jitter: CropJitter = CropJitter(torch.tensor([torch.pi / 2]), torch.tensor([2.0]), torch.tensor([[0.25, 0.0]]))
    # Shifted centre (72,48), positive rotation turns crop axes clockwise in image coordinates.
    points: Float32[Tensor, '1 2 2'] = torch.tensor([[[72.0, 48.0], [72.0, 144.0]]])
    torch.testing.assert_close(apply_affine(crop_from_net(boxes, mirror, jitter), points), torch.tensor([[[47.5, 47.5], [95.5, 47.5]]]))


def test_cut_crops_samples_selected_frames_and_zero_padding() -> None:
    frames: UInt8[Tensor, '2 480 640'] = torch.zeros(2, 480, 640, dtype=torch.uint8)
    frames[1, 220:240, 110:130] = 255
    boxes: Float32[Tensor, '3 4'] = torch.tensor([[99.5, 199.5, 195.5, 295.5], [99.5, 199.5, 195.5, 295.5], [-96.5, -96.5, -0.5, -0.5]])
    crops: Float32[Tensor, '3 1 96 96'] = cut_crops(frames, torch.tensor([1, 0, 1]), crop_from_net(boxes, torch.tensor([False, False, False])))
    expected: Float32[Tensor, '3 1 96 96'] = torch.zeros_like(crops)
    expected[0, 0, 20:40, 10:30] = 1.0
    torch.testing.assert_close(crops, expected, atol=4e-5, rtol=0.0)


def test_inside_crop_excludes_behind_nonfinite_and_upper_edges() -> None:
    points: Float32[Tensor, '1 6 2'] = torch.tensor([[[0.0, 0.0], [95.99, 95.99], [96.0, 1.0], [-0.1, 1.0], [1.0, 1.0], [float('nan'), 0.0]]])
    assert count_inside_crop(points, torch.tensor([[True, True, True, True, False, True]])).tolist() == [2]


def test_seeded_jitter_bounds_and_neutral_draws() -> None:
    jitter: CropJitter = sample_jitter(100, torch.Generator().manual_seed(7), 'cpu', 0.3, (0.8, 1.2), 0.1)
    repeated: CropJitter = sample_jitter(100, torch.Generator().manual_seed(7), 'cpu', 0.3, (0.8, 1.2), 0.1)
    torch.testing.assert_close(jitter.rotation, repeated.rotation)
    assert jitter.rotation.abs().max() <= 0.3
    assert jitter.scale.min() >= 0.8 and jitter.scale.max() <= 1.2
    assert jitter.shift.abs().max() <= 0.1
    neutral: CropJitter = sample_jitter(2, torch.Generator(), 'cpu', 0.0, (1.0, 1.0), 0.0)
    torch.testing.assert_close(neutral.rotation, torch.zeros(2))
    torch.testing.assert_close(neutral.scale, torch.ones(2))
    torch.testing.assert_close(neutral.shift, torch.zeros(2, 2))


def test_boundary_rectangles_and_probability() -> None:
    assert not boundary_occlusion(10, torch.Generator(), 'cpu', 0.0, 0.3).any()
    masks: Bool[Tensor, '100 96 96'] = boundary_occlusion(100, torch.Generator().manual_seed(18), 'cpu', 1.0, 0.3)
    assert masks.flatten(1).any(dim=1).all()
    assert (masks[:, 0].any(dim=1) | masks[:, -1].any(dim=1) | masks[:, :, 0].any(dim=1) | masks[:, :, -1].any(dim=1)).all()
    for mask in masks:
        occupied: Int64[Tensor, 'n 2'] = torch.nonzero(mask)
        height: int = int(occupied[:, 0].max() - occupied[:, 0].min() + 1)
        width: int = int(occupied[:, 1].max() - occupied[:, 1].min() + 1)
        assert int(mask.sum()) == height * width
        assert min(height, width) <= 28
    assert not boundary_occlusion(2, torch.Generator(), 'cpu', 1.0, 0.0).any()


def test_intensity_is_one_factor_per_image_and_clamped() -> None:
    images: Float32[Tensor, '8 1 3 3'] = torch.full((8, 1, 3, 3), 0.5)
    output: Float32[Tensor, '8 1 3 3'] = scale_intensity(images, torch.Generator().manual_seed(3), 0.5, 1.5)
    assert output.min() >= 0.25 and output.max() <= 0.75
    assert output[:, 0, 0, 0].unique().numel() == 8
    torch.testing.assert_close(output, output[:, :, :1, :1].expand_as(output))
    torch.testing.assert_close(scale_intensity(images, torch.Generator(), 3.0, 3.0), torch.ones_like(images))
