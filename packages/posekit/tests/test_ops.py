"""CPU-runnable tests for posekit GPU ops (geometry and decoders)."""

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

from posekit.ops.crops import CropBatch, CropSpec, bbox_xyxy_to_center_scale, crop_coords_to_image, crop_frames
from posekit.ops.decode import decode_classic_heatmaps, decode_simcc, decode_udp_heatmaps
from posekit.ops.letterbox import letterbox_frames
from posekit.ops.yolox import decode_yolox_head_outputs
from posekit.skeletons import COCO_17, COCO_133, HAND_21


def test_skeleton_registry() -> None:
    assert COCO_133.num_keypoints == 133
    assert COCO_17.num_keypoints == 17
    assert HAND_21.num_keypoints == 21
    for skeleton in (COCO_17, COCO_133, HAND_21):
        for a, b in skeleton.links:
            assert 0 <= a < skeleton.num_keypoints
            assert 0 <= b < skeleton.num_keypoints


def test_bbox_center_scale_aspect_expansion() -> None:
    bboxes: Float[Tensor, "1 4"] = torch.tensor([[10.0, 20.0, 110.0, 40.0]])
    centers, scales = bbox_xyxy_to_center_scale(bboxes, aspect_wh=0.75, padding=1.0)
    assert torch.allclose(centers[0], torch.tensor([60.0, 30.0]))
    # Wide box: width dominates, height expands to width / aspect.
    assert torch.allclose(scales[0], torch.tensor([100.0, 100.0 / 0.75]))


def test_crop_constant_region_and_roundtrip() -> None:
    frames: UInt8[Tensor, "1 64 48 3"] = torch.full((1, 64, 48, 3), 77, dtype=torch.uint8)
    spec = CropSpec(input_size=(24, 32), padding=1.0, align="udp", mean_rgb=None, std_rgb=None)
    batch: CropBatch = crop_frames(
        frames,
        frame_indices=torch.tensor([0]),
        bboxes_xyxy=torch.tensor([[8.0, 8.0, 40.0, 56.0]]),
        spec=spec,
    )
    assert batch.inputs.shape == (1, 3, 32, 24)
    assert torch.allclose(batch.inputs, torch.full_like(batch.inputs, 77.0))
    # The crop-space center maps back to the bbox center.
    center_crop: Float[Tensor, "1 1 2"] = torch.tensor([[[12.0, 16.0]]])
    xy_image: Float[Tensor, "1 1 2"] = crop_coords_to_image(
        center_crop, centers=batch.centers, scales=batch.scales, input_size=spec.input_size
    )
    assert torch.allclose(xy_image[0, 0], torch.tensor([24.0, 32.0]), atol=1e-4)


def test_crop_empty_boxes() -> None:
    frames: UInt8[Tensor, "1 32 32 3"] = torch.zeros((1, 32, 32, 3), dtype=torch.uint8)
    spec = CropSpec(input_size=(16, 16), padding=1.25)
    batch: CropBatch = crop_frames(
        frames, frame_indices=torch.empty((0,), dtype=torch.long), bboxes_xyxy=torch.empty((0, 4)), spec=spec
    )
    assert batch.inputs.shape == (0, 3, 16, 16)


def test_letterbox_shapes_and_padding() -> None:
    frames: UInt8[Tensor, "2 30 40 3"] = torch.zeros((2, 30, 40, 3), dtype=torch.uint8)
    inputs, ratios = letterbox_frames(frames, output_size=(64, 64), pad_value=114.0)
    assert inputs.shape == (2, 3, 64, 64)
    assert torch.allclose(ratios, torch.tensor([1.6, 1.6]))
    # Region outside the resized 48x64 area keeps the pad value.
    assert torch.allclose(inputs[:, :, 60:, :], torch.full_like(inputs[:, :, 60:, :], 114.0))


def test_decode_simcc_onehot() -> None:
    simcc_x: Float[Tensor, "1 2 96"] = torch.zeros((1, 2, 96))
    simcc_y: Float[Tensor, "1 2 128"] = torch.zeros((1, 2, 128))
    simcc_x[0, 0, 40] = 5.0
    simcc_y[0, 0, 60] = 7.0
    simcc_x[0, 1, 10] = 1.0
    simcc_y[0, 1, 20] = 1.0
    xy, scores = decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0)
    assert torch.allclose(xy[0, 0], torch.tensor([20.0, 30.0]))
    assert torch.allclose(scores[0, 0], torch.tensor(6.0))


def test_decode_udp_heatmap_peak() -> None:
    heatmap_w, heatmap_h = 48, 64
    yy, xx = torch.meshgrid(torch.arange(heatmap_h).float(), torch.arange(heatmap_w).float(), indexing="ij")
    peak_x, peak_y = 20.0, 33.0
    gaussian: Float[Tensor, "hm_h hm_w"] = torch.exp(-((xx - peak_x) ** 2 + (yy - peak_y) ** 2) / (2 * 2.0**2))
    heatmaps: Float[Tensor, "1 1 hm_h hm_w"] = gaussian[None, None]
    xy, scores = decode_udp_heatmaps(heatmaps, input_size=(192, 256), heatmap_size=(heatmap_w, heatmap_h), blur_kernel_size=11)
    expected_x: float = peak_x * 192.0 / (heatmap_w - 1)
    expected_y: float = peak_y * 256.0 / (heatmap_h - 1)
    assert torch.allclose(xy[0, 0], torch.tensor([expected_x, expected_y]), atol=1.0)
    assert float(scores[0, 0]) > 0.9


def test_decode_yolox_head_outputs_threshold_and_nms() -> None:
    boxes: Float[Tensor, "1 3 4"] = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [40.0, 40.0, 50.0, 50.0]]])
    scores: Float[Tensor, "1 3"] = torch.tensor([[0.9, 0.8, 0.2]])
    detections = decode_yolox_head_outputs(boxes, scores, resize_ratios=torch.tensor([2.0]), score_thr=0.5, nms_thr=0.45)
    # Overlapping pair collapses to the higher-score box; low-score box dropped; coords rescaled by ratio.
    assert detections.num_detections == 1
    assert torch.allclose(detections.xyxy[0], torch.tensor([0.0, 0.0, 5.0, 5.0]))
    assert torch.allclose(detections.scores[0], torch.tensor(0.9))
    assert int(detections.frame_indices[0]) == 0

def test_decode_classic_heatmaps_peak_and_quarter_shift() -> None:
    heatmaps: Float[Tensor, "1 1 16 12"] = torch.zeros((1, 1, 16, 12))
    heatmaps[0, 0, 10, 5] = 1.0
    heatmaps[0, 0, 10, 6] = 0.5  # neighbor gradient pulls +0.25 px in x
    xy, scores = decode_classic_heatmaps(heatmaps, input_size=(48, 64), heatmap_size=(12, 16))
    assert float(scores[0, 0]) == 1.0
    # (5 + 0.25) * 4 in x, 10 * 4 in y (symmetric neighbors -> no y shift).
    assert torch.allclose(xy[0, 0], torch.tensor([21.0, 40.0]))
