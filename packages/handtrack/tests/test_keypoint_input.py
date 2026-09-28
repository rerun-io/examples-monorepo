from dataclasses import replace

import pytest
import torch
from jaxtyping import Float32
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.geometry.letterbox import letterbox_for
from handtrack.hand.pose import generic_hand_model
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.crops import apply_affine, crop_boxes, crop_from_net
from handtrack.labels.heatmaps import DISTANCE_RANGE_MM, decode_distance, decode_heatmaps, render_distance, render_heatmaps
from handtrack.labels.keypoint_input import ZERO_INPUT, add_input_noise, hand_scale, keypoint_input, relative_distances


def test_hand_scale_uses_wrist_centred_rest_geometry() -> None:
    generic: HandModelTorch = generic_hand_model()
    assert hand_scale(generic, generic) == pytest.approx(1.0)
    scaled: HandModelTorch = replace(generic, landmark_rest_positions=generic.landmark_rest_positions * 1.1 + 12.0, joint_rest_positions=generic.joint_rest_positions * 1.1 + 12.0)
    assert hand_scale(scaled, generic) == pytest.approx(1.1)


def test_relative_distance_is_radial_centred_and_in_generic_millimetres() -> None:
    points: Float32[Tensor, '2 3 21 3'] = torch.zeros(2, 3, 21, 3)
    points[..., 0] = torch.linspace(0.1, 0.3, 21)
    values: Float32[Tensor, '2 3 21'] = relative_distances(points, torch.full((2, 3), 2.0))
    torch.testing.assert_close(values[0, 0], torch.linspace(-50.0, 50.0, 21), atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(values.mean(dim=-1), torch.zeros(2, 3), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(relative_distances(points * 1.1, torch.full((2, 3), 2.2)), values, atol=3e-5, rtol=1e-5)


def test_input_is_interleaved_normalized_and_preserves_mirror() -> None:
    points: Float32[Tensor, '1 21 2'] = torch.full((1, 21, 2), 47.5)
    points[0, 0] = torch.tensor([-0.5, 95.5])
    distance: Float32[Tensor, '1 21'] = torch.zeros(1, 21)
    distance[0, 0] = DISTANCE_RANGE_MM
    vector: Float32[Tensor, '1 63'] = keypoint_input(points, distance)
    torch.testing.assert_close(vector[0, :6], torch.tensor([0.0, 1.0, 1.0, 0.5, 0.5, 0.0]))
    points[..., 0] = 95 - points[..., 0]
    torch.testing.assert_close(keypoint_input(points, distance)[0, :3], torch.tensor([1.0, 1.0, 1.0]))
    torch.testing.assert_close(ZERO_INPUT, torch.zeros(63))


def test_seeded_input_noise_has_separate_uv_and_distance_scales() -> None:
    vectors: Float32[Tensor, '2000 63'] = torch.zeros(2000, 63)
    noisy: Float32[Tensor, '2000 63'] = add_input_noise(vectors, torch.Generator().manual_seed(8), 0.02, 0.1)
    torch.testing.assert_close(noisy, add_input_noise(vectors, torch.Generator().manual_seed(8), 0.02, 0.1))
    assert float(noisy[:, 0::3].std()) == pytest.approx(0.02, rel=0.03)
    assert float(noisy[:, 1::3].std()) == pytest.approx(0.02, rel=0.03)
    assert float(noisy[:, 2::3].std()) == pytest.approx(0.1, rel=0.03)
    assert not vectors.any()
    torch.testing.assert_close(add_input_noise(vectors, torch.Generator(), 0.0, 0.0), vectors)


def test_labels_flow_from_source_pixels_to_keynet_features() -> None:
    source: Float32[Tensor, '1 21 2'] = torch.stack((torch.linspace(300.0, 500.0, 21), torch.linspace(500.0, 700.0, 21)), dim=-1)[None]
    net: Float32[Tensor, '1 21 2'] = letterbox_for(1024, 1280).to_net(source)
    circles: Float32[Tensor, '1 3'] = torch.from_numpy(enclosing_circles(net.numpy(), torch.ones(1, 21, dtype=torch.bool).numpy()))
    affine: Float32[Tensor, '1 3 3'] = crop_from_net(crop_boxes(circles), torch.tensor([True]))
    crop: Float32[Tensor, '1 21 2'] = apply_affine(affine, net)
    distances: Float32[Tensor, '1 21'] = torch.linspace(-75.0, 75.0, 21)[None]
    decoded: Float32[Tensor, '1 21 2'] = decode_heatmaps(render_heatmaps(crop))[0]
    torch.testing.assert_close(keypoint_input(decoded, decode_distance(render_distance(distances))), keypoint_input(crop, distances), atol=1e-5, rtol=1e-5)
    recovered: Float32[Tensor, '1 21 2'] = letterbox_for(1024, 1280).from_net(apply_affine(torch.linalg.inv(affine), decoded))
    torch.testing.assert_close(recovered, source, atol=1e-4, rtol=1e-5)


def test_input_noise_preserves_reference_rng_order() -> None:
    vectors: Float32[Tensor, '7 63'] = torch.linspace(-1.0, 1.0, 7 * 63).reshape(7, 63)
    generator: torch.Generator = torch.Generator().manual_seed(31)
    reference_generator: torch.Generator = torch.Generator().manual_seed(31)
    noise: Float32[Tensor, '7 21 3'] = torch.randn((7, 21, 3), generator=reference_generator) * torch.tensor([0.02, 0.02, 0.1])
    torch.testing.assert_close(add_input_noise(vectors, generator, 0.02, 0.1), vectors + noise.reshape(7, 63), atol=1e-6, rtol=0.0)
    assert torch.equal(generator.get_state(), reference_generator.get_state())
