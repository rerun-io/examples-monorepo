"""KeyNet from the catalog stream: palm centre, coco3d priors, per-draw augmentation and KeyNet echo pools (no catalog, no GPU)."""

from collections import Counter
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from simplecv.umetrack_temp.generic_hand_model_torch import LANDMARK
from torch import Tensor

from handtrack.data.batches import CropKind, KeyNetBatch
from handtrack.data.catalog import _generic_model
from handtrack.data.echo import EchoStream
from handtrack.data.keynet_rerun import (
    DrawAugment,
    _point_priors,
    coco_landmarks,
    coherent_prior_noise,
    keynet_finish,
    palm_centre,
    palm_coefficients,
)
from handtrack.data.stream import KeyNetSamples, empty_keynet_samples
from handtrack.labels.coco2d import COCO_SLOTS


def _rest() -> Tensor:
    return _generic_model().landmark_rest_positions.float()


def _rotation(angle: float) -> Tensor:
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def test_palm_centre_reproduces_the_generic_hand_and_follows_a_rigid_motion_and_scale() -> None:
    rest: Tensor = _rest()
    coefficients: Tensor = palm_coefficients(rest)
    torch.testing.assert_close(palm_centre(rest, coefficients), rest[int(LANDMARK.PALM_CENTER)], atol=1e-3, rtol=0)
    moved: Tensor = 0.9 * rest @ _rotation(0.7).T + torch.tensor([10.0, -5.0, 300.0])
    torch.testing.assert_close(palm_centre(moved, coefficients), moved[int(LANDMARK.PALM_CENTER)], atol=1e-3, rtol=0)


def test_coco_landmarks_map_the_slots_and_build_the_palm_of_both_hands() -> None:
    """The left slots hold the generic (left) rest hand, the right slots its MIRROR image (a real right hand), moved and rotated."""
    rest_m: Tensor = _rest() / 1000.0
    mirrored: Tensor = (rest_m * torch.tensor([-1.0, 1.0, 1.0])) @ _rotation(1.1).T + torch.tensor([0.3, -0.1, 0.5])
    xyz: Tensor = torch.full((1, 133, 3), torch.nan)
    for slot, landmark in COCO_SLOTS.items():
        xyz[0, slot] = rest_m[landmark]
        xyz[0, slot + 21] = mirrored[landmark]
    confidence: Tensor = torch.ones((1, 133))
    points, known = coco_landmarks(xyz, confidence, palm_coefficients(_rest()))
    assert bool(known.all())
    torch.testing.assert_close(points[0, 0], rest_m, atol=1e-5, rtol=0)
    torch.testing.assert_close(points[0, 1], mirrored, atol=1e-5, rtol=0)  # landmark 20 included: on the palm side of the right hand


def test_coco_landmarks_mark_unconfident_points_unknown_and_drop_the_palm_with_them() -> None:
    xyz: Tensor = torch.zeros((1, 133, 3))
    confidence: Tensor = torch.ones((1, 133))
    confidence[0, 91] = 0.0  # the left wrist
    _, known = coco_landmarks(xyz, confidence, palm_coefficients(_rest()))
    assert not bool(known[0, 0, int(LANDMARK.WRIST_JOINT)]) and not bool(known[0, 0, int(LANDMARK.PALM_CENTER)])
    assert bool(known[0, 1].all())


def test_point_priors_extrapolate_from_the_two_previous_steps_and_fall_back_to_the_last() -> None:
    frames: np.ndarray = np.arange(30, dtype=np.int64)
    points: Tensor = torch.arange(30, dtype=torch.float32)[:, None, None, None].expand(30, 2, 21, 3).clone()
    valid: Tensor = torch.ones((30, 2), dtype=torch.bool)
    valid[3, 0] = False
    extrapolated, extrapolated_valid, stale, stale_valid = _point_priors(points, valid, frames, step=1)
    assert float(extrapolated[10, 0, 0, 0]) == 10.0  # 2 * 9 - 8
    assert float(extrapolated[5, 0, 0, 0]) == 4.0  # frame 3 is missing: the last pose as it is
    assert not bool(extrapolated_valid[4, 0]) and bool(extrapolated_valid[4, 1])
    assert not bool(extrapolated_valid[0].any())
    assert bool(stale_valid[25].all()) and float(stale[25, 0, 0, 0]) == 5.0 and not bool(stale_valid[19].any())


def test_coherent_prior_noise_keeps_untracked_priors_zero_and_moves_whole_hands() -> None:
    generator = torch.Generator().manual_seed(0)
    keypoints: Tensor = torch.rand((64, 63))
    keypoints[:8] = 0.0
    noisy: Tensor = coherent_prior_noise(keypoints, generator, DrawAugment(prior_probability=1.0))
    assert bool((noisy[:8] == 0).all())
    moved = noisy[8:].reshape(-1, 21, 3) - keypoints[8:].reshape(-1, 21, 3)
    d_offset = moved[..., 2]
    torch.testing.assert_close(d_offset, d_offset[:, :1].expand_as(d_offset))  # one d offset per hand
    assert bool((moved[..., :2].abs().sum(dim=(1, 2)) > 0).all())


def _samples(n: int, dataset: int = 0) -> KeyNetSamples:
    samples: KeyNetSamples = empty_keynet_samples(n, torch.device("cpu"))
    crops: Tensor = torch.full((n, 96, 96), 128, dtype=torch.uint8)
    crops[:, 0, 0] = torch.arange(n, dtype=torch.uint8)
    points: Tensor = torch.rand((n, 21, 2)) * 80 + 8
    kind: Tensor = torch.where(torch.arange(n) % 4 == 0, int(CropKind.DRIFT), int(CropKind.POSITIVE))
    return KeyNetSamples(crops=crops, points_crop=points, d_rel_mm=torch.zeros((n, 21)), keypoints=torch.rand((n, 63)), presence=(kind == 0).float(),
                         kind=kind, dataset=torch.full((n,), dataset), crop_from_net=torch.eye(3).expand(n, 3, 3).clone(), other_inside=samples.other_inside,
                         visible=samples.visible, pinch_contact_mm=samples.pinch_contact_mm)


def test_keynet_finish_renders_targets_only_for_positives_and_keeps_crops_in_range() -> None:
    batch: KeyNetBatch = keynet_finish(DrawAugment())(_samples(32), torch.Generator().manual_seed(1))
    assert batch.crops.shape == (32, 1, 96, 96) and float(batch.crops.min()) >= 0.0 and float(batch.crops.max()) <= 1.0
    negative: Tensor = ~batch.positive
    assert bool((batch.heatmaps[negative] == 0).all()) and bool((batch.heatmaps[batch.positive].amax(dim=(-1, -2)) > 0.3).all())
    assert batch.presence_mask.all()


def test_keynet_finish_without_augmentation_is_the_stored_crop() -> None:
    samples: KeyNetSamples = _samples(8)
    batch: KeyNetBatch = keynet_finish(None)(samples, torch.Generator().manual_seed(1))
    torch.testing.assert_close(batch.crops[:, 0], samples.crops.float() / 255.0)
    torch.testing.assert_close(batch.keypoints, samples.keypoints)


def test_draw_augment_rejects_bad_probabilities() -> None:
    with pytest.raises(ValueError):
        DrawAugment(blur_probability=1.5)


def _keynet_chunks(dataset: int, crops: int) -> Iterator[KeyNetSamples]:
    for start in range(0, crops, 16):
        yield _samples(min(16, crops - start), dataset)


def test_echo_stream_serves_keynet_batches_split_equally_across_datasets() -> None:
    producers = [lambda epoch, d=d: _keynet_chunks(d, 400) for d in range(4)]
    stream = EchoStream(producers, batch_size=32, pool_capacity=256, min_fill=32, max_echo=2, device=torch.device("cpu"), seed=0,
                        finish=keynet_finish(DrawAugment()), keynet=True)
    stream.start_epoch(0)
    batches: list[KeyNetBatch] = []
    while (batch := stream.next_keynet_batch()) is not None:
        batches.append(batch)
    stream.close()
    assert batches
    for batch in batches:
        assert Counter(batch.dataset.tolist()) == {0: 8, 1: 8, 2: 8, 3: 8}
    with pytest.raises(RuntimeError):
        stream.next_detnet_batch()


_RACE = """
import threading, torch
from handtrack.data.keynet_rerun import warm_linalg
device = torch.device("cuda")
warm_linalg(device)
errors = []
barrier = threading.Barrier(8)
def work():
    barrier.wait()
    try:
        for _ in range(20):
            eye = torch.eye(3, device=device).expand(16, 3, 3).contiguous()
            torch.linalg.inv(eye); torch.linalg.inv_ex(eye); torch.linalg.cross(eye[:, 0], eye[:, 1])
    except Exception as error:
        errors.append(repr(error))
threads = [threading.Thread(target=work) for _ in range(8)]
[t.start() for t in threads]; [t.join() for t in threads]
print("ERRORS", errors)
assert not errors
"""


@pytest.mark.integration
def test_producer_threads_can_call_linalg_at_once_after_warm_up() -> None:
    """Regression (2026-11-09): eight threads hit torch.linalg at the same moment in a fresh process, after warm_linalg."""
    import subprocess
    import sys

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    result = subprocess.run([sys.executable, "-c", _RACE], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
