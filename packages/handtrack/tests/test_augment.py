import math

import torch

from handtrack.data.augment import DetNetAugment, augment_detnet
from handtrack.data.batches import DetNetBatch

OFF = dict(geometric_probability=0.0, flip_probability=0.0, wide_probability=0.0, noise_probability=0.0, blur_probability=0.0,
           gamma_range=(1.0, 1.0), contrast_range=(1.0, 1.0))


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(120) + 0.5, torch.arange(160) + 0.5, indexing='ij')
    return xx * 4, yy * 4  # pooled pixel centres in net-frame pixels


def _batch(cx: float, cy: float, r: float, copies: int = 1) -> tuple[DetNetBatch, torch.Tensor, torch.Tensor]:
    """A left hand drawn as a bright disk with 21 keypoints on a ring inside it; no right hand."""
    xx, yy = _grid()
    disk = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2).float()
    pooled = (0.1 + 0.8 * disk)[None, None].repeat(copies, 1, 1, 1)
    angles = torch.linspace(0, 2 * math.pi, 22)[:21]
    points = torch.zeros(copies, 2, 21, 2)
    points[:, 0] = torch.stack([cx + 0.9 * r * angles.cos(), cy + 0.9 * r * angles.sin()], -1)
    in_front = torch.zeros(copies, 2, 21, dtype=torch.bool)
    in_front[:, 0] = True
    batch = DetNetBatch(pooled=pooled, circle=torch.tensor([[[cx / 640, cy / 480, r / 640], [0.0, 0.0, 0.0]]]).repeat(copies, 1, 1),
                        presence=torch.tensor([[1.0, 0.0]]).repeat(copies, 1), circle_mask=torch.tensor([[True, False]]).repeat(copies, 1),
                        presence_mask=torch.ones(copies, 2, dtype=torch.bool), dataset=torch.zeros(copies, dtype=torch.int64))
    return batch, points, in_front


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_enabled_with_every_draw_off_is_the_identity() -> None:
    batch, points, in_front = _batch(300.0, 200.0, 40.0)
    out = augment_detnet(batch, points, in_front, DetNetAugment(enabled=True, **OFF), _generator())
    torch.testing.assert_close(out.pooled, batch.pooled, atol=1e-5, rtol=0)
    torch.testing.assert_close(out.circle, batch.circle)
    assert out.circle_mask.tolist() == [[True, False]] and out.presence.tolist() == [[1.0, 0.0]]


def test_a_mirror_flips_the_image_and_moves_the_left_hand_to_the_right_slot() -> None:
    batch, points, in_front = _batch(200.0, 240.0, 40.0)
    out = augment_detnet(batch, points, in_front, DetNetAugment(enabled=True, **{**OFF, 'flip_probability': 1.0}), _generator())
    torch.testing.assert_close(out.pooled, batch.pooled.flip(-1), atol=1e-5, rtol=0)
    assert out.circle_mask.tolist() == [[False, True]] and out.presence.tolist() == [[0.0, 1.0]]
    torch.testing.assert_close(out.circle[0, 1], torch.tensor([1 - 200.0 / 640, 240.0 / 480, 40.0 / 640]))


def test_a_quarter_turn_moves_the_label_with_the_drawn_hand() -> None:
    batch, points, in_front = _batch(400.0, 240.0, 40.0)
    settings = DetNetAugment(enabled=True, **{**OFF, 'geometric_probability': 1.0, 'quarter_turn_probability': 1.0,
                                              'max_rotation_deg': 0.0, 'scale_range': (1.0, 1.0), 'max_shift': 0.0})
    out = augment_detnet(batch, points, in_front, settings, _generator(1))
    xx, yy = _grid()
    bright = out.pooled[0, 0] > 0.5
    centroid = (float(xx[bright].mean()), float(yy[bright].mean()))
    label = (float(out.circle[0, 0, 0]) * 640, float(out.circle[0, 0, 1]) * 480)
    assert abs(label[0] - 320.0) < 1e-3 and abs(abs(label[1] - 240.0) - 80.0) < 1e-3  # +-90 degrees about the frame centre
    assert math.dist(centroid, label) < 4.0 and bool(out.circle_mask[0, 0])


def test_16_9_bars_are_black_and_hands_inside_them_lose_their_label() -> None:
    batch, points, in_front = _batch(320.0, 40.0, 30.0, copies=64)  # a hand near the top edge
    out = augment_detnet(batch, points, in_front, DetNetAugment(enabled=True, **{**OFF, 'wide_probability': 1.0}), _generator(2))
    black_rows = (out.pooled[:, 0] == 0).all(-1).sum(-1)
    assert black_rows.tolist() == [30] * 64  # 480 - 360 net rows = 30 pooled rows, above and below the band
    xx, yy = _grid()
    disk = (((xx - 320.0) ** 2 + (yy - 40.0) ** 2) <= 30.0 ** 2)
    shown = (out.pooled[:, 0][:, disk] > 0.5).float().mean(-1)
    present, absent = out.circle_mask[:, 0], ~out.circle_mask[:, 0] & out.presence_mask[:, 0]
    assert bool((shown[present] > 0.8).all()) and bool((shown[absent] == 0).all())
    assert int(present.sum()) > 0 and int(absent.sum()) > 0


def test_random_geometry_keeps_labels_consistent_with_the_image() -> None:
    batch, points, in_front = _batch(320.0, 240.0, 35.0, copies=256)
    settings = DetNetAugment(enabled=True, **{**OFF, 'geometric_probability': 1.0, 'max_shift': 0.6, 'scale_range': (0.6, 1.6)})
    out = augment_detnet(batch, points, in_front, settings, _generator(3))
    xx, yy = _grid()
    for i in range(256):
        cx, cy, r = (float(v) for v in out.circle[i, 0] * torch.tensor([640.0, 480.0, 640.0]))
        bright = int((out.pooled[i, 0] > 0.5).sum())
        if out.circle_mask[i, 0]:
            inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.8 * r) ** 2
            assert float((out.pooled[i, 0][inside] > 0.5).float().mean()) > 0.9  # the label sits on the drawn hand
        elif out.presence_mask[i, 0]:
            # Absent: every keypoint left the frame; at most a sliver of the disk rim (it is larger than the keypoint ring) remains.
            assert bright <= 0.05 * math.pi * (35.0 / 4) ** 2
    assert 0 < int(out.circle_mask[:, 0].sum()) < 256
