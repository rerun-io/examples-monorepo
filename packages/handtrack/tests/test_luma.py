import torch

from handtrack.data.luma import grey_from_luma


def test_full_range_luma_is_already_grey() -> None:
    y = torch.tensor([[[0, 17, 128, 255]]], dtype=torch.uint8)
    assert grey_from_luma(y, torch.tensor([True])).tolist() == [[[0, 17, 128, 255]]]


def test_limited_range_luma_is_stretched_like_an_rgb_decode() -> None:
    # BT.601/709 limited range: Y 16 is black, 235 is white, and R = G = B = 255 * (Y - 16) / 219 for grey pixels; outside values clip.
    y = torch.tensor([[[16, 235, 126, 0, 255]]], dtype=torch.uint8)
    assert grey_from_luma(y, torch.tensor([False])).tolist() == [[[0, 255, 128, 0, 255]]]


def test_each_image_uses_its_own_range() -> None:
    y = torch.full((2, 1, 2), 235, dtype=torch.uint8)
    assert grey_from_luma(y, torch.tensor([True, False])).tolist() == [[[235, 235]], [[255, 255]]]


def test_rgb_frames_become_bt709_luma_and_grey_pixels_stay_unchanged() -> None:
    from handtrack.data.luma import grey_from_rgb

    rgb = torch.tensor([[[[0, 128, 255, 255]], [[0, 128, 255, 0]], [[0, 128, 255, 0]]]], dtype=torch.uint8)  # 1 x 3 x 1 x 4
    # Pure red: 0.2126 * 255 = 54.2.
    assert grey_from_rgb(rgb).tolist() == [[[0, 128, 255, 54]]]
