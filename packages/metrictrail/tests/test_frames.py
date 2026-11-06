"""Input classification, ViPE frame order, VSLAM-LAB frame linking and the display raster fit."""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import UInt8
from numpy import ndarray

from metrictrail.frames import fit_for_display, input_kind, link_sequence_frames, list_frames, open_frames


def touch_frames(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_list_frames_is_vipe_sorted_order_with_any_case_extension(tmp_path: Path) -> None:
    touch_frames(tmp_path, ["000010.png", "000002.PNG", "notes.txt", "000001.jpg"])
    assert list_frames(tmp_path) == ["000001.jpg", "000002.PNG", "000010.png"]


def test_input_kind(tmp_path: Path) -> None:
    sequence: Path = tmp_path / "rgbd_dataset_freiburg1_desk"
    touch_frames(sequence / "rgb_0", ["1.png"])
    (sequence / "rgb.csv").write_text("ts_rgb_0 (ns),path_rgb_0\n")
    video: Path = tmp_path / "clip.MP4"
    video.write_bytes(b"")
    assert input_kind(sequence) == "vslamlab_sequence"
    assert input_kind(sequence / "rgb_0") == "frame_dir"
    assert input_kind(video) == "video"
    with pytest.raises(ValueError, match="not a frame directory"):
        input_kind(tmp_path / "rgb.csv")


def test_link_sequence_frames_names_the_directory_and_keeps_order(tmp_path: Path) -> None:
    touch_frames(tmp_path / "seq" / "rgb_0", ["1305031102.211214.png", "1305031102.175304.png"])
    names: list[str] = link_sequence_frames(tmp_path / "seq" / "rgb_0", tmp_path / "frames" / "desk")
    assert names == ["1305031102.175304.png", "1305031102.211214.png"]
    assert list_frames(tmp_path / "frames" / "desk") == names
    assert all((tmp_path / "frames" / "desk" / name).is_symlink() for name in names)
    assert link_sequence_frames(tmp_path / "seq" / "rgb_0", tmp_path / "frames" / "desk") == names  # idempotent
    (tmp_path / "frames" / "desk" / names[0]).unlink()
    (tmp_path / "frames" / "desk" / names[0]).write_bytes(b"other")
    with pytest.raises(FileExistsError):
        link_sequence_frames(tmp_path / "seq" / "rgb_0", tmp_path / "frames" / "desk")


def test_open_frames_reads_rgb(tmp_path: Path) -> None:
    import cv2

    bgr: UInt8[ndarray, "4 6 3"] = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[..., 2] = 200  # red in OpenCV's BGR
    cv2.imwrite(str(tmp_path / "000000.png"), bgr)
    source = open_frames(tmp_path)
    rgb: UInt8[ndarray, "h w 3"] = source.reader()(0)
    assert source.kind == "frame_dir" and rgb.shape == (4, 6, 3)
    assert rgb[0, 0].tolist() == [200, 0, 0]


@pytest.mark.parametrize(("policy", "source_hw"), [("resize", (480, 640)), ("cover_crop", (720, 1280)), ("pipeline_default", (300, 1000))])
def test_fit_for_display_lands_on_the_raster(policy: str, source_hw: tuple[int, int]) -> None:
    image: UInt8[ndarray, "h w 3"] = np.full((*source_hw, 3), 7, dtype=np.uint8)
    fitted: UInt8[ndarray, "H W 3"] = fit_for_display(image, (280, 504), policy)  # pyrefly: ignore[bad-argument-type]
    assert fitted.shape == (280, 504, 3)
    assert int(fitted.min()) == int(fitted.max()) == 7
    same: UInt8[ndarray, "h w 3"] = np.zeros((280, 504, 3), dtype=np.uint8)
    assert fit_for_display(same, (280, 504), "unchanged") is same
