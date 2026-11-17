"""The insta360_io module (packages/insta360-io) as dataforge calls it: the .insv reader and the CPU panorama stitch."""

from __future__ import annotations

import os
from pathlib import Path

import insta360_io
import numpy as np
import pytest
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

V6: str = (
    "2_2.314940_7096.760_7095.390_3834.830_3864.120_0.119_0.407_90.629_0.000000_0.000000_0.000000_0.93861496_-1.68402135_"
    "2.19166017_3.85762072_0.00000000_-0.00495962_0.00156076_-0.00330722_0.00397486_0.00827898_-0.00125315_0.02416405_"
    "-0.00143547_15360_7680_131_2.314940_7149.220_7149.700_11543.300_3858.430_-0.212_0.339_88.279_0.000132_-0.000043_"
    "-0.033291_0.95675713_-1.80496192_2.88544273_1.76948690_0.00000000_0.00131924_0.00023973_0.00279902_-0.00570679_"
    "0.00069985_0.00064811_0.00799088_0.02677096_15360_7680_131_394240"
)
"""The X4 Air clip's V6 factory string (clip VID_20261116_125200_00_003)."""
CROP: tuple[int, int, int, int, int, int] = (7680, 7680, 7424, 7424, 0, 0)

CLIP: Path | None = Path(clip) if (clip := os.environ.get("SLAM_RS_INSV")) else None
"""The X4 Air clip VID_20261116_125200_00_003.insv, from ``SLAM_RS_INSV``; the golden test skips without it."""


def test_a_file_that_is_not_an_insv_clip_is_refused(tmp_path: Path) -> None:
    text: Path = tmp_path / "notes.insv"
    text.write_bytes(b"not an mp4 container, and no Insta360 trailer either\n" * 40)
    with pytest.raises(ValueError):
        insta360_io.InsvClip(str(text))
    with pytest.raises(ValueError):
        insta360_io.InsvClip(str(tmp_path / "absent.insv"))


def test_each_lens_fills_its_own_hemisphere() -> None:
    stitch = insta360_io.PanoramaStitch(V6, CROP, (1920, 1920))
    bright: UInt8[ndarray, "h w"] = np.full((1920, 1920), 200, dtype=np.uint8)
    dark: UInt8[ndarray, "h w"] = np.full((1920, 1920), 40, dtype=np.uint8)
    pano: UInt8[ndarray, "960 1920"] = stitch.panorama(bright, dark)
    assert pano.shape == (960, 1920)
    assert pano.dtype == np.uint8
    # Lens 0's axis is the left half's centre, lens 1's the right half's.
    assert pano[480, 480] == 200
    assert pano[480, 1440] == 40
    swapped: UInt8[ndarray, "960 1920"] = stitch.panorama(dark, bright)
    assert swapped[480, 480] == 40


def test_a_lens_of_the_wrong_size_or_layout_or_a_bad_string_is_refused() -> None:
    stitch = insta360_io.PanoramaStitch(V6, CROP, (1920, 1920))
    small: UInt8[ndarray, "h w"] = np.zeros((1080, 1920), dtype=np.uint8)
    with pytest.raises(ValueError, match="lens image"):
        stitch.panorama(small, small)
    full: UInt8[ndarray, "h w"] = np.zeros((1920, 1920), dtype=np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        stitch.panorama(np.asfortranarray(full), full)
    with pytest.raises(ValueError, match="uint8"):
        stitch.panorama(full.astype(np.uint16), full)
    with pytest.raises(ValueError):
        insta360_io.PanoramaStitch("2_not_a_lens_string", CROP, (1920, 1920))
    with pytest.raises(ValueError, match="below 4x4"):
        insta360_io.PanoramaStitch(V6, CROP, (2, 2))


@pytest.mark.golden
def test_the_clip_reads_as_slam_rs_read_it() -> None:
    """Counts and lens strings of the X4 Air clip, as ``slam_rs._core.InsvClip`` returned them on the base branch (2026-11-17)."""
    if CLIP is None or not CLIP.is_file():
        pytest.skip(f"asset VID_20261116_125200_00_003.insv is absent: set SLAM_RS_INSV to the clip (now {CLIP})")
    clip = insta360_io.InsvClip(str(CLIP))
    assert (clip.camera_type, clip.resolution, clip.lens_track, clip.window_crop) == ("Insta360 X4 Air", (1920, 1920), (1, 0), CROP)
    assert clip.num_frames == 7724
    times: Int64[ndarray, " n"]
    accel: Float64[ndarray, "n 3"]
    times, accel, _ = clip.imu()
    assert times.shape == (257600,) and accel.shape == (257600, 3)
    strings: dict[str, str] = dict(clip.factory_strings)
    assert sorted(strings) == ["v1", "v2", "v3", "v6"]
    assert strings["v6"] == V6
    assert len(clip.lenses) == len(clip.lenses_v6) == 2
