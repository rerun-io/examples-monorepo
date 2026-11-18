"""The insta360_io module (packages/insta360-io) as dataforge calls it: the .insv reader and the CPU panorama stitch."""

from __future__ import annotations

import math
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


def test_a_file_that_is_not_an_insv_clip_is_refused(tmp_path: Path) -> None:
    text: Path = tmp_path / "notes.insv"
    text.write_bytes(b"not an mp4 container, and no Insta360 trailer either\n" * 40)
    with pytest.raises(ValueError):
        insta360_io.InsvClip(str(text))
    with pytest.raises(ValueError):
        insta360_io.InsvClip(str(tmp_path / "absent.insv"))


def factory_camera(lens: int) -> insta360_io.MeiCamera:
    """Lens ``lens`` of ``V6`` at the 1920 px track (window crop applied), rounded to float32 as a recording stores it."""
    fx, fy, cx, cy = [(1835.3690185546875, 1835.0146484375, 958.292236328125, 965.8672485351562), (1848.9361572265625, 1849.060302734375, 965.6551513671875, 964.3956909179688)][lens]
    tokens: list[float] = [float(token) for token in V6.split("_")]
    first: int = 1 + 27 * lens
    coefficients: list[float] = [float(np.float32(value)) for value in (tokens[first], math.radians(100.0), *tokens[first + 11 : first + 24])]
    return insta360_io.MeiCamera("mei_v6", fx, fy, cx, cy, coefficients)


BACK_TO_BACK: list[list[float]] = [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.0333], [0.0, 0.0, 0.0, 1.0]]
"""Camera 1 turned 180 deg about camera 0's y, 33.3 mm behind it."""


def test_each_camera_fills_its_own_hemisphere() -> None:
    stitch = insta360_io.PanoramaStitch(factory_camera(0), factory_camera(1), BACK_TO_BACK, (1920, 1920))
    bright: UInt8[ndarray, "h w"] = np.full((1920, 1920), 200, dtype=np.uint8)
    dark: UInt8[ndarray, "h w"] = np.full((1920, 1920), 40, dtype=np.uint8)
    pano: UInt8[ndarray, "960 1920"] = stitch.panorama(bright, dark)
    assert pano.shape == (960, 1920)
    assert pano.dtype == np.uint8
    # Camera 0's axis is the left half's centre, camera 1's the right half's.
    assert pano[480, 480] == 200
    assert pano[480, 1440] == 40
    swapped: UInt8[ndarray, "960 1920"] = stitch.panorama(dark, bright)
    assert swapped[480, 480] == 40


def test_a_camera_states_its_model_and_coefficients_back() -> None:
    camera = factory_camera(1)
    assert (camera.model, camera.fx, camera.cy) == ("mei_v6", 1848.9361572265625, 964.3956909179688)
    assert len(camera.coefficients) == 15


@pytest.mark.parametrize(
    ("model", "coefficients", "message"),
    [
        ("kannala_brandt", [0.0] * 4, "not a Mei model"),
        ("mei_v6", [2.3, 1.7] + [0.0] * 5, "mei_v6 takes 15 coefficients"),
        ("mei_v3", [2.3, 1.7] + [0.0] * 13, "mei_v3 takes 7 coefficients"),
        ("mei_v3", [2.3, float("nan")] + [0.0] * 5, "not a valid Mei camera"),
    ],
)
def test_a_camera_outside_the_mei_models_is_refused_with_the_reason(model: str, coefficients: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        insta360_io.MeiCamera(model, 1835.0, 1835.0, 960.0, 960.0, coefficients)


def test_a_lens_of_the_wrong_size_or_layout_or_a_pose_that_is_not_rigid_is_refused() -> None:
    stitch = insta360_io.PanoramaStitch(factory_camera(0), factory_camera(1), BACK_TO_BACK, (1920, 1920))
    small: UInt8[ndarray, "h w"] = np.zeros((1080, 1920), dtype=np.uint8)
    with pytest.raises(ValueError, match="lens image"):
        stitch.panorama(small, small)
    full: UInt8[ndarray, "h w"] = np.zeros((1920, 1920), dtype=np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        stitch.panorama(np.asfortranarray(full), full)
    with pytest.raises(ValueError, match="uint8"):
        stitch.panorama(full.astype(np.uint16), full)
    sheared: list[list[float]] = [row.copy() for row in BACK_TO_BACK]
    sheared[0][1] = 0.1
    with pytest.raises(ValueError, match="not a rigid transform"):
        insta360_io.PanoramaStitch(factory_camera(0), factory_camera(1), sheared, (1920, 1920))
    with pytest.raises(ValueError, match="below 4x4"):
        insta360_io.PanoramaStitch(factory_camera(0), factory_camera(1), BACK_TO_BACK, (2, 2))


def patched_clip(clip: Path, tmp_path: Path, old: bytes, new: bytes) -> Path:
    """A sparse copy of ``clip`` holding only its head and its trailer end, ``old`` replaced by ``new`` (same length) everywhere.

    The reader opens a clip from those two ends alone; the samples between stay unread holes.
    """
    assert len(old) == len(new)
    size: int = clip.stat().st_size
    with clip.open("rb") as source:
        head: bytes = source.read(4 << 20)
        source.seek(size - (64 << 20))
        tail: bytes = source.read()
    assert old in tail, f"{old!r} is not in the trailer"
    copy: Path = tmp_path / clip.name
    with copy.open("wb") as out:
        out.truncate(size)
        out.write(head)
        out.seek(size - len(tail))
        out.write(tail.replace(old, new))
    return copy


@pytest.mark.integration
@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # Lens 1's z translation of the V6 string becomes infinite.
        (b"_-0.033291_0.95675713", b"_+infinity_0.95675713", "token \"\\+infinity\" is not a finite number"),
        # Lens 0 of the lens strings is not the 200 deg lens class 131.
        (b"_131_2.314940_7149.220", b"_132_2.314940_7149.220", "lens 0 has type 132"),
    ],
)
def test_a_clip_with_a_non_finite_token_or_an_unsupported_lens_is_refused_naming_the_reason(
    insv_clip: Path, tmp_path: Path, old: bytes, new: bytes, message: str
) -> None:
    insta360_io.InsvClip(str(patched_clip(insv_clip, tmp_path, old, old)))  # the sparse copy itself opens
    with pytest.raises(ValueError, match=message):
        insta360_io.InsvClip(str(patched_clip(insv_clip, tmp_path, old, new)))


@pytest.mark.golden
def test_the_clip_reads_as_slam_rs_read_it(insv_clip: Path) -> None:
    """Counts and lens strings of the X4 Air clip, as ``slam_rs._core.InsvClip`` returned them on the base branch (2026-11-17)."""
    clip = insta360_io.InsvClip(str(insv_clip))
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


@pytest.mark.golden
def test_the_lenses_are_mei_cameras_with_the_factory_coefficients(insv_clip: Path) -> None:
    """Lens 0 of the V6 string: xi 2.31494, d0..d12 as written; the V3 lenses carry seven coefficients."""
    clip = insta360_io.InsvClip(str(insv_clip))
    camera = clip.lenses_v6[0].camera
    assert camera.model == "mei_v6"
    tokens: list[float] = [float(token) for token in V6.split("_")]
    assert camera.coefficients == [tokens[1], math.radians(100.0), *tokens[12:25]]
    # 7096.760 px per unit at the 7680 px calibration frame, cropped to 7424 and scaled to the 1920 px track.
    assert (camera.fx, camera.fy) == pytest.approx((7096.76 * 1920 / 7424, 7095.39 * 1920 / 7424), abs=1e-6)
    assert [(lens.camera.model, len(lens.camera.coefficients)) for lens in clip.lenses] == [("mei_v3", 7)] * 2
