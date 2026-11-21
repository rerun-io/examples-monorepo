"""The insta360_io package (packages/insta360-io) as dataforge calls it: the clip, the reader and the CPU panorama stitch."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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
    ],
)
def test_a_clip_with_a_non_finite_token_is_refused_naming_the_reason(
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


@pytest.mark.integration
def test_a_lens_of_a_type_without_a_field_entry_gets_the_190_degree_default(insv_clip: Path, tmp_path: Path) -> None:
    """Lens 0 of the V6 string becomes type 132 (no entry in the field table): it opens with a 95 degree half field, and
    the rig, whose cameras are the V6 lenses, gets no device profile although the V3 string still states 131 twice."""
    patched: Path = patched_clip(insv_clip, tmp_path, b"_131_2.314940_7149.220", b"_132_2.314940_7149.220")
    clip = insta360_io.InsvClip(str(patched))
    assert clip.lenses_v6[0].camera.coefficients[1] == pytest.approx(math.radians(95.0))
    assert clip.lenses_v6[1].camera.coefficients[1] == pytest.approx(math.radians(100.0))
    rig = insta360_io.InsvRig(str(patched), READER_FRAMES)
    assert (rig.profile_name, rig.pano_from_imu) == (None, None)
    assert "lenses of types 132 and 131" in rig.profile_note


# ── InsvReader ────────────────────────────────────────────────────────────

READER_FRAMES: int = 90
"""Framesets the reader tests read: three GOPs on the X4 Air, one and a half on the X5."""


def test_a_clip_the_rig_cannot_open_is_refused(tmp_path: Path) -> None:
    text: Path = tmp_path / "notes.insv"
    text.write_bytes(b"not an mp4 container\n" * 40)
    with pytest.raises(ValueError):
        insta360_io.InsvReader(text)


def chunk_shapes(stream: object, entity: str) -> list[tuple[bool, list[str], list[tuple[str, str]]]]:
    """``(static, timelines, [(component, arrow type)])`` of each chunk at ``entity``, in stream order."""
    import rerun.chunk as rrc

    shapes: list[tuple[bool, list[str], list[tuple[str, str]]]] = []
    for chunk in stream:  # pyrefly: ignore[not-iterable]
        assert isinstance(chunk, rrc.Chunk)
        if chunk.entity_path == entity:
            schema = chunk.to_record_batch().schema
            shapes.append((chunk.is_static, chunk.timeline_names, [(f.name, str(f.type)) for f in schema if not f.name.startswith("rerun.")]))
    return shapes


@pytest.mark.integration
def test_each_lens_video_has_the_chunk_shapes_of_rerun_mp4_reader(insv_clip: Path, tmp_path: Path) -> None:
    """Per lens, as ``Mp4Reader`` emits an HEVC mp4: a static codec chunk, one sample chunk per GOP, one keyframe chunk.

    The reference is ``Mp4Reader`` itself on a three-GOP HEVC clip encoded here, compared type for type.
    """
    import subprocess

    import pyarrow as pa
    import rerun as rr
    from rerun.experimental import Mp4Reader

    mp4: Path = tmp_path / "gops.mp4"
    encode: list[str] = ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=128x64:rate=30", "-t", "3", "-c:v", "libx265"]
    subprocess.run([*encode, "-x265-params", "bframes=0:keyint=30:log-level=0", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    expected = chunk_shapes(Mp4Reader(mp4, entity_path="/v").stream(), "/v")
    reader = insta360_io.InsvReader(insv_clip, frame_limit=READER_FRAMES)
    assert reader.entity_path == str(insv_clip.absolute())
    chunks = reader.stream().to_chunks()
    for lens in range(2):
        entity: str = f"{reader.entity_path}/cam_{lens:02d}/pinhole/video"
        assert chunk_shapes(chunks, entity) == expected
        ours = [c.to_record_batch() for c in chunks if c.entity_path == entity]
        assert ours[0].column("VideoStream:codec").to_pylist() == [[rr.VideoCodec.H265.value]]
        gop_starts: list[int] = [batch.column("video").cast(pa.int64())[0].as_py() for batch in ours[1:-1]]
        assert [len(batch) for batch in ours[1:-1]] == [30, 30, 30]
        assert ours[-1].column("video").cast(pa.int64()).to_pylist() == gop_starts
        assert ours[-1].column("VideoStream:is_keyframe").to_pylist() == [[True]] * 3
        # Annex B with the parameter sets on the IRAP: VPS, SPS, PPS, then the IDR/CRA slice.
        first: bytes = bytes(ours[1].column("VideoStream:sample")[0].as_py()[0])
        assert first.startswith(b"\x00\x00\x00\x01\x40\x01")


GOLDEN: dict[str, dict[str, str]] = {
    "x4-air": {
        "lens0_samples": "6d0f685ebb4bf1f80056c43f53a5af35f096df513328991a78a3c783aa07c2df",
        "lens1_samples": "d6ba3d7ac26e54f974058a47bfa3d1ef8a68d08c91c56ffcfc1423cf9a762819",
        "lens0_sample_lengths": "e40f4055cbddcd3a979f2e5137c53dda4dafc424738ee8a116b95612146a567b",
        "lens1_sample_lengths": "dcb3b64b50ae0ce93e2579f06172e6a30b4f9cd204f3d71f48461e12acc0098b",
        "frame_times": "d7a487ba4ec123477415d2677dcb98040c3b5a25d98b36d075560a2d472e70f2",
        "lens0_exposure": "0fda121e4860f8931120d0e236793502ad7a28784a1193845d0c20b890dfe6d8",
        "lens1_exposure": "0fda121e4860f8931120d0e236793502ad7a28784a1193845d0c20b890dfe6d8",
        "imu_times": "8a06cd3f6f23ff4cf14ad4a26349bc8f7b6cafa61ed6cafedc110195dc0eb700",
        "gyro": "1837b9859847af7fa76e6ff4c79665e70beb3e3264f2c0087d63acb5cb89a586",
        "accel": "bca134114d15d3c27806976b4db88defda698ae1a40954c588d3f06f197d0565",
    },
    "x5": {
        "lens0_samples": "039214f0b3a4ae8cd53e8a4d92521fb9dc5af94b583bb3e95968a27a7544d3c9",
        "lens1_samples": "7a740602c5e6de7876a966886828fcca320ad73a00d6e9b782b1646b30e5ad1d",
        "lens0_sample_lengths": "fc08d795802907adc6bfec5a9064d78afff70ba0d05e737a0e1f271ec52bf8cc",
        "lens1_sample_lengths": "f4520321d16b04e4fdfb727490a0d894879c84060250e99a46901b630f027625",
        "frame_times": "e4fbe34cac1471cdd9d2c93235452bf320e63d6121023d8832efbabf2e00b1a9",
        "lens0_exposure": "8b253eb6b3d79e2b01baeb29a715559104bacfe641cb39e4f02c64f9c451e6ab",
        "lens1_exposure": "8b253eb6b3d79e2b01baeb29a715559104bacfe641cb39e4f02c64f9c451e6ab",
        "imu_times": "17887a82cf7f29786516da262a12141530ed8daeb6652f56317e26136eba6305",
        "gyro": "dea90afff2e59c34b4f3398e63cb71acb310c047dc8e97f40d02c18b1cae8c97",
        "accel": "dca348101a4680794312b31574b79513675a7f8ecab3a075be2e0b030c25d0fe",
    },
}
"""Per-signal sha256 of the first 90 framesets: the samples' bytes in time order, their lengths and the frame times
(little-endian int64, ns on the reader's timeline), the exposures (float64, s), and the IMU up to 1 s past the last
frame (times int64; gyro and accel float64, row-major ``n x 3``). The sample bytes, first and last frame time and IMU
row count are the pre-reader converter's (2026-11-18); the other digests were frozen from the reader at 77065551a,
which reproduced those."""


def digest(values: object, dtype: str) -> str:
    """sha256 of ``values`` as a C-ordered array of ``dtype``."""
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=dtype)).tobytes()).hexdigest()


@pytest.mark.golden
@pytest.mark.parametrize("model", ["x4-air", "x5"])
def test_the_reader_reproduces_the_pre_reader_conversion(model: str, request: pytest.FixtureRequest) -> None:
    """Samples, sample lengths, frame times, exposures and IMU of the first 90 framesets, each bit for bit, on the X4
    Air and the X5."""
    import hashlib

    import pyarrow as pa

    clip: Path = request.getfixturevalue("insv_clip" if model == "x4-air" else "insv_x5_clip")
    reader = insta360_io.InsvReader(clip, entity_path="/rig", frame_limit=READER_FRAMES)
    assert reader.rig.profile_name == model
    batches: list[tuple[str, pa.RecordBatch]] = [(c.entity_path, c.to_record_batch()) for c in reader.chunks() if not c.is_static]

    def rows(entity: str, column: str) -> list[Any]:
        return [row for path, batch in batches if path == entity and column in batch.schema.names for row in batch.column(column).to_pylist()]

    def times(entity: str, column: str) -> list[int]:
        return [t for path, batch in batches if path == entity and column in batch.schema.names for t in batch.column("video").cast(pa.int64()).to_pylist()]

    signals: dict[str, str] = {}
    frame_times: list[list[int]] = []
    for lens in range(2):
        video: str = f"/rig/cam_{lens:02d}/pinhole/video"
        samples: list[bytes] = [bytes(sample[0]) for sample in rows(video, "VideoStream:sample")]
        signals[f"lens{lens}_samples"] = hashlib.sha256(b"".join(samples)).hexdigest()
        signals[f"lens{lens}_sample_lengths"] = digest([len(sample) for sample in samples], "<i8")
        frame_times.append(times(video, "VideoStream:sample"))
        signals[f"lens{lens}_exposure"] = digest(rows(f"/rig/cam_{lens:02d}/exposure", "Scalars:scalars"), "<f8")
    assert frame_times[0] == frame_times[1] and len(frame_times[0]) == READER_FRAMES
    signals["frame_times"] = digest(frame_times[0], "<i8")
    imu_times: list[int] = times("/rig/imu_00/gyro", "Scalars:scalars")
    assert imu_times == times("/rig/imu_00/accel", "Scalars:scalars")
    signals["imu_times"] = digest(imu_times, "<i8")
    for name in ("gyro", "accel"):
        signals[name] = digest(rows(f"/rig/imu_00/{name}", "Scalars:scalars"), "<f8")
    assert signals == GOLDEN[model]


@pytest.mark.integration
def test_a_body_without_a_profile_is_read_loudly_without_an_imu_pose(insv_x3_clip: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The X3 (two files, H.264, no measured profile): one warning names the body and the reason, the property says so,
    the IMU gets no pose, and both lenses still read in Mp4Reader's shapes."""
    import logging

    import rerun as rr

    with caplog.at_level(logging.WARNING, logger="insta360_io.reader"):
        reader = insta360_io.InsvReader(insv_x3_clip, entity_path="/rig", frame_limit=30)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "Insta360 X3" in warnings[0] and 'no device profile for a "Insta360 X3"' in warnings[0], warnings
    rig = reader.rig
    assert (rig.profile_name, rig.imu_noise, rig.pano_from_imu, rig.codec, rig.lens_track) == (None, None, None, "h264", (0, 1))
    chunks = reader.stream().to_chunks()
    entities = {c.entity_path for c in chunks}
    assert "/rig/imu_00" not in entities and {"/rig/imu_00/gyro", "/rig/cam_00/pinhole/video", "/rig/cam_01/pinhole/video"} <= entities
    insta360 = next(c for c in chunks if c.entity_path == "/__properties/insta360").to_record_batch()
    assert insta360.column("profile").to_pylist() == [["none"]]
    assert insta360.column("profile_note").to_pylist() == [[rig.profile_note]]
    for lens in range(2):
        codec = next(c for c in chunks if c.entity_path == f"/rig/cam_{lens:02d}/pinhole/video" and c.is_static).to_record_batch()
        assert codec.column("VideoStream:codec").to_pylist() == [[rr.VideoCodec.H264.value]]
        samples = sum(c.num_rows for c in chunks if c.entity_path == f"/rig/cam_{lens:02d}/pinhole/video" and "VideoStream:sample" in c.to_record_batch().schema.names)
        assert samples == 30


@pytest.mark.integration
def test_a_zero_frame_limit_and_reads_outside_the_rig_raise_instead_of_panicking(insv_x3_clip: Path) -> None:
    with pytest.raises(ValueError, match="frame limit 0"):
        insta360_io.InsvReader(insv_x3_clip, frame_limit=0)
    rig = insta360_io.InsvRig(str(insv_x3_clip), 30)
    for lens, start, stop in [(2, 0, 1), (0, 5, 3), (0, 0, 31), (1, 30, 31)]:
        with pytest.raises(IndexError):
            rig.read_packed(lens, start, stop)
    with pytest.raises(IndexError):
        rig.gops(2)


@pytest.mark.golden
def test_the_reader_logs_the_typed_mei_model_and_a_paraxial_pinhole(insv_clip: Path, tmp_path: Path) -> None:
    """Each lens: ``DistortionModel`` ``mei_v6_unified`` with ``gamma_x, gamma_y, xi, half_field_rad, d0 .. d12``, and a
    ``Pinhole`` at the paraxial ``gamma / (1 + xi)`` for the viewer (exoego:v2 section 4). Read back through simplecv, the
    camera is the clip's own."""
    import rerun.chunk as rrc
    from simplecv.catalog_calibration import read_camera_models_from_store

    reader = insta360_io.InsvReader(insv_clip, entity_path="/world/rig_00", frame_limit=2)
    statics: list[Any] = [chunk for chunk in reader.static_chunks()]
    path: Path = tmp_path / "statics.rrd"
    rrc.ChunkStore.from_chunks(statics).stream().write_rrd(path, application_id="dataforge", recording_id="clip")
    nodes: list[str] = ["/world/rig_00/cam_00", "/world/rig_00/cam_01"]
    models = read_camera_models_from_store(rrc.RrdReader(path).store(), nodes, str(path))
    for lens, node in enumerate(nodes):
        camera = reader.rig.lenses[lens].camera
        model = models[node]
        assert model.distortion is not None and model.distortion.model == f"{camera.model}_unified"
        np.testing.assert_allclose(model.distortion.coefficients, [camera.fx, camera.fy, *camera.coefficients], rtol=1e-6, atol=1e-7)
        xi: float = camera.coefficients[0]
        k = model.pinhole.intrinsics.k_matrix
        assert k is not None
        np.testing.assert_allclose(np.diag(k)[:2], [camera.fx / (1 + xi), camera.fy / (1 + xi)], rtol=1e-6)
        assert model.mei is not None and model.mei.model == camera.model
        np.testing.assert_allclose([model.mei.gamma_x, model.mei.gamma_y, model.mei.cx, model.mei.cy], [camera.fx, camera.fy, camera.cx, camera.cy], rtol=1e-6)
