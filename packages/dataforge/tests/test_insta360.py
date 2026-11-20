"""The Insta360 dataset: the clips it finds and the clip converted into its layers (the device profiles and the frame-time
model are insta360-io's, tested in crates/insta360-io-core)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.chunk as rrc
from conftest import assert_raw_root_guarded, read_back, read_chunks, recording_properties, static_values
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

FRAMES: int = 90
"""Framesets the asset tests convert: three seconds."""
MIN_PSNR_DB: float = 38.0
"""Floor for the first decoded panorama frame against the stitch at the default cq 34 (measured 41.7 dB on 2026-11-17)."""


def test_each_clip_in_the_root_is_one_sequence_and_an_empty_root_is_refused(tmp_path: Path) -> None:
    from dataforge.datasets.insta360 import ClipSource, Insta360Config

    (tmp_path / "VID_2.insv").write_bytes(b"")
    (tmp_path / "VID_1.insv").write_bytes(b"")
    (tmp_path / "VID_1.mp4").write_bytes(b"")
    found = Insta360Config(root=tmp_path).setup().discover()
    assert [(identity.recording_id, source) for identity, source in found] == [
        ("insta360__VID_1", ClipSource(tmp_path / "VID_1.insv")),
        ("insta360__VID_2", ClipSource(tmp_path / "VID_2.insv")),
    ]
    with pytest.raises(FileNotFoundError, match="no .insv clip"):
        Insta360Config(root=tmp_path / "empty").setup().download()


def test_a_two_file_clip_is_one_sequence_named_by_its_lens_0_file(tmp_path: Path) -> None:
    """An X3 / ONE X2 clip is a `_00_` and a `_10_` file; either opens the whole clip, so only the `_00_` one is a sequence.
    A `_10_` file without its partner stays a sequence, so converting it names the missing file."""
    from dataforge.datasets.insta360 import ClipSource, Insta360Config

    names: list[str] = ["VID_20231218_150323_00_022.insv", "VID_20231218_150323_10_022.insv", "VID_20261116_125200_00_003.insv", "VID_20231126_111301_10_004.insv"]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    found = Insta360Config(root=tmp_path).setup().discover()
    assert [(identity.recording_id, source) for identity, source in found] == [
        ("insta360__VID_20231126_111301_10_004", ClipSource(tmp_path / "VID_20231126_111301_10_004.insv")),
        ("insta360__VID_20231218_150323_00_022", ClipSource(tmp_path / "VID_20231218_150323_00_022.insv")),
        ("insta360__VID_20261116_125200_00_003", ClipSource(tmp_path / "VID_20261116_125200_00_003.insv")),
    ]


def test_convert_refuses_an_output_root_beneath_the_raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataforge.datasets.insta360 import ClipSource

    assert_raw_root_guarded("insta360", ClipSource(tmp_path / "clip.insv"), tmp_path, monkeypatch)


# ── the clip ─────────────────────────────────────────────────────────────


class Converted(NamedTuple):
    """The first ``FRAMES`` framesets of the clip, converted under one output root."""

    clip: Path
    """The source clip."""
    root: Path
    """The output root."""
    targets: dict[str, Path]
    """Layer files by layer name; the panorama exists only once ``panorama_rrd`` ran."""


def convert_layers(clip: Path, output_root: Path, **layers: object) -> dict[str, Path]:
    """Convert the clip's first ``FRAMES`` framesets under ``output_root``: the base, and the layers ``Insta360Config`` options ask for."""
    from dataforge.datasets.insta360 import ClipSource, Insta360Config
    from dataforge.identity import SequenceIdentity

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATAFORGE_OUTPUT_ROOT", str(output_root))
        dataset = Insta360Config(root=clip.parent, frame_limit=FRAMES, **layers).setup()  # pyrefly: ignore[bad-argument-type]
        identity: SequenceIdentity = SequenceIdentity(dataset.config.name, (clip.stem,))
        dataset.convert(identity, ClipSource(clip), force=False)
        return dataset.targets(identity)


@pytest.fixture(scope="module")
def converted(insv_clip: Path, tmp_path_factory: pytest.TempPathFactory) -> Converted:
    """The base layer of the clip's first framesets (no GPU needed): a default conversion writes no panorama."""
    from dataforge.datasets.insta360 import PANORAMA_LAYER

    root: Path = tmp_path_factory.mktemp("insta360")
    targets: dict[str, Path] = convert_layers(insv_clip, root, panorama=False)
    assert not targets[PANORAMA_LAYER].exists()
    return Converted(insv_clip, root, targets)


@pytest.fixture(scope="module")
def panorama_rrd(converted: Converted, nvenc_ffmpeg: Path) -> Path:
    """The panorama layer added beside ``converted``'s base layer, which it leaves as written."""
    from dataforge import paths
    from dataforge.datasets.insta360 import PANORAMA_LAYER

    base_mtime_ns: int = converted.targets[paths.BASE_LAYER].stat().st_mtime_ns
    panorama: Path = convert_layers(converted.clip, converted.root, panorama=True)[PANORAMA_LAYER]
    assert converted.targets[paths.BASE_LAYER].stat().st_mtime_ns == base_mtime_ns
    return panorama


def video_times(rrd: Path, entity: str) -> Int64[ndarray, " n"]:
    """Sorted ``video_time`` of every ``VideoStream`` sample at ``entity``."""
    chunks: list[rrc.Chunk] = [chunk for chunk in read_chunks(rrd) if chunk.entity_path == entity and "VideoStream:sample" in chunk.to_record_batch().schema.names]
    return np.sort(np.concatenate([np.asarray(chunk.to_record_batch().column("video_time").cast(pa.int64())) for chunk in chunks]))


@pytest.mark.integration
def test_the_clip_converts_into_mei_cameras_on_the_imu_with_their_readout_and_exposure(converted: Converted) -> None:
    import insta360_io

    from dataforge import paths, schema

    clip = insta360_io.InsvClip(str(converted.clip))
    store: rrc.ChunkStore = read_back(converted.targets[paths.BASE_LAYER])
    insta360: dict[str, object] = recording_properties(store, "insta360")
    assert sorted(insta360) == ["camera_type", "firmware", "pano_from_imu", "profile", "profile_version", "source_file"]
    assert (insta360["source_file"], insta360["firmware"], insta360["profile"], insta360["profile_version"]) == (converted.clip.name, clip.firmware, "x4-air", 1)
    assert recording_properties(store, "capture")["num_frames"] == FRAMES
    assert static_values(store, schema.rig_path(0))["reference"] == "imu_00"
    assert clip.rolling_shutter_ms is not None
    frame_ns: Int64[ndarray, " n"] = video_times(converted.targets[paths.BASE_LAYER], schema.video_path(0, 0))
    assert frame_ns.size == FRAMES
    for cam, lens in enumerate(clip.lenses_v6):
        camera = lens.camera
        node: dict[str, object] = static_values(store, schema.cam_path(0, cam))
        assert (node["name"], node["camera_model"], node["stream_id"]) == (f"lens{cam}", "mei_v6", f"track{clip.lens_track[cam]}")
        assert node["rolling_shutter_readout_s"] == pytest.approx(clip.rolling_shutter_ms * 1e-3)
        pinhole: dict[str, object] = static_values(store, schema.pinhole_path(0, cam))
        assert pinhole["simplecv.components.DistortionModel"] == "mei_v6"
        np.testing.assert_allclose(np.asarray(pinhole["simplecv.components.DistortionCoefficients"], dtype=np.float64), camera.coefficients, rtol=1e-7)  # stored as float32
        np.testing.assert_allclose(np.asarray(pinhole["image_from_camera"], dtype=np.float64), [camera.fx, 0.0, 0.0, 0.0, camera.fy, 0.0, camera.cx, camera.cy, 1.0], rtol=1e-7)
        np.testing.assert_array_equal(video_times(converted.targets[paths.BASE_LAYER], schema.video_path(0, cam)), frame_ns)
        exposure: pa.Table = store.reader(index=schema.TIMELINE, contents=f"{schema.cam_path(0, cam)}/exposure").to_arrow_table()
        assert exposure.num_rows == FRAMES


@pytest.mark.integration
def test_a_clip_without_a_profile_converts_without_a_rig_reference_or_imu_noise(insv_x3_clip: Path, tmp_path: Path) -> None:
    """The base of an unprofiled body names no rig reference and carries no IMU noise model, so slam-rs refuses it."""
    from dataforge import paths, schema

    store: rrc.ChunkStore = read_back(convert_layers(insv_x3_clip, tmp_path, panorama=False)[paths.BASE_LAYER])
    assert "reference" not in static_values(store, schema.rig_path(0))
    assert not any(key.startswith("ImuCalibration") or "noise" in key for key in static_values(store, schema.imu_path(0, 0)))
    assert recording_properties(store, "insta360")["profile"] == "none"


@pytest.mark.integration
def test_the_panorama_decodes_both_lenses_of_an_h264_body(insv_x3_clip: Path) -> None:
    """The X3 records H.264: the panorama's lens decoder follows the clip's codec."""
    import insta360_io

    from dataforge.datasets.insta360 import gray_frames

    rig = insta360_io.InsvRig(str(insv_x3_clip), 30)
    width, height = rig.resolution
    for lens in range(2):
        shapes = [frame.shape for frame in gray_frames(rig, lens, threads=1)]
        assert shapes == [(height, width)] * 30


@pytest.mark.integration
def test_the_cube_lens_decoder_reads_an_h264_body_on_nvdec_as_pyav_does(insv_x3_clip: Path) -> None:
    """The X3 (H.264, a GOP of 16, its lenses in two files) at stride 7 in batches of 5, so batches span several GOP segments."""
    import insta360_io
    import torch

    from dataforge.datasets.insta360 import lens_pictures

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rig = insta360_io.InsvRig(str(insv_x3_clip), 120)
    indices: Int64[ndarray, " k"] = np.arange(0, 120, 7, dtype=np.int64)
    pairs = {device: list(lens_pictures(rig, indices, torch.device(device), batch=5, threads=4)) for device in ("cuda", "cpu")}
    for lens in range(2):
        cuda, cpu = (torch.cat([pair[lens] for pair in pairs[device]]).cpu().numpy().astype(np.int16) for device in ("cuda", "cpu"))
        assert cuda.shape == (indices.size, 3, rig.resolution[1], rig.resolution[0])
        assert np.abs(cuda - cpu).max() <= 1, f"lens {lens}"


@pytest.mark.golden
def test_the_lenses_sit_33_3_mm_apart_along_the_lens_axis_with_the_factory_focal(converted: Converted) -> None:
    """Rig frame = IMU: lens 1 sits 33.29 mm behind lens 0 along +y_imu (V6 string), lens 0 looks along z_P = -y_imu."""
    from dataforge import paths, schema

    store: rrc.ChunkStore = read_back(converted.targets[paths.BASE_LAYER])
    centres: list[Float64[ndarray, "3"]] = []
    for cam, focal in enumerate((1835.37, 1848.94)):
        transform: dict[str, object] = static_values(store, schema.cam_path(0, cam))
        cam_R_imu: Float64[ndarray, "3 3"] = np.asarray(transform["mat3x3"], dtype=np.float64).reshape(3, 3).T  # column-major
        centres.append(-cam_R_imu.T @ np.asarray(transform["translation"], dtype=np.float64))
        if cam == 0:
            assert cam_R_imu[2] @ np.array([0.0, -1.0, 0.0]) == pytest.approx(1.0, abs=1e-3)
        image_from_camera: list[float] = static_values(store, schema.pinhole_path(0, cam))["image_from_camera"]  # pyrefly: ignore[bad-assignment]
        assert image_from_camera[0] == pytest.approx(focal, abs=0.1)
    baseline: Float64[ndarray, "3"] = centres[1] - centres[0]
    assert np.linalg.norm(baseline) == pytest.approx(0.03329, abs=1e-4)
    assert baseline[1] > 0.99 * np.linalg.norm(baseline)


@pytest.mark.integration
def test_the_panorama_layer_is_av1_on_the_base_frame_times(converted: Converted, panorama_rrd: Path) -> None:
    from dataforge import paths, schema
    from dataforge.datasets.insta360 import PANO_ENTITY

    assert rrc.RrdReader(panorama_rrd).recordings() == rrc.RrdReader(converted.targets[paths.BASE_LAYER]).recordings()
    pano: list[rrc.Chunk] = [chunk for chunk in read_chunks(panorama_rrd) if chunk.entity_path == PANO_ENTITY]
    codecs: list[int] = [
        value
        for chunk in pano
        if "VideoStream:codec" in chunk.to_record_batch().schema.names
        for value in chunk.to_record_batch().column("VideoStream:codec").flatten().to_pylist()
    ]
    assert codecs == [rr.VideoCodec.AV1.value]
    np.testing.assert_array_equal(video_times(panorama_rrd, PANO_ENTITY), video_times(converted.targets[paths.BASE_LAYER], schema.video_path(0, 0)))


@pytest.mark.golden
def test_the_first_panorama_frame_is_the_stitch_within_38_db(converted: Converted, panorama_rrd: Path) -> None:
    import insta360_io

    from dataforge import paths
    from dataforge.datasets.insta360 import PANO_ENTITY, gray_frames, stitch_from_recording

    chunks: list[rrc.Chunk] = [c for c in read_chunks(panorama_rrd) if c.entity_path == PANO_ENTITY and "VideoStream:sample" in c.to_record_batch().schema.names]
    rows: list[tuple[int, bytes]] = [
        (int(time), bytes(sample[0]))
        for chunk in chunks
        for time, sample in zip(
            chunk.to_record_batch().column("video_time").cast(pa.int64()).to_pylist(), chunk.to_record_batch().column("VideoStream:sample").to_pylist(), strict=True
        )
    ]
    first: bytes = min(rows)[1]
    decoder = av.CodecContext.create("libdav1d", "r")
    decoded: UInt8[ndarray, "h w"] = (decoder.decode(av.Packet(first)) + decoder.decode(None))[0].to_ndarray(format="gray")  # pyrefly: ignore[missing-attribute]
    rig = insta360_io.InsvRig(str(converted.clip), 1)
    stitch: insta360_io.PanoramaStitch = stitch_from_recording(converted.targets[paths.BASE_LAYER])
    lenses: list[Iterator[UInt8[ndarray, "h w"]]] = [gray_frames(rig, lens, threads=1) for lens in range(2)]
    expected: UInt8[ndarray, "960 1920"] = stitch.panorama(next(lenses[0]), next(lenses[1]))
    mse: float = float(np.mean((decoded.astype(np.float64) - expected.astype(np.float64)) ** 2))
    psnr: float = 10.0 * np.log10(255.0**2 / mse)
    print(f"first panorama frame PSNR {psnr:.2f} dB")
    assert psnr > MIN_PSNR_DB


def decoded_rgb(rrd: Path, entity: str) -> UInt8[ndarray, "k h w 3"]:
    """Every ``VideoStream`` sample at ``entity``, in time order, decoded (libdav1d) to RGB."""
    rows: list[tuple[int, bytes]] = []
    for chunk in read_chunks(rrd):
        batch = chunk.to_record_batch()
        if chunk.entity_path == entity and "VideoStream:sample" in batch.schema.names:
            times: list[int] = batch.column("video_time").cast(pa.int64()).to_pylist()
            rows += [(t, bytes(sample[0])) for t, sample in zip(times, batch.column("VideoStream:sample").to_pylist(), strict=True)]
    decoder = av.CodecContext.create("libdav1d", "r")
    assert isinstance(decoder, av.video.codeccontext.VideoCodecContext)  # narrows the type under every PyAV stub version
    frames = [frame for _, sample in sorted(rows) for frame in decoder.decode(av.Packet(sample))] + list(decoder.decode(None))
    return np.stack([frame.to_ndarray(format="rgb24") for frame in frames])  # pyrefly: ignore[missing-attribute]


@pytest.mark.integration
def test_the_cube_layer_on_cuda_is_the_cpu_layer_within_the_nvenc_band(insv_clip: Path, nvenc_ffmpeg: Path, tmp_path: Path) -> None:
    """Every tenth frameset of the first ``FRAMES``: six face videos on the base's stamps, CUDA and CPU within 40 dB."""
    import torch

    from dataforge import paths, schema
    from dataforge.datasets.insta360 import CUBE_LAYER, FACE_CAMS

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    layers: dict[str, dict[str, Path]] = {device: convert_layers(insv_clip, tmp_path / device, cube=True, cube_device=device) for device in ("cuda", "cpu")}
    stamps: Int64[ndarray, " k"] = video_times(layers["cuda"][paths.BASE_LAYER], schema.video_path(0, 0))[::10]
    for cam in FACE_CAMS:
        entity: str = schema.video_path(0, cam)
        np.testing.assert_array_equal(video_times(layers["cuda"][CUBE_LAYER], entity), stamps)
        np.testing.assert_array_equal(video_times(layers["cpu"][CUBE_LAYER], entity), stamps)
        cuda, cpu = (decoded_rgb(layers[device][CUBE_LAYER], entity) for device in ("cuda", "cpu"))
        assert cuda.shape == cpu.shape == (stamps.size, 800, 800, 3)
        mse: float = float(np.mean((cuda.astype(np.float64) - cpu.astype(np.float64)) ** 2))
        assert 10.0 * np.log10(255.0**2 / mse) > 40.0, f"cam_{cam:02d}"


@pytest.mark.integration
def test_the_cube_layer_without_nvenc_encodes_in_software_on_the_base_stamps(insv_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-GPU lane: PyAV decode, OpenCV remap and SVT-AV1, six face videos on every tenth base stamp."""
    from dataforge import paths, schema
    from dataforge.datasets import insta360

    monkeypatch.setattr(insta360, "av1_nvenc_encodes", lambda: False)
    targets: dict[str, Path] = convert_layers(insv_clip, tmp_path, cube=True, cube_device="cpu")
    stamps: Int64[ndarray, " k"] = video_times(targets[paths.BASE_LAYER], schema.video_path(0, 0))[::10]
    for cam in insta360.FACE_CAMS:
        entity: str = schema.video_path(0, cam)
        np.testing.assert_array_equal(video_times(targets[insta360.CUBE_LAYER], entity), stamps)
        assert decoded_rgb(targets[insta360.CUBE_LAYER], entity).shape == (stamps.size, 800, 800, 3)
