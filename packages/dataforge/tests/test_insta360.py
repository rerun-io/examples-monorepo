"""The Insta360 dataset: device profiles and their selection, the frame-time model, and the clip converted into its layers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.chunk as rrc
from conftest import assert_raw_root_guarded, read_back, read_chunks, recording_properties, static_values
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

if TYPE_CHECKING:
    from dataforge.datasets.insta360 import DeviceProfile

FRAMES: int = 90
"""Framesets the asset tests convert: three seconds."""
MIN_PSNR_DB: float = 36.0
"""Floor for the first decoded panorama frame (RGB, 3840x1920) against the join at the default cq 34 (measured 37.4 dB on 2026-11-19)."""


def angle_to_nearest_permutation_deg(rotation: Float64[ndarray, "3 3"]) -> float:
    """Angle in degrees between ``rotation`` and the signed axis permutation nearest to it (row by row)."""
    permutation: Float64[ndarray, "3 3"] = np.zeros((3, 3))
    for row in range(3):
        column: int = int(np.argmax(np.abs(rotation[row])))
        permutation[row, column] = np.sign(rotation[row, column])
    cosine: float = (float(np.trace(permutation.T @ rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def x4_air() -> DeviceProfile:
    from dataforge.datasets.insta360 import load_profiles

    return next(profile for profile in load_profiles() if profile.name == "x4-air")


def test_the_measured_imu_rotation_orthonormalises_to_0_13_deg_from_its_permutation() -> None:
    """The X4 Air profile's R_P_imu is 0.13 deg from x_P = -z_imu, y_P = +x_imu, z_P = -y_imu."""
    from dataforge.datasets.insta360 import orthonormal

    profile: DeviceProfile = x4_air()
    assert (profile.name, profile.version, profile.camera_type) == ("x4-air", 1, "Insta360 X4 Air")
    rotation: Float64[ndarray, "3 3"] = orthonormal(profile.pano_from_imu_measured)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.abs(rotation - profile.pano_from_imu_measured).max() < 1e-4
    assert angle_to_nearest_permutation_deg(rotation) == pytest.approx(0.13, abs=0.02)
    permutation: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert angle_to_nearest_permutation_deg(permutation) == pytest.approx(0.0, abs=1e-6)


def test_orthonormal_returns_a_proper_rotation_for_a_reflection_free_input() -> None:
    from dataforge.datasets.insta360 import orthonormal

    noisy: Float64[ndarray, "3 3"] = np.eye(3) + np.array([[0.0, 0.01, 0.0], [0.0, 0.0, 0.02], [0.005, 0.0, 0.0]])
    rotation: Float64[ndarray, "3 3"] = orthonormal(noisy)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[[profile]]\nname = 'x'\nunknown = 1\n", "not an Insta360 device profile file"),
        ("[[profile]\n", "not an Insta360 device profile file"),
    ],
)
def test_a_profile_file_with_an_unknown_key_or_broken_toml_is_refused_naming_the_file(tmp_path: Path, text: str, message: str) -> None:
    from dataforge.datasets.insta360 import load_profiles

    path: Path = tmp_path / "profiles.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message) as error:
        load_profiles(path)
    assert str(path) in str(error.value)


def test_a_hand_written_integer_widens_to_a_float(tmp_path: Path) -> None:
    from dataforge.datasets.insta360 import PROFILES_TOML, load_profiles

    path: Path = tmp_path / "profiles.toml"
    path.write_text(PROFILES_TOML.read_text().replace("frame_time_constant_s = -0.0001", "frame_time_constant_s = 0"))
    profile: DeviceProfile = next(profile for profile in load_profiles(path) if profile.name == "x4-air")
    assert profile.frame_time_constant_s == 0.0


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({}, None),
        ({"readout_s": None}, None),
        ({"readout_s": 0.015882978}, None),
        ({"lens_type": 113}, "lens type 113 is not 131"),
        ({"frame_rate_hz": 59.94}, "frame rate 59.9400 Hz is not 29.97"),
        ({"readout_s": 0.0079}, "rolling-shutter readout 7.9000 ms is not 15.8830"),
        ({"camera_type": "Insta360 X5"}, "no device profile for a 'Insta360 X5'"),
    ],
)
def test_the_clip_mode_selects_the_profile_or_names_what_differs(change: dict[str, object], reason: str | None) -> None:
    from dataforge.datasets.insta360 import CaptureMode, select_profile

    mode: CaptureMode = replace(CaptureMode("Insta360 X4 Air", 131, 29.9697, 0.015883), **change)  # pyrefly: ignore[bad-argument-type]
    if reason is None:
        assert select_profile([x4_air()], mode).name == "x4-air"
    else:
        with pytest.raises(ValueError, match=reason):
            select_profile([x4_air()], mode)


X5_V3: str = (
    "2_2.000000_4267.220_4267.930_2693.490_2692.190_0.073_0.192_89.740_0.000000_0.000000_0.000000_0.19010369_2.01127100"
    "_-3.03693724_0.00169806_-0.00122825_10752_5376_113_2.000000_4249.040_4248.200_8059.560_2694.530_-0.294_0.371_90.620"
    "_-0.000469_0.000039_-0.032061_0.18495527_2.02530622_-3.03083992_-0.00055454_0.00157740_10752_5376_113_197632"
)
"""The V3 lens string of the X5 clip VID_20261113_145021_00_003."""


def test_an_x5_clip_selects_the_x5_profile_by_its_lens_type() -> None:
    from dataforge.datasets.insta360 import CaptureMode, lens_type, load_profiles, select_profile

    assert lens_type(X5_V3) == 113
    with pytest.raises(ValueError, match=r"lens types \[113, 131\]"):
        lens_type(X5_V3.replace("_10752_5376_113_197632", "_10752_5376_131_197632"))
    profile: DeviceProfile = select_profile(load_profiles(), CaptureMode("Insta360 X5", 113, 29.9688, 0.021244))
    assert (profile.name, profile.readout_s) == ("x5", 0.021244)


def test_frames_land_on_the_middle_row_time_and_a_non_increasing_model_is_refused() -> None:
    from dataforge.datasets.insta360 import frame_times_ns

    profile: DeviceProfile = x4_air()
    stamps: Int64[ndarray, " n"] = np.array([1_000_000, 1_033_367, 1_066_733], dtype=np.int64)
    shutter: Float64[ndarray, " n"] = np.array([0.01, 0.002, 0.0])
    times: Int64[ndarray, " n"] = frame_times_ns(stamps, shutter, profile)
    # -0.44 * shutter - 0.1 ms, on the stamp.
    np.testing.assert_array_equal(times, stamps * 1000 + np.array([-4_500_000, -980_000, -100_000]))
    with pytest.raises(ValueError, match="non-increasing"):
        frame_times_ns(stamps, np.array([0.0, 0.2, 0.0]), profile)


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
def test_the_first_panorama_frame_is_the_flow_seam_join_within_36_db(converted: Converted, panorama_rrd: Path) -> None:
    import insta360_io
    import torch

    from dataforge import paths
    from dataforge.datasets.insta360 import PANO_ENTITY, cameras_from_recording, lens_pictures
    from dataforge.datasets.insta360_panorama import SEAM_BAND_DEG, PanoramaJoin, default_panorama_width, panorama_table

    decoded: UInt8[ndarray, "k h w 3"] = decoded_rgb(panorama_rrd, PANO_ENTITY)
    assert decoded.shape[1:] == (1920, 3840, 3)
    clip = insta360_io.InsvClip(str(converted.clip))
    lenses = cameras_from_recording(converted.targets[paths.BASE_LAYER])
    table = panorama_table(lenses.cameras, lenses.lens1_from_lens0, lenses.size, default_panorama_width(lenses.size[0]), SEAM_BAND_DEG)
    join = PanoramaJoin(table, lenses.size, torch.device("cuda"), flow=True)
    try:
        lens0, lens1 = next(lens_pictures(clip, np.zeros(1, dtype=np.int64), torch.device("cuda"), batch=1, batch_frames=1, threads=1))
        expected: UInt8[ndarray, "h w 3"] = join.panoramas(lens0, lens1)[0].permute(1, 2, 0).cpu().numpy()
    finally:
        join.close()
    mse: float = float(np.mean((decoded[0].astype(np.float64) - expected.astype(np.float64)) ** 2))
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
