"""Driver-run HO-Cap checks; all assets are local and no model download is allowed."""

from pathlib import Path
from zipfile import ZipFile

import av
import numpy as np
import pyarrow as pa
import pytest
from conftest import read_chunks
from simplecv.ops.mano.mano_np import MANOLayerNP

from dataforge import paths, schema
from dataforge.datasets.hocap import HocapConfig
from dataforge.datasets.hocap_source import EGO_RIG, EXO_RIGS, SequenceData, read_labels, read_sequence
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4

KEY: str = "subject_5/20231027_113202"


@pytest.fixture
def hocap_root() -> Path:
    root: Path = paths.raw_root() / "hocap"
    for name in ("calibration", "models", "poses", "labels", "subject_5"):
        if not (root / f"{name}.zip").is_file():
            pytest.skip(f"HOCap asset absent: {root / f'{name}.zip'} ({KEY})")
    return root


@pytest.fixture
def real_scene(hocap_root: Path) -> SequenceData:
    with (
        ZipFile(hocap_root / "subject_5.zip") as subject,
        ZipFile(hocap_root / "calibration.zip") as calibration,
        ZipFile(hocap_root / "poses.zip") as poses,
    ):
        return read_sequence(subject, calibration, poses, KEY, frame_limit=30, members=frozenset(subject.namelist()))


@pytest.mark.integration
def test_real_sequence_all_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hocap_root: Path, nvenc_ffmpeg: Path) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    dataset = HocapConfig(root=hocap_root, sequences=(KEY,), frame_limit=30).setup()
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=True)
    targets = dataset.targets(identity)
    assert set(targets) == {"base", "hand_pose"}
    for layer, target in targets.items():
        chunks = read_chunks(target)
        assert chunks, layer
        temporal = [chunk for chunk in chunks if not chunk.is_static]
        assert temporal, layer
        for chunk in temporal:
            batch = chunk.to_record_batch()
            assert set(chunk.timeline_names) == {"video_time", "frame_index"}
            frames = batch.column("frame_index").to_pylist()
            assert all(0 <= frame < 30 for frame in frames)
            np.testing.assert_array_equal(
                batch.column("video_time").cast(pa.int64()).to_numpy(), np.rint(np.array(frames) / 30 * 1e9).astype(np.int64)
            )
    base = read_chunks(targets["base"])
    for rig in (*EXO_RIGS, EGO_RIG):
        video = [c for c in base if c.entity_path == schema.video_path(rig, 0) and "VideoStream:sample" in c.to_record_batch().schema.names]
        assert sum(c.num_rows for c in video) == 30
    hands = read_chunks(targets["hand_pose"])
    assert not any(f"rig_{EGO_RIG:02}" in chunk.entity_path for chunk in hands)
    stamps = {layer: target.stat().st_mtime_ns for layer, target in targets.items()}
    dataset.convert(identity, source, force=False)
    assert stamps == {layer: target.stat().st_mtime_ns for layer, target in targets.items()}


@pytest.mark.integration
def test_jpeg_nvenc_retains_color_count_and_no_b_frames(tmp_path: Path, hocap_root: Path, nvenc_ffmpeg: Path) -> None:
    output: Path = tmp_path / "jpeg.mp4"
    with ZipFile(hocap_root / "subject_5.zip") as archive:
        count = encode_frames_to_mp4(
            (archive.read(f"{KEY}/hololens_kv5h72/color_{frame:06d}.jpg") for frame in range(30)),
            output,
            source=FrameSource("jpeg"),
            fps=30,
            gop=60,
            cq=36,
            ffmpeg=nvenc_ffmpeg,
        )
    assert count == 30
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.codec.canonical_name == "av1"
        packets = [packet for packet in container.demux(stream) if packet.pts is not None]
        assert len(packets) == 30
        assert all(packet.pts == packet.dts for packet in packets)
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 30
        assert (frames[0].width, frames[0].height) == (1280, 720)
        rgb = frames[0].to_ndarray(format="rgb24").astype(np.float32)
        assert np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 2])) > 1.0


@pytest.mark.golden
def test_shipped_pixels_match_world_projection(hocap_root: Path, real_scene: SequenceData) -> None:
    with ZipFile(hocap_root / "labels.zip") as archive:
        labels = read_labels(archive, KEY, real_scene, members=frozenset(archive.namelist()))
    for rig, calibration in ((camera.rig, real_scene.intrinsics[camera.rig]) for camera in real_scene.exo):
        camera_from_world = np.linalg.inv(real_scene.world_T_cam[rig])
        camera = labels.xyz @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        projected = camera @ calibration.matrix.T
        projected = projected[:, :, :2] / projected[:, :, 2:]
        valid = np.isfinite(labels.uv[rig]).all(axis=-1) & np.isfinite(projected).all(axis=-1)
        errors = np.linalg.norm(projected[valid] - labels.uv[rig][valid], axis=-1)
        assert len(errors) > 100
        assert float(np.median(errors)) <= 1.0, (rig, float(np.median(errors)))


@pytest.mark.golden
def test_mano_joints_agree_with_shipped_world_joints(hocap_root: Path, real_scene: SequenceData) -> None:
    with ZipFile(hocap_root / "labels.zip") as archive:
        labels = read_labels(archive, KEY, real_scene, members=frozenset(archive.namelist()))
    for hand, side, start, tolerance in ((0, "right", 112, 1e-4), (1, "left", 91, 0.005)):
        model = MANOLayerNP(side=side, betas=real_scene.betas, use_pca=True)
        _, joints = model(real_scene.mano[hand, :, :48], real_scene.mano[hand, :, 48:])
        error = np.linalg.norm(joints - labels.xyz[:, start : start + 21], axis=-1)
        # Official MANO v1.2 left labels vs tracked wilor-nano mano_clean left asset: 5 mm band.
        assert float(np.nanmax(error)) <= tolerance
