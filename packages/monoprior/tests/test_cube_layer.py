"""``cube_layer.iter_framesets`` on the chunked reader: NVDEC whenever CUDA is available, the whole-stream decoder's frames."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

import av  # noqa: E402
import rerun as rr  # noqa: E402

pytestmark = pytest.mark.integration

SIZE: int = 64
FACES: tuple[str, ...] = ("cube6_front0", "cube6_left")


def _samples(frames: int, seed: int) -> list[tuple[bytes, bool]]:
    """H.264 of moving noise, closed GOPs of 8, no B-frames."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = SIZE, SIZE, "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.options = {"g": "8", "keyint_min": "8", "bf": "0", "sc_threshold": "0"}
    base: np.ndarray = np.random.default_rng(seed).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    packets: list = []
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(np.roll(base, 3 * index, axis=1), format="rgb24")
        frame.pts = index
        packets += codec.encode(frame)
    packets += codec.encode(None)
    return [(bytes(packet), bool(packet.is_keyframe)) for packet in packets]


def test_framesets_decode_on_nvdec_whenever_cuda_is_available(tmp_path: Path) -> None:
    from simplecv.catalog_video import read_catalog_videos
    from simplecv.catalog_video_codec import wrap_mp4
    from simplecv.catalog_video_reader import select_decode_device
    from torchcodec.decoders import VideoDecoder

    from monopriors.cube_faces import CubeFace
    from monopriors.cube_layer import iter_framesets, open_rrd

    path: Path = tmp_path / "faces.rrd"
    times = (33_333_333 * np.arange(40, dtype=np.int64)).astype("timedelta64[ns]")
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        for seed, _ in enumerate(FACES):
            samples: list[tuple[bytes, bool]] = _samples(40, seed)
            entity: str = f"/world/rig_00/cam_0{seed}/pinhole/video"
            rec.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            rec.send_columns(
                entity,
                indexes=[rr.TimeColumn("video_time", duration=times)],
                columns=rr.VideoStream.columns(sample=[s for s, _ in samples], is_keyframe=[k for _, k in samples]),
            )
    k_matrix = np.array([[SIZE / 2, 0.0, (SIZE - 1) / 2], [0.0, SIZE / 2, (SIZE - 1) / 2], [0.0, 0.0, 1.0]])
    faces = tuple(CubeFace(entity=f"/world/rig_00/cam_0{i}", name=name, k_matrix=k_matrix, width=SIZE, height=SIZE) for i, name in enumerate(FACES))
    wanted: list[int] = [int(times[i].astype(np.int64)) for i in (1, 9, 10, 33)]
    device: str = select_decode_device()
    with open_rrd(path, "faces") as layer:
        framesets = list(iter_framesets(layer, faces, "video_time", set(wanted)))
        videos = read_catalog_videos(layer.dataset, layer.segment_id, [f"{face.entity}/pinhole/video" for face in faces], "video_time")
    assert [frameset.t_ns for frameset in framesets] == wanted
    for face, video in zip(faces, videos, strict=True):
        whole = VideoDecoder(wrap_mp4(video.samples, video.keyframes, 30, codec=video.codec), device=device, seek_mode="exact")
        reference = whole.get_frames_at([1, 9, 10, 33]).data.permute(0, 2, 3, 1).cpu().numpy()
        np.testing.assert_array_equal(np.stack([frameset.rgb_by_face[face.name] for frameset in framesets]), reference)
    if torch.cuda.is_available():
        assert device == "cuda", "the faces decode on NVDEC whenever CUDA is available"
