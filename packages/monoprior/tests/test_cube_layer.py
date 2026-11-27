"""``cube_layer.iter_framesets`` on the chunked reader: NVDEC whenever CUDA is available, the whole-stream decoder's frames."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

import rerun as rr  # noqa: E402
from simplecv.testing import encode_h264

pytestmark = pytest.mark.integration

SIZE: int = 64
FACES: tuple[str, ...] = ("cube6_front0", "cube6_left")


def test_framesets_decode_on_nvdec_whenever_cuda_is_available(tmp_path: Path) -> None:
    from simplecv.catalog_video_codec import wrap_mp4
    from simplecv.catalog_video_reader import select_decode_device
    from simplecv.testing import whole_column
    from torchcodec.decoders import VideoDecoder

    from monopriors.cube_faces import CubeFace
    from monopriors.cube_layer import iter_framesets, open_rrd

    path: Path = tmp_path / "faces.rrd"
    times = (33_333_333 * np.arange(40, dtype=np.int64)).astype("timedelta64[ns]")
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        for seed, _ in enumerate(FACES):
            samples: list[tuple[bytes, bool]] = encode_h264(40, 8, seed, width=SIZE, height=SIZE)
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
        videos = whole_column(layer.dataset, layer.segment_id, [f"{face.entity}/pinhole/video" for face in faces], "video_time")
    assert [frameset.t_ns for frameset in framesets] == wanted
    for face, video in zip(faces, videos, strict=True):
        whole = VideoDecoder(wrap_mp4(video.samples, video.keyframes, 30, codec=video.codec), device=device, seek_mode="exact")
        reference = whole.get_frames_at([1, 9, 10, 33]).data.permute(0, 2, 3, 1).cpu().numpy()
        np.testing.assert_array_equal(np.stack([frameset.rgb_by_face[face.name] for frameset in framesets]), reference)
    if torch.cuda.is_available():
        assert device == "cuda", "the faces decode on NVDEC whenever CUDA is available"
