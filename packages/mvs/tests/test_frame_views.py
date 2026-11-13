from pathlib import Path

import depth_png
import numpy as np
import pytest
import rerun as rr
import rerun.chunk as rrc
import torch

from mvs.blueprint import CURRENT_CAM_COLOR, CURRENT_CAM_PLANE_M, SOURCE_CAM_PLANE_M, EntityPaths
from mvs.depth_engine import DepthInputs
from mvs.frame_views import (
    depth_normals,
    log_current_cam_style,
    log_lowest_cost,
    log_normals,
    log_sources,
    normals_rgb,
    source_cameras,
    video_offsets_ns,
)

PATHS: EntityPaths = EntityPaths()
K_S1_33: torch.Tensor = torch.tensor([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])
VIDEO_WH: tuple[int, int] = (1920, 1440)


def _inputs() -> DepthInputs:
    """A depth query whose source view ``i`` sits at x = i metres, looking down +z."""
    world_T_cam_m44 = torch.eye(4).repeat(7, 1, 1)
    world_T_cam_m44[:, 0, 3] = torch.arange(7, dtype=torch.float32)
    K_s1_m44 = torch.eye(4).repeat(7, 1, 1)
    K_s1_m44[:, :3, :3] = K_S1_33
    return DepthInputs(
        cur_image_b3hw=torch.zeros(1, 3, 384, 512),
        src_image_bm3hw=torch.zeros(1, 7, 3, 384, 512),
        src_K_bm44=K_s1_m44[None],
        cur_invK_b44=torch.eye(4)[None],
        src_cam_T_world_bm44=torch.linalg.inv(world_T_cam_m44)[None],
        cur_world_T_cam_b44=torch.eye(4)[None],
    )


def _columns(target: Path) -> dict[str, dict[str, list]]:
    columns: dict[str, dict[str, list]] = {}
    for chunk in rrc.RrdReader(target).stream():
        batch = chunk.to_record_batch()
        columns.setdefault(str(chunk.entity_path), {}).update({name: batch.column(name).to_pylist() for name in batch.schema.names})
    return columns


def test_source_cameras_are_world_from_camera_at_the_video_size() -> None:
    """s1 intrinsics are the native ones scaled to 128x96, so a 1920x1440 video gets them times 15."""
    world_T_cam_m44, K_m33 = source_cameras(_inputs(), VIDEO_WH)
    assert world_T_cam_m44.shape == (7, 4, 4) and K_m33.shape == (7, 3, 3)
    assert np.allclose(world_T_cam_m44[:, 0, 3], np.arange(7))
    assert np.allclose(K_m33[3], [[1500.0, 0.0, 960.0], [0.0, 1500.0, 720.0], [0.0, 0.0, 1.0]])


def test_video_offsets_name_the_decoded_sample_relative_to_the_stream_start() -> None:
    """A grid time maps to the latest video sample at or before it (what the decoder showed the model), as an
    offset from the relayed stream's first sample (what a VideoFrameReference into a VideoStream counts from)."""
    video_times_ns = np.array([1_000, 17_000, 34_000, 50_000], dtype=np.int64)
    grid_ns = np.array([17_500, 33_999, 60_000], dtype=np.int64)
    assert video_offsets_ns(grid_ns, video_times_ns, 1_000).tolist() == [16_000, 16_000, 49_000]
    # The relay may open after sample 0 (at its first keyframe): offsets count from there.
    assert video_offsets_ns(grid_ns, video_times_ns, 17_000).tolist() == [0, 0, 33_000]
    with pytest.raises(ValueError, match="before the relayed stream"):
        video_offsets_ns(np.array([1_500], dtype=np.int64), video_times_ns, 17_000)


def test_depth_normals_of_a_wall_face_the_camera_axis() -> None:
    K_33 = torch.tensor([[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]])
    normals_b3hw = depth_normals(torch.full((1, 1, 192, 256), 2.0), torch.linalg.inv(K_33)[None])
    interior = normals_b3hw[0, :, 10:-10, 10:-10]
    assert torch.allclose(interior, torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1), atol=1e-4)
    assert normals_rgb(interior)[0, 0].tolist() == [127, 127, 255]


def test_log_sources_writes_seven_small_frusta_and_their_video_frames(tmp_path: Path) -> None:
    target: Path = tmp_path / "views.rrd"
    recording = rr.RecordingStream("mvs-test")
    recording.save(target)
    with recording:
        log_current_cam_style(PATHS)
        rr.set_time("frame", sequence=3)
        log_sources(PATHS, _inputs(), VIDEO_WH, np.arange(7, dtype=np.int64) * 1_000_000)
        log_normals(PATHS, torch.full((1, 1, 192, 256), 2.0), np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]], dtype=np.float32))
        log_lowest_cost(PATHS, torch.full((1, 96, 128), 0.25))
    recording.flush()
    recording.disconnect()

    columns = _columns(target)
    current = columns[f"/{PATHS.current_pinhole}"]
    assert current["Pinhole:color"] == [[int.from_bytes(bytes(CURRENT_CAM_COLOR), "big")]]
    assert current["Pinhole:image_plane_distance"] == [[pytest.approx(CURRENT_CAM_PLANE_M)]]
    for index in range(7):
        pinhole = columns[f"/{PATHS.source_pinhole(index)}"]
        assert pinhole["Pinhole:resolution"] == [[[1920.0, 1440.0]]]
        assert pinhole["Pinhole:image_plane_distance"] == [[pytest.approx(SOURCE_CAM_PLANE_M)]]
        frame = columns[f"/{PATHS.source_rgb(index)}"]
        assert frame["VideoFrameReference:timestamp"] == [[index * 1_000_000]]
        assert frame["VideoFrameReference:video_reference"] == [[f"/{PATHS.video}"]]
        assert not any(name.startswith(("Image:", "EncodedImage:")) for name in frame)
    normal = columns[f"/{PATHS.normal}"]
    assert normal["EncodedImage:media_type"] == [["image/jpeg"]]
    assert columns[f"/{PATHS.source_cam(6)}"]["Transform3D:translation"] == [[[6.0, 0.0, 0.0]]]
    lowest_cost = columns[f"/{PATHS.lowest_cost}"]
    assert lowest_cost["EncodedDepthImage:meter"] == [[1000.0]]
    decoded = depth_png.decode(bytes(lowest_cost["EncodedDepthImage:blob"][0][0]))
    assert decoded.dtype == np.uint16 and decoded.shape == (96, 128)
    assert (decoded == 250).all()
