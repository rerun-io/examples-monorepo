from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.chunk as rrc
import torch

from mvs.blueprint import CURRENT_CAM_COLOR, CURRENT_CAM_PLANE_M, SOURCE_CAM_PLANE_M, EntityPaths
from mvs.depth_engine import IMAGENET_MEAN, IMAGENET_STD, DepthInputs
from mvs.frame_views import (
    LogConfig,
    depth_normals,
    log_current_cam_style,
    log_lowest_cost,
    log_sources,
    normals_rgb,
    rgb_at_depth_raster,
    source_cameras,
)

PATHS: EntityPaths = EntityPaths()
K_S1_33: torch.Tensor = torch.tensor([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])


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


def test_log_config_defaults_log_the_full_layout_and_check_the_jpeg_quality() -> None:
    config = LogConfig()
    assert config.camera and config.source_cams and config.normals and config.lowest_cost and config.timings
    assert config.jpeg_quality is None
    assert LogConfig(jpeg_quality=85).jpeg_quality == 85
    for quality in (0, 101):
        with pytest.raises(ValueError, match="jpeg_quality"):
            LogConfig(jpeg_quality=quality)


def test_rgb_at_depth_raster_undoes_the_model_normalisation() -> None:
    """The fused colour and the source images are the model's input images, un-normalised and averaged 2x2."""
    rgb_3hw = torch.zeros(3, 384, 512)
    rgb_3hw[0] = 1.0  # red
    rgb_3hw[2, :, 256:] = 0.5  # blue on the right half
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rgb_nhw3 = rgb_at_depth_raster(((rgb_3hw - mean) / std).repeat(2, 1, 1, 1))
    assert rgb_nhw3.shape == (2, 192, 256, 3) and rgb_nhw3.dtype.name == "uint8"
    assert rgb_nhw3[1, 0, 0].tolist() == [255, 0, 0]
    assert rgb_nhw3[1, 0, 255].tolist() == [255, 0, 128]


def test_source_cameras_are_world_from_camera_at_the_depth_raster() -> None:
    world_T_cam_m44, K_m33 = source_cameras(_inputs())
    assert world_T_cam_m44.shape == (7, 4, 4) and K_m33.shape == (7, 3, 3)
    assert np.allclose(world_T_cam_m44[:, 0, 3], np.arange(7))
    assert np.allclose(K_m33[3], [[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]])


def test_depth_normals_of_a_wall_face_the_camera_axis() -> None:
    K_33 = torch.tensor([[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]])
    normals_b3hw = depth_normals(torch.full((1, 1, 192, 256), 2.0), torch.linalg.inv(K_33)[None])
    interior = normals_b3hw[0, :, 10:-10, 10:-10]
    assert torch.allclose(interior, torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1), atol=1e-4)
    assert normals_rgb(interior)[0, 0].tolist() == [127, 127, 255]


def test_log_sources_writes_seven_small_frusta_and_their_images(tmp_path: Path) -> None:
    target: Path = tmp_path / "views.rrd"
    recording = rr.RecordingStream("mvs-test")
    recording.save(target)
    with recording:
        log_current_cam_style(PATHS)
        rr.set_time("frame", sequence=3)
        log_sources(PATHS, _inputs(), jpeg_quality=None)
        log_lowest_cost(PATHS, torch.full((1, 96, 128), 0.25))
    recording.flush()
    recording.disconnect()

    columns = _columns(target)
    current = columns[f"/{PATHS.current_pinhole}"]
    assert current["Pinhole:color"] == [[int.from_bytes(bytes(CURRENT_CAM_COLOR), "big")]]
    assert current["Pinhole:image_plane_distance"] == [[pytest.approx(CURRENT_CAM_PLANE_M)]]
    for index in range(7):
        pinhole = columns[f"/{PATHS.source_pinhole(index)}"]
        assert pinhole["Pinhole:resolution"] == [[[256.0, 192.0]]]
        assert pinhole["Pinhole:image_plane_distance"] == [[pytest.approx(SOURCE_CAM_PLANE_M)]]
        assert f"/{PATHS.source_rgb(index)}" in columns
    assert columns[f"/{PATHS.source_cam(6)}"]["Transform3D:translation"] == [[[6.0, 0.0, 0.0]]]
    assert columns[f"/{PATHS.lowest_cost}"]["DepthImage:meter"] == [[1.0]]
